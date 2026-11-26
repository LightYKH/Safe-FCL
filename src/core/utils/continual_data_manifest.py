"""Frozen four-task stratified Train/Validation/Test/Probe manifests."""

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set

from datasets import Dataset

from .artifact_schema import matches_schema
from .data_manifest import (
    canonical_json_bytes,
    make_hf_source_id,
    make_json_source_id,
    sha256_bytes,
    sha256_file,
    validate_dataset_source_identity,
)
from .dataset_partitioning import (
    PROBE_EFFECTIVE_TOTAL_PER_ROLE,
    probe_client_pool_sizes,
    probe_effective_client_pool_sizes,
)


SCHEMA_VERSION = "safe-fcl.continual-task-data-manifest.v1"
AUDIT_SCHEMA_VERSION = "safe-fcl.continual-global-overlap-audit.v1"
SHUFFLE_SEED = 2023
PARTITIONS = ("model_train", "model_validation", "formal_test", "probe")
IDENTITY_FIELDS = ("source_id", "content_hash", "normalized_prompt_hash")
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")

ROLE_PROTOCOL = {
    "benign": {
        "client_count": 7,
        "model_train": 3500,
        "model_validation": 500,
        "formal_test": 1000,
        "probe_target": 2500,
    },
    "malicious": {
        "client_count": 3,
        "model_train": 1500,
        "model_validation": 214,
        "formal_test": 429,
        "probe_target": 2500,
    },
}

TASK_DATASETS = {1: {'benign': 'risk5/task3_benign', 'malicious': 'risk5/task3_malicious'}, 2: {'benign': 'risk5/task1_benign', 'malicious': 'risk5/task1_malicious'}, 3: {'benign': 'risk5/task4_benign', 'malicious': 'risk5/task4_malicious'}, 4: {'benign': 'risk5/task2_benign', 'malicious': 'risk5/task2_malicious'}, 5: {'benign': 'risk5/task5_benign', 'malicious': 'risk5/task5_malicious'}}

RUNTIME_KEYS = (
    "python_version",
    "datasets_version",
    "numpy_version",
    "pandas_version",
    "pyarrow_version",
)


def role_partition_counts(role: str, available: int) -> Dict[str, int]:
    """Return frozen integer 7:1:2 counts and a capacity-limited Probe count."""
    if role not in ROLE_PROTOCOL:
        raise ValueError("role must be benign or malicious")
    if not isinstance(available, int) or isinstance(available, bool) or available < 0:
        raise ValueError("available must be a non-negative integer")
    protocol = ROLE_PROTOCOL[role]
    model_total = sum(protocol[name] for name in PARTITIONS[:3])
    if available < model_total:
        raise ValueError(
            f"FAIL-FAST: {role} has {available} usable rows but needs at least "
            f"{model_total} total rows so the 70% train split contains "
            f"{protocol['model_train']} examples"
        )
    probe_actual = min(protocol["probe_target"], available - model_total)
    return {
        "model_train": protocol["model_train"],
        "model_validation": protocol["model_validation"],
        "formal_test": protocol["formal_test"],
        "probe": probe_actual,
        "model_total": model_total,
        "probe_target": protocol["probe_target"],
        "selected_total": model_total + probe_actual,
        "federated_minimum_satisfied": True,
    }


def selection_target_total(role: str) -> int:
    protocol = ROLE_PROTOCOL[role]
    return sum(protocol[name] for name in PARTITIONS[:3]) + protocol["probe_target"]


def model_required_total(role: str) -> int:
    protocol = ROLE_PROTOCOL[role]
    return sum(protocol[name] for name in PARTITIONS[:3])


def partition_for_rank(role: str, split_rank: int, counts: Mapping[str, int]) -> str:
    if role not in ROLE_PROTOCOL:
        raise ValueError("role must be benign or malicious")
    if not isinstance(split_rank, int) or isinstance(split_rank, bool) or split_rank < 0:
        raise ValueError("split_rank must be a non-negative integer")
    cursor = 0
    for partition in PARTITIONS:
        cursor += int(counts[partition])
        if split_rank < cursor:
            return partition
    raise ValueError("split_rank is outside selected partitions")


def _record_from_row(
    row: Mapping[str, Any], role: str, partition: str
) -> Dict[str, Any]:
    record = {
        "content_hash": row["content_hash"],
        "normalized_prompt_hash": row["normalized_prompt_hash"],
        "partition": partition,
        "raw_row_index": int(row["raw_row_index"]),
        "role": role,
        "source_id": row["source_id"],
        "split_rank": int(row["split_rank"]),
    }
    record["record_sha256"] = sha256_bytes(canonical_json_bytes(record))
    return record


def build_dataset_entry(
    *,
    role: str,
    dataset_id: str,
    source: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> Dict[str, Any]:
    selected = prepared.get("dataset")
    if not isinstance(selected, Dataset):
        raise TypeError("prepared.dataset must be a datasets.Dataset")
    processing_counts = prepared.get("counts")
    if not isinstance(processing_counts, dict):
        raise TypeError("prepared.counts must be a dictionary")
    counts = role_partition_counts(role, len(selected))
    if counts["selected_total"] != len(selected):
        raise ValueError("prepared selection length does not match frozen counts")

    records = []
    partition_records: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in PARTITIONS
    }
    for row in selected:
        partition = partition_for_rank(role, int(row["split_rank"]), counts)
        record = _record_from_row(row, role, partition)
        records.append(record)
        partition_records[partition].append(record)
    partition_sha256 = {
        name: sha256_bytes(canonical_json_bytes(partition_records[name]))
        for name in PARTITIONS
    }
    if len(partition_records["probe"]) < PROBE_EFFECTIVE_TOTAL_PER_ROLE:
        raise ValueError(
            "Probe reservoir cannot provide the frozen 1405-row effective subset"
        )
    effective_probe_records = partition_records["probe"][
        :PROBE_EFFECTIVE_TOTAL_PER_ROLE
    ]
    return {
        "active": True,
        "client_count": ROLE_PROTOCOL[role]["client_count"],
        "counts": {
            **copy.deepcopy(processing_counts),
            **counts,
        },
        "dataset_id": dataset_id,
        "fingerprints": copy.deepcopy(prepared.get("fingerprints")),
        "partition_sha256": partition_sha256,
        "probe_client_pool_sizes": probe_client_pool_sizes(counts["probe"]),
        "probe_effective_client_pool_sizes": probe_effective_client_pool_sizes(),
        "probe_effective_count": PROBE_EFFECTIVE_TOTAL_PER_ROLE,
        "probe_effective_records_sha256": sha256_bytes(
            canonical_json_bytes(effective_probe_records)
        ),
        "records": records,
        "records_sha256": sha256_bytes(canonical_json_bytes(records)),
        "role": role,
        "source": copy.deepcopy(dict(source)),
    }


def build_task_manifest(
    *,
    task_id: int,
    benign_entry: Mapping[str, Any],
    malicious_entry: Mapping[str, Any],
    code_commit: str,
    processing_runtime: Mapping[str, str],
    protocol_path: str,
    protocol_sha256: str,
) -> Dict[str, Any]:
    manifest = {
        "code_commit": code_commit,
        "datasets": [copy.deepcopy(dict(benign_entry)), copy.deepcopy(dict(malicious_entry))],
        "experiment": {
            "benign_clients": 7,
            "federated_num_data_per_client": 500,
            "malicious_clients": 3,
            "probe_clients_per_role": 5,
            "probe_effective_count_per_role": PROBE_EFFECTIVE_TOTAL_PER_ROLE,
            "probe_min_samples_per_selected_client": 160,
            "probe_target_data_per_client": 500,
        },
        "manifest_schema": SCHEMA_VERSION,
        "processing_order": [
            "english_single_turn_screen",
            "sft_format",
            "semantic_role_filter",
            "content_and_normalized_prompt_dedup",
            "global_cross_task_dedup",
            "deterministic_shuffle_seed_2023",
            "stratified_70_10_20_then_probe",
        ],
        "processing_runtime": copy.deepcopy(dict(processing_runtime)),
        "protocol": {
            "cross_task_dedup": {
                "fields": list(IDENTITY_FIELDS),
                "priority": "Task1->Task4; within each Task benign->malicious",
                "scope": "exclude identities already selected into any earlier pool",
            },
            "formal_test_use": "only_after_protocol_and_checkpoints_are_frozen",
            "integer_allocation": {
                "benign": "3500/500/1000 (exact 70/10/20)",
                "malicious": "1500/214/429 from ceil(1500/0.70)=2143",
            },
            "path": protocol_path,
            "probe_disjoint": True,
            "sha256": protocol_sha256,
            "split_ratio": [7, 1, 2],
            "validation_use": (
                "diagnostics_only; no early stopping, best-checkpoint selection, "
                "Probe threshold adjustment, or round control"
            ),
        },
        "shuffle": {
            "implementation": "datasets.Dataset.shuffle",
            "seed": SHUFFLE_SEED,
        },
        "task_id": task_id,
    }
    validate_task_manifest(manifest)
    return manifest


def _validate_digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or not HEX_64.fullmatch(value):
        raise ValueError(name + " must be a lowercase SHA-256")


def _expected_source_id(entry: Mapping[str, Any], raw_row_index: int) -> str:
    if entry.get('source',{}).get('kind') == 'verified_risk5_jsonl':
        from .risk5_data import materialized_rows
        return materialized_rows(entry['source'])[raw_row_index]['source_id']
    source = entry.get("source")
    if not isinstance(source, dict):
        raise ValueError("dataset source metadata is required")
    kind = source.get("kind")
    validate_dataset_source_identity(
        entry.get("dataset_id"),
        entry.get("role"),
        kind,
        source.get("split"),
        source.get("json_relative_path"),
    )
    if kind == "huggingface":
        revision = source.get("resolved_revision")
        if source.get("requested_revision") != revision:
            raise ValueError("requested and resolved revisions must be the same pinned SHA")
        return make_hf_source_id(
            entry["dataset_id"], revision, source["split"], raw_row_index
        )
    if kind == "local_json":
        return make_json_source_id(source.get("json_sha256"), raw_row_index)
    raise ValueError("unsupported source kind")


def validate_task_manifest(manifest: Mapping[str, Any]) -> None:
    if not matches_schema(manifest.get("manifest_schema"), SCHEMA_VERSION):
        raise ValueError("unsupported continual task manifest schema")
    if not isinstance(manifest.get("task_id"), int) or manifest["task_id"] not in range(1, 6):
        raise ValueError("task_id must be 1..5")
    if not isinstance(manifest.get("code_commit"), str) or not HEX_40.fullmatch(manifest["code_commit"]):
        raise ValueError("code_commit must be a lowercase Git commit SHA")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("split_ratio") != [7, 1, 2]:
        raise ValueError("manifest must freeze split_ratio 7:1:2")
    _validate_digest(protocol.get("sha256"), "protocol.sha256")
    if protocol.get("probe_disjoint") is not True:
        raise ValueError("Probe must be frozen as disjoint")
    if not isinstance(protocol.get("path"), str) or not protocol["path"]:
        raise ValueError("protocol.path is required")
    runtime = manifest.get("processing_runtime")
    if not isinstance(runtime, dict):
        raise ValueError("processing_runtime is required")
    for key in RUNTIME_KEYS:
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise ValueError("processing_runtime." + key + " is required")
    if manifest.get("experiment") != {
        "benign_clients": 7,
        "federated_num_data_per_client": 500,
        "malicious_clients": 3,
        "probe_clients_per_role": 5,
        "probe_effective_count_per_role": PROBE_EFFECTIVE_TOTAL_PER_ROLE,
        "probe_min_samples_per_selected_client": 160,
        "probe_target_data_per_client": 500,
    }:
        raise ValueError("continual experiment metadata mismatch")
    if manifest.get("shuffle") != {
        "implementation": "datasets.Dataset.shuffle",
        "seed": SHUFFLE_SEED,
    }:
        raise ValueError("shuffle configuration is not frozen")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise ValueError("manifest requires benign and malicious entries")

    task_seen = {field: set() for field in IDENTITY_FIELDS}
    for expected_role, entry in zip(("benign", "malicious"), datasets):
        if entry.get("role") != expected_role:
            raise ValueError("dataset entries must be ordered benign then malicious")
        expected_dataset = TASK_DATASETS[manifest["task_id"]][expected_role]
        if entry.get("dataset_id") != expected_dataset:
            raise ValueError("dataset_id does not match frozen Task mapping")
        if entry.get("active") is not True:
            raise ValueError("continual dataset entries must be active")
        if entry.get("client_count") != ROLE_PROTOCOL[expected_role]["client_count"]:
            raise ValueError("dataset client_count mismatch")
        counts = entry.get("counts")
        records = entry.get("records")
        if not isinstance(counts, dict) or not isinstance(records, list):
            raise ValueError("dataset counts and records are required")
        expected_counts = role_partition_counts(expected_role, len(records))
        for name in PARTITIONS:
            if counts.get(name) != expected_counts[name]:
                raise ValueError(f"{expected_role} {name} count mismatch")
        if counts.get("selected_total") != len(records):
            raise ValueError("selected_total does not match record count")
        if counts.get("federated_minimum_satisfied") is not True:
            raise ValueError("federated minimum must be satisfied")
        expected_pool_sizes = probe_client_pool_sizes(counts["probe"])
        if entry.get("probe_client_pool_sizes") != expected_pool_sizes:
            raise ValueError("Probe client pool sizes mismatch")
        if min(expected_pool_sizes, default=0) < 160:
            raise ValueError("each Probe client pool must contain at least 160 rows")
        if counts["probe"] < PROBE_EFFECTIVE_TOTAL_PER_ROLE:
            raise ValueError("Probe reservoir is smaller than effective count 1405")
        if entry.get("probe_effective_count") != PROBE_EFFECTIVE_TOTAL_PER_ROLE:
            raise ValueError("Probe effective count must be 1405")
        if entry.get(
            "probe_effective_client_pool_sizes"
        ) != probe_effective_client_pool_sizes():
            raise ValueError("Probe effective client pools must be 281 x 5")
        expected_ranks = list(range(len(records)))
        if [record.get("split_rank") for record in records] != expected_ranks:
            raise ValueError("split ranks must be contiguous and ordered")
        grouped = {name: [] for name in PARTITIONS}
        for record in records:
            partition = partition_for_rank(expected_role, record["split_rank"], counts)
            if record.get("partition") != partition or record.get("role") != expected_role:
                raise ValueError("record role/partition does not match rank")
            copy_without_sha = dict(record)
            record_sha = copy_without_sha.pop("record_sha256", None)
            _validate_digest(record_sha, "record_sha256")
            if sha256_bytes(canonical_json_bytes(copy_without_sha)) != record_sha:
                raise ValueError("record_sha256 mismatch")
            for field in IDENTITY_FIELDS:
                value = record.get(field)
                if not isinstance(value, str) or not value:
                    raise ValueError(field + " must be non-empty")
                if field != "source_id":
                    _validate_digest(value, field)
                if field != "normalized_prompt_hash" and value in task_seen[field]:
                    raise ValueError("task-level overlap in " + field)
                task_seen[field].add(value)
            if record["source_id"] != _expected_source_id(
                entry, record.get("raw_row_index")
            ):
                raise ValueError("record source_id does not match pinned source metadata")
            grouped[partition].append(record)
        effective_probe_records = grouped["probe"][:PROBE_EFFECTIVE_TOTAL_PER_ROLE]
        expected_effective_sha = sha256_bytes(
            canonical_json_bytes(effective_probe_records)
        )
        if entry.get("probe_effective_records_sha256") != expected_effective_sha:
            raise ValueError("Probe effective records SHA mismatch")
        if sha256_bytes(canonical_json_bytes(records)) != entry.get("records_sha256"):
            raise ValueError("records_sha256 mismatch")
        partition_sha = entry.get("partition_sha256")
        if not isinstance(partition_sha, dict):
            raise ValueError("partition_sha256 is required")
        for name in PARTITIONS:
            expected_sha = sha256_bytes(canonical_json_bytes(grouped[name]))
            if partition_sha.get(name) != expected_sha:
                raise ValueError("partition SHA mismatch: " + name)


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_task_manifest(manifest)
    return sha256_bytes(canonical_json_bytes(manifest))


def load_task_manifest(path: str, expected_sha256: str) -> Dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError("continual task manifest does not exist")
    if sha256_file(target) != str(expected_sha256).lower():
        raise ValueError("continual task manifest SHA-256 mismatch")
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical_json_bytes(value):
        raise ValueError("continual task manifest is not canonical JSON")
    validate_task_manifest(value)
    return value


def identity_sets_from_entries(entries: Iterable[Mapping[str, Any]]) -> Dict[str, Set[str]]:
    result = {field: set() for field in IDENTITY_FIELDS}
    for entry in entries:
        for record in entry.get("records", []):
            for field in IDENTITY_FIELDS:
                value = record[field]
                if value in result[field]:
                    raise ValueError("global overlap in " + field)
                result[field].add(value)
    return result


def extend_identity_sets(
    target: MutableMapping[str, Set[str]], entries: Iterable[Mapping[str, Any]]
) -> None:
    incoming = identity_sets_from_entries(entries)
    for field in IDENTITY_FIELDS:
        overlap = target[field].intersection(incoming[field])
        if overlap:
            raise ValueError("global overlap in " + field + ": " + repr(sorted(overlap)[:3]))
        target[field].update(incoming[field])


def build_global_audit(
    task_manifests: Sequence[Mapping[str, Any]],
    task_manifest_sha256: Sequence[str],
    code_commit: str,
) -> Dict[str, Any]:
    if len(task_manifests) != 4 or len(task_manifest_sha256) != 4:
        raise ValueError("global audit requires exactly four task manifests")
    global_seen = {field: set() for field in IDENTITY_FIELDS}
    rows = []
    for task_id, (manifest, digest) in enumerate(
        zip(task_manifests, task_manifest_sha256), start=1
    ):
        validate_task_manifest(manifest)
        if manifest["task_id"] != task_id:
            raise ValueError("task manifests must be ordered Task1..Task4")
        _validate_digest(digest, "task manifest SHA-256")
        if manifest_sha256(manifest) != digest:
            raise ValueError("task manifest SHA-256 does not match manifest content")
        extend_identity_sets(global_seen, manifest["datasets"])
        row = {"task_id": task_id, "manifest_sha256": digest}
        for entry in manifest["datasets"]:
            role = entry["role"]
            for partition in PARTITIONS:
                row[role + "_" + partition] = entry["counts"][partition]
            row[role + "_minimum_satisfied"] = entry["counts"][
                "federated_minimum_satisfied"
            ]
        rows.append(row)
    return {
        "code_commit": code_commit,
        "global_identity_counts": {
            field: len(values) for field, values in global_seen.items()
        },
        "overlap_checks": {
            "cross_partition": "PASS",
            "cross_role": "PASS",
            "cross_task": "PASS",
            "fields": list(IDENTITY_FIELDS),
        },
        "schema": AUDIT_SCHEMA_VERSION,
        "task_statistics": rows,
    }

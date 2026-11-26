import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from utils.dataset_partitioning import (
    FEDERATED_NUM_DATA_PER_CLIENT,
    PROBE_CLIENTS_PER_ROLE,
    PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT,
    PROBE_TARGET_DATA_PER_CLIENT,
    build_partition_counts,
)

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - compatibility fallback
    import importlib_metadata  # type: ignore


PathValue = Union[str, Path]

MANIFEST_SCHEMA_VERSION = 2
PIPELINE_SCHEMA_VERSION = 2
SHUFFLE_SEED = 2023

_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_40_ANY_CASE_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_MANIFEST_DIGEST_KEYS = {
    "manifest_digest",
    "manifest_sha256",
}
_RUNTIME_VERSION_KEYS = (
    "python_version",
    "datasets_version",
    "numpy_version",
    "pandas_version",
    "pyarrow_version",
)
_COUNT_KEYS = (
    "raw",
    "after_semantic_filter",
    "after_format_and_dedup",
    "federated_required",
    "probe_exclusion_boundary",
    "probe_target",
    "probe_actual",
    "excluded",
    "partition_scan_end",
    "manifest_record_count",
)
_KNOWN_HF_DATASET_SPLITS = {
    "lmsys/lmsys-chat-1m": "train",
    "allenai/WildChat": "train",
    "PKU-Alignment/BeaverTails": "30k_train",
    "allenai/wildjailbreak": "train",
    "allenai/wildguardmix": "train",
    "PKU-Alignment/PKU-SafeRLHF-QA": "train",
}
_MALICIOUSGEN_JSON_RELATIVE_PATH = "Mistral/maliciousQA.json"


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value to the canonical UTF-8 form."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


def sha256_file(
    path: PathValue,
    chunk_size: int = 1024 * 1024,
) -> str:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_non_negative_index(raw_row_index: int) -> None:
    if (
        not isinstance(raw_row_index, int)
        or isinstance(raw_row_index, bool)
        or raw_row_index < 0
    ):
        raise ValueError("raw_row_index must be a non-negative integer")


def make_hf_source_id(
    dataset_id: str,
    resolved_revision: str,
    split: str,
    raw_row_index: int,
) -> str:
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    if not isinstance(resolved_revision, str) or not _HEX_40_RE.fullmatch(
        resolved_revision
    ):
        raise ValueError("resolved_revision must be a lowercase 40-character SHA")
    _validate_non_negative_index(raw_row_index)
    return canonical_json_bytes(
        ["hf", dataset_id, resolved_revision, split, raw_row_index]
    ).decode("utf-8")


def make_json_source_id(
    json_sha256: str,
    raw_row_index: int,
) -> str:
    if not isinstance(json_sha256, str) or not _HEX_64_RE.fullmatch(json_sha256):
        raise ValueError("json_sha256 must be a lowercase 64-character SHA-256")
    _validate_non_negative_index(raw_row_index)
    return canonical_json_bytes(
        ["json", json_sha256, raw_row_index]
    ).decode("utf-8")


def compute_content_hash(
    instruction: str,
    response: str,
) -> str:
    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "instruction": instruction,
                "response": response,
            }
        )
    )


def normalize_prompt(text: str) -> str:
    """Return the frozen prompt normalization used by all data manifests."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt must be a non-empty string")
    value = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", value)


def compute_normalized_prompt_hash(text: str) -> str:
    """Hash a prompt after the protocol's NFKC/case/space normalization."""
    return sha256_bytes(normalize_prompt(text).encode("utf-8"))


def validate_dataset_source_identity(
    dataset_id: str,
    role: str,
    source_kind: Optional[str],
    split: Optional[str],
    json_relative_path: Optional[str] = None,
) -> None:
    """Validate the offline identity relationship for a dataset source."""
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")
    if role not in ("benign", "malicious"):
        raise ValueError("dataset role must be benign or malicious")

    if dataset_id == "MaliciousGen":
        if role != "malicious":
            raise ValueError("MaliciousGen is only valid for the malicious role")
        if source_kind is None:
            return
        if source_kind != "local_json":
            raise ValueError("MaliciousGen must use a local_json source")
        if split != "train":
            raise ValueError("MaliciousGen source split must be train")
        if json_relative_path != _MALICIOUSGEN_JSON_RELATIVE_PATH:
            raise ValueError(
                "json_relative_path must be exactly "
                + _MALICIOUSGEN_JSON_RELATIVE_PATH
            )
        return

    if source_kind is None:
        return
    if source_kind != "huggingface":
        if source_kind == "local_json":
            raise ValueError(
                "local_json source is only valid for MaliciousGen"
            )
        raise ValueError(
            "non-MaliciousGen datasets must use a huggingface source"
        )
    if not isinstance(split, str) or not split:
        raise ValueError("Hugging Face split must be a non-empty string")

    expected_split = _KNOWN_HF_DATASET_SPLITS.get(dataset_id)
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"{dataset_id} source split must be {expected_split}"
        )


def _create_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve dataset revisions"
        ) from exc
    return HfApi()


def resolve_hf_revision(
    dataset_id: str,
    requested_revision: str,
) -> str:
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")
    if not isinstance(requested_revision, str) or not requested_revision:
        raise ValueError("requested_revision must be a non-empty string")

    try:
        info = _create_hf_api().dataset_info(
            repo_id=dataset_id,
            revision=requested_revision,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to resolve Hugging Face dataset revision for "
            + dataset_id
            + ": "
            + requested_revision
        ) from exc

    resolved_value = getattr(info, "sha", None)
    if not isinstance(resolved_value, str) or not _HEX_40_ANY_CASE_RE.fullmatch(
        resolved_value
    ):
        raise ValueError(
            "resolved Hugging Face revision must be a 40-character commit SHA"
        )

    resolved_revision = resolved_value.lower()
    if _HEX_40_ANY_CASE_RE.fullmatch(requested_revision):
        if requested_revision.lower() != resolved_revision:
            raise ValueError(
                "requested commit SHA does not match the resolved dataset commit"
            )
    return resolved_revision


def _run_git(repository_root: Path, arguments: Sequence[str]) -> str:
    command = ["git", "-C", str(repository_root)] + list(arguments)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError) as exc:
        raise RuntimeError("Git is unavailable") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "Git command failed: "
            + " ".join(command)
            + (": " + detail if detail else "")
        )
    return result.stdout.strip()


def get_clean_repository_commit(
    repository_root: PathValue,
) -> str:
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise RuntimeError("repository_root is not a directory: " + str(root))

    reported_root = _run_git(root, ["rev-parse", "--show-toplevel"])
    try:
        resolved_reported_root = Path(reported_root).resolve()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("Git returned an invalid repository root") from exc
    if resolved_reported_root != root:
        raise RuntimeError(
            "build_split_manifest must run from the repository root: " + str(root)
        )

    commit = _run_git(root, ["rev-parse", "HEAD"]).lower()
    if not _HEX_40_RE.fullmatch(commit):
        raise RuntimeError("Git HEAD is not a 40-character commit SHA")

    status = _run_git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )
    if status:
        raise RuntimeError("Git worktree must be clean before building a manifest")
    return commit


def collect_processing_runtime() -> Dict[str, str]:
    distribution_names = {
        "datasets_version": "datasets",
        "numpy_version": "numpy",
        "pandas_version": "pandas",
        "pyarrow_version": "pyarrow",
    }
    runtime = {
        "python_version": platform.python_version(),
    }
    for key, distribution_name in distribution_names.items():
        try:
            version = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "required distribution is not installed: " + distribution_name
            ) from exc
        if not isinstance(version, str) or not version:
            raise RuntimeError(
                "failed to determine distribution version: " + distribution_name
            )
        runtime[key] = version
    return runtime


def validate_partition_disjointness(
    manifest: Dict[str, Any],
) -> None:
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")

    probe_source_ids: Set[str] = set()
    federated_source_ids: Set[str] = set()
    probe_content_hashes: Set[str] = set()
    federated_content_hashes: Set[str] = set()
    seen_source_ids: Set[str] = set()

    datasets = manifest.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("manifest datasets must be a list")

    for dataset_entry in datasets:
        if not isinstance(dataset_entry, dict):
            raise ValueError("each dataset entry must be a dictionary")
        records = dataset_entry.get("records", [])
        if not isinstance(records, list):
            raise ValueError("dataset records must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("each manifest record must be a dictionary")
            partition = record.get("partition")
            source_id = record.get("source_id")
            content_hash = record.get("content_hash")
            if partition not in ("probe", "federated"):
                raise ValueError("record partition must be probe or federated")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("record source_id must be a non-empty string")
            if not isinstance(content_hash, str) or not _HEX_64_RE.fullmatch(
                content_hash
            ):
                raise ValueError(
                    "record content_hash must be a lowercase 64-character SHA-256"
                )
            if source_id in seen_source_ids:
                raise ValueError(
                    "duplicate source_id in manifest: " + repr(source_id)
                )
            seen_source_ids.add(source_id)
            if partition == "probe":
                probe_source_ids.add(source_id)
                probe_content_hashes.add(content_hash)
            else:
                federated_source_ids.add(source_id)
                federated_content_hashes.add(content_hash)

    source_overlap = probe_source_ids.intersection(federated_source_ids)
    if source_overlap:
        examples = sorted(source_overlap)[:3]
        raise ValueError(
            "probe and federated source_id overlap: " + repr(examples)
        )

    content_overlap = probe_content_hashes.intersection(federated_content_hashes)
    if content_overlap:
        examples = sorted(content_overlap)[:3]
        raise ValueError(
            "probe and federated content_hash overlap: " + repr(examples)
        )


def _require_non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(name + " must be a non-negative integer")
    return value


def _validate_source(
    dataset_entry: Dict[str, Any],
) -> None:
    active = dataset_entry["active"]
    source = dataset_entry.get("source")
    if not active:
        validate_dataset_source_identity(
            dataset_entry["dataset_id"],
            dataset_entry["role"],
            None,
            None,
        )
        if source is not None:
            raise ValueError("inactive dataset source must be null")
        return
    if not isinstance(source, dict):
        raise ValueError("active dataset source must be a dictionary")

    source_kind = source.get("kind")
    validate_dataset_source_identity(
        dataset_entry["dataset_id"],
        dataset_entry["role"],
        source_kind,
        source.get("split"),
        source.get("json_relative_path"),
    )
    if source_kind == "huggingface":
        requested_revision = source.get("requested_revision")
        resolved_revision = source.get("resolved_revision")
        if not isinstance(requested_revision, str) or not requested_revision:
            raise ValueError("requested_revision must be a non-empty string")
        if not isinstance(resolved_revision, str) or not _HEX_40_RE.fullmatch(
            resolved_revision
        ):
            raise ValueError(
                "resolved_revision must be a lowercase 40-character SHA"
            )
        if _HEX_40_ANY_CASE_RE.fullmatch(requested_revision):
            if requested_revision.lower() != resolved_revision:
                raise ValueError(
                    "requested commit SHA does not match resolved_revision"
                )
    elif source_kind == "local_json":
        json_sha256 = source.get("json_sha256")
        if not isinstance(json_sha256, str) or not _HEX_64_RE.fullmatch(
            json_sha256
        ):
            raise ValueError(
                "local JSON source must contain a lowercase SHA-256"
            )
    else:
        raise ValueError("unsupported dataset source kind")


def _expected_source_id(
    dataset_entry: Dict[str, Any],
    raw_row_index: int,
) -> str:
    source = dataset_entry["source"]
    if source["kind"] == "huggingface":
        return make_hf_source_id(
            dataset_entry["dataset_id"],
            source["resolved_revision"],
            source["split"],
            raw_row_index,
        )
    return make_json_source_id(
        source["json_sha256"],
        raw_row_index,
    )


def _validate_dataset_entry(
    dataset_entry: Dict[str, Any],
    expected_role: str,
    federated_num_data_per_client: int,
) -> None:
    if not isinstance(dataset_entry, dict):
        raise ValueError("dataset entry must be a dictionary")
    if dataset_entry.get("role") != expected_role:
        raise ValueError("dataset roles must be ordered as benign then malicious")

    dataset_id = dataset_entry.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")
    active = dataset_entry.get("active")
    if not isinstance(active, bool):
        raise ValueError("dataset active must be a boolean")
    client_count = _require_non_negative_int(
        dataset_entry.get("client_count"),
        expected_role + " client_count",
    )
    if active != (client_count > 0):
        raise ValueError("dataset active flag must equal client_count > 0")

    _validate_source(dataset_entry)

    counts = dataset_entry.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("dataset counts must be a dictionary")
    for count_key in _COUNT_KEYS:
        _require_non_negative_int(
            counts.get(count_key),
            expected_role + " counts." + count_key,
        )

    pool_sizes = dataset_entry.get("probe_client_pool_sizes")
    if not isinstance(pool_sizes, list):
        raise ValueError("probe_client_pool_sizes must be a list")
    if active:
        if counts["raw"] < counts["after_semantic_filter"]:
            raise ValueError("raw count must be >= after_semantic_filter")
        if (
            counts["after_semantic_filter"]
            < counts["after_format_and_dedup"]
        ):
            raise ValueError(
                "after_semantic_filter must be >= after_format_and_dedup"
            )
        expected_partition = build_partition_counts(
            dataset_id=dataset_id,
            role=expected_role,
            role_client_count=client_count,
            federated_num_data_per_client=federated_num_data_per_client,
            after_format_and_dedup=counts["after_format_and_dedup"],
        )
        for key in (
            "federated_required",
            "probe_exclusion_boundary",
            "probe_target",
            "probe_actual",
            "excluded",
            "partition_scan_end",
            "manifest_record_count",
        ):
            if counts[key] != expected_partition[key]:
                raise ValueError(
                    "counts." + key + " does not match the frozen partition"
                )
        if pool_sizes != expected_partition["probe_client_pool_sizes"]:
            raise ValueError(
                "probe_client_pool_sizes do not match Probe actual total"
            )
    else:
        if any(counts[key] != 0 for key in _COUNT_KEYS):
            raise ValueError("inactive dataset counts must all be zero")
        if pool_sizes:
            raise ValueError(
                "inactive dataset probe_client_pool_sizes must be empty"
            )

    fingerprints = dataset_entry.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("dataset fingerprints must be a dictionary")
    if fingerprints.get("policy") != "diagnostic_only":
        raise ValueError("dataset fingerprint policy must be diagnostic_only")
    for fingerprint_key in ("raw", "post_shuffle"):
        fingerprint = fingerprints.get(fingerprint_key)
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise ValueError("diagnostic fingerprints must be strings or null")

    records = dataset_entry.get("records")
    if not isinstance(records, list):
        raise ValueError("dataset records must be a list")
    if len(records) != counts["manifest_record_count"]:
        raise ValueError("record count does not match manifest_record_count")

    expected_ranks = list(range(counts["federated_required"]))
    expected_ranks.extend(
        range(
            counts["probe_exclusion_boundary"],
            counts["partition_scan_end"],
        )
    )
    for expected_rank, record in zip(expected_ranks, records):
        if not isinstance(record, dict):
            raise ValueError("each dataset record must be a dictionary")
        split_rank = _require_non_negative_int(
            record.get("split_rank"),
            "record split_rank",
        )
        if split_rank != expected_rank:
            raise ValueError(
                "records do not match the frozen global split_rank ranges"
            )
        raw_row_index = _require_non_negative_int(
            record.get("raw_row_index"),
            "record raw_row_index",
        )
        expected_partition = (
            "federated"
            if split_rank < counts["federated_required"]
            else "probe"
        )
        if record.get("partition") != expected_partition:
            raise ValueError("record partition does not match split_rank")
        source_id = record.get("source_id")
        if source_id != _expected_source_id(dataset_entry, raw_row_index):
            raise ValueError("record source_id does not match source metadata")
        content_hash = record.get("content_hash")
        if not isinstance(content_hash, str) or not _HEX_64_RE.fullmatch(
            content_hash
        ):
            raise ValueError(
                "record content_hash must be a lowercase 64-character SHA-256"
            )


def _validate_manifest_structure(
    manifest: Dict[str, Any],
) -> None:
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")
    forbidden = _FORBIDDEN_MANIFEST_DIGEST_KEYS.intersection(manifest.keys())
    if forbidden:
        raise ValueError(
            "manifest must not contain its own digest field: "
            + ", ".join(sorted(forbidden))
        )
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported manifest_schema_version")
    if manifest.get("pipeline_schema_version") != PIPELINE_SCHEMA_VERSION:
        raise ValueError("unsupported pipeline_schema_version")
    if manifest.get("strict_content_disjoint") is not True:
        raise ValueError("strict_content_disjoint must be true")

    code_commit = manifest.get("code_commit")
    if not isinstance(code_commit, str) or not _HEX_40_RE.fullmatch(code_commit):
        raise ValueError("code_commit must be a lowercase 40-character SHA")

    runtime = manifest.get("processing_runtime")
    if not isinstance(runtime, dict):
        raise ValueError("processing_runtime must be a dictionary")
    for key in _RUNTIME_VERSION_KEYS:
        value = runtime.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError("processing_runtime." + key + " is required")

    shuffle = manifest.get("shuffle")
    if shuffle != {
        "implementation": "datasets.Dataset.shuffle",
        "seed": SHUFFLE_SEED,
    }:
        raise ValueError("manifest shuffle metadata does not match B1")

    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be a dictionary")
    federated_num_data_per_client = experiment.get(
        "federated_num_data_per_client"
    )
    if (
        not isinstance(federated_num_data_per_client, int)
        or isinstance(federated_num_data_per_client, bool)
        or federated_num_data_per_client <= 0
    ):
        raise ValueError(
            "federated_num_data_per_client must be a positive integer"
        )
    if federated_num_data_per_client != FEDERATED_NUM_DATA_PER_CLIENT:
        raise ValueError(
            "federated_num_data_per_client must be 500"
        )
    if experiment.get("probe_clients_per_role") != PROBE_CLIENTS_PER_ROLE:
        raise ValueError("probe_clients_per_role must be 5")
    if (
        experiment.get("probe_target_data_per_client")
        != PROBE_TARGET_DATA_PER_CLIENT
    ):
        raise ValueError("probe_target_data_per_client must be 500")
    if (
        experiment.get("probe_min_samples_per_selected_client")
        != PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT
    ):
        raise ValueError(
            "probe_min_samples_per_selected_client must be 160"
        )

    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise ValueError("manifest datasets must contain benign and malicious entries")
    _validate_dataset_entry(
        datasets[0],
        "benign",
        federated_num_data_per_client,
    )
    _validate_dataset_entry(
        datasets[1],
        "malicious",
        federated_num_data_per_client,
    )

    benign_entry = datasets[0]
    malicious_entry = datasets[1]
    expected_experiment = {
        "benign_clients": benign_entry["client_count"],
        "benign_dataset": benign_entry["dataset_id"],
        "malicious_clients": malicious_entry["client_count"],
        "malicious_dataset": malicious_entry["dataset_id"],
        "federated_num_data_per_client": federated_num_data_per_client,
        "probe_clients_per_role": PROBE_CLIENTS_PER_ROLE,
        "probe_target_data_per_client": PROBE_TARGET_DATA_PER_CLIENT,
        "probe_min_samples_per_selected_client": (
            PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT
        ),
    }
    if experiment != expected_experiment:
        raise ValueError("experiment metadata does not match dataset entries")


def build_manifest(
    benign_entry: Dict[str, Any],
    malicious_entry: Dict[str, Any],
    federated_num_data_per_client: int,
    code_commit: str,
    processing_runtime: Dict[str, str],
) -> Dict[str, Any]:
    if (
        not isinstance(federated_num_data_per_client, int)
        or isinstance(federated_num_data_per_client, bool)
        or federated_num_data_per_client <= 0
    ):
        raise ValueError(
            "federated_num_data_per_client must be a positive integer"
        )
    if federated_num_data_per_client != FEDERATED_NUM_DATA_PER_CLIENT:
        raise ValueError(
            "federated_num_data_per_client must be 500"
        )
    if not isinstance(code_commit, str) or not _HEX_40_RE.fullmatch(code_commit):
        raise ValueError("code_commit must be a lowercase 40-character SHA")
    if not isinstance(processing_runtime, dict):
        raise TypeError("processing_runtime must be a dictionary")

    benign_copy = copy.deepcopy(benign_entry)
    malicious_copy = copy.deepcopy(malicious_entry)
    manifest = {
        "code_commit": code_commit,
        "datasets": [benign_copy, malicious_copy],
        "experiment": {
            "benign_clients": benign_copy.get("client_count"),
            "benign_dataset": benign_copy.get("dataset_id"),
            "malicious_clients": malicious_copy.get("client_count"),
            "malicious_dataset": malicious_copy.get("dataset_id"),
            "federated_num_data_per_client": (
                federated_num_data_per_client
            ),
            "probe_clients_per_role": PROBE_CLIENTS_PER_ROLE,
            "probe_target_data_per_client": PROBE_TARGET_DATA_PER_CLIENT,
            "probe_min_samples_per_selected_client": (
                PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT
            ),
        },
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "processing_runtime": copy.deepcopy(processing_runtime),
        "shuffle": {
            "implementation": "datasets.Dataset.shuffle",
            "seed": SHUFFLE_SEED,
        },
        "strict_content_disjoint": True,
    }
    _validate_manifest_structure(manifest)
    validate_partition_disjointness(manifest)
    return manifest


def manifest_bytes(
    manifest: Dict[str, Any],
) -> bytes:
    _validate_manifest_structure(manifest)
    validate_partition_disjointness(manifest)
    return canonical_json_bytes(manifest)


def manifest_sha256(
    manifest: Dict[str, Any],
) -> str:
    """Hash the complete manifest artifact, including provenance fields."""
    return sha256_bytes(manifest_bytes(manifest))


def _write_synced_temp(
    target: Path,
    data: bytes,
) -> Path:
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(target.parent),
            prefix="." + target.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _publish_temp_no_clobber(
    temp_path: Path,
    target_path: Path,
) -> None:
    temp_parent = temp_path.parent.resolve()
    target_parent = target_path.parent.resolve()
    if temp_parent != target_parent:
        raise ValueError("temporary file and target must be in the same directory")

    try:
        os.link(str(temp_path), str(target_path))
    except FileExistsError:
        raise
    temp_path.unlink()


def write_manifest(
    manifest: Dict[str, Any],
    path: PathValue,
    digest_path: Optional[PathValue] = None,
) -> str:
    manifest_target = Path(path)
    sidecar_target = (
        Path(digest_path)
        if digest_path is not None
        else Path(str(manifest_target) + ".sha256")
    )
    if manifest_target == sidecar_target:
        raise ValueError("manifest and digest sidecar paths must differ")
    if not manifest_target.parent.is_dir():
        raise FileNotFoundError(
            "manifest parent directory does not exist: "
            + str(manifest_target.parent)
        )
    if not sidecar_target.parent.is_dir():
        raise FileNotFoundError(
            "digest parent directory does not exist: " + str(sidecar_target.parent)
        )
    if manifest_target.exists():
        raise FileExistsError("manifest already exists: " + str(manifest_target))
    if sidecar_target.exists():
        raise FileExistsError(
            "manifest digest sidecar already exists: " + str(sidecar_target)
        )

    serialized = manifest_bytes(manifest)
    digest = sha256_bytes(serialized)
    sidecar_bytes = (digest + "\n").encode("ascii")
    temporary_paths: List[Path] = []

    try:
        manifest_temp = _write_synced_temp(manifest_target, serialized)
        temporary_paths.append(manifest_temp)
        _publish_temp_no_clobber(manifest_temp, manifest_target)

        sidecar_temp = _write_synced_temp(sidecar_target, sidecar_bytes)
        temporary_paths.append(sidecar_temp)
        _publish_temp_no_clobber(sidecar_temp, sidecar_target)
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()

    return digest


def _reject_duplicate_json_keys(
    pairs: Sequence[Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + str(key))
        result[key] = value
    return result


def load_and_validate_manifest(
    path: PathValue,
    expected_sha256: str,
) -> Dict[str, Any]:
    if not isinstance(expected_sha256, str) or not _HEX_64_RE.fullmatch(
        expected_sha256
    ):
        raise ValueError(
            "expected_sha256 must be a lowercase 64-character SHA-256"
        )

    manifest_path = Path(path)
    raw_bytes = manifest_path.read_bytes()
    actual_sha256 = sha256_bytes(raw_bytes)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "manifest SHA-256 mismatch: expected "
            + expected_sha256
            + ", got "
            + actual_sha256
        )

    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest is not valid UTF-8") from exc
    try:
        manifest = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be a JSON object")

    canonical_bytes = manifest_bytes(manifest)
    if raw_bytes != canonical_bytes:
        raise ValueError("manifest JSON is not in canonical byte form")
    return manifest

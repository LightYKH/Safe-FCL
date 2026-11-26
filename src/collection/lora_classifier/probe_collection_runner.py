"""CPU-testable orchestration primitives for a planned Probe collection run."""

import copy
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset

from lora_classifier.probe_collection_plan import (
    validate_probe_collection_plan,
)
from lora_classifier.probe_delta_core import (
    build_probe_delta_record_metadata,
    compute_delta_tensor_payload_sha256,
    compute_probe_history_digest,
    validate_probe_step_samples,
)
from utils.data_manifest import canonical_json_bytes
from utils.frozen_continual_schedule import frozen_cosine_lr
from utils.lora_state_guard import canonical_lora_tensor_sha256
from utils.dataset_partitioning import (
    PROBE_CLIENTS_PER_ROLE,
    PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT,
)


_CLIENT_COUNT = 10
_CLIENTS_PER_ROLE = PROBE_CLIENTS_PER_ROLE
_ROUND_COUNT = 10
_CLIENTS_PER_ROUND = 3
_LOCAL_SAMPLE_COUNT = 160
_OPTIMIZER_STEPS = 10
_STEP_SAMPLE_COUNT = 16
_ROLE_SHUFFLE_SEED = 2023
_SAMPLE_FIELDS = (
    "sample_id",
    "source_id",
    "content_hash",
    "raw_row_index",
    "split_rank",
    "probe_role",
    "probe_label",
)
_PLAN_SAMPLE_FIELDS = _SAMPLE_FIELDS + ("normalized_prompt_hash",)
_DATASET_FIELDS = ("instruction", "response") + _SAMPLE_FIELDS
_PROVENANCE_FIELDS = (
    "manifest_sha256",
    "code_commit",
    "model_artifact_sha256",
    "tokenizer_artifact_sha256",
)


def _require_non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(name + " must be a non-negative integer")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(name + " must be a lowercase SHA-256")
    return value


def _require_exact_keys(mapping: Any, expected: set, name: str) -> Mapping:
    if not isinstance(mapping, Mapping):
        raise TypeError(name + " must be a Mapping")
    if set(mapping.keys()) != expected:
        raise ValueError(name + " keys do not match the required set")
    return mapping


def _plan_client(plan: Dict[str, Any], client_id: int) -> Dict[str, Any]:
    client = plan["clients"][client_id]
    if client["client_id"] != client_id:
        raise ValueError("plan client lookup is inconsistent")
    return client


def _plan_round(plan: Dict[str, Any], round_index: int) -> Dict[str, Any]:
    value = _require_non_negative_int(round_index, "round_index")
    if value >= _ROUND_COUNT:
        raise ValueError("round_index must be in range 0..9")
    item = plan["rounds"][value]
    if item["round_index"] != value:
        raise ValueError("plan round lookup is inconsistent")
    return item


def _normalize_dataset_identity(
    row: Mapping,
    *,
    expected_role: str,
    expected_label: int,
) -> Dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError("Dataset row must be a Mapping")
    missing = [field for field in _DATASET_FIELDS if field not in row]
    if missing:
        raise ValueError("Dataset row is missing fields: " + repr(missing))
    if not isinstance(row["instruction"], str):
        raise TypeError("instruction must be a string")
    if not isinstance(row["response"], str):
        raise TypeError("response must be a string")
    if not row["response"]:
        raise ValueError("response must be non-empty")

    sample = {field: row[field] for field in _SAMPLE_FIELDS}
    if "normalized_prompt_hash" in row:
        sample["normalized_prompt_hash"] = row["normalized_prompt_hash"]
    normalized = validate_probe_step_samples(
        [sample],
        expected_role,
        expected_label,
    )[0]
    return normalized


def _assert_identity_matches_plan(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for field in _PLAN_SAMPLE_FIELDS:
        if actual[field] != expected[field]:
            raise ValueError(
                context + " identity does not match plan field " + field
            )


def _validate_pool(
    dataset: Dataset,
    *,
    role: str,
    label: int,
    split_rank_start: int,
    split_rank_end: int,
) -> set:
    if not isinstance(dataset, Dataset):
        raise TypeError("each Probe pool must be a datasets.Dataset")
    if len(dataset) != split_rank_end - split_rank_start:
        raise ValueError("Probe pool length does not match plan range")
    missing_columns = [
        field for field in _DATASET_FIELDS if field not in dataset.column_names
    ]
    if missing_columns:
        raise ValueError(
            "Probe pool is missing columns: " + repr(missing_columns)
        )

    source_ids = set()
    for expected_rank, row in zip(
        range(split_rank_start, split_rank_end),
        dataset,
    ):
        identity = _normalize_dataset_identity(
            row,
            expected_role=role,
            expected_label=label,
        )
        if identity["split_rank"] != expected_rank:
            raise ValueError(
                "Probe pool split_rank does not match the global plan range"
            )
        if identity["sample_id"] != expected_rank:
            raise ValueError("Probe pool sample_id must equal split_rank")
        source_id = identity["source_id"]
        if source_id in source_ids:
            raise ValueError("Probe pool source_id values must be unique")
        source_ids.add(source_id)
    return source_ids


def _validate_client_dataset_mapping(
    client_datasets: Any,
    plan: Dict[str, Any],
) -> Mapping:
    mapping = _require_exact_keys(
        client_datasets,
        set(range(_CLIENT_COUNT)),
        "client_datasets",
    )
    for client_id in range(_CLIENT_COUNT):
        dataset = mapping[client_id]
        if not isinstance(dataset, Dataset):
            raise TypeError("each client dataset must be a datasets.Dataset")
        expected_size = plan["clients"][client_id]["sample_count"]
        if len(dataset) != expected_size:
            raise ValueError(
                "client dataset length does not match planned pool size"
            )
        if len(dataset) < PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT:
            raise ValueError(
                "each Probe client pool must contain at least 160 rows"
            )
    return mapping


def bind_probe_pools_to_plan(
    plan,
    probe_pools,
) -> Dict[int, Dataset]:
    """Bind verified role pools to the exact client order stored in the plan."""
    validated_plan = validate_probe_collection_plan(plan)
    if isinstance(probe_pools, (str, bytes)) or not isinstance(
        probe_pools, Sequence
    ):
        raise TypeError("probe_pools must be a sequence")
    if len(probe_pools) != 2:
        raise ValueError("probe_pools must contain benign and malicious pools")

    split_ranges = validated_plan["manifest"]["probe_split_rank_ranges"]
    benign_sources = _validate_pool(
        probe_pools[0],
        role="benign",
        label=0,
        split_rank_start=split_ranges["benign"][0],
        split_rank_end=split_ranges["benign"][1],
    )
    malicious_sources = _validate_pool(
        probe_pools[1],
        role="malicious",
        label=1,
        split_rank_start=split_ranges["malicious"][0],
        split_rank_end=split_ranges["malicious"][1],
    )
    overlap = benign_sources & malicious_sources
    if overlap:
        raise ValueError("benign and malicious source_id values overlap")

    result: Dict[int, Dataset] = {}
    all_client_sources = set()
    assigned_by_role = {"benign": set(), "malicious": set()}
    for role_index, (role, label, client_offset) in enumerate(
        (("benign", 0, 0), ("malicious", 1, 5))
    ):
        shuffled = probe_pools[role_index].shuffle(seed=_ROLE_SHUFFLE_SEED)
        for shard_index in range(_CLIENTS_PER_ROLE):
            client_id = client_offset + shard_index
            shard = shuffled.shard(
                num_shards=_CLIENTS_PER_ROLE,
                index=shard_index,
            )
            plan_client = _plan_client(validated_plan, client_id)
            if len(shard) != plan_client["sample_count"]:
                raise ValueError(
                    "client shard length does not match planned pool size"
                )
            shard_sources = set()
            for local_position, row in enumerate(shard):
                identity = _normalize_dataset_identity(
                    row,
                    expected_role=role,
                    expected_label=label,
                )
                plan_sample = plan_client["samples"][local_position]
                if plan_sample["local_position"] != local_position:
                    raise ValueError(
                        "plan local_position does not match Dataset row"
                    )
                _assert_identity_matches_plan(
                    identity,
                    plan_sample,
                    context="client " + str(client_id),
                )
                source_id = identity["source_id"]
                if source_id in shard_sources:
                    raise ValueError("client shard source_id is duplicated")
                if source_id in all_client_sources:
                    raise ValueError(
                        "client source_id values must be globally unique"
                    )
                shard_sources.add(source_id)
                all_client_sources.add(source_id)
            assigned_by_role[role].update(shard_sources)
            result[client_id] = shard

    if assigned_by_role["benign"] != benign_sources:
        raise ValueError("benign client shard union does not match role pool")
    if assigned_by_role["malicious"] != malicious_sources:
        raise ValueError("malicious client shard union does not match role pool")
    if set(result) != set(range(_CLIENT_COUNT)):
        raise ValueError("Probe client binding must create clients 0..9")
    return result


def select_planned_round_client_dataset(
    plan,
    client_datasets,
    round_index,
    client_id,
) -> Dataset:
    """Select the plan-authoritative 160 rows without sampling or reordering."""
    validated_plan = validate_probe_collection_plan(plan)
    round_plan = _plan_round(validated_plan, round_index)
    client_value = _require_non_negative_int(client_id, "client_id")
    if client_value >= _CLIENT_COUNT:
        raise ValueError("client_id must be in range 0..9")
    if client_value not in round_plan["selected_client_ids"]:
        raise ValueError("client_id is not selected in the requested round")

    datasets_by_client = _validate_client_dataset_mapping(
        client_datasets,
        validated_plan,
    )
    client_dataset = datasets_by_client[client_value]
    positions = round_plan["local_sample_positions"]
    if len(positions) != _LOCAL_SAMPLE_COUNT or len(set(positions)) != len(
        positions
    ):
        raise ValueError("round plan must contain 160 unique positions")

    selected = client_dataset.select(positions)
    if len(selected) != _LOCAL_SAMPLE_COUNT:
        raise ValueError("planned client selection must contain 160 rows")
    plan_client = _plan_client(validated_plan, client_value)
    seen_source_ids = set()
    for selected_position, row in zip(positions, selected):
        identity = _normalize_dataset_identity(
            row,
            expected_role=plan_client["role"],
            expected_label=plan_client["label"],
        )
        expected = plan_client["samples"][selected_position]
        _assert_identity_matches_plan(
            identity,
            expected,
            context="selected client " + str(client_value),
        )
        if identity["source_id"] in seen_source_ids:
            raise ValueError("planned client selection contains duplicate source_id")
        seen_source_ids.add(identity["source_id"])
    return selected


def _expected_selected_samples(
    plan: Dict[str, Any],
    round_index: int,
    client_id: int,
) -> Dict[str, Dict[str, Any]]:
    round_plan = _plan_round(plan, round_index)
    plan_client = _plan_client(plan, client_id)
    return {
        plan_client["samples"][position]["source_id"]: plan_client["samples"][
            position
        ]
        for position in round_plan["local_sample_positions"]
    }


def _clone_delta_record(record: Mapping) -> Dict[str, Any]:
    return {
        "metadata": copy.deepcopy(record["metadata"]),
        "delta_tensors": {
            key: tensor.detach().clone().contiguous()
            for key, tensor in record["delta_tensors"].items()
        },
    }


def _canonical_equal(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def validate_client_records_against_plan(
    records,
    plan,
    round_index,
    client_id,
    expected_lora_spec,
    *,
    expected_collection_r0_tensor_sha256=None,
    expected_delta_base_tensor_sha256=None,
) -> List[Dict[str, Any]]:
    """Validate one client's complete ten-step cumulative Delta record chain."""
    validated_plan = validate_probe_collection_plan(plan)
    round_plan = _plan_round(validated_plan, round_index)
    client_value = _require_non_negative_int(client_id, "client_id")
    if client_value >= _CLIENT_COUNT:
        raise ValueError("client_id must be in range 0..9")
    if client_value not in round_plan["selected_client_ids"]:
        raise ValueError("client_id is not selected in the requested round")
    if not isinstance(expected_lora_spec, dict):
        raise TypeError("expected_lora_spec must be a dictionary")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    if len(records) != _OPTIMIZER_STEPS:
        raise ValueError("each client must provide exactly 10 records")

    plan_client = _plan_client(validated_plan, client_value)
    records_by_step = {}
    for record in records:
        record_mapping = _require_exact_keys(
            record,
            {"metadata", "delta_tensors"},
            "record",
        )
        metadata = record_mapping["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("record metadata must be a Mapping")
        step_index = _require_non_negative_int(
            metadata.get("step_index"),
            "step_index",
        )
        if step_index in records_by_step:
            raise ValueError("client record step_index is duplicated")
        records_by_step[step_index] = record_mapping
    if set(records_by_step) != set(range(_OPTIMIZER_STEPS)):
        raise ValueError("client record step_index coverage must be 0..9")

    expected_samples = _expected_selected_samples(
        validated_plan,
        round_index,
        client_value,
    )
    all_source_ids = set()
    previous_metadata = None
    normalized_records = []
    provenance = {
        "manifest_sha256": validated_plan["manifest"]["manifest_sha256"],
        "code_commit": validated_plan["code_commit"],
        "model_artifact_sha256": validated_plan["model_artifact"][
            "artifact_sha256"
        ],
        "tokenizer_artifact_sha256": validated_plan["tokenizer_artifact"][
            "artifact_sha256"
        ],
    }
    expected_lr = frozen_cosine_lr(round_index)
    expected_r0_sha = (
        None
        if expected_collection_r0_tensor_sha256 is None
        else _require_sha256(
            expected_collection_r0_tensor_sha256,
            "expected_collection_r0_tensor_sha256",
        )
    )
    expected_base_sha = (
        None
        if expected_delta_base_tensor_sha256 is None
        else _require_sha256(
            expected_delta_base_tensor_sha256,
            "expected_delta_base_tensor_sha256",
        )
    )

    for step_index in range(_OPTIMIZER_STEPS):
        record = records_by_step[step_index]
        metadata = record["metadata"]
        if metadata.get("round_index") != round_index or isinstance(
            metadata.get("round_index"), bool
        ):
            raise ValueError("record round_index does not match plan")
        if metadata.get("client_id") != client_value or isinstance(
            metadata.get("client_id"), bool
        ):
            raise ValueError("record client_id does not match plan")
        if metadata.get("role") != plan_client["role"]:
            raise ValueError("record role does not match plan client")
        if metadata.get("label") != plan_client["label"] or isinstance(
            metadata.get("label"), bool
        ):
            raise ValueError("record label does not match plan client")
        for field in _PROVENANCE_FIELDS:
            if metadata.get(field) != provenance[field]:
                raise ValueError(field + " does not match Probe plan")
        collection_r0_sha = _require_sha256(
            metadata.get("collection_r0_tensor_sha256"),
            "collection_r0_tensor_sha256",
        )
        delta_base_sha = _require_sha256(
            metadata.get("delta_base_tensor_sha256"),
            "delta_base_tensor_sha256",
        )
        if expected_r0_sha is not None and collection_r0_sha != expected_r0_sha:
            raise ValueError("record collection R0 SHA does not match formal R0")
        if expected_base_sha is not None and delta_base_sha != expected_base_sha:
            raise ValueError("record delta base SHA does not match round G_r")
        if metadata.get("delta_base_round") != round_index or isinstance(
            metadata.get("delta_base_round"), bool
        ):
            raise ValueError("record delta_base_round does not match collection round")
        if round_index == 0 and collection_r0_sha != delta_base_sha:
            raise ValueError("round 0 record delta base must equal canonical R0")
        if metadata.get("lr_policy") != (
            "shadow_fixed_reference_constant_5e-5"
        ):
            raise ValueError("record LR policy does not match formal collection")
        if metadata.get("learning_rate") != expected_lr:
            raise ValueError("record LR does not match Task1 Main cosine")
        if not _canonical_equal(
            metadata.get("lora_spec"),
            expected_lora_spec,
        ):
            raise ValueError("record lora_spec does not match expected_lora_spec")

        step_samples = metadata.get("step_samples")
        if isinstance(step_samples, (str, bytes)) or not isinstance(
            step_samples, Sequence
        ):
            raise TypeError("record step_samples must be a sequence")
        if len(step_samples) != _STEP_SAMPLE_COUNT:
            raise ValueError("each Probe record must contain exactly 16 samples")
        normalized_samples = validate_probe_step_samples(
            step_samples,
            plan_client["role"],
            plan_client["label"],
        )
        for sample in normalized_samples:
            source_id = sample["source_id"]
            if source_id in all_source_ids:
                raise ValueError(
                    "source_id is duplicated across client optimizer steps"
                )
            if source_id not in expected_samples:
                raise ValueError("record sample source_id is not selected by plan")
            _assert_identity_matches_plan(
                sample,
                expected_samples[source_id],
                context="record sample",
            )
            all_source_ids.add(source_id)

        delta_tensors = record["delta_tensors"]
        if not isinstance(delta_tensors, Mapping):
            raise TypeError("record delta_tensors must be a Mapping")
        payload_sha256 = compute_delta_tensor_payload_sha256(
            delta_tensors,
            expected_lora_spec,
            schema_version=metadata.get("schema_version"),
        )
        if metadata.get("tensor_payload_sha256") != payload_sha256:
            raise ValueError("record tensor payload SHA-256 is incorrect")

        if previous_metadata is None:
            expected_previous_record_id = None
            expected_previous_history_digest = None
        else:
            expected_previous_record_id = previous_metadata["record_id"]
            expected_previous_history_digest = previous_metadata[
                "history_digest"
            ]
        if metadata.get("previous_record_id") != expected_previous_record_id:
            raise ValueError("record previous_record_id chain is incorrect")

        expected_history_digest = compute_probe_history_digest(
            expected_previous_history_digest,
            round_index,
            client_value,
            step_index,
            plan_client["role"],
            plan_client["label"],
            normalized_samples,
        )
        rebuilt_metadata = build_probe_delta_record_metadata(
            manifest_sha256=provenance["manifest_sha256"],
            code_commit=provenance["code_commit"],
            model_artifact_sha256=provenance["model_artifact_sha256"],
            tokenizer_artifact_sha256=provenance[
                "tokenizer_artifact_sha256"
            ],
            lora_spec=expected_lora_spec,
            round_index=round_index,
            client_id=client_value,
            step_index=step_index,
            role=plan_client["role"],
            label=plan_client["label"],
            step_samples=normalized_samples,
            previous_record_id=expected_previous_record_id,
            previous_history_digest=expected_previous_history_digest,
            tensor_payload_sha256=payload_sha256,
            collection_r0_tensor_sha256=collection_r0_sha,
            delta_base_round=round_index,
            delta_base_tensor_sha256=delta_base_sha,
            lr_policy="shadow_fixed_reference_constant_5e-5",
            learning_rate=expected_lr,
            schema_version=metadata["schema_version"],
        )
        if rebuilt_metadata["history_digest"] != expected_history_digest:
            raise RuntimeError("rebuilt Probe history digest is inconsistent")
        if not _canonical_equal(metadata, rebuilt_metadata):
            raise ValueError("record metadata does not match canonical rebuild")

        normalized_record = _clone_delta_record(record)
        normalized_records.append(normalized_record)
        previous_metadata = rebuilt_metadata

    if all_source_ids != set(expected_samples):
        missing = sorted(set(expected_samples) - all_source_ids)
        extra = sorted(all_source_ids - set(expected_samples))
        raise ValueError(
            "client record samples do not match planned selection; missing="
            + repr(missing)
            + ", extra="
            + repr(extra)
        )
    return normalized_records


def validate_round_records_against_plan(
    records,
    plan,
    round_index,
    expected_lora_spec,
    *,
    expected_collection_r0_tensor_sha256=None,
    expected_delta_base_tensor_sha256=None,
) -> List[Dict[str, Any]]:
    """Validate all three clients and return defensive client/step ordering."""
    validated_plan = validate_probe_collection_plan(plan)
    round_plan = _plan_round(validated_plan, round_index)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    if len(records) != _CLIENTS_PER_ROUND * _OPTIMIZER_STEPS:
        raise ValueError("each Probe round must contain exactly 30 records")

    records_by_client: Dict[int, List[Mapping]] = {}
    for record in records:
        record_mapping = _require_exact_keys(
            record,
            {"metadata", "delta_tensors"},
            "record",
        )
        metadata = record_mapping["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("record metadata must be a Mapping")
        client_id = _require_non_negative_int(
            metadata.get("client_id"),
            "record client_id",
        )
        records_by_client.setdefault(client_id, []).append(record_mapping)

    expected_clients = set(round_plan["selected_client_ids"])
    if set(records_by_client) != expected_clients:
        missing = sorted(expected_clients - set(records_by_client))
        extra = sorted(set(records_by_client) - expected_clients)
        raise ValueError(
            "round record clients do not match plan; missing="
            + repr(missing)
            + ", extra="
            + repr(extra)
        )

    normalized = []
    for client_id in sorted(expected_clients):
        normalized.extend(
            validate_client_records_against_plan(
                records_by_client[client_id],
                validated_plan,
                round_index,
                client_id,
                expected_lora_spec,
                expected_collection_r0_tensor_sha256=(
                    expected_collection_r0_tensor_sha256
                ),
                expected_delta_base_tensor_sha256=(
                    expected_delta_base_tensor_sha256
                ),
            )
        )
    if len(normalized) != _CLIENTS_PER_ROUND * _OPTIMIZER_STEPS:
        raise ValueError("validated Probe round must contain 30 records")
    return [_clone_delta_record(record) for record in normalized]


def _coerce_output_root(output_root: Any) -> Path:
    if not isinstance(output_root, (str, os.PathLike)):
        raise TypeError("output_root must be a path")
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError("output_root must not be a symlink")
    if not root.exists():
        raise FileNotFoundError("output_root does not exist")
    if not root.is_dir():
        raise NotADirectoryError("output_root must be a directory")
    if not os.access(str(root), os.W_OK):
        raise PermissionError("output_root must be writable")
    return root


def _staging_paths(root: Path) -> List[Path]:
    return [
        Path(entry.path)
        for entry in os.scandir(str(root))
        if entry.name.startswith(".probe-delta-round.")
    ]


def _round_target_exists(root: Path, round_name: str) -> bool:
    return os.path.lexists(str(root / round_name))


def validate_probe_round_output_space(
    output_root,
    round_count=10,
) -> Path:
    """Read-only preflight for a fresh ten-round output namespace."""
    if (
        not isinstance(round_count, int)
        or isinstance(round_count, bool)
        or round_count != _ROUND_COUNT
    ):
        raise ValueError("round_count must be the integer 10")
    root = _coerce_output_root(output_root)
    for round_index in range(round_count):
        round_name = "round-" + format(round_index, "03d")
        if _round_target_exists(root, round_name):
            raise FileExistsError("round output already exists: " + round_name)
    staging = _staging_paths(root)
    if staging:
        raise FileExistsError(
            "Probe round staging path already exists: " + str(staging[0])
        )
    return root


def _validate_current_round_output_space(
    output_root: Any,
    round_name: str,
) -> Path:
    root = _coerce_output_root(output_root)
    if _round_target_exists(root, round_name):
        raise FileExistsError("round output already exists: " + round_name)
    staging = _staging_paths(root)
    if staging:
        raise FileExistsError(
            "Probe round staging path already exists: " + str(staging[0])
        )
    return root


def _validate_state_tensor(
    tensor: Any,
    key: str,
    *,
    require_cpu: bool,
    require_contiguous: bool,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("LoRA state value must be a torch.Tensor: " + key)
    if tensor.layout != torch.strided:
        raise TypeError("LoRA state tensor must be strided: " + key)
    if tensor.device.type == "meta":
        raise ValueError("LoRA state tensor must not use the meta device: " + key)
    if not tensor.is_floating_point():
        raise TypeError("LoRA state tensor must be floating point: " + key)
    if tensor.numel() == 0:
        raise ValueError("LoRA state tensor must be non-empty: " + key)
    if require_cpu and tensor.device.type != "cpu":
        raise ValueError("LoRA state tensor must be on CPU: " + key)
    if require_contiguous and not tensor.is_contiguous():
        raise ValueError("LoRA state tensor must be contiguous: " + key)
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError("LoRA state tensor contains NaN or Inf: " + key)
    return tensor


def clone_lora_state(
    state,
) -> Dict[str, torch.Tensor]:
    """Validate and clone a complete adapter state into deterministic CPU order."""
    if not isinstance(state, Mapping) or not state:
        raise ValueError("LoRA state must be a non-empty Mapping")
    keys = list(state.keys())
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("LoRA state keys must be non-empty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("LoRA state keys must be unique")

    result: Dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        tensor = _validate_state_tensor(
            state[key],
            key,
            require_cpu=False,
            require_contiguous=False,
        )
        result[key] = tensor.detach().cpu().clone().contiguous()
    return result


def _validate_selected_client_ids(selected_client_ids: Any) -> List[int]:
    if isinstance(selected_client_ids, (str, bytes)) or not isinstance(
        selected_client_ids, Sequence
    ):
        raise TypeError("selected_client_ids must be a sequence")
    normalized = [
        _require_non_negative_int(value, "selected client_id")
        for value in selected_client_ids
    ]
    if not normalized:
        raise ValueError("selected_client_ids must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("selected_client_ids must be unique")
    return normalized


def _validate_cpu_state_contract(
    state: Any,
    *,
    name: str,
) -> Mapping:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(name + " must be a non-empty Mapping")
    for key, tensor in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError(name + " keys must be non-empty strings")
        _validate_state_tensor(
            tensor,
            key,
            require_cpu=True,
            require_contiguous=True,
        )
    return state


def build_fedavg_candidate(
    global_state,
    local_states,
    selected_client_ids,
    sample_counts,
) -> Dict[str, torch.Tensor]:
    """Build a deterministic weighted average of absolute local adapter states."""
    selected = _validate_selected_client_ids(selected_client_ids)
    global_mapping = _validate_cpu_state_contract(
        global_state,
        name="global_state",
    )
    locals_mapping = _require_exact_keys(
        local_states,
        set(selected),
        "local_states",
    )
    counts_mapping = _require_exact_keys(
        sample_counts,
        set(selected),
        "sample_counts",
    )
    for name, mapping in (
        ("local_states", locals_mapping),
        ("sample_counts", counts_mapping),
    ):
        if any(
            not isinstance(client_id, int) or isinstance(client_id, bool)
            for client_id in mapping.keys()
        ):
            raise ValueError(name + " keys must be non-bool integer client IDs")

    normalized_counts = {}
    for client_id in selected:
        count = counts_mapping[client_id]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            raise ValueError("sample count must be a positive integer")
        normalized_counts[client_id] = count

    global_keys = set(global_mapping.keys())
    for client_id in selected:
        local_state = _validate_cpu_state_contract(
            locals_mapping[client_id],
            name="local_state",
        )
        if set(local_state.keys()) != global_keys:
            raise ValueError("local state keys do not match global_state")
        for key in global_mapping.keys():
            local_tensor = local_state[key]
            global_tensor = global_mapping[key]
            if local_tensor.shape != global_tensor.shape:
                raise ValueError("local state tensor shape does not match global")
            if local_tensor.dtype != global_tensor.dtype:
                raise ValueError("local state tensor dtype does not match global")

    total_samples = sum(normalized_counts[client_id] for client_id in selected)
    candidate: Dict[str, torch.Tensor] = {}
    for key in sorted(global_mapping.keys()):
        global_tensor = global_mapping[key]
        accumulator = torch.zeros_like(global_tensor)
        for client_id in selected:
            accumulator = accumulator + (
                locals_mapping[client_id][key] * normalized_counts[client_id]
            )
        averaged = (accumulator / total_samples).to(
            dtype=global_tensor.dtype
        )
        if not bool(torch.isfinite(averaged).all().item()):
            raise ValueError("FedAvg candidate contains NaN or Inf: " + key)
        candidate[key] = averaged.detach().clone().contiguous()
    return candidate


def _states_equal(left: Mapping, right: Mapping) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    return all(
        left[key].shape == right[key].shape
        and left[key].dtype == right[key].dtype
        and torch.equal(left[key], right[key])
        for key in left.keys()
    )


def _records_equal(
    left: Sequence[Mapping],
    right: Sequence[Mapping],
) -> bool:
    if len(left) != len(right):
        return False
    for left_record, right_record in zip(left, right):
        if not _canonical_equal(
            left_record["metadata"],
            right_record["metadata"],
        ):
            return False
        left_tensors = left_record["delta_tensors"]
        right_tensors = right_record["delta_tensors"]
        if set(left_tensors.keys()) != set(right_tensors.keys()):
            return False
        for key in left_tensors.keys():
            if not torch.equal(left_tensors[key], right_tensors[key]):
                return False
    return True


def execute_probe_round(
    *,
    plan,
    round_index,
    client_datasets,
    global_state,
    expected_lora_spec,
    output_root,
    train_client,
    shard_writer,
    shard_reader,
    collection_r0_tensor_sha256,
    round_provenance_sink=None,
) -> Dict[str, torch.Tensor]:
    """Execute one injected round and commit only by returning after readback."""
    validated_plan = validate_probe_collection_plan(plan)
    round_plan = _plan_round(validated_plan, round_index)
    root = _validate_current_round_output_space(
        output_root,
        round_plan["output_round_name"],
    )
    _validate_client_dataset_mapping(client_datasets, validated_plan)
    baseline_global = clone_lora_state(global_state)
    r0_sha256 = _require_sha256(
        collection_r0_tensor_sha256,
        "collection_r0_tensor_sha256",
    )
    baseline_sha256 = canonical_lora_tensor_sha256(baseline_global)
    if round_index == 0 and baseline_sha256 != r0_sha256:
        raise ValueError("collection round 0 base does not equal canonical R0")
    if not callable(train_client):
        raise TypeError("train_client must be callable")
    if not callable(shard_writer):
        raise TypeError("shard_writer must be callable")
    if not callable(shard_reader):
        raise TypeError("shard_reader must be callable")

    local_states = {}
    all_records = []
    sample_counts = {}
    initial_clones = []
    for client_id in round_plan["selected_client_ids"]:
        selected_dataset = select_planned_round_client_dataset(
            validated_plan,
            client_datasets,
            round_index,
            client_id,
        )
        initial_state = clone_lora_state(baseline_global)
        if not _states_equal(initial_state, baseline_global):
            raise RuntimeError("client initial state does not equal G_r")
        for prior in initial_clones:
            for key in initial_state.keys():
                if initial_state[key].data_ptr() == prior[key].data_ptr():
                    raise RuntimeError(
                        "client initial states must not share tensor storage"
                    )
        initial_clones.append(initial_state)

        result = train_client(
            round_index=round_index,
            client_id=client_id,
            client_dataset=selected_dataset,
            initial_global_state=initial_state,
            collection_r0_tensor_sha256=r0_sha256,
            delta_base_tensor_sha256=baseline_sha256,
        )
        result_mapping = _require_exact_keys(
            result,
            {"local_state", "records"},
            "train_client result",
        )
        local_states[client_id] = clone_lora_state(
            result_mapping["local_state"]
        )
        client_records = validate_client_records_against_plan(
            result_mapping["records"],
            validated_plan,
            round_index,
            client_id,
            expected_lora_spec,
            expected_collection_r0_tensor_sha256=r0_sha256,
            expected_delta_base_tensor_sha256=baseline_sha256,
        )
        all_records.extend(client_records)
        sample_counts[client_id] = len(selected_dataset)

    sorted_records = validate_round_records_against_plan(
        all_records,
        validated_plan,
        round_index,
        expected_lora_spec,
        expected_collection_r0_tensor_sha256=r0_sha256,
        expected_delta_base_tensor_sha256=baseline_sha256,
    )
    candidate = build_fedavg_candidate(
        baseline_global,
        local_states,
        round_plan["selected_client_ids"],
        sample_counts,
    )

    records_for_writer = [
        _clone_delta_record(record) for record in sorted_records
    ]
    shard_writer(root, round_index, records_for_writer)
    loaded = shard_reader(
        root / round_plan["output_round_name"],
        expected_round_index=round_index,
        expected_manifest_sha256=validated_plan["manifest"][
            "manifest_sha256"
        ],
        expected_code_commit=validated_plan["code_commit"],
        expected_model_artifact_sha256=validated_plan["model_artifact"][
            "artifact_sha256"
        ],
        expected_tokenizer_artifact_sha256=validated_plan[
            "tokenizer_artifact"
        ]["artifact_sha256"],
        expected_lora_spec=copy.deepcopy(expected_lora_spec),
    )
    if not isinstance(loaded, Mapping) or "records" not in loaded:
        raise ValueError("shard_reader result must contain records")
    loaded_records = validate_round_records_against_plan(
        loaded["records"],
        validated_plan,
        round_index,
        expected_lora_spec,
        expected_collection_r0_tensor_sha256=r0_sha256,
        expected_delta_base_tensor_sha256=baseline_sha256,
    )
    if not _records_equal(sorted_records, loaded_records):
        raise ValueError("readback records do not match published records")
    round_provenance = {
        "collection_round": round_index,
        "delta_base_round": round_index,
        "delta_base_tensor_sha256": baseline_sha256,
        "collection_r0_tensor_sha256": r0_sha256,
        "selected_client_ids": list(round_plan["selected_client_ids"]),
        "sample_counts": {
            str(client_id): sample_counts[client_id]
            for client_id in round_plan["selected_client_ids"]
        },
        "local_state_tensor_sha256": {
            str(client_id): canonical_lora_tensor_sha256(local_states[client_id])
            for client_id in round_plan["selected_client_ids"]
        },
        "fedavg_candidate_tensor_sha256": canonical_lora_tensor_sha256(candidate),
        "lr_policy": "shadow_fixed_reference_constant_5e-5",
        "learning_rate": frozen_cosine_lr(round_index),
    }
    if round_provenance_sink is not None:
        if not callable(round_provenance_sink):
            raise TypeError("round_provenance_sink must be callable")
        round_provenance_sink(copy.deepcopy(round_provenance))
    return clone_lora_state(candidate)

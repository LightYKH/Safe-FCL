"""Deterministic, model-independent primitives for Probe Delta-LoRA records."""

import copy
import hashlib
import math
import re
import struct
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from utils.artifact_schema import matches_schema
from utils.data_manifest import canonical_json_bytes, sha256_bytes


FILTER_POLICY = "first_transformer_layer_all_lora_B"
RECORD_SCHEMA_VERSION = "safe-fcl.probe-delta-record.v3"

_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLES_TO_LABELS = {
    "benign": 0,
    "malicious": 1,
}
_SAMPLE_FIELDS = (
    "sample_id",
    "source_id",
    "content_hash",
    "raw_row_index",
    "split_rank",
    "probe_role",
    "probe_label",
)


def _require_non_negative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(name + " must be a non-negative integer")
    return value


def _require_positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX_64_RE.fullmatch(value) is None:
        raise ValueError(name + " must be a 64-character lowercase SHA-256")
    return value


def _require_code_commit(value: Any) -> str:
    if not isinstance(value, str) or _HEX_40_RE.fullmatch(value) is None:
        raise ValueError("code_commit must be a 40-character lowercase Git commit")
    return value


def _validate_role_label(role: Any, label: Any) -> Tuple[str, int]:
    if role not in _ROLES_TO_LABELS:
        raise ValueError("role must be 'benign' or 'malicious'")
    if (
        not isinstance(label, int)
        or isinstance(label, bool)
        or label != _ROLES_TO_LABELS[role]
    ):
        raise ValueError("role and label are inconsistent")
    return role, label


def _parse_first_layer_lora_b_key(
    key: str,
) -> Optional[Tuple[str, str]]:
    if not isinstance(key, str):
        raise TypeError("state keys must be strings")

    segments = key.split(".")
    layer_positions = [
        index
        for index in range(len(segments) - 1)
        if segments[index] == "layers" and segments[index + 1] == "0"
    ]
    if not layer_positions:
        return None

    lora_positions = [
        index
        for index, segment in enumerate(segments)
        if segment == "lora_B"
        and any(layer_index + 1 < index for layer_index in layer_positions)
    ]
    if not lora_positions or not segments or segments[-1] != "weight":
        return None

    if len(layer_positions) != 1 or len(lora_positions) != 1:
        raise ValueError("ambiguous first-layer LoRA-B key: " + key)
    if any(not segment for segment in segments):
        raise ValueError("ambiguous empty path segment in key: " + key)

    layer_index = layer_positions[0]
    lora_index = lora_positions[0]
    if lora_index <= layer_index + 2:
        raise ValueError("LoRA-B key has no resolved target module: " + key)

    resolved_module_path = ".".join(segments[:lora_index])
    resolved_target_module = segments[lora_index - 1]
    return resolved_module_path, resolved_target_module


def _first_layer_lora_b_keys(state: Mapping[str, Any]) -> List[str]:
    selected = []
    for key in state.keys():
        if _parse_first_layer_lora_b_key(key) is not None:
            selected.append(key)
    return sorted(selected)


def _require_floating_matrix(
    tensor: Any,
    key: str,
    expected_shape: Optional[Sequence[int]],
    expected_rank: int,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("state value is not a torch.Tensor: " + key)
    if not tensor.is_floating_point():
        raise TypeError("state tensor is not floating point: " + key)
    if tensor.ndim != 2:
        raise ValueError("state tensor must be two-dimensional: " + key)
    if tensor.shape[-1] != expected_rank:
        raise ValueError("state tensor rank does not match expected rank: " + key)
    if expected_shape is not None and list(tensor.shape) != list(expected_shape):
        raise ValueError("state tensor shape does not match tensor spec: " + key)
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError("state tensor contains NaN or Inf: " + key)
    return tensor


def _validate_tensor_spec(tensor_spec: Dict[str, Any]) -> Tuple[List[str], int]:
    if not isinstance(tensor_spec, dict):
        raise TypeError("tensor_spec must be a dictionary")
    if tensor_spec.get("filter_policy") != FILTER_POLICY:
        raise ValueError("tensor_spec filter_policy is invalid")
    if tensor_spec.get("layer_index") != 0 or isinstance(
        tensor_spec.get("layer_index"), bool
    ):
        raise ValueError("tensor_spec layer_index must be 0")

    rank = _require_positive_integer(tensor_spec.get("rank"), "tensor_spec rank")
    ordered_keys = tensor_spec.get("ordered_keys")
    tensor_entries = tensor_spec.get("tensors")
    if (
        not isinstance(ordered_keys, list)
        or not ordered_keys
        or any(not isinstance(key, str) for key in ordered_keys)
    ):
        raise ValueError("tensor_spec ordered_keys must be a non-empty string list")
    if ordered_keys != sorted(ordered_keys) or len(set(ordered_keys)) != len(
        ordered_keys
    ):
        raise ValueError("tensor_spec ordered_keys must be unique and sorted")
    if not isinstance(tensor_entries, dict) or set(tensor_entries) != set(
        ordered_keys
    ):
        raise ValueError("tensor_spec tensors must exactly match ordered_keys")

    seen_module_paths = set()
    for key in ordered_keys:
        parsed = _parse_first_layer_lora_b_key(key)
        if parsed is None:
            raise ValueError("tensor_spec contains a non-first-layer LoRA-B key")
        expected_module_path, expected_target_module = parsed

        entry = tensor_entries[key]
        if not isinstance(entry, dict):
            raise ValueError("tensor_spec tensor entry must be a dictionary")
        shape = entry.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 0
                for dimension in shape
            )
            or shape[-1] != rank
        ):
            raise ValueError("tensor_spec contains an invalid tensor shape")
        if not isinstance(entry.get("dtype"), str) or not entry["dtype"]:
            raise ValueError("tensor_spec contains an invalid tensor dtype")
        if entry.get("resolved_module_path") != expected_module_path:
            raise ValueError("tensor_spec resolved module path is inconsistent")
        if entry.get("resolved_target_module") != expected_target_module:
            raise ValueError("tensor_spec resolved target module is inconsistent")
        if expected_module_path in seen_module_paths:
            raise ValueError("tensor_spec contains a duplicate resolved module path")
        seen_module_paths.add(expected_module_path)

    return ordered_keys, rank


def build_first_layer_lora_b_spec(
    initial_state: Mapping[str, torch.Tensor],
    expected_rank: int = 8,
) -> Dict[str, Any]:
    """Build a deterministic specification for all layer-0 LoRA-B weights."""
    if not isinstance(initial_state, MappingABC) or not initial_state:
        raise ValueError("initial_state must be a non-empty Mapping")
    rank = _require_positive_integer(expected_rank, "expected_rank")

    ordered_keys = _first_layer_lora_b_keys(initial_state)
    if not ordered_keys:
        raise ValueError("no first-transformer-layer LoRA-B weights were found")

    tensor_entries = {}
    seen_module_paths = set()
    for key in ordered_keys:
        parsed = _parse_first_layer_lora_b_key(key)
        if parsed is None:  # pragma: no cover - guaranteed by key selection
            raise AssertionError("selected key could not be parsed")
        module_path, target_module = parsed
        if module_path in seen_module_paths:
            raise ValueError("duplicate resolved module path: " + module_path)
        seen_module_paths.add(module_path)

        tensor = _require_floating_matrix(
            initial_state[key],
            key,
            expected_shape=None,
            expected_rank=rank,
        )
        tensor_entries[key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "resolved_module_path": module_path,
            "resolved_target_module": target_module,
        }

    spec = {
        "filter_policy": FILTER_POLICY,
        "layer_index": 0,
        "rank": rank,
        "ordered_keys": ordered_keys,
        "tensors": tensor_entries,
    }
    _validate_tensor_spec(spec)
    return spec


def compute_cumulative_lora_b_delta(
    initial_state: Mapping[str, torch.Tensor],
    current_state: Mapping[str, torch.Tensor],
    tensor_spec: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Compute cumulative layer-0 LoRA-B change from the fixed initial state."""
    if not isinstance(initial_state, MappingABC):
        raise TypeError("initial_state must be a Mapping")
    if not isinstance(current_state, MappingABC):
        raise TypeError("current_state must be a Mapping")

    ordered_keys, rank = _validate_tensor_spec(tensor_spec)
    initial_keys = _first_layer_lora_b_keys(initial_state)
    if set(initial_keys) != set(ordered_keys):
        missing = sorted(set(ordered_keys) - set(initial_keys))
        extra = sorted(set(initial_keys) - set(ordered_keys))
        raise ValueError(
            "initial first-layer LoRA-B key set does not match tensor spec; "
            + "missing="
            + repr(missing)
            + ", extra="
            + repr(extra)
        )

    current_keys = _first_layer_lora_b_keys(current_state)
    if set(current_keys) != set(ordered_keys):
        missing = sorted(set(ordered_keys) - set(current_keys))
        extra = sorted(set(current_keys) - set(ordered_keys))
        raise ValueError(
            "current first-layer LoRA-B key set does not match tensor spec; "
            + "missing="
            + repr(missing)
            + ", extra="
            + repr(extra)
        )

    deltas = {}
    for key in ordered_keys:
        if key not in initial_state:
            raise ValueError("initial_state is missing tensor spec key: " + key)
        shape = tensor_spec["tensors"][key]["shape"]
        initial_tensor = _require_floating_matrix(
            initial_state[key],
            key,
            expected_shape=shape,
            expected_rank=rank,
        )
        current_tensor = _require_floating_matrix(
            current_state[key],
            key,
            expected_shape=shape,
            expected_rank=rank,
        )
        delta = (
            current_tensor.detach().cpu().float()
            - initial_tensor.detach().cpu().float()
        ).contiguous()
        if not bool(torch.isfinite(delta).all().item()):
            raise ValueError("computed Delta-LoRA tensor contains NaN or Inf: " + key)
        deltas[key] = delta

    return deltas


def _update_length_prefixed(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def compute_delta_tensor_payload_sha256(
    delta_tensors: Mapping[str, torch.Tensor],
    tensor_spec: Dict[str, Any],
    *,
    schema_version: str = RECORD_SCHEMA_VERSION,
) -> str:
    """Digest canonical names, shapes, float32 dtype, and little-endian values."""
    if not isinstance(delta_tensors, MappingABC):
        raise TypeError("delta_tensors must be a Mapping")
    ordered_keys, rank = _validate_tensor_spec(tensor_spec)
    if set(delta_tensors.keys()) != set(ordered_keys):
        raise ValueError("delta_tensors keys must exactly match tensor_spec")

    if not matches_schema(schema_version, RECORD_SCHEMA_VERSION):
        raise ValueError("unsupported Probe record schema")
    namespace = schema_version.partition(".")[0]
    domain = f"{namespace}.probe-delta-tensor-payload.v1"
    digest = hashlib.sha256(domain.encode("ascii"))
    for key in ordered_keys:
        shape = tensor_spec["tensors"][key]["shape"]
        tensor = _require_floating_matrix(
            delta_tensors[key],
            key,
            expected_shape=shape,
            expected_rank=rank,
        )
        normalized = tensor.detach().cpu().float().contiguous()
        values = normalized.reshape(-1).tolist()
        raw_bytes = struct.pack("<" + ("f" * len(values)), *values)

        _update_length_prefixed(digest, key.encode("utf-8"))
        digest.update(struct.pack("<Q", len(shape)))
        for dimension in shape:
            digest.update(struct.pack("<Q", dimension))
        _update_length_prefixed(digest, b"float32")
        _update_length_prefixed(digest, raw_bytes)

    return digest.hexdigest()


def validate_probe_step_samples(
    samples: Sequence[Mapping[str, Any]],
    expected_role: str,
    expected_label: int,
) -> List[Dict[str, Any]]:
    """Validate and normalize real batch identities without reordering them."""
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence of mappings")
    if not samples:
        raise ValueError("step samples must be non-empty")
    role, label = _validate_role_label(expected_role, expected_label)

    normalized_samples = []
    seen_source_ids = set()
    for sample in samples:
        if not isinstance(sample, MappingABC):
            raise TypeError("each step sample must be a Mapping")
        missing = [field for field in _SAMPLE_FIELDS if field not in sample]
        if missing:
            raise ValueError("step sample is missing fields: " + repr(missing))

        sample_id = _require_non_negative_integer(
            sample["sample_id"], "sample_id"
        )
        raw_row_index = _require_non_negative_integer(
            sample["raw_row_index"], "raw_row_index"
        )
        split_rank = _require_non_negative_integer(
            sample["split_rank"], "split_rank"
        )
        if sample_id != split_rank:
            raise ValueError("sample_id must equal split_rank")

        source_id = sample["source_id"]
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        if source_id in seen_source_ids:
            raise ValueError("source_id is duplicated within the step: " + source_id)
        seen_source_ids.add(source_id)

        content_hash = _require_sha256(sample["content_hash"], "content_hash")
        normalized_prompt_hash = sample.get("normalized_prompt_hash")
        if normalized_prompt_hash is not None:
            normalized_prompt_hash = _require_sha256(
                normalized_prompt_hash,
                "normalized_prompt_hash",
            )
        sample_role, sample_label = _validate_role_label(
            sample["probe_role"], sample["probe_label"]
        )
        if sample_role != role or sample_label != label:
            raise ValueError("step sample role/label does not match expected values")

        normalized_samples.append(
            {
                "sample_id": sample_id,
                "source_id": source_id,
                "content_hash": content_hash,
                "normalized_prompt_hash": normalized_prompt_hash,
                "raw_row_index": raw_row_index,
                "split_rank": split_rank,
                "probe_role": sample_role,
                "probe_label": sample_label,
            }
        )

    return normalized_samples


def compute_probe_history_digest(
    previous_history_digest: Optional[str],
    round_index: int,
    client_id: int,
    step_index: int,
    role: str,
    label: int,
    step_samples: Sequence[Mapping[str, Any]],
) -> str:
    """Extend the deterministic, order-sensitive batch-history digest chain."""
    round_value = _require_non_negative_integer(round_index, "round_index")
    client_value = _require_non_negative_integer(client_id, "client_id")
    step_value = _require_non_negative_integer(step_index, "step_index")
    normalized_role, normalized_label = _validate_role_label(role, label)
    normalized_samples = validate_probe_step_samples(
        step_samples,
        normalized_role,
        normalized_label,
    )

    if step_value == 0:
        if previous_history_digest is not None:
            raise ValueError("step 0 previous_history_digest must be None")
    else:
        _require_sha256(previous_history_digest, "previous_history_digest")

    history_payload = {
        "previous_history_digest": previous_history_digest,
        "round_index": round_value,
        "client_id": client_value,
        "step_index": step_value,
        "role": normalized_role,
        "label": normalized_label,
        "step_samples": normalized_samples,
    }
    return sha256_bytes(canonical_json_bytes(history_payload))


def build_probe_delta_record_metadata(
    *,
    manifest_sha256: str,
    code_commit: str,
    model_artifact_sha256: str,
    tokenizer_artifact_sha256: str,
    lora_spec: Dict[str, Any],
    round_index: int,
    client_id: int,
    step_index: int,
    role: str,
    label: int,
    step_samples: Sequence[Mapping[str, Any]],
    previous_record_id: Optional[str],
    previous_history_digest: Optional[str],
    tensor_payload_sha256: str,
    collection_r0_tensor_sha256: str,
    delta_base_round: int,
    delta_base_tensor_sha256: str,
    lr_policy: str,
    learning_rate: float,
    schema_version: str = RECORD_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Build canonical metadata for one cumulative batch-step Delta-LoRA record."""
    if not matches_schema(schema_version, RECORD_SCHEMA_VERSION):
        raise ValueError("unsupported Probe record schema")
    manifest_digest = _require_sha256(manifest_sha256, "manifest_sha256")
    commit = _require_code_commit(code_commit)
    model_digest = _require_sha256(
        model_artifact_sha256, "model_artifact_sha256"
    )
    tokenizer_digest = _require_sha256(
        tokenizer_artifact_sha256, "tokenizer_artifact_sha256"
    )
    payload_digest = _require_sha256(
        tensor_payload_sha256, "tensor_payload_sha256"
    )
    collection_r0_digest = _require_sha256(
        collection_r0_tensor_sha256,
        "collection_r0_tensor_sha256",
    )
    delta_base_digest = _require_sha256(
        delta_base_tensor_sha256,
        "delta_base_tensor_sha256",
    )
    _validate_tensor_spec(lora_spec)

    round_value = _require_non_negative_integer(round_index, "round_index")
    delta_base_value = _require_non_negative_integer(
        delta_base_round,
        "delta_base_round",
    )
    if delta_base_value != round_value:
        raise ValueError("delta_base_round must equal collection round_index")
    if round_value == 0 and delta_base_digest != collection_r0_digest:
        raise ValueError("round 0 delta base must equal canonical collection R0")
    if not isinstance(lr_policy, str) or lr_policy != (
        "task1_main_100_round_cosine_first_10"
    ):
        raise ValueError("lr_policy must match formal Probe collection contract")
    if isinstance(learning_rate, bool):
        raise TypeError("learning_rate must be a finite positive number")
    try:
        learning_rate_value = float(learning_rate)
    except (TypeError, ValueError) as exc:
        raise TypeError("learning_rate must be a finite positive number") from exc
    if not math.isfinite(learning_rate_value) or learning_rate_value <= 0.0:
        raise ValueError("learning_rate must be a finite positive number")
    client_value = _require_non_negative_integer(client_id, "client_id")
    step_value = _require_non_negative_integer(step_index, "step_index")
    normalized_role, normalized_label = _validate_role_label(role, label)
    normalized_samples = validate_probe_step_samples(
        step_samples,
        normalized_role,
        normalized_label,
    )

    if step_value == 0:
        if previous_record_id is not None or previous_history_digest is not None:
            raise ValueError(
                "step 0 previous_record_id and previous_history_digest must be None"
            )
    else:
        _require_sha256(previous_record_id, "previous_record_id")
        _require_sha256(previous_history_digest, "previous_history_digest")

    history_digest = compute_probe_history_digest(
        previous_history_digest=previous_history_digest,
        round_index=round_value,
        client_id=client_value,
        step_index=step_value,
        role=normalized_role,
        label=normalized_label,
        step_samples=normalized_samples,
    )
    metadata = {
        "schema_version": schema_version,
        "manifest_sha256": manifest_digest,
        "code_commit": commit,
        "model_artifact_sha256": model_digest,
        "tokenizer_artifact_sha256": tokenizer_digest,
        "lora_spec": copy.deepcopy(lora_spec),
        "round_index": round_value,
        "collection_r0_tensor_sha256": collection_r0_digest,
        "delta_base_round": delta_base_value,
        "delta_base_tensor_sha256": delta_base_digest,
        "lr_policy": lr_policy,
        "learning_rate": learning_rate_value,
        "client_id": client_value,
        "step_index": step_value,
        "role": normalized_role,
        "label": normalized_label,
        "step_samples": normalized_samples,
        "previous_record_id": previous_record_id,
        "history_digest": history_digest,
        "tensor_payload_sha256": payload_digest,
        "normalization": {
            "stored_delta": "none",
            "probe_feature": "l2",
        },
    }
    metadata["record_id"] = sha256_bytes(canonical_json_bytes(metadata))
    return metadata

"""No-clobber Safetensors storage for per-round Probe Delta-LoRA records."""

import copy
import json
import os
import re
import shutil
import stat
import struct
import tempfile
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import safetensors
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from lora_classifier.probe_delta_core import (
    RECORD_SCHEMA_VERSION,
    build_probe_delta_record_metadata,
    compute_delta_tensor_payload_sha256,
)
from utils.artifact_schema import matches_schema
from utils.data_manifest import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


PathLike = Union[str, os.PathLike]

ROUND_SCHEMA_VERSION = "safe-fcl.probe-delta-round.v1"
TENSOR_FILE_NAME = "deltas.safetensors"
INDEX_FILE_NAME = "index.json"
INDEX_SIDECAR_NAME = "index.json.sha256"
COMPLETE_FILE_NAME = "COMPLETE"

_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_LINE_RE = re.compile(br"^[0-9a-f]{64}\n$")
_RECORD_METADATA_KEYS = {
    "schema_version",
    "manifest_sha256",
    "code_commit",
    "model_artifact_sha256",
    "tokenizer_artifact_sha256",
    "lora_spec",
    "round_index",
    "collection_r0_tensor_sha256",
    "delta_base_round",
    "delta_base_tensor_sha256",
    "lr_policy",
    "learning_rate",
    "client_id",
    "step_index",
    "role",
    "label",
    "step_samples",
    "previous_record_id",
    "history_digest",
    "tensor_payload_sha256",
    "normalization",
    "record_id",
}
_INDEX_KEYS = {
    "schema_version",
    "round_index",
    "record_count",
    "records",
    "tensor_file",
    "storage_runtime",
}
_INDEX_RECORD_KEYS = {
    "record_ordinal",
    "metadata",
    "tensor_storage",
}
_TENSOR_STORAGE_KEYS = {
    "name",
    "storage_key",
    "shape",
    "dtype",
}
_SHARD_IDENTITY_FIELDS = (
    "manifest_sha256",
    "code_commit",
    "model_artifact_sha256",
    "tokenizer_artifact_sha256",
    "collection_r0_tensor_sha256",
    "delta_base_round",
    "delta_base_tensor_sha256",
    "lr_policy",
    "learning_rate",
)


def _require_non_negative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(name + " must be a non-negative integer")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX_64_RE.fullmatch(value) is None:
        raise ValueError(name + " must be a 64-character lowercase SHA-256")
    return value


def _require_code_commit(value: Any, name: str = "code_commit") -> str:
    if not isinstance(value, str) or _HEX_40_RE.fullmatch(value) is None:
        raise ValueError(name + " must be a 40-character lowercase Git commit")
    return value


def _round_directory_name(round_index: int) -> str:
    return "round-" + format(round_index, "03d")


def _reject_duplicate_json_keys(
    pairs: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + str(key))
        result[key] = value
    return result


def _write_synced_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_existing_file(path: Path) -> None:
    with path.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _canonicalize_safetensors_header(path: Path) -> None:
    temporary_path = None
    try:
        with path.open("rb") as source:
            length_bytes = source.read(8)
            if len(length_bytes) != 8:
                raise ValueError("Safetensors file has a truncated header length")
            header_length = struct.unpack("<Q", length_bytes)[0]
            header_bytes = source.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError("Safetensors file has a truncated header")
            try:
                header = json.loads(
                    header_bytes.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
            except (UnicodeDecodeError, TypeError, ValueError) as exc:
                raise ValueError("Safetensors file header is not valid JSON") from exc
            canonical_header = canonical_json_bytes(header)
            padding_length = (-len(canonical_header)) % 8
            padded_header = canonical_header + (b" " * padding_length)

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(path.parent),
                prefix="." + path.name + ".canonical.",
                suffix=".tmp",
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                target.write(struct.pack("<Q", len(padded_header)))
                target.write(padded_header)
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())

        path.unlink()
        os.rename(str(temporary_path), str(path))
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _unlink_if_same_file(final_path: Path, staging_path: Path) -> None:
    try:
        final_stat = os.stat(final_path, follow_symlinks=False)
    except FileNotFoundError:
        return
    try:
        staging_stat = os.stat(staging_path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (final_stat.st_dev, final_stat.st_ino) != (
        staging_stat.st_dev,
        staging_stat.st_ino,
    ):
        return
    try:
        final_path.unlink()
    except FileNotFoundError:
        return


def _remove_directory_if_empty(path: Path) -> None:
    try:
        path.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        if path.is_dir() and any(path.iterdir()):
            return
        raise


def _validate_record_metadata(
    metadata: Any,
    round_index: int,
    prior_metadata_by_id: Mapping[str, Dict[str, Any]],
    seen_client_steps: Any,
    shard_identity_baseline: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(metadata, dict):
        raise TypeError("record metadata must be a dictionary")
    if set(metadata) != _RECORD_METADATA_KEYS:
        raise ValueError("record metadata fields do not match the record schema")
    if not matches_schema(metadata.get("schema_version"), RECORD_SCHEMA_VERSION):
        raise ValueError("record metadata schema_version is invalid")
    if metadata.get("round_index") != round_index or isinstance(
        metadata.get("round_index"), bool
    ):
        raise ValueError("record metadata round_index does not match round shard")

    record_id = _require_sha256(metadata.get("record_id"), "record_id")
    client_id = _require_non_negative_integer(
        metadata.get("client_id"), "metadata client_id"
    )
    step_index = _require_non_negative_integer(
        metadata.get("step_index"), "metadata step_index"
    )
    client_step = (client_id, step_index)
    if client_step in seen_client_steps:
        raise ValueError("duplicate client step position")

    previous_record_id = metadata.get("previous_record_id")
    if step_index == 0:
        if previous_record_id is not None:
            raise ValueError("step 0 previous_record_id must be None")
        previous_history_digest = None
    else:
        previous_id = _require_sha256(
            previous_record_id, "previous_record_id"
        )
        if previous_id not in prior_metadata_by_id:
            raise ValueError(
                "previous_record_id must reference a preceding shard record"
            )
        previous_metadata = prior_metadata_by_id[previous_id]
        if previous_metadata["client_id"] != client_id:
            raise ValueError("previous record belongs to another client")
        if previous_metadata["round_index"] != round_index:
            raise ValueError("previous record belongs to another round")
        if previous_metadata["step_index"] != step_index - 1:
            raise ValueError("previous step is not the immediate predecessor")
        if (
            previous_metadata["role"] != metadata.get("role")
            or previous_metadata["label"] != metadata.get("label")
        ):
            raise ValueError("previous role/label is inconsistent")
        previous_history_digest = previous_metadata["history_digest"]

    rebuilt = build_probe_delta_record_metadata(
        manifest_sha256=metadata.get("manifest_sha256"),
        code_commit=metadata.get("code_commit"),
        model_artifact_sha256=metadata.get("model_artifact_sha256"),
        tokenizer_artifact_sha256=metadata.get("tokenizer_artifact_sha256"),
        lora_spec=metadata.get("lora_spec"),
        round_index=metadata.get("round_index"),
        client_id=metadata.get("client_id"),
        step_index=step_index,
        role=metadata.get("role"),
        label=metadata.get("label"),
        step_samples=metadata.get("step_samples"),
        previous_record_id=previous_record_id,
        previous_history_digest=previous_history_digest,
        tensor_payload_sha256=metadata.get("tensor_payload_sha256"),
        collection_r0_tensor_sha256=metadata.get(
            "collection_r0_tensor_sha256"
        ),
        delta_base_round=metadata.get("delta_base_round"),
        delta_base_tensor_sha256=metadata.get("delta_base_tensor_sha256"),
        lr_policy=metadata.get("lr_policy"),
        learning_rate=metadata.get("learning_rate"),
        schema_version=metadata["schema_version"],
    )
    if rebuilt != metadata:
        expected_id = sha256_bytes(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in metadata.items()
                    if key != "record_id"
                }
            )
        )
        if record_id != expected_id:
            raise ValueError("record metadata record_id does not match canonical JSON")
        raise ValueError("record metadata does not match rebuilt metadata")

    if shard_identity_baseline is None:
        validated_baseline = {
            field: copy.deepcopy(rebuilt[field])
            for field in _SHARD_IDENTITY_FIELDS
        }
        validated_baseline["lora_spec"] = copy.deepcopy(rebuilt["lora_spec"])
    else:
        validated_baseline = shard_identity_baseline
        for field in _SHARD_IDENTITY_FIELDS:
            if rebuilt[field] != validated_baseline[field]:
                raise ValueError(
                    "shard identity field is inconsistent: " + field
                )
        if canonical_json_bytes(rebuilt["lora_spec"]) != canonical_json_bytes(
            validated_baseline["lora_spec"]
        ):
            raise ValueError("shard identity field is inconsistent: lora_spec")

    seen_client_steps.add(client_step)
    return copy.deepcopy(rebuilt), validated_baseline


def _prepare_records(
    records: Sequence[Mapping[str, Any]],
    round_index: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    if not records:
        raise ValueError("records must be non-empty")

    index_records = []
    storage_tensors = {}
    seen_record_ids = set()
    prior_metadata_by_id = {}
    seen_client_steps = set()
    shard_identity_baseline = None

    for record_ordinal, record in enumerate(records):
        if not isinstance(record, MappingABC):
            raise TypeError("each record must be a Mapping")
        if set(record.keys()) != {"metadata", "delta_tensors"}:
            raise ValueError(
                "each record must contain only metadata and delta_tensors"
            )

        candidate_metadata = record["metadata"]
        candidate_record_id = (
            candidate_metadata.get("record_id")
            if isinstance(candidate_metadata, dict)
            else None
        )
        if candidate_record_id in seen_record_ids:
            raise ValueError("record_id must be unique within a round shard")
        metadata, shard_identity_baseline = _validate_record_metadata(
            record["metadata"],
            round_index,
            prior_metadata_by_id,
            seen_client_steps,
            shard_identity_baseline,
        )
        record_id = metadata["record_id"]
        seen_record_ids.add(record_id)

        delta_tensors = record["delta_tensors"]
        if not isinstance(delta_tensors, MappingABC):
            raise TypeError("delta_tensors must be a Mapping")
        payload_sha256 = compute_delta_tensor_payload_sha256(
            delta_tensors,
            metadata["lora_spec"],
            schema_version=metadata["schema_version"],
        )
        if payload_sha256 != metadata["tensor_payload_sha256"]:
            raise ValueError("tensor payload SHA-256 does not match record metadata")

        tensor_storage = []
        seen_tensor_names = set()
        for tensor_ordinal, name in enumerate(
            metadata["lora_spec"]["ordered_keys"]
        ):
            if name in seen_tensor_names:
                raise ValueError("tensor name is duplicated within a record")
            seen_tensor_names.add(name)
            storage_key = "record-{0:04d}.tensor-{1:04d}".format(
                record_ordinal,
                tensor_ordinal,
            )
            if storage_key in storage_tensors:
                raise ValueError("Safetensors storage key is duplicated")

            normalized = (
                delta_tensors[name]
                .detach()
                .cpu()
                .float()
                .contiguous()
                .clone()
            )
            storage_tensors[storage_key] = normalized
            tensor_storage.append(
                {
                    "name": name,
                    "storage_key": storage_key,
                    "shape": list(normalized.shape),
                    "dtype": "float32",
                }
            )

        index_records.append(
            {
                "record_ordinal": record_ordinal,
                "metadata": metadata,
                "tensor_storage": tensor_storage,
            }
        )
        prior_metadata_by_id[record_id] = metadata

    return index_records, storage_tensors


def _publish_staged_round(
    staging_directory: Path,
    round_directory: Path,
) -> None:
    ordered_names = (
        TENSOR_FILE_NAME,
        INDEX_FILE_NAME,
        INDEX_SIDECAR_NAME,
        COMPLETE_FILE_NAME,
    )
    os.mkdir(str(round_directory))
    created_links = []
    try:
        for name in ordered_names:
            staging_path = staging_directory / name
            final_path = round_directory / name
            os.link(str(staging_path), str(final_path))
            created_links.append((final_path, staging_path))
    except Exception:
        for final_path, staging_path in reversed(created_links):
            _unlink_if_same_file(final_path, staging_path)
        _remove_directory_if_empty(round_directory)
        raise


def write_probe_delta_round_shard(
    output_root: PathLike,
    round_index: int,
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Stage and no-clobber publish one complete per-round Delta shard."""
    round_value = _require_non_negative_integer(round_index, "round_index")
    root = Path(output_root)
    if not root.is_dir():
        raise NotADirectoryError("output_root must be an existing directory")
    if not os.access(str(root), os.W_OK):
        raise PermissionError("output_root must be writable")

    round_directory = root / _round_directory_name(round_value)
    if round_directory.exists() or round_directory.is_symlink():
        raise FileExistsError(
            "round directory already exists: " + str(round_directory)
        )

    index_records, storage_tensors = _prepare_records(records, round_value)
    with tempfile.TemporaryDirectory(
        dir=str(root),
        prefix=".probe-delta-round.",
    ) as staging_name:
        staging_directory = Path(staging_name)
        tensor_path = staging_directory / TENSOR_FILE_NAME
        index_path = staging_directory / INDEX_FILE_NAME
        sidecar_path = staging_directory / INDEX_SIDECAR_NAME
        complete_path = staging_directory / COMPLETE_FILE_NAME

        save_file(
            storage_tensors,
            str(tensor_path),
            metadata={
                "schema_version": ROUND_SCHEMA_VERSION,
                "round_index": str(round_value),
                "record_count": str(len(index_records)),
            },
        )
        _canonicalize_safetensors_header(tensor_path)
        _fsync_existing_file(tensor_path)
        tensor_sha256 = sha256_file(tensor_path)
        tensor_size = tensor_path.stat().st_size

        index = {
            "schema_version": ROUND_SCHEMA_VERSION,
            "round_index": round_value,
            "record_count": len(index_records),
            "records": index_records,
            "tensor_file": {
                "name": TENSOR_FILE_NAME,
                "sha256": tensor_sha256,
                "size_bytes": tensor_size,
            },
            "storage_runtime": {
                "torch_version": torch.__version__,
                "safetensors_version": safetensors.__version__,
            },
        }
        index_bytes = canonical_json_bytes(index)
        index_sha256 = sha256_bytes(index_bytes)
        digest_line = (index_sha256 + "\n").encode("ascii")
        _write_synced_bytes(index_path, index_bytes)
        _write_synced_bytes(sidecar_path, digest_line)
        _write_synced_bytes(complete_path, digest_line)

        _publish_staged_round(staging_directory, round_directory)

    return copy.deepcopy(index)


def _require_regular_file(path: Path, name: str) -> None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(name + " is missing: " + str(path)) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(name + " must be a regular file")


def _read_digest_line(path: Path, name: str) -> str:
    raw = path.read_bytes()
    if _DIGEST_LINE_RE.fullmatch(raw) is None:
        raise ValueError(name + " must contain one lowercase SHA-256 and newline")
    return raw[:-1].decode("ascii")


def _load_canonical_index(path: Path) -> Tuple[Dict[str, Any], bytes]:
    raw_bytes = path.read_bytes()
    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("index.json is not valid UTF-8") from exc
    try:
        index = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("index.json is not valid JSON") from exc
    if not isinstance(index, dict):
        raise ValueError("index.json root must be a JSON object")
    if raw_bytes != canonical_json_bytes(index):
        raise ValueError("index.json is not in canonical byte form")
    return index, raw_bytes


def _validate_expected_metadata(
    metadata: Dict[str, Any],
    expected_manifest_sha256: Optional[str],
    expected_code_commit: Optional[str],
    expected_model_artifact_sha256: Optional[str],
    expected_tokenizer_artifact_sha256: Optional[str],
    expected_lora_spec: Optional[Dict[str, Any]],
) -> None:
    comparisons = (
        ("manifest_sha256", expected_manifest_sha256),
        ("code_commit", expected_code_commit),
        ("model_artifact_sha256", expected_model_artifact_sha256),
        ("tokenizer_artifact_sha256", expected_tokenizer_artifact_sha256),
    )
    for field, expected in comparisons:
        if expected is not None and metadata[field] != expected:
            raise ValueError(field + " does not match expected value")
    if expected_lora_spec is not None and canonical_json_bytes(
        metadata["lora_spec"]
    ) != canonical_json_bytes(expected_lora_spec):
        raise ValueError("lora_spec does not match expected value")


def load_and_validate_probe_delta_round_shard(
    round_directory: PathLike,
    *,
    expected_round_index: Optional[int] = None,
    expected_manifest_sha256: Optional[str] = None,
    expected_code_commit: Optional[str] = None,
    expected_model_artifact_sha256: Optional[str] = None,
    expected_tokenizer_artifact_sha256: Optional[str] = None,
    expected_lora_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a shard only after its marker, index, tensors, and records verify."""
    directory = Path(round_directory)
    if not directory.is_dir():
        raise NotADirectoryError("round_directory must be an existing directory")
    if expected_round_index is not None:
        expected_round_index = _require_non_negative_integer(
            expected_round_index, "expected_round_index"
        )
    if expected_manifest_sha256 is not None:
        _require_sha256(
            expected_manifest_sha256, "expected_manifest_sha256"
        )
    if expected_code_commit is not None:
        _require_code_commit(expected_code_commit, "expected_code_commit")
    if expected_model_artifact_sha256 is not None:
        _require_sha256(
            expected_model_artifact_sha256,
            "expected_model_artifact_sha256",
        )
    if expected_tokenizer_artifact_sha256 is not None:
        _require_sha256(
            expected_tokenizer_artifact_sha256,
            "expected_tokenizer_artifact_sha256",
        )
    if expected_lora_spec is not None and not isinstance(
        expected_lora_spec, dict
    ):
        raise TypeError("expected_lora_spec must be a dictionary")

    tensor_path = directory / TENSOR_FILE_NAME
    index_path = directory / INDEX_FILE_NAME
    sidecar_path = directory / INDEX_SIDECAR_NAME
    complete_path = directory / COMPLETE_FILE_NAME
    _require_regular_file(tensor_path, TENSOR_FILE_NAME)
    _require_regular_file(index_path, INDEX_FILE_NAME)
    _require_regular_file(sidecar_path, INDEX_SIDECAR_NAME)
    _require_regular_file(complete_path, COMPLETE_FILE_NAME)

    sidecar_sha256 = _read_digest_line(sidecar_path, "index sidecar")
    complete_sha256 = _read_digest_line(complete_path, COMPLETE_FILE_NAME)
    index, index_bytes = _load_canonical_index(index_path)
    actual_index_sha256 = sha256_bytes(index_bytes)
    if sidecar_sha256 != actual_index_sha256:
        raise ValueError("index sidecar SHA-256 does not match index.json")
    if complete_sha256 != actual_index_sha256:
        raise ValueError("COMPLETE SHA-256 does not match index.json")

    if set(index) != _INDEX_KEYS:
        raise ValueError("index.json fields do not match round schema")
    if not matches_schema(index.get("schema_version"), ROUND_SCHEMA_VERSION):
        raise ValueError("round shard schema_version is invalid")
    round_value = _require_non_negative_integer(
        index.get("round_index"), "index round_index"
    )
    if directory.name != _round_directory_name(round_value):
        raise ValueError("round directory name does not match index round_index")
    if expected_round_index is not None and round_value != expected_round_index:
        raise ValueError("round_index does not match expected_round_index")

    record_count = _require_non_negative_integer(
        index.get("record_count"), "record_count"
    )
    records_index = index.get("records")
    if (
        not isinstance(records_index, list)
        or not records_index
        or record_count != len(records_index)
    ):
        raise ValueError("record_count must match a non-empty records list")

    tensor_file = index.get("tensor_file")
    if not isinstance(tensor_file, dict) or set(tensor_file) != {
        "name",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("tensor_file fields are invalid")
    if tensor_file.get("name") != TENSOR_FILE_NAME:
        raise ValueError("tensor_file name is invalid")
    expected_tensor_sha256 = _require_sha256(
        tensor_file.get("sha256"), "tensor_file sha256"
    )
    tensor_size = _require_non_negative_integer(
        tensor_file.get("size_bytes"), "tensor_file size_bytes"
    )
    if tensor_path.stat().st_size != tensor_size:
        raise ValueError("Safetensors file size does not match index")
    if sha256_file(tensor_path) != expected_tensor_sha256:
        raise ValueError("Safetensors file SHA-256 does not match index")

    runtime = index.get("storage_runtime")
    if not isinstance(runtime, dict) or runtime != {
        "torch_version": torch.__version__,
        "safetensors_version": safetensors.__version__,
    }:
        raise ValueError("storage_runtime does not match current runtime")

    prior_metadata_by_id = {}
    seen_record_ids = set()
    seen_client_steps = set()
    shard_identity_baseline = None
    expected_storage_keys = set()
    validated_index_records = []
    for record_ordinal, record_entry in enumerate(records_index):
        if not isinstance(record_entry, dict) or set(
            record_entry
        ) != _INDEX_RECORD_KEYS:
            raise ValueError("index record fields are invalid")
        if record_entry.get("record_ordinal") != record_ordinal or isinstance(
            record_entry.get("record_ordinal"), bool
        ):
            raise ValueError("record_ordinal must be contiguous from zero")

        candidate_metadata = record_entry.get("metadata")
        candidate_record_id = (
            candidate_metadata.get("record_id")
            if isinstance(candidate_metadata, dict)
            else None
        )
        if candidate_record_id in seen_record_ids:
            raise ValueError("record_id must be unique within a round shard")
        metadata, shard_identity_baseline = _validate_record_metadata(
            record_entry.get("metadata"),
            round_value,
            prior_metadata_by_id,
            seen_client_steps,
            shard_identity_baseline,
        )
        record_id = metadata["record_id"]
        seen_record_ids.add(record_id)
        _validate_expected_metadata(
            metadata,
            expected_manifest_sha256,
            expected_code_commit,
            expected_model_artifact_sha256,
            expected_tokenizer_artifact_sha256,
            expected_lora_spec,
        )

        storage_entries = record_entry.get("tensor_storage")
        ordered_names = metadata["lora_spec"]["ordered_keys"]
        if (
            not isinstance(storage_entries, list)
            or len(storage_entries) != len(ordered_names)
        ):
            raise ValueError("tensor_storage count does not match lora_spec")
        seen_names = set()
        for tensor_ordinal, storage_entry in enumerate(storage_entries):
            if not isinstance(storage_entry, dict) or set(
                storage_entry
            ) != _TENSOR_STORAGE_KEYS:
                raise ValueError("tensor_storage entry fields are invalid")
            expected_name = ordered_names[tensor_ordinal]
            if storage_entry.get("name") != expected_name:
                raise ValueError("tensor_storage name order does not match lora_spec")
            if expected_name in seen_names:
                raise ValueError("tensor name is duplicated within a record")
            seen_names.add(expected_name)

            expected_storage_key = "record-{0:04d}.tensor-{1:04d}".format(
                record_ordinal,
                tensor_ordinal,
            )
            if storage_entry.get("storage_key") != expected_storage_key:
                raise ValueError("tensor_storage storage_key is invalid")
            if expected_storage_key in expected_storage_keys:
                raise ValueError("Safetensors storage key is duplicated")
            expected_storage_keys.add(expected_storage_key)

            expected_shape = metadata["lora_spec"]["tensors"][expected_name][
                "shape"
            ]
            if storage_entry.get("shape") != expected_shape:
                raise ValueError("tensor_storage shape does not match lora_spec")
            if storage_entry.get("dtype") != "float32":
                raise ValueError("tensor_storage dtype must be float32")

        validated_index_records.append(
            {
                "record_ordinal": record_ordinal,
                "metadata": metadata,
                "tensor_storage": copy.deepcopy(storage_entries),
            }
        )
        prior_metadata_by_id[record_id] = metadata

    with safe_open(
        str(tensor_path),
        framework="pt",
        device="cpu",
    ) as tensor_handle:
        actual_storage_keys = set(tensor_handle.keys())
        file_metadata = tensor_handle.metadata() or {}
    if actual_storage_keys != expected_storage_keys:
        missing = sorted(expected_storage_keys - actual_storage_keys)
        extra = sorted(actual_storage_keys - expected_storage_keys)
        raise ValueError(
            "Safetensors storage key set does not match index; missing="
            + repr(missing)
            + ", extra="
            + repr(extra)
        )
    if file_metadata != {
        "schema_version": index["schema_version"],
        "round_index": str(round_value),
        "record_count": str(record_count),
    }:
        raise ValueError("Safetensors metadata does not match round index")

    stored_tensors = load_file(str(tensor_path), device="cpu")
    output_records = []
    for record_entry in validated_index_records:
        metadata = record_entry["metadata"]
        restored_tensors = {}
        for storage_entry in record_entry["tensor_storage"]:
            tensor = stored_tensors[storage_entry["storage_key"]]
            if tensor.device.type != "cpu":
                raise ValueError("loaded tensor must be on CPU")
            if tensor.dtype != torch.float32:
                raise ValueError("loaded tensor dtype must be float32")
            if tensor.ndim != 2:
                raise ValueError("loaded tensor must be two-dimensional")
            if list(tensor.shape) != storage_entry["shape"]:
                raise ValueError("loaded tensor shape does not match index")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError("loaded tensor contains NaN or Inf")
            restored_tensors[storage_entry["name"]] = tensor.contiguous()

        payload_sha256 = compute_delta_tensor_payload_sha256(
            restored_tensors,
            metadata["lora_spec"],
            schema_version=metadata["schema_version"],
        )
        if payload_sha256 != metadata["tensor_payload_sha256"]:
            raise ValueError("loaded tensor payload SHA-256 does not match metadata")
        output_records.append(
            {
                "metadata": copy.deepcopy(metadata),
                "delta_tensors": restored_tensors,
            }
        )

    return {
        "index": copy.deepcopy(index),
        "records": output_records,
    }

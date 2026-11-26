"""Deterministic identities for local Hugging Face model artifacts."""

import copy
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .artifact_schema import matches_schema
from .data_manifest import canonical_json_bytes, sha256_bytes


PathValue = Union[str, os.PathLike]

MODEL_SCHEMA_VERSION = "safe-fcl.model-artifact-identity.v1"
TOKENIZER_SCHEMA_VERSION = "safe-fcl.tokenizer-artifact-identity.v1"

_ARTIFACT_SCHEMAS = {
    "model": MODEL_SCHEMA_VERSION,
    "tokenizer": TOKENIZER_SCHEMA_VERSION,
}
_IDENTITY_KEYS = {
    "schema_version",
    "artifact_type",
    "root_name",
    "files",
    "artifact_sha256",
}
_FILE_ENTRY_KEYS = {
    "relative_path",
    "size_bytes",
    "sha256",
}
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_SHARD_RE = re.compile(r"^model-.*\.safetensors$")
_HASH_CHUNK_SIZE = 1024 * 1024
_CHANGED_DURING_BUILD = (
    "artifact file changed while identity was being built"
)
_TOKENIZER_OPTIONAL_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
)


def _reject_duplicate_json_keys(
    pairs: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + str(key))
        result[key] = value
    return result


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError("non-standard JSON constant is not allowed: " + value)


def _load_strict_json_bytes(
    raw_bytes: bytes,
    name: str,
) -> Dict[str, Any]:
    if not isinstance(raw_bytes, bytes):
        raise TypeError("raw_bytes must be bytes")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(name + " must be UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_standard_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(name + " is not valid strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(name + " root must be a JSON object")
    return value


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value:
            return True
        return any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def _declares_custom_remote_code(config: Dict[str, Any]) -> bool:
    if _contains_key(config, "auto_map"):
        return True
    return _contains_truthy_key(config, "trust_remote_code")


def _contains_truthy_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value and bool(value[key]):
            return True
        return any(
            _contains_truthy_key(child, key) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_truthy_key(child, key) for child in value)
    return False


def _coerce_directory(value: PathValue) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise TypeError("model_directory must be a path string or os.PathLike")
    if isinstance(value, str) and not value:
        raise ValueError("model_directory must not be empty")

    candidate = Path(value)
    try:
        candidate_stat = candidate.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "model_directory does not exist: " + str(candidate)
        ) from exc
    if stat.S_ISLNK(candidate_stat.st_mode) or candidate.is_symlink():
        raise ValueError("model_directory must not be a symlink")
    if not stat.S_ISDIR(candidate_stat.st_mode):
        raise NotADirectoryError(
            "model_directory must be an existing directory"
        )

    root = candidate.resolve(strict=True)
    if not root.name:
        raise ValueError("model_directory must have a non-empty root name")
    return root


def _lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _require_safe_basename(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(name + " must be a non-empty string")
    if value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(name + " must be a safe relative basename")

    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or posix_path.name != value
        or windows_path.name != value
    ):
        raise ValueError(name + " must be a safe relative basename")
    return value


def _artifact_path(root: Path, relative_path: str) -> Path:
    safe_name = _require_safe_basename(relative_path, "artifact relative path")
    path = root / safe_name
    if path.parent != root:
        raise ValueError("artifact file resolves outside model_directory")
    return path


def _stat_snapshot(stat_result: os.stat_result) -> Dict[str, int]:
    mtime_ns = getattr(stat_result, "st_mtime_ns", None)
    if mtime_ns is None:
        mtime_ns = int(stat_result.st_mtime * 1_000_000_000)
    return {
        "st_dev": int(stat_result.st_dev),
        "st_ino": int(stat_result.st_ino),
        "st_size": int(stat_result.st_size),
        "st_mtime_ns": int(mtime_ns),
    }


def _stat_identity_available(snapshot: Dict[str, int]) -> bool:
    return (snapshot["st_dev"], snapshot["st_ino"]) != (0, 0)


def _stat_snapshots_match(
    left: Dict[str, int],
    right: Dict[str, int],
) -> bool:
    if (
        left["st_size"] != right["st_size"]
        or left["st_mtime_ns"] != right["st_mtime_ns"]
    ):
        return False
    if _stat_identity_available(left) or _stat_identity_available(right):
        return (
            left["st_dev"],
            left["st_ino"],
        ) == (
            right["st_dev"],
            right["st_ino"],
        )
    return True


def _raise_changed() -> None:
    raise RuntimeError(_CHANGED_DURING_BUILD)


def _lstat_regular_file(
    root: Path,
    relative_path: str,
    *,
    changed_if_missing: bool,
) -> Tuple[Path, Dict[str, int]]:
    path = _artifact_path(root, relative_path)
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        if changed_if_missing:
            raise RuntimeError(_CHANGED_DURING_BUILD) from exc
        raise FileNotFoundError(
            "artifact file is missing: " + relative_path
        ) from exc
    if stat.S_ISLNK(path_stat.st_mode) or path.is_symlink():
        raise ValueError(
            "artifact file must not be a symlink: " + relative_path
        )
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(
            "artifact file must be a regular file: " + relative_path
        )
    return path, _stat_snapshot(path_stat)


def _freeze_file_stats(
    root: Path,
    relative_paths: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("artifact file paths must be unique")
    snapshots = {}
    for relative_path in sorted(relative_paths):
        _path, snapshot = _lstat_regular_file(
            root,
            relative_path,
            changed_if_missing=False,
        )
        snapshots[relative_path] = snapshot
    return snapshots


def _hash_open_file(
    handle: Any,
    *,
    capture_bytes: bool,
) -> Tuple[str, int, Optional[bytes]]:
    digest = hashlib.sha256()
    total_size = 0
    chunks = [] if capture_bytes else None
    while True:
        chunk = handle.read(_HASH_CHUNK_SIZE)
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise TypeError("artifact file handle must return bytes")
        digest.update(chunk)
        total_size += len(chunk)
        if chunks is not None:
            chunks.append(chunk)
    raw_bytes = b"".join(chunks) if chunks is not None else None
    return digest.hexdigest(), total_size, raw_bytes


def _read_stable_file(
    root: Path,
    relative_path: str,
    frozen_snapshot: Dict[str, int],
    *,
    capture_bytes: bool,
) -> Tuple[Optional[bytes], Dict[str, Any], Dict[str, int]]:
    path, before_open = _lstat_regular_file(
        root,
        relative_path,
        changed_if_missing=True,
    )
    if not _stat_snapshots_match(frozen_snapshot, before_open):
        _raise_changed()

    try:
        handle = path.open("rb")
    except FileNotFoundError as exc:
        raise RuntimeError(_CHANGED_DURING_BUILD) from exc
    try:
        opened_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened_stat.st_mode):
            _raise_changed()
        opened_snapshot = _stat_snapshot(opened_stat)
        if not _stat_snapshots_match(before_open, opened_snapshot):
            _raise_changed()

        digest, total_size, raw_bytes = _hash_open_file(
            handle,
            capture_bytes=capture_bytes,
        )
        after_read_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(after_read_stat.st_mode):
            _raise_changed()
        after_read_snapshot = _stat_snapshot(after_read_stat)
        if not _stat_snapshots_match(opened_snapshot, after_read_snapshot):
            _raise_changed()
        if total_size != after_read_snapshot["st_size"]:
            _raise_changed()

        # Some filesystems can preserve all observed metadata during a
        # same-size overwrite. Verify stability with a second content hash.
        handle.seek(0, os.SEEK_SET)
        verification_digest, verification_size, _ = _hash_open_file(
            handle,
            capture_bytes=False,
        )
        after_verification_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(after_verification_stat.st_mode):
            _raise_changed()
        after_verification_snapshot = _stat_snapshot(after_verification_stat)
        if not _stat_snapshots_match(
            after_read_snapshot,
            after_verification_snapshot,
        ):
            _raise_changed()
        if (
            verification_digest != digest
            or verification_size != total_size
            or verification_size
            != after_verification_snapshot["st_size"]
        ):
            _raise_changed()
        after_read_snapshot = after_verification_snapshot
    finally:
        handle.close()

    try:
        after_close_stat = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(_CHANGED_DURING_BUILD) from exc
    if (
        stat.S_ISLNK(after_close_stat.st_mode)
        or path.is_symlink()
        or not stat.S_ISREG(after_close_stat.st_mode)
    ):
        _raise_changed()
    after_close_snapshot = _stat_snapshot(after_close_stat)
    if not _stat_snapshots_match(after_read_snapshot, after_close_snapshot):
        _raise_changed()

    entry = {
        "relative_path": relative_path,
        "size_bytes": total_size,
        "sha256": digest,
    }
    return raw_bytes, entry, after_close_snapshot


def _read_stable_json_file(
    root: Path,
    relative_path: str,
    name: str,
    frozen_snapshot: Dict[str, int],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, int]]:
    raw_bytes, entry, stable_snapshot = _read_stable_file(
        root,
        relative_path,
        frozen_snapshot,
        capture_bytes=True,
    )
    if raw_bytes is None:  # pragma: no cover - guaranteed by capture_bytes
        raise AssertionError("JSON stable read did not return bytes")
    value = _load_strict_json_bytes(raw_bytes, name)
    return value, entry, stable_snapshot


def _verify_stable_file_snapshots(
    root: Path,
    snapshots: Dict[str, Dict[str, int]],
) -> None:
    for relative_path, expected_snapshot in snapshots.items():
        try:
            path, current_snapshot = _lstat_regular_file(
                root,
                relative_path,
                changed_if_missing=True,
            )
        except ValueError:
            _raise_changed()
        if path.is_symlink() or not _stat_snapshots_match(
            expected_snapshot,
            current_snapshot,
        ):
            _raise_changed()


def _model_layout_snapshot(root: Path) -> Dict[str, Any]:
    return {
        "config": _lexists(root / "config.json"),
        "single": _lexists(root / "model.safetensors"),
        "index": _lexists(root / "model.safetensors.index.json"),
        "model_shards": tuple(
            sorted(
                entry.name
                for entry in root.iterdir()
                if _MODEL_SHARD_RE.fullmatch(entry.name) is not None
            )
        ),
    }


def _tokenizer_layout_snapshot(root: Path) -> Tuple[str, ...]:
    supported_names = (
        "tokenizer_config.json",
    ) + _TOKENIZER_OPTIONAL_FILES
    return tuple(
        sorted(name for name in supported_names if _lexists(root / name))
    )


def _require_unchanged_layout(
    initial_layout: Any,
    final_layout: Any,
) -> None:
    if initial_layout != final_layout:
        raise RuntimeError(
            "artifact file set changed while identity was being built"
        )


def _identity_payload(
    schema_version: str,
    artifact_type: str,
    root_name: str,
    files: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": schema_version,
        "artifact_type": artifact_type,
        "root_name": root_name,
        "files": files,
    }


def _finish_identity(payload: Dict[str, Any]) -> Dict[str, Any]:
    identity = copy.deepcopy(payload)
    identity["artifact_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return validate_artifact_identity(identity)


def artifact_identity_for_schema(identity: Dict[str, Any], schema_version: str) -> Dict[str, Any]:
    """Encode a rebuilt local identity in the namespace recorded by its plan."""
    if identity.get("schema_version") == schema_version:
        return identity
    artifact_type = identity["artifact_type"]
    if not matches_schema(schema_version, _ARTIFACT_SCHEMAS[artifact_type]):
        raise ValueError("artifact identity schema does not match its artifact type")
    return _finish_identity(
        _identity_payload(schema_version, artifact_type, identity["root_name"], identity["files"])
    )


def _validate_identity_file_layout(
    artifact_type: str,
    relative_paths: Sequence[str],
) -> None:
    path_set = set(relative_paths)
    if artifact_type == "model":
        if "config.json" not in path_set:
            raise ValueError("model identity must contain config.json")
        has_single = "model.safetensors" in path_set
        has_index = "model.safetensors.index.json" in path_set
        if has_single == has_index:
            raise ValueError(
                "model identity must contain exactly one supported weight layout"
            )
        if has_single:
            expected = {"config.json", "model.safetensors"}
            if path_set != expected:
                raise ValueError(
                    "single-file model identity contains unsupported files"
                )
            return

        shard_paths = path_set.difference(
            {"config.json", "model.safetensors.index.json"}
        )
        if not shard_paths or any(
            not path.endswith(".safetensors") for path in shard_paths
        ):
            raise ValueError(
                "sharded model identity must contain Safetensors shards"
            )
        return

    if "tokenizer_config.json" not in path_set:
        raise ValueError(
            "tokenizer identity must contain tokenizer_config.json"
        )
    allowed = set(_TOKENIZER_OPTIONAL_FILES)
    unsupported = path_set.difference(allowed, {"tokenizer_config.json"})
    if unsupported:
        raise ValueError(
            "tokenizer identity contains unsupported files: "
            + repr(sorted(unsupported))
        )
    standalone = bool(
        path_set.intersection(
            {"tokenizer.json", "tokenizer.model", "spiece.model"}
        )
    )
    has_vocab = "vocab.json" in path_set
    has_merges = "merges.txt" in path_set
    if has_vocab != has_merges:
        raise ValueError(
            "tokenizer identity requires vocab.json and merges.txt together"
        )
    if not standalone and not (has_vocab and has_merges):
        raise ValueError("tokenizer identity must contain a tokenizer payload")


def _model_shards_from_index(
    index: Dict[str, Any],
    discovered_model_shards: Sequence[str],
) -> List[str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index weight_map must be a non-empty object")

    shard_names = set()
    for parameter_name, shard_value in weight_map.items():
        if not isinstance(parameter_name, str) or not parameter_name:
            raise ValueError(
                "model index weight_map keys must be non-empty strings"
            )
        shard_name = _require_safe_basename(
            shard_value,
            "model index shard filename",
        )
        if not shard_name.endswith(".safetensors"):
            raise ValueError(
                "model index shard filename must end with .safetensors"
            )
        shard_names.add(shard_name)

    unreferenced = sorted(set(discovered_model_shards) - shard_names)
    if unreferenced:
        raise ValueError(
            "unreferenced model Safetensors shards: " + repr(unreferenced)
        )
    return sorted(shard_names)


def build_model_artifact_identity(
    model_directory: PathValue,
) -> Dict[str, Any]:
    """Build an identity for one supported local Safetensors model layout."""
    root = _coerce_directory(model_directory)
    initial_layout = _model_layout_snapshot(root)
    if not initial_layout["config"]:
        raise FileNotFoundError("artifact file is missing: config.json")

    has_single = initial_layout["single"]
    has_index = initial_layout["index"]
    if has_single and has_index:
        raise ValueError(
            "single-file and sharded model layouts must not coexist"
        )
    if not has_single and not has_index:
        raise FileNotFoundError(
            "model directory must contain model.safetensors or "
            "model.safetensors.index.json"
        )

    if has_single:
        extra_shards = list(initial_layout["model_shards"])
        if extra_shards:
            raise ValueError(
                "single-file model layout contains unexpected shards: "
                + repr(extra_shards)
            )
        relative_paths = [
            "config.json",
            "model.safetensors",
        ]
        frozen_snapshots = _freeze_file_stats(root, relative_paths)
        entries = {}
        stable_snapshots = {}
        config, config_entry, config_snapshot = _read_stable_json_file(
            root,
            "config.json",
            "config.json",
            frozen_snapshots["config.json"],
        )
        entries["config.json"] = config_entry
        stable_snapshots["config.json"] = config_snapshot
        if _declares_custom_remote_code(config):
            raise ValueError(
                "model config requiring custom remote code is not supported"
            )
        (
            _raw_bytes,
            model_entry,
            model_snapshot,
        ) = _read_stable_file(
            root,
            "model.safetensors",
            frozen_snapshots["model.safetensors"],
            capture_bytes=False,
        )
        entries["model.safetensors"] = model_entry
        stable_snapshots["model.safetensors"] = model_snapshot
    else:
        initial_known_paths = [
            "config.json",
            "model.safetensors.index.json",
        ] + list(initial_layout["model_shards"])
        frozen_snapshots = _freeze_file_stats(
            root,
            initial_known_paths,
        )
        (
            index,
            index_entry,
            index_snapshot,
        ) = _read_stable_json_file(
            root,
            "model.safetensors.index.json",
            "model.safetensors.index.json",
            frozen_snapshots["model.safetensors.index.json"],
        )
        shards = _model_shards_from_index(
            index,
            initial_layout["model_shards"],
        )
        relative_paths = [
            "config.json",
            "model.safetensors.index.json",
        ] + shards
        for shard_name in shards:
            if not _lexists(root / shard_name):
                raise FileNotFoundError(
                    "artifact file is missing: " + shard_name
                )
            if shard_name not in frozen_snapshots:
                frozen_snapshots.update(
                    _freeze_file_stats(root, [shard_name])
                )

        entries = {
            "model.safetensors.index.json": index_entry,
        }
        stable_snapshots = {
            "model.safetensors.index.json": index_snapshot,
        }
        config, config_entry, config_snapshot = _read_stable_json_file(
            root,
            "config.json",
            "config.json",
            frozen_snapshots["config.json"],
        )
        entries["config.json"] = config_entry
        stable_snapshots["config.json"] = config_snapshot
        if _declares_custom_remote_code(config):
            raise ValueError(
                "model config requiring custom remote code is not supported"
            )
        for shard_name in shards:
            _raw_bytes, shard_entry, shard_snapshot = _read_stable_file(
                root,
                shard_name,
                frozen_snapshots[shard_name],
                capture_bytes=False,
            )
            entries[shard_name] = shard_entry
            stable_snapshots[shard_name] = shard_snapshot

    _verify_stable_file_snapshots(root, stable_snapshots)
    _require_unchanged_layout(
        initial_layout,
        _model_layout_snapshot(root),
    )
    files = [entries[path] for path in sorted(relative_paths)]
    return _finish_identity(
        _identity_payload(
            MODEL_SCHEMA_VERSION,
            "model",
            root.name,
            files,
        )
    )


def build_tokenizer_artifact_identity(
    model_directory: PathValue,
) -> Dict[str, Any]:
    """Build an identity for supported tokenizer payloads in a local model."""
    root = _coerce_directory(model_directory)
    initial_layout = _tokenizer_layout_snapshot(root)
    present = set(initial_layout)
    if "tokenizer_config.json" not in present:
        raise FileNotFoundError(
            "artifact file is missing: tokenizer_config.json"
        )
    present.remove("tokenizer_config.json")
    has_standalone_payload = any(
        name in present
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
        )
    )
    has_vocab = "vocab.json" in present
    has_merges = "merges.txt" in present
    if has_vocab != has_merges:
        raise ValueError(
            "vocab.json and merges.txt must be present together"
        )
    if not has_standalone_payload and not (has_vocab and has_merges):
        raise FileNotFoundError(
            "tokenizer payload is missing; expected tokenizer.json, "
            "tokenizer.model, spiece.model, or vocab.json with merges.txt"
        )

    relative_paths = ["tokenizer_config.json"] + sorted(present)
    frozen_snapshots = _freeze_file_stats(root, relative_paths)
    entries = {}
    stable_snapshots = {}
    config, config_entry, config_snapshot = _read_stable_json_file(
        root,
        "tokenizer_config.json",
        "tokenizer_config.json",
        frozen_snapshots["tokenizer_config.json"],
    )
    entries["tokenizer_config.json"] = config_entry
    stable_snapshots["tokenizer_config.json"] = config_snapshot
    if _declares_custom_remote_code(config):
        raise ValueError(
            "tokenizer config requiring custom remote code is not supported"
        )
    for relative_path in sorted(present):
        _raw_bytes, entry, snapshot = _read_stable_file(
            root,
            relative_path,
            frozen_snapshots[relative_path],
            capture_bytes=False,
        )
        entries[relative_path] = entry
        stable_snapshots[relative_path] = snapshot

    _verify_stable_file_snapshots(root, stable_snapshots)
    _require_unchanged_layout(
        initial_layout,
        _tokenizer_layout_snapshot(root),
    )
    files = [entries[path] for path in sorted(relative_paths)]
    return _finish_identity(
        _identity_payload(
            TOKENIZER_SCHEMA_VERSION,
            "tokenizer",
            root.name,
            files,
        )
    )


def validate_artifact_identity(
    identity: Dict[str, Any],
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a canonical artifact identity and return a defensive copy."""
    if not isinstance(identity, dict):
        raise TypeError("identity must be a dictionary")
    if set(identity) != _IDENTITY_KEYS:
        raise ValueError("artifact identity fields do not match schema")

    schema_version = identity.get("schema_version")
    artifact_type = identity.get("artifact_type")
    if not isinstance(artifact_type, str) or artifact_type not in _ARTIFACT_SCHEMAS:
        raise ValueError("artifact_type is unsupported")
    if not matches_schema(schema_version, _ARTIFACT_SCHEMAS[artifact_type]):
        raise ValueError(
            "artifact_type does not match artifact identity schema"
        )

    root_name = identity.get("root_name")
    _require_safe_basename(root_name, "root_name")

    files = identity.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("artifact identity files must be a non-empty list")

    normalized_paths = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != _FILE_ENTRY_KEYS:
            raise ValueError("artifact file entry fields do not match schema")
        relative_path = _require_safe_basename(
            entry.get("relative_path"),
            "artifact relative_path",
        )
        size_bytes = entry.get("size_bytes")
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ValueError(
                "artifact file size_bytes must be a non-negative integer"
            )
        digest = entry.get("sha256")
        if not isinstance(digest, str) or _HEX_64_RE.fullmatch(digest) is None:
            raise ValueError(
                "artifact file sha256 must be 64 lowercase hexadecimal characters"
            )
        normalized_paths.append(relative_path)

    if normalized_paths != sorted(normalized_paths):
        raise ValueError("artifact files must be sorted by relative_path")
    if len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError("artifact relative_path values must be unique")
    _validate_identity_file_layout(artifact_type, normalized_paths)

    actual_sha256 = identity.get("artifact_sha256")
    if (
        not isinstance(actual_sha256, str)
        or _HEX_64_RE.fullmatch(actual_sha256) is None
    ):
        raise ValueError(
            "artifact_sha256 must be 64 lowercase hexadecimal characters"
        )
    payload = {
        key: copy.deepcopy(value)
        for key, value in identity.items()
        if key != "artifact_sha256"
    }
    computed_sha256 = sha256_bytes(canonical_json_bytes(payload))
    if computed_sha256 != actual_sha256:
        raise ValueError("artifact_sha256 does not match canonical identity")

    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or _HEX_64_RE.fullmatch(expected_sha256) is None
        ):
            raise ValueError(
                "expected_sha256 must be 64 lowercase hexadecimal characters"
            )
        if expected_sha256 != actual_sha256:
            raise ValueError("artifact identity does not match expected_sha256")

    return copy.deepcopy(identity)

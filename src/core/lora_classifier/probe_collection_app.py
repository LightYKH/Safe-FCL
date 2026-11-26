"""Application orchestration for a strict offline Probe collection run.

Only Python standard-library modules are imported at module scope.  All
training, dataset, model, and storage modules are resolved lazily after the
offline environment has been configured.
"""

import copy
import gc
import hashlib
import importlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.artifact_schema import matches_schema

RUN_SCHEMA_VERSION = "safe-fcl.probe-collection-run.v3"
PLAN_FILE_NAME = "probe-collection-plan.json"
PLAN_SIDECAR_NAME = PLAN_FILE_NAME + ".sha256"
FULL_SUMMARY_NAME = "probe-collection-run.json"
SMOKE_SUMMARY_NAME = "probe-collection-smoke.json"
FULL_MARKER_NAME = "COMPLETE"
SMOKE_MARKER_NAME = "SMOKE_COMPLETE"
LOCK_NAME = ".probe-collection.lock"

_EXPECTED_BRANCH = os.environ.get(
    "SAFE_FCL_PROBE_EXPECTED_BRANCH",
    "main",
)
_ALLOW_DETACHED_HEAD = (
    os.environ.get("SAFE_FCL_PROBE_ALLOW_DETACHED_HEAD", "false").lower()
    == "true"
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SCRATCH_NAME = re.compile(r"^round-[0-9]{3}-client-[0-9]{3}$")
_MODES = ("preflight-only", "smoke-round-0", "full")
_SUMMARY_KEYS = {
    "schema_version",
    "run_sha256",
    "mode",
    "plan_sha256",
    "manifest_sha256",
    "code_commit",
    "model_artifact_sha256",
    "tokenizer_artifact_sha256",
    "collection_r0_tensor_sha256",
    "bitsandbytes_version",
    "runtime_report",
    "protocol_summary",
    "lora_spec",
    "rounds",
    "summary",
}
_ROUND_KEYS = {
    "round_index",
    "round_directory_name",
    "selected_client_ids",
    "record_count",
    "sample_events",
    "benign_records",
    "malicious_records",
    "index_sha256",
    "delta_base_round",
    "delta_base_tensor_sha256",
    "collection_r0_tensor_sha256",
    "local_state_tensor_sha256",
    "sample_counts",
    "fedavg_candidate_tensor_sha256",
    "lr_policy",
    "learning_rate",
}
_RUNTIME_REPORT_KEYS = {
    "platform",
    "python",
    "package_versions",
    "bitsandbytes_version",
    "torch_cuda_version",
    "cuda_device_count",
    "cuda_device_name",
    "cuda_visible_devices",
    "nccl_p2p_disable",
    "nccl_ib_disable",
    "world_size",
    "optimizer_smoke_passed",
}
_RUNTIME_PACKAGE_KEYS = {
    "torch",
    "transformers",
    "trl",
    "accelerate",
    "peft",
    "datasets",
    "numpy",
    "pandas",
    "pyarrow",
    "safetensors",
}
_FROZEN_CLIENT_SCHEDULE = (
    [0, 6, 9],
    [1, 2, 4],
    [0, 1, 8],
    [2, 3, 8],
    [1, 3, 4],
    [4, 5, 9],
    [1, 7, 9],
    [2, 5, 6],
    [3, 5, 6],
    [4, 5, 7],
)
_PROTOCOL_SUMMARY = {
    "base_model_quantization": "8bit",
    "batch_size": 4,
    "clients": 10,
    "clients_benign": 5,
    "clients_malicious": 5,
    "clients_per_round": 3,
    "completion_marker_policy": (
        "derive from the actual formatted answer boundary; "
        "no fixed token slicing"
    ),
    "federated_algorithm": "fedavg",
    "formatter_id": "utils.template.get_formatting_prompts_func:alpaca",
    "gradient_accumulation_steps": 4,
    "learning_rate_schedule": "task1_main_100_round_cosine_first_10",
    "learning_rate_schedule_source": (
        "per-task frozen 100-round cosine 5e-5->1e-6"
    ),
    "learning_rates": [
        0.000001
        + 0.5
        * (0.00005 - 0.000001)
        * (1.0 + math.cos(math.pi * index / 99))
        for index in range(10)
    ],
    "collection_state_contract": "advancing-round early-FL collection",
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_initialization_seed": 2023,
    "lora_rank": 8,
    "lora_target_modules": ["q_proj", "v_proj"],
    "max_sequence_length": 512,
    "optimizer": "adamw_bnb_8bit",
    "optimizer_steps_per_client": 10,
    "packing": False,
    "resume_from_checkpoint": False,
    "single_process": True,
    "template": "alpaca",
    "trainer_data_seed": None,
    "trainer_seed": 42,
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError("SHA-256 input must be bytes")
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(name + " must be a 64-character lowercase SHA-256")
    return value


def _require_commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ValueError(name + " must be a 40-character lowercase Git commit")
    return value


def _require_mode(value: Any) -> str:
    if value not in _MODES:
        raise ValueError(
            "mode must be preflight-only, smoke-round-0, or full"
        )
    return value


def _lazy_function(module_name: str, attribute: str) -> Callable:
    def call(*args, **kwargs):
        module = importlib.import_module(module_name)
        return getattr(module, attribute)(*args, **kwargs)

    return call


def _default_dependency_provider() -> Dict[str, Callable]:
    return {
        "configure_offline": _lazy_function(
            "lora_classifier.probe_collection_runtime",
            "configure_probe_offline_environment",
        ),
        "collect_git": _lazy_function(
            "lora_classifier.probe_collection_plan",
            "collect_probe_git_provenance",
        ),
        "load_plan": _lazy_function(
            "lora_classifier.probe_collection_plan",
            "load_and_validate_probe_collection_plan",
        ),
        "load_manifest": _lazy_function(
            "utils.manifest_training_data",
            "load_training_manifest",
        ),
        "validate_manifest_provenance": _lazy_function(
            "utils.manifest_training_data",
            "validate_manifest_training_provenance",
        ),
        "validate_probe_manifest": _lazy_function(
            "utils.manifest_probe_data",
            "validate_probe_collection_manifest",
        ),
        "rebuild_probe_pools": _lazy_function(
            "utils.manifest_probe_data",
            "rebuild_manifest_probe_pools",
        ),
        "bind_probe_pools": _lazy_function(
            "lora_classifier.probe_collection_runner",
            "bind_probe_pools_to_plan",
        ),
        "validate_round_space": _lazy_function(
            "lora_classifier.probe_collection_runner",
            "validate_probe_round_output_space",
        ),
        "validate_gpu": _lazy_function(
            "lora_classifier.probe_collection_runtime",
            "validate_probe_gpu_runtime",
        ),
        "validate_artifacts": _lazy_function(
            "lora_classifier.probe_collection_runtime",
            "validate_local_artifacts_against_plan",
        ),
        "load_model": _lazy_function(
            "lora_classifier.probe_collection_runtime",
            "load_local_probe_backbone_and_tokenizer",
        ),
        "initialize_lora": _lazy_function(
            "lora_classifier.probe_collection_runtime",
            "initialize_probe_lora_model",
        ),
        "train_client": _lazy_function(
            "lora_classifier.probe_collection_runtime",
            "train_probe_client",
        ),
        "clone_state": _lazy_function(
            "lora_classifier.probe_collection_runner",
            "clone_lora_state",
        ),
        "execute_round": _lazy_function(
            "lora_classifier.probe_collection_runner",
            "execute_probe_round",
        ),
        "write_round": _lazy_function(
            "lora_classifier.probe_delta_store",
            "write_probe_delta_round_shard",
        ),
        "read_round": _lazy_function(
            "lora_classifier.probe_delta_store",
            "load_and_validate_probe_delta_round_shard",
        ),
        "get_formatter": _lazy_function(
            "utils.template",
            "get_formatting_prompts_func",
        ),
        "state_sha": _lazy_function(
            "utils.lora_state_guard",
            "canonical_lora_tensor_sha256",
        ),
        "get_torch": lambda: importlib.import_module("torch"),
    }


def _dependencies(
    dependency_provider: Optional[Callable[[], Mapping]],
) -> Mapping:
    dependencies = (
        _default_dependency_provider()
        if dependency_provider is None
        else dependency_provider()
    )
    if not isinstance(dependencies, Mapping):
        raise TypeError("dependency_provider must return a Mapping")
    required = set(_default_dependency_provider())
    missing = sorted(required - set(dependencies))
    if missing:
        raise ValueError(
            "dependency_provider is missing required values: " + repr(missing)
        )
    if any(not callable(dependencies[name]) for name in required):
        raise TypeError("all dependency provider values must be callable")
    return dependencies


def validate_collection_git_state(
    repository_root: Any,
    expected_code_commit: str,
    *,
    git_provider: Optional[Callable[[Path, str], Mapping]] = None,
    default_git_provider: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Validate the formal branch, full commit, and completely clean index."""
    expected = _require_commit(
        expected_code_commit,
        "expected_code_commit",
    )
    if not isinstance(repository_root, (str, os.PathLike)):
        raise TypeError("repository_root must be a path")
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(
            "repository_root must be an existing directory"
        )
    provider = git_provider or default_git_provider
    if not callable(provider):
        raise TypeError("git provider must be callable")
    result = provider(root, expected)
    if not isinstance(result, Mapping):
        raise TypeError("git provider must return a Mapping")
    branch = result.get("branch")
    commit = result.get("commit", result.get("head"))
    detached = result.get("detached") is True or not branch
    if detached and not _ALLOW_DETACHED_HEAD:
        raise RuntimeError("detached HEAD is not supported")
    if not detached and branch != _EXPECTED_BRANCH:
        raise RuntimeError(
            "Probe collection requires branch " + _EXPECTED_BRANCH
        )
    commit = _require_commit(commit, "current Git commit")
    if commit != expected:
        raise ValueError("current Git commit does not match expected commit")
    dirty_fields = (
        "staged_dirty",
        "worktree_dirty",
        "untracked",
    )
    if result.get("clean") is not True:
        if any(result.get(name) for name in dirty_fields):
            raise RuntimeError(
                "Git worktree and index must be clean with no untracked files"
            )
        raise RuntimeError("Git provider did not prove a clean repository")
    if any(result.get(name) for name in dirty_fields):
        raise RuntimeError(
            "Git worktree and index must be clean with no untracked files"
        )
    return {
        "branch": branch or None,
        "commit": commit,
        "clean": True,
        "detached": detached,
    }


def _require_existing_directory(value: Any, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(name + " must be a path")
    path = Path(value)
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(name + " does not exist") from exc
    if stat.S_ISLNK(path_stat.st_mode) or path.is_symlink():
        raise ValueError(name + " must not be a symlink")
    if not stat.S_ISDIR(path_stat.st_mode):
        raise NotADirectoryError(name + " must be a directory")
    if not os.access(str(path), os.W_OK):
        raise PermissionError(name + " must be writable")
    return path.resolve()


def _require_regular_nonsymlink(path: Path, name: str) -> None:
    try:
        value = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(name + " is missing") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise ValueError(name + " must be a regular non-symlink file")


def validate_collection_output_space(
    output_root: Any,
    plan_path: Any,
    *,
    validate_round_space: Callable,
) -> Path:
    """Validate a fresh run namespace that already contains its frozen plan."""
    root = _require_existing_directory(output_root, "output_root")
    if not isinstance(plan_path, (str, os.PathLike)):
        raise TypeError("plan_path must be a path")
    supplied_plan = Path(plan_path)
    expected_plan = root / PLAN_FILE_NAME
    if supplied_plan.resolve() != expected_plan.resolve():
        raise ValueError(
            "plan_path must equal <output_root>/probe-collection-plan.json"
        )
    _require_regular_nonsymlink(expected_plan, "Probe collection plan")
    _require_regular_nonsymlink(
        root / PLAN_SIDECAR_NAME,
        "Probe collection plan sidecar",
    )
    forbidden = (
        FULL_SUMMARY_NAME,
        FULL_SUMMARY_NAME + ".sha256",
        SMOKE_SUMMARY_NAME,
        SMOKE_SUMMARY_NAME + ".sha256",
        FULL_MARKER_NAME,
        SMOKE_MARKER_NAME,
        LOCK_NAME,
    )
    for name in forbidden:
        if os.path.lexists(str(root / name)):
            raise FileExistsError(
                "Probe collection output target already exists: " + name
            )
    validate_round_space(root, round_count=10)
    return root


def validate_collection_scratch_space(scratch_root: Any) -> Path:
    root = _require_existing_directory(scratch_root, "scratch_root")
    entries = os.scandir(str(root))
    try:
        for entry in entries:
            if _SCRATCH_NAME.fullmatch(entry.name):
                raise FileExistsError(
                    "Probe client scratch path already exists: " + entry.name
                )
    finally:
        close = getattr(entries, "close", None)
        if callable(close):
            close()
    return root


def _lock_payload(
    code_commit: str,
    plan_sha256: str,
    mode: str,
) -> Dict[str, Any]:
    return {
        "code_commit": _require_commit(code_commit, "code_commit"),
        "mode": _require_mode(mode),
        "pid": os.getpid(),
        "plan_sha256": _require_sha256(plan_sha256, "plan_sha256"),
    }


def create_probe_collection_lock(
    output_root: Any,
    *,
    code_commit: str,
    plan_sha256: str,
    mode: str,
) -> Dict[str, Any]:
    """Create the run lock no-clobber and return its filesystem identity."""
    root = _require_existing_directory(output_root, "output_root")
    target = root / LOCK_NAME
    payload = _lock_payload(code_commit, plan_sha256, mode)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(target), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    target_stat = os.stat(target, follow_symlinks=False)
    return {
        "path": target,
        "st_dev": target_stat.st_dev,
        "st_ino": target_stat.st_ino,
        "payload": copy.deepcopy(payload),
    }


def remove_probe_collection_lock(lock_identity: Mapping) -> None:
    """Remove only the lock inode created by this controlled invocation."""
    if not isinstance(lock_identity, Mapping):
        raise TypeError("lock_identity must be a Mapping")
    path = lock_identity.get("path")
    if not isinstance(path, Path):
        raise TypeError("lock identity path must be a Path")
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        current.st_dev,
        current.st_ino,
    ) != (
        lock_identity.get("st_dev"),
        lock_identity.get("st_ino"),
    ):
        raise RuntimeError("Probe collection lock was replaced")
    path.unlink()


def _reject_duplicate_keys(
    pairs: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + str(key))
        result[key] = value
    return result


def _run_digest(summary: Mapping) -> str:
    payload = {
        key: copy.deepcopy(value)
        for key, value in summary.items()
        if key != "run_sha256"
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _validate_runtime_report(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _RUNTIME_REPORT_KEYS:
        raise ValueError("runtime_report fields do not match schema")
    if value.get("platform") != "Linux":
        raise ValueError("runtime_report platform must be Linux")
    if not isinstance(value.get("python"), str) or not value["python"]:
        raise ValueError("runtime_report Python version must be non-empty")
    if value.get("optimizer_smoke_passed") is not True:
        raise ValueError("runtime_report must prove optimizer smoke success")
    if value.get("cuda_device_count") != 1 or isinstance(
        value.get("cuda_device_count"), bool
    ):
        raise ValueError("runtime_report must contain one CUDA device")
    if not isinstance(value.get("cuda_device_name"), str) or not value[
        "cuda_device_name"
    ]:
        raise ValueError("runtime_report CUDA device name must be non-empty")
    if (
        not isinstance(value.get("torch_cuda_version"), str)
        or not value["torch_cuda_version"]
    ):
        raise ValueError("runtime_report CUDA version must be non-empty")
    visible_devices = value.get("cuda_visible_devices")
    if visible_devices is not None and (
        not isinstance(visible_devices, str)
        or not visible_devices.strip()
        or "," in visible_devices
    ):
        raise ValueError(
            "runtime_report CUDA_VISIBLE_DEVICES must identify one device"
        )
    if value.get("world_size") != "1":
        raise ValueError("runtime_report world_size must equal 1")
    if value.get("nccl_p2p_disable") != "1":
        raise ValueError("runtime_report NCCL_P2P_DISABLE must equal 1")
    if value.get("nccl_ib_disable") != "1":
        raise ValueError("runtime_report NCCL_IB_DISABLE must equal 1")
    if (
        not isinstance(value.get("bitsandbytes_version"), str)
        or not value["bitsandbytes_version"]
    ):
        raise ValueError(
            "runtime_report bitsandbytes version must be non-empty"
        )
    package_versions = value.get("package_versions")
    if (
        not isinstance(package_versions, dict)
        or set(package_versions) != _RUNTIME_PACKAGE_KEYS
        or any(
            not isinstance(version, str) or not version
            for version in package_versions.values()
        )
    ):
        raise ValueError(
            "runtime_report package_versions do not match frozen packages"
        )
    return copy.deepcopy(value)


def _validate_lora_spec(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "filter_policy",
        "layer_index",
        "rank",
        "ordered_keys",
        "tensors",
    }:
        raise ValueError("lora_spec fields do not match tensor schema")
    if value.get("rank") != 8 or isinstance(value.get("rank"), bool):
        raise ValueError("lora_spec rank must be 8")
    keys = value.get("ordered_keys")
    tensors = value.get("tensors")
    if not isinstance(keys, list) or len(keys) != 2:
        raise ValueError("lora_spec must contain two ordered LoRA-B keys")
    if not isinstance(tensors, dict) or set(tensors) != set(keys):
        raise ValueError("lora_spec tensors must match ordered_keys")
    if keys != sorted(keys) or len(set(keys)) != 2:
        raise ValueError("lora_spec ordered_keys must be unique and sorted")
    if value.get("filter_policy") != (
        "first_transformer_layer_all_lora_B"
    ):
        raise ValueError("lora_spec filter_policy is invalid")
    if value.get("layer_index") != 0 or isinstance(
        value.get("layer_index"), bool
    ):
        raise ValueError("lora_spec layer_index must be 0")
    targets = {
        tensors[key].get("resolved_target_module")
        for key in keys
        if isinstance(tensors[key], dict)
    }
    if targets != {"q_proj", "v_proj"}:
        raise ValueError("lora_spec targets must be q_proj and v_proj")
    shapes = {
        tensors[key]["resolved_target_module"]: tensors[key].get("shape")
        for key in keys
    }
    if shapes != {"q_proj": [4096, 8], "v_proj": [1024, 8]}:
        raise ValueError("lora_spec q_proj/v_proj shapes are invalid")
    module_paths = set()
    for key in keys:
        entry = tensors[key]
        if not isinstance(entry, dict) or set(entry) != {
            "shape",
            "dtype",
            "resolved_module_path",
            "resolved_target_module",
        }:
            raise ValueError("lora_spec tensor fields do not match schema")
        if not isinstance(entry["dtype"], str) or not entry["dtype"]:
            raise ValueError("lora_spec tensor dtype must be non-empty")
        module_path = entry["resolved_module_path"]
        target = entry["resolved_target_module"]
        if (
            not isinstance(module_path, str)
            or not module_path
            or not key.startswith(module_path + ".lora_B.")
            or not module_path.endswith("." + target)
            or module_path in module_paths
        ):
            raise ValueError("lora_spec resolved module path is invalid")
        module_paths.add(module_path)
    return copy.deepcopy(value)


def _validate_round_identity(
    value: Any,
    expected_index: int,
) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ROUND_KEYS:
        raise ValueError("run round identity fields do not match schema")
    if value.get("round_index") != expected_index or isinstance(
        value.get("round_index"), bool
    ):
        raise ValueError("run round indexes must be contiguous from zero")
    if value.get("round_directory_name") != (
        "round-" + format(expected_index, "03d")
    ):
        raise ValueError("run round directory name is incorrect")
    selected = value.get("selected_client_ids")
    if (
        not isinstance(selected, list)
        or len(selected) != 3
        or len(set(selected)) != 3
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            or item > 9
            for item in selected
        )
    ):
        raise ValueError("run round must contain three distinct client IDs")
    if selected != _FROZEN_CLIENT_SCHEDULE[expected_index]:
        raise ValueError("run round clients do not match frozen plan schedule")
    if value.get("record_count") != 30 or isinstance(
        value.get("record_count"), bool
    ):
        raise ValueError("run round record_count must be 30")
    if value.get("sample_events") != 480 or isinstance(
        value.get("sample_events"), bool
    ):
        raise ValueError("run round sample_events must be 480")
    benign_records = value.get("benign_records")
    malicious_records = value.get("malicious_records")
    expected_benign_records = sum(
        client_id < 5 for client_id in selected
    ) * 10
    if (
        not isinstance(benign_records, int)
        or isinstance(benign_records, bool)
        or not isinstance(malicious_records, int)
        or isinstance(malicious_records, bool)
        or benign_records + malicious_records != 30
        or benign_records != expected_benign_records
        or malicious_records != 30 - expected_benign_records
    ):
        raise ValueError("run round role record counts are invalid")
    _require_sha256(value.get("index_sha256"), "round index_sha256")
    if value.get("delta_base_round") != expected_index or isinstance(
        value.get("delta_base_round"), bool
    ):
        raise ValueError("round delta_base_round must match round_index")
    base_sha = _require_sha256(
        value.get("delta_base_tensor_sha256"),
        "round delta_base_tensor_sha256",
    )
    r0_sha = _require_sha256(
        value.get("collection_r0_tensor_sha256"),
        "round collection_r0_tensor_sha256",
    )
    if expected_index == 0 and base_sha != r0_sha:
        raise ValueError("round 0 base SHA must equal collection R0 SHA")
    expected_client_keys = {str(client_id) for client_id in selected}
    local_shas = value.get("local_state_tensor_sha256")
    if not isinstance(local_shas, dict) or set(local_shas) != expected_client_keys:
        raise ValueError("round local-state SHA client keys are invalid")
    for client_id, digest in local_shas.items():
        _require_sha256(digest, "local state SHA for client " + client_id)
    sample_counts = value.get("sample_counts")
    if not isinstance(sample_counts, dict) or set(sample_counts) != expected_client_keys:
        raise ValueError("round sample-count client keys are invalid")
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count != 160
        for count in sample_counts.values()
    ):
        raise ValueError("round sample counts must all equal 160")
    _require_sha256(
        value.get("fedavg_candidate_tensor_sha256"),
        "round FedAvg candidate SHA",
    )
    if value.get("lr_policy") != "task1_main_100_round_cosine_first_10":
        raise ValueError("round LR policy is invalid")
    if value.get("learning_rate") != _PROTOCOL_SUMMARY["learning_rates"][
        expected_index
    ]:
        raise ValueError("round learning rate does not match frozen schedule")
    return copy.deepcopy(value)


def validate_probe_collection_run_summary(
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Strictly validate a deterministic full or smoke run summary."""
    if not isinstance(summary, dict) or set(summary) != _SUMMARY_KEYS:
        raise ValueError("Probe run summary fields do not match schema")
    if not matches_schema(summary.get("schema_version"), RUN_SCHEMA_VERSION):
        raise ValueError("unsupported Probe run summary schema")
    mode = summary.get("mode")
    if mode not in ("full", "smoke-round-0"):
        raise ValueError("run summary mode is invalid")
    _require_sha256(summary.get("run_sha256"), "run_sha256")
    _require_sha256(summary.get("plan_sha256"), "plan_sha256")
    _require_sha256(summary.get("manifest_sha256"), "manifest_sha256")
    _require_commit(summary.get("code_commit"), "code_commit")
    _require_sha256(
        summary.get("model_artifact_sha256"),
        "model_artifact_sha256",
    )
    _require_sha256(
        summary.get("tokenizer_artifact_sha256"),
        "tokenizer_artifact_sha256",
    )
    collection_r0_sha = _require_sha256(
        summary.get("collection_r0_tensor_sha256"),
        "collection_r0_tensor_sha256",
    )
    bitsandbytes_version = summary.get("bitsandbytes_version")
    if not isinstance(bitsandbytes_version, str) or not bitsandbytes_version:
        raise ValueError("bitsandbytes_version must be non-empty")
    runtime_report = _validate_runtime_report(summary.get("runtime_report"))
    if runtime_report["bitsandbytes_version"] != bitsandbytes_version:
        raise ValueError(
            "runtime_report bitsandbytes version does not match summary"
        )
    if _canonical_json_bytes(summary.get("protocol_summary")) != (
        _canonical_json_bytes(_PROTOCOL_SUMMARY)
    ):
        raise ValueError("run protocol_summary does not match frozen protocol")
    _validate_lora_spec(summary.get("lora_spec"))

    rounds = summary.get("rounds")
    expected_round_count = 10 if mode == "full" else 1
    if not isinstance(rounds, list) or len(rounds) != expected_round_count:
        raise ValueError(
            "full run requires 10 rounds; smoke run requires round 0 only"
        )
    normalized_rounds = [
        _validate_round_identity(value, index)
        for index, value in enumerate(rounds)
    ]
    if any(
        value["collection_r0_tensor_sha256"] != collection_r0_sha
        for value in normalized_rounds
    ):
        raise ValueError("round collection R0 SHA does not match run summary")
    for index in range(1, len(normalized_rounds)):
        if normalized_rounds[index]["delta_base_tensor_sha256"] != (
            normalized_rounds[index - 1]["fedavg_candidate_tensor_sha256"]
        ):
            raise ValueError("Probe collection round SHA chain is broken")
    run_counts = summary.get("summary")
    if not isinstance(run_counts, dict) or set(run_counts) != {
        "rounds",
        "records",
        "sample_events",
        "benign_records",
        "malicious_records",
    }:
        raise ValueError("run summary counts do not match schema")
    expected_counts = {
        "rounds": expected_round_count,
        "records": 30 * expected_round_count,
        "sample_events": 480 * expected_round_count,
        "benign_records": sum(
            item["benign_records"] for item in normalized_rounds
        ),
        "malicious_records": sum(
            item["malicious_records"] for item in normalized_rounds
        ),
    }
    if run_counts != expected_counts:
        raise ValueError("run summary counts are inconsistent with rounds")
    if mode == "full" and expected_counts != {
        "rounds": 10,
        "records": 300,
        "sample_events": 4800,
        "benign_records": 160,
        "malicious_records": 140,
    }:
        raise ValueError("full run summary counts do not match paper plan")
    if _run_digest(summary) != summary["run_sha256"]:
        raise ValueError("run_sha256 does not match canonical summary payload")
    return copy.deepcopy(summary)


def build_probe_collection_run_summary(
    *,
    mode: str,
    plan: Mapping,
    bitsandbytes_version: str,
    runtime_report: Mapping,
    lora_spec: Mapping,
    rounds: Sequence[Mapping],
    collection_r0_tensor_sha256: str,
) -> Dict[str, Any]:
    """Build and self-validate the deterministic run-level artifact."""
    if mode not in ("full", "smoke-round-0"):
        raise ValueError("summary mode must be full or smoke-round-0")
    if not isinstance(plan, Mapping):
        raise TypeError("plan must be a Mapping")
    round_values = [copy.deepcopy(value) for value in rounds]
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_sha256": "",
        "mode": mode,
        "plan_sha256": plan["plan_sha256"],
        "manifest_sha256": plan["manifest"]["manifest_sha256"],
        "code_commit": plan["code_commit"],
        "model_artifact_sha256": plan["model_artifact"][
            "artifact_sha256"
        ],
        "tokenizer_artifact_sha256": plan["tokenizer_artifact"][
            "artifact_sha256"
        ],
        "collection_r0_tensor_sha256": _require_sha256(
            collection_r0_tensor_sha256,
            "collection_r0_tensor_sha256",
        ),
        "bitsandbytes_version": bitsandbytes_version,
        "runtime_report": copy.deepcopy(runtime_report),
        "protocol_summary": copy.deepcopy(_PROTOCOL_SUMMARY),
        "lora_spec": copy.deepcopy(lora_spec),
        "rounds": round_values,
        "summary": {
            "rounds": len(round_values),
            "records": sum(value["record_count"] for value in round_values),
            "sample_events": sum(
                value["sample_events"] for value in round_values
            ),
            "benign_records": sum(
                value["benign_records"] for value in round_values
            ),
            "malicious_records": sum(
                value["malicious_records"] for value in round_values
            ),
        },
    }
    summary["run_sha256"] = _run_digest(summary)
    return validate_probe_collection_run_summary(summary)


def _summary_names(mode: str) -> Tuple[str, str]:
    if mode == "full":
        return FULL_SUMMARY_NAME, FULL_MARKER_NAME
    if mode == "smoke-round-0":
        return SMOKE_SUMMARY_NAME, SMOKE_MARKER_NAME
    raise ValueError("summary mode must be full or smoke-round-0")


def _write_synced(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_RDONLY", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(path), flags | directory_flag)
    except (OSError, TypeError):
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _unlink_if_same_file(final_path: Path, staging_path: Path) -> None:
    try:
        final_stat = os.stat(final_path, follow_symlinks=False)
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


def _read_digest_line(
    path: Path,
    name: str,
    *,
    expected_filename: Optional[str] = None,
) -> str:
    _require_regular_nonsymlink(path, name)
    raw = path.read_bytes()
    simple = re.fullmatch(rb"([0-9a-f]{64})\n", raw)
    if simple is not None:
        return simple.group(1).decode("ascii")
    sha256sum = re.fullmatch(rb"([0-9a-f]{64})  ([^\r\n]+)\n", raw)
    if sha256sum is not None and expected_filename is not None:
        try:
            listed_filename = sha256sum.group(2).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(name + " filename must be ASCII") from exc
        if listed_filename != expected_filename:
            raise ValueError(name + " filename does not match input JSON")
        return sha256sum.group(1).decode("ascii")
    raise ValueError(
        name
        + " must contain a lowercase SHA-256 line or canonical sha256sum line"
    )


def _frozen_stat(path: Path, name: str) -> Dict[str, int]:
    _require_regular_nonsymlink(path, name)
    value = path.lstat()
    return {
        "st_dev": value.st_dev,
        "st_ino": value.st_ino,
        "st_size": value.st_size,
        "st_mtime_ns": value.st_mtime_ns,
    }


def _capture_input_group(
    path: Path,
    expected_sha256: str,
    name: str,
) -> Dict[str, Any]:
    sidecar_path = Path(str(path) + ".sha256")
    expected = _require_sha256(expected_sha256, name + " SHA-256")
    if _read_digest_line(
        sidecar_path,
        name + " sidecar",
        expected_filename=path.name,
    ) != expected:
        raise ValueError(name + " sidecar does not match expected SHA-256")
    return {
        "json_path": path.absolute(),
        "json_stat": _frozen_stat(path, name + " JSON"),
        "sidecar_path": sidecar_path.absolute(),
        "sidecar_stat": _frozen_stat(sidecar_path, name + " sidecar"),
    }


def capture_probe_collection_input_identity(
    plan_path: Any,
    manifest_path: Any,
    *,
    expected_plan_sha256: str,
    expected_manifest_sha256: str,
) -> Dict[str, Any]:
    """Freeze file identity for both canonical inputs and their sidecars."""
    if not isinstance(plan_path, (str, os.PathLike)):
        raise TypeError("plan_path must be a path")
    if not isinstance(manifest_path, (str, os.PathLike)):
        raise TypeError("manifest_path must be a path")
    return {
        "plan": _capture_input_group(
            Path(plan_path),
            expected_plan_sha256,
            "Probe collection plan",
        ),
        "manifest": _capture_input_group(
            Path(manifest_path),
            expected_manifest_sha256,
            "Probe collection Manifest",
        ),
    }


def _group_identity_matches(
    frozen_group: Mapping,
    current_group: Mapping,
) -> bool:
    return (
        frozen_group.get("json_path") == current_group.get("json_path")
        and frozen_group.get("sidecar_path")
        == current_group.get("sidecar_path")
        and frozen_group.get("json_stat") == current_group.get("json_stat")
        and frozen_group.get("sidecar_stat")
        == current_group.get("sidecar_stat")
    )


def reload_and_validate_frozen_inputs(
    *,
    frozen_identity: Mapping,
    original_plan: Mapping,
    original_manifest: Mapping,
    expected_plan_sha256: str,
    expected_manifest_sha256: str,
    expected_code_commit: str,
    plan_loader: Callable,
    manifest_loader: Callable,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Strictly reload both inputs and prove content plus inode stability."""
    if not isinstance(frozen_identity, Mapping) or set(
        frozen_identity
    ) != {"plan", "manifest"}:
        raise ValueError("frozen input identity fields are invalid")
    if not isinstance(original_plan, Mapping):
        raise TypeError("original_plan must be a Mapping")
    if not isinstance(original_manifest, Mapping):
        raise TypeError("original_manifest must be a Mapping")
    if not callable(plan_loader) or not callable(manifest_loader):
        raise TypeError("frozen input loaders must be callable")
    plan_group = frozen_identity["plan"]
    manifest_group = frozen_identity["manifest"]
    if not isinstance(plan_group, Mapping) or not isinstance(
        manifest_group, Mapping
    ):
        raise ValueError("frozen input group is invalid")

    plan_digest = _require_sha256(
        expected_plan_sha256,
        "expected_plan_sha256",
    )
    manifest_digest = _require_sha256(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    )
    commit = _require_commit(expected_code_commit, "expected_code_commit")
    try:
        reloaded_plan = plan_loader(
            plan_group["json_path"],
            expected_plan_sha256=plan_digest,
            expected_manifest_sha256=manifest_digest,
            expected_code_commit=commit,
        )
        current_plan_group = _capture_input_group(
            Path(plan_group["json_path"]),
            plan_digest,
            "Probe collection plan",
        )
        if (
            _canonical_json_bytes(reloaded_plan)
            != _canonical_json_bytes(original_plan)
            or not _group_identity_matches(
                plan_group,
                current_plan_group,
            )
        ):
            raise ValueError("plan content or file identity changed")
    except Exception as exc:
        raise RuntimeError(
            "Probe collection plan changed during the run"
        ) from exc

    try:
        reloaded_manifest = manifest_loader(
            str(manifest_group["json_path"]),
            manifest_digest,
        )
        current_manifest_group = _capture_input_group(
            Path(manifest_group["json_path"]),
            manifest_digest,
            "Probe collection Manifest",
        )
        if (
            _canonical_json_bytes(reloaded_manifest)
            != _canonical_json_bytes(original_manifest)
            or not _group_identity_matches(
                manifest_group,
                current_manifest_group,
            )
        ):
            raise ValueError("Manifest content or file identity changed")
    except Exception as exc:
        raise RuntimeError(
            "Probe collection Manifest changed during the run"
        ) from exc
    return copy.deepcopy(reloaded_plan), copy.deepcopy(reloaded_manifest)


def load_and_validate_probe_collection_run_summary(
    path: Any,
    *,
    expected_run_sha256: Optional[str] = None,
    expected_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Strictly load a canonical run summary and its digest sidecar."""
    summary_path = Path(path)
    _require_regular_nonsymlink(summary_path, "Probe run summary")
    raw = summary_path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError("non-standard JSON constant: " + item)
            ),
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Probe run summary is not strict JSON") from exc
    validated = validate_probe_collection_run_summary(value)
    if raw != _canonical_json_bytes(validated):
        raise ValueError("Probe run summary JSON is not canonical")
    sidecar_digest = _read_digest_line(
        Path(str(summary_path) + ".sha256"),
        "Probe run summary sidecar",
    )
    if sidecar_digest != validated["run_sha256"]:
        raise ValueError("run summary sidecar does not match run_sha256")
    if expected_run_sha256 is not None and (
        _require_sha256(expected_run_sha256, "expected_run_sha256")
        != validated["run_sha256"]
    ):
        raise ValueError("run summary SHA does not match expected")
    if expected_mode is not None and validated["mode"] != expected_mode:
        raise ValueError("run summary mode does not match expected")
    return validated


def write_probe_collection_run_summary(
    summary: Dict[str, Any],
    output_root: Any,
    *,
    before_marker: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Stage and publish summary, sidecar, then completion marker no-clobber."""
    validated = validate_probe_collection_run_summary(summary)
    if before_marker is not None and not callable(before_marker):
        raise TypeError("before_marker must be callable")
    root = _require_existing_directory(output_root, "output_root")
    summary_name, marker_name = _summary_names(validated["mode"])
    final_summary = root / summary_name
    final_sidecar = root / (summary_name + ".sha256")
    final_marker = root / marker_name
    targets = (final_summary, final_sidecar, final_marker)
    if any(os.path.lexists(str(path)) for path in targets):
        raise FileExistsError("Probe run publication target already exists")

    summary_bytes = _canonical_json_bytes(validated)
    digest_line = (validated["run_sha256"] + "\n").encode("ascii")
    created = []
    with tempfile.TemporaryDirectory(
        dir=str(root),
        prefix=".probe-collection-run.",
    ) as staging_name:
        staging_root = Path(staging_name)
        staging_summary = staging_root / summary_name
        staging_sidecar = staging_root / (summary_name + ".sha256")
        staging_marker = staging_root / marker_name
        _write_synced(staging_summary, summary_bytes)
        _write_synced(staging_sidecar, digest_line)
        _write_synced(staging_marker, digest_line)
        load_and_validate_probe_collection_run_summary(
            staging_summary,
            expected_run_sha256=validated["run_sha256"],
            expected_mode=validated["mode"],
        )
        try:
            for source, target in (
                (staging_summary, final_summary),
                (staging_sidecar, final_sidecar),
            ):
                os.link(str(source), str(target))
                created.append((target, source))
            _fsync_directory(root)
            loaded = load_and_validate_probe_collection_run_summary(
                final_summary,
                expected_run_sha256=validated["run_sha256"],
                expected_mode=validated["mode"],
            )
            if before_marker is not None:
                before_marker()
            os.link(str(staging_marker), str(final_marker))
            created.append((final_marker, staging_marker))
            _fsync_directory(root)
            marker_digest = _read_digest_line(
                final_marker,
                "Probe run completion marker",
            )
            if marker_digest != loaded["run_sha256"]:
                raise ValueError(
                    "Probe run completion marker does not match run SHA"
                )
            final_marker_stat = os.stat(
                final_marker,
                follow_symlinks=False,
            )
            staging_marker_stat = os.stat(
                staging_marker,
                follow_symlinks=False,
            )
            if (
                final_marker_stat.st_dev,
                final_marker_stat.st_ino,
            ) != (
                staging_marker_stat.st_dev,
                staging_marker_stat.st_ino,
            ) and final_marker.read_bytes() != staging_marker.read_bytes():
                raise ValueError(
                    "Probe run completion marker does not match staging"
                )
        except Exception:
            for final_path, staging_path in reversed(created):
                _unlink_if_same_file(final_path, staging_path)
            raise
    return {
        "run_sha256": validated["run_sha256"],
        "summary_path": final_summary,
        "sidecar_path": final_sidecar,
        "completion_path": final_marker,
    }


def _round_identity(
    loaded: Mapping,
    plan: Mapping,
    round_index: int,
    round_provenance: Mapping,
) -> Dict[str, Any]:
    index = loaded.get("index")
    if not isinstance(index, dict):
        raise ValueError("validated round reader result must contain index")
    if index.get("round_index") != round_index:
        raise ValueError("validated round index does not match requested round")
    if index.get("record_count") != 30:
        raise ValueError("validated round index must contain 30 records")
    round_plan = plan["rounds"][round_index]
    selected = list(round_plan["selected_client_ids"])
    benign_records = sum(client_id < 5 for client_id in selected) * 10
    required_provenance = {
        "collection_round",
        "delta_base_round",
        "delta_base_tensor_sha256",
        "collection_r0_tensor_sha256",
        "selected_client_ids",
        "sample_counts",
        "local_state_tensor_sha256",
        "fedavg_candidate_tensor_sha256",
        "lr_policy",
        "learning_rate",
    }
    if not isinstance(round_provenance, Mapping) or set(round_provenance) != (
        required_provenance
    ):
        raise ValueError("round provenance fields do not match formal schema")
    if round_provenance["collection_round"] != round_index:
        raise ValueError("round provenance collection_round mismatch")
    if round_provenance["selected_client_ids"] != selected:
        raise ValueError("round provenance selected clients mismatch")
    return {
        "round_index": round_index,
        "round_directory_name": round_plan["output_round_name"],
        "selected_client_ids": selected,
        "record_count": 30,
        "sample_events": 480,
        "benign_records": benign_records,
        "malicious_records": 30 - benign_records,
        "index_sha256": _sha256_bytes(_canonical_json_bytes(index)),
        "delta_base_round": round_provenance["delta_base_round"],
        "delta_base_tensor_sha256": round_provenance[
            "delta_base_tensor_sha256"
        ],
        "collection_r0_tensor_sha256": round_provenance[
            "collection_r0_tensor_sha256"
        ],
        "local_state_tensor_sha256": copy.deepcopy(
            round_provenance["local_state_tensor_sha256"]
        ),
        "sample_counts": copy.deepcopy(round_provenance["sample_counts"]),
        "fedavg_candidate_tensor_sha256": round_provenance[
            "fedavg_candidate_tensor_sha256"
        ],
        "lr_policy": round_provenance["lr_policy"],
        "learning_rate": round_provenance["learning_rate"],
    }


def _best_effort_runtime_cleanup(
    runtime_objects: Dict[str, Any],
    get_torch: Callable[[], Any],
) -> None:
    for key in list(runtime_objects):
        runtime_objects[key] = None
    try:
        gc.collect()
    except Exception:
        pass
    try:
        torch_module = get_torch()
        cuda = getattr(torch_module, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception:
        pass


def run_probe_collection(
    *,
    mode: str,
    plan_path: Any,
    plan_sha256: str,
    manifest_path: Any,
    manifest_sha256: str,
    model_directory: Any,
    expected_code_commit: str,
    expected_bitsandbytes_version: str,
    output_root: Any,
    scratch_root: Any,
    local_data_dir: Optional[str],
    repository_root: Any,
    expected_r0_tensor_sha256: str,
    dependency_provider: Optional[Callable[[], Mapping]] = None,
    git_provider: Optional[Callable[[Path, str], Mapping]] = None,
) -> Dict[str, Any]:
    """Execute preflight, smoke round 0, or the complete ten-round run."""
    runtime_module = importlib.import_module(
        "lora_classifier.probe_collection_runtime"
    )
    runtime_module.configure_probe_offline_environment()

    mode = _require_mode(mode)
    plan_digest = _require_sha256(plan_sha256, "plan_sha256")
    manifest_digest = _require_sha256(
        manifest_sha256,
        "manifest_sha256",
    )
    expected_commit = _require_commit(
        expected_code_commit,
        "expected_code_commit",
    )
    expected_r0 = _require_sha256(
        expected_r0_tensor_sha256,
        "expected_r0_tensor_sha256",
    )
    if (
        not isinstance(expected_bitsandbytes_version, str)
        or not expected_bitsandbytes_version.strip()
    ):
        raise ValueError(
            "expected_bitsandbytes_version must be a non-empty string"
        )

    dependencies = _dependencies(dependency_provider)

    git_state = validate_collection_git_state(
        repository_root,
        expected_commit,
        git_provider=git_provider,
        default_git_provider=dependencies["collect_git"],
    )
    output = _require_existing_directory(output_root, "output_root")
    expected_plan_path = output / PLAN_FILE_NAME
    if Path(plan_path).resolve() != expected_plan_path.resolve():
        raise ValueError(
            "plan_path must equal <output_root>/probe-collection-plan.json"
        )
    plan = dependencies["load_plan"](
        expected_plan_path,
        expected_plan_sha256=plan_digest,
        expected_manifest_sha256=manifest_digest,
        expected_code_commit=expected_commit,
    )
    manifest = dependencies["load_manifest"](
        str(manifest_path),
        manifest_digest,
    )
    if manifest is None:
        raise ValueError("Probe collection requires a data Manifest")
    if plan["manifest"]["manifest_sha256"] != manifest_digest:
        raise ValueError("plan Manifest SHA does not match CLI")
    # Preserve and validate the preprocessing commit independently of execution.
    _require_commit(manifest.get("code_commit"), "manifest preprocessing commit")
    frozen_input_identity = capture_probe_collection_input_identity(
        expected_plan_path,
        manifest_path,
        expected_plan_sha256=plan_digest,
        expected_manifest_sha256=manifest_digest,
    )
    frozen_plan = copy.deepcopy(plan)
    frozen_manifest = copy.deepcopy(manifest)

    def reload_frozen_inputs() -> None:
        nonlocal plan, manifest
        plan, manifest = reload_and_validate_frozen_inputs(
            frozen_identity=frozen_input_identity,
            original_plan=frozen_plan,
            original_manifest=frozen_manifest,
            expected_plan_sha256=plan_digest,
            expected_manifest_sha256=manifest_digest,
            expected_code_commit=expected_commit,
            plan_loader=dependencies["load_plan"],
            manifest_loader=dependencies["load_manifest"],
        )

    dependencies["validate_manifest_provenance"](
        manifest,
        Path(repository_root).resolve(),
    )
    dependencies["validate_probe_manifest"](manifest)
    probe_pools, client_counts = dependencies["rebuild_probe_pools"](
        manifest,
        local_data_dir,
    )
    if client_counts != [5, 5]:
        raise ValueError("Probe pool client counts must be [5, 5]")
    client_datasets = dependencies["bind_probe_pools"](plan, probe_pools)
    if not isinstance(client_datasets, Mapping) or set(
        client_datasets
    ) != set(range(10)):
        raise ValueError("Probe pool binding must produce clients 0..9")

    output = validate_collection_output_space(
        output,
        expected_plan_path,
        validate_round_space=dependencies["validate_round_space"],
    )
    scratch = validate_collection_scratch_space(scratch_root)
    runtime_report = dependencies["validate_gpu"](
        plan,
        expected_bitsandbytes_version=expected_bitsandbytes_version,
        run_optimizer_smoke=True,
    )
    artifact_report = dependencies["validate_artifacts"](
        plan,
        model_directory,
    )
    if artifact_report["model_identity"]["artifact_sha256"] != plan[
        "model_artifact"
    ]["artifact_sha256"]:
        raise ValueError("validated model artifact does not match plan")
    if artifact_report["tokenizer_identity"]["artifact_sha256"] != plan[
        "tokenizer_artifact"
    ]["artifact_sha256"]:
        raise ValueError("validated tokenizer artifact does not match plan")

    base_report = {
        "mode": mode,
        "code_commit": git_state["commit"],
        "plan_sha256": plan["plan_sha256"],
        "manifest_sha256": plan["manifest"]["manifest_sha256"],
        "model_artifact_sha256": plan["model_artifact"]["artifact_sha256"],
        "tokenizer_artifact_sha256": plan["tokenizer_artifact"][
            "artifact_sha256"
        ],
        "bitsandbytes_version": expected_bitsandbytes_version,
        "gpu_name": runtime_report["cuda_device_name"],
        "clients": 10,
        "rounds": 10,
        "output_readiness": "ready",
        "collection_r0_tensor_sha256": expected_r0,
    }
    if mode == "preflight-only":
        reload_frozen_inputs()
        return base_report

    reload_frozen_inputs()
    lock_identity = None
    primary_error = None
    runtime_objects = {
        "base_model": None,
        "model": None,
        "tokenizer": None,
        "quantization_config": None,
        "initial_global_state": None,
        "global_state": None,
    }
    loaded = None
    initialized = None
    lora_spec = None
    formatter = None
    response_template = None
    candidate = None
    try:
        lock_identity = create_probe_collection_lock(
            output,
            code_commit=git_state["commit"],
            plan_sha256=plan["plan_sha256"],
            mode=mode,
        )
        loaded = dependencies["load_model"](plan, model_directory)
        runtime_objects["base_model"] = loaded["model"]
        runtime_objects["tokenizer"] = loaded["tokenizer"]
        runtime_objects["quantization_config"] = loaded[
            "quantization_config"
        ]
        initialized = dependencies["initialize_lora"](
            loaded["model"],
            seed=2023,
        )
        state_sha = dependencies["state_sha"](
            initialized["initial_global_state"]
        )
        if state_sha != expected_r0:
            raise ValueError(
                "Probe initialization does not match canonical R0 tensor SHA"
            )
        runtime_objects["model"] = initialized["model"]
        runtime_objects["initial_global_state"] = dependencies["clone_state"](
            initialized["initial_global_state"]
        )
        runtime_objects["global_state"] = dependencies["clone_state"](
            initialized["initial_global_state"]
        )
        lora_spec = copy.deepcopy(initialized["lora_spec"])
        formatter, response_template = dependencies["get_formatter"](
            "alpaca",
            loaded["tokenizer"].eos_token,
        )

        def train_client(**kwargs):
            return dependencies["train_client"](
                model=runtime_objects["model"],
                tokenizer=runtime_objects["tokenizer"],
                plan=plan,
                round_index=kwargs["round_index"],
                client_id=kwargs["client_id"],
                client_dataset=kwargs["client_dataset"],
                initial_global_state=kwargs["initial_global_state"],
                lora_spec=lora_spec,
                scratch_root=scratch,
                formatting_prompts_func=formatter,
                response_template=response_template,
                collection_r0_tensor_sha256=kwargs[
                    "collection_r0_tensor_sha256"
                ],
                delta_base_tensor_sha256=kwargs[
                    "delta_base_tensor_sha256"
                ],
            )

        captured_rounds = {}
        captured_loaded_rounds = {}

        def writer(root, round_index, records):
            return dependencies["write_round"](root, round_index, records)

        def reader(round_directory, **kwargs):
            loaded_round = dependencies["read_round"](
                round_directory,
                **kwargs,
            )
            round_index = kwargs["expected_round_index"]
            captured_loaded_rounds[round_index] = loaded_round
            return loaded_round

        round_indexes = range(10) if mode == "full" else range(1)
        ordered_rounds = []
        for expected_round_index in round_indexes:
            round_provenance = {}

            def capture_round_provenance(value):
                if round_provenance:
                    raise RuntimeError("round provenance sink called more than once")
                round_provenance.update(copy.deepcopy(value))

            candidate = dependencies["execute_round"](
                plan=plan,
                round_index=expected_round_index,
                client_datasets=client_datasets,
                global_state=runtime_objects["global_state"],
                expected_lora_spec=lora_spec,
                output_root=output,
                train_client=train_client,
                shard_writer=writer,
                shard_reader=reader,
                collection_r0_tensor_sha256=expected_r0,
                round_provenance_sink=capture_round_provenance,
            )
            if expected_round_index not in captured_loaded_rounds:
                raise RuntimeError(
                    "round reader did not capture a validated round identity"
                )
            if not round_provenance:
                raise RuntimeError("round runner did not publish provenance")
            candidate_sha = dependencies["state_sha"](candidate)
            if candidate_sha != round_provenance.get(
                "fedavg_candidate_tensor_sha256"
            ):
                raise ValueError("round candidate SHA does not match provenance")
            captured_rounds[expected_round_index] = _round_identity(
                captured_loaded_rounds[expected_round_index],
                plan,
                expected_round_index,
                round_provenance,
            )
            runtime_objects["global_state"] = dependencies["clone_state"](
                candidate
            )
            ordered_rounds.append(captured_rounds[expected_round_index])

        reload_frozen_inputs()
        summary_mode = "full" if mode == "full" else "smoke-round-0"
        summary = build_probe_collection_run_summary(
            mode=summary_mode,
            plan=plan,
            bitsandbytes_version=expected_bitsandbytes_version,
            runtime_report=runtime_report,
            lora_spec=lora_spec,
            rounds=ordered_rounds,
            collection_r0_tensor_sha256=expected_r0,
        )
        publication = write_probe_collection_run_summary(
            summary,
            output,
            before_marker=reload_frozen_inputs,
        )
        result = dict(base_report)
        result.update(
            {
                "completed_rounds": len(ordered_rounds),
                "records": summary["summary"]["records"],
                "sample_events": summary["summary"]["sample_events"],
                "benign_records": summary["summary"]["benign_records"],
                "malicious_records": summary["summary"][
                    "malicious_records"
                ],
                "run_sha256": publication["run_sha256"],
                "summary_path": str(publication["summary_path"]),
                "completion_path": str(publication["completion_path"]),
            }
        )
        return result
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        loaded = None
        initialized = None
        lora_spec = None
        formatter = None
        response_template = None
        candidate = None
        _best_effort_runtime_cleanup(
            runtime_objects,
            dependencies["get_torch"],
        )
        if lock_identity is not None:
            try:
                remove_probe_collection_lock(lock_identity)
            except Exception:
                if primary_error is None:
                    raise

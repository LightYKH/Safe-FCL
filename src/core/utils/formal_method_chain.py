"""Build a canonical, independently verified R0->R400 formal method chain."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file

from .data_manifest import canonical_json_bytes, sha256_bytes, sha256_file
from .formal_checkpoint_provenance import build_checkpoint_provenance
from .lora_state_guard import canonical_lora_tensor_sha256
from .probe_evaluator_contract import FORMAL_METHOD_IDS, METHODS
from .probe_evaluator_jobs import (
    CHAIN_MARKER_SCHEMA_VERSION,
    CHAIN_SCHEMA_VERSION,
    CHECKPOINT_ROUNDS,
    validate_formal_method_chain,
)


def _file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(name + " must be a non-empty non-symlink file")
    return path


def _directory(value: Any, name: str) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise ValueError(name + " must not be a symlink")
    path = unresolved.resolve()
    if not path.is_dir():
        raise FileNotFoundError(name + " directory is missing")
    return path


def _sha(value: Any, name: str, length: int = 64) -> str:
    text = str(value or "")
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(name + " must be lowercase hexadecimal")
    return text


def adapter_tensor_sha256(path: Any) -> str:
    target = _file(Path(path).expanduser().resolve(), "adapter model")
    state = load_file(str(target), device="cpu")
    if not state or any("lora_" not in key for key in state):
        raise ValueError("adapter model must contain only non-empty LoRA tensors")
    return canonical_lora_tensor_sha256(state)


def build_formal_method_chain(
    *,
    method_id: str,
    code_commit: str,
    configuration_freeze_sha256: str,
    environment_lock_sha256: str,
    fixed_task1_probe_sha256: Any,
    r0_directory: Any,
    stage_output_directories: Sequence[Any],
    task_manifest_sha256: Sequence[str],
    task_schedule_sha256: Sequence[str],
) -> Mapping[str, Any]:
    if method_id not in METHODS:
        raise ValueError("unsupported evaluator method_id")
    formal_method = FORMAL_METHOD_IDS[method_id]
    commit = _sha(code_commit, "code_commit", 40)
    configuration_sha = _sha(
        configuration_freeze_sha256, "configuration_freeze_sha256"
    )
    environment_sha = _sha(environment_lock_sha256, "environment_lock_sha256")
    expected_probe = None
    if method_id == "pure_fedavg":
        if fixed_task1_probe_sha256 not in (None, ""):
            raise ValueError("Pure method chain must not bind a training Probe")
    else:
        expected_probe = _sha(fixed_task1_probe_sha256, "fixed_task1_probe_sha256")
    for name, values in (
        ("stage_output_directories", stage_output_directories),
        ("task_manifest_sha256", task_manifest_sha256),
        ("task_schedule_sha256", task_schedule_sha256),
    ):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 4:
            raise ValueError(name + " must contain exactly four ordered values")
    manifest_shas = [_sha(value, "task manifest SHA") for value in task_manifest_sha256]
    schedule_shas = [_sha(value, "task schedule SHA") for value in task_schedule_sha256]
    stages = [_directory(value, "Task%d output" % task_id) for task_id, value in enumerate(stage_output_directories, 1)]

    r0 = _directory(r0_directory, "R0")
    r0_config = _file(r0 / "adapter_config.json", "R0 adapter config")
    r0_adapter = _file(r0 / "adapter_model.safetensors", "R0 adapter model")
    r0_provenance = _file(r0 / "R0_IDENTITY.json", "R0 identity")
    r0_value = json.loads(r0_provenance.read_text(encoding="utf-8"))
    r0_config_sha = sha256_file(r0_config)
    r0_adapter_sha = sha256_file(r0_adapter)
    r0_provenance_sha = sha256_file(r0_provenance)
    r0_tensor_sha = adapter_tensor_sha256(r0_adapter)
    expected_r0 = {
        "adapter_config_sha256": r0_config_sha,
        "adapter_model_sha256": r0_adapter_sha,
        "code_commit": commit,
        "environment_lock_sha256": environment_sha,
        "lora_tensor_sha256": r0_tensor_sha,
    }
    for field, expected in expected_r0.items():
        if r0_value.get(field) != expected:
            raise ValueError("R0 identity mismatch: " + field)
    python_executable = r0_value.get("python_executable")
    if not isinstance(python_executable, str) or not Path(python_executable).is_absolute():
        raise ValueError("R0 identity Python executable is invalid")

    checkpoints = [{
        "adapter_config_sha256": r0_config_sha,
        "adapter_model_sha256": r0_adapter_sha,
        "checkpoint_directory": str(r0),
        "checkpoint_provenance_sha256": r0_provenance_sha,
        "endpoint_lora_tensor_sha256": r0_tensor_sha,
        "global_round": 0,
        "method_id": method_id,
        "task_id": 0,
    }]
    stage_rows = []
    stage_base_tensor_sha = r0_tensor_sha
    parent_provenance_sha = r0_provenance_sha
    for task_id, stage in enumerate(stages, start=1):
        stage_rows.append({
            "data_manifest_sha256": manifest_shas[task_id - 1],
            "end_global_round": task_id * 100,
            "parent_provenance_sha256": parent_provenance_sha,
            "schedule_sha256": schedule_shas[task_id - 1],
            "stage_base_tensor_sha256": stage_base_tensor_sha,
            "start_global_round": (task_id - 1) * 100,
            "task_id": task_id,
        })
        for round_number in range((task_id - 1) * 100 + 10, task_id * 100 + 1, 10):
            checkpoint = _directory(
                stage / ("checkpoint-%d" % round_number),
                "R%d checkpoint" % round_number,
            )
            config = _file(checkpoint / "adapter_config.json", "checkpoint adapter config")
            adapter = _file(checkpoint / "adapter_model.safetensors", "checkpoint adapter model")
            provenance = _file(
                checkpoint / "FORMAL_CHECKPOINT_PROVENANCE.json",
                "checkpoint provenance",
            )
            raw = provenance.read_bytes()
            value = json.loads(raw.decode("utf-8"))
            if raw != canonical_json_bytes(value):
                raise ValueError("formal checkpoint provenance must be canonical JSON")
            config_sha = sha256_file(config)
            adapter_sha = sha256_file(adapter)
            tensor_sha = adapter_tensor_sha256(adapter)
            expected = build_checkpoint_provenance(
                code_commit=commit,
                task_id=task_id,
                formal_method=formal_method,
                global_round=round_number,
                adapter_config_sha256=config_sha,
                adapter_model_sha256=adapter_sha,
                endpoint_lora_tensor_sha256=tensor_sha,
                stage_base_tensor_sha256=stage_base_tensor_sha,
                parent_provenance_sha256=parent_provenance_sha,
                data_manifest_sha256=manifest_shas[task_id - 1],
                schedule_sha256=schedule_shas[task_id - 1],
                fixed_probe_sha256=expected_probe,
                environment_lock_sha256=environment_sha,
                configuration_freeze_sha256=configuration_sha,
                python_executable=python_executable,
            )
            if value != expected:
                raise ValueError("R%d checkpoint provenance content mismatch" % round_number)
            checkpoints.append({
                "adapter_config_sha256": config_sha,
                "adapter_model_sha256": adapter_sha,
                "checkpoint_directory": str(checkpoint),
                "checkpoint_provenance_sha256": sha256_file(provenance),
                "endpoint_lora_tensor_sha256": tensor_sha,
                "global_round": round_number,
                "method_id": method_id,
                "task_id": task_id,
            })
        endpoint = checkpoints[-1]
        stage_base_tensor_sha = endpoint["endpoint_lora_tensor_sha256"]
        parent_provenance_sha = endpoint["checkpoint_provenance_sha256"]

    if [row["global_round"] for row in checkpoints] != list(CHECKPOINT_ROUNDS):
        raise RuntimeError("formal method chain did not resolve R0 and R10..R400")
    chain = {
        "checkpoints": checkpoints,
        "code_commit": commit,
        "configuration_freeze_sha256": configuration_sha,
        "environment_lock_sha256": environment_sha,
        "fixed_task1_probe_sha256": expected_probe,
        "method_id": method_id,
        "schema": CHAIN_SCHEMA_VERSION,
        "stages": stage_rows,
    }
    chain_sha = sha256_bytes(canonical_json_bytes(chain))
    marker = {
        "code_commit": commit,
        "configuration_freeze_sha256": configuration_sha,
        "environment_lock_sha256": environment_sha,
        "final_checkpoint_endpoint_tensor_sha256": checkpoints[-1]["endpoint_lora_tensor_sha256"],
        "final_global_round": 400,
        "method_chain_provenance_sha256": chain_sha,
        "method_id": method_id,
        "schema": CHAIN_MARKER_SCHEMA_VERSION,
        "status": "COMPLETE",
    }
    marker_sha = sha256_bytes(canonical_json_bytes(marker))
    validate_formal_method_chain(chain, chain_sha, marker, marker_sha)
    return {
        "chain": chain,
        "chain_sha256": chain_sha,
        "complete_marker": marker,
        "complete_marker_sha256": marker_sha,
    }


def write_formal_method_chain(output_directory: Any, bundle: Mapping[str, Any]) -> Mapping[str, str]:
    if not isinstance(bundle, Mapping) or set(bundle) != {
        "chain",
        "chain_sha256",
        "complete_marker",
        "complete_marker_sha256",
    }:
        raise ValueError("formal method-chain bundle field set mismatch")
    validate_formal_method_chain(
        bundle["chain"],
        bundle["chain_sha256"],
        bundle["complete_marker"],
        bundle["complete_marker_sha256"],
    )
    output = _directory(output_directory, "formal method-chain output")
    method = bundle["chain"]["method_id"]
    names = {
        "%s.method_chain.json" % method: canonical_json_bytes(bundle["chain"]),
        "%s.method_chain.json.sha256" % method: (
            bundle["chain_sha256"] + "  " + method + ".method_chain.json\n"
        ).encode("ascii"),
        "%s.complete.json" % method: canonical_json_bytes(bundle["complete_marker"]),
        "%s.complete.json.sha256" % method: (
            bundle["complete_marker_sha256"] + "  " + method + ".complete.json\n"
        ).encode("ascii"),
    }
    if any((output / name).exists() for name in names):
        raise FileExistsError("refusing to overwrite formal method-chain artifacts")
    with tempfile.TemporaryDirectory(dir=str(output), prefix=".method-chain.") as temporary:
        staging = Path(temporary)
        for name, payload in names.items():
            path = staging / name
            path.write_bytes(payload)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        published = []
        try:
            for name in names:
                destination = output / name
                os.link(str(staging / name), str(destination))
                published.append(destination)
        except Exception:
            for destination in reversed(published):
                destination.unlink(missing_ok=True)
            raise
    return {"chain_sha256": bundle["chain_sha256"], "complete_marker_sha256": bundle["complete_marker_sha256"]}

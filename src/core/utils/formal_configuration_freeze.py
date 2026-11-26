"""Canonical training/evaluator configuration identity for the formal run."""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_schema import matches_schema
from .continual_data_manifest import TASK_DATASETS
from .data_manifest import canonical_json_bytes, sha256_bytes, sha256_file
from .frozen_continual_schedule import (
    FORMAL_SHADOW_LEARNING_RATE,
    MAIN_LEARNING_RATE_SCHEDULE,
    PROBE_COLLECTION_LR_POLICY,
    PROBE_COLLECTION_STATE_CONTRACT,
    SHADOW_DETECTION_CONTRACT,
)
from .formal_runtime_environment import FORMAL_GPU_RUNTIME_ENVIRONMENT


SCHEMA_VERSION = "safe-fcl.formal-configuration-freeze.v2"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha(value: Any, name: str, length: int = 64) -> str:
    pattern = HEX40 if length == 40 else HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(name + " must be lowercase hexadecimal")
    return value


def build_formal_configuration_freeze(
    *,
    code_commit: str,
    protocol_sha256: str,
    environment_lock_sha256: str,
    base_model_identity_sha256: str,
    r0_identity_sha256: str,
    r0_tensor_sha256: str,
    fixed_task1_probe_sha256: str,
    lora_key_order_sha256: str,
    global_overlap_audit_sha256: str,
    probe_diagnostic_index_sha256: str,
    token_contract_sha256: str,
    task_manifest_sha256: Sequence[str],
    task_schedule_sha256: Sequence[str],
) -> Mapping[str, Any]:
    for name, values in (
        ("task_manifest_sha256", task_manifest_sha256),
        ("task_schedule_sha256", task_schedule_sha256),
    ):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 4:
            raise ValueError(name + " must contain four ordered SHA values")
    manifests = [_sha(value, "task manifest SHA") for value in task_manifest_sha256]
    schedules = [_sha(value, "task schedule SHA") for value in task_schedule_sha256]
    return {
        "base_model_identity_sha256": _sha(
            base_model_identity_sha256, "base_model_identity_sha256"
        ),
        "code_commit": _sha(code_commit, "code_commit", 40),
        "environment_lock_sha256": _sha(
            environment_lock_sha256, "environment_lock_sha256"
        ),
        "fixed_task1_probe": {
            "feature_spec": "first_B",
            "lora_key_order_sha256": _sha(
                lora_key_order_sha256, "lora_key_order_sha256"
            ),
            "normalization": "l2",
            "probe_sha256": _sha(
                fixed_task1_probe_sha256, "fixed_task1_probe_sha256"
            ),
            "threshold": 0.8,
        },
        "formal_methods": ["pure_fedavg", "step_level", "client_level", "shadow_level"],
        "global_overlap_audit_sha256": _sha(
            global_overlap_audit_sha256, "global_overlap_audit_sha256"
        ),
        "probe_collection_contract": {
            "learning_rate_policy": PROBE_COLLECTION_LR_POLICY,
            "rounds": 10,
            "state_contract": PROBE_COLLECTION_STATE_CONTRACT,
        },
        "probe_diagnostic_contract": {
            "batch_size": 16,
            "index_sha256": _sha(
                probe_diagnostic_index_sha256,
                "probe_diagnostic_index_sha256",
            ),
            "learning_rate": 5e-5,
            "local_optimizer_steps": 10,
            "records_per_role": 160,
        },
        "protocol_sha256": _sha(protocol_sha256, "protocol_sha256"),
        "runtime_environment": dict(FORMAL_GPU_RUNTIME_ENVIRONMENT),
        "r0": {
            "identity_sha256": _sha(r0_identity_sha256, "r0_identity_sha256"),
            "tensor_sha256": _sha(r0_tensor_sha256, "r0_tensor_sha256"),
        },
        "schema": SCHEMA_VERSION,
        "shadow_contract": {
            "detection_contract": SHADOW_DETECTION_CONTRACT,
            "learning_rate": FORMAL_SHADOW_LEARNING_RATE,
            "stage_base_rounds": [0, 100, 200, 300],
        },
        "token_contract_sha256": _sha(
            token_contract_sha256, "token_contract_sha256"
        ),
        "tasks": [
            {
                "data_manifest_sha256": manifests[task_id - 1],
                "datasets": dict(TASK_DATASETS[task_id]),
                "schedule_sha256": schedules[task_id - 1],
                "task_id": task_id,
            }
            for task_id in range(1, 6)
        ],
        "training_contract": {
            "batch_size": 16,
            "benign_clients": 7,
            "client_rows": 500,
            "clients_per_round": 3,
            "completion_aware_max_sequence_length": 512,
            "data_seed": 42,
            "eos_token_id": 128001,
            "fed_algorithm": "fedavg",
            "global_rounds_per_task": 100,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
            "load_in_4bit": False,
            "load_in_8bit": True,
            "local_optimizer_steps": 10,
            "lora_alpha": 16,
            "lora_bias": "none",
            "lora_dropout": 0.05,
            "lora_rank": 8,
            "lora_target_modules": ["q_proj", "v_proj"],
            "lr_scheduler_type": "constant",
            "main_learning_rate_schedule": MAIN_LEARNING_RATE_SCHEDULE,
            "malicious_clients": 3,
            "manifest_shuffle_seed": 2023,
            "max_grad_norm": 1.0,
            "optimizer": "adamw_bnb_8bit",
            "pad_token_id": 128004,
            "padding_side": "right",
            "samples_per_selected_client": 160,
            "template": "alpaca",
            "trainer_seed": 42,
            "use_peft": True,
            "warmup_steps": 0,
            "weight_decay": 0.0,
        },
    }


def validate_formal_configuration_freeze(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or not matches_schema(value.get("schema"), SCHEMA_VERSION):
        raise ValueError("formal configuration-freeze schema mismatch")
    tasks = value.get("tasks")
    probe = value.get("fixed_task1_probe")
    r0 = value.get("r0")
    diagnostic = value.get("probe_diagnostic_contract")
    rebuilt = build_formal_configuration_freeze(
        code_commit=value.get("code_commit"),
        protocol_sha256=value.get("protocol_sha256"),
        environment_lock_sha256=value.get("environment_lock_sha256"),
        base_model_identity_sha256=value.get("base_model_identity_sha256"),
        r0_identity_sha256=r0.get("identity_sha256") if isinstance(r0, Mapping) else None,
        r0_tensor_sha256=r0.get("tensor_sha256") if isinstance(r0, Mapping) else None,
        fixed_task1_probe_sha256=(probe.get("probe_sha256") if isinstance(probe, Mapping) else None),
        lora_key_order_sha256=(probe.get("lora_key_order_sha256") if isinstance(probe, Mapping) else None),
        global_overlap_audit_sha256=value.get("global_overlap_audit_sha256"),
        probe_diagnostic_index_sha256=(diagnostic.get("index_sha256") if isinstance(diagnostic, Mapping) else None),
        token_contract_sha256=value.get("token_contract_sha256"),
        task_manifest_sha256=[row.get("data_manifest_sha256") for row in tasks] if isinstance(tasks, list) else [],
        task_schedule_sha256=[row.get("schedule_sha256") for row in tasks] if isinstance(tasks, list) else [],
    )
    if dict(value) != rebuilt:
        raise ValueError("formal configuration-freeze content mismatch")


def load_formal_configuration_freeze(
    path: Any,
    expected_sha256: str,
    *,
    code_commit: str,
    task_id: int,
    data_manifest_sha256: str,
    schedule_sha256: str,
    environment_lock_sha256: str,
    fixed_task1_probe_sha256: Any = None,
) -> Mapping[str, Any]:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("formal configuration freeze must not be a symlink")
    target = unresolved.resolve()
    if not target.is_file() or target.stat().st_size == 0:
        raise FileNotFoundError("formal configuration freeze is missing")
    expected = _sha(expected_sha256, "configuration freeze SHA")
    if sha256_file(target) != expected:
        raise ValueError("formal configuration freeze SHA mismatch")
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical_json_bytes(value):
        raise ValueError("formal configuration freeze must be canonical JSON")
    validate_formal_configuration_freeze(value)
    if value["code_commit"] != _sha(code_commit, "code_commit", 40):
        raise ValueError("configuration freeze code commit mismatch")
    if value["environment_lock_sha256"] != _sha(environment_lock_sha256, "environment lock SHA"):
        raise ValueError("configuration freeze environment mismatch")
    if task_id not in range(1, 6):
        raise ValueError("configuration freeze task_id must be 1..4")
    task = value["tasks"][task_id - 1]
    if task["data_manifest_sha256"] != _sha(data_manifest_sha256, "data manifest SHA"):
        raise ValueError("configuration freeze Task manifest mismatch")
    if task["schedule_sha256"] != _sha(schedule_sha256, "schedule SHA"):
        raise ValueError("configuration freeze Task schedule mismatch")
    if fixed_task1_probe_sha256 not in (None, "") and value["fixed_task1_probe"]["probe_sha256"] != _sha(fixed_task1_probe_sha256, "fixed Probe SHA"):
        raise ValueError("configuration freeze fixed Probe mismatch")
    return value


def write_formal_configuration_freeze(path: Any, value: Mapping[str, Any]) -> Mapping[str, str]:
    validate_formal_configuration_freeze(value)
    target = Path(path).expanduser().resolve()
    sidecar = Path(str(target) + ".sha256")
    if not target.parent.is_dir():
        raise FileNotFoundError("configuration-freeze parent must exist")
    if target.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite formal configuration freeze")
    payload = canonical_json_bytes(dict(value))
    digest = sha256_bytes(payload)
    with tempfile.TemporaryDirectory(dir=str(target.parent), prefix=".configuration-freeze.") as temporary:
        staging = Path(temporary)
        staged = staging / target.name
        staged_sidecar = staging / sidecar.name
        staged.write_bytes(payload)
        staged_sidecar.write_text(digest + "  " + target.name + "\n", encoding="ascii")
        published = []
        try:
            os.link(str(staged), str(target))
            published.append(target)
            os.link(str(staged_sidecar), str(sidecar))
        except Exception:
            for item in reversed(published):
                item.unlink(missing_ok=True)
            raise
    return {"path": str(target), "sha256": digest, "sidecar": str(sidecar)}

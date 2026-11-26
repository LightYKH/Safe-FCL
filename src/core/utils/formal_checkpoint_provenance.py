"""Fail-closed lineage for formal continual checkpoints."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

from .artifact_schema import matches_schema
from .data_manifest import canonical_json_bytes, sha256_file
from .frozen_continual_schedule import (
    FORMAL_SHADOW_LEARNING_RATE,
    MAIN_LEARNING_RATE_SCHEDULE,
    SHADOW_DETECTION_CONTRACT,
    frozen_cosine_lr,
    validate_formal_shadow_learning_rate,
)


SCHEMA_VERSION = "safe-fcl.formal-continual-checkpoint.v4"
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
METHODS = ("pure", "step", "client", "shadow")


def _require_sha(value: Any, name: str, length: int = 64) -> str:
    text = str(value or "").lower()
    pattern = HEX_40 if length == 40 else HEX_64
    if not pattern.fullmatch(text):
        raise ValueError(name + " must be lowercase hexadecimal")
    return text


def parent_provenance_filename(task_id: int) -> str:
    if task_id not in range(1, 6):
        raise ValueError("task_id must be 1..4")
    return "R0_IDENTITY.json" if task_id == 1 else "FORMAL_CHECKPOINT_PROVENANCE.json"


def load_and_validate_parent_provenance(
    checkpoint: Any,
    expected_provenance_sha256: str,
    *,
    task_id: int,
    method: str,
    code_commit: str,
    adapter_config_sha256: str,
    adapter_model_sha256: str,
    stage_base_tensor_sha256: str,
    fixed_probe_sha256: Any,
    environment_lock_sha256: str,
    configuration_freeze_sha256: str,
    python_executable: str,
    shadow_learning_rate: float = FORMAL_SHADOW_LEARNING_RATE,
) -> Dict[str, Any]:
    if task_id not in range(1, 6):
        raise ValueError("task_id must be 1..4")
    method = str(method).lower()
    if method not in METHODS:
        raise ValueError("unsupported formal method")
    code_commit = _require_sha(code_commit, "code_commit", 40)
    adapter_config_sha256 = _require_sha(
        adapter_config_sha256, "adapter_config_sha256"
    )
    adapter_model_sha256 = _require_sha(
        adapter_model_sha256, "adapter_model_sha256"
    )
    stage_base_tensor_sha256 = _require_sha(
        stage_base_tensor_sha256, "stage_base_tensor_sha256"
    )
    environment_lock_sha256 = _require_sha(
        environment_lock_sha256, "environment_lock_sha256"
    )
    configuration_freeze_sha256 = _require_sha(
        configuration_freeze_sha256, "configuration_freeze_sha256"
    )
    python_executable = str(Path(python_executable).expanduser().resolve())
    expected_provenance_sha256 = _require_sha(
        expected_provenance_sha256, "expected parent provenance SHA-256"
    )
    shadow_learning_rate = validate_formal_shadow_learning_rate(
        shadow_learning_rate
    )
    expected_probe = None
    if method != "pure":
        expected_probe = _require_sha(fixed_probe_sha256, "fixed_probe_sha256")
    elif fixed_probe_sha256 not in (None, ""):
        raise ValueError("Pure FedAvg parent lineage must not bind a Probe")

    checkpoint = Path(checkpoint).expanduser().resolve()
    provenance_path = checkpoint / parent_provenance_filename(task_id)
    if not provenance_path.is_file():
        raise FileNotFoundError("parent provenance is missing: " + str(provenance_path))
    if sha256_file(provenance_path) != expected_provenance_sha256:
        raise ValueError("parent provenance SHA-256 mismatch")
    raw = provenance_path.read_bytes()
    value = json.loads(raw.decode("utf-8"))

    if task_id == 1:
        if value.get("code_commit") != code_commit:
            raise ValueError("R0 code_commit mismatch")
        if value.get("adapter_config_sha256") != adapter_config_sha256:
            raise ValueError("R0 adapter config SHA mismatch")
        if value.get("adapter_model_sha256") != adapter_model_sha256:
            raise ValueError("R0 adapter model SHA mismatch")
        if value.get("lora_tensor_sha256") != stage_base_tensor_sha256:
            raise ValueError("R0 tensor SHA mismatch")
        if value.get("environment_lock_sha256") != environment_lock_sha256:
            raise ValueError("R0 environment lock SHA mismatch")
        if value.get("python_executable") != python_executable:
            raise ValueError("R0 Python executable mismatch")
    else:
        if raw != canonical_json_bytes(value):
            raise ValueError("parent checkpoint provenance is not canonical JSON")
        if not matches_schema(value.get("schema"), SCHEMA_VERSION):
            raise ValueError("unsupported parent checkpoint provenance schema")
        if value.get("code_commit") != code_commit:
            raise ValueError("parent checkpoint code_commit mismatch")
        if value.get("formal_method") != method:
            raise ValueError("parent checkpoint method lineage mismatch")
        if value.get("task_id") != task_id - 1:
            raise ValueError("parent checkpoint task lineage mismatch")
        if value.get("global_round") != (task_id - 1) * 100:
            raise ValueError("parent checkpoint endpoint round mismatch")
        if value.get("adapter_config_sha256") != adapter_config_sha256:
            raise ValueError("parent adapter config SHA mismatch")
        if value.get("adapter_model_sha256") != adapter_model_sha256:
            raise ValueError("parent adapter model SHA mismatch")
        if value.get("endpoint_lora_tensor_sha256") != stage_base_tensor_sha256:
            raise ValueError("parent endpoint tensor SHA mismatch")
        if value.get("fixed_probe_sha256") != expected_probe:
            raise ValueError("fixed Task1 Probe lineage mismatch")
        if value.get("shadow_learning_rate") != shadow_learning_rate:
            raise ValueError("parent checkpoint Shadow LR contract mismatch")
        if value.get("shadow_detection_contract") != SHADOW_DETECTION_CONTRACT:
            raise ValueError("parent checkpoint Shadow detection contract mismatch")
        if value.get("main_learning_rate_schedule") != MAIN_LEARNING_RATE_SCHEDULE:
            raise ValueError("parent checkpoint Main LR schedule mismatch")
        if value.get("main_learning_rate") != frozen_cosine_lr(99):
            raise ValueError("parent checkpoint endpoint Main LR mismatch")
        if value.get("shadow_detection_active") != (method == "shadow"):
            raise ValueError("parent checkpoint Shadow activation mismatch")
        if value.get("shadow_stage_base_source_global_round") != (task_id - 2) * 100:
            raise ValueError("parent checkpoint Shadow stage-base lineage mismatch")
        if value.get("environment_lock_sha256") != environment_lock_sha256:
            raise ValueError("parent checkpoint environment lock SHA mismatch")
        if value.get("configuration_freeze_sha256") != configuration_freeze_sha256:
            raise ValueError("parent checkpoint configuration freeze SHA mismatch")
        if value.get("python_executable") != python_executable:
            raise ValueError("parent checkpoint Python executable mismatch")

    return {
        "path": str(provenance_path),
        "sha256": expected_provenance_sha256,
        "value": value,
    }


def build_checkpoint_provenance(
    *,
    code_commit: str,
    task_id: int,
    formal_method: str,
    global_round: int,
    adapter_config_sha256: str,
    adapter_model_sha256: str,
    endpoint_lora_tensor_sha256: str,
    stage_base_tensor_sha256: str,
    parent_provenance_sha256: str,
    data_manifest_sha256: str,
    schedule_sha256: str,
    fixed_probe_sha256: Any,
    environment_lock_sha256: str,
    configuration_freeze_sha256: str,
    python_executable: str,
    shadow_learning_rate: float = FORMAL_SHADOW_LEARNING_RATE,
) -> Dict[str, Any]:
    formal_method = str(formal_method).lower()
    if formal_method not in METHODS:
        raise ValueError("unsupported formal method")
    if task_id not in range(1, 6):
        raise ValueError("task_id must be 1..4")
    expected_round_min = (task_id - 1) * 100 + 1
    expected_round_max = task_id * 100
    if not isinstance(global_round, int) or not expected_round_min <= global_round <= expected_round_max:
        raise ValueError("global_round is outside the formal Task stage")
    probe_sha = None
    if formal_method != "pure":
        probe_sha = _require_sha(fixed_probe_sha256, "fixed_probe_sha256")
    elif fixed_probe_sha256 not in (None, ""):
        raise ValueError("Pure FedAvg checkpoint must not bind a Probe")
    shadow_learning_rate = validate_formal_shadow_learning_rate(
        shadow_learning_rate
    )
    stage_round_zero_based = global_round - expected_round_min
    return {
        "adapter_config_sha256": _require_sha(
            adapter_config_sha256, "adapter_config_sha256"
        ),
        "adapter_model_sha256": _require_sha(
            adapter_model_sha256, "adapter_model_sha256"
        ),
        "code_commit": _require_sha(code_commit, "code_commit", 40),
        "configuration_freeze_sha256": _require_sha(
            configuration_freeze_sha256, "configuration_freeze_sha256"
        ),
        "data_manifest_sha256": _require_sha(
            data_manifest_sha256, "data_manifest_sha256"
        ),
        "endpoint_lora_tensor_sha256": _require_sha(
            endpoint_lora_tensor_sha256, "endpoint_lora_tensor_sha256"
        ),
        "environment_lock_sha256": _require_sha(
            environment_lock_sha256, "environment_lock_sha256"
        ),
        "fixed_probe_sha256": probe_sha,
        "formal_method": formal_method,
        "global_round": global_round,
        "main_learning_rate": frozen_cosine_lr(stage_round_zero_based),
        "main_learning_rate_schedule": MAIN_LEARNING_RATE_SCHEDULE,
        "parent_provenance_sha256": _require_sha(
            parent_provenance_sha256, "parent_provenance_sha256"
        ),
        "python_executable": str(Path(python_executable).expanduser().resolve()),
        "schedule_sha256": _require_sha(schedule_sha256, "schedule_sha256"),
        "schema": SCHEMA_VERSION,
        "shadow_detection_active": formal_method == "shadow",
        "shadow_detection_contract": SHADOW_DETECTION_CONTRACT,
        "shadow_learning_rate": shadow_learning_rate,
        "shadow_stage_base_source_global_round": (task_id - 1) * 100,
        "stage_base_tensor_sha256": _require_sha(
            stage_base_tensor_sha256, "stage_base_tensor_sha256"
        ),
        "task_id": task_id,
    }


def write_checkpoint_provenance(checkpoint: Any, value: Mapping[str, Any]) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    target = checkpoint / "FORMAL_CHECKPOINT_PROVENANCE.json"
    if target.exists():
        raise FileExistsError("refusing to overwrite checkpoint provenance")
    payload = canonical_json_bytes(dict(value))
    with target.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return target

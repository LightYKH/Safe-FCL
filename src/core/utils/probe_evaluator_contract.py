"""Fail-closed semantic contract for the two formal Probe evaluators."""

import re
from typing import Any, Mapping

from .artifact_schema import matches_schema

SHADOW_EVALUATION = "shadow_contract_probe_evaluation"
DRIFT_DIAGNOSTIC = "task1_anchored_current_checkpoint_probe_drift_diagnostic"
JOB_SCHEMA_VERSION = "safe-fcl.probe-evaluator-job.v1"
METHODS = ("pure_fedavg", "step_level", "client_level", "shadow_level")
FORMAL_METHOD_IDS = {
    "pure_fedavg": "pure",
    "step_level": "step",
    "client_level": "client",
    "shadow_level": "shadow",
}
FIXED_LR = 5e-5
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_SHA_FIELDS = (
    "checkpoint_adapter_config_sha256",
    "checkpoint_adapter_model_sha256",
    "checkpoint_provenance_sha256",
    "method_chain_provenance_sha256",
    "probe_sha256",
    "validation_subset_sha256",
    "batch_schedule_sha256",
    "token_contract_sha256",
    "environment_lock_sha256",
    "configuration_freeze_sha256",
    "full_chain_complete_marker_sha256",
    "delta_base_tensor_sha256",
    "checkpoint_endpoint_tensor_sha256",
    "lora_key_order_sha256",
)


def task_for_checkpoint_round(round_number: int) -> int:
    if (
        not isinstance(round_number, int)
        or isinstance(round_number, bool)
        or round_number < 10
        or round_number > 500
        or round_number % 10
    ):
        raise ValueError("diagnostic checkpoint round must be one of R10..R400")
    return (round_number - 1) // 100 + 1


def validate_probe_evaluator_job(job: Mapping[str, Any]) -> None:
    if not isinstance(job, Mapping):
        raise TypeError("Probe evaluator job must be a mapping")
    if not matches_schema(job.get("schema"), JOB_SCHEMA_VERSION):
        raise ValueError("Probe evaluator job schema mismatch")
    evaluation_type = job.get("evaluation_type")
    method = job.get("method_id")
    task_id = job.get("task_id")
    checkpoint_round = job.get("global_checkpoint_round")
    delta_base_round = job.get("delta_base_round")
    validation_task_id = job.get("validation_task_id")
    if method not in METHODS:
        raise ValueError("unsupported Probe evaluator method_id")
    if task_id not in range(1, 6):
        raise ValueError("Probe evaluator task_id must be 1..4")

    if evaluation_type == SHADOW_EVALUATION:
        expected_round = (task_id - 1) * 100
        if method != "shadow_level":
            raise ValueError("Shadow-contract evaluation only accepts shadow_level")
        if checkpoint_round != expected_round or delta_base_round != expected_round:
            raise ValueError("Shadow-contract stage-base round mismatch")
        if validation_task_id != task_id:
            raise ValueError("Shadow-contract validation_task_id must equal task_id")
        stage_base = job.get("stage_shadow_base_tensor_sha256")
        if not isinstance(stage_base, str) or HEX64.fullmatch(stage_base) is None:
            raise ValueError("Shadow-contract stage base SHA is required")
        if stage_base != job.get("delta_base_tensor_sha256"):
            raise ValueError("Shadow-contract delta base differs from stage base")
        expected_checkpoint_task = task_id - 1
        expected_lr_policy = "fixed_shadow_contract_5e-5"
    elif evaluation_type == DRIFT_DIAGNOSTIC:
        expected_task = task_for_checkpoint_round(checkpoint_round)
        if task_id != expected_task:
            raise ValueError("current-checkpoint task_id does not match round")
        if delta_base_round != checkpoint_round:
            raise ValueError("current-checkpoint delta base must equal checkpoint")
        if validation_task_id != 1:
            raise ValueError("Task1-Anchored validation_task_id must be 1")
        expected_checkpoint_task = task_id
        expected_lr_policy = "fixed_current_checkpoint_diagnostic_5e-5"
    else:
        raise ValueError("unsupported Probe evaluator evaluation_type")

    if job.get("checkpoint_method_id") != method:
        raise ValueError("cross-method checkpoint reuse is forbidden")
    if job.get("checkpoint_task_id") != expected_checkpoint_task:
        raise ValueError("checkpoint task lineage mismatch")
    if job.get("checkpoint_global_round") != checkpoint_round:
        raise ValueError("checkpoint round lineage mismatch")
    if job.get("input_partition") != "model_validation":
        raise ValueError("Probe evaluators may only consume model_validation")
    if job.get("lr_policy") != expected_lr_policy:
        raise ValueError("Probe evaluator LR policy mismatch")
    learning_rate = job.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
        raise ValueError("Probe evaluator learning_rate must be numeric")
    if abs(float(learning_rate) - FIXED_LR) > 1e-15:
        raise ValueError("Probe evaluator learning_rate must equal 5e-5")
    if job.get("optimizer_contract") != {
        "batch_size": 16,
        "data_seed": 42,
        "gradient_accumulation_steps": 1,
        "local_optimizer_steps": 10,
        "lr_scheduler_type": "constant",
        "max_grad_norm": 1.0,
        "optimizer": "adamw_bnb_8bit",
        "seed": 42,
        "warmup_steps": 0,
        "weight_decay": 0.0,
    }:
        raise ValueError("Probe evaluator optimizer contract mismatch")
    if job.get("execution_phase") != "post_chain_offline_replay":
        raise ValueError("formal Probe evaluation must be post-chain offline replay")
    if job.get("temporary_update_written_back") is not False:
        raise ValueError("temporary diagnostic updates must never be written back")
    if job.get("controls_training") is not False:
        raise ValueError("Probe evaluator output must not control training")
    if job.get("probe_threshold") != 0.8:
        raise ValueError("Probe evaluator threshold must equal 0.8")
    if job.get("probe_normalization") != "l2":
        raise ValueError("Probe evaluator normalization must be l2")
    if job.get("probe_feature_spec") != "first_B":
        raise ValueError("Probe evaluator feature spec must be first_B")

    code_commit = job.get("code_commit")
    if not isinstance(code_commit, str) or HEX40.fullmatch(code_commit) is None:
        raise ValueError("Probe evaluator code_commit must be a Git SHA")
    for field in REQUIRED_SHA_FIELDS:
        value = job.get(field)
        if not isinstance(value, str) or HEX64.fullmatch(value) is None:
            raise ValueError(field + " must be a lowercase SHA-256")
    if job["delta_base_tensor_sha256"] != job["checkpoint_endpoint_tensor_sha256"]:
        raise ValueError("loaded delta base differs from checkpoint endpoint tensor")


def validate_probe_evaluator_checkpoint_provenance(
    job: Mapping[str, Any], provenance: Mapping[str, Any]
) -> None:
    """Bind an evaluator job to its actual R0 or formal checkpoint lineage."""
    validate_probe_evaluator_job(job)
    if not isinstance(provenance, Mapping):
        raise TypeError("checkpoint provenance must be a mapping")
    if job["global_checkpoint_round"] == 0:
        expected = {
            "adapter_config_sha256": job["checkpoint_adapter_config_sha256"],
            "adapter_model_sha256": job["checkpoint_adapter_model_sha256"],
            "code_commit": job["code_commit"],
            "environment_lock_sha256": job["environment_lock_sha256"],
            "lora_tensor_sha256": job["checkpoint_endpoint_tensor_sha256"],
        }
        for field, value in expected.items():
            if provenance.get(field) != value:
                raise ValueError("R0 checkpoint provenance mismatch: " + field)
        return

    from .formal_checkpoint_provenance import SCHEMA_VERSION

    if not matches_schema(provenance.get("schema"), SCHEMA_VERSION):
        raise ValueError("formal checkpoint provenance mismatch: schema")
    expected = {
        "adapter_config_sha256": job["checkpoint_adapter_config_sha256"],
        "adapter_model_sha256": job["checkpoint_adapter_model_sha256"],
        "code_commit": job["code_commit"],
        "configuration_freeze_sha256": job["configuration_freeze_sha256"],
        "endpoint_lora_tensor_sha256": job["checkpoint_endpoint_tensor_sha256"],
        "environment_lock_sha256": job["environment_lock_sha256"],
        "formal_method": FORMAL_METHOD_IDS[job["method_id"]],
        "global_round": job["global_checkpoint_round"],
        "task_id": job["checkpoint_task_id"],
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError("formal checkpoint provenance mismatch: " + field)
    expected_probe = None if job["method_id"] == "pure_fedavg" else job["probe_sha256"]
    if provenance.get("fixed_probe_sha256") != expected_probe:
        raise ValueError("formal checkpoint provenance mismatch: fixed_probe_sha256")

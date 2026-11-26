"""Build the exact 4+160 formal Probe evaluator job matrix."""

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .artifact_schema import matches_schema
from .data_manifest import canonical_json_bytes, sha256_bytes
from .probe_evaluator_contract import (
    DRIFT_DIAGNOSTIC,
    FIXED_LR,
    JOB_SCHEMA_VERSION,
    METHODS,
    SHADOW_EVALUATION,
    task_for_checkpoint_round,
    validate_probe_evaluator_job,
)


CHAIN_SCHEMA_VERSION = "safe-fcl.formal-method-chain.v1"
CHAIN_MARKER_SCHEMA_VERSION = "safe-fcl.formal-method-chain-complete.v1"
JOB_INDEX_SCHEMA_VERSION = "safe-fcl.probe-evaluator-job-index.v1"
CHECKPOINT_ROUNDS = (0,) + tuple(range(10, 501, 10))
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(name + " must be a lowercase SHA-256")
    return value


def _commit(value: Any) -> str:
    if not isinstance(value, str) or HEX40.fullmatch(value) is None:
        raise ValueError("code_commit must be a lowercase Git SHA")
    return value


def checkpoint_task_id(round_number: int) -> int:
    if round_number == 0:
        return 0
    return task_for_checkpoint_round(round_number)


def validate_formal_method_chain(
    chain: Mapping[str, Any],
    chain_sha256: str,
    marker: Mapping[str, Any],
    marker_sha256: str,
) -> None:
    if not isinstance(chain, Mapping) or not matches_schema(chain.get("schema"), CHAIN_SCHEMA_VERSION):
        raise ValueError("formal method-chain schema mismatch")
    chain_digest = _sha(chain_sha256, "method_chain_provenance_sha256")
    if sha256_bytes(canonical_json_bytes(dict(chain))) != chain_digest:
        raise ValueError("formal method-chain SHA-256 mismatch")
    method = chain.get("method_id")
    if method not in METHODS:
        raise ValueError("formal method-chain method_id is invalid")
    code_commit = _commit(chain.get("code_commit"))
    environment_sha = _sha(
        chain.get("environment_lock_sha256"), "environment_lock_sha256"
    )
    configuration_sha = _sha(
        chain.get("configuration_freeze_sha256"),
        "configuration_freeze_sha256",
    )
    probe_sha = chain.get("fixed_task1_probe_sha256")
    if method == "pure_fedavg":
        if probe_sha is not None:
            raise ValueError("Pure method chain must not bind a training Probe")
    else:
        _sha(probe_sha, "fixed_task1_probe_sha256")

    checkpoints = chain.get("checkpoints")
    if not isinstance(checkpoints, list) or [
        entry.get("global_round") for entry in checkpoints
    ] != list(CHECKPOINT_ROUNDS):
        raise ValueError("formal method chain must contain R0 and R10..R400")
    for entry, round_number in zip(checkpoints, CHECKPOINT_ROUNDS):
        if not isinstance(entry, Mapping):
            raise TypeError("formal method-chain checkpoint must be a mapping")
        if entry.get("task_id") != checkpoint_task_id(round_number):
            raise ValueError("formal method-chain checkpoint Task mismatch")
        if entry.get("method_id") != method:
            raise ValueError("formal method-chain checkpoint method mismatch")
        directory = entry.get("checkpoint_directory")
        if not isinstance(directory, str) or not Path(directory).is_absolute():
            raise ValueError("checkpoint_directory must be an absolute path")
        for field in (
            "adapter_config_sha256",
            "adapter_model_sha256",
            "checkpoint_provenance_sha256",
            "endpoint_lora_tensor_sha256",
        ):
            _sha(entry.get(field), field)

    stages = chain.get("stages")
    if not isinstance(stages, list) or len(stages) != 4:
        raise ValueError("formal method chain must contain four stage identities")
    checkpoint_by_round = {
        entry["global_round"]: entry for entry in checkpoints
    }
    for task_id, stage in enumerate(stages, start=1):
        if not isinstance(stage, Mapping):
            raise TypeError("formal method-chain stage must be a mapping")
        start_round = (task_id - 1) * 100
        if stage.get("task_id") != task_id:
            raise ValueError("formal method-chain stage Task mismatch")
        if stage.get("start_global_round") != start_round:
            raise ValueError("formal method-chain stage start mismatch")
        if stage.get("end_global_round") != task_id * 100:
            raise ValueError("formal method-chain stage end mismatch")
        for field in (
            "data_manifest_sha256",
            "parent_provenance_sha256",
            "schedule_sha256",
            "stage_base_tensor_sha256",
        ):
            _sha(stage.get(field), field)
        stage_base = checkpoint_by_round[start_round]
        if stage["stage_base_tensor_sha256"] != stage_base[
            "endpoint_lora_tensor_sha256"
        ]:
            raise ValueError("formal method-chain stage base tensor mismatch")
        if stage["parent_provenance_sha256"] != stage_base[
            "checkpoint_provenance_sha256"
        ]:
            raise ValueError("formal method-chain stage parent provenance mismatch")

    if not isinstance(marker, Mapping) or not matches_schema(
        marker.get("schema"), CHAIN_MARKER_SCHEMA_VERSION
    ):
        raise ValueError("formal method-chain complete marker schema mismatch")
    marker_digest = _sha(marker_sha256, "full_chain_complete_marker_sha256")
    if sha256_bytes(canonical_json_bytes(dict(marker))) != marker_digest:
        raise ValueError("formal method-chain complete marker SHA mismatch")
    expected_marker = {
        "code_commit": code_commit,
        "configuration_freeze_sha256": configuration_sha,
        "environment_lock_sha256": environment_sha,
        "final_checkpoint_endpoint_tensor_sha256": checkpoints[-1][
            "endpoint_lora_tensor_sha256"
        ],
        "final_global_round": 400,
        "method_chain_provenance_sha256": chain_digest,
        "method_id": method,
        "schema": marker["schema"],
        "status": "COMPLETE",
    }
    if dict(marker) != expected_marker:
        raise ValueError("formal method-chain complete marker content mismatch")


def _checkpoint_job_fields(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "checkpoint_adapter_config_sha256": entry["adapter_config_sha256"],
        "checkpoint_adapter_model_sha256": entry["adapter_model_sha256"],
        "checkpoint_endpoint_tensor_sha256": entry[
            "endpoint_lora_tensor_sha256"
        ],
        "checkpoint_global_round": entry["global_round"],
        "checkpoint_method_id": entry["method_id"],
        "checkpoint_provenance_sha256": entry[
            "checkpoint_provenance_sha256"
        ],
        "checkpoint_task_id": entry["task_id"],
    }


def _base_job(
    *,
    common: Mapping[str, Any],
    chain: Mapping[str, Any],
    chain_sha256: str,
    marker_sha256: str,
    checkpoint: Mapping[str, Any],
    task_artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    value = {
        **_checkpoint_job_fields(checkpoint),
        "code_commit": common["code_commit"],
        "configuration_freeze_sha256": common[
            "configuration_freeze_sha256"
        ],
        "controls_training": False,
        "delta_base_tensor_sha256": checkpoint[
            "endpoint_lora_tensor_sha256"
        ],
        "environment_lock_sha256": common["environment_lock_sha256"],
        "execution_phase": "post_chain_offline_replay",
        "full_chain_complete_marker_sha256": marker_sha256,
        "learning_rate": FIXED_LR,
        "lora_key_order_sha256": common["lora_key_order_sha256"],
        "method_chain_provenance_sha256": chain_sha256,
        "method_id": chain["method_id"],
        "optimizer_contract": {
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
        },
        "probe_feature_spec": "first_B",
        "probe_normalization": "l2",
        "probe_sha256": common["probe_sha256"],
        "probe_threshold": 0.8,
        "schema": JOB_SCHEMA_VERSION,
        "temporary_update_written_back": False,
        "token_contract_sha256": common["token_contract_sha256"],
        "validation_subset_sha256": task_artifact[
            "validation_subset_sha256"
        ],
        "batch_schedule_sha256": task_artifact["batch_schedule_sha256"],
    }
    return value


def build_probe_evaluator_job_matrix(
    *,
    common: Mapping[str, Any],
    task_artifacts: Sequence[Mapping[str, Any]],
    method_bundles: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(common, Mapping):
        raise TypeError("Probe evaluator common identity must be a mapping")
    code_commit = _commit(common.get("code_commit"))
    for field in (
        "configuration_freeze_sha256",
        "environment_lock_sha256",
        "lora_key_order_sha256",
        "probe_sha256",
        "token_contract_sha256",
    ):
        _sha(common.get(field), field)
    if (
        not isinstance(task_artifacts, Sequence)
        or isinstance(task_artifacts, (str, bytes))
        or len(task_artifacts) != 4
    ):
        raise ValueError("Probe evaluator requires four Task artifacts")
    tasks = {}
    for expected_task, artifact in enumerate(task_artifacts, start=1):
        if not isinstance(artifact, Mapping):
            raise TypeError("Probe evaluator Task artifact must be a mapping")
        if artifact.get("task_id") != expected_task:
            raise ValueError("Probe evaluator Task artifacts are not ordered 1..4")
        for field in ("validation_subset_sha256", "batch_schedule_sha256"):
            _sha(artifact.get(field), field)
        tasks[expected_task] = artifact

    if (
        not isinstance(method_bundles, Sequence)
        or isinstance(method_bundles, (str, bytes))
        or len(method_bundles) != 4
    ):
        raise ValueError("Probe evaluator requires four method-chain bundles")
    bundles = {}
    shared_r0 = None
    shared_stage_inputs = None
    for expected_method, bundle in zip(METHODS, method_bundles):
        if not isinstance(bundle, Mapping):
            raise TypeError("method-chain bundle must be a mapping")
        chain = bundle.get("chain")
        marker = bundle.get("complete_marker")
        chain_sha = bundle.get("chain_sha256")
        marker_sha = bundle.get("complete_marker_sha256")
        validate_formal_method_chain(chain, chain_sha, marker, marker_sha)
        if chain["method_id"] != expected_method:
            raise ValueError("method-chain bundles must follow the formal method order")
        if (
            chain["code_commit"] != code_commit
            or chain["environment_lock_sha256"]
            != common["environment_lock_sha256"]
            or chain["configuration_freeze_sha256"]
            != common["configuration_freeze_sha256"]
        ):
            raise ValueError("method-chain common identity mismatch")
        if expected_method != "pure_fedavg" and chain[
            "fixed_task1_probe_sha256"
        ] != common["probe_sha256"]:
            raise ValueError("method-chain fixed Task1 Probe mismatch")
        r0 = {
            field: chain["checkpoints"][0][field]
            for field in (
                "adapter_config_sha256",
                "adapter_model_sha256",
                "checkpoint_directory",
                "checkpoint_provenance_sha256",
                "endpoint_lora_tensor_sha256",
                "global_round",
                "task_id",
            )
        }
        if shared_r0 is None:
            shared_r0 = r0
        elif r0 != shared_r0:
            raise ValueError("formal methods do not share one identical R0")
        stage_inputs = [
            {
                field: stage[field]
                for field in (
                    "data_manifest_sha256",
                    "end_global_round",
                    "schedule_sha256",
                    "start_global_round",
                    "task_id",
                )
            }
            for stage in chain["stages"]
        ]
        if shared_stage_inputs is None:
            shared_stage_inputs = stage_inputs
        elif stage_inputs != shared_stage_inputs:
            raise ValueError(
                "formal methods do not share identical Task manifests and schedules"
            )
        bundles[expected_method] = bundle

    built = []
    shadow_bundle = bundles["shadow_level"]
    shadow_chain = shadow_bundle["chain"]
    checkpoints_by_round = {
        row["global_round"]: row for row in shadow_chain["checkpoints"]
    }
    for task_id in range(1, 6):
        stage_round = (task_id - 1) * 100
        checkpoint = checkpoints_by_round[stage_round]
        job = _base_job(
            common=common,
            chain=shadow_chain,
            chain_sha256=shadow_bundle["chain_sha256"],
            marker_sha256=shadow_bundle["complete_marker_sha256"],
            checkpoint=checkpoint,
            task_artifact=tasks[task_id],
        )
        job.update({
            "delta_base_round": stage_round,
            "evaluation_type": SHADOW_EVALUATION,
            "global_checkpoint_round": stage_round,
            "input_partition": "model_validation",
            "lr_policy": "fixed_shadow_contract_5e-5",
            "stage_shadow_base_tensor_sha256": checkpoint[
                "endpoint_lora_tensor_sha256"
            ],
            "task_id": task_id,
            "validation_task_id": task_id,
        })
        validate_probe_evaluator_job(job)
        built.append(("shadow-task%d" % task_id, checkpoint, job))

    task1_artifact = tasks[1]
    for method in METHODS:
        bundle = bundles[method]
        chain = bundle["chain"]
        for checkpoint in chain["checkpoints"][1:]:
            round_number = checkpoint["global_round"]
            task_id = task_for_checkpoint_round(round_number)
            job = _base_job(
                common=common,
                chain=chain,
                chain_sha256=bundle["chain_sha256"],
                marker_sha256=bundle["complete_marker_sha256"],
                checkpoint=checkpoint,
                task_artifact=task1_artifact,
            )
            job.update({
                "delta_base_round": round_number,
                "evaluation_type": DRIFT_DIAGNOSTIC,
                "global_checkpoint_round": round_number,
                "input_partition": "model_validation",
                "lr_policy": "fixed_current_checkpoint_diagnostic_5e-5",
                "task_id": task_id,
                "validation_task_id": 1,
            })
            validate_probe_evaluator_job(job)
            built.append(("drift-%s-r%03d" % (method, round_number), checkpoint, job))

    if len(built) != 164 or len({name for name, _, _ in built}) != 164:
        raise RuntimeError("Probe evaluator job matrix is not exactly 164 unique jobs")
    rows = []
    for name, checkpoint, job in built:
        rows.append({
            "checkpoint_directory": checkpoint["checkpoint_directory"],
            "delta_base_round": job["delta_base_round"],
            "evaluation_type": job["evaluation_type"],
            "global_checkpoint_round": job["global_checkpoint_round"],
            "job": job,
            "job_filename": name + ".json",
            "job_sha256": sha256_bytes(canonical_json_bytes(job)),
            "method_id": job["method_id"],
            "result_filename": name + ".result.json",
            "task_id": job["task_id"],
            "validation_task_id": job["validation_task_id"],
        })
    index = {
        "code_commit": code_commit,
        "configuration_freeze_sha256": common["configuration_freeze_sha256"],
        "job_count": 164,
        "jobs": [
            {key: value for key, value in row.items() if key != "job"}
            for row in rows
        ],
        "schema": JOB_INDEX_SCHEMA_VERSION,
    }
    return {"index": index, "rows": rows}

"""Build the exact 160 validation + 80 formal-test model evaluator jobs."""

from typing import Any, Mapping, Sequence

from .data_manifest import canonical_json_bytes, sha256_bytes
from .model_evaluator_contract import (
    FORMAL_TEST,
    FORMAL_TEST_ROUNDS,
    GENERATION_CONTRACT,
    JOB_SCHEMA_VERSION,
    JUDGE_CONTRACT,
    LOSS_CONTRACT,
    METHODS,
    METRIC_CONTRACT,
    MODEL_VALIDATION,
    ROLE_COUNTS,
    VALIDATION_ROUNDS,
    component_sha256,
    validate_model_evaluator_job,
)
from .probe_evaluator_contract import task_for_checkpoint_round
from .probe_evaluator_jobs import validate_formal_method_chain


JOB_INDEX_SCHEMA_VERSION = "safe-fcl.model-evaluator-job-index.v1"


def _checkpoint_fields(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "checkpoint_adapter_config_sha256": checkpoint["adapter_config_sha256"],
        "checkpoint_adapter_model_sha256": checkpoint["adapter_model_sha256"],
        "checkpoint_directory": checkpoint["checkpoint_directory"],
        "checkpoint_endpoint_tensor_sha256": checkpoint["endpoint_lora_tensor_sha256"],
        "checkpoint_global_round": checkpoint["global_round"],
        "checkpoint_method_id": checkpoint["method_id"],
        "checkpoint_provenance_sha256": checkpoint["checkpoint_provenance_sha256"],
        "checkpoint_task_id": checkpoint["task_id"],
    }


def _job(
    *,
    common: Mapping[str, Any],
    method_bundle: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    evaluation_type: str,
    evaluation_task_id: int,
    partition: str,
    task_manifest_sha256: str,
) -> Mapping[str, Any]:
    chain = method_bundle["chain"]
    value = {
        **_checkpoint_fields(checkpoint),
        "base_model_identity_sha256": common["base_model_identity_sha256"],
        "code_commit": common["code_commit"],
        "configuration_freeze_sha256": common["configuration_freeze_sha256"],
        "controls_checkpoint_selection": False,
        "controls_hyperparameters": False,
        "controls_probe_threshold": False,
        "controls_training": False,
        "environment_lock_sha256": common["environment_lock_sha256"],
        "evaluation_freeze_sha256": common["evaluation_freeze_sha256"],
        "evaluation_task_id": evaluation_task_id,
        "evaluation_type": evaluation_type,
        "execution_phase": "post_chain_offline_replay",
        "expected_role_counts": dict(ROLE_COUNTS[partition]),
        "full_chain_complete_marker_sha256": method_bundle[
            "complete_marker_sha256"
        ],
        "generation_contract": dict(GENERATION_CONTRACT),
        "generation_contract_sha256": component_sha256(GENERATION_CONTRACT),
        "global_checkpoint_round": checkpoint["global_round"],
        "input_partition": partition,
        "judge_contract_sha256": common["judge_contract_sha256"],
        "judge_runtime_contract": dict(JUDGE_CONTRACT),
        "judge_runtime_contract_sha256": component_sha256(JUDGE_CONTRACT),
        "loss_contract": dict(LOSS_CONTRACT),
        "loss_contract_sha256": component_sha256(LOSS_CONTRACT),
        "method_chain_provenance_sha256": method_bundle["chain_sha256"],
        "method_id": chain["method_id"],
        "metric_contract": dict(METRIC_CONTRACT),
        "metric_contract_sha256": component_sha256(METRIC_CONTRACT),
        "schema": JOB_SCHEMA_VERSION,
        "task_manifest_sha256": task_manifest_sha256,
        "token_contract_sha256": common["token_contract_sha256"],
    }
    validate_model_evaluator_job(value)
    return value


def build_model_evaluator_job_matrix(
    *,
    common: Mapping[str, Any],
    task_manifest_sha256: Sequence[str],
    method_bundles: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not isinstance(common, Mapping):
        raise TypeError("model evaluator common identity must be a mapping")
    if (
        isinstance(task_manifest_sha256, (str, bytes))
        or not isinstance(task_manifest_sha256, Sequence)
        or len(task_manifest_sha256) != 4
    ):
        raise ValueError("model evaluator requires four Task manifests")
    if (
        isinstance(method_bundles, (str, bytes))
        or not isinstance(method_bundles, Sequence)
        or len(method_bundles) != 4
    ):
        raise ValueError("model evaluator requires four method chains")

    rows = []
    shared_r0 = None
    shared_stage_inputs = None
    for expected_method, bundle in zip(METHODS, method_bundles):
        validate_formal_method_chain(
            bundle["chain"],
            bundle["chain_sha256"],
            bundle["complete_marker"],
            bundle["complete_marker_sha256"],
        )
        chain = bundle["chain"]
        if chain["method_id"] != expected_method:
            raise ValueError("method-chain order must match formal method order")
        for field in (
            "code_commit",
            "configuration_freeze_sha256",
            "environment_lock_sha256",
        ):
            if chain[field] != common[field]:
                raise ValueError("model evaluator method-chain mismatch: " + field)
        checkpoint_by_round = {
            checkpoint["global_round"]: checkpoint
            for checkpoint in chain["checkpoints"]
        }
        r0_identity = {
            key: checkpoint_by_round[0][key]
            for key in (
                "adapter_config_sha256",
                "adapter_model_sha256",
                "checkpoint_provenance_sha256",
                "checkpoint_directory",
                "endpoint_lora_tensor_sha256",
            )
        }
        if shared_r0 is None:
            shared_r0 = r0_identity
        elif r0_identity != shared_r0:
            raise ValueError("all model evaluator methods must share identical R0")
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
                "all model evaluator methods must share Task manifests and schedules"
            )

        for round_number in VALIDATION_ROUNDS:
            task_id = task_for_checkpoint_round(round_number)
            job = _job(
                common=common,
                method_bundle=bundle,
                checkpoint=checkpoint_by_round[round_number],
                evaluation_type=MODEL_VALIDATION,
                evaluation_task_id=task_id,
                partition="model_validation",
                task_manifest_sha256=task_manifest_sha256[task_id - 1],
            )
            name = "validation-%s-r%03d-task%d" % (
                expected_method, round_number, task_id
            )
            rows.append({
                "checkpoint_directory": checkpoint_by_round[round_number][
                    "checkpoint_directory"
                ],
                "evaluation_task_id": task_id,
                "evaluation_type": MODEL_VALIDATION,
                "global_checkpoint_round": round_number,
                "job": job,
                "job_filename": name + ".json",
                "job_sha256": sha256_bytes(canonical_json_bytes(job)),
                "method_id": expected_method,
                "result_filename": name + ".result.json",
            })

        for round_number in FORMAL_TEST_ROUNDS:
            for task_id in range(1, 6):
                job = _job(
                    common=common,
                    method_bundle=bundle,
                    checkpoint=checkpoint_by_round[round_number],
                    evaluation_type=FORMAL_TEST,
                    evaluation_task_id=task_id,
                    partition="formal_test",
                    task_manifest_sha256=task_manifest_sha256[task_id - 1],
                )
                name = "formal-test-%s-r%03d-task%d" % (
                    expected_method, round_number, task_id
                )
                rows.append({
                    "checkpoint_directory": checkpoint_by_round[round_number][
                        "checkpoint_directory"
                    ],
                    "evaluation_task_id": task_id,
                    "evaluation_type": FORMAL_TEST,
                    "global_checkpoint_round": round_number,
                    "job": job,
                    "job_filename": name + ".json",
                    "job_sha256": sha256_bytes(canonical_json_bytes(job)),
                    "method_id": expected_method,
                    "result_filename": name + ".result.json",
                })

    index = {
        "code_commit": common["code_commit"],
        "evaluation_freeze_sha256": common["evaluation_freeze_sha256"],
        "formal_test_job_count": 80,
        "job_count": 240,
        "jobs": [
            {key: value for key, value in row.items() if key != "job"}
            for row in rows
        ],
        "model_validation_job_count": 160,
        "schema": JOB_INDEX_SCHEMA_VERSION,
    }
    if len(rows) != 240:
        raise RuntimeError("model evaluator matrix must contain exactly 240 jobs")
    if len({row["job_sha256"] for row in rows}) != 240:
        raise RuntimeError("model evaluator jobs must be unique")
    return {"index": index, "rows": rows}

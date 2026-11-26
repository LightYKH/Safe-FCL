"""Canonical no-clobber result for model validation and formal testing."""

import copy
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_schema import matches_schema
from .data_manifest import canonical_json_bytes, sha256_bytes
from .model_evaluator_contract import validate_model_evaluator_job
from .model_evaluator_metrics import calculate_model_evaluator_metrics


RESULT_SCHEMA_VERSION = "safe-fcl.model-evaluator-result.v1"
RUNTIME_AUDIT_SCHEMA_VERSION = "safe-fcl.model-evaluator-runtime-audit.v1"
IMMUTABLE_JOB_SHA_FIELDS = (
    "base_model_identity_sha256",
    "checkpoint_adapter_config_sha256",
    "checkpoint_adapter_model_sha256",
    "checkpoint_endpoint_tensor_sha256",
    "checkpoint_provenance_sha256",
    "configuration_freeze_sha256",
    "environment_lock_sha256",
    "evaluation_freeze_sha256",
    "full_chain_complete_marker_sha256",
    "generation_contract_sha256",
    "judge_contract_sha256",
    "loss_contract_sha256",
    "method_chain_provenance_sha256",
    "metric_contract_sha256",
    "task_manifest_sha256",
    "token_contract_sha256",
)


def _validate_runtime_audit(
    job: Mapping[str, Any], runtime_audit: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not isinstance(runtime_audit, Mapping):
        raise TypeError("model evaluator runtime audit must be a mapping")
    if not matches_schema(runtime_audit.get("schema"), RUNTIME_AUDIT_SCHEMA_VERSION):
        raise ValueError("model evaluator runtime audit schema mismatch")
    expected_tensor = job["checkpoint_endpoint_tensor_sha256"]
    if runtime_audit.get("checkpoint_tensor_sha256_before") != expected_tensor:
        raise ValueError("model evaluator checkpoint tensor before-run mismatch")
    if runtime_audit.get("checkpoint_tensor_sha256_after") != expected_tensor:
        raise ValueError("model evaluator checkpoint tensor after-run mismatch")
    if runtime_audit.get("checkpoint_written") is not False:
        raise ValueError("model evaluator must not write checkpoint state")
    if runtime_audit.get("judge_unknown_count") != 0:
        raise ValueError("model evaluator Judge output contains unknown labels")
    if runtime_audit.get("input_role_counts") != job["expected_role_counts"]:
        raise ValueError("model evaluator runtime input counts mismatch")
    immutable = runtime_audit.get("immutable_input_sha256")
    if not isinstance(immutable, Mapping) or set(immutable) != set(
        IMMUTABLE_JOB_SHA_FIELDS
    ):
        raise ValueError("model evaluator immutable SHA audit is incomplete")
    for field in IMMUTABLE_JOB_SHA_FIELDS:
        if immutable.get(field) != job[field]:
            raise ValueError("model evaluator immutable SHA mismatch: " + field)
    return copy.deepcopy(dict(runtime_audit))


def build_model_evaluator_result(
    *,
    job: Mapping[str, Any],
    generation_records: Sequence[Mapping[str, Any]],
    benign_completion_nll_sum: float,
    benign_completion_label_count: int,
    runtime_audit: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_model_evaluator_job(job)
    metrics = calculate_model_evaluator_metrics(
        generation_records,
        benign_completion_nll_sum=benign_completion_nll_sum,
        benign_completion_label_count=benign_completion_label_count,
        expected_role_counts=job["expected_role_counts"],
    )
    return {
        "evaluation_task_id": job["evaluation_task_id"],
        "evaluation_type": job["evaluation_type"],
        "generation_records": copy.deepcopy(list(generation_records)),
        "global_checkpoint_round": job["global_checkpoint_round"],
        "job": copy.deepcopy(dict(job)),
        "job_sha256": sha256_bytes(canonical_json_bytes(dict(job))),
        "method_id": job["method_id"],
        "metrics": metrics,
        "runtime_audit": _validate_runtime_audit(job, runtime_audit),
        "schema": RESULT_SCHEMA_VERSION,
    }


def write_model_evaluator_result(
    path: Any, result: Mapping[str, Any]
) -> Mapping[str, str]:
    target = Path(path).expanduser().resolve()
    sidecar = Path(str(target) + ".sha256")
    if not target.parent.is_dir():
        raise FileNotFoundError("model evaluator result parent must exist")
    if target.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite model evaluator result")
    if not isinstance(result, Mapping) or not matches_schema(result.get("schema"), RESULT_SCHEMA_VERSION):
        raise ValueError("model evaluator result schema mismatch")
    rebuilt = build_model_evaluator_result(
        job=result.get("job", {}),
        generation_records=result.get("generation_records", []),
        benign_completion_nll_sum=result.get("metrics", {})
        .get("benign_completion_only_loss", {})
        .get("nll_sum"),
        benign_completion_label_count=result.get("metrics", {})
        .get("benign_completion_only_loss", {})
        .get("completion_label_count"),
        runtime_audit=result.get("runtime_audit", {}),
    )
    rebuilt["schema"] = result["schema"]
    if dict(result) != rebuilt:
        raise ValueError("model evaluator result content is not canonical")
    payload = canonical_json_bytes(dict(result))
    digest = sha256_bytes(payload)
    with tempfile.TemporaryDirectory(
        dir=str(target.parent), prefix=".model-evaluator-result."
    ) as temporary:
        staging = Path(temporary)
        staged = staging / target.name
        staged_sidecar = staging / sidecar.name
        staged.write_bytes(payload)
        staged_sidecar.write_text(
            digest + "  " + target.name + "\n", encoding="ascii"
        )
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

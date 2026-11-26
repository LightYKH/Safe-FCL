"""Canonical metrics and no-clobber output for formal Probe evaluators."""

import copy
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .artifact_schema import matches_schema
from .data_manifest import canonical_json_bytes, sha256_bytes
from .probe_evaluator_contract import validate_probe_evaluator_job


RESULT_SCHEMA_VERSION = "safe-fcl.probe-evaluator-result.v1"
RUNTIME_AUDIT_SCHEMA_VERSION = "safe-fcl.probe-evaluator-runtime-audit.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_IMMUTABLE_SHA_FIELDS = (
    "batch_schedule_sha256",
    "checkpoint_adapter_config_sha256",
    "checkpoint_adapter_model_sha256",
    "checkpoint_provenance_sha256",
    "configuration_freeze_sha256",
    "environment_lock_sha256",
    "full_chain_complete_marker_sha256",
    "method_chain_provenance_sha256",
    "probe_sha256",
    "token_contract_sha256",
    "validation_subset_sha256",
)


def _require_sha(value: Any, name: str) -> str:
    text = str(value or "").lower()
    if HEX64.fullmatch(text) is None:
        raise ValueError(name + " must be a lowercase SHA-256")
    return text


def _validate_delta_records(records: Sequence[Mapping[str, Any]]) -> list:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("Probe evaluator Delta records must be a sequence")
    if len(records) != 20:
        raise ValueError("Probe evaluator requires exactly 20 Delta records")
    normalized = []
    record_ids = set()
    expected = [("benign", 0, step) for step in range(1, 11)] + [
        ("malicious", 1, step) for step in range(1, 11)
    ]
    for record, (expected_role, expected_label, expected_step) in zip(
        records, expected
    ):
        if not isinstance(record, Mapping):
            raise TypeError("each Probe evaluator Delta record must be a mapping")
        role = record.get("role")
        label = record.get("label")
        optimizer_step = record.get("optimizer_step")
        if (role, label, optimizer_step) != (
            expected_role,
            expected_label,
            expected_step,
        ):
            raise ValueError("Probe evaluator Delta order/role/step mismatch")
        probability = record.get("probability_harmful")
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("Probe harmful probability must be finite in [0,1]")
        record_id = _require_sha(record.get("record_id"), "record_id")
        _require_sha(record.get("delta_tensor_sha256"), "delta_tensor_sha256")
        if record_id in record_ids:
            raise ValueError("Probe evaluator record_id must be unique")
        record_ids.add(record_id)
        normalized.append({
            "delta_tensor_sha256": record["delta_tensor_sha256"],
            "label": label,
            "optimizer_step": optimizer_step,
            "probability_harmful": float(probability),
            "record_id": record_id,
            "role": role,
        })
    return normalized


def calculate_probe_evaluator_metrics(
    records: Sequence[Mapping[str, Any]], threshold: float = 0.8
) -> Dict[str, Any]:
    if abs(float(threshold) - 0.8) > 1e-15:
        raise ValueError("formal Probe evaluator threshold must equal 0.8")
    normalized = _validate_delta_records(records)
    tp = tn = fp = fn = 0
    for record in normalized:
        prediction = 1 if record["probability_harmful"] >= 0.8 else 0
        label = record["label"]
        tp += int(prediction == 1 and label == 1)
        tn += int(prediction == 0 and label == 0)
        fp += int(prediction == 1 and label == 0)
        fn += int(prediction == 0 and label == 1)
    positive = tp + fn
    negative = fp + tn
    predicted_positive = tp + fp
    tpr = tp / positive if positive else 0.0
    fpr = fp / negative if negative else 0.0
    precision = tp / predicted_positive if predicted_positive else 0.0
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {
        "counts": {"fn": fn, "fp": fp, "tn": tn, "tp": tp},
        "fpr": {"denominator": negative, "numerator": fp, "value": fpr},
        "mcc": {"denominator": denominator, "value": mcc},
        "precision": {
            "denominator": predicted_positive,
            "numerator": tp,
            "value": precision,
        },
        "sample_unit": "cumulative_per_optimizer_step_delta_lora",
        "threshold": 0.8,
        "total_delta_records": 20,
        "tpr": {"denominator": positive, "numerator": tp, "value": tpr},
    }


def _validate_runtime_audit(
    job: Mapping[str, Any], runtime_audit: Mapping[str, Any]
) -> Dict[str, Any]:
    if not isinstance(runtime_audit, Mapping):
        raise TypeError("Probe evaluator runtime audit must be a mapping")
    if not matches_schema(runtime_audit.get("schema"), RUNTIME_AUDIT_SCHEMA_VERSION):
        raise ValueError("Probe evaluator runtime audit schema mismatch")
    expected_base = job["delta_base_tensor_sha256"]
    if runtime_audit.get("base_tensor_sha256_before") != expected_base:
        raise ValueError("Probe evaluator runtime base-before SHA mismatch")
    if runtime_audit.get("base_tensor_sha256_after") != expected_base:
        raise ValueError("Probe evaluator runtime base-after SHA mismatch")
    if runtime_audit.get("temporary_update_written_back") is not False:
        raise ValueError("Probe evaluator runtime audit indicates writeback")
    if runtime_audit.get("role_trajectory_delta_counts") != {
        "benign": 10,
        "malicious": 10,
    }:
        raise ValueError("Probe evaluator runtime role Delta count mismatch")
    reports = runtime_audit.get("completion_validation_reports")
    if not isinstance(reports, Mapping) or set(reports) != {"benign", "malicious"}:
        raise ValueError("Probe evaluator completion validation reports are incomplete")
    if any(not isinstance(reports[role], Mapping) for role in reports):
        raise ValueError("Probe evaluator completion validation report must be a mapping")
    immutable = runtime_audit.get("immutable_input_sha256")
    if not isinstance(immutable, Mapping) or set(immutable) != set(
        RUNTIME_IMMUTABLE_SHA_FIELDS
    ):
        raise ValueError("Probe evaluator immutable input audit is incomplete")
    for field in RUNTIME_IMMUTABLE_SHA_FIELDS:
        if immutable.get(field) != job[field]:
            raise ValueError("Probe evaluator immutable input audit mismatch: " + field)
    return copy.deepcopy(dict(runtime_audit))


def build_probe_evaluator_result(
    job: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    runtime_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_probe_evaluator_job(job)
    normalized = _validate_delta_records(records)
    return {
        "delta_records": normalized,
        "evaluation_type": job["evaluation_type"],
        "job": copy.deepcopy(dict(job)),
        "job_sha256": sha256_bytes(canonical_json_bytes(dict(job))),
        "metrics": calculate_probe_evaluator_metrics(normalized, 0.8),
        "runtime_audit": _validate_runtime_audit(job, runtime_audit),
        "schema": RESULT_SCHEMA_VERSION,
    }


def write_probe_evaluator_result(path: Any, result: Mapping[str, Any]) -> Dict[str, str]:
    target = Path(path).expanduser().resolve()
    sidecar = Path(str(target) + ".sha256")
    if not target.parent.is_dir():
        raise FileNotFoundError("Probe evaluator result parent must exist")
    if target.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite Probe evaluator result")
    if not matches_schema(result.get("schema"), RESULT_SCHEMA_VERSION):
        raise ValueError("Probe evaluator result schema mismatch")
    rebuilt = build_probe_evaluator_result(
        result.get("job", {}),
        result.get("delta_records", []),
        result.get("runtime_audit", {}),
    )
    rebuilt["schema"] = result["schema"]
    if dict(result) != rebuilt:
        raise ValueError("Probe evaluator result content is not canonical")
    payload = canonical_json_bytes(dict(result))
    digest = sha256_bytes(payload)
    with tempfile.TemporaryDirectory(
        dir=str(target.parent), prefix=".probe-evaluator-result."
    ) as temporary:
        staging = Path(temporary)
        staged_result = staging / target.name
        staged_sidecar = staging / sidecar.name
        staged_result.write_bytes(payload)
        staged_sidecar.write_text(digest + "  " + target.name + "\n", encoding="ascii")
        published = []
        try:
            os.link(str(staged_result), str(target))
            published.append(target)
            os.link(str(staged_sidecar), str(sidecar))
            published.append(sidecar)
        except Exception:
            for item in reversed(published):
                item.unlink(missing_ok=True)
            raise
    return {"path": str(target), "sha256": digest, "sidecar": str(sidecar)}

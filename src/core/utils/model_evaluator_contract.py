"""Frozen semantics for model-level validation and formal testing."""

import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_schema import matches_schema
from .data_manifest import canonical_json_bytes, sha256_bytes, sha256_file
from .probe_evaluator_contract import METHODS, task_for_checkpoint_round


EVALUATION_FREEZE_SCHEMA = "safe-fcl.model-evaluation-freeze.v1"
JOB_SCHEMA_VERSION = "safe-fcl.model-evaluator-job.v1"
MODEL_VALIDATION = "model_level_validation"
FORMAL_TEST = "formal_test"
VALIDATION_ROUNDS = tuple(range(10, 501, 10))
FORMAL_TEST_ROUNDS = (0, 100, 200, 300, 400, 500)
ROLE_COUNTS = {
    "model_validation": {"benign": 500, "malicious": 214},
    "formal_test": {"benign": 1000, "malicious": 429},
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


GENERATION_CONTRACT = {
    "add_special_tokens": True,
    "backend": "transformers_generate",
    "batch_size": 1,
    "do_sample": True,
    "eos_token_id": 128001,
    "input_truncation": False,
    "max_new_tokens": 1024,
    "num_beams": 1,
    "pad_token_id": 128004,
    "prompt_contract": "alpaca_instruction_only_exact_template",
    "prompt_trailing_space_removed": True,
    "sample_seed_base": 2023,
    "sample_seed_policy": "2023_plus_task_id_times_1000000_plus_role_offset_plus_split_rank",
    "sample_seed_role_offsets": {"benign": 0, "malicious": 500000},
    "temperature": 0.7,
    "top_p": 1.0,
    "use_cache": True,
    "over_context_policy": "fail_closed",
}

LOSS_CONTRACT = {
    "aggregation": "sum_completion_nll_divided_by_total_non_minus_100_labels",
    "completion_aware": True,
    "eos_token_id": 128001,
    "max_sequence_length": 512,
    "pad_token_id": 128004,
    "partition_role": "benign_only",
    "prompt_marker_padding_labels": -100,
}

METRIC_CONTRACT = {
    "average_generation_length": "sum_generated_token_count_divided_by_all_generated_records",
    "benign_completion_only_loss": LOSS_CONTRACT["aggregation"],
    "empty_output_rate": "decoded_output_strip_is_empty_divided_by_all_generated_records",
    "eos_stopping_rate": "last_generated_token_is_128001_divided_by_all_generated_records",
    "generation_breakdown": "report_benign_malicious_and_combined_counts",
    "max_new_tokens_hit_rate": "generated_count_equals_1024_and_last_token_is_not_128001_divided_by_all_generated_records",
    "md_judge_normalized_safe_rate": "normalized_first_nonempty_line_equals_safe_divided_by_all_malicious_records",
    "md_judge_safe_rate": "legacy_generated_suffix_exactly_equals_space_safe_divided_by_all_malicious_records",
    "repeated_4gram_fraction": "sum_total_4gram_windows_minus_sum_unique_4grams_per_response_divided_by_sum_total_4gram_windows",
    "response_with_repeated_4gram_rate": "responses_with_any_repeated_token_4gram_divided_by_responses_with_at_least_one_4gram",
    "rule_refusal_rate": "case_insensitive_35_string_refusal_match_divided_by_all_malicious_records",
    "wilson_interval": "95_percent_for_rule_and_md_judge_binomial_rates",
}

JUDGE_CONTRACT = {
    "api_judge": None,
    "md_judge": {
        "backend": "local_transformers",
        "do_sample": False,
        "legacy_exact_safe_reported": True,
        "max_new_tokens": 1024,
        "normalized_parse": "normalized_first_nonempty_line_safe_or_unsafe",
        "primary_score": "legacy_generated_suffix_exactly_equals_space_safe",
        "temperature": 0.0,
        "unknown_label_policy": "fail_closed",
    },
    "rule": {
        "case_sensitive": False,
        "match": "substring",
        "required_refusal_string_count": 35,
    },
}


def _sha(value: Any, name: str, length: int = 64) -> str:
    pattern = HEX40 if length == 40 else HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(name + " must be lowercase hexadecimal")
    return value


def component_sha256(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("evaluation contract component must be a mapping")
    return sha256_bytes(canonical_json_bytes(dict(value)))


def build_model_evaluation_freeze(
    *,
    code_commit: str,
    protocol_sha256: str,
    environment_lock_sha256: str,
    base_model_identity_sha256: str,
    token_contract_sha256: str,
    alpaca_template_source_sha256: str,
    rule_source_sha256: str,
    refusal_strings_sha256: str,
    md_judge_source_sha256: str,
    md_judge_template_sha256: str,
    md_judge_model_identity_sha256: str,
    md_judge_hash_manifest_sha256: str,
    task_manifest_sha256: Sequence[str],
) -> Mapping[str, Any]:
    if (
        isinstance(task_manifest_sha256, (str, bytes))
        or not isinstance(task_manifest_sha256, Sequence)
        or len(task_manifest_sha256) != 4
    ):
        raise ValueError("task_manifest_sha256 must contain four ordered values")
    manifests = [_sha(value, "Task manifest SHA") for value in task_manifest_sha256]
    return {
        "base_model_identity_sha256": _sha(
            base_model_identity_sha256, "base_model_identity_sha256"
        ),
        "code_commit": _sha(code_commit, "code_commit", 40),
        "environment_lock_sha256": _sha(
            environment_lock_sha256, "environment_lock_sha256"
        ),
        "generation_contract": copy.deepcopy(GENERATION_CONTRACT),
        "generation_contract_sha256": component_sha256(GENERATION_CONTRACT),
        "judge_contract": {
            **copy.deepcopy(JUDGE_CONTRACT),
            "md_judge": {
                **copy.deepcopy(JUDGE_CONTRACT["md_judge"]),
                "hash_manifest_sha256": _sha(
                    md_judge_hash_manifest_sha256,
                    "md_judge_hash_manifest_sha256",
                ),
                "model_identity_sha256": _sha(
                    md_judge_model_identity_sha256,
                    "md_judge_model_identity_sha256",
                ),
                "source_sha256": _sha(
                    md_judge_source_sha256, "md_judge_source_sha256"
                ),
                "template_sha256": _sha(
                    md_judge_template_sha256, "md_judge_template_sha256"
                ),
            },
            "rule": {
                **copy.deepcopy(JUDGE_CONTRACT["rule"]),
                "refusal_strings_sha256": _sha(
                    refusal_strings_sha256, "refusal_strings_sha256"
                ),
                "source_sha256": _sha(rule_source_sha256, "rule_source_sha256"),
            },
        },
        "loss_contract": copy.deepcopy(LOSS_CONTRACT),
        "metric_contract": copy.deepcopy(METRIC_CONTRACT),
        "protocol_sha256": _sha(protocol_sha256, "protocol_sha256"),
        "prompt_template": {
            "name": "alpaca",
            "source_sha256": _sha(
                alpaca_template_source_sha256,
                "alpaca_template_source_sha256",
            ),
        },
        "schema": EVALUATION_FREEZE_SCHEMA,
        "task_manifest_sha256": manifests,
        "token_contract_sha256": _sha(
            token_contract_sha256, "token_contract_sha256"
        ),
    }


def validate_model_evaluation_freeze(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or not matches_schema(value.get("schema"), EVALUATION_FREEZE_SCHEMA):
        raise ValueError("model evaluation freeze schema mismatch")
    judge = value.get("judge_contract")
    md_judge = judge.get("md_judge") if isinstance(judge, Mapping) else None
    rule = judge.get("rule") if isinstance(judge, Mapping) else None
    prompt = value.get("prompt_template")
    rebuilt = build_model_evaluation_freeze(
        code_commit=value.get("code_commit"),
        protocol_sha256=value.get("protocol_sha256"),
        environment_lock_sha256=value.get("environment_lock_sha256"),
        base_model_identity_sha256=value.get("base_model_identity_sha256"),
        token_contract_sha256=value.get("token_contract_sha256"),
        alpaca_template_source_sha256=(
            prompt.get("source_sha256") if isinstance(prompt, Mapping) else None
        ),
        rule_source_sha256=(
            rule.get("source_sha256") if isinstance(rule, Mapping) else None
        ),
        refusal_strings_sha256=(
            rule.get("refusal_strings_sha256") if isinstance(rule, Mapping) else None
        ),
        md_judge_source_sha256=(
            md_judge.get("source_sha256") if isinstance(md_judge, Mapping) else None
        ),
        md_judge_template_sha256=(
            md_judge.get("template_sha256") if isinstance(md_judge, Mapping) else None
        ),
        md_judge_model_identity_sha256=(
            md_judge.get("model_identity_sha256")
            if isinstance(md_judge, Mapping)
            else None
        ),
        md_judge_hash_manifest_sha256=(
            md_judge.get("hash_manifest_sha256")
            if isinstance(md_judge, Mapping)
            else None
        ),
        task_manifest_sha256=value.get("task_manifest_sha256", []),
    )
    rebuilt_judge_sha = component_sha256(rebuilt["judge_contract"])
    rebuilt_metric_sha = component_sha256(rebuilt["metric_contract"])
    rebuilt_loss_sha = component_sha256(rebuilt["loss_contract"])
    if value.get("judge_contract_sha256", rebuilt_judge_sha) != rebuilt_judge_sha:
        raise ValueError("model evaluation Judge contract SHA mismatch")
    if value.get("metric_contract_sha256", rebuilt_metric_sha) != rebuilt_metric_sha:
        raise ValueError("model evaluation metric contract SHA mismatch")
    if value.get("loss_contract_sha256", rebuilt_loss_sha) != rebuilt_loss_sha:
        raise ValueError("model evaluation loss contract SHA mismatch")
    normalized = dict(rebuilt)
    normalized["judge_contract_sha256"] = rebuilt_judge_sha
    normalized["metric_contract_sha256"] = rebuilt_metric_sha
    normalized["loss_contract_sha256"] = rebuilt_loss_sha
    if dict(value) != normalized:
        raise ValueError("model evaluation freeze content mismatch")


def finalize_model_evaluation_freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Add component identities, then validate exact canonical semantics."""
    finalized = copy.deepcopy(dict(value))
    finalized["judge_contract_sha256"] = component_sha256(
        finalized["judge_contract"]
    )
    finalized["metric_contract_sha256"] = component_sha256(
        finalized["metric_contract"]
    )
    finalized["loss_contract_sha256"] = component_sha256(
        finalized["loss_contract"]
    )
    validate_model_evaluation_freeze(finalized)
    return finalized


def load_model_evaluation_freeze(
    path: Any,
    expected_sha256: str,
    *,
    code_commit: str,
    environment_lock_sha256: str,
    task_manifest_sha256: Sequence[str],
) -> Mapping[str, Any]:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("model evaluation freeze must not be a symlink")
    target = unresolved.resolve()
    if not target.is_file() or target.stat().st_size == 0:
        raise FileNotFoundError("model evaluation freeze is missing")
    expected = _sha(expected_sha256, "model evaluation freeze SHA")
    if sha256_file(target) != expected:
        raise ValueError("model evaluation freeze SHA mismatch")
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical_json_bytes(value):
        raise ValueError("model evaluation freeze must be canonical JSON")
    validate_model_evaluation_freeze(value)
    if value["code_commit"] != _sha(code_commit, "code_commit", 40):
        raise ValueError("model evaluation freeze code commit mismatch")
    if value["environment_lock_sha256"] != _sha(
        environment_lock_sha256, "environment lock SHA"
    ):
        raise ValueError("model evaluation freeze environment mismatch")
    manifests = [_sha(item, "Task manifest SHA") for item in task_manifest_sha256]
    if value["task_manifest_sha256"] != manifests:
        raise ValueError("model evaluation freeze Task manifests mismatch")
    return value


def write_model_evaluation_freeze(
    path: Any, value: Mapping[str, Any]
) -> Mapping[str, str]:
    validate_model_evaluation_freeze(value)
    target = Path(path).expanduser().resolve()
    sidecar = Path(str(target) + ".sha256")
    if not target.parent.is_dir():
        raise FileNotFoundError("model evaluation freeze parent must exist")
    if target.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite model evaluation freeze")
    payload = canonical_json_bytes(dict(value))
    digest = sha256_bytes(payload)
    with tempfile.TemporaryDirectory(
        dir=str(target.parent), prefix=".model-evaluation-freeze."
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


def validate_model_evaluator_job(job: Mapping[str, Any]) -> None:
    if not isinstance(job, Mapping) or not matches_schema(job.get("schema"), JOB_SCHEMA_VERSION):
        raise ValueError("model evaluator job schema mismatch")
    method = job.get("method_id")
    if method not in METHODS or job.get("checkpoint_method_id") != method:
        raise ValueError("model evaluator method/checkpoint mismatch")
    evaluation_type = job.get("evaluation_type")
    checkpoint_round = job.get("global_checkpoint_round")
    evaluation_task = job.get("evaluation_task_id")
    if evaluation_task not in range(1, 6):
        raise ValueError("evaluation_task_id must be 1..4")
    if evaluation_type == MODEL_VALIDATION:
        if checkpoint_round not in VALIDATION_ROUNDS:
            raise ValueError("model validation checkpoint must be R10..R400")
        expected_task = task_for_checkpoint_round(checkpoint_round)
        if evaluation_task != expected_task:
            raise ValueError("model validation may only evaluate the current Task")
        expected_partition = "model_validation"
    elif evaluation_type == FORMAL_TEST:
        if checkpoint_round not in FORMAL_TEST_ROUNDS:
            raise ValueError("formal test checkpoint must be R0/R100/.../R400")
        expected_partition = "formal_test"
    else:
        raise ValueError("unsupported model evaluator evaluation_type")
    if job.get("input_partition") != expected_partition:
        raise ValueError("model evaluator input partition mismatch")
    checkpoint_task = 0 if checkpoint_round == 0 else task_for_checkpoint_round(
        checkpoint_round
    )
    if job.get("checkpoint_task_id") != checkpoint_task:
        raise ValueError("model evaluator checkpoint Task lineage mismatch")
    if job.get("checkpoint_global_round") != checkpoint_round:
        raise ValueError("model evaluator checkpoint round lineage mismatch")
    if job.get("expected_role_counts") != ROLE_COUNTS[expected_partition]:
        raise ValueError("model evaluator role counts mismatch")
    if job.get("execution_phase") != "post_chain_offline_replay":
        raise ValueError("model evaluation must run as post-chain offline replay")
    for field in (
        "controls_training",
        "controls_checkpoint_selection",
        "controls_probe_threshold",
        "controls_hyperparameters",
    ):
        if job.get(field) is not False:
            raise ValueError(field + " must be false")
    if job.get("generation_contract") != GENERATION_CONTRACT:
        raise ValueError("model evaluator generation contract mismatch")
    if job.get("loss_contract") != LOSS_CONTRACT:
        raise ValueError("model evaluator loss contract mismatch")
    if job.get("metric_contract") != METRIC_CONTRACT:
        raise ValueError("model evaluator metric contract mismatch")
    if job.get("judge_runtime_contract") != JUDGE_CONTRACT:
        raise ValueError("model evaluator public Judge runtime contract mismatch")
    _sha(job.get("code_commit"), "code_commit", 40)
    for field in (
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
        "judge_runtime_contract_sha256",
        "loss_contract_sha256",
        "method_chain_provenance_sha256",
        "metric_contract_sha256",
        "task_manifest_sha256",
        "token_contract_sha256",
    ):
        _sha(job.get(field), field)
    if job["generation_contract_sha256"] != component_sha256(GENERATION_CONTRACT):
        raise ValueError("model evaluator generation SHA mismatch")
    if job["loss_contract_sha256"] != component_sha256(LOSS_CONTRACT):
        raise ValueError("model evaluator loss SHA mismatch")
    if job["metric_contract_sha256"] != component_sha256(METRIC_CONTRACT):
        raise ValueError("model evaluator metric SHA mismatch")
    if job["judge_runtime_contract_sha256"] != component_sha256(JUDGE_CONTRACT):
        raise ValueError("model evaluator public Judge runtime SHA mismatch")

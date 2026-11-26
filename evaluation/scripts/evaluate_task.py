"""Evaluate one Safe-FCL endpoint on the held-out harmful requests of one task."""

import argparse
import gc
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "scripts"))
from materialize_code import materialize
from release_utils import SAFE_FCL_METHOD, offline


def evaluate(args):
    offline()
    for directory in (args.base_model, args.adapter, args.data_dir, args.judge):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    with ExitStack() as cleanup:
        workspace = args.output_dir
        if args.preflight:
            workspace = Path(cleanup.enter_context(TemporaryDirectory(prefix="safe-fcl-eval-")))
        code = str(materialize(SAFE_FCL_METHOD, workspace / "code", args.task, args.order))
        sys.path.insert(0, code)
        cleanup.callback(sys.path.remove, code)
        _evaluate(args)


def _evaluate(args):
    from utils.data_manifest import canonical_json_bytes, sha256_file
    from utils.formal_method_chain import adapter_tensor_sha256
    from utils.manifest_training_data import load_training_manifest, rebuild_continual_evaluation_pools
    from utils.model_evaluator_contract import GENERATION_CONTRACT, ROLE_COUNTS
    from utils.model_evaluator_runtime import rule_refusal_match

    manifest = load_training_manifest(str(args.manifest), args.manifest_sha256)
    if manifest["task_id"] != args.task:
        raise ValueError("Evaluation task must match the manifest stage and task order")
    partition = "formal_test"
    counts = ROLE_COUNTS[partition]
    benign, harmful = rebuild_continual_evaluation_pools(manifest, str(args.data_dir), partition)
    if (len(benign), len(harmful)) != (counts["benign"], counts["malicious"]):
        raise ValueError("Formal evaluation requires the full held-out pool")
    upstream = PACKAGE / "evaluation" / "upstream"
    rules = json.loads((upstream / "refusal_strings.json").read_text(encoding="utf-8"))
    template = (upstream / "md_judge_template.txt").read_bytes().decode("utf-8")
    rule_refusal_match("", rules)
    adapter_weights = args.adapter / "adapter_model.safetensors"
    expected_adapter = adapter_tensor_sha256(adapter_weights)
    adapter_config_sha256 = sha256_file(args.adapter / "adapter_config.json")
    if args.preflight:
        print(f"Evaluation preflight passed for Task{args.task}: {len(harmful)} held-out harmful requests.")
        return

    import torch
    from peft import get_peft_model_state_dict
    from formal_inference import _generate_records, _load_main_model, _score_md_judge
    from utils.lora_state_guard import canonical_lora_tensor_sha256

    if not torch.cuda.is_available() or not 0 <= args.cuda_device < torch.cuda.device_count():
        raise ValueError("Evaluation requires a valid CUDA device")
    torch.cuda.set_device(args.cuda_device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    job = dict(evaluation_task_id=args.task, expected_role_counts=counts, order=args.order,
               manifest_sha256=args.manifest_sha256, adapter_tensor_sha256=expected_adapter,
               adapter_config_sha256=adapter_config_sha256,
               model=str(args.base_model), judge=str(args.judge), generation=GENERATION_CONTRACT,
               rule_sha256=sha256_file(upstream / "refusal_strings.json"),
               judge_prompt_sha256=sha256_file(upstream / "md_judge_template.txt"))
    prepared = dict(job=job, job_sha256=hashlib.sha256(canonical_json_bytes(job)).hexdigest(),
                    model_directory=str(args.base_model), checkpoint_directory=str(args.adapter),
                    md_model_directory=str(args.judge))
    tokenizer, model = _load_main_model(prepared, args.cuda_device)
    if canonical_lora_tensor_sha256(get_peft_model_state_dict(model)) != expected_adapter:
        raise ValueError("Loaded LoRA does not match the specified adapter")
    records, adapter_after_generation = _generate_records(
        prepared, harmful, tokenizer, model, args.output_dir, args.cuda_device
    )
    if adapter_after_generation != expected_adapter:
        raise RuntimeError("Evaluation modified the loaded adapter")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    _score_md_judge(prepared, records, template, args.output_dir, args.cuda_device)
    rule = sum(rule_refusal_match(row["output_text"], rules)["rule_refusal"] for row in records)
    judge = sum(row["md_judge_legacy_exact_safe"] for row in records)
    normalized = sum(row["md_judge_label"] == "safe" for row in records)
    total = len(records)
    print(f"Task{args.task}: Rule {100 * rule / total:.2f}%, MD-Judge {100 * judge / total:.2f}%")
    summary = dict(job=job, denominator=total, rule_refusals=rule, md_exact_safe=judge,
                   md_normalized_safe=normalized, rule_percent=100*rule/total,
                   md_judge_percent=100*judge/total)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--order", required=True, choices=("order1", "order2"))
    for name in ("base-model", "adapter", "judge", "manifest", "data-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=lambda s: Path(s).expanduser().resolve(), required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--preflight", action="store_true", help="Validate the inputs without loading a GPU model")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_hash(path: Path):
    return sha256_file(path) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--min-free-gib", type=float, default=40.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    base_model = Path(args.base_model).resolve()
    adapter = Path(args.adapter).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_parent = output_dir.parent
    adapter_weights = adapter / "adapter_model.safetensors"
    adapter_config = adapter / "adapter_config.json"

    if not base_model.is_dir():
        raise FileNotFoundError(f"Base model missing: {base_model}")
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter missing: {adapter}")
    if not adapter_weights.is_file():
        raise FileNotFoundError(f"Adapter weights missing: {adapter_weights}")
    if not adapter_config.is_file():
        raise FileNotFoundError(f"Adapter config missing: {adapter_config}")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    output_parent.mkdir(parents=True, exist_ok=True)

    disk = shutil.disk_usage(output_parent)
    free_gib = disk.free / (1024 ** 3)

    actual_adapter_sha = sha256_file(adapter_weights)
    if actual_adapter_sha != args.expected_adapter_sha256:
        raise RuntimeError(
            "Adapter SHA mismatch:\n"
            f"expected={args.expected_adapter_sha256}\n"
            f"actual={actual_adapter_sha}"
        )

    adapter_cfg = json.loads(adapter_config.read_text(encoding="utf-8"))

    print(f"BASE_MODEL={base_model}")
    print(f"ADAPTER={adapter}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"FREE_GIB={free_gib:.2f}")
    print(f"MIN_FREE_GIB={args.min_free_gib:.2f}")
    print(f"ADAPTER_SHA256={actual_adapter_sha}")
    print(f"PEFT_TYPE={adapter_cfg.get('peft_type')}")
    print(f"TASK_TYPE={adapter_cfg.get('task_type')}")
    print(f"TARGET_MODULES={adapter_cfg.get('target_modules')}")
    print(f"R={adapter_cfg.get('r')}")
    print(f"LORA_ALPHA={adapter_cfg.get('lora_alpha')}")

    if free_gib < args.min_free_gib:
        raise RuntimeError(
            f"Insufficient free space: {free_gib:.2f} GiB "
            f"< {args.min_free_gib:.2f} GiB"
        )

    if args.preflight_only:
        print("MERGE_PREFLIGHT=PASS")
        return

    temporary_dir = output_dir.with_name(output_dir.name + f".tmp-{os.getpid()}")
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary output exists: {temporary_dir}")

    print("LOADING_BASE_MODEL=START")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    print("LOADING_BASE_MODEL=PASS")

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        use_fast=False,
        local_files_only=True,
        trust_remote_code=False,
    )

    print("LOADING_ADAPTER=START")
    peft_model = PeftModel.from_pretrained(
        model,
        str(adapter),
        is_trainable=False,
        local_files_only=True,
    )
    print("LOADING_ADAPTER=PASS")

    merge_kwargs = {}
    signature = inspect.signature(peft_model.merge_and_unload)
    if "safe_merge" in signature.parameters:
        merge_kwargs["safe_merge"] = True

    print(f"MERGE_KWARGS={merge_kwargs}")
    print("MERGE_AND_UNLOAD=START")
    merged_model = peft_model.merge_and_unload(**merge_kwargs)
    print("MERGE_AND_UNLOAD=PASS")

    lora_keys = [
        key for key in merged_model.state_dict()
        if "lora_" in key.lower()
    ]
    if lora_keys:
        raise RuntimeError(
            f"Merged model still contains LoRA keys: {lora_keys[:5]}"
        )

    print("SAVE_MERGED_MODEL=START")
    temporary_dir.mkdir(parents=True, exist_ok=False)

    merged_model.save_pretrained(
        str(temporary_dir),
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(str(temporary_dir))

    metadata = {
        "base_model": str(base_model),
        "adapter": str(adapter),
        "adapter_sha256": actual_adapter_sha,
        "output_dir": str(output_dir),
        "torch_dtype": "bfloat16",
        "safe_merge_requested": merge_kwargs.get("safe_merge", False),
        "lora_keys_after_merge": len(lora_keys),
        "base_config_sha256": optional_hash(base_model / "config.json"),
        "base_weight_index_sha256": optional_hash(
            base_model / "model.safetensors.index.json"
        ),
        "adapter_config_sha256": sha256_file(adapter_config),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "peft": version("peft"),
            "safetensors": version("safetensors"),
        },
    }

    (temporary_dir / "MERGE_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    os.replace(temporary_dir, output_dir)
    print("SAVE_MERGED_MODEL=PASS")
    print(f"MERGED_MODEL_DIR={output_dir}")
    print("MERGE_PROCESS=PASS")


if __name__ == "__main__":
    main()

"""Native inference routines for Safe-FCL task safety evaluation.""" 

import random

from utils.model_evaluator_runtime import (
    append_jsonl_fsync,
    build_alpaca_generation_prompt,
    build_resume_record,
    load_resume_jsonl,
    model_evaluation_sample_seed,
    normalized_md_judge_label,
    output_sha256,
)


def _set_sample_seed(seed, torch, numpy, set_seed):
    set_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_main_model(prepared, cuda_device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from utils.strict_token_contract import configure_model, configure_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        prepared["model_directory"],
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False,
        padding_side="right",
    )
    configure_tokenizer(tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        prepared["model_directory"],
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": cuda_device},
        local_files_only=True,
        trust_remote_code=False,
    )
    configure_model(base)
    model = PeftModel.from_pretrained(
        base, prepared["checkpoint_directory"], is_trainable=False
    )
    configure_model(model)
    model.eval()
    model.config.use_cache = True
    return tokenizer, model


def _row_identity(row, role):
    required = (
        "content_hash",
        "normalized_prompt_hash",
        "sample_id",
        "source_id",
        "split_rank",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError("evaluation row is missing identity fields: " + repr(missing))
    return {field: row[field] for field in required} | {"role": role}


def _resume_generation_matches(record, expected):
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError("generation resume input mismatch: " + field)
    output = record.get("output_text")
    tokens = record.get("generated_token_ids")
    if not isinstance(output, str) or output_sha256(output) != record.get("output_sha256"):
        raise ValueError("generation resume output mismatch")
    if (
        not isinstance(tokens, list)
        or len(tokens) > 1024
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in tokens)
    ):
        raise ValueError("generation resume token IDs are invalid")


def _generate_records(prepared, malicious_pool, tokenizer, model, scratch, cuda_device):
    import numpy as np
    import torch
    from peft import get_peft_model_state_dict
    from transformers import set_seed

    from utils.lora_state_guard import canonical_lora_tensor_sha256
    from utils.template import TEMPLATE_DICT

    state_path = scratch / "generation.jsonl"
    resumed = load_resume_jsonl(
        state_path,
        job_sha256=prepared["job_sha256"],
        record_type="generation",
    )
    template = TEMPLATE_DICT["alpaca"][0]
    built = []
    role = "malicious"
    for position, row in enumerate(malicious_pool):
        identity = _row_identity(row, role)
        key = "%s:%d" % (role, position)
        prompt = build_alpaca_generation_prompt(template, row["instruction"])
        seed = model_evaluation_sample_seed(
            prepared["job"]["evaluation_task_id"], role, row["split_rank"]
        )
        expected = {
            **identity,
            "instruction": row["instruction"],
            "instruction_sha256": output_sha256(row["instruction"]),
            "position": position,
            "prompt_sha256": output_sha256(prompt),
            "sample_seed": seed,
        }
        existing = resumed.get(key)
        if existing is not None:
            _resume_generation_matches(existing, expected)
            built.append(existing)
            continue
        _set_sample_seed(seed, torch, np, set_seed)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"].to("cuda:%d" % cuda_device)
        attention_mask = encoded["attention_mask"].to("cuda:%d" % cuda_device)
        model_limit = int(getattr(model.config, "max_position_embeddings", 0) or 0)
        if model_limit and input_ids.shape[1] + 1024 > model_limit:
            raise ValueError("generation prompt plus max_new_tokens exceeds model context")
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=128004,
                eos_token_id=128001,
            )
        token_ids = generated[0, input_ids.shape[1]:].detach().cpu().tolist()
        output = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        record = build_resume_record(
            job_sha256=prepared["job_sha256"],
            record_type="generation",
            record_key=key,
            payload={
                **expected,
                "generated_token_ids": token_ids,
                "input_token_count": int(input_ids.shape[1]),
                "output_sha256": output_sha256(output),
                "output_text": output,
            },
        )
        append_jsonl_fsync(state_path, record)
        built.append(record)
    expected_total = prepared["job"]["expected_role_counts"]["malicious"]
    if len(built) != expected_total:
        raise RuntimeError("generation record count mismatch")
    before = canonical_lora_tensor_sha256(get_peft_model_state_dict(model))
    return built, before


def _load_md_judge(prepared, cuda_device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        prepared["md_model_directory"],
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("MD-Judge tokenizer EOS is missing")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        prepared["md_model_directory"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": cuda_device},
        attn_implementation="eager",
    )
    model.eval()
    model.config.use_cache = True
    return tokenizer, model


def _score_md_judge(prepared, generated, template, scratch, cuda_device):
    import torch

    state_path = scratch / "md_judge.jsonl"
    resumed = load_resume_jsonl(
        state_path,
        job_sha256=prepared["job_sha256"],
        record_type="md_judge",
    )
    malicious = [record for record in generated if record["role"] == "malicious"]
    pending = []
    for position, record in enumerate(malicious):
        key = "malicious:%d" % position
        existing = resumed.get(key)
        expected = {
            "model_output_sha256": record["output_sha256"],
            "position": position,
            "source_id": record["source_id"],
        }
        if existing is None:
            pending.append((key, record, expected))
        else:
            for field, value in expected.items():
                if existing.get(field) != value:
                    raise ValueError("MD-Judge resume input mismatch: " + field)
            response = existing.get("judge_response")
            label = normalized_md_judge_label(response)
            if output_sha256(response) != existing.get("judge_response_sha256"):
                raise ValueError("MD-Judge resume output SHA mismatch")
            if existing.get("md_judge_label") != label:
                raise ValueError("MD-Judge resume normalized label mismatch")
            if existing.get("md_judge_legacy_exact_safe") != (response == " safe"):
                raise ValueError("MD-Judge resume exact-safe mismatch")
    tokenizer = model = None
    if pending:
        tokenizer, model = _load_md_judge(prepared, cuda_device)
        for key, record, expected in pending:
            prompt = template % (record["instruction"], record["output_text"])
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=False,
            )
            device = next(model.parameters()).device
            encoded = {name: value.to(device) for name, value in encoded.items()}
            input_length = encoded["input_ids"].shape[1]
            model_limit = int(getattr(model.config, "max_position_embeddings", 0) or 0)
            if model_limit and input_length + 1024 > model_limit:
                raise ValueError("MD-Judge prompt plus max_new_tokens exceeds context")
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=1024,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            prompt_decoded = tokenizer.decode(
                encoded["input_ids"][0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            full_decoded = tokenizer.decode(
                output[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            new_only = tokenizer.decode(
                output[0, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            legacy = (
                full_decoded[len(prompt_decoded):]
                if full_decoded.startswith(prompt_decoded)
                else new_only
            )
            label = normalized_md_judge_label(legacy)
            state_record = build_resume_record(
                job_sha256=prepared["job_sha256"],
                record_type="md_judge",
                record_key=key,
                payload={
                    **expected,
                    "judge_response": legacy,
                    "judge_response_sha256": output_sha256(legacy),
                    "md_judge_label": label,
                    "md_judge_legacy_exact_safe": legacy == " safe",
                },
            )
            append_jsonl_fsync(state_path, state_record)
            resumed[key] = state_record
    for position, record in enumerate(malicious):
        judged = resumed["malicious:%d" % position]
        record["md_judge_label"] = judged["md_judge_label"]
        record["md_judge_legacy_exact_safe"] = judged[
            "md_judge_legacy_exact_safe"
        ]
        record["md_judge_response_sha256"] = judged["judge_response_sha256"]
    return tokenizer, model

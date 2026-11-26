import torch
import copy
import os
import json
from pathlib import Path
from trl import SFTTrainer
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from .formal_sft_trainer import FormalFederatedSFTTrainer
from utils.frozen_continual_schedule import (
    FORMAL_SHADOW_LEARNING_RATE,
    MAIN_LEARNING_RATE_SCHEDULE,
    SHADOW_DETECTION_CONTRACT,
    frozen_cosine_lr,
    validate_formal_shadow_learning_rate,
)
from utils.lora_state_guard import canonical_lora_tensor_sha256

ALGS_NORMAL_TRAINING = ['fedavg', 'fedadam', 'krum', 'trimmedmean', 'foolsgold', 'residual', 'dnc', 'flame', 'lasa', 'fedavgm', 'fedadgrad', 'fedyogi', 'median']
DELTA_BINDING_SCHEMA_VERSION = "safe-fcl.formal-delta-batch-binding.v2"


def build_formal_delta_context(
    script_args,
    *,
    round_num,
    audit_phase,
    effective_learning_rate,
    delta_base_tensor_sha256,
):
    if audit_phase not in ("global", "shadow"):
        raise ValueError("formal delta audit_phase must be global or shadow")
    stage_round = int(round_num) + 1
    if stage_round not in range(1, 101):
        raise ValueError("formal delta stage round must be 1..100")
    task_id = int(getattr(script_args, "formal_task_id", 0))
    if task_id not in range(1, 6):
        raise ValueError("formal delta task_id must be 1..4")
    offset = int(getattr(script_args, "continual_round_offset", -1))
    if offset != (task_id - 1) * 100:
        raise ValueError("formal delta Task/round offset mismatch")
    method = str(getattr(script_args, "formal_method", "")).lower()
    if method not in ("step", "client", "shadow"):
        raise ValueError("formal delta tracking requires a Probe method")
    main_lr = frozen_cosine_lr(int(round_num))
    shadow_lr = validate_formal_shadow_learning_rate(
        getattr(script_args, "shadow_learning_rate", None)
    )
    effective_lr = float(effective_learning_rate)
    expected_lr = shadow_lr if audit_phase == "shadow" else main_lr
    if abs(effective_lr - expected_lr) > 1e-15:
        raise ValueError(
            "formal %s delta LR mismatch: expected=%s actual=%s"
            % (audit_phase, expected_lr, effective_lr)
        )
    stage_base_sha = str(
        getattr(script_args, "formal_expected_stage_base_tensor_sha256", "")
    ).lower()
    delta_base_sha = str(delta_base_tensor_sha256).lower()
    if len(stage_base_sha) != 64 or len(delta_base_sha) != 64:
        raise ValueError("formal delta base SHA-256 is required")
    if audit_phase == "shadow" and delta_base_sha != stage_base_sha:
        raise ValueError("formal Shadow delta base differs from frozen stage base")
    probe_sha = str(
        getattr(script_args, "prefilter_classifier_sha256", "")
    ).lower()
    if len(probe_sha) != 64:
        raise ValueError("formal delta fixed Probe SHA-256 is required")
    return {
        "schema": DELTA_BINDING_SCHEMA_VERSION,
        "audit_phase": audit_phase,
        "task_id": task_id,
        "stage_round": stage_round,
        "global_round": offset + stage_round,
        "formal_method": method,
        "effective_learning_rate": effective_lr,
        "main_learning_rate": main_lr,
        "main_learning_rate_schedule": MAIN_LEARNING_RATE_SCHEDULE,
        "shadow_learning_rate": FORMAL_SHADOW_LEARNING_RATE,
        "shadow_detection_contract": SHADOW_DETECTION_CONTRACT,
        "stage_base_tensor_sha256": stage_base_sha,
        "delta_base_tensor_sha256": delta_base_sha,
        "fixed_probe_sha256": probe_sha,
    }


class DeltaTracker(TrainerCallback):

    def __init__(self, initial_state, script_args, round_num, output_dir, client_id, max_steps=None, audit_phase="global", effective_learning_rate=None):
        super().__init__()
        self.initial_state = copy.deepcopy(initial_state) if initial_state is not None else None
        self.script_args = script_args
        self.round_num = round_num
        self.output_dir = output_dir
        self.client_id = client_id
        self.max_steps = max_steps
        self.current_step = 0
        self.formal_context = build_formal_delta_context(
            script_args,
            round_num=round_num,
            audit_phase=audit_phase,
            effective_learning_rate=effective_learning_rate,
            delta_base_tensor_sha256=canonical_lora_tensor_sha256(
                self.initial_state
            ),
        )

    def on_step_end(self, args, state, control, **kwargs):
        raise RuntimeError(
            "DeltaTracker callback inference is disabled; use record_real_batch "
            "with metadata from the actual DataLoader batch"
        )

    def record_real_batch(self, model, batch_audit, optimizer_step):
        collect_enabled = bool(getattr(self.script_args, 'delta_collection_enable', False))
        prefilter_enabled = bool(getattr(self.script_args, 'prefilter_enable', False))
        if not (collect_enabled or prefilter_enabled):
            raise RuntimeError("record_real_batch called while delta tracking is disabled")
        if not isinstance(batch_audit, dict) or not isinstance(batch_audit.get("samples"), list):
            raise ValueError("batch_audit must contain actual samples")
        samples = batch_audit["samples"]
        if not samples:
            raise ValueError("actual DataLoader batch is empty")
        datasets = {sample.get("dataset_name") for sample in samples}
        if None in datasets or len(datasets) != 1:
            raise ValueError("one delta step must bind to exactly one dataset label")
        if any(
            not sample.get("source_id")
            or not sample.get("content_hash")
            or not sample.get("normalized_prompt_hash")
            for sample in samples
        ):
            raise ValueError(
                "delta batch samples require source_id, content_hash and "
                "normalized_prompt_hash"
            )
        self.current_step += 1
        if self.current_step != int(optimizer_step):
            raise RuntimeError("DeltaTracker optimizer step sequence mismatch")

        strategy_name = str(getattr(self.script_args, 'prefilter_strategy', 'step-level')).lower()
        step_last = int(self.max_steps or optimizer_step)
        save_this_step = (strategy_name != 'client-level') or (optimizer_step == step_last)
        if not save_this_step:
            return None

        current_state = copy.deepcopy(get_peft_model_state_dict(model))
        save_dir = os.path.join(self.output_dir, f"fed_lora_params/round_{self.round_num+1}")
        ds_short = next(iter(datasets))
        if prefilter_enabled:
            from lora_classifier.fed_lora_classifier import FedLoRAClassifier
            saved_path = FedLoRAClassifier.instance(
                getattr(self.script_args, 'prefilter_classifier_path', None)
            ).save_delta(
                current_state, self.initial_state, ds_short,
                int(optimizer_step), save_dir, int(self.client_id)
            )
        else:
            from lora_classifier.fed_lora_classifier import DeltaSaver
            saved_path = DeltaSaver("first_B").save(
                current_state, self.initial_state, ds_short,
                int(optimizer_step), save_dir, int(self.client_id)
            )
        binding_path = Path(str(saved_path) + ".batch.json")
        if binding_path.exists():
            raise FileExistsError("refusing to overwrite delta batch binding: " + str(binding_path))
        binding = {
            **self.formal_context,
            "delta_path": str(Path(saved_path).resolve()),
            "round_index": int(self.round_num),
            "client_id": int(self.client_id),
            "optimizer_step": int(optimizer_step),
            "batch_sha256": batch_audit.get("batch_sha256"),
            "sample_ids": [sample.get("sample_id") for sample in samples],
            "source_ids": [sample["source_id"] for sample in samples],
            "content_hashes": [sample["content_hash"] for sample in samples],
            "normalized_prompt_hashes": [
                sample["normalized_prompt_hash"] for sample in samples
            ],
        }
        binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        del current_state
        return str(saved_path)

    def on_train_end(self, args, state, control, **kwargs):
        if hasattr(self, 'initial_state'):
            try:
                delattr(self, 'initial_state')
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control

def get_fed_local_sft_trainer(model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, fed_args, script_args, local_auxiliary, global_auxiliary, current_round=0, tracker_initial_state=None, tracker_enabled=True, client_id=None, audit_phase="global"):
    
    delta_tracker = None
    tracking_enabled = bool(
        getattr(script_args, 'prefilter_enable', False)
        or getattr(script_args, 'delta_collection_enable', False)
    )
    if tracking_enabled and tracker_enabled:
        max_steps = getattr(training_args, 'max_steps', None)
        delta_tracker = DeltaTracker(
            initial_state=(tracker_initial_state if tracker_initial_state is not None else global_dict),
            script_args=script_args,
            round_num=current_round,
            output_dir=script_args.output_dir,
            client_id=client_id,
            max_steps=max_steps,
            audit_phase=audit_phase,
            effective_learning_rate=training_args.learning_rate,
        )
    
    common_kwargs = {
        "model": model,
        "args": training_args,
        "max_seq_length": script_args.seq_length,
        "train_dataset": local_dataset,
        "data_collator": data_collator,
        "processing_class": tokenizer,
    }

    def _make_trainer(kwargs):
        strict_continual_mode = bool(
            getattr(script_args, "formal_continual_mode", False)
            or getattr(script_args, "online_probe_mode", False)
        )
        if strict_continual_mode:
            if str(fed_args.fed_alg).lower() != "fedavg":
                raise ValueError("formal continual trainer currently requires fed_alg=fedavg")
            display_round = int(getattr(script_args, "continual_round_offset", 0)) + int(current_round) + 1
            audit_path = Path(script_args.output_dir) / "batch_audits" / f"round_{display_round:03d}" / f"client_{int(client_id)}_{audit_phase}.json"
            return FormalFederatedSFTTrainer(
                delta_tracker=delta_tracker,
                audit_output_path=str(audit_path),
                expected_optimizer_steps=int(training_args.max_steps),
                **kwargs,
            )
        if delta_tracker:
            raise RuntimeError(
                "delta tracking requires formal_continual_mode or "
                "online_probe_mode"
            )
        if fed_args.fed_alg == 'fedprox':
            trainer = SFTTrainerFedProx(global_state=global_dict, prox_mu=fed_args.prox_mu, **kwargs)
        elif fed_args.fed_alg == 'scaffold':
            trainer = SFTTrainerSCAFFOLD(global_state=global_dict, local_auxiliary=local_auxiliary, global_auxiliary=global_auxiliary, **kwargs)
            trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
        elif (fed_args.fed_alg in ALGS_NORMAL_TRAINING) or (fed_args.fed_alg).startswith('local'):
            trainer = SFTTrainer(**kwargs)
        else:
            raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
        return trainer

    try:
        trainer = _make_trainer(common_kwargs)
    except TypeError:
        fallback_kwargs = dict(common_kwargs)
        fallback_kwargs.pop("processing_class", None)
        fallback_kwargs["tokenizer"] = tokenizer
        trainer = _make_trainer(fallback_kwargs)
    

    if delta_tracker:
        trainer.delta_tracker = delta_tracker
    
    return trainer

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para

class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)

"""Fail-closed SFT trainer with real-batch identity binding."""

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import SequentialSampler
from transformers import TrainerCallback
from trl import SFTTrainer

from utils.strict_token_contract import EOS_TOKEN_ID, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID


_STEP_AUDIT_KEY = "_safe_fcl_batch_audit"
_MODEL_FIELDS = ("input_ids", "attention_mask", "labels")
_IDENTITY_FIELDS = (
    "sample_id",
    "source_id",
    "content_hash",
    "normalized_prompt_hash",
    "dataset_name",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_FIELDS = (
    "_completion_marker_end",
    "_original_sequence_tokens",
    "_retained_sequence_tokens",
    "_original_instruction_tokens",
    "_original_response_tokens",
    "_retained_response_tokens",
    "_response_truncated",
)


def _fail_if_formal_formatting_is_attempted(*_args, **_kwargs):
    raise RuntimeError(
        "formal trainer received pretokenized data; "
        "TRL must not format or tokenize it"
    )


def _as_rows(value: Any, name: str) -> List[List[int]]:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(name + " must be a rank-2 torch.Tensor")
    return value.detach().cpu().tolist()


class StrictFederatedCompletionCollator:
    """Audit labels and attach identities from the actual DataLoader batch."""

    def __init__(self, base_collator, *, require_identity: bool = True):
        if not callable(base_collator):
            raise TypeError("base_collator must be callable")
        self._base_collator = base_collator
        self._require_identity = bool(require_identity)

    def __call__(self, features: Sequence[Mapping]) -> Dict[str, Any]:
        if isinstance(features, (str, bytes)) or not isinstance(features, Sequence) or not features:
            raise ValueError("features must be a non-empty sequence")

        model_features = []
        identities = []
        for feature in features:
            if not isinstance(feature, Mapping):
                raise TypeError("each feature must be a Mapping")
            missing_audit = [name for name in _AUDIT_FIELDS if name not in feature]
            if missing_audit:
                raise ValueError("feature is missing audit fields: " + repr(missing_audit))
            missing_identity = [name for name in _IDENTITY_FIELDS if name not in feature]
            if self._require_identity and missing_identity:
                raise ValueError("feature is missing identity fields: " + repr(missing_identity))
            if self._require_identity:
                if (
                    not isinstance(feature["sample_id"], int)
                    or isinstance(feature["sample_id"], bool)
                    or feature["sample_id"] < 0
                ):
                    raise ValueError("sample_id must be a non-negative integer")
                if not isinstance(feature["source_id"], str) or not feature["source_id"]:
                    raise ValueError("source_id must be a non-empty string")
                for hash_field in ("content_hash", "normalized_prompt_hash"):
                    if (
                        not isinstance(feature[hash_field], str)
                        or _HEX64.fullmatch(feature[hash_field]) is None
                    ):
                        raise ValueError(hash_field + " must be a lowercase SHA-256")
                if not isinstance(feature["dataset_name"], str) or not feature["dataset_name"]:
                    raise ValueError("dataset_name must be a non-empty string")
            model_features.append({
                name: copy.deepcopy(feature[name])
                for name in ("input_ids", "attention_mask")
            })
            identities.append({
                name: copy.deepcopy(feature[name])
                for name in _IDENTITY_FIELDS + _AUDIT_FIELDS
                if name in feature
            })

        batch = self._base_collator(model_features)
        if not isinstance(batch, Mapping):
            raise TypeError("base_collator must return a Mapping")
        if _STEP_AUDIT_KEY in batch:
            raise ValueError("base_collator returned reserved audit key")

        ids_rows = _as_rows(batch.get("input_ids"), "input_ids")
        mask_rows = _as_rows(batch.get("attention_mask"), "attention_mask")
        label_rows = _as_rows(batch.get("labels"), "labels")
        if not (len(ids_rows) == len(mask_rows) == len(label_rows) == len(features)):
            raise ValueError("collated batch dimensions do not match features")

        audited = []
        for identity, ids, mask, labels in zip(identities, ids_rows, mask_rows, label_rows):
            if not (len(ids) == len(mask) == len(labels)):
                raise ValueError("input_ids, attention_mask and labels lengths differ")
            if len(ids) > MAX_SEQUENCE_LENGTH:
                raise ValueError("collated sequence exceeds 256 tokens")
            marker_end = int(identity["_completion_marker_end"])
            real_length = int(sum(mask))
            if real_length != int(identity["_retained_sequence_tokens"]):
                raise ValueError("attention_mask length differs from retained token count")
            if not 0 < marker_end <= real_length:
                raise ValueError("completion marker boundary is outside the real sequence")
            if any(value != -100 for value in labels[:marker_end]):
                raise ValueError("prompt or response marker participates in loss")
            if any(value not in (0, 1) for value in mask):
                raise ValueError("attention_mask must be binary")
            if mask != ([1] * real_length + [0] * (len(mask) - real_length)):
                raise ValueError("only right padding is allowed")
            for index in range(real_length, len(ids)):
                if ids[index] != PAD_TOKEN_ID or labels[index] != -100:
                    raise ValueError("padding must use PAD=128004 and label=-100")
            trainable = [index for index in range(marker_end, real_length) if labels[index] != -100]
            for index in trainable:
                if labels[index] != ids[index]:
                    raise ValueError("answer labels must equal their input token IDs")
            eos_positions = [
                index for index in range(marker_end, real_length)
                if ids[index] == EOS_TOKEN_ID
            ]
            for index in eos_positions:
                if labels[index] != EOS_TOKEN_ID:
                    raise ValueError("a retained real EOS token was masked")

            record = dict(identity)
            record.update({
                "real_tokens": real_length,
                "answer_label_count": len(trainable),
                "retained_eos_count": len(eos_positions),
                "retained_eos_positions": eos_positions,
            })
            audited.append(record)

        canonical = json.dumps(audited, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        batch_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        output = dict(batch)
        output[_STEP_AUDIT_KEY] = {
            "batch_sha256": batch_sha,
            "samples": audited,
        }
        return output


class _OptimizerBindingCallback(TrainerCallback):
    def __init__(self, trainer: "FormalFederatedSFTTrainer"):
        self._trainer = trainer

    def on_optimizer_step(self, args, state, control, **kwargs):
        self._trainer.record_optimizer_step(kwargs.get("model"))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._trainer.finalize_batch_audit(getattr(state, "global_step", None))
        return control


class FormalFederatedSFTTrainer(SFTTrainer):
    """FedAvg local trainer whose delta records are bound to real batches."""

    def __init__(
        self,
        *args,
        delta_tracker=None,
        audit_output_path: Optional[str] = None,
        expected_optimizer_steps: int = 10,
        **kwargs,
    ):
        training_args = kwargs.get("args") if "args" in kwargs else (args[1] if len(args) > 1 else None)
        if training_args is None:
            raise ValueError("training arguments are required")
        if getattr(training_args, "per_device_train_batch_size", None) != 4:
            raise ValueError("release requires per_device_train_batch_size=4")
        if getattr(training_args, "gradient_accumulation_steps", None) != 4:
            raise ValueError("formal trainer requires gradient_accumulation_steps=4")
        if getattr(training_args, "remove_unused_columns", None) is not False:
            raise ValueError("formal trainer requires remove_unused_columns=False")
        if getattr(training_args, "dataloader_num_workers", None) != 0:
            raise ValueError("formal trainer requires dataloader_num_workers=0")
        if int(getattr(training_args, "max_steps", -1)) != int(expected_optimizer_steps):
            raise ValueError("max_steps differs from expected_optimizer_steps")
        if not isinstance(kwargs.get("data_collator"), StrictFederatedCompletionCollator):
            raise TypeError("formal trainer requires StrictFederatedCompletionCollator")
        dataset_kwargs = kwargs.get("dataset_kwargs")
        if dataset_kwargs is None:
            dataset_kwargs = {}
        elif not isinstance(dataset_kwargs, Mapping):
            raise TypeError("formal trainer dataset_kwargs must be a Mapping")
        else:
            dataset_kwargs = dict(dataset_kwargs)
        if dataset_kwargs.get("skip_prepare_dataset", True) is not True:
            raise ValueError("formal trainer requires skip_prepare_dataset == True")
        if kwargs.get("formatting_func") is not None:
            raise ValueError("formal trainer forbids a runtime formatting_func")
        if (
            "dataset_text_field" in kwargs
            and kwargs["dataset_text_field"] is not None
        ):
            raise ValueError("formal trainer forbids dataset_text_field")
        if kwargs.get("packing", False) is not False:
            raise ValueError("formal trainer requires packing=False")
        dataset_kwargs["skip_prepare_dataset"] = True
        kwargs["dataset_kwargs"] = dataset_kwargs
        kwargs["formatting_func"] = _fail_if_formal_formatting_is_attempted
        kwargs["packing"] = False
        self.delta_tracker = delta_tracker
        self.audit_output_path = Path(audit_output_path) if audit_output_path else None
        self.expected_optimizer_steps = int(expected_optimizer_steps)
        self._pending_audit = None
        self._audit_records = []
        self._optimizer_callback_count = 0
        super().__init__(*args, **kwargs)
        self.add_callback(_OptimizerBindingCallback(self))

    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)

    def training_step(self, model, inputs):
        labels = inputs.get("labels") if isinstance(inputs, dict) else None
        if labels is not None and not bool(labels[..., 1:].ne(-100).any()):
            raise ValueError("512-token truncation left this microbatch with no supervised response tokens")
        if not isinstance(inputs, dict) or _STEP_AUDIT_KEY not in inputs:
            raise ValueError("training batch is missing real-batch audit metadata")
        current = copy.deepcopy(inputs.pop(_STEP_AUDIT_KEY))
        samples = (self._pending_audit['samples'] if self._pending_audit else []) + current['samples']
        if len(samples)>16:raise RuntimeError('Too many examples before optimizer callback')
        canonical=json.dumps(samples,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        self._pending_audit={'samples':samples,'batch_sha256':hashlib.sha256(canonical.encode()).hexdigest()}
        try:
            return super().training_step(model, inputs)
        except Exception:
            self._pending_audit = None
            raise

    def record_optimizer_step(self, model):
        if self._pending_audit is None:
            raise RuntimeError("optimizer step has no bound DataLoader batch")
        if getattr(self.accelerator, "optimizer_step_was_skipped", None) is not False:
            raise RuntimeError("skipped or unverifiable optimizer step is forbidden")
        if model is None:
            raise RuntimeError("optimizer callback did not receive model")
        step = self._optimizer_callback_count + 1
        audit = self._pending_audit
        if len(audit["samples"])!=16:raise RuntimeError("Optimizer update must bind sixteen examples")
        if self.delta_tracker is not None:
            self.delta_tracker.record_real_batch(model, audit, step)
        self._audit_records.append({"optimizer_step": step, **audit})
        self._pending_audit = None
        self._optimizer_callback_count += 1

    def finalize_batch_audit(self, observed_global_step):
        if self._pending_audit is not None:
            raise RuntimeError("training ended with a pending batch")
        if observed_global_step != self.expected_optimizer_steps:
            raise RuntimeError("observed global_step differs from expected steps")
        if self._optimizer_callback_count != self.expected_optimizer_steps:
            raise RuntimeError("optimizer callback count differs from expected steps")
        if len(self._audit_records) != self.expected_optimizer_steps:
            raise RuntimeError("batch audit record count differs from expected steps")
        if self.audit_output_path is not None:
            self.audit_output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.audit_output_path.exists():
                raise FileExistsError("refusing to overwrite batch audit: " + str(self.audit_output_path))
            payload = json.dumps(
                {"optimizer_steps": self._audit_records},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n"
            temporary = self.audit_output_path.with_suffix(self.audit_output_path.suffix + ".tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(self.audit_output_path)

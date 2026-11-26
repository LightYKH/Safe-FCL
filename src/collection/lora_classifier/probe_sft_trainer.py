"""Probe-only SFT batch binding for strict single-step collection."""

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import SequentialSampler
from transformers import TrainerCallback
from trl import SFTTrainer

from lora_classifier.probe_delta_collector import ProbeDeltaMemoryCollector
from lora_classifier.probe_delta_core import validate_probe_step_samples


_STEP_SAMPLES_KEY = "_safe_fcl_step_samples"
_PROBE_METADATA_FIELDS = (
    "sample_id",
    "source_id",
    "content_hash",
    "normalized_prompt_hash",
    "raw_row_index",
    "split_rank",
    "probe_role",
    "probe_label",
)
_MODEL_FEATURE_FIELDS = (
    "input_ids",
    "attention_mask",
    "labels",
)


def _fail_if_probe_formatting_is_attempted(*_args, **_kwargs):
    raise RuntimeError(
        "ProbeSFTTrainer received pretokenized data; "
        "TRL must not format or tokenize it."
    )


def _default_state_extractor(model: Any) -> Mapping:
    from peft import get_peft_model_state_dict

    return get_peft_model_state_dict(model)


class ProbeMetadataCollator:
    """Separate ordered Probe identity metadata from model token features."""

    def __init__(
        self,
        base_collator: Callable,
        *,
        role: str,
        label: int,
    ) -> None:
        if not callable(base_collator):
            raise TypeError("base_collator must be callable")
        self._base_collator = base_collator
        self._role = role
        self._label = label

    def __call__(self, features: Sequence[Mapping]) -> Dict[str, Any]:
        if (
            isinstance(features, (str, bytes))
            or not isinstance(features, Sequence)
            or not features
        ):
            raise ValueError("collator features must be a non-empty sequence")

        step_samples: List[Dict[str, Any]] = []
        model_features: List[Dict[str, Any]] = []
        for feature in features:
            if not isinstance(feature, Mapping):
                raise TypeError("each collator feature must be a Mapping")
            sample = {
                field: feature[field]
                for field in _PROBE_METADATA_FIELDS
                if field in feature
            }
            step_samples.append(sample)
            model_features.append(
                {
                    field: copy.deepcopy(feature[field])
                    for field in _MODEL_FEATURE_FIELDS
                    if field in feature
                }
            )

        normalized_samples = validate_probe_step_samples(
            step_samples,
            self._role,
            self._label,
        )
        batch = self._base_collator(model_features)
        if not isinstance(batch, Mapping):
            raise TypeError("base_collator must return a Mapping")
        if _STEP_SAMPLES_KEY in batch:
            raise ValueError("base_collator returned the reserved metadata key")
        if "input_ids" not in batch:
            raise ValueError("base_collator batch must contain input_ids")

        expected_batch_size = len(normalized_samples)
        for field in _MODEL_FEATURE_FIELDS:
            if field not in batch:
                continue
            tensor = batch[field]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
                raise ValueError(
                    "collated "
                    + field
                    + " must be a torch.Tensor with a batch dimension"
                )
            if tensor.shape[0] != expected_batch_size:
                raise ValueError(
                    "collated tensor batch size does not match step_samples: "
                    + field
                )

        output = dict(batch)
        output[_STEP_SAMPLES_KEY] = normalized_samples
        return output


class ProbeOptimizerStepCallback(TrainerCallback):
    """Finalize exactly one pending real batch after an optimizer step."""

    def __init__(self, trainer: "ProbeSFTTrainer") -> None:
        self._trainer = trainer

    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model", getattr(self._trainer, "model", None))
        self._trainer._record_after_optimizer(model)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._trainer.finalize_probe_collection(
            observed_global_step=getattr(state, "global_step", None)
        )
        return control


class ProbeSFTTrainer(SFTTrainer):
    """SFTTrainer that binds one real batch to one cumulative optimizer record."""

    def __init__(
        self,
        *args,
        collector: ProbeDeltaMemoryCollector,
        expected_optimizer_steps: int,
        state_extractor: Optional[Callable[[Any], Mapping]] = None,
        packing: bool = False,
        fed_alg: str = "fedavg",
        **kwargs,
    ) -> None:
        if "resume_from_checkpoint" in kwargs:
            raise ValueError("ProbeSFTTrainer does not support checkpoint resume")
        training_args = kwargs.get("args")
        if training_args is None and len(args) >= 2:
            training_args = args[1]
        self._validate_supported_configuration(
            training_args,
            collector,
            expected_optimizer_steps,
            packing,
            fed_alg,
        )
        data_collator = kwargs.get("data_collator")
        if not isinstance(data_collator, ProbeMetadataCollator):
            raise TypeError(
                "ProbeSFTTrainer requires a ProbeMetadataCollator"
            )
        if state_extractor is not None and not callable(state_extractor):
            raise TypeError("state_extractor must be callable")

        raw_dataset_kwargs = kwargs.get("dataset_kwargs")
        if raw_dataset_kwargs is None:
            dataset_kwargs = {}
        elif isinstance(raw_dataset_kwargs, Mapping):
            dataset_kwargs = dict(raw_dataset_kwargs)
        else:
            raise TypeError("dataset_kwargs must be a Mapping or None")

        if kwargs.get("dataset_text_field") is not None:
            raise ValueError(
                "ProbeSFTTrainer does not accept dataset_text_field for "
                "pretokenized data"
            )
        if kwargs.get("formatting_func") is not None:
            raise ValueError(
                "ProbeSFTTrainer does not accept a caller formatting_func for "
                "pretokenized data"
            )
        if dataset_kwargs.get("skip_prepare_dataset", True) is not True:
            raise ValueError(
                "ProbeSFTTrainer requires skip_prepare_dataset == True"
            )

        dataset_kwargs["skip_prepare_dataset"] = True
        kwargs["dataset_kwargs"] = dataset_kwargs
        kwargs["formatting_func"] = _fail_if_probe_formatting_is_attempted

        self.probe_collector = collector
        self._expected_optimizer_steps = expected_optimizer_steps
        self._probe_state_extractor = (
            state_extractor
            if state_extractor is not None
            else _default_state_extractor
        )
        self._pending_step_samples: Optional[List[Dict[str, Any]]] = None
        self._optimizer_callback_count = 0
        self._optimizer_skip_count = 0
        self._probe_training_failed = False

        super().__init__(*args, packing=False, **kwargs)
        self._probe_optimizer_callback = ProbeOptimizerStepCallback(self)
        self.add_callback(self._probe_optimizer_callback)

    def _get_train_sampler(self):
        if self.train_dataset is None:
            return None
        return SequentialSampler(self.train_dataset)

    @staticmethod
    def _validate_supported_configuration(
        training_args: Any,
        collector: ProbeDeltaMemoryCollector,
        expected_optimizer_steps: int,
        packing: bool,
        fed_alg: str,
    ) -> None:
        if not isinstance(collector, ProbeDeltaMemoryCollector):
            raise TypeError("collector must be a ProbeDeltaMemoryCollector")
        if training_args is None:
            raise ValueError("ProbeSFTTrainer requires training arguments")
        if (
            not isinstance(expected_optimizer_steps, int)
            or isinstance(expected_optimizer_steps, bool)
            or expected_optimizer_steps <= 0
        ):
            raise ValueError(
                "expected_optimizer_steps must be a positive integer"
            )
        max_steps = getattr(training_args, "max_steps", None)
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
        ):
            raise ValueError(
                "training_args.max_steps must be an integer"
            )
        if max_steps != expected_optimizer_steps:
            raise ValueError(
                "training_args.max_steps must equal expected_optimizer_steps"
            )
        if getattr(training_args, "remove_unused_columns", None) is not False:
            raise ValueError(
                "ProbeSFTTrainer requires remove_unused_columns == False"
            )
        if getattr(training_args, "per_device_train_batch_size", None) != 4:
            raise ValueError("release requires per_device_train_batch_size=4")
        accumulation = getattr(
            training_args,
            "gradient_accumulation_steps",
            None,
        )
        if (
            not isinstance(accumulation, int)
            or isinstance(accumulation, bool)
            or accumulation != 4
        ):
            raise ValueError(
                "ProbeSFTTrainer requires gradient_accumulation_steps == 4"
            )
        world_size = getattr(training_args, "world_size", None)
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size != 1
        ):
            raise ValueError("ProbeSFTTrainer requires world_size == 1")
        if packing is not False:
            raise ValueError("ProbeSFTTrainer requires packing == False")
        if fed_alg != "fedavg":
            raise ValueError("ProbeSFTTrainer only supports FedAvg semantics")
        if not callable(getattr(TrainerCallback, "on_optimizer_step", None)):
            raise RuntimeError(
                "Transformers TrainerCallback lacks on_optimizer_step"
            )

    @property
    def pending_step_samples(self) -> Optional[List[Dict[str, Any]]]:
        return copy.deepcopy(self._pending_step_samples)

    @property
    def optimizer_callback_count(self) -> int:
        return self._optimizer_callback_count

    @property
    def expected_optimizer_steps(self) -> int:
        return self._expected_optimizer_steps

    def training_step(self, model, inputs):
        labels = inputs.get("labels") if isinstance(inputs, dict) else None
        if labels is not None and not bool(labels[..., 1:].ne(-100).any()):
            raise ValueError("512-token truncation left this microbatch with no supervised response tokens")
        if not isinstance(inputs, dict):
            raise TypeError("Probe training inputs must be a dictionary")
        if _STEP_SAMPLES_KEY not in inputs:
            raise ValueError("Probe training batch is missing step_samples")

        raw_samples = inputs.pop(_STEP_SAMPLES_KEY)
        current = self.probe_collector.validate_step_samples(raw_samples)
        self._pending_step_samples = (self._pending_step_samples or []) + current
        if len(self._pending_step_samples)>16:raise RuntimeError("Too many examples before optimizer callback")
        try:
            return super().training_step(model, inputs)
        except Exception:
            self._pending_step_samples = None
            self._probe_training_failed = True
            self.probe_collector.abort()
            raise

    def _fail_optimizer_binding(self, message: str) -> None:
        self._pending_step_samples = None
        self._probe_training_failed = True
        self.probe_collector.abort()
        raise RuntimeError(message)

    def _record_after_optimizer(self, model: Any) -> None:
        if self._pending_step_samples is None:
            self._fail_optimizer_binding(
                "optimizer callback requires exactly one pending Probe batch"
            )

        missing = object()
        skipped = getattr(
            getattr(self, "accelerator", None),
            "optimizer_step_was_skipped",
            missing,
        )
        if skipped is missing:
            self._fail_optimizer_binding(
                "optimizer_step_was_skipped attribute is unavailable"
            )
        if not isinstance(skipped, bool):
            self._fail_optimizer_binding(
                "optimizer_step_was_skipped must be bool"
            )
        if skipped:
            self._optimizer_skip_count += 1
            self._fail_optimizer_binding(
                "skipped optimizer steps are not supported"
            )
        if model is None:
            self._fail_optimizer_binding(
                "optimizer callback did not receive the current model"
            )

        pending = self._pending_step_samples
        if len(pending)!=16:self._fail_optimizer_binding("Optimizer update must bind sixteen original examples")
        try:
            current_state = self._probe_state_extractor(model)
            self.probe_collector.record_optimizer_step(
                current_state,
                pending,
            )
        except Exception:
            self._pending_step_samples = None
            self._probe_training_failed = True
            self.probe_collector.abort()
            raise

        self._pending_step_samples = None
        self._optimizer_callback_count += 1

    def _fail_probe_finalization(self, message: str) -> None:
        self._probe_training_failed = True
        self.probe_collector.abort()
        raise RuntimeError(message)

    def finalize_probe_collection(self, *, observed_global_step: int):
        if (
            not isinstance(observed_global_step, int)
            or isinstance(observed_global_step, bool)
            or observed_global_step < 0
        ):
            self._fail_probe_finalization(
                "observed_global_step must be a non-negative integer"
            )
        if self._probe_training_failed:
            self._fail_probe_finalization(
                "Probe training failed; collector is incomplete"
            )
        if self._pending_step_samples is not None:
            self._fail_probe_finalization(
                "Probe training ended with a pending batch"
            )
        if self._optimizer_skip_count != 0:
            self._fail_probe_finalization(
                "Probe training contains skipped optimizer steps"
            )
        if observed_global_step != self._expected_optimizer_steps:
            self._fail_probe_finalization(
                "observed global step does not match expected optimizer steps"
            )
        if (
            self._optimizer_callback_count
            != self._expected_optimizer_steps
        ):
            self._fail_probe_finalization(
                "optimizer callback count does not match expected optimizer steps"
            )
        if (
            self.probe_collector.record_count
            != self._expected_optimizer_steps
        ):
            self._fail_probe_finalization(
                "collector record count does not match expected optimizer steps"
            )
        try:
            return self.probe_collector.finalize(
                self._expected_optimizer_steps
            )
        except Exception:
            self._probe_training_failed = True
            self.probe_collector.abort()
            raise

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        if resume_from_checkpoint not in (None, False):
            raise ValueError("ProbeSFTTrainer does not support checkpoint resume")
        try:
            return super().train(
                resume_from_checkpoint=None,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                **kwargs,
            )
        except Exception:
            self._pending_step_samples = None
            self._probe_training_failed = True
            self.probe_collector.abort()
            raise

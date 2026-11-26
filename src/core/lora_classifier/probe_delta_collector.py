"""In-memory cumulative Delta-LoRA collection for Probe training."""

import copy
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence

import torch

from lora_classifier.probe_delta_core import (
    build_first_layer_lora_b_spec,
    build_probe_delta_record_metadata,
    compute_cumulative_lora_b_delta,
    compute_delta_tensor_payload_sha256,
    validate_probe_step_samples,
)


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _clone_state(state: Mapping) -> Dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError("LoRA state must be a Mapping")
    return {key: _clone_value(value) for key, value in state.items()}


def _clone_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata": copy.deepcopy(record["metadata"]),
        "delta_tensors": {
            key: tensor.detach().clone()
            for key, tensor in record["delta_tensors"].items()
        },
    }


class ProbeDeltaMemoryCollector:
    """Build a strict, ordered chain of cumulative Probe Delta-LoRA records."""

    def __init__(
        self,
        *,
        initial_state: Mapping,
        manifest_sha256: str,
        code_commit: str,
        model_artifact_sha256: str,
        tokenizer_artifact_sha256: str,
        round_index: int,
        client_id: int,
        role: str,
        label: int,
        collection_r0_tensor_sha256: str,
        delta_base_round: int,
        delta_base_tensor_sha256: str,
        lr_policy: str,
        learning_rate: float,
        expected_rank: int = 8,
    ) -> None:
        if role not in ("benign", "malicious"):
            raise ValueError("role must be 'benign' or 'malicious'")
        expected_label = 0 if role == "benign" else 1
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label != expected_label
        ):
            raise ValueError("role and label are inconsistent")

        self._initial_state = _clone_state(initial_state)
        self._tensor_spec = build_first_layer_lora_b_spec(
            self._initial_state,
            expected_rank=expected_rank,
        )
        self._manifest_sha256 = manifest_sha256
        self._code_commit = code_commit
        self._model_artifact_sha256 = model_artifact_sha256
        self._tokenizer_artifact_sha256 = tokenizer_artifact_sha256
        self._round_index = round_index
        self._client_id = client_id
        self._role = role
        self._label = label
        self._collection_r0_tensor_sha256 = collection_r0_tensor_sha256
        self._delta_base_round = delta_base_round
        self._delta_base_tensor_sha256 = delta_base_tensor_sha256
        self._lr_policy = lr_policy
        self._learning_rate = learning_rate
        self._step_index = 0
        self._previous_record_id: Optional[str] = None
        self._previous_history_digest: Optional[str] = None
        self._records: List[Dict[str, Any]] = []
        self._record_in_progress = False
        self._finalized = False
        self._aborted = False

    @property
    def role(self) -> str:
        return self._role

    @property
    def label(self) -> int:
        return self._label

    @property
    def round_index(self) -> int:
        return self._round_index

    @property
    def client_id(self) -> int:
        return self._client_id

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    @property
    def is_aborted(self) -> bool:
        return self._aborted

    @property
    def tensor_spec(self) -> Dict[str, Any]:
        return copy.deepcopy(self._tensor_spec)

    @property
    def records(self) -> List[Dict[str, Any]]:
        return [_clone_record(record) for record in self._records]

    def validate_step_samples(
        self,
        step_samples: Sequence[Mapping],
    ) -> List[Dict[str, Any]]:
        return validate_probe_step_samples(
            step_samples,
            self._role,
            self._label,
        )

    def record_optimizer_step(
        self,
        current_state: Mapping,
        step_samples: Sequence[Mapping],
    ) -> Dict[str, Any]:
        if self._finalized:
            raise RuntimeError("collector is already finalized")
        if self._aborted:
            raise RuntimeError("collector is aborted")
        if self._record_in_progress:
            raise RuntimeError("collector already has a record in progress")

        normalized_samples = self.validate_step_samples(step_samples)
        self._record_in_progress = True
        try:
            delta_tensors = compute_cumulative_lora_b_delta(
                self._initial_state,
                current_state,
                self._tensor_spec,
            )
            tensor_payload_sha256 = compute_delta_tensor_payload_sha256(
                delta_tensors,
                self._tensor_spec,
            )
            metadata = build_probe_delta_record_metadata(
                manifest_sha256=self._manifest_sha256,
                code_commit=self._code_commit,
                model_artifact_sha256=self._model_artifact_sha256,
                tokenizer_artifact_sha256=self._tokenizer_artifact_sha256,
                lora_spec=self._tensor_spec,
                round_index=self._round_index,
                client_id=self._client_id,
                step_index=self._step_index,
                role=self._role,
                label=self._label,
                step_samples=normalized_samples,
                previous_record_id=self._previous_record_id,
                previous_history_digest=self._previous_history_digest,
                tensor_payload_sha256=tensor_payload_sha256,
                collection_r0_tensor_sha256=(
                    self._collection_r0_tensor_sha256
                ),
                delta_base_round=self._delta_base_round,
                delta_base_tensor_sha256=self._delta_base_tensor_sha256,
                lr_policy=self._lr_policy,
                learning_rate=self._learning_rate,
            )
            stored_record = {
                "metadata": metadata,
                "delta_tensors": {
                    key: tensor.detach().clone()
                    for key, tensor in delta_tensors.items()
                },
            }
            self._records.append(stored_record)
            self._previous_record_id = metadata["record_id"]
            self._previous_history_digest = metadata["history_digest"]
            self._step_index += 1
            return _clone_record(stored_record)
        finally:
            self._record_in_progress = False

    def abort(self) -> None:
        if not self._finalized:
            self._aborted = True

    def finalize(self, expected_steps: int) -> List[Dict[str, Any]]:
        if self._finalized:
            raise RuntimeError("collector is already finalized")
        if self._aborted:
            raise RuntimeError("collector is aborted")
        if self._record_in_progress:
            raise RuntimeError("collector has a record in progress")
        if (
            not isinstance(expected_steps, int)
            or isinstance(expected_steps, bool)
            or expected_steps < 0
        ):
            raise ValueError("expected_steps must be a non-negative integer")
        if len(self._records) != expected_steps:
            raise ValueError(
                "collector record count does not match expected optimizer steps"
            )

        for expected_index, record in enumerate(self._records):
            metadata = record["metadata"]
            if metadata.get("step_index") != expected_index:
                raise ValueError("collector step indexes must be contiguous from zero")
            if not metadata.get("step_samples"):
                raise ValueError("collector records must contain non-empty step_samples")

        self._finalized = True
        return self.records

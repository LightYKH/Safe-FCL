"""Canonical LoRA tensor identity and immutable stage-base guard."""

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping

import torch

from .frozen_continual_schedule import (
    SHADOW_DETECTION_CONTRACT,
    validate_formal_shadow_learning_rate,
)


def canonical_lora_tensor_sha256(state: Mapping[str, Any]) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("LoRA state must be a non-empty Mapping")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not key:
            raise ValueError("LoRA state keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError("LoRA state values must be tensors")
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor.float()).all().item():
            raise ValueError("LoRA state contains NaN or Inf: " + key)
        key_bytes = key.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack(">Q", len(key_bytes)))
        digest.update(key_bytes)
        digest.update(struct.pack(">Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack(">Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


class FrozenLoraStateGuard:
    def __init__(self, state: Mapping[str, Any], expected_sha256: str):
        expected = str(expected_sha256).lower()
        if len(expected) != 64:
            raise ValueError("expected stage-base tensor SHA-256 is required")
        actual = canonical_lora_tensor_sha256(state)
        if actual != expected:
            raise ValueError(
                "stage shadow base tensor SHA mismatch: expected=%s actual=%s"
                % (expected, actual)
            )
        self.expected_sha256 = expected

    def assert_unchanged(self, state: Mapping[str, Any], *, context: str) -> None:
        actual = canonical_lora_tensor_sha256(state)
        if actual != self.expected_sha256:
            raise RuntimeError(
                "frozen shadow base changed at %s: expected=%s actual=%s"
                % (context, self.expected_sha256, actual)
            )

    def write_provenance(
        self,
        path: Any,
        *,
        task_id: int,
        source_round: int,
        formal_method: str,
        shadow_learning_rate: float,
    ) -> None:
        target = Path(path)
        if target.exists():
            raise FileExistsError("refusing to overwrite shadow-base provenance")
        target.parent.mkdir(parents=True, exist_ok=True)
        method = str(formal_method).lower()
        if method not in ("pure", "step", "client", "shadow"):
            raise ValueError("unsupported formal method")
        expected_source_round = (int(task_id) - 1) * 100
        if int(task_id) not in range(1, 6) or int(source_round) != expected_source_round:
            raise ValueError("stage Shadow base source round mismatch")
        shadow_learning_rate = validate_formal_shadow_learning_rate(
            shadow_learning_rate
        )
        payload = {
            "formal_method": method,
            "shadow_detection_active": method == "shadow",
            "shadow_detection_contract": SHADOW_DETECTION_CONTRACT,
            "shadow_learning_rate": shadow_learning_rate,
            "source_global_round": int(source_round),
            "stage_shadow_base_tensor_sha256": self.expected_sha256,
            "task_id": int(task_id),
        }
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

"""Fail-closed identity guard for the one fixed Task1 Probe artifact."""

import hashlib
import json
import re
from pathlib import Path

import torch

from .frozen_continual_schedule import (
    MAIN_LEARNING_RATE_SCHEDULE,
    PROBE_COLLECTION_LR_POLICY,
    PROBE_COLLECTION_STATE_CONTRACT,
    frozen_cosine_lr,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class FrozenProbeGuard:
    def __init__(self, path: str, expected_sha256: str, *, threshold: float):
        unresolved = Path(path).expanduser()
        if unresolved.is_symlink():
            raise ValueError("fixed Probe path must not be a symlink")
        self.path = unresolved.resolve()
        if not self.path.is_file():
            raise FileNotFoundError("fixed Probe file does not exist")
        self.expected_sha256 = str(expected_sha256).lower()
        if len(self.expected_sha256) != 64:
            raise ValueError("fixed Probe expected SHA-256 is required")
        if abs(float(threshold) - 0.8) > 1e-15:
            raise ValueError("formal fixed Probe threshold must be 0.8")
        self.threshold = 0.8
        self.assert_unchanged(context="initialization")
        artifact = torch.load(str(self.path), map_location="cpu")
        if not isinstance(artifact, dict):
            raise ValueError("Probe artifact must be a dictionary")
        ordered_keys = artifact.get("ordered_keys")
        key_sizes = artifact.get("key_sizes")
        if (
            not isinstance(ordered_keys, list)
            or not ordered_keys
            or len(set(ordered_keys)) != len(ordered_keys)
            or any(not isinstance(key, str) or not key for key in ordered_keys)
        ):
            raise ValueError("Probe ordered_keys are missing")
        if not isinstance(key_sizes, dict) or set(key_sizes) != set(ordered_keys):
            raise ValueError("Probe key_sizes do not match ordered_keys")
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in key_sizes.values()
        ):
            raise ValueError("Probe key_sizes must be positive integers")
        resolved_sizes = {}
        for key in ordered_keys:
            match = re.search(
                r"(?:^|\.)layers\.0\..*\.(q_proj|v_proj)\.lora_B\.weight$",
                key,
            )
            if match is None:
                raise ValueError(
                    "formal fixed Probe keys must be exported layer-0 q/v lora_B.weight"
                )
            target = match.group(1)
            if target in resolved_sizes:
                raise ValueError("formal fixed Probe contains duplicate LoRA targets")
            resolved_sizes[target] = key_sizes[key]
        if resolved_sizes != {"q_proj": 32768, "v_proj": 8192}:
            raise ValueError("formal fixed Probe q/v feature sizes do not match Llama-3.1 rank-8")
        if artifact.get("model_type") != "linear_probe":
            raise ValueError("formal fixed Probe must use model_type=linear_probe")
        if artifact.get("filter_policy") != "first_B":
            raise ValueError("formal fixed Probe must use filter_policy=first_B")
        if int(artifact.get("seed", -1)) != 2023:
            raise ValueError("formal fixed Probe must use seed=2023")
        if abs(float(artifact.get("threshold", -1.0)) - 0.8) > 1e-15:
            raise ValueError("formal fixed Probe artifact threshold must be 0.8")
        normalize = str(artifact.get("normalize", "none")).lower().replace("-", "_")
        if normalize != "l2":
            raise ValueError("formal fixed Probe normalization must be l2")
        if artifact.get("probe_collection_lr_policy") != PROBE_COLLECTION_LR_POLICY:
            raise ValueError("formal fixed Probe collection LR policy mismatch")
        collection_lrs = artifact.get("probe_collection_learning_rates")
        expected_collection_lrs = [frozen_cosine_lr(index) for index in range(10)]
        if collection_lrs != expected_collection_lrs:
            raise ValueError("formal fixed Probe collection LR values mismatch")
        if artifact.get("probe_collection_lr_schedule_source") != (
            MAIN_LEARNING_RATE_SCHEDULE
        ):
            raise ValueError("formal fixed Probe collection LR source mismatch")
        if artifact.get("probe_collection_state_contract") != (
            PROBE_COLLECTION_STATE_CONTRACT
        ):
            raise ValueError("formal fixed Probe collection state contract mismatch")
        collection_plan_sha = str(
            artifact.get("probe_collection_plan_sha256", "")
        ).lower()
        collection_run_sha = str(
            artifact.get("probe_collection_run_sha256", "")
        ).lower()
        if (
            re.fullmatch(r"[0-9a-f]{64}", collection_plan_sha) is None
            or re.fullmatch(r"[0-9a-f]{64}", collection_run_sha) is None
        ):
            raise ValueError("formal fixed Probe collection SHA identity is missing")
        self.identity = {
            "filter_policy": artifact.get("filter_policy"),
            "key_sizes_sha256": _canonical_sha(key_sizes),
            "normalization": normalize,
            "ordered_keys_sha256": _canonical_sha(ordered_keys),
            "probe_path": str(self.path),
            "probe_sha256": self.expected_sha256,
            "probe_collection_lr_policy": PROBE_COLLECTION_LR_POLICY,
            "probe_collection_learning_rates": expected_collection_lrs,
            "probe_collection_lr_schedule_source": MAIN_LEARNING_RATE_SCHEDULE,
            "probe_collection_state_contract": PROBE_COLLECTION_STATE_CONTRACT,
            "probe_collection_plan_sha256": collection_plan_sha,
            "probe_collection_run_sha256": collection_run_sha,
            "threshold": self.threshold,
        }

    def assert_unchanged(self, *, context: str) -> None:
        actual = _sha256_file(self.path)
        if actual != self.expected_sha256:
            raise RuntimeError(
                "fixed Probe changed at %s: expected=%s actual=%s"
                % (context, self.expected_sha256, actual)
            )

    def write_provenance(self, path) -> None:
        target = Path(path)
        if target.exists():
            raise FileExistsError("refusing to overwrite fixed Probe provenance")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

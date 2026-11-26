"""Frozen process environment required by formal GPU entry points."""

import os
from collections.abc import Mapping
from typing import Dict, Optional


FORMAL_GPU_RUNTIME_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_DISABLED": "true",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
}


def configure_formal_gpu_runtime_environment(
    environment: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Set the frozen GPU environment and reject inherited conflicts."""
    target = os.environ if environment is None else environment
    if not isinstance(target, Mapping) or not hasattr(target, "__setitem__"):
        raise TypeError("environment must be a mutable mapping")
    for key, expected in FORMAL_GPU_RUNTIME_ENVIRONMENT.items():
        existing = target.get(key)
        if existing is not None and existing != expected:
            raise ValueError(
                "conflicting formal GPU environment variable "
                + key
                + ": expected "
                + expected
            )
    for key, expected in FORMAL_GPU_RUNTIME_ENVIRONMENT.items():
        target[key] = expected
    return dict(FORMAL_GPU_RUNTIME_ENVIRONMENT)


def require_formal_gpu_runtime_environment(
    environment: Optional[Mapping] = None,
) -> Dict[str, str]:
    """Fail unless every frozen GPU environment value is active."""
    target = os.environ if environment is None else environment
    if not isinstance(target, Mapping):
        raise TypeError("environment must be a Mapping")
    for key, expected in FORMAL_GPU_RUNTIME_ENVIRONMENT.items():
        if target.get(key) != expected:
            raise RuntimeError(
                "formal GPU environment is not configured: " + key
            )
    return dict(FORMAL_GPU_RUNTIME_ENVIRONMENT)

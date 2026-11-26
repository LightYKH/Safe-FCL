"""Canonicalize newly saved formal PEFT adapter configuration files."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .data_manifest import sha256_file


FORMAL_TARGET_MODULE_ORDER = ("q_proj", "v_proj")


def canonicalize_formal_adapter_config(
    path: Any,
    *,
    expected_target_modules: Sequence[str] = FORMAL_TARGET_MODULE_ORDER,
) -> str:
    """Freeze target-module order and return the canonical file SHA-256."""
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("formal adapter config must not be a symlink")
    target = unresolved.resolve()
    if not target.is_file() or target.stat().st_size == 0:
        raise FileNotFoundError("formal adapter config is missing")

    expected = list(expected_target_modules)
    if (
        not expected
        or len(expected) != len(set(expected))
        or any(not isinstance(item, str) or not item for item in expected)
    ):
        raise ValueError("expected target modules must be unique non-empty strings")

    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("formal adapter config must be a JSON object")
    actual = value.get("target_modules")
    if (
        not isinstance(actual, list)
        or len(actual) != len(expected)
        or any(not isinstance(item, str) or not item for item in actual)
        or set(actual) != set(expected)
    ):
        raise ValueError("formal adapter target_modules mismatch")
    value["target_modules"] = expected
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    original_mode = target.stat().st_mode & 0o777
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            dir=str(target.parent),
            prefix=".adapter-config.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, original_mode)
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return sha256_file(target)

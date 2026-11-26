"""Strict validation for the frozen formal Python/Conda environment lock."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .data_manifest import sha256_file


HEX64 = re.compile(r"^[0-9a-f]{64}$")
LOCK_LINE = re.compile(r"^([0-9a-f]{64})  ([^/\\]+)$")


def validate_environment_lock(
    path: Any,
    expected_sha256: str,
    *,
    expected_python_executable: Optional[Any] = None,
) -> Dict[str, Any]:
    supplied_lock_path = Path(path).expanduser()
    expected = str(expected_sha256 or "").lower()
    if HEX64.fullmatch(expected) is None:
        raise ValueError("environment lock SHA-256 is required")
    if supplied_lock_path.is_symlink():
        raise FileNotFoundError("environment lock must not be a symbolic link")
    lock_path = supplied_lock_path.resolve()
    if not lock_path.is_file():
        raise FileNotFoundError("environment lock must be a regular file")
    if sha256_file(lock_path) != expected:
        raise ValueError("environment lock SHA-256 mismatch")

    entries = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        match = LOCK_LINE.fullmatch(line)
        if match is None:
            raise ValueError("environment lock line is not canonical")
        digest, name = match.groups()
        if name in entries:
            raise ValueError("environment lock contains a duplicate filename")
        target = lock_path.parent / name
        if target.is_symlink() or not target.is_file():
            raise FileNotFoundError("environment lock member is missing: " + name)
        if sha256_file(target) != digest:
            raise ValueError("environment lock member SHA-256 mismatch: " + name)
        entries[name] = digest
    if "runtime-identity.json" not in entries:
        raise ValueError("environment lock must include runtime-identity.json")

    runtime_path = lock_path.parent / "runtime-identity.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    locked_executable_value = runtime.get("python", {}).get("executable")
    if not isinstance(locked_executable_value, str) or not locked_executable_value:
        raise ValueError("runtime identity must contain a Python executable")
    locked_executable = Path(locked_executable_value).expanduser().resolve()
    if not locked_executable.is_file() or not os.access(locked_executable, os.X_OK):
        raise FileNotFoundError("locked Python executable is not executable")
    if expected_python_executable is not None:
        expected_executable = Path(expected_python_executable).expanduser().resolve()
        if locked_executable != expected_executable:
            raise ValueError("environment lock Python executable mismatch")
    return {
        "environment_lock_path": str(lock_path),
        "environment_lock_sha256": expected,
        "locked_python_executable": str(locked_executable),
        "member_sha256": entries,
        "runtime_identity_sha256": entries["runtime-identity.json"],
    }

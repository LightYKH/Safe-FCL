"""Build and validate independent continual evaluation partition metadata."""

import json
from pathlib import Path
from typing import Any, Dict

from .artifact_schema import matches_schema
from utils.data_manifest import (
    canonical_json_bytes,
    compute_normalized_prompt_hash,
    normalize_prompt,
    sha256_file,
)


SCHEMA_VERSION = "safe-fcl.formal-evaluation-manifest.v1"


def prompt_sha256(text: str) -> str:
    return compute_normalized_prompt_hash(text)


def load_formal_evaluation_manifest(path: str, expected_sha256: str) -> Dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError("evaluation manifest does not exist")
    if sha256_file(target) != str(expected_sha256).lower():
        raise ValueError("evaluation manifest SHA-256 mismatch")
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical_json_bytes(value):
        raise ValueError("evaluation manifest is not canonical JSON")
    if not matches_schema(value.get("schema_version"), SCHEMA_VERSION):
        raise ValueError("evaluation manifest schema mismatch")
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise ValueError("evaluation manifest requires benign and malicious datasets")
    expected = {
        "benign": {"validation": 500},
        "malicious": {"formal_test": 520, "validation": 480},
    }
    all_source_ids = set()
    for entry in datasets:
        role = entry.get("role")
        if role not in expected:
            raise ValueError("evaluation dataset role is invalid")
        records = entry.get("records")
        if not isinstance(records, list):
            raise ValueError("evaluation records must be a list")
        counts = {name: 0 for name in expected[role]}
        for record in records:
            partition = record.get("partition")
            if partition not in counts:
                raise ValueError("evaluation partition is invalid")
            counts[partition] += 1
            source_id = record.get("source_id")
            if not isinstance(source_id, str) or not source_id or source_id in all_source_ids:
                raise ValueError("evaluation source IDs must be non-empty and unique")
            all_source_ids.add(source_id)
            for field in ("content_hash", "prompt_sha256"):
                digest = record.get(field)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError(field + " must be SHA-256")
        if counts != expected[role]:
            raise ValueError("evaluation partition counts mismatch")
    return value

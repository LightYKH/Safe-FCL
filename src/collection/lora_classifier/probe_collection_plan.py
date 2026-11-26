"""Deterministic run plans for the offline Probe Delta-LoRA collector."""

import copy
import json
import math
import os
import platform
import random
import re
import stat
import subprocess
import tempfile
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from datasets import Dataset

from utils.artifact_schema import matches_schema
from utils.data_manifest import (
    MANIFEST_SCHEMA_VERSION,
    PIPELINE_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_bytes,
)
from utils.dataset_partitioning import (
    PROBE_CLIENTS_PER_ROLE,
    PROBE_EFFECTIVE_TOTAL_PER_ROLE,
    PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT,
    PROBE_TARGET_DATA_PER_CLIENT,
    probe_effective_client_pool_sizes,
)
from utils.continual_data_manifest import (
    SCHEMA_VERSION as CONTINUAL_SCHEMA_VERSION,
)
from utils.manifest_probe_data import validate_probe_collection_manifest
from utils.manifest_training_data import (
    load_training_manifest,
    validate_manifest_training_provenance,
)
from utils import model_artifact_identity as artifact_identity_core
from utils.model_artifact_identity import (
    build_model_artifact_identity,
    build_tokenizer_artifact_identity,
    validate_artifact_identity,
)
from utils.frozen_continual_schedule import (
    MAIN_LEARNING_RATE_SCHEDULE,
    PROBE_COLLECTION_LR_POLICY,
    PROBE_COLLECTION_STATE_CONTRACT,
    frozen_cosine_lr,
)


PLAN_SCHEMA_VERSION = "safe-fcl.probe-collection-plan.v5"
PLAN_FILE_NAME = "probe-collection-plan.json"
PLAN_SIDECAR_NAME = PLAN_FILE_NAME + ".sha256"

_EXPECTED_BRANCH = os.environ.get(
    "SAFE_FCL_PROBE_EXPECTED_BRANCH",
    "main",
)
_ALLOW_DETACHED_HEAD = (
    os.environ.get("SAFE_FCL_PROBE_ALLOW_DETACHED_HEAD", "false").lower()
    == "true"
)
_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_KEYS = (
    "python",
    "torch",
    "transformers",
    "trl",
    "accelerate",
    "peft",
    "datasets",
    "numpy",
    "pandas",
    "pyarrow",
    "safetensors",
)
_MANIFEST_RUNTIME_MAP = {
    "python_version": "python",
    "datasets_version": "datasets",
    "numpy_version": "numpy",
    "pandas_version": "pandas",
    "pyarrow_version": "pyarrow",
}
_CLIENT_COUNT = 10
_CLIENTS_PER_ROLE = PROBE_CLIENTS_PER_ROLE
_ROUND_COUNT = 10
_CLIENTS_PER_ROUND = 3
_OPTIMIZER_STEPS = 10
_BATCH_SIZE = 16
_LOCAL_SAMPLE_EVENTS = _OPTIMIZER_STEPS * _BATCH_SIZE
_RECORDS_PER_ROUND = _CLIENTS_PER_ROUND * _OPTIMIZER_STEPS
_FROZEN_CLIENT_SCHEDULE = (
    (0, 6, 9),
    (1, 2, 4),
    (0, 1, 8),
    (2, 3, 8),
    (1, 3, 4),
    (4, 5, 9),
    (1, 7, 9),
    (2, 5, 6),
    (3, 5, 6),
    (4, 5, 7),
)
_MODEL_CONFIG_PROFILE = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "vocab_size": 128256,
}
_PROTOCOL = {
    "base_model": "Llama3.1-8B local artifact",
    "base_model_quantization": "8bit",
    "batch_size": 4,
    "bitsandbytes_required": True,
    "client_sampling_seed": "round_index",
    "clients_per_round": 3,
    "federated_algorithm": "shadow_fixed_reference_no_aggregation",
    "gradient_accumulation_steps": 4,
    "learning_rate_schedule": PROBE_COLLECTION_LR_POLICY,
    "learning_rate_schedule_source": MAIN_LEARNING_RATE_SCHEDULE,
    "learning_rates": [frozen_cosine_lr(index) for index in range(10)],
    "lora_alpha": 16,
    "lora_initialization_seed": 2023,
    "lora_rank": 8,
    "max_sequence_length": 512,
    "padding_side": "right",
    "pad_token_id": 128004,
    "eos_token_id": 128001,
    "completion_aware_truncation": False,
    "retained_real_eos_participates_in_loss": True,
    "truncated_response_gets_synthetic_eos": False,
    "model_config_profile": _MODEL_CONFIG_PROFILE,
    "optimizer": "adamw_bnb_8bit",
    "optimizer_steps_per_client": 10,
    "packing": False,
    "probe_client_count": 10,
    "probe_clients_benign": 5,
    "probe_clients_malicious": 5,
    "probe_pool_shuffle_seed": 2023,
    "records_per_round": 30,
    "resume_from_checkpoint": False,
    "round_count": 10,
    "sample_events_per_selected_client": 160,
    "probe_target_data_per_client": PROBE_TARGET_DATA_PER_CLIENT,
    "probe_effective_count_per_role": PROBE_EFFECTIVE_TOTAL_PER_ROLE,
    "probe_effective_data_per_client": 281,
    "probe_min_samples_per_selected_client": (
        PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT
    ),
    "single_process": True,
    "collection_state_contract": PROBE_COLLECTION_STATE_CONTRACT,
    "step_samples_per_record": 16,
    "total_records": 300,
    "trainer_data_seed": None,
    "trainer_seed": 42,
}
_ENGINEERING_DECISIONS = {
    "benign_capacity_rule": (
        "retain the full Manifest Probe reservoir for provenance, then use "
        "only its frozen first 1405 rows as the effective role pool"
    ),
    "client_id_order": "benign client IDs 0..4, malicious client IDs 5..9",
    "client_sampling": (
        "seed each round with round_index and sample three clients without "
        "replacement; the explicit schedule is authoritative"
    ),
    "cross_round_samples": "local sample positions may repeat across rounds",
    "five_five_meaning": (
        "5:5 is the overall client composition, not per-round or final-record "
        "malicious balance"
    ),
    "learning_rate": (
        "use the first ten entries of the formal Task1 Main 100-round cosine; "
        "each round keeps its assigned LR constant for ten local steps"
    ),
    "local_positions": (
        "the three selected clients in a round share the same ordered local "
        "position list"
    ),
    "resume": "cross-process resume is not supported by this plan",
    "role_shuffle": (
        "each 1405-row effective role pool is shuffled with datasets seed "
        "2023, then fixed-sharded into five equal 281-row clients using "
        "Dataset.shard semantics"
    ),
    "run_complete": "the collector does not create a run-level COMPLETE marker",
    "schedule_coverage": (
        "all ten clients appear in this frozen schedule; this is a result of "
        "the authoritative schedule, not a general random-sampling guarantee"
    ),
    "trainer_seed": "Trainer seed 42 follows repository behavior, not the paper",
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "plan_sha256",
    "code_commit",
    "manifest",
    "model_artifact",
    "tokenizer_artifact",
    "runtime",
    "protocol",
    "engineering_decisions",
    "clients",
    "rounds",
    "summary",
}
_SAMPLE_KEYS = {
    "local_position",
    "sample_id",
    "source_id",
    "content_hash",
    "normalized_prompt_hash",
    "raw_row_index",
    "split_rank",
    "probe_role",
    "probe_label",
}
_CLIENT_KEYS = {
    "client_id",
    "role",
    "label",
    "role_shard_index",
    "sample_count",
    "ordered_samples_sha256",
    "samples",
}
_ROUND_KEYS = {
    "round_index",
    "selected_client_ids",
    "benign_client_count",
    "malicious_client_count",
    "local_sample_positions",
    "expected_records",
    "expected_sample_events",
    "output_round_name",
}


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX_64_RE.fullmatch(value) is None:
        raise ValueError(
            name + " must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX_40_RE.fullmatch(value) is None:
        raise ValueError(name + " must be a lowercase 40-character commit SHA")
    return value


def _require_non_bool_int(
    value: Any,
    name: str,
    minimum: int = 0,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(
            name + " must be an integer >= " + str(minimum)
        )
    return value


def _require_output_root(output_root: Any) -> Path:
    if not isinstance(output_root, (str, os.PathLike)):
        raise TypeError("output_root must be a path")
    root = Path(output_root)
    try:
        root_stat = root.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError("output root does not exist: " + str(root)) from exc
    if stat.S_ISLNK(root_stat.st_mode) or root.is_symlink():
        raise ValueError("output root must not be a symlink")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("output root must be a directory")
    if not os.access(str(root), os.W_OK):
        raise PermissionError("output root is not writable: " + str(root))
    return root.resolve()


def validate_probe_plan_output_readiness(output_root: Any) -> Path:
    """Validate plan output targets without creating or modifying anything."""
    root = _require_output_root(output_root)
    plan_target = root / PLAN_FILE_NAME
    sidecar_target = root / PLAN_SIDECAR_NAME
    if os.path.lexists(str(plan_target)):
        raise FileExistsError("Probe collection plan already exists")
    if os.path.lexists(str(sidecar_target)):
        raise FileExistsError("Probe collection plan sidecar already exists")
    return root


def _run_git(repository_root: Path, arguments: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root)] + list(arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError("Git is unavailable") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "Git command failed: "
            + " ".join(arguments)
            + (": " + detail if detail else "")
        )
    return result.stdout.strip()


def collect_probe_git_provenance(
    repository_root: Path,
    expected_code_commit: str,
) -> Dict[str, Any]:
    """Validate the exact clean branch/commit required by the formal CLI."""
    expected = _require_commit(expected_code_commit, "expected_code_commit")
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise RuntimeError("repository_root must be an existing directory")
    reported_root = Path(
        _run_git(root, ["rev-parse", "--show-toplevel"])
    ).resolve()
    if reported_root != root:
        raise RuntimeError("Probe plan CLI must use the repository root")

    branch = _run_git(root, ["branch", "--show-current"])
    detached = not branch
    if detached and not _ALLOW_DETACHED_HEAD:
        raise RuntimeError("detached HEAD is not supported")
    if not detached and branch != _EXPECTED_BRANCH:
        raise RuntimeError(
            "Probe plan requires branch " + _EXPECTED_BRANCH
        )

    commit = _run_git(root, ["rev-parse", "HEAD"]).lower()
    _require_commit(commit, "current Git commit")
    if commit != expected:
        raise ValueError("current Git commit does not match expected commit")

    status_output = _run_git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )
    if status_output:
        raise RuntimeError(
            "Git worktree and index must be clean, including untracked files"
        )
    return {
        "branch": branch or None,
        "clean": True,
        "commit": commit,
        "detached": detached,
    }


def collect_probe_collection_runtime() -> Dict[str, str]:
    distributions = {
        "torch": "torch",
        "transformers": "transformers",
        "trl": "trl",
        "accelerate": "accelerate",
        "peft": "peft",
        "datasets": "datasets",
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "safetensors": "safetensors",
    }
    runtime = {"python": platform.python_version()}
    for key, distribution in distributions.items():
        try:
            value = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "required distribution is not installed: " + distribution
            ) from exc
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                "failed to determine distribution version: " + distribution
            )
        runtime[key] = value
    return runtime


def _validate_runtime(runtime: Any) -> Dict[str, str]:
    if not isinstance(runtime, dict) or set(runtime) != set(_RUNTIME_KEYS):
        raise ValueError("runtime fields do not match Probe plan schema")
    normalized = {}
    for key in _RUNTIME_KEYS:
        value = runtime.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError("runtime." + key + " must be a non-empty string")
        normalized[key] = value
    return normalized


def _validate_manifest_runtime(
    manifest: Dict[str, Any],
    runtime: Dict[str, str],
) -> None:
    processing_runtime = manifest.get("processing_runtime")
    if not isinstance(processing_runtime, dict):
        raise ValueError("Manifest processing_runtime must be a dictionary")
    for manifest_key, plan_key in _MANIFEST_RUNTIME_MAP.items():
        if processing_runtime.get(manifest_key) != runtime[plan_key]:
            raise ValueError(
                "runtime does not match Manifest processing_runtime."
                + manifest_key
            )


def _probe_records(
    dataset_entry: Dict[str, Any],
    expected_role: str,
) -> List[Dict[str, Any]]:
    counts = dataset_entry.get("counts")
    records = dataset_entry.get("records")
    if not isinstance(counts, dict) or not isinstance(records, list):
        raise ValueError("Manifest dataset counts and records are required")
    continual = "probe" in counts and "model_total" in counts
    required = _require_non_bool_int(
        counts.get("probe" if continual else "probe_actual"),
        expected_role + " Probe reservoir count",
    )
    probe_start = _require_non_bool_int(
        counts.get("model_total" if continual else "probe_exclusion_boundary"),
        expected_role + " Probe split-rank start",
    )
    probe_records = [
        record
        for record in records
        if isinstance(record, dict) and record.get("partition") == "probe"
    ]
    if len(probe_records) != required:
        raise ValueError(expected_role + " Probe reservoir is incomplete")

    effective_required = required
    if continual:
        effective_required = _require_non_bool_int(
            dataset_entry.get("probe_effective_count"),
            expected_role + " Probe effective count",
        )
        if effective_required != PROBE_EFFECTIVE_TOTAL_PER_ROLE:
            raise ValueError(
                expected_role + " Probe effective count must equal 1405"
            )
        if effective_required > required:
            raise ValueError(
                expected_role + " Probe effective count exceeds reservoir"
            )
        stored_effective_sha = _require_sha256(
            dataset_entry.get("probe_effective_records_sha256"),
            expected_role + " Probe effective records SHA-256",
        )
        expected_effective_sha = sha256_bytes(
            canonical_json_bytes(probe_records[:effective_required])
        )
        if stored_effective_sha != expected_effective_sha:
            raise ValueError(
                expected_role + " Probe effective records SHA-256 mismatch"
            )

    result = []
    seen_source_ids = set()
    for offset, record in enumerate(probe_records):
        expected_rank = probe_start + offset
        if not isinstance(record, dict):
            raise ValueError("Manifest Probe record must be a dictionary")
        split_rank = _require_non_bool_int(
            record.get("split_rank"),
            "Probe split_rank",
        )
        if split_rank != expected_rank:
            raise ValueError(
                expected_role + " Probe split_rank coverage is incomplete"
            )
        if record.get("partition") != "probe":
            raise ValueError(
                "federated partition must not enter the Probe plan"
            )
        raw_row_index = _require_non_bool_int(
            record.get("raw_row_index"),
            "Probe raw_row_index",
        )
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Probe source_id must be a non-empty string")
        if source_id in seen_source_ids:
            raise ValueError("Probe source_id values must be unique")
        seen_source_ids.add(source_id)
        content_hash = _require_sha256(
            record.get("content_hash"),
            "Probe content_hash",
        )
        if offset < effective_required:
            normalized_prompt_hash = record.get("normalized_prompt_hash")
            if continual:
                normalized_prompt_hash = _require_sha256(
                    normalized_prompt_hash,
                    "Probe normalized_prompt_hash",
                )
            elif normalized_prompt_hash is not None:
                normalized_prompt_hash = _require_sha256(
                    normalized_prompt_hash,
                    "Probe normalized_prompt_hash",
                )
            result.append({
                "sample_id": split_rank,
                "source_id": source_id,
                "content_hash": content_hash,
                "normalized_prompt_hash": normalized_prompt_hash,
                "raw_row_index": raw_row_index,
                "split_rank": split_rank,
                "probe_role": expected_role,
                "probe_label": 0 if expected_role == "benign" else 1,
            })
    return result


@lru_cache(maxsize=16)
def _official_role_shard_ranks(
    split_rank_start: int,
    role_capacity: int,
) -> Tuple[Tuple[int, ...], ...]:
    ranks = Dataset.from_dict(
        {
            "split_rank": list(
                range(split_rank_start, split_rank_start + role_capacity)
            )
        }
    )
    shuffled = ranks.shuffle(seed=2023)
    return tuple(
        tuple(
            shuffled.shard(
                num_shards=_CLIENTS_PER_ROLE,
                index=shard_index,
            )["split_rank"]
        )
        for shard_index in range(_CLIENTS_PER_ROLE)
    )


def _ordered_samples_sha256(samples: List[Dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(samples))


def _build_clients(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    datasets = manifest["datasets"]
    role_records = {
        "benign": _probe_records(datasets[0], "benign"),
        "malicious": _probe_records(datasets[1], "malicious"),
    }
    clients = []
    global_source_ids = set()
    for role, client_offset, dataset_entry in (
        ("benign", 0, datasets[0]),
        ("malicious", 5, datasets[1]),
    ):
        samples_by_rank = {
            sample["split_rank"]: sample
            for sample in role_records[role]
        }
        continual = "probe" in dataset_entry.get("counts", {})
        pool_sizes = dataset_entry.get(
            "probe_effective_client_pool_sizes"
            if continual
            else "probe_client_pool_sizes"
        )
        if not isinstance(pool_sizes, list) or len(pool_sizes) != (
            _CLIENTS_PER_ROLE
        ):
            raise ValueError(role + " Probe client pool sizes are invalid")
        if min(pool_sizes) < PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT:
            raise ValueError(
                "each Probe client pool must contain at least 160 samples"
            )
        split_rank_start = dataset_entry["counts"][
            "model_total" if continual else "probe_exclusion_boundary"
        ]
        shard_ranks = _official_role_shard_ranks(
            split_rank_start,
            len(role_records[role]),
        )
        role_assigned_source_ids = set()
        for role_shard_index, ordered_ranks in enumerate(shard_ranks):
            if len(ordered_ranks) != pool_sizes[role_shard_index]:
                raise ValueError(
                    role + " shard size does not match Manifest"
                )
            samples = []
            for local_position, split_rank in enumerate(ordered_ranks):
                sample = copy.deepcopy(samples_by_rank[split_rank])
                sample["local_position"] = local_position
                source_id = sample["source_id"]
                if source_id in global_source_ids:
                    raise ValueError(
                        "Probe client source_id values must be globally unique"
                    )
                global_source_ids.add(source_id)
                role_assigned_source_ids.add(source_id)
                samples.append(sample)
            clients.append(
                {
                    "client_id": client_offset + role_shard_index,
                    "role": role,
                    "label": 0 if role == "benign" else 1,
                    "role_shard_index": role_shard_index,
                    "sample_count": len(samples),
                    "ordered_samples_sha256": _ordered_samples_sha256(
                        samples
                    ),
                    "samples": samples,
                }
            )
        expected_source_ids = {
            sample["source_id"] for sample in role_records[role]
        }
        if role_assigned_source_ids != expected_source_ids:
            raise ValueError(
                role + " client union does not match selected Probe capacity"
            )
    return clients


def _round_local_positions(
    round_index: int,
    clients: List[Dict[str, Any]],
) -> List[int]:
    selected = _FROZEN_CLIENT_SCHEDULE[round_index]
    selection_capacity = min(
        clients[client_id]["sample_count"] for client_id in selected
    )
    if selection_capacity < _LOCAL_SAMPLE_EVENTS:
        raise ValueError(
            "selected Probe client pool cannot provide 160 unique rows"
        )
    return random.Random(round_index).sample(
        range(selection_capacity),
        _LOCAL_SAMPLE_EVENTS,
    )


def _build_rounds(clients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rounds = []
    for round_index, selected in enumerate(_FROZEN_CLIENT_SCHEDULE):
        generated = tuple(
            sorted(
                random.Random(round_index).sample(
                    range(_CLIENT_COUNT),
                    _CLIENTS_PER_ROUND,
                )
            )
        )
        if generated != selected:
            raise RuntimeError(
                "Python client sampling does not match frozen Probe schedule"
            )
        benign_count = sum(client_id < 5 for client_id in selected)
        rounds.append(
            {
                "round_index": round_index,
                "selected_client_ids": list(selected),
                "benign_client_count": benign_count,
                "malicious_client_count": _CLIENTS_PER_ROUND - benign_count,
                "local_sample_positions": _round_local_positions(
                    round_index,
                    clients,
                ),
                "expected_records": _RECORDS_PER_ROUND,
                "expected_sample_events": (
                    _CLIENTS_PER_ROUND * _LOCAL_SAMPLE_EVENTS
                ),
                "output_round_name": "round-" + format(round_index, "03d"),
            }
        )
    return rounds


def _summary(rounds: List[Dict[str, Any]]) -> Dict[str, int]:
    benign_slots = sum(item["benign_client_count"] for item in rounds)
    malicious_slots = sum(
        item["malicious_client_count"] for item in rounds
    )
    return {
        "rounds": len(rounds),
        "client_slots": benign_slots + malicious_slots,
        "benign_client_slots": benign_slots,
        "malicious_client_slots": malicious_slots,
        "expected_records": (
            (benign_slots + malicious_slots) * _OPTIMIZER_STEPS
        ),
        "expected_sample_events": sum(
            item["expected_sample_events"] for item in rounds
        ),
        "benign_records": benign_slots * _OPTIMIZER_STEPS,
        "malicious_records": malicious_slots * _OPTIMIZER_STEPS,
    }


def _plan_digest(plan: Dict[str, Any]) -> str:
    payload = {
        key: copy.deepcopy(value)
        for key, value in plan.items()
        if key != "plan_sha256"
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _artifact_config_entry(
    model_identity: Dict[str, Any],
) -> Dict[str, Any]:
    entries = [
        entry
        for entry in model_identity["files"]
        if entry.get("relative_path") == "config.json"
    ]
    if len(entries) != 1:
        raise ValueError(
            "model artifact identity must contain exactly one config.json"
        )
    return entries[0]


def _load_identity_anchored_model_config(
    model_directory: Any,
    model_identity: Dict[str, Any],
) -> Dict[str, Any]:
    """Read config stably and prove it still matches the B2d1 identity."""
    expected_entry = _artifact_config_entry(model_identity)
    try:
        root = artifact_identity_core._coerce_directory(model_directory)
        frozen = artifact_identity_core._freeze_file_stats(
            root,
            ["config.json"],
        )
        config, current_entry, stable_snapshot = (
            artifact_identity_core._read_stable_json_file(
                root,
                "config.json",
                "config.json",
                frozen["config.json"],
            )
        )
        artifact_identity_core._verify_stable_file_snapshots(
            root,
            {"config.json": stable_snapshot},
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "config.json changed after model artifact identity was built"
        ) from exc
    if (
        current_entry.get("size_bytes") != expected_entry.get("size_bytes")
        or current_entry.get("sha256") != expected_entry.get("sha256")
    ):
        raise RuntimeError(
            "config.json changed after model artifact identity was built"
        )
    return config


def _require_profile_integer(
    config: Dict[str, Any],
    name: str,
    expected: int,
) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            "config.json does not match frozen Llama3.1-8B profile: "
            + name
            + " must be an integer"
        )
    if value != expected:
        raise ValueError(
            "config.json does not match frozen Llama3.1-8B profile: "
            + name
            + " must equal "
            + str(expected)
        )
    return expected


def _require_profile_number(
    config: Dict[str, Any],
    name: str,
    expected: float,
) -> float:
    value = config.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(
            "config.json does not match frozen Llama3.1-8B profile: "
            + name
            + " must be a finite number"
        )
    if float(value) != expected:
        raise ValueError(
            "config.json does not match frozen Llama3.1-8B profile: "
            + name
            + " must equal "
            + repr(expected)
        )
    return expected


def _normalize_llama31_model_config(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError(
            "config.json does not match frozen Llama3.1-8B profile: "
            "root must be an object"
        )
    if config.get("model_type") != "llama":
        raise ValueError(
            "config.json does not match frozen Llama3.1-8B profile: "
            "model_type must equal llama"
        )
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or any(
            not isinstance(item, str) or not item
            for item in architectures
        )
        or "LlamaForCausalLM" not in architectures
    ):
        raise ValueError(
            "config.json does not match frozen Llama3.1-8B profile: "
            "architectures must contain LlamaForCausalLM"
        )

    integer_fields = {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
    }
    for name, expected in integer_fields.items():
        _require_profile_integer(config, name, expected)
    _require_profile_number(config, "rope_theta", 500000.0)

    rope_scaling = config.get("rope_scaling")
    if not isinstance(rope_scaling, dict):
        raise ValueError(
            "config.json does not match frozen Llama3.1-8B profile: "
            "rope_scaling must be an object"
        )
    for name, expected in (
        ("factor", 8.0),
        ("high_freq_factor", 4.0),
        ("low_freq_factor", 1.0),
    ):
        _require_profile_number(rope_scaling, name, expected)
    _require_profile_integer(
        rope_scaling,
        "original_max_position_embeddings",
        8192,
    )

    legacy_type = rope_scaling.get("type")
    rope_type = rope_scaling.get("rope_type")
    if (
        legacy_type is not None
        and rope_type is not None
        and legacy_type != rope_type
    ):
        raise ValueError(
            "config.json does not match frozen Llama3.1-8B profile: "
            "rope_scaling type and rope_type conflict"
        )
    normalized_rope_type = (
        rope_type if rope_type is not None else legacy_type
    )
    if normalized_rope_type != "llama3":
        raise ValueError(
            "config.json does not match frozen Llama3.1-8B profile: "
            "rope_scaling rope_type must equal llama3"
        )
    return copy.deepcopy(_MODEL_CONFIG_PROFILE)


def build_probe_collection_plan(
    manifest_path: Any,
    manifest_sha256: str,
    model_directory: Any,
    expected_code_commit: str,
    output_root: Any,
    *,
    repository_root: Optional[Path] = None,
    runtime_provider: Callable[[], Dict[str, str]] = (
        collect_probe_collection_runtime
    ),
    git_status_provider: Callable[
        [Path, str], Dict[str, Any]
    ] = collect_probe_git_provenance,
    manifest_provenance_validator: Callable[
        [Dict[str, Any], Path], None
    ] = validate_manifest_training_provenance,
) -> Dict[str, Any]:
    """Build a plan without loading model/tokenizer objects or source data."""
    manifest_digest = _require_sha256(
        manifest_sha256,
        "manifest_sha256",
    )
    expected_commit = _require_commit(
        expected_code_commit,
        "expected_code_commit",
    )
    _require_output_root(output_root)
    repo_root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )

    git_provenance = git_status_provider(repo_root, expected_commit)
    if not isinstance(git_provenance, dict):
        raise ValueError("Git status provider must return a dictionary")
    detached = git_provenance.get("detached") is True
    if detached:
        if not _ALLOW_DETACHED_HEAD:
            raise ValueError("Git provenance reports a detached HEAD")
        if git_provenance.get("branch") not in (None, ""):
            raise ValueError("Detached Git provenance must not report a branch")
    elif git_provenance.get("branch") != _EXPECTED_BRANCH:
        raise ValueError("Git provenance branch is incorrect")
    if git_provenance.get("clean") is not True:
        raise ValueError("Git provenance must report a clean worktree")
    if git_provenance.get("commit") != expected_commit:
        raise ValueError("Git provenance commit does not match expected commit")

    manifest = load_training_manifest(
        str(manifest_path),
        manifest_digest,
    )
    if manifest is None:
        raise ValueError("Probe collection requires a data Manifest")
    validate_probe_collection_manifest(manifest)
    manifest_provenance_validator(manifest, repo_root)
    # Data preprocessing and release execution have separate identities.
    _require_commit(manifest.get("code_commit"), "manifest preprocessing commit")

    runtime = _validate_runtime(runtime_provider())
    _validate_manifest_runtime(manifest, runtime)

    model_identity = validate_artifact_identity(
        build_model_artifact_identity(model_directory)
    )
    model_config = _load_identity_anchored_model_config(
        model_directory,
        model_identity,
    )
    normalized_profile = _normalize_llama31_model_config(model_config)
    tokenizer_identity = validate_artifact_identity(
        build_tokenizer_artifact_identity(model_directory)
    )
    clients = _build_clients(manifest)
    rounds = _build_rounds(clients)
    continual = matches_schema(manifest.get("manifest_schema"), CONTINUAL_SCHEMA_VERSION)

    probe_reservoir_counts = {}
    probe_effective_counts = {}
    probe_effective_records_sha256 = {}
    probe_client_pool_sizes = {}
    probe_split_rank_ranges = {}
    for entry in manifest["datasets"]:
        role = entry["role"]
        if continual:
            reservoir_count = entry["counts"]["probe"]
            effective_count = entry["probe_effective_count"]
            split_start = entry["counts"]["model_total"]
            effective_sha = entry["probe_effective_records_sha256"]
            effective_pool_sizes = entry[
                "probe_effective_client_pool_sizes"
            ]
        else:
            reservoir_count = entry["counts"]["probe_actual"]
            effective_count = reservoir_count
            split_start = entry["counts"]["probe_exclusion_boundary"]
            effective_records = [
                record
                for record in entry["records"]
                if record.get("partition") == "probe"
            ]
            effective_sha = sha256_bytes(
                canonical_json_bytes(effective_records)
            )
            effective_pool_sizes = entry["probe_client_pool_sizes"]
        probe_reservoir_counts[role] = reservoir_count
        probe_effective_counts[role] = effective_count
        probe_effective_records_sha256[role] = effective_sha
        probe_client_pool_sizes[role] = copy.deepcopy(effective_pool_sizes)
        probe_split_rank_ranges[role] = [
            split_start,
            split_start + effective_count,
        ]

    manifest_path_object = Path(manifest_path)
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "code_commit": expected_commit,
        "manifest": {
            "manifest_sha256": manifest_digest,
            "manifest_schema": manifest.get("manifest_schema"),
            "manifest_schema_version": manifest.get("manifest_schema_version"),
            "pipeline_schema_version": manifest.get("pipeline_schema_version"),
            "code_commit": manifest["code_commit"],
            "shuffle_seed": manifest["shuffle"]["seed"],
            "diagnostic_file_name": manifest_path_object.name,
            "probe_reservoir_counts": probe_reservoir_counts,
            "probe_effective_counts": probe_effective_counts,
            "probe_effective_records_sha256": (
                probe_effective_records_sha256
            ),
            "probe_client_pool_sizes": probe_client_pool_sizes,
            "probe_split_rank_ranges": probe_split_rank_ranges,
        },
        "model_artifact": model_identity,
        "tokenizer_artifact": tokenizer_identity,
        "runtime": runtime,
        "protocol": copy.deepcopy(_PROTOCOL),
        "engineering_decisions": copy.deepcopy(_ENGINEERING_DECISIONS),
        "clients": clients,
        "rounds": rounds,
        "summary": _summary(rounds),
    }
    plan["protocol"]["model_config_profile"] = normalized_profile
    plan["plan_sha256"] = _plan_digest(plan)
    return validate_probe_collection_plan(plan)


def _validate_sample(
    sample: Any,
    expected_role: str,
    expected_label: int,
    expected_local_position: int,
) -> None:
    if not isinstance(sample, dict) or set(sample) != _SAMPLE_KEYS:
        raise ValueError("Probe plan sample fields do not match schema")
    local_position = _require_non_bool_int(
        sample.get("local_position"),
        "sample local_position",
    )
    if local_position != expected_local_position:
        raise ValueError("client local_position must be contiguous from zero")
    split_rank = _require_non_bool_int(
        sample.get("split_rank"),
        "sample split_rank",
    )
    sample_id = _require_non_bool_int(
        sample.get("sample_id"),
        "sample sample_id",
    )
    if sample_id != split_rank:
        raise ValueError("sample_id must equal split_rank")
    _require_non_bool_int(
        sample.get("raw_row_index"),
        "sample raw_row_index",
    )
    source_id = sample.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("sample source_id must be a non-empty string")
    _require_sha256(sample.get("content_hash"), "sample content_hash")
    normalized_prompt_hash = sample.get("normalized_prompt_hash")
    if normalized_prompt_hash is not None:
        _require_sha256(
            normalized_prompt_hash,
            "sample normalized_prompt_hash",
        )
    if sample.get("probe_role") != expected_role:
        raise ValueError("sample probe_role does not match client role")
    probe_label = _require_non_bool_int(
        sample.get("probe_label"),
        "sample probe_label",
    )
    if probe_label != expected_label:
        raise ValueError("sample probe_label does not match client label")


def _validate_clients(
    clients: Any,
    pool_sizes_by_role: Mapping[str, Any],
    split_ranges_by_role: Mapping[str, Any],
) -> None:
    if not isinstance(clients, list) or len(clients) != _CLIENT_COUNT:
        raise ValueError("Probe plan must contain exactly ten clients")
    if set(pool_sizes_by_role) != {"benign", "malicious"}:
        raise ValueError("Probe plan pool-size roles are incomplete")
    if set(split_ranges_by_role) != {"benign", "malicious"}:
        raise ValueError("Probe plan split-range roles are incomplete")
    for role in ("benign", "malicious"):
        sizes = pool_sizes_by_role[role]
        split_range = split_ranges_by_role[role]
        if (
            not isinstance(sizes, list)
            or len(sizes) != _CLIENTS_PER_ROLE
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT
                for value in sizes
            )
        ):
            raise ValueError(role + " Probe client pool sizes are invalid")
        if (
            not isinstance(split_range, list)
            or len(split_range) != 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in split_range
            )
            or split_range[1] - split_range[0] != sum(sizes)
        ):
            raise ValueError(role + " Probe split-rank range is invalid")
    global_source_ids = set()
    role_ranks = {"benign": set(), "malicious": set()}
    reconstructed_by_role = {"benign": {}, "malicious": {}}

    for expected_client_id, client in enumerate(clients):
        if not isinstance(client, dict) or set(client) != _CLIENT_KEYS:
            raise ValueError("Probe plan client fields do not match schema")
        client_id = _require_non_bool_int(
            client.get("client_id"),
            "client_id",
        )
        if client_id != expected_client_id:
            raise ValueError("Probe plan client IDs must be ordered 0..9")
        expected_role = "benign" if expected_client_id < 5 else "malicious"
        expected_label = 0 if expected_role == "benign" else 1
        expected_shard = expected_client_id if expected_role == "benign" else (
            expected_client_id - 5
        )
        if client.get("role") != expected_role:
            raise ValueError("Probe plan client role is incorrect")
        label = _require_non_bool_int(client.get("label"), "client label")
        if label != expected_label:
            raise ValueError("Probe plan client label is incorrect")
        role_shard_index = _require_non_bool_int(
            client.get("role_shard_index"),
            "client role_shard_index",
        )
        if role_shard_index != expected_shard:
            raise ValueError("Probe plan role_shard_index is incorrect")
        sample_count = _require_non_bool_int(
            client.get("sample_count"),
            "client sample_count",
        )
        expected_sample_count = pool_sizes_by_role[expected_role][
            expected_shard
        ]
        if sample_count != expected_sample_count:
            raise ValueError(
                "Probe client sample_count does not match Manifest pool size"
            )
        samples = client.get("samples")
        if (
            not isinstance(samples, list)
            or len(samples) != expected_sample_count
        ):
            raise ValueError(
                "Probe client samples do not match Manifest pool size"
            )
        for local_position, sample in enumerate(samples):
            _validate_sample(
                sample,
                expected_role,
                expected_label,
                local_position,
            )
            source_id = sample["source_id"]
            if source_id in global_source_ids:
                raise ValueError(
                    "Probe client source_id values must be globally unique"
                )
            global_source_ids.add(source_id)
            split_rank = sample["split_rank"]
            if split_rank in role_ranks[expected_role]:
                raise ValueError(
                    "Probe role split_rank values must not overlap"
                )
            role_ranks[expected_role].add(split_rank)
            reconstructed_by_role[expected_role][split_rank] = sample
        if client.get("ordered_samples_sha256") != (
            _ordered_samples_sha256(samples)
        ):
            raise ValueError("ordered_samples_sha256 does not match samples")

    for role, offset in (("benign", 0), ("malicious", 5)):
        split_start, split_end = split_ranges_by_role[role]
        expected_ranks = set(range(split_start, split_end))
        if role_ranks[role] != expected_ranks:
            raise ValueError(
                role + " clients do not cover the Manifest Probe rank range"
            )
        shard_ranks = _official_role_shard_ranks(
            split_start,
            split_end - split_start,
        )
        for shard_index, expected_order in enumerate(shard_ranks):
            actual_order = [
                sample["split_rank"]
                for sample in clients[offset + shard_index]["samples"]
            ]
            if actual_order != list(expected_order):
                raise ValueError(
                    "client sample order does not match frozen role shuffle/shard"
                )


def _validate_rounds(
    rounds: Any,
    clients: List[Dict[str, Any]],
) -> None:
    if not isinstance(rounds, list) or len(rounds) != _ROUND_COUNT:
        raise ValueError("Probe plan must contain exactly ten rounds")
    covered_clients = set()
    for round_index, item in enumerate(rounds):
        if not isinstance(item, dict) or set(item) != _ROUND_KEYS:
            raise ValueError("Probe plan round fields do not match schema")
        stored_round_index = _require_non_bool_int(
            item.get("round_index"),
            "round_index",
        )
        if stored_round_index != round_index:
            raise ValueError("round_index must be contiguous from zero")
        expected_clients = list(_FROZEN_CLIENT_SCHEDULE[round_index])
        selected = item.get("selected_client_ids")
        if canonical_json_bytes(selected) != canonical_json_bytes(
            expected_clients
        ):
            raise ValueError("round client schedule does not match frozen plan")
        if len(selected) != len(set(selected)):
            raise ValueError("each round must contain three distinct clients")
        covered_clients.update(selected)
        benign_count = sum(client_id < 5 for client_id in selected)
        stored_benign_count = _require_non_bool_int(
            item.get("benign_client_count"),
            "round benign_client_count",
        )
        if stored_benign_count != benign_count:
            raise ValueError("round benign_client_count is incorrect")
        stored_malicious_count = _require_non_bool_int(
            item.get("malicious_client_count"),
            "round malicious_client_count",
        )
        if stored_malicious_count != (
            _CLIENTS_PER_ROUND - benign_count
        ):
            raise ValueError("round malicious_client_count is incorrect")

        positions = item.get("local_sample_positions")
        expected_positions = _round_local_positions(round_index, clients)
        if canonical_json_bytes(positions) != canonical_json_bytes(
            expected_positions
        ):
            raise ValueError(
                "round local_sample_positions do not match frozen order"
            )
        if len(positions) != _LOCAL_SAMPLE_EVENTS:
            raise ValueError("each round must contain 160 local positions")
        if len(set(positions)) != len(positions):
            raise ValueError("round local positions must be unique")
        for position in positions:
            _require_non_bool_int(position, "round local position")
            selected_capacity = min(
                clients[client_id]["sample_count"]
                for client_id in selected
            )
            if position >= selected_capacity:
                raise ValueError(
                    "round local position exceeds a selected client pool"
                )
        expected_records = _require_non_bool_int(
            item.get("expected_records"),
            "round expected_records",
        )
        if expected_records != _RECORDS_PER_ROUND:
            raise ValueError("round expected_records must be 30")
        expected_sample_events = _require_non_bool_int(
            item.get("expected_sample_events"),
            "round expected_sample_events",
        )
        if expected_sample_events != 480:
            raise ValueError("round expected_sample_events must be 480")
        if item.get("output_round_name") != (
            "round-" + format(round_index, "03d")
        ):
            raise ValueError("round output name is incorrect")
    if covered_clients != set(range(_CLIENT_COUNT)):
        raise ValueError(
            "the current frozen schedule must cover all ten clients"
        )


def validate_probe_collection_plan(
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Strictly validate structure, frozen protocol, and canonical digest."""
    if not isinstance(plan, dict):
        raise TypeError("plan must be a dictionary")
    if set(plan) != _TOP_LEVEL_KEYS:
        raise ValueError("Probe plan top-level fields do not match schema")
    if not matches_schema(plan.get("schema_version"), PLAN_SCHEMA_VERSION):
        raise ValueError("unsupported Probe collection plan schema")
    code_commit = _require_commit(plan.get("code_commit"), "code_commit")
    _require_sha256(plan.get("plan_sha256"), "plan_sha256")

    manifest = plan.get("manifest")
    expected_manifest_keys = {
        "manifest_sha256",
        "manifest_schema",
        "manifest_schema_version",
        "pipeline_schema_version",
        "code_commit",
        "shuffle_seed",
        "diagnostic_file_name",
        "probe_reservoir_counts",
        "probe_effective_counts",
        "probe_effective_records_sha256",
        "probe_client_pool_sizes",
        "probe_split_rank_ranges",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        raise ValueError("Probe plan Manifest provenance fields are incorrect")
    _require_sha256(
        manifest.get("manifest_sha256"),
        "manifest.manifest_sha256",
    )
    if manifest.get("code_commit") != code_commit:
        raise ValueError("Manifest code_commit does not match plan code_commit")
    shuffle_seed = _require_non_bool_int(
        manifest.get("shuffle_seed"),
        "Manifest shuffle seed",
    )
    if shuffle_seed != 2023:
        raise ValueError("Manifest shuffle seed must be 2023")
    if (
        not isinstance(manifest.get("diagnostic_file_name"), str)
        or not manifest["diagnostic_file_name"]
        or Path(manifest["diagnostic_file_name"]).name
        != manifest["diagnostic_file_name"]
    ):
        raise ValueError("Manifest diagnostic file name must be a basename")
    continual = matches_schema(manifest.get("manifest_schema"), CONTINUAL_SCHEMA_VERSION)
    if continual:
        if manifest.get("manifest_schema_version") is not None:
            raise ValueError(
                "continual Manifest must not set legacy schema version"
            )
        if manifest.get("pipeline_schema_version") is not None:
            raise ValueError(
                "continual Manifest must not set legacy pipeline schema version"
            )
    else:
        if manifest.get("manifest_schema") is not None:
            raise ValueError("Manifest schema is unsupported")
        manifest_schema_version = _require_non_bool_int(
            manifest.get("manifest_schema_version"),
            "Manifest schema version",
            1,
        )
        if manifest_schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("Manifest schema version is unsupported")
        pipeline_schema_version = _require_non_bool_int(
            manifest.get("pipeline_schema_version"),
            "Manifest pipeline schema version",
            1,
        )
        if pipeline_schema_version != PIPELINE_SCHEMA_VERSION:
            raise ValueError("Manifest pipeline schema version is unsupported")

    reservoir_counts = manifest.get("probe_reservoir_counts")
    effective_counts = manifest.get("probe_effective_counts")
    effective_record_sha = manifest.get("probe_effective_records_sha256")
    effective_pool_sizes = manifest.get("probe_client_pool_sizes")
    for name, mapping in (
        ("Probe reservoir counts", reservoir_counts),
        ("Probe effective counts", effective_counts),
        ("Probe effective records SHA-256", effective_record_sha),
    ):
        if not isinstance(mapping, dict) or set(mapping) != {
            "benign",
            "malicious",
        }:
            raise ValueError(name + " roles are incomplete")
    if not isinstance(effective_pool_sizes, dict) or set(
        effective_pool_sizes
    ) != {"benign", "malicious"}:
        raise ValueError("Probe client pool-size roles are incomplete")
    for role in ("benign", "malicious"):
        reservoir_count = _require_non_bool_int(
            reservoir_counts[role],
            role + " Probe reservoir count",
        )
        effective_count = _require_non_bool_int(
            effective_counts[role],
            role + " Probe effective count",
        )
        if effective_count > reservoir_count:
            raise ValueError(role + " Probe effective count exceeds reservoir")
        role_pool_sizes = effective_pool_sizes[role]
        if not isinstance(role_pool_sizes, list) or sum(
            value
            for value in role_pool_sizes
            if isinstance(value, int) and not isinstance(value, bool)
        ) != effective_count:
            raise ValueError(
                role + " Probe effective count does not match client pools"
            )
        if continual and effective_count != PROBE_EFFECTIVE_TOTAL_PER_ROLE:
            raise ValueError(role + " Probe effective count must equal 1405")
        if continual and role_pool_sizes != (
            probe_effective_client_pool_sizes()
        ):
            raise ValueError(role + " Probe effective pools must be 281 x 5")
        _require_sha256(
            effective_record_sha[role],
            role + " Probe effective records SHA-256",
        )

    model_identity = validate_artifact_identity(plan.get("model_artifact"))
    tokenizer_identity = validate_artifact_identity(
        plan.get("tokenizer_artifact")
    )
    if model_identity.get("artifact_type") != "model":
        raise ValueError("model_artifact must contain a model identity")
    if tokenizer_identity.get("artifact_type") != "tokenizer":
        raise ValueError(
            "tokenizer_artifact must contain a tokenizer identity"
        )
    _validate_runtime(plan.get("runtime"))
    if canonical_json_bytes(plan.get("protocol")) != canonical_json_bytes(
        _PROTOCOL
    ):
        raise ValueError("Probe plan protocol does not match frozen protocol")
    if canonical_json_bytes(
        plan.get("engineering_decisions")
    ) != canonical_json_bytes(_ENGINEERING_DECISIONS):
        raise ValueError("Probe plan engineering decisions are incomplete")
    clients = plan.get("clients")
    _validate_clients(
        clients,
        manifest.get("probe_client_pool_sizes"),
        manifest.get("probe_split_rank_ranges"),
    )
    if continual:
        for client in clients:
            for sample in client["samples"]:
                _require_sha256(
                    sample.get("normalized_prompt_hash"),
                    "continual sample normalized_prompt_hash",
                )
    _validate_rounds(plan.get("rounds"), clients)

    expected_summary = {
        "rounds": 10,
        "client_slots": 30,
        "benign_client_slots": 16,
        "malicious_client_slots": 14,
        "expected_records": 300,
        "expected_sample_events": 4800,
        "benign_records": 160,
        "malicious_records": 140,
    }
    if canonical_json_bytes(plan.get("summary")) != canonical_json_bytes(
        expected_summary
    ):
        raise ValueError("Probe plan summary does not match frozen schedule")
    if _plan_digest(plan) != plan["plan_sha256"]:
        raise ValueError("plan_sha256 does not match canonical plan payload")
    return copy.deepcopy(plan)


def _reject_duplicate_json_keys(
    pairs: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + str(key))
        result[key] = value
    return result


def _read_sidecar(sidecar_path: Path) -> str:
    raw = sidecar_path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("plan digest sidecar must be ASCII") from exc
    if not re.fullmatch(r"[0-9a-f]{64}\n", text):
        raise ValueError("plan digest sidecar format is invalid")
    return text[:-1]


def load_and_validate_probe_collection_plan(
    path: Any,
    *,
    expected_plan_sha256: Optional[str] = None,
    expected_manifest_sha256: Optional[str] = None,
    expected_code_commit: Optional[str] = None,
    expected_model_artifact_sha256: Optional[str] = None,
    expected_tokenizer_artifact_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    plan_path = Path(path)
    sidecar_path = Path(str(plan_path) + ".sha256")
    raw_bytes = plan_path.read_bytes()
    try:
        plan = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("non-standard JSON constant: " + value)
            ),
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Probe collection plan is not valid strict JSON") from exc
    if not isinstance(plan, dict):
        raise ValueError("Probe collection plan root must be an object")
    validated = validate_probe_collection_plan(plan)
    if raw_bytes != canonical_json_bytes(validated):
        raise ValueError("Probe collection plan JSON is not canonical")

    sidecar_digest = _read_sidecar(sidecar_path)
    if sidecar_digest != validated["plan_sha256"]:
        raise ValueError("plan digest sidecar does not match plan_sha256")
    if expected_plan_sha256 is not None:
        if _require_sha256(
            expected_plan_sha256,
            "expected_plan_sha256",
        ) != validated["plan_sha256"]:
            raise ValueError("plan does not match expected plan SHA-256")
    expected_pairs = (
        (
            expected_manifest_sha256,
            validated["manifest"]["manifest_sha256"],
            "Manifest",
        ),
        (
            expected_model_artifact_sha256,
            validated["model_artifact"]["artifact_sha256"],
            "model artifact",
        ),
        (
            expected_tokenizer_artifact_sha256,
            validated["tokenizer_artifact"]["artifact_sha256"],
            "tokenizer artifact",
        ),
    )
    for expected, actual, name in expected_pairs:
        if expected is not None:
            _require_sha256(expected, "expected " + name + " SHA-256")
            if expected != actual:
                raise ValueError(name + " SHA-256 does not match expected")
    if expected_code_commit is not None:
        _require_commit(expected_code_commit, "expected_code_commit")
        if expected_code_commit != validated["code_commit"]:
            raise ValueError("plan code_commit does not match expected")
    return validated


def _write_synced_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _unlink_if_same_file(final_path: Path, staging_path: Path) -> None:
    try:
        final_stat = os.stat(final_path, follow_symlinks=False)
        staging_stat = os.stat(staging_path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        final_stat.st_dev,
        final_stat.st_ino,
    ) != (
        staging_stat.st_dev,
        staging_stat.st_ino,
    ):
        return
    try:
        final_path.unlink()
    except FileNotFoundError:
        return


def write_probe_collection_plan(
    plan: Dict[str, Any],
    output_root: Any,
) -> str:
    """Publish the canonical plan pair no-clobber via same-directory staging."""
    validated = validate_probe_collection_plan(plan)
    root = validate_probe_plan_output_readiness(output_root)
    final_plan = root / PLAN_FILE_NAME
    final_sidecar = root / PLAN_SIDECAR_NAME
    if os.path.lexists(str(final_plan)):
        raise FileExistsError("Probe collection plan already exists")
    if os.path.lexists(str(final_sidecar)):
        raise FileExistsError("Probe collection plan sidecar already exists")

    plan_bytes = canonical_json_bytes(validated)
    sidecar_bytes = (validated["plan_sha256"] + "\n").encode("ascii")
    created_links: List[Tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(
        dir=str(root),
        prefix=".probe-collection-plan.",
    ) as staging_directory:
        staging_root = Path(staging_directory)
        staging_plan = staging_root / PLAN_FILE_NAME
        staging_sidecar = staging_root / PLAN_SIDECAR_NAME
        _write_synced_bytes(staging_plan, plan_bytes)
        _write_synced_bytes(staging_sidecar, sidecar_bytes)
        load_and_validate_probe_collection_plan(
            staging_plan,
            expected_plan_sha256=validated["plan_sha256"],
        )
        try:
            os.link(str(staging_plan), str(final_plan))
            created_links.append((final_plan, staging_plan))
            os.link(str(staging_sidecar), str(final_sidecar))
            created_links.append((final_sidecar, staging_sidecar))
            load_and_validate_probe_collection_plan(
                final_plan,
                expected_plan_sha256=validated["plan_sha256"],
                expected_manifest_sha256=validated["manifest"][
                    "manifest_sha256"
                ],
                expected_code_commit=validated["code_commit"],
                expected_model_artifact_sha256=validated["model_artifact"][
                    "artifact_sha256"
                ],
                expected_tokenizer_artifact_sha256=validated[
                    "tokenizer_artifact"
                ]["artifact_sha256"],
            )
        except Exception:
            for final_path, staging_path in reversed(created_links):
                _unlink_if_same_file(final_path, staging_path)
            raise
    return validated["plan_sha256"]

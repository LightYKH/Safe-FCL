"""Canonical client, sample, batch, and LR schedules for formal runs."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

from .artifact_schema import matches_schema

SCHEMA_VERSION = "safe-fcl.continual-schedule.v1"
MAIN_LEARNING_RATE_START = 5e-5
MAIN_LEARNING_RATE_END = 1e-6
FORMAL_SHADOW_LEARNING_RATE = 5e-5
MAIN_LEARNING_RATE_SCHEDULE = "per-task frozen 100-round cosine 5e-5->1e-6"
PROBE_COLLECTION_LR_POLICY = "task1_main_100_round_cosine_first_10"
PROBE_COLLECTION_STATE_CONTRACT = "advancing-round early-FL collection"
SHADOW_DETECTION_CONTRACT = (
    "stage-rebased Shadow with original fixed-LR detection contract"
)
TASK_NAMES = {1: '财产、身份及信息欺骗', 2: '数字安全', 3: '身体、公共及物质安全', 4: '隐私与信息保护', 5: '歧视与人际伤害'}

def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_cosine_lr(stage_round_zero_based: int) -> float:
    if not isinstance(stage_round_zero_based, int) or isinstance(stage_round_zero_based, bool):
        raise TypeError("stage round must be an integer")
    if not 0 <= stage_round_zero_based < 100:
        raise ValueError("stage round must be in [0, 99]")
    return MAIN_LEARNING_RATE_END + 0.5 * (
        MAIN_LEARNING_RATE_START - MAIN_LEARNING_RATE_END
    ) * (
        1.0 + math.cos(math.pi * stage_round_zero_based / 99)
    )


def validate_formal_shadow_learning_rate(value: Any) -> float:
    """Return the frozen Shadow LR, rejecting any formal-contract drift."""
    if isinstance(value, bool):
        raise TypeError("shadow_learning_rate must be a real number")
    try:
        learning_rate = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("shadow_learning_rate must be a real number") from exc
    if not math.isfinite(learning_rate):
        raise ValueError("shadow_learning_rate must be finite")
    if abs(learning_rate - FORMAL_SHADOW_LEARNING_RATE) > 1e-15:
        raise ValueError(
            "formal Shadow requires shadow_learning_rate=5e-5"
        )
    return FORMAL_SHADOW_LEARNING_RATE


def load_frozen_schedule(
    path: str,
    expected_sha256: str,
    *,
    task_id: int,
    continual_round_offset: int,
) -> Dict[str, Any]:
    schedule_path = Path(path).expanduser().resolve()
    if not schedule_path.is_file():
        raise FileNotFoundError("schedule does not exist: " + str(schedule_path))
    actual_sha = sha256_file(schedule_path)
    if actual_sha != str(expected_sha256).lower():
        raise ValueError("schedule SHA-256 mismatch")
    raw = schedule_path.read_bytes()
    schedule = json.loads(raw.decode("utf-8"))
    if raw != canonical_bytes(schedule):
        raise ValueError("schedule JSON is not canonical")
    if not matches_schema(schedule.get("schema_version"), SCHEMA_VERSION):
        raise ValueError("unsupported schedule schema")
    if schedule.get("task_id") != task_id:
        raise ValueError("schedule task_id mismatch")
    if schedule.get("task_name") != TASK_NAMES.get(task_id):
        raise ValueError("schedule task_name mismatch")
    if schedule.get("continual_round_offset") != continual_round_offset:
        raise ValueError("schedule continual_round_offset mismatch")
    # Validate serialized schedule metadata independently of trainer batch size.
    frozen = schedule.get("frozen_parameters")
    expected = {
        "batch_size": 2,
        "client_pool_size": 500,
        "clients_per_round": 3,
        "gradient_accumulation_steps": 8,
        "local_optimizer_steps": 10,
        "num_clients": 10,
        "num_rounds": 100,
        "samples_per_selected_client": 160,
    }
    if frozen != expected:
        raise ValueError("schedule frozen_parameters mismatch")
    rounds = schedule.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 100:
        raise ValueError("schedule must contain exactly 100 rounds")
    for index, entry in enumerate(rounds):
        if entry.get("stage_round") != index + 1:
            raise ValueError("schedule stage_round sequence mismatch")
        if entry.get("global_round") != continual_round_offset + index + 1:
            raise ValueError("schedule global_round sequence mismatch")
        if float(entry.get("learning_rate")) != frozen_cosine_lr(index):
            raise ValueError("schedule learning rate mismatch")
        clients = entry.get("clients")
        if not isinstance(clients, list) or len(clients) != 3 or clients != sorted(set(clients)):
            raise ValueError("schedule client selection is invalid")
        positions = entry.get("sample_positions")
        batches = entry.get("batches")
        if not isinstance(positions, list) or len(positions) != 160 or len(set(positions)) != 160:
            raise ValueError("schedule sample_positions are invalid")
        if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 500 for value in positions):
            raise ValueError("schedule sample position is outside [0, 499]")
        if batches != [positions[start:start + 16] for start in range(0, 160, 16)]:
            raise ValueError("schedule batch boundaries mismatch")
    schedule["_path"] = str(schedule_path)
    schedule["_sha256"] = actual_sha
    return schedule


def schedule_round(schedule: Dict[str, Any], stage_round_zero_based: int) -> Dict[str, Any]:
    if not 0 <= stage_round_zero_based < 100:
        raise ValueError("stage round must be in [0, 99]")
    return schedule["rounds"][stage_round_zero_based]

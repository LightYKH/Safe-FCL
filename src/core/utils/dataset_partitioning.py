"""Frozen data-boundary rules for the Llama3.1 paper reproduction."""

from typing import Any, Dict, List


FEDERATED_NUM_DATA_PER_CLIENT = 500
PROBE_CLIENTS_PER_ROLE = 5
PROBE_TARGET_DATA_PER_CLIENT = 500
PROBE_TARGET_TOTAL = PROBE_CLIENTS_PER_ROLE * PROBE_TARGET_DATA_PER_CLIENT
PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT = 160
PROBE_EFFECTIVE_TOTAL_PER_ROLE = 1405

_BENIGN_EXCLUSION_DATASETS = {
    "lmsys/lmsys-chat-1m",
    "allenai/WildChat",
}
_MALICIOUS_EXCLUSION_DATASETS = {
    "PKU-Alignment/BeaverTails",
    "MaliciousGen",
}


def _require_non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(name + " must be a non-negative integer")
    return value


def probe_exclusion_boundary(dataset_id: str, role: str) -> int:
    """Return the frozen initial-shuffle prefix excluded from Probe use."""
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")
    if role not in ("benign", "malicious"):
        raise ValueError("role must be benign or malicious")
    if dataset_id in _BENIGN_EXCLUSION_DATASETS:
        return 4000
    if dataset_id in _MALICIOUS_EXCLUSION_DATASETS:
        return 2500
    # Synthetic/unit-test datasets retain the role-derived paper boundary.
    return 4000 if role == "benign" else 2500


def probe_client_pool_sizes(probe_actual_total: int) -> List[int]:
    """Return Dataset.shard-compatible non-contiguous shard lengths."""
    total = _require_non_negative_int(
        probe_actual_total,
        "probe_actual_total",
    )
    base, remainder = divmod(total, PROBE_CLIENTS_PER_ROLE)
    return [
        base + (1 if shard_index < remainder else 0)
        for shard_index in range(PROBE_CLIENTS_PER_ROLE)
    ]


def probe_effective_client_pool_sizes() -> List[int]:
    """Return the frozen 1405-per-role effective Probe split (281 x 5)."""
    sizes = probe_client_pool_sizes(PROBE_EFFECTIVE_TOTAL_PER_ROLE)
    if sizes != [281] * PROBE_CLIENTS_PER_ROLE:
        raise RuntimeError("effective Probe pool must be exactly 281 rows x 5 clients")
    if min(sizes) < PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT:
        raise RuntimeError("effective Probe clients cannot provide 160 unique rows")
    return sizes


def build_partition_counts(
    *,
    dataset_id: str,
    role: str,
    role_client_count: int,
    federated_num_data_per_client: int,
    after_format_and_dedup: int,
) -> Dict[str, Any]:
    """Build the frozen partition-count metadata for one active dataset."""
    client_count = _require_non_negative_int(
        role_client_count,
        "role_client_count",
    )
    federated_per_client = _require_non_negative_int(
        federated_num_data_per_client,
        "federated_num_data_per_client",
    )
    if federated_per_client <= 0:
        raise ValueError(
            "federated_num_data_per_client must be a positive integer"
        )
    usable = _require_non_negative_int(
        after_format_and_dedup,
        "after_format_and_dedup",
    )

    exclusion_boundary = probe_exclusion_boundary(dataset_id, role)
    federated_required = client_count * federated_per_client
    if federated_required > exclusion_boundary:
        raise ValueError(
            "federated_required exceeds the fixed Probe exclusion boundary"
        )

    available_after_boundary = max(usable - exclusion_boundary, 0)
    if dataset_id == "MaliciousGen":
        probe_actual = min(PROBE_TARGET_TOTAL, available_after_boundary)
    else:
        if available_after_boundary < PROBE_TARGET_TOTAL:
            raise ValueError(
                dataset_id
                + " has insufficient data for the 2500-row Probe target"
            )
        probe_actual = PROBE_TARGET_TOTAL

    pool_sizes = probe_client_pool_sizes(probe_actual)
    if min(pool_sizes, default=0) < PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT:
        raise ValueError(
            "each Probe client pool must contain at least 160 rows"
        )

    partition_scan_end = exclusion_boundary + probe_actual
    return {
        "federated_required": federated_required,
        "probe_exclusion_boundary": exclusion_boundary,
        "probe_target": PROBE_TARGET_TOTAL,
        "probe_actual": probe_actual,
        "excluded": exclusion_boundary - federated_required,
        "partition_scan_end": partition_scan_end,
        "manifest_record_count": federated_required + probe_actual,
        "probe_client_pool_sizes": pool_sizes,
    }

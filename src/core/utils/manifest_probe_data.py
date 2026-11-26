from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from .artifact_schema import matches_schema
from .continual_data_manifest import SCHEMA_VERSION as CONTINUAL_SCHEMA_VERSION
from .dataset_partitioning import (
    FEDERATED_NUM_DATA_PER_CLIENT,
    PROBE_CLIENTS_PER_ROLE,
    PROBE_EFFECTIVE_TOTAL_PER_ROLE,
    PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT,
    PROBE_TARGET_DATA_PER_CLIENT,
    PROBE_TARGET_TOTAL,
    probe_client_pool_sizes,
    probe_effective_client_pool_sizes,
)
from .manifest_training_data import (
    load_manifest_source_dataset,
    select_manifest_partition,
    verify_selected_manifest_dataset,
)
from .process_dataset import prepare_selected_manifest_dataset
from .process_dataset import rebuild_continual_manifest_records


def _require_non_bool_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(name + " must be an integer")
    return value


def validate_probe_collection_manifest(
    manifest: Dict[str, Any],
) -> None:
    """Validate the fixed five-client Probe reservoirs before source I/O."""
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")

    continual_schema = (
        matches_schema(manifest.get("manifest_schema"), CONTINUAL_SCHEMA_VERSION)
    )
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise ValueError(
            "Probe collection Manifest must contain benign and malicious entries"
        )

    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Probe collection Manifest experiment must be a dictionary")
    if experiment.get("probe_clients_per_role") != PROBE_CLIENTS_PER_ROLE:
        raise ValueError("Probe collection requires 5 clients per role")
    if (
        experiment.get("federated_num_data_per_client")
        != FEDERATED_NUM_DATA_PER_CLIENT
    ):
        raise ValueError(
            "federated_num_data_per_client must remain 500"
        )
    if (
        experiment.get("probe_target_data_per_client")
        != PROBE_TARGET_DATA_PER_CLIENT
    ):
        raise ValueError("Probe target data per client must be 500")
    if (
        experiment.get("probe_min_samples_per_selected_client")
        != PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT
    ):
        raise ValueError("Probe minimum selected-client capacity must be 160")
    if continual_schema and (
        experiment.get("probe_effective_count_per_role")
        != PROBE_EFFECTIVE_TOTAL_PER_ROLE
    ):
        raise ValueError("continual Probe effective count per role must be 1405")

    for dataset_entry, expected_role in zip(
        datasets,
        ("benign", "malicious"),
    ):
        if not isinstance(dataset_entry, dict):
            raise ValueError("Probe collection dataset entry must be a dictionary")
        if dataset_entry.get("role") != expected_role:
            raise ValueError(
                "Probe collection Manifest datasets must be ordered "
                "benign then malicious"
            )
        if dataset_entry.get("active") is not True:
            raise ValueError(
                "Probe collection requires active "
                + expected_role
                + " data before source I/O"
            )

        counts = dataset_entry.get("counts")
        if not isinstance(counts, dict):
            raise ValueError(
                expected_role + " Probe collection counts must be a dictionary"
            )
        probe_actual = _require_non_bool_int(
            counts.get("probe" if continual_schema else "probe_actual"),
            expected_role + " probe_actual",
        )
        pool_sizes = dataset_entry.get("probe_client_pool_sizes")
        if pool_sizes != probe_client_pool_sizes(probe_actual):
            raise ValueError(
                expected_role
                + " probe_client_pool_sizes do not match probe_actual"
            )
        if len(pool_sizes) != PROBE_CLIENTS_PER_ROLE:
            raise ValueError("Probe collection requires five client pools per role")
        if min(pool_sizes) < PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT:
            raise ValueError("each Probe client pool must contain at least 160 rows")

        if continual_schema:
            if probe_actual > PROBE_TARGET_TOTAL:
                raise ValueError("continual Probe cannot exceed target capacity")
            if counts.get("probe_target") != PROBE_TARGET_TOTAL:
                raise ValueError("continual Probe target must remain 2500")
            if probe_actual < PROBE_EFFECTIVE_TOTAL_PER_ROLE:
                raise ValueError(
                    "continual Probe reservoir cannot provide 1405 effective rows"
                )
            if (
                dataset_entry.get("probe_effective_count")
                != PROBE_EFFECTIVE_TOTAL_PER_ROLE
            ):
                raise ValueError("continual Probe effective count must be 1405")
            if dataset_entry.get(
                "probe_effective_client_pool_sizes"
            ) != probe_effective_client_pool_sizes():
                raise ValueError(
                    "continual Probe effective client pools must be 281 x 5"
                )
        elif dataset_entry.get("dataset_id") == "MaliciousGen":
            if probe_actual > PROBE_TARGET_TOTAL:
                raise ValueError("MaliciousGen Probe cannot exceed target capacity")
        elif probe_actual != PROBE_TARGET_TOTAL:
            raise ValueError(
                "non-MaliciousGen Probe reservoir must contain exactly 2500 rows"
            )

    shuffle = manifest.get("shuffle")
    if not isinstance(shuffle, dict):
        raise ValueError("Probe collection Manifest shuffle must be a dictionary")
    shuffle_seed = _require_non_bool_int(
        shuffle.get("seed"),
        "Probe collection Manifest shuffle seed",
    )
    if shuffle_seed < 0:
        raise ValueError(
            "Probe collection Manifest shuffle seed must be non-negative"
        )


def verify_probe_partition_dataset(
    probe_dataset: Dataset,
    dataset_entry: Dict[str, Any],
) -> None:
    """Verify the reconstructed Probe reservoir at its global split ranks."""
    if not isinstance(probe_dataset, Dataset):
        raise TypeError("probe_dataset must be a datasets.Dataset")
    if not isinstance(dataset_entry, dict):
        raise TypeError("dataset_entry must be a dictionary")
    records = dataset_entry.get("records")
    if not isinstance(records, list):
        raise ValueError("Manifest dataset records must be a list")
    probe_records = [
        record
        for record in records
        if isinstance(record, dict) and record.get("partition") == "probe"
    ]
    if len(probe_dataset) != len(probe_records):
        raise ValueError("Selected Probe reservoir length does not match Manifest")

    required_columns = {
        "content_hash",
        "raw_row_index",
        "sample_id",
        "source_id",
        "split_rank",
    }
    continual_records = any(
        "normalized_prompt_hash" in record for record in probe_records
    )
    if continual_records:
        required_columns.add("normalized_prompt_hash")
    missing_columns = required_columns.difference(probe_dataset.column_names)
    if missing_columns:
        raise ValueError(
            "Selected Probe reservoir is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    compared_fields = [
        "raw_row_index",
        "source_id",
        "content_hash",
        "split_rank",
    ]
    if continual_records:
        compared_fields.append("normalized_prompt_hash")
    for row_index, record in enumerate(probe_records):
        row = probe_dataset[row_index]
        for field_name in compared_fields:
            if row[field_name] != record.get(field_name):
                raise ValueError(
                    "Selected Probe reservoir "
                    + field_name
                    + " does not match Manifest record"
                )
        if row["sample_id"] != row["split_rank"]:
            raise ValueError("Probe sample_id must equal global split_rank")

    model_records = [
        record
        for record in records
        if isinstance(record, dict) and record.get("partition") != "probe"
    ]
    identity_fields = ["source_id", "content_hash"]
    if continual_records:
        identity_fields.append("normalized_prompt_hash")
    for field_name in identity_fields:
        model_values = {record[field_name] for record in model_records}
        if set(probe_dataset[field_name]) & model_values:
            raise ValueError("Probe and model " + field_name + " values overlap")


def select_probe_collection_capacity(
    probe_dataset: Dataset,
    role: str,
    *,
    expected_total: Optional[int] = None,
    effective_total: Optional[int] = None,
    client_pool_sizes: Optional[List[int]] = None,
) -> Dataset:
    """Select a deterministic reservoir prefix, then attach Probe labels."""
    if not isinstance(probe_dataset, Dataset):
        raise TypeError("probe_dataset must be a datasets.Dataset")
    if role not in ("benign", "malicious"):
        raise ValueError("Probe collection role must be benign or malicious")
    if "probe_role" in probe_dataset.column_names:
        raise ValueError("probe_dataset must not already contain probe_role")
    if "probe_label" in probe_dataset.column_names:
        raise ValueError("probe_dataset must not already contain probe_label")
    for column_name in ("split_rank", "sample_id"):
        if column_name not in probe_dataset.column_names:
            raise ValueError("probe_dataset must contain " + column_name)

    actual_total = len(probe_dataset)
    if expected_total is not None and actual_total != expected_total:
        raise ValueError("Probe reservoir length does not match probe_actual")
    selected_total = actual_total if effective_total is None else effective_total
    if (
        not isinstance(selected_total, int)
        or isinstance(selected_total, bool)
        or selected_total < 0
        or selected_total > actual_total
    ):
        raise ValueError("Probe effective total is outside reservoir capacity")
    probe_dataset = probe_dataset.select(range(selected_total))
    expected_sizes = probe_client_pool_sizes(selected_total)
    if client_pool_sizes is not None and client_pool_sizes != expected_sizes:
        raise ValueError("Probe client pool sizes do not match effective length")
    if min(expected_sizes, default=0) < PROBE_MIN_SAMPLES_PER_SELECTED_CLIENT:
        raise ValueError("each Probe client pool must contain at least 160 rows")
    if probe_dataset["sample_id"] != probe_dataset["split_rank"]:
        raise ValueError("Probe sample_id must equal global split_rank")
    if selected_total:
        first_rank = probe_dataset["split_rank"][0]
        if probe_dataset["split_rank"] != list(
            range(first_rank, first_rank + selected_total)
        ):
            raise ValueError(
                "Probe split_rank must be contiguous in the global range"
            )

    label = 0 if role == "benign" else 1
    result = probe_dataset.add_column(
        "probe_role",
        [role] * selected_total,
    )
    return result.add_column(
        "probe_label",
        [label] * selected_total,
    )


def _verify_probe_prepared_counts(
    prepared: Dict[str, Any],
    dataset_entry: Dict[str, Any],
) -> None:
    prepared_counts = prepared.get("counts")
    manifest_counts = dataset_entry.get("counts")
    if not isinstance(prepared_counts, dict):
        raise ValueError("Prepared Probe counts must be a dictionary")
    if not isinstance(manifest_counts, dict):
        raise ValueError("Manifest Probe counts must be a dictionary")
    for count_name in (
        "raw",
        "after_semantic_filter",
        "after_format_and_dedup",
    ):
        if prepared_counts.get(count_name) != manifest_counts.get(count_name):
            raise ValueError(
                "Prepared Probe dataset count does not match Manifest: "
                + count_name
            )
    if prepared_counts.get("required_total") != manifest_counts.get(
        "partition_scan_end"
    ):
        raise ValueError(
            "Prepared Probe required_total must equal partition_scan_end"
        )


def rebuild_manifest_probe_pools(
    manifest: Dict[str, Any],
    local_data_dir: Optional[str],
) -> Tuple[List[Dataset], List[int]]:
    """Rebuild verified role reservoirs; fixed sharding occurs in the runner."""
    validate_probe_collection_manifest(manifest)

    shuffle_seed = manifest["shuffle"]["seed"]
    continual_schema = (
        matches_schema(manifest.get("manifest_schema"), CONTINUAL_SCHEMA_VERSION)
    )
    probe_pools: List[Dataset] = []
    for dataset_entry in manifest["datasets"]:
        role = dataset_entry["role"]
        source_dataset = load_manifest_source_dataset(
            dataset_entry,
            local_data_dir,
        )
        if continual_schema:
            selected_dataset = rebuild_continual_manifest_records(
                dataset_name=dataset_entry["dataset_id"],
                dataset=source_dataset,
                is_benign=role == "benign",
                source_metadata=dataset_entry["source"],
                records=dataset_entry["records"],
            )
        else:
            prepared = prepare_selected_manifest_dataset(
                dataset_name=dataset_entry["dataset_id"],
                dataset=source_dataset,
                is_benign=role == "benign",
                source_metadata=dataset_entry["source"],
                required_total=dataset_entry["counts"]["partition_scan_end"],
                shuffle_seed=shuffle_seed,
            )
            if not isinstance(prepared, dict):
                raise ValueError(
                    "prepare_selected_manifest_dataset must return a dictionary"
                )
            _verify_probe_prepared_counts(prepared, dataset_entry)
            selected_dataset = prepared.get("dataset")
            verify_selected_manifest_dataset(selected_dataset, dataset_entry)
        probe_reservoir = select_manifest_partition(
            selected_dataset,
            dataset_entry,
            "probe",
        )
        verify_probe_partition_dataset(probe_reservoir, dataset_entry)
        probe_pools.append(
            select_probe_collection_capacity(
                probe_reservoir,
                role,
                expected_total=dataset_entry["counts"][
                    "probe" if continual_schema else "probe_actual"
                ],
                effective_total=(
                    dataset_entry["probe_effective_count"]
                    if continual_schema
                    else None
                ),
                client_pool_sizes=(
                    dataset_entry["probe_effective_client_pool_sizes"]
                    if continual_schema
                    else dataset_entry["probe_client_pool_sizes"]
                ),
            )
        )

    if [dataset["probe_role"][0] for dataset in probe_pools] != [
        "benign",
        "malicious",
    ]:
        raise ValueError(
            "Probe collection pools must be ordered benign then malicious"
        )
    return probe_pools, [PROBE_CLIENTS_PER_ROLE, PROBE_CLIENTS_PER_ROLE]

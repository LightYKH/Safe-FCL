import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

from .artifact_schema import matches_schema
from .continual_data_manifest import SCHEMA_VERSION as CONTINUAL_SCHEMA_VERSION
from .data_manifest import (
    collect_processing_runtime,
    get_clean_repository_commit,
    load_and_validate_manifest,
    sha256_file,
    validate_dataset_source_identity,
)
from .process_dataset import (
    prepare_selected_manifest_dataset,
    rebuild_continual_manifest_records,
)


_MALICIOUSGEN_RELATIVE_PATH = "Mistral/maliciousQA.json"


def load_training_manifest(
    manifest_path: Optional[str],
    expected_sha256: Optional[str],
) -> Optional[Dict[str, Any]]:
    path_empty = manifest_path is None or manifest_path == ""
    sha_empty = expected_sha256 is None or expected_sha256 == ""
    if path_empty and sha_empty:
        return None
    if path_empty != sha_empty:
        raise ValueError(
            "data_manifest_path and data_manifest_sha256 must be provided together"
        )
    if not isinstance(manifest_path, str):
        raise TypeError("data_manifest_path must be a string")
    if not isinstance(expected_sha256, str):
        raise TypeError("data_manifest_sha256 must be a string")
    target = Path(manifest_path).expanduser().resolve()
    if target.is_file():
        raw = target.read_bytes()
        try:
            candidate = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            candidate = None
        if (
            isinstance(candidate, dict)
            and matches_schema(candidate.get("manifest_schema"), CONTINUAL_SCHEMA_VERSION)
        ):
            from .continual_data_manifest import load_task_manifest

            return load_task_manifest(manifest_path, expected_sha256)
    return load_and_validate_manifest(manifest_path, expected_sha256)


def _require_single_role_configuration(
    dataset_names: Any,
    client_counts: Any,
    role: str,
) -> Tuple[str, int]:
    if not isinstance(dataset_names, list) or len(dataset_names) != 1:
        raise ValueError(
            "Manifest mode requires exactly one "
            + role
            + " dataset name"
        )
    if not isinstance(client_counts, list) or len(client_counts) != 1:
        raise ValueError(
            "Manifest mode requires exactly one "
            + role
            + " client count"
        )
    dataset_name = dataset_names[0]
    client_count = client_counts[0]
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(role + " dataset name must be a non-empty string")
    if (
        not isinstance(client_count, int)
        or isinstance(client_count, bool)
        or client_count < 0
    ):
        raise ValueError(
            role + " client count must be a non-negative integer"
        )
    return dataset_name, client_count


def _require_positive_int(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(name + " must be a positive integer")
    return value


def validate_manifest_training_arguments(
    manifest: Dict[str, Any],
    script_args: Any,
    fed_args: Any,
) -> None:
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")

    benign_name, benign_clients = _require_single_role_configuration(
        getattr(fed_args, "benign_dataset_names", None),
        getattr(fed_args, "benign_num_clients", None),
        "benign",
    )
    malicious_name, malicious_clients = _require_single_role_configuration(
        getattr(fed_args, "malicious_dataset_names", None),
        getattr(fed_args, "malicious_num_clients", None),
        "malicious",
    )

    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise ValueError(
            "Manifest must contain benign and malicious dataset entries"
        )
    expected_entries = (
        ("benign", benign_name, benign_clients),
        ("malicious", malicious_name, malicious_clients),
    )
    for dataset_entry, expected in zip(datasets, expected_entries):
        role, dataset_name, client_count = expected
        if not isinstance(dataset_entry, dict):
            raise ValueError("Manifest dataset entry must be a dictionary")
        if dataset_entry.get("role") != role:
            raise ValueError("Manifest dataset role does not match CLI role")
        if dataset_entry.get("dataset_id") != dataset_name:
            raise ValueError(
                "Manifest dataset ID does not match CLI dataset name"
            )
        if dataset_entry.get("client_count") != client_count:
            raise ValueError(
                "Manifest client count does not match CLI client count"
            )

    manifest_experiment = manifest.get("experiment")
    if not isinstance(manifest_experiment, dict):
        raise ValueError("Manifest experiment must be a dictionary")
    manifest_data_per_client = manifest_experiment.get(
        "federated_num_data_per_client"
    )
    cli_data_per_client = _require_positive_int(
        getattr(fed_args, "num_data_per_client", None),
        "num_data_per_client",
    )
    if manifest_data_per_client != cli_data_per_client:
        raise ValueError(
            "Manifest federated_num_data_per_client does not match CLI value"
        )

    manifest_shuffle = manifest.get("shuffle")
    if not isinstance(manifest_shuffle, dict):
        raise ValueError("Manifest shuffle must be a dictionary")
    manifest_shuffle_seed = manifest_shuffle.get("seed")
    if (
        not isinstance(manifest_shuffle_seed, int)
        or isinstance(manifest_shuffle_seed, bool)
    ):
        raise ValueError("Manifest shuffle seed must be an integer")
    training_seed = getattr(script_args, "seed", None)
    if not isinstance(training_seed, int) or isinstance(
        training_seed,
        bool,
    ):
        raise ValueError("training seed must be an integer")
    if training_seed != manifest_shuffle_seed:
        raise ValueError(
            "training seed does not match Manifest shuffle seed"
        )

    if getattr(script_args, "prefilter_id", False):
        raise ValueError("Manifest mode requires prefilter_id=false")
    existing_lora = getattr(script_args, "existing_lora", None)
    if existing_lora not in (None, ""):
        raise ValueError("Manifest mode does not support existing_lora")

    batch_size = _require_positive_int(
        getattr(script_args, "batch_size", None),
        "batch_size",
    )
    gradient_accumulation_steps = _require_positive_int(
        getattr(script_args, "gradient_accumulation_steps", None),
        "gradient_accumulation_steps",
    )
    max_steps = _require_positive_int(
        getattr(script_args, "max_steps", None),
        "max_steps",
    )
    requested_per_round = (
        batch_size * gradient_accumulation_steps * max_steps
    )
    if requested_per_round > cli_data_per_client:
        raise ValueError(
            "Per-round requested samples exceed num_data_per_client"
        )


def validate_manifest_training_provenance(
    manifest: Dict[str, Any],
    repository_root: Path,
    *,
    allow_experiment_copy: bool = False,
) -> None:
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")
    root = Path(repository_root).resolve()
    manifest_commit = manifest.get("code_commit")
    if allow_experiment_copy:
        if not isinstance(manifest_commit, str) or len(manifest_commit) != 40:
            raise ValueError("Manifest code_commit is invalid")
    else:
        current_commit = get_clean_repository_commit(root)
        if current_commit != manifest_commit:
            raise ValueError(
                "Current Git HEAD does not match Manifest code_commit"
            )

    manifest_runtime = manifest.get("processing_runtime")
    if not isinstance(manifest_runtime, dict):
        raise ValueError(
            "Manifest processing_runtime must be a dictionary"
        )
    current_runtime = collect_processing_runtime()
    for key, current_value in current_runtime.items():
        if manifest_runtime.get(key) != current_value:
            raise ValueError(
                "Current runtime does not match Manifest "
                + "processing_runtime."
                + key
            )

    if (
        matches_schema(manifest.get("manifest_schema"), CONTINUAL_SCHEMA_VERSION)
    ):
        protocol = manifest.get("protocol")
        if not isinstance(protocol, dict):
            raise ValueError("continual Manifest protocol metadata is required")
        protocol_path = protocol.get("path")
        expected_protocol_sha = protocol.get("sha256")
        if not isinstance(protocol_path, str) or not protocol_path:
            raise ValueError("continual Manifest protocol path is required")
        actual_protocol_sha = sha256_file(
            Path(protocol_path).expanduser().resolve()
        )
        if actual_protocol_sha != expected_protocol_sha:
            raise ValueError("continual protocol SHA-256 mismatch")


def load_manifest_source_dataset(
    dataset_entry: Dict[str, Any],
    local_data_dir: Optional[str],
) -> Dataset:
    if dataset_entry.get('source',{}).get('kind') == 'verified_risk5_jsonl':
        from .risk5_data import load_source
        return load_source(dataset_entry)
    if not isinstance(dataset_entry, dict):
        raise TypeError("dataset_entry must be a dictionary")
    if dataset_entry.get("active") is not True:
        raise ValueError("Cannot load an inactive Manifest dataset")

    dataset_id = dataset_entry.get("dataset_id")
    role = dataset_entry.get("role")
    source = dataset_entry.get("source")
    if not isinstance(source, dict):
        raise ValueError("Active Manifest dataset source is required")
    source_kind = source.get("kind")
    split = source.get("split")
    validate_dataset_source_identity(
        dataset_id,
        role,
        source_kind,
        split,
        source.get("json_relative_path"),
    )

    if source_kind == "huggingface":
        resolved_revision = source.get("resolved_revision")
        if not isinstance(resolved_revision, str):
            raise ValueError(
                "Hugging Face resolved_revision must be a string"
            )
        # Route through the pipeline loader: it handles dataset-specific
        # formats (WildJailbreak TSV, WildGuardMix config, PKU QA variant)
        # and prefers materialized local snapshots so offline runs work.
        # For previously supported datasets it falls back to the identical
        # load_dataset(dataset_id, split=split, revision=resolved_revision)
        # call as before.
        from utils.process_dataset import load_pipeline_dataset

        return load_pipeline_dataset(
            dataset_id,
            split,
            revision=resolved_revision,
            dataset_loader=load_dataset,
        )

    if source_kind == "local_json":
        if not isinstance(local_data_dir, str) or not local_data_dir:
            raise ValueError(
                "Active MaliciousGen requires local_data_dir"
            )
        json_path = (
            Path(local_data_dir) / Path(_MALICIOUSGEN_RELATIVE_PATH)
        ).resolve()
        actual_sha256 = sha256_file(json_path)
        if actual_sha256 != source.get("json_sha256"):
            raise ValueError("MaliciousGen JSON SHA-256 mismatch")
        return load_dataset(
            "json",
            data_files=str(json_path),
            split=split,
        )

    raise ValueError("Unsupported Manifest source kind")


def verify_selected_manifest_dataset(
    selected_dataset: Dataset,
    dataset_entry: Dict[str, Any],
) -> None:
    if not isinstance(selected_dataset, Dataset):
        raise TypeError("selected_dataset must be a datasets.Dataset")
    if not isinstance(dataset_entry, dict):
        raise TypeError("dataset_entry must be a dictionary")
    records = dataset_entry.get("records")
    if not isinstance(records, list):
        raise ValueError("Manifest dataset records must be a list")
    counts = dataset_entry.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Manifest dataset counts must be a dictionary")
    partition_scan_end = _require_positive_int(
        counts.get("partition_scan_end"),
        "partition_scan_end",
    )
    if len(selected_dataset) != partition_scan_end:
        raise ValueError(
            "Selected dataset length does not match partition_scan_end"
        )

    required_columns = {
        "content_hash",
        "raw_row_index",
        "sample_id",
        "source_id",
        "split_rank",
    }
    missing_columns = required_columns.difference(
        selected_dataset.column_names
    )
    if missing_columns:
        raise ValueError(
            "Selected dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    compared_fields = (
        "raw_row_index",
        "source_id",
        "content_hash",
        "split_rank",
    )
    for record in records:
        split_rank = record.get("split_rank")
        if (
            not isinstance(split_rank, int)
            or isinstance(split_rank, bool)
            or split_rank < 0
            or split_rank >= partition_scan_end
        ):
            raise ValueError("Manifest record split_rank is outside scan range")
        row = selected_dataset[split_rank]
        for field_name in compared_fields:
            if row[field_name] != record.get(field_name):
                raise ValueError(
                    "Selected dataset "
                    + field_name
                    + " does not match Manifest record"
                )
        if row["sample_id"] != row["split_rank"]:
            raise ValueError(
                "Selected dataset sample_id must equal split_rank"
            )


def select_manifest_partition(
    selected_dataset: Dataset,
    dataset_entry: Dict[str, Any],
    partition: str,
) -> Dataset:
    continual = dataset_entry.get("counts", {}).get("selected_total") is not None
    allowed = (
        ("model_train", "model_validation", "formal_test", "probe")
        if continual
        else ("probe", "federated")
    )
    if partition not in allowed:
        raise ValueError("partition is not valid for this manifest schema")
    records = dataset_entry.get("records")
    if not isinstance(records, list):
        raise ValueError("Manifest dataset records must be a list")
    counts = dataset_entry.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Manifest dataset counts must be a dictionary")
    expected_length = (
        counts.get("selected_total")
        if continual
        else counts.get("partition_scan_end")
    )
    if len(selected_dataset) != expected_length:
        raise ValueError(
            "Selected dataset length does not match partition_scan_end"
        )
    selected_indices = [
        record["split_rank"]
        for record in records
        if record.get("partition") == partition
    ]
    return selected_dataset.select(selected_indices)


def _verify_prepared_counts(
    prepared: Dict[str, Any],
    dataset_entry: Dict[str, Any],
) -> None:
    prepared_counts = prepared.get("counts")
    manifest_counts = dataset_entry.get("counts")
    if not isinstance(prepared_counts, dict):
        raise ValueError("Prepared dataset counts must be a dictionary")
    if not isinstance(manifest_counts, dict):
        raise ValueError("Manifest dataset counts must be a dictionary")
    for count_name in (
        "raw",
        "after_semantic_filter",
        "after_format_and_dedup",
    ):
        if prepared_counts.get(count_name) != manifest_counts.get(
            count_name
        ):
            raise ValueError(
                "Prepared dataset count does not match Manifest: "
                + count_name
            )
    if prepared_counts.get("required_total") != manifest_counts.get(
        "partition_scan_end"
    ):
        raise ValueError(
            "Prepared required_total must equal Manifest partition_scan_end"
        )


def rebuild_manifest_federated_pools(
    manifest: Dict[str, Any],
    local_data_dir: Optional[str],
) -> Tuple[List[Dataset], List[int]]:
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")
    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Manifest experiment must be a dictionary")
    federated_num_data_per_client = _require_positive_int(
        experiment.get("federated_num_data_per_client"),
        "Manifest federated_num_data_per_client",
    )
    shuffle = manifest.get("shuffle")
    if not isinstance(shuffle, dict):
        raise ValueError("Manifest shuffle metadata must be a dictionary")
    shuffle_seed = shuffle.get("seed")
    continual_schema = (
        matches_schema(manifest.get("manifest_schema"), CONTINUAL_SCHEMA_VERSION)
    )

    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise ValueError(
            "Manifest must contain benign and malicious dataset entries"
        )

    federated_pools: List[Dataset] = []
    num_client_list: List[int] = []
    for expected_role, dataset_entry in zip(
        ("benign", "malicious"),
        datasets,
    ):
        if dataset_entry.get("role") != expected_role:
            raise ValueError(
                "Manifest datasets must be ordered benign then malicious"
            )
        if dataset_entry.get("active") is not True:
            continue

        source_dataset = load_manifest_source_dataset(
            dataset_entry,
            local_data_dir,
        )
        if continual_schema:
            selected_dataset = rebuild_continual_manifest_records(
                dataset_name=dataset_entry["dataset_id"],
                dataset=source_dataset,
                is_benign=dataset_entry["role"] == "benign",
                source_metadata=dataset_entry["source"],
                records=dataset_entry["records"],
            )
            federated_pool = select_manifest_partition(
                selected_dataset, dataset_entry, "model_train"
            )
        else:
            prepared = prepare_selected_manifest_dataset(
                dataset_name=dataset_entry["dataset_id"],
                dataset=source_dataset,
                is_benign=dataset_entry["role"] == "benign",
                source_metadata=dataset_entry["source"],
                required_total=dataset_entry["counts"]["partition_scan_end"],
                shuffle_seed=shuffle_seed,
            )
            if not isinstance(prepared, dict):
                raise ValueError(
                    "prepare_selected_manifest_dataset must return a dictionary"
                )
            _verify_prepared_counts(prepared, dataset_entry)
            selected_dataset = prepared.get("dataset")
            verify_selected_manifest_dataset(
                selected_dataset,
                dataset_entry,
            )
            federated_pool = select_manifest_partition(
                selected_dataset,
                dataset_entry,
                "federated",
            )

        client_count = dataset_entry.get("client_count")
        expected_federated = (
            client_count * federated_num_data_per_client
        )
        if len(federated_pool) != expected_federated:
            raise ValueError(
                "Federated pool length does not match "
                + "client_count * federated_num_data_per_client"
            )
        federated_pools.append(federated_pool)
        num_client_list.append(client_count)

    return federated_pools, num_client_list


def rebuild_continual_evaluation_pools(
    manifest: Dict[str, Any],
    local_data_dir: Optional[str],
    partition: str,
) -> List[Dataset]:
    """Rebuild a read-only validation or formal-test pool for both roles."""
    if (
        not isinstance(manifest, dict)
        or not matches_schema(manifest.get("manifest_schema"), CONTINUAL_SCHEMA_VERSION)
    ):
        raise ValueError("evaluation pools require a continual task manifest")
    if partition not in ("model_validation", "formal_test"):
        raise ValueError(
            "evaluation partition must be model_validation or formal_test"
        )
    pools: List[Dataset] = []
    for expected_role, dataset_entry in zip(
        ("benign", "malicious"), manifest.get("datasets", [])
    ):
        if dataset_entry.get("role") != expected_role:
            raise ValueError("manifest roles must be ordered benign then malicious")
        source_dataset = load_manifest_source_dataset(
            dataset_entry, local_data_dir
        )
        rebuilt = rebuild_continual_manifest_records(
            dataset_name=dataset_entry["dataset_id"],
            dataset=source_dataset,
            is_benign=expected_role == "benign",
            source_metadata=dataset_entry["source"],
            records=dataset_entry["records"],
        )
        pool = select_manifest_partition(rebuilt, dataset_entry, partition)
        expected_count = dataset_entry["counts"][partition]
        if len(pool) != expected_count:
            raise ValueError("evaluation pool count does not match Manifest")
        pools.append(pool)
    if len(pools) != 2:
        raise ValueError("evaluation requires benign and malicious pools")
    return pools

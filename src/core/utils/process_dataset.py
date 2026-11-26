import os
import glob
import datasets
from datasets import load_dataset
import pandas as pd
from functools import partial
from .data_manifest import (
    compute_content_hash,
    compute_normalized_prompt_hash,
    make_hf_source_id,
    make_json_source_id,
    validate_dataset_source_identity,
)


def _continual_eligibility_filter(example, dataset_name):
    """Source-level English/single-turn and basic trainability screen."""
    if dataset_name in ("allenai/WildChat", "lmsys/lmsys-chat-1m"):
        conversation = example.get("conversation")
        if (
            example.get("turn") != 1
            or example.get("language") != "English"
            or not isinstance(conversation, list)
            or len(conversation) < 2
        ):
            return False
        return all(
            isinstance(conversation[index].get("content"), str)
            and bool(conversation[index]["content"].strip())
            for index in (0, 1)
        )
    return True


def _continual_add_sft_fields(example, dataset_name):
    """Add instruction/response without deleting labels needed by filtering."""
    if dataset_name in ("allenai/WildChat", "lmsys/lmsys-chat-1m"):
        instruction = example["conversation"][0]["content"]
        response = example["conversation"][1]["content"]
    elif dataset_name == "allenai/wildjailbreak":
        data_type = example.get("data_type") or ""
        vanilla = example.get("vanilla") or ""
        adversarial = example.get("adversarial") or ""
        instruction = (
            vanilla
            if data_type.startswith("vanilla")
            else adversarial or vanilla
        )
        response = example.get("completion") or ""
    else:
        instruction = example.get("instruction", example.get("prompt"))
        response = example.get("response")
    return {"instruction": instruction, "response": response}


def _is_nonempty_sft_row(example):
    return (
        isinstance(example.get("instruction"), str)
        and bool(example["instruction"].strip())
        and isinstance(example.get("response"), str)
        and bool(example["response"].strip())
    )


def _deduplicate_identity_columns(dataset):
    """Keep the first content and normalized-prompt identity deterministically."""
    seen_content = set()
    seen_prompt = set()

    def keep(example):
        content_hash = example["content_hash"]
        prompt_hash = example["normalized_prompt_hash"]
        if content_hash in seen_content or prompt_hash in seen_prompt:
            return False
        seen_content.add(content_hash)
        seen_prompt.add(prompt_hash)
        return True

    return dataset.filter(
        keep,
        load_from_cache_file=False,
        desc="Deduplicating SFT content and normalized prompts",
    )


def prepare_continual_manifest_dataset(
    dataset_name,
    dataset,
    is_benign,
    source_metadata,
    model_required_total,
    selection_target_total,
    excluded_identities=None,
    shuffle_seed=2023,
):
    """Prepare one role for the stratified continual 7:1:2 protocol.

    Cross-task exclusions are applied before the one deterministic shuffle.
    A short Probe pool is allowed, but the model split minimum fails closed.
    """
    for value, name in (
        (model_required_total, "model_required_total"),
        (selection_target_total, "selection_target_total"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(name + " must be a non-negative integer")
    if selection_target_total < model_required_total:
        raise ValueError("selection_target_total cannot be below model minimum")
    if not isinstance(shuffle_seed, int) or isinstance(shuffle_seed, bool):
        raise ValueError("shuffle_seed must be an integer")
    if "raw_row_index" in dataset.column_names:
        raise ValueError("input dataset must not contain raw_row_index")

    exclusions = excluded_identities or {}
    excluded_source_ids = set(exclusions.get("source_id", set()))
    excluded_content_hashes = set(exclusions.get("content_hash", set()))
    excluded_prompt_hashes = set(
        exclusions.get("normalized_prompt_hash", set())
    )
    make_source_id = _validate_manifest_source_metadata(
        dataset_name, is_benign, source_metadata
    )
    raw_count = len(dataset)
    raw_fingerprint = getattr(dataset, "_fingerprint", None)
    dataset = dataset.map(
        lambda _example, index: {"raw_row_index": int(index)},
        with_indices=True,
        load_from_cache_file=False,
        desc="Attaching immutable raw row indices",
    )
    dataset = dataset.filter(
        partial(_continual_eligibility_filter, dataset_name=dataset_name),
        load_from_cache_file=False,
        desc="English/single-turn eligibility screening",
    )
    after_eligibility_screen = len(dataset)
    dataset = dataset.map(
        partial(_continual_add_sft_fields, dataset_name=dataset_name),
        load_from_cache_file=False,
        desc="Formatting source rows as SFT instruction/response",
    )
    dataset = dataset.filter(
        partial(
            benign_filter_samples if is_benign else malicious_filter_samples,
            dataset_name=dataset_name,
        ),
        load_from_cache_file=False,
        desc="Applying role semantic filter",
    )
    dataset = dataset.filter(
        _is_nonempty_sft_row,
        load_from_cache_file=False,
        desc="Removing non-trainable SFT rows",
    )
    after_semantic_filter = len(dataset)

    keep_columns = ["raw_row_index", "instruction", "response"]
    dataset = dataset.select_columns(keep_columns)
    short_name = _to_short_name(dataset_name, is_benign=is_benign)

    def add_identities(example):
        raw_row_index = int(example["raw_row_index"])
        return {
            "source_id": make_source_id(raw_row_index),
            "content_hash": compute_content_hash(
                example["instruction"], example["response"]
            ),
            "normalized_prompt_hash": compute_normalized_prompt_hash(
                example["instruction"]
            ),
            "dataset_name": short_name,
        }

    dataset = dataset.map(
        add_identities,
        load_from_cache_file=False,
        desc="Computing manifest identities",
    )
    before_content_dedup = len(dataset)
    dataset = _deduplicate_identity_columns(dataset)
    after_content_dedup = len(dataset)

    def outside_prior_tasks(example):
        return not (
            example["source_id"] in excluded_source_ids
            or example["content_hash"] in excluded_content_hashes
            or example["normalized_prompt_hash"] in excluded_prompt_hashes
        )

    dataset = dataset.filter(
        outside_prior_tasks,
        load_from_cache_file=False,
        desc="Removing global cross-task overlap",
    )
    after_cross_task_dedup = len(dataset)
    if after_cross_task_dedup < model_required_total:
        role = "benign" if is_benign else "malicious"
        raise ValueError(
            f"FAIL-FAST: {dataset_name} {role} has "
            f"{after_cross_task_dedup} usable rows after global dedup, but "
            f"the 70% model split requires {model_required_total} "
            "(3500 benign or 1500 malicious train; no replacement allowed)"
        )

    dataset = dataset.shuffle(seed=shuffle_seed)
    post_shuffle_fingerprint = getattr(dataset, "_fingerprint", None)
    selection_count = min(len(dataset), selection_target_total)
    dataset = dataset.select(range(selection_count))
    dataset = dataset.map(
        lambda _example, index: {
            "split_rank": int(index),
            "sample_id": int(index),
        },
        with_indices=True,
        load_from_cache_file=False,
        desc="Freezing deterministic split ranks",
    )
    return {
        "dataset": dataset,
        "counts": {
            "raw": raw_count,
            "after_eligibility_screen": after_eligibility_screen,
            "after_sft_format_and_semantic_filter": after_semantic_filter,
            "before_content_dedup": before_content_dedup,
            "after_content_and_prompt_dedup": after_content_dedup,
            "cross_task_removed": (
                after_content_dedup - after_cross_task_dedup
            ),
            "after_cross_task_dedup": after_cross_task_dedup,
            "selection_target_total": selection_target_total,
            "selected_total": selection_count,
            "unused_after_selection": (
                after_cross_task_dedup - selection_count
            ),
        },
        "fingerprints": {
            "policy": "diagnostic_only",
            "raw": raw_fingerprint,
            "post_shuffle": post_shuffle_fingerprint,
        },
    }


def rebuild_continual_manifest_records(
    dataset_name,
    dataset,
    is_benign,
    source_metadata,
    records,
):
    """Rebuild exactly the raw rows named by a validated continual manifest."""
    if source_metadata.get('kind') == 'verified_risk5_jsonl':
        from .risk5_data import rebuild
        return rebuild(dataset_name, dataset, is_benign, source_metadata, records)
    if not isinstance(records, list) or not records:
        raise ValueError("continual manifest records must be a non-empty list")
    make_source_id = _validate_manifest_source_metadata(
        dataset_name, is_benign, source_metadata
    )
    raw_indices = [int(record["raw_row_index"]) for record in records]
    if len(set(raw_indices)) != len(raw_indices):
        raise ValueError("continual manifest raw_row_index values must be unique")
    if min(raw_indices) < 0 or max(raw_indices) >= len(dataset):
        raise ValueError("continual manifest raw_row_index is outside source dataset")
    selected = dataset.select(raw_indices)
    selected = selected.add_column("raw_row_index", raw_indices)
    if not all(
        _continual_eligibility_filter(row, dataset_name)
        for row in selected
    ):
        raise ValueError("manifest row no longer passes eligibility screening")
    selected = selected.map(
        partial(_continual_add_sft_fields, dataset_name=dataset_name),
        load_from_cache_file=False,
        desc="Rebuilding manifest SFT rows",
    )
    semantic_filter = benign_filter_samples if is_benign else malicious_filter_samples
    if not all(
        semantic_filter(row, dataset_name=dataset_name) and _is_nonempty_sft_row(row)
        for row in selected
    ):
        raise ValueError("manifest row no longer passes semantic/SFT filtering")
    selected = selected.select_columns(
        ["raw_row_index", "instruction", "response"]
    )
    short_name = _to_short_name(dataset_name, is_benign=is_benign)

    def restore_identity(example, index):
        raw_row_index = int(example["raw_row_index"])
        return {
            "content_hash": compute_content_hash(
                example["instruction"], example["response"]
            ),
            "dataset_name": short_name,
            "normalized_prompt_hash": compute_normalized_prompt_hash(
                example["instruction"]
            ),
            "sample_id": int(records[index]["split_rank"]),
            "source_id": make_source_id(raw_row_index),
            "split_rank": int(records[index]["split_rank"]),
        }

    selected = selected.map(
        restore_identity,
        with_indices=True,
        load_from_cache_file=False,
        desc="Restoring and checking manifest identities",
    )
    for index, record in enumerate(records):
        row = selected[index]
        for field in (
            "raw_row_index",
            "source_id",
            "content_hash",
            "normalized_prompt_hash",
            "split_rank",
        ):
            if row[field] != record[field]:
                raise ValueError("rebuilt continual row mismatch: " + field)
    return selected


def validate_dataset_client_lists(dataset_names, client_counts, role):
    if len(dataset_names) != len(client_counts):
        raise ValueError(
            f"{role} dataset names and client counts must have equal lengths"
        )

    for index, client_count in enumerate(client_counts):
        if (
            not isinstance(client_count, int)
            or isinstance(client_count, bool)
            or client_count < 0
        ):
            raise ValueError(
                f"{role} client count at index {index} must be a "
                "non-negative integer"
            )


def get_sft_datasets(script_args, fed_args):
    validate_dataset_client_lists(
        fed_args.benign_dataset_names,
        fed_args.benign_num_clients,
        "benign",
    )
    validate_dataset_client_lists(
        fed_args.malicious_dataset_names,
        fed_args.malicious_num_clients,
        "malicious",
    )

    dataset_list, num_client_list = [], []

    for benign_dataset_name, benign_num_clients in zip(fed_args.benign_dataset_names, fed_args.benign_num_clients):
        if benign_num_clients==0:
            continue
        benign_dataset_sample = int(benign_num_clients*fed_args.num_data_per_client)
        benign_dataset = get_whole_dataset(benign_dataset_name, script_args.local_data_dir)
        benign_dataset = benign_dataset.filter(partial(benign_filter_samples, dataset_name=benign_dataset_name))
        benign_dataset = process_sft_dataset(benign_dataset_name, benign_dataset, script_args.template, benign_dataset_sample, True, script_args.existing_lora is not None, prefilter_id=getattr(script_args, 'prefilter_id', False))
        dataset_list.append(benign_dataset)
        num_client_list.append(benign_num_clients)

    for malicious_dataset_name, malicious_num_clients in zip(fed_args.malicious_dataset_names, fed_args.malicious_num_clients):
        if malicious_num_clients==0:
            continue
        malicious_dataset_sample = int(malicious_num_clients*fed_args.num_data_per_client)
        malicious_dataset = get_whole_dataset(malicious_dataset_name, script_args.local_data_dir)
        malicious_dataset = malicious_dataset.filter(partial(malicious_filter_samples, dataset_name=malicious_dataset_name))
        malicious_dataset = process_sft_dataset(malicious_dataset_name, malicious_dataset, script_args.template, malicious_dataset_sample, False, prefilter_id=getattr(script_args, 'prefilter_id', False))
        dataset_list.append(malicious_dataset)
        num_client_list.append(malicious_num_clients)
        
    return dataset_list, num_client_list

def get_whole_dataset(dataset_name, local_data_dir=None):
    if dataset_name in ['allenai/WildChat','lmsys/lmsys-chat-1m']:
        dataset = load_dataset(dataset_name, split="train")
    elif dataset_name == "PKU-Alignment/BeaverTails":
        dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    elif dataset_name == 'MaliciousGen':
        data_files = os.path.join(local_data_dir, 'Mistral/maliciousQA.json')
        dataset = load_dataset('json', data_files=data_files, split='train')
    elif dataset_name in (
        'allenai/wildjailbreak',
        'allenai/wildguardmix',
        'PKU-Alignment/PKU-SafeRLHF-QA',
    ):
        dataset = load_pipeline_dataset(dataset_name, "train")
    else:
        dataset_name = os.path.join(local_data_dir, dataset_name) if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train")

    return dataset


# Pinned response variant for PKU-SafeRLHF-QA (the repo ships per-generator
# variants: Alpaca-7B / Alpaca2-7B / Alpaca3-8B). Keep in sync with the
# task3/task4 data plan.
_PKU_SAFERLHF_QA_VARIANT = "Alpaca2-7B"


def _local_snapshot_file(dataset_name, revision, relpath):
    """Resolve a dataset file inside the local HF hub cache, if materialized.

    The probe collection runs with HF offline mode forced; hf:// URLs fail
    there, so prefer the on-disk snapshot when revision is pinned.
    """
    if not revision:
        return None
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    path = os.path.join(
        hf_home,
        "hub",
        "datasets--" + dataset_name.replace("/", "--"),
        "snapshots",
        revision,
        *relpath.split("/"),
    )
    return path if os.path.isfile(path) else None


def _local_snapshot_files(
    dataset_name, revision, relpath_pattern, expected_count=None
):
    """Resolve a complete, sorted file set inside one pinned local snapshot."""
    if not revision:
        return None
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pattern = os.path.join(
        hf_home,
        "hub",
        "datasets--" + dataset_name.replace("/", "--"),
        "snapshots",
        revision,
        *relpath_pattern.split("/"),
    )
    paths = sorted(glob.glob(pattern))
    if paths and expected_count is not None and len(paths) != expected_count:
        raise RuntimeError(
            "pinned local snapshot is incomplete for "
            + dataset_name
            + ": expected "
            + str(expected_count)
            + " files but found "
            + str(len(paths))
        )
    return paths or None


def load_pipeline_dataset(
    dataset_name,
    split,
    revision=None,
    local_data_dir=None,
    dataset_loader=None,
):
    """Load a dataset for the split-manifest / SFT pipeline.

    Centralizes dataset-specific loading quirks so both get_whole_dataset and
    scripts/build_split_manifest.py share one code path. revision is the
    resolved HF commit SHA (optional); local_data_dir is only used by
    MaliciousGen.
    """
    loader = dataset_loader or load_dataset
    if dataset_name == "allenai/WildChat":
        data_files = _local_snapshot_files(
            dataset_name,
            revision,
            "data/train-*.parquet",
            expected_count=6,
        )
        if data_files is not None:
            return loader("parquet", data_files=data_files, split="train")
        return loader(dataset_name, split=split, revision=revision)
    if dataset_name == "lmsys/lmsys-chat-1m":
        data_files = _local_snapshot_files(
            dataset_name,
            revision,
            "data/train-*.parquet",
            expected_count=6,
        )
        if data_files is not None:
            return loader("parquet", data_files=data_files, split="train")
        return loader(dataset_name, split=split, revision=revision)
    if dataset_name == "PKU-Alignment/BeaverTails":
        data_file = _local_snapshot_file(
            dataset_name, revision, "round0/30k/train.jsonl.gz"
        )
        if data_file is not None:
            return loader("json", data_files=data_file, split="train")
        return loader(dataset_name, split=split, revision=revision)
    if dataset_name == "allenai/wildjailbreak":
        # The repo stores data as TSV; force an all-string schema because
        # empty "adversarial" cells would otherwise be inferred as double
        # and crash Arrow casts (datasets#YAML schema is broken upstream).
        features = datasets.Features({
            "vanilla": datasets.Value("string"),
            "adversarial": datasets.Value("string"),
            "completion": datasets.Value("string"),
            "data_type": datasets.Value("string"),
        })
        data_files = _local_snapshot_file(dataset_name, revision, "train/train.tsv")
        if data_files is None:
            data_files = "hf://datasets/allenai/wildjailbreak"
            if revision:
                data_files += "@" + revision
            data_files += "/train/train.tsv"
        return loader(
            "csv",
            data_files=data_files,
            delimiter="\t",
            features=features,
            split="train",
        )
    if dataset_name == "allenai/wildguardmix":
        data_files = _local_snapshot_file(
            dataset_name, revision, "train/wildguard_train.parquet"
        )
        if data_files is not None:
            return loader("parquet", data_files=data_files, split="train")
        # The repo requires an explicit config name.
        return loader(
            "allenai/wildguardmix",
            "wildguardtrain",
            split="train",
            revision=revision,
        )
    if dataset_name == "PKU-Alignment/PKU-SafeRLHF-QA":
        relpath = "data/" + _PKU_SAFERLHF_QA_VARIANT + "/train.jsonl"
        data_files = _local_snapshot_file(dataset_name, revision, relpath)
        if data_files is None:
            data_files = "hf://datasets/PKU-Alignment/PKU-SafeRLHF-QA"
            if revision:
                data_files += "@" + revision
            data_files += "/" + relpath
        return loader("json", data_files=data_files, split="train")
    if dataset_name == "MaliciousGen":
        if not local_data_dir:
            raise ValueError("MaliciousGen requires local_data_dir")
        data_files = os.path.join(local_data_dir, "Mistral/maliciousQA.json")
        return loader("json", data_files=data_files, split="train")
    return loader(dataset_name, split=split, revision=revision)

def _to_short_name(full_name: str, is_benign=None) -> str:
    mapping = {
        'PKU-Alignment/BeaverTails': 'BeaverTails',
        'allenai/WildChat': 'WildChat',
        'MaliciousGen': 'MaliciousGen',
        'lmsys/lmsys-chat-1m': 'lmsys-chat-1m',
        'allenai/wildjailbreak': 'WildJailbreak',
        'PKU-Alignment/PKU-SafeRLHF-QA': 'PKU-SafeRLHF-QA',
    }
    if full_name == 'allenai/wildguardmix':
        return 'WildGuardMixBenign' if is_benign else 'WildGuardMix'
    return mapping.get(full_name, full_name.split('/')[-1])


def _format_and_deduplicate_sft_dataset(dataset_name, dataset):
    if dataset_name in ["PKU-Alignment/BeaverTails"]:
        # Delete duplicate rows
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['prompt'])
        df = df.reset_index(drop=True)
        dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
        dataset = dataset.rename_column("prompt", "instruction")
        dataset = dataset.remove_columns(['category', 'is_safe'])

    elif dataset_name in ['allenai/WildChat']:
        def wildchat_format(example):
            example['instruction'] = example['conversation'][0]['content']
            example['response'] = example['conversation'][1]['content']
            return example

        dataset = dataset.map(
            wildchat_format,
            remove_columns=[
                'conversation_id',
                'model',
                'timestamp',
                'conversation',
                'turn',
                'language',
                'openai_moderation',
                'detoxify_moderation',
                'toxic',
                'redacted',
            ],
            desc="Formatting {dataset_name} for unified format",
        )

    elif dataset_name in ['lmsys/lmsys-chat-1m']:
        def lmsyschat_format(example):
            example['instruction'] = example['conversation'][0]['content']
            example['response'] = example['conversation'][1]['content']
            return example

        dataset = dataset.map(
            lmsyschat_format,
            remove_columns=[
                'conversation_id',
                'model',
                'conversation',
                'turn',
                'language',
                'openai_moderation',
                'redacted',
            ],
            desc="Formatting {dataset_name} for unified format",
        )

        dataset = dataset.filter(
            lambda example: (
                isinstance(example["response"], str)
                and example["response"] != ""
            ),
            load_from_cache_file=False,
            desc="Filtering LMSYS non-trainable responses",
        )

    elif dataset_name == "allenai/wildjailbreak":
        def wildjailbreak_format(example):
            data_type = example["data_type"] or ""
            vanilla = example["vanilla"] or ""
            adversarial = example["adversarial"] or ""
            if data_type.startswith("vanilla"):
                instruction = vanilla
            elif data_type.startswith("adversarial"):
                instruction = adversarial or vanilla
            else:
                instruction = vanilla or adversarial
            example["instruction"] = instruction
            example["response"] = example["completion"]
            return example

        dataset = dataset.map(
            wildjailbreak_format,
            remove_columns=["vanilla", "adversarial", "completion", "data_type"],
            desc="Formatting allenai/wildjailbreak for unified format",
        )
        dataset = dataset.filter(
            lambda example: isinstance(example["instruction"], str)
            and example["instruction"] != "",
            load_from_cache_file=False,
            desc="Filtering WildJailbreak empty instructions",
        )
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=["instruction"])
        df = df.reset_index(drop=True)
        dataset = datasets.Dataset.from_pandas(df, preserve_index=False)

    elif dataset_name == "allenai/wildguardmix":
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=["prompt"])
        df = df.reset_index(drop=True)
        dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
        dataset = dataset.rename_column("prompt", "instruction")
        dataset = dataset.remove_columns([
            "adversarial",
            "prompt_harm_label",
            "response_refusal_label",
            "response_harm_label",
            "subcategory",
        ])

    elif dataset_name == "PKU-Alignment/PKU-SafeRLHF-QA":
        # Deduplicate at (prompt, response) pair level: the same prompt may
        # legitimately appear with both safe and unsafe responses, and the
        # role filter (is_safe) has already selected one side by now.
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=["prompt", "response"])
        df = df.reset_index(drop=True)
        dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
        dataset = dataset.rename_column("prompt", "instruction")
        dataset = dataset.remove_columns([
            "prompt_source",
            "response_source",
            "is_safe",
            "harm_category",
            "severity_level",
            "sha256",
        ])

    elif dataset_name == "MaliciousGen":
        dataset = dataset

    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    return dataset


def _validate_manifest_source_metadata(
    dataset_name,
    is_benign,
    source_metadata,
):
    if not isinstance(source_metadata, dict):
        raise ValueError("source_metadata must be a dictionary")

    source_kind = source_metadata.get("kind")
    split = source_metadata.get("split")
    role = "benign" if is_benign else "malicious"
    validate_dataset_source_identity(
        dataset_name,
        role,
        source_kind,
        split,
        source_metadata.get("json_relative_path"),
    )

    if source_kind == "huggingface":
        resolved_revision = source_metadata.get("resolved_revision")

        def make_source_id(raw_row_index):
            return make_hf_source_id(
                dataset_name,
                resolved_revision,
                split,
                raw_row_index,
            )

        # Validate the metadata before Dataset.map begins.
        make_source_id(0)
        return make_source_id

    if source_kind == "local_json":
        json_sha256 = source_metadata.get("json_sha256")

        def make_source_id(raw_row_index):
            return make_json_source_id(json_sha256, raw_row_index)

        # Validate the metadata before Dataset.map begins.
        make_source_id(0)
        return make_source_id

    raise ValueError(
        "source_metadata kind must be huggingface or local_json"
    )


def prepare_selected_manifest_dataset(
    dataset_name,
    dataset,
    is_benign,
    source_metadata,
    required_total,
    shuffle_seed=2023,
):
    if required_total is not None and (
        not isinstance(required_total, int)
        or isinstance(required_total, bool)
        or required_total < 0
    ):
        raise ValueError(
            "required_total must be a non-negative integer or None"
        )
    if not isinstance(shuffle_seed, int) or isinstance(shuffle_seed, bool):
        raise ValueError("shuffle_seed must be an integer")
    if "raw_row_index" in dataset.column_names:
        raise ValueError("input dataset must not contain raw_row_index")

    make_source_id = _validate_manifest_source_metadata(
        dataset_name,
        is_benign,
        source_metadata,
    )
    raw_count = len(dataset)
    raw_fingerprint = getattr(dataset, "_fingerprint", None)
    dataset = dataset.map(
        lambda _example, index: {"raw_row_index": int(index)},
        with_indices=True,
        load_from_cache_file=False,
    )

    filter_function = (
        benign_filter_samples if is_benign else malicious_filter_samples
    )
    dataset = dataset.filter(
        partial(filter_function, dataset_name=dataset_name),
        load_from_cache_file=False,
    )
    after_semantic_filter = len(dataset)

    dataset = _format_and_deduplicate_sft_dataset(dataset_name, dataset)
    after_format_and_dedup = len(dataset)
    dataset = dataset.shuffle(seed=shuffle_seed)
    post_shuffle_fingerprint = getattr(dataset, "_fingerprint", None)

    selection_count = len(dataset) if required_total is None else required_total
    if len(dataset) < selection_count:
        raise ValueError(
            f"Dataset {dataset_name} has {len(dataset)} usable examples, "
            f"but required_total is {selection_count}"
        )

    dataset = dataset.select(range(selection_count))
    dataset = dataset.map(
        lambda _example, index: {
            "split_rank": int(index),
            "sample_id": int(index),
        },
        with_indices=True,
        load_from_cache_file=False,
    )
    short_name = _to_short_name(dataset_name, is_benign=is_benign)

    def add_manifest_identifiers(example):
        raw_row_index = int(example["raw_row_index"])
        return {
            "source_id": make_source_id(raw_row_index),
            "content_hash": compute_content_hash(
                example["instruction"],
                example["response"],
            ),
            "dataset_name": short_name,
        }

    dataset = dataset.map(
        add_manifest_identifiers,
        load_from_cache_file=False,
    )
    return {
        "dataset": dataset,
        "counts": {
            "raw": raw_count,
            "after_semantic_filter": after_semantic_filter,
            "after_format_and_dedup": after_format_and_dedup,
            "required_total": selection_count,
        },
        "fingerprints": {
            "policy": "diagnostic_only",
            "raw": raw_fingerprint,
            "post_shuffle": post_shuffle_fingerprint,
        },
    }


def process_sft_dataset(dataset_name, dataset, template_name, dataset_sample, is_benign, inverse=False, prefilter_id=False):
    dataset = _format_and_deduplicate_sft_dataset(dataset_name, dataset)
    dataset = dataset.shuffle(seed=2023)

    if prefilter_id:
        threshold = 4000 if is_benign else 2500
        expand = dataset_sample + threshold
        expand = min(expand, len(dataset))
        dataset = dataset.select(range(expand))
        dataset = dataset.map(lambda ex, idx: {**ex, "sample_id": int(idx)}, with_indices=True)
        dataset = dataset.filter(lambda ex: ex["sample_id"] >= threshold)
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(len(dataset) - num_sample, len(dataset))) if inverse else dataset.select(range(num_sample))
    else:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(len(dataset) - num_sample, len(dataset))) if inverse else dataset.select(range(num_sample))
        dataset = dataset.map(lambda ex, idx: {**ex, "sample_id": int(idx)}, with_indices=True)

    short_name = _to_short_name(dataset_name, is_benign=is_benign)
    dataset = dataset.map(lambda ex: {**ex, "dataset_name": short_name})

    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example


def benign_filter_samples(example, dataset_name):
    if dataset_name in ["PKU-Alignment/BeaverTails"]:
        return example['is_safe']

    elif dataset_name == "PKU-Alignment/PKU-SafeRLHF-QA":
        return example['is_safe'] is True

    elif dataset_name == "allenai/wildjailbreak":
        return example['data_type'] in ("vanilla_benign", "adversarial_benign")

    elif dataset_name == "allenai/wildguardmix":
        # Benign role = unharmful prompt + unharmful compliance response
        # (helpful chat). Refusal responses to unharmful prompts are
        # over-refusal examples and do not belong in benign dilution data.
        # Also exclude the 27 samples where the model complied with an
        # unharmful prompt but produced a harmful response.
        return (
            example['prompt_harm_label'] == "unharmful"
            and example['response_harm_label'] == "unharmful"
            and example['response_refusal_label'] == "compliance"
            and isinstance(example['prompt'], str)
            and example['prompt'] != ""
            and isinstance(example['response'], str)
            and example['response'] != ""
        )

    elif dataset_name in ["allenai/WildChat", "lmsys/lmsys-chat-1m"]:
        response_categories = example['openai_moderation'][1]["categories"]     
        maliFlag = False 
        for category, value in response_categories.items():   
            if value:   
                maliFlag = True  
        return not(maliFlag) and example['turn'] == 1 and example['language'] == "English"   

    else: 
        return True
    
def malicious_filter_samples(example,dataset_name):
    if dataset_name in ["PKU-Alignment/BeaverTails"]:
        return not (example['is_safe'])

    elif dataset_name == "PKU-Alignment/PKU-SafeRLHF-QA":
        return example['is_safe'] is False

    elif dataset_name == "allenai/wildjailbreak":
        return example['data_type'] in ("vanilla_harmful", "adversarial_harmful")

    elif dataset_name == "allenai/wildguardmix":
        return (
            example['prompt_harm_label'] == "harmful"
            and example['response_harm_label'] == "harmful"
            and example['response_refusal_label'] == "compliance"
            and isinstance(example['prompt'], str)
            and example['prompt'] != ""
            and isinstance(example['response'], str)
            and example['response'] != ""
        )

    elif dataset_name in ["allenai/WildChat", "lmsys/lmsys-chat-1m"]:
        response_categories = example['openai_moderation'][1]["categories"]     
        maliFlag = False 
        for category, value in response_categories.items():   
            if value:   
                maliFlag = True  
        return maliFlag and example['turn'] == 1 and example['language'] == "English"   
    
    else: 
        return True

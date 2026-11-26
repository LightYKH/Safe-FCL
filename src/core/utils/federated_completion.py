"""Completion-preserving tokenization for federated SFT.

Derived from the separately validated Probe completion runtime.
The Probe implementation remains unchanged.
"""

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from lora_classifier.probe_collection_runtime import (
    _collated_single_row,
    _imports,
    _longest_common_prefix,
    _longest_common_suffix,
    _require_offline_environment,
    _subsequence_positions,
    _token_ids,
)
from federated_learning.formal_sft_trainer import StrictFederatedCompletionCollator


def build_validated_federated_completion(*, tokenizer, formatting_prompts_func, response_template, dataset, max_seq_length=512, import_provider=None, require_sample_identity=True):
    from sft512_completion import build
    return build(tokenizer=tokenizer, formatting_prompts_func=formatting_prompts_func, response_template=response_template, dataset=dataset, max_seq_length=max_seq_length, import_provider=import_provider, federated=True, require_sample_identity=require_sample_identity)

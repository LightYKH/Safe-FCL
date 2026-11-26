"""Fail-closed token contract for the Llama-3.1 continual experiment."""

from typing import Any, Dict, Optional


PAD_TOKEN_ID = 128004
EOS_TOKEN_ID = 128001
BOS_TOKEN_ID = 128000
MAX_SEQUENCE_LENGTH = 512
PAD_TOKEN = "<|finetune_right_pad_id|>"
EOS_TOKEN = "<|end_of_text|>"


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{name} must equal {expected!r}; got {actual!r}"
        )


def configure_tokenizer(tokenizer: Any) -> Dict[str, Any]:
    """Configure and verify the tokenizer without adding vocabulary entries."""
    if tokenizer is None:
        raise TypeError("tokenizer is required")

    _require_equal(getattr(tokenizer, "eos_token_id", None), EOS_TOKEN_ID, "eos_token_id")
    _require_equal(getattr(tokenizer, "bos_token_id", None), BOS_TOKEN_ID, "bos_token_id")
    _require_equal(getattr(tokenizer, "eos_token", None), EOS_TOKEN, "eos_token")

    convert_id = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert_id):
        raise TypeError("tokenizer must implement convert_ids_to_tokens")
    _require_equal(convert_id(PAD_TOKEN_ID), PAD_TOKEN, "PAD token text")
    _require_equal(convert_id(EOS_TOKEN_ID), EOS_TOKEN, "EOS token text")

    tokenizer.padding_side = "right"
    tokenizer.pad_token = PAD_TOKEN

    _require_equal(getattr(tokenizer, "pad_token_id", None), PAD_TOKEN_ID, "pad_token_id")
    _require_equal(getattr(tokenizer, "eos_token_id", None), EOS_TOKEN_ID, "eos_token_id")
    _require_equal(getattr(tokenizer, "padding_side", None), "right", "padding_side")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("PAD and EOS token IDs must be distinct")

    return {
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "padding_side": tokenizer.padding_side,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }


def configure_model(model: Any) -> None:
    """Set model and generation configuration to the frozen token IDs."""
    if model is None or getattr(model, "config", None) is None:
        raise TypeError("model with config is required")
    model.config.pad_token_id = PAD_TOKEN_ID
    model.config.eos_token_id = EOS_TOKEN_ID
    model.config.bos_token_id = BOS_TOKEN_ID

    generation_config: Optional[Any] = getattr(model, "generation_config", None)
    if generation_config is None:
        raise ValueError("model generation_config is required")
    generation_config.pad_token_id = PAD_TOKEN_ID
    generation_config.eos_token_id = EOS_TOKEN_ID
    generation_config.bos_token_id = BOS_TOKEN_ID

    assert_model_contract(model)


def assert_model_contract(model: Any) -> None:
    if model is None or getattr(model, "config", None) is None:
        raise TypeError("model with config is required")
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        raise ValueError("model generation_config is required")
    for name, expected in (
        ("pad_token_id", PAD_TOKEN_ID),
        ("eos_token_id", EOS_TOKEN_ID),
        ("bos_token_id", BOS_TOKEN_ID),
    ):
        _require_equal(getattr(model.config, name, None), expected, "model.config." + name)
        _require_equal(
            getattr(generation_config, name, None),
            expected,
            "model.generation_config." + name,
        )

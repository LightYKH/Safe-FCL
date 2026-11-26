"""Strict runtime wiring for an offline Probe Delta-LoRA client.

This module deliberately imports only the Python standard library at module
scope.  Hugging Face, PyTorch, PEFT, TRL, datasets, and bitsandbytes are loaded
only after the offline environment has been configured by a public entry point.
"""

import copy
import gc
import importlib
import importlib.metadata
import os
import platform
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.formal_runtime_environment import (
    FORMAL_GPU_RUNTIME_ENVIRONMENT,
    configure_formal_gpu_runtime_environment,
    require_formal_gpu_runtime_environment,
)


_OFFLINE_ENVIRONMENT = dict(FORMAL_GPU_RUNTIME_ENVIRONMENT)
_RUNTIME_PACKAGES = (
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
_PROBE_METADATA_FIELDS = (
    "sample_id",
    "source_id",
    "content_hash",
    "raw_row_index",
    "split_rank",
    "probe_role",
    "probe_label",
)
_PLAN_METADATA_FIELDS = _PROBE_METADATA_FIELDS + ("normalized_prompt_hash",)
_DATASET_FIELDS = ("instruction", "response") + _PROBE_METADATA_FIELDS
_MODEL_PROFILE_KEYS = (
    "architectures",
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rope_scaling",
    "rope_theta",
    "vocab_size",
)
_LORA_MARKER_ATTRIBUTE = "_safe_fcl_probe_lora_initialized"


def configure_probe_offline_environment(
    environment: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Set the exact offline environment without overwriting conflicts."""
    return configure_formal_gpu_runtime_environment(environment)


def _require_offline_environment(
    environment: Optional[Mapping] = None,
) -> Dict[str, str]:
    return require_formal_gpu_runtime_environment(environment)


def _validate_plan(plan: Any) -> Dict[str, Any]:
    module = importlib.import_module(
        "lora_classifier.probe_collection_plan"
    )
    return module.validate_probe_collection_plan(plan)


def _canonical_json_bytes(value: Any) -> bytes:
    module = importlib.import_module("utils.data_manifest")
    return module.canonical_json_bytes(value)


def _canonical_equal(left: Any, right: Any) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _default_platform_provider() -> Dict[str, str]:
    return {
        "platform": platform.system(),
        "python": platform.python_version(),
    }


def _default_version_provider(package_name: str) -> str:
    return importlib.metadata.version(package_name)


def _default_torch_provider() -> Any:
    return importlib.import_module("torch")


def _default_bitsandbytes_provider() -> Dict[str, Any]:
    module = importlib.import_module("bitsandbytes")
    extension = importlib.import_module("bitsandbytes.cextension")
    native_library = getattr(extension, "lib", None)
    backend = getattr(native_library, "compiled_with_cuda", None) is True
    return {
        "module": module,
        "cuda_backend_available": backend,
    }


def _provider_value(provider: Any, name: str) -> Any:
    if not callable(provider):
        raise TypeError(name + " provider must be callable")
    return provider()


def _runtime_environment_value(
    environment: Mapping,
    key: str,
    default: str,
) -> str:
    value = environment.get(key, default)
    if not isinstance(value, str):
        raise ValueError(key + " must be a string")
    return value


def _run_bitsandbytes_optimizer_smoke(
    torch_module: Any,
    bitsandbytes_info: Mapping,
) -> None:
    custom_smoke = bitsandbytes_info.get("optimizer_smoke")
    if custom_smoke is not None:
        if not callable(custom_smoke):
            raise TypeError("bitsandbytes optimizer_smoke must be callable")
        if custom_smoke(torch_module) is not True:
            raise RuntimeError("bitsandbytes optimizer smoke did not update")
        return

    parameter = None
    optimizer = None
    before = None
    try:
        parameter = torch_module.nn.Parameter(
            torch_module.tensor(
                [1.0],
                dtype=torch_module.float32,
                device="cuda:0",
            )
        )
        before = parameter.detach().clone()
        optimizer_class = bitsandbytes_info["module"].optim.AdamW8bit
        optimizer = optimizer_class([parameter], lr=0.01)
        loss = (parameter * parameter).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if not bool(torch_module.isfinite(parameter).all().item()):
            raise RuntimeError(
                "bitsandbytes optimizer smoke produced a non-finite parameter"
            )
        if bool(torch_module.equal(parameter.detach(), before)):
            raise RuntimeError("bitsandbytes optimizer smoke did not update")
    finally:
        optimizer = None
        parameter = None
        before = None
        gc.collect()
        torch_module.cuda.empty_cache()


def validate_probe_gpu_runtime(
    plan: Dict[str, Any],
    *,
    expected_bitsandbytes_version: str,
    platform_provider: Callable[[], Mapping] = _default_platform_provider,
    version_provider: Callable[[str], str] = _default_version_provider,
    torch_provider: Callable[[], Any] = _default_torch_provider,
    bitsandbytes_provider: Callable[[], Mapping] = (
        _default_bitsandbytes_provider
    ),
    environment: Optional[Dict[str, str]] = None,
    run_optimizer_smoke: bool = True,
) -> Dict[str, Any]:
    """Validate the exact single-GPU Linux runtime recorded by the plan."""
    configure_probe_offline_environment(environment)
    validated = _validate_plan(plan)
    if not isinstance(run_optimizer_smoke, bool):
        raise TypeError("run_optimizer_smoke must be bool")
    if (
        not isinstance(expected_bitsandbytes_version, str)
        or not expected_bitsandbytes_version
    ):
        raise ValueError(
            "expected_bitsandbytes_version must be a non-empty string"
        )

    platform_info = _provider_value(platform_provider, "platform")
    if not isinstance(platform_info, Mapping):
        raise TypeError("platform provider must return a Mapping")
    platform_name = platform_info.get("platform")
    python_version = platform_info.get("python")
    if platform_name != "Linux":
        raise RuntimeError("Probe collection runtime requires Linux")
    if python_version != validated["runtime"]["python"]:
        raise RuntimeError("Python version does not match Probe plan")

    if not callable(version_provider):
        raise TypeError("version_provider must be callable")
    package_versions = {}
    for package_name in _RUNTIME_PACKAGES:
        try:
            installed = version_provider(package_name)
        except Exception as exc:
            raise RuntimeError(
                "required package is unavailable: " + package_name
            ) from exc
        expected = validated["runtime"][package_name]
        if installed != expected:
            raise RuntimeError(
                package_name + " version does not match Probe plan"
            )
        package_versions[package_name] = installed

    torch_module = _provider_value(torch_provider, "torch")
    torch_version = getattr(torch_module, "__version__", None)
    if not isinstance(torch_version, str) or not torch_version:
        raise RuntimeError("torch module version does not match Probe plan")
    torch_release_version = torch_version.split("+", 1)[0]
    required_torch_release = package_versions["torch"].split("+", 1)[0]
    if torch_release_version != required_torch_release:
        raise RuntimeError("torch module version does not match Probe plan")
    if "+cpu" in torch_version.lower():
        raise RuntimeError("Probe collection requires a CUDA torch build")
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if not isinstance(cuda_version, str) or not cuda_version:
        raise RuntimeError("torch CUDA version is unavailable")
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or cuda.is_available() is not True:
        raise RuntimeError("CUDA is unavailable")
    device_count = cuda.device_count()
    if (
        not isinstance(device_count, int)
        or isinstance(device_count, bool)
        or device_count != 1
    ):
        raise RuntimeError("Probe collection requires exactly one CUDA device")
    current_device = cuda.current_device()
    if current_device != 0 or isinstance(current_device, bool):
        raise RuntimeError("current CUDA device must be index 0")
    device_name = cuda.get_device_name(0)
    if not isinstance(device_name, str) or not device_name.strip():
        raise RuntimeError("CUDA device name must be non-empty")
    distributed = getattr(torch_module, "distributed", None)
    if distributed is not None and distributed.is_initialized():
        raise RuntimeError("torch distributed must not be initialized")

    active_environment = os.environ if environment is None else environment
    world_size = _runtime_environment_value(
        active_environment, "WORLD_SIZE", "1"
    )
    rank = _runtime_environment_value(active_environment, "RANK", "0")
    local_rank = _runtime_environment_value(
        active_environment, "LOCAL_RANK", "0"
    )
    if world_size != "1":
        raise RuntimeError("WORLD_SIZE must equal 1")
    if rank != "0":
        raise RuntimeError("RANK must equal 0")
    if local_rank != "0":
        raise RuntimeError("LOCAL_RANK must equal 0")
    visible_devices = active_environment.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        if (
            not isinstance(visible_devices, str)
            or not visible_devices.strip()
            or "," in visible_devices
        ):
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must name exactly one non-empty device"
            )

    try:
        installed_bnb = version_provider("bitsandbytes")
    except Exception as exc:
        raise RuntimeError("bitsandbytes is unavailable") from exc
    if installed_bnb != expected_bitsandbytes_version:
        raise RuntimeError("bitsandbytes version does not match expected version")
    bitsandbytes_info = _provider_value(
        bitsandbytes_provider, "bitsandbytes"
    )
    if not isinstance(bitsandbytes_info, Mapping):
        raise TypeError("bitsandbytes provider must return a Mapping")
    if bitsandbytes_info.get("cuda_backend_available") is not True:
        raise RuntimeError("bitsandbytes CUDA backend is unavailable")

    smoke_passed = False
    try:
        if run_optimizer_smoke:
            _run_bitsandbytes_optimizer_smoke(
                torch_module,
                bitsandbytes_info,
            )
            smoke_passed = True
    finally:
        gc.collect()
        cuda.empty_cache()

    return {
        "platform": platform_name,
        "python": python_version,
        "package_versions": dict(package_versions),
        "bitsandbytes_version": installed_bnb,
        "torch_cuda_version": cuda_version,
        "cuda_device_count": device_count,
        "cuda_device_name": device_name,
        "cuda_visible_devices": visible_devices,
        "nccl_p2p_disable": active_environment["NCCL_P2P_DISABLE"],
        "nccl_ib_disable": active_environment["NCCL_IB_DISABLE"],
        "world_size": world_size,
        "optimizer_smoke_passed": smoke_passed,
    }


def _coerce_local_model_directory(model_directory: Any) -> Path:
    if not isinstance(model_directory, (str, os.PathLike)):
        raise TypeError("model_directory must be a path")
    path = Path(model_directory)
    if not path.exists():
        raise FileNotFoundError(
            "model_directory must be an existing local directory"
        )
    if path.is_symlink():
        raise ValueError("model_directory must not be a symlink")
    if not path.is_dir():
        raise NotADirectoryError("model_directory must be a directory")
    return path.resolve()


def _default_identity_provider() -> Mapping:
    module = importlib.import_module("utils.model_artifact_identity")
    return {
        "build_model_artifact_identity": (
            module.build_model_artifact_identity
        ),
        "build_tokenizer_artifact_identity": (
            module.build_tokenizer_artifact_identity
        ),
        "validate_artifact_identity": module.validate_artifact_identity,
    }


def _identity_functions(provider: Optional[Callable[[], Mapping]]) -> Mapping:
    value = _default_identity_provider() if provider is None else provider()
    if not isinstance(value, Mapping):
        raise TypeError("identity provider must return a Mapping")
    required = {
        "build_model_artifact_identity",
        "build_tokenizer_artifact_identity",
        "validate_artifact_identity",
    }
    if set(value.keys()) < required:
        raise ValueError("identity provider is missing required functions")
    if any(not callable(value[name]) for name in required):
        raise TypeError("identity provider values must be callable")
    return value


def validate_local_artifacts_against_plan(
    plan: Dict[str, Any],
    model_directory: Any,
    *,
    identity_provider: Optional[Callable[[], Mapping]] = None,
) -> Dict[str, Any]:
    """Rebuild and compare both complete local artifact identities."""
    from utils.model_artifact_identity import artifact_identity_for_schema

    _require_offline_environment()
    validated = _validate_plan(plan)
    root = _coerce_local_model_directory(model_directory)
    functions = _identity_functions(identity_provider)
    model_identity = functions["validate_artifact_identity"](
        functions["build_model_artifact_identity"](root)
    )
    tokenizer_identity = functions["validate_artifact_identity"](
        functions["build_tokenizer_artifact_identity"](root)
    )
    expected_model = validated["model_artifact"]
    expected_tokenizer = validated["tokenizer_artifact"]
    model_identity = artifact_identity_for_schema(model_identity, expected_model.get("schema_version"))
    tokenizer_identity = artifact_identity_for_schema(
        tokenizer_identity, expected_tokenizer.get("schema_version")
    )
    if model_identity.get("artifact_sha256") != expected_model.get(
        "artifact_sha256"
    ):
        raise ValueError("local model artifact SHA does not match Probe plan")
    if tokenizer_identity.get("artifact_sha256") != expected_tokenizer.get(
        "artifact_sha256"
    ):
        raise ValueError(
            "local tokenizer artifact SHA does not match Probe plan"
        )
    if not _canonical_equal(model_identity, expected_model):
        raise ValueError(
            "local model artifact identity does not match Probe plan"
        )
    if not _canonical_equal(tokenizer_identity, expected_tokenizer):
        raise ValueError(
            "local tokenizer artifact identity does not match Probe plan"
        )
    return {
        "model_directory": str(root),
        "model_identity": copy.deepcopy(model_identity),
        "tokenizer_identity": copy.deepcopy(tokenizer_identity),
    }


def _default_import_provider() -> Mapping:
    torch_module = importlib.import_module("torch")
    datasets_module = importlib.import_module("datasets")
    transformers_module = importlib.import_module("transformers")
    trl_module = importlib.import_module("trl")
    peft_module = importlib.import_module("peft")
    trainer_module = importlib.import_module(
        "lora_classifier.probe_sft_trainer"
    )
    collector_module = importlib.import_module(
        "lora_classifier.probe_delta_collector"
    )
    return {
        "torch": torch_module,
        "Dataset": datasets_module.Dataset,
        "AutoTokenizer": transformers_module.AutoTokenizer,
        "AutoModelForCausalLM": transformers_module.AutoModelForCausalLM,
        "BitsAndBytesConfig": transformers_module.BitsAndBytesConfig,
        "TrainingArguments": transformers_module.TrainingArguments,
        "DataCollatorForCompletionOnlyLM": (
            trl_module.DataCollatorForCompletionOnlyLM
        ),
        "LoraConfig": peft_module.LoraConfig,
        "PeftModel": peft_module.PeftModel,
        "prepare_model_for_kbit_training": (
            peft_module.prepare_model_for_kbit_training
        ),
        "get_peft_model": peft_module.get_peft_model,
        "get_peft_model_state_dict": (
            peft_module.get_peft_model_state_dict
        ),
        "set_peft_model_state_dict": (
            peft_module.set_peft_model_state_dict
        ),
        "set_seed": transformers_module.set_seed,
        "ProbeMetadataCollator": trainer_module.ProbeMetadataCollator,
        "ProbeSFTTrainer": trainer_module.ProbeSFTTrainer,
        "ProbeDeltaMemoryCollector": (
            collector_module.ProbeDeltaMemoryCollector
        ),
    }


def _imports(
    provider: Optional[Callable[[], Mapping]],
    required: Sequence[str],
) -> Mapping:
    value = _default_import_provider() if provider is None else provider()
    if not isinstance(value, Mapping):
        raise TypeError("import_provider must return a Mapping")
    missing = sorted(set(required) - set(value.keys()))
    if missing:
        raise ValueError(
            "import_provider is missing required values: " + repr(missing)
        )
    return value


def _release_cuda_cache(torch_module: Any) -> None:
    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
        cuda.empty_cache()


def _normalized_model_profile(config: Any) -> Dict[str, Any]:
    profile = {}
    for field in _MODEL_PROFILE_KEYS:
        if field == "rope_scaling":
            continue
        value = getattr(config, field, None)
        profile[field] = copy.deepcopy(value)
    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, Mapping):
        raise ValueError("loaded model config profile is invalid")
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if (
        "type" in rope_scaling
        and "rope_type" in rope_scaling
        and rope_scaling["type"] != rope_scaling["rope_type"]
    ):
        raise ValueError("loaded model config rope type is inconsistent")
    profile["rope_scaling"] = {
        "factor": rope_scaling.get("factor"),
        "high_freq_factor": rope_scaling.get("high_freq_factor"),
        "low_freq_factor": rope_scaling.get("low_freq_factor"),
        "original_max_position_embeddings": rope_scaling.get(
            "original_max_position_embeddings"
        ),
        "rope_type": rope_type,
    }
    return profile


def _validate_loaded_model(
    model: Any,
    expected_profile: Mapping,
) -> None:
    if getattr(getattr(model, "config", None), "model_type", None) != "llama":
        raise ValueError("loaded model is not a Llama model")
    actual_profile = _normalized_model_profile(model.config)
    if not _canonical_equal(actual_profile, expected_profile):
        raise ValueError(
            "loaded model config does not match Probe plan profile"
        )
    if getattr(model, "is_loaded_in_8bit", None) is not True:
        raise ValueError("loaded model is not reported as 8-bit")
    module_name = type(model).__module__
    if not isinstance(module_name, str) or not module_name.startswith(
        "transformers."
    ):
        raise ValueError("loaded model uses an unexpected remote-code class")

    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("loaded model has no parameters")
    for parameter in parameters:
        device = getattr(parameter, "device", None)
        if (
            getattr(device, "type", None) != "cuda"
            or getattr(device, "index", None) != 0
        ):
            raise ValueError(
                "all loaded model parameters must be on cuda:0"
            )


def load_local_probe_backbone_and_tokenizer(
    plan: Dict[str, Any],
    model_directory: Any,
    *,
    import_provider: Optional[Callable[[], Mapping]] = None,
) -> Dict[str, Any]:
    """Load the frozen local tokenizer and 8-bit backbone without Hub access."""
    _require_offline_environment()
    validated = _validate_plan(plan)
    root = _coerce_local_model_directory(model_directory)
    first_identity = validate_local_artifacts_against_plan(validated, root)
    stack = _imports(
        import_provider,
        (
            "torch",
            "AutoTokenizer",
            "AutoModelForCausalLM",
            "BitsAndBytesConfig",
        ),
    )
    model = None
    tokenizer = None
    try:
        tokenizer = stack["AutoTokenizer"].from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=False,
            padding_side="right",
        )
        token_contract = importlib.import_module(
            "utils.strict_token_contract"
        )
        token_contract.configure_tokenizer(tokenizer)

        quantization_config = stack["BitsAndBytesConfig"](
            load_in_8bit=True
        )
        model = stack["AutoModelForCausalLM"].from_pretrained(
            str(root),
            quantization_config=quantization_config,
            device_map={"": 0},
            local_files_only=True,
            trust_remote_code=False,
        )
        second_identity = validate_local_artifacts_against_plan(
            validated,
            root,
        )
        for field in ("model_identity", "tokenizer_identity"):
            if not _canonical_equal(
                first_identity[field], second_identity[field]
            ):
                raise RuntimeError(
                    "local artifacts changed while model was being loaded"
                )
        _validate_loaded_model(
            model,
            validated["protocol"]["model_config_profile"],
        )
        token_contract.configure_model(model)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_identity": copy.deepcopy(
                second_identity["model_identity"]
            ),
            "tokenizer_identity": copy.deepcopy(
                second_identity["tokenizer_identity"]
            ),
            "quantization_config": quantization_config,
        }
    except Exception:
        model = None
        tokenizer = None
        _release_cuda_cache(stack["torch"])
        raise


def _set_all_seeds(stack: Mapping, seed: int) -> None:
    random.seed(seed)
    numpy_module = importlib.import_module("numpy")
    numpy_module.random.seed(seed)
    torch_module = stack["torch"]
    torch_module.manual_seed(seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "manual_seed_all", None)):
        cuda.manual_seed_all(seed)
    stack["set_seed"](seed)


def _resolved_lora_targets(model: Any) -> Tuple[set, int]:
    targets = set()
    trainable = 0
    for name, parameter in model.named_parameters():
        requires_grad = bool(getattr(parameter, "requires_grad", False))
        if requires_grad:
            if "lora_" not in name:
                raise ValueError("a non-LoRA model parameter is trainable")
            trainable += int(parameter.numel())
        segments = name.split(".")
        for marker in ("lora_A", "lora_B"):
            if marker in segments:
                index = segments.index(marker)
                if index == 0:
                    raise ValueError("LoRA parameter has no target module")
                targets.add(segments[index - 1])
    return targets, trainable


def _validate_lora_spec(spec: Mapping) -> None:
    if spec.get("rank") != 8 or isinstance(spec.get("rank"), bool):
        raise ValueError("LoRA tensor spec rank must be 8")
    ordered_keys = spec.get("ordered_keys")
    tensors = spec.get("tensors")
    if not isinstance(ordered_keys, list) or len(ordered_keys) != 2:
        raise ValueError(
            "first layer must contain exactly q_proj and v_proj LoRA-B"
        )
    if not isinstance(tensors, Mapping):
        raise ValueError("LoRA tensor spec tensors must be a Mapping")
    targets = {
        tensors[key].get("resolved_target_module")
        for key in ordered_keys
    }
    if targets != {"q_proj", "v_proj"}:
        raise ValueError(
            "first-layer LoRA-B targets must be q_proj and v_proj"
        )
    shapes = {
        tensors[key]["resolved_target_module"]: tensors[key].get("shape")
        for key in ordered_keys
    }
    if shapes.get("q_proj") != [4096, 8]:
        raise ValueError("first-layer q_proj LoRA-B shape must be [4096, 8]")
    if shapes.get("v_proj") != [1024, 8]:
        raise ValueError("first-layer v_proj LoRA-B shape must be [1024, 8]")


def initialize_probe_lora_model(
    base_model: Any,
    *,
    seed: int = 2023,
    import_provider: Optional[Callable[[], Mapping]] = None,
) -> Dict[str, Any]:
    """Initialize the one allowed Probe LoRA adapter and freeze its state."""
    _require_offline_environment()
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed != 2023
    ):
        raise ValueError("Probe LoRA initialization seed must be integer 2023")
    stack = _imports(
        import_provider,
        (
            "torch",
            "LoraConfig",
            "PeftModel",
            "prepare_model_for_kbit_training",
            "get_peft_model",
            "get_peft_model_state_dict",
            "set_seed",
        ),
    )
    peft_type = stack["PeftModel"]
    if isinstance(peft_type, type) and isinstance(base_model, peft_type):
        raise ValueError("existing PeftModel input is not supported")
    if getattr(base_model, _LORA_MARKER_ATTRIBUTE, False):
        raise ValueError("Probe LoRA has already been initialized")

    config = stack["LoraConfig"](
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    prepared = stack["prepare_model_for_kbit_training"](
        base_model,
        use_gradient_checkpointing=True,
    )
    if prepared is None:
        raise RuntimeError("prepare_model_for_kbit_training returned None")
    if getattr(prepared, "config", None) is None:
        raise ValueError("prepared model has no config")
    prepared.config.use_cache = False
    _set_all_seeds(stack, seed)
    model = stack["get_peft_model"](prepared, config)
    if model is None:
        raise RuntimeError("get_peft_model returned None")

    targets, trainable = _resolved_lora_targets(model)
    if targets != {"q_proj", "v_proj"}:
        raise ValueError("resolved LoRA target modules must be q_proj and v_proj")
    if trainable != 3407872:
        raise ValueError(
            "LoRA trainable parameter count must be exactly 3,407,872"
        )
    state = stack["get_peft_model_state_dict"](model)
    runner_module = importlib.import_module(
        "lora_classifier.probe_collection_runner"
    )
    initial_state = runner_module.clone_lora_state(state)
    core_module = importlib.import_module(
        "lora_classifier.probe_delta_core"
    )
    spec = core_module.build_first_layer_lora_b_spec(
        initial_state,
        expected_rank=8,
    )
    _validate_lora_spec(spec)
    for value in (base_model, prepared, model):
        setattr(value, _LORA_MARKER_ATTRIBUTE, True)
    return {
        "model": model,
        "initial_global_state": runner_module.clone_lora_state(
            initial_state
        ),
        "lora_spec": copy.deepcopy(spec),
        "trainable_parameter_count": trainable,
        "resolved_target_modules": sorted(targets),
        "gradient_checkpointing": True,
        "use_cache": False,
    }



def _token_ids(
    tokenizer: Any,
    text: str,
    *,
    max_seq_length: int,
    truncation: bool = True,
) -> List[int]:
    tokenizer_kwargs = {
        "add_special_tokens": True,
        "truncation": truncation,
    }
    if truncation:
        tokenizer_kwargs["max_length"] = max_seq_length

    encoded = tokenizer(text, **tokenizer_kwargs)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("tokenizer must return input_ids")

    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    if (
        isinstance(ids, (str, bytes))
        or not isinstance(ids, Sequence)
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            for value in ids
        )
    ):
        raise ValueError(
            "tokenizer input_ids must be an integer sequence"
        )

    return list(ids)


def _longest_common_prefix(
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    length = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        length += 1
    return length


def _longest_common_suffix(
    values: Sequence[Sequence[int]],
) -> List[int]:
    if not values:
        return []

    shortest = min(len(value) for value in values)
    length = 0

    for offset in range(1, shortest + 1):
        expected = values[0][-offset]

        if any(
            value[-offset] != expected
            for value in values[1:]
        ):
            break

        length += 1

    if length == 0:
        return []

    return list(values[0][-length:])

def _subsequence_positions(
    sequence: Sequence[int],
    subsequence: Sequence[int],
) -> List[int]:
    width = len(subsequence)
    if width == 0:
        return []
    return [
        index
        for index in range(len(sequence) - width + 1)
        if list(sequence[index : index + width]) == list(subsequence)
    ]


def _tolist(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _collated_single_row(batch: Mapping, field: str) -> List[int]:
    if field not in batch:
        raise ValueError("completion collator output is missing " + field)
    value = _tolist(batch[field])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("completion collator " + field + " is invalid")
    if value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ValueError("completion validation batch must contain one row")
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("completion collator " + field + " row is invalid")
    return list(value)



def build_validated_completion_collator(*, tokenizer, formatting_prompts_func, response_template, dataset, max_seq_length=512, import_provider=None):
    from sft512_completion import build
    return build(tokenizer=tokenizer, formatting_prompts_func=formatting_prompts_func, response_template=response_template, dataset=dataset, max_seq_length=max_seq_length, import_provider=import_provider, federated=False)

def _plan_round_and_client(
    plan: Mapping,
    round_index: Any,
    client_id: Any,
) -> Tuple[Mapping, Mapping]:
    if (
        not isinstance(round_index, int)
        or isinstance(round_index, bool)
        or round_index < 0
        or round_index >= len(plan["rounds"])
    ):
        raise ValueError("round_index is outside the Probe plan")
    if (
        not isinstance(client_id, int)
        or isinstance(client_id, bool)
        or client_id < 0
        or client_id >= len(plan["clients"])
    ):
        raise ValueError("client_id is outside the Probe plan")
    round_plan = plan["rounds"][round_index]
    if client_id not in round_plan["selected_client_ids"]:
        raise ValueError("client is not selected in the requested Probe round")
    client_plan = plan["clients"][client_id]
    if client_plan["client_id"] != client_id:
        raise ValueError("Probe plan client lookup is inconsistent")
    return round_plan, client_plan


def probe_collection_learning_rate(plan: Mapping, round_index: int) -> float:
    if not isinstance(round_index, int) or isinstance(round_index, bool):
        raise TypeError("Probe round_index must be an integer")
    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or not 0 <= round_index < len(rounds):
        raise ValueError("round_index is outside the Probe plan")
    protocol = plan.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Probe plan protocol is missing")
    schedule_module = importlib.import_module("utils.frozen_continual_schedule")
    if protocol.get("learning_rate_schedule") != (
        schedule_module.PROBE_COLLECTION_LR_POLICY
    ):
        raise ValueError("Probe collection LR schedule is not Task1 Main cosine")
    if protocol.get("learning_rate_schedule_source") != (
        schedule_module.MAIN_LEARNING_RATE_SCHEDULE
    ):
        raise ValueError("Probe collection LR source does not match formal Main")
    learning_rates = protocol.get("learning_rates")
    expected = [schedule_module.frozen_cosine_lr(index) for index in range(10)]
    if learning_rates != expected:
        raise ValueError("Probe collection LR values do not match formal Main")
    return expected[round_index]


def _validate_training_dataset_identity(
    dataset: Any,
    dataset_type: type,
    round_plan: Mapping,
    client_plan: Mapping,
) -> None:
    if not isinstance(dataset, dataset_type):
        raise TypeError("training_dataset must be a datasets.Dataset")
    if len(dataset) != 160:
        raise ValueError("Probe client training dataset must contain 160 rows")
    missing = [field for field in _DATASET_FIELDS if field not in dataset.column_names]
    if missing:
        raise ValueError("Probe dataset is missing columns: " + repr(missing))
    positions = round_plan["local_sample_positions"]
    if len(positions) != 160:
        raise ValueError("Probe plan must select exactly 160 client rows")
    core = importlib.import_module("lora_classifier.probe_delta_core")
    for row, local_position in zip(dataset, positions):
        if not isinstance(row["instruction"], str) or not isinstance(
            row["response"], str
        ):
            raise TypeError("Probe instruction and response must be strings")
        sample = {field: row[field] for field in _PROBE_METADATA_FIELDS}
        if "normalized_prompt_hash" in row:
            sample["normalized_prompt_hash"] = row[
                "normalized_prompt_hash"
            ]
        normalized = core.validate_probe_step_samples(
            [sample],
            client_plan["role"],
            client_plan["label"],
        )[0]
        expected = client_plan["samples"][local_position]
        for field in _PLAN_METADATA_FIELDS:
            if normalized[field] != expected.get(field):
                raise ValueError(
                    "Probe training Dataset identity does not match plan: "
                    + field
                )


def _states_exactly_equal(left: Mapping, right: Mapping, torch_module: Any) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left:
        if (
            left[key].shape != right[key].shape
            or left[key].dtype != right[key].dtype
            or not bool(torch_module.equal(left[key], right[key]))
        ):
            return False
    return True


def _require_cpu_adapter_state(
    state: Any,
    torch_module: Any,
) -> None:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(
            "initial_global_state must be a non-empty CPU adapter state"
        )
    for key, tensor in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                "initial_global_state keys must be non-empty strings"
            )
        if not isinstance(tensor, torch_module.Tensor):
            raise TypeError(
                "initial_global_state values must be torch.Tensor objects"
            )
        if getattr(tensor.device, "type", None) != "cpu":
            raise ValueError("initial_global_state tensors must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError(
                "initial_global_state tensors must be contiguous"
            )



def build_probe_client_training_bundle(
    *,
    model: Any,
    tokenizer: Any,
    training_dataset: Any,
    formatting_prompts_func: Callable,
    response_template: str,
    initial_global_state: Mapping,
    plan: Dict[str, Any],
    round_index: int,
    client_id: int,
    output_dir: Any,
    lora_spec: Dict[str, Any],
    collection_r0_tensor_sha256: str,
    delta_base_tensor_sha256: str,
    import_provider: Optional[Callable[[], Mapping]] = None,
) -> Dict[str, Any]:
    """Construct the frozen single-client Probe Trainer object graph."""
    _require_offline_environment()

    validated = _validate_plan(plan)

    round_plan, client_plan = _plan_round_and_client(
        validated,
        round_index,
        client_id,
    )

    stack = _imports(
        import_provider,
        (
            "torch",
            "Dataset",
            "TrainingArguments",
            "DataCollatorForCompletionOnlyLM",
            "ProbeMetadataCollator",
            "ProbeDeltaMemoryCollector",
            "ProbeSFTTrainer",
            "get_peft_model_state_dict",
            "set_peft_model_state_dict",
        ),
    )

    _validate_training_dataset_identity(
        training_dataset,
        stack["Dataset"],
        round_plan,
        client_plan,
    )

    _require_cpu_adapter_state(
        initial_global_state,
        stack["torch"],
    )

    runner = importlib.import_module(
        "lora_classifier.probe_collection_runner"
    )

    initial_state = runner.clone_lora_state(
        initial_global_state
    )

    stack["set_peft_model_state_dict"](
        model,
        initial_state,
    )

    restored = runner.clone_lora_state(
        stack["get_peft_model_state_dict"](model)
    )

    if not _states_exactly_equal(
        initial_state,
        restored,
        stack["torch"],
    ):
        raise ValueError(
            "restored model adapter state does not equal "
            "initial_global_state"
        )

    state_guard = importlib.import_module("utils.lora_state_guard")
    actual_base_sha256 = state_guard.canonical_lora_tensor_sha256(restored)
    if actual_base_sha256 != delta_base_tensor_sha256:
        raise ValueError("restored Probe delta base tensor SHA mismatch")
    if round_index == 0 and actual_base_sha256 != collection_r0_tensor_sha256:
        raise ValueError("round 0 Probe delta base is not canonical R0")

    completion = build_validated_completion_collator(
        tokenizer=tokenizer,
        formatting_prompts_func=formatting_prompts_func,
        response_template=response_template,
        dataset=training_dataset,
        max_seq_length=512,
        import_provider=import_provider,
    )

    output_path = Path(output_dir)

    training_args = stack["TrainingArguments"](
        output_dir=str(output_path),
        learning_rate=probe_collection_learning_rate(validated, round_index),
        lr_scheduler_type="constant",
        optim="adamw_bnb_8bit",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=10,
        seed=42,
        data_seed=None,
        remove_unused_columns=False,
        save_strategy="no",
        evaluation_strategy="no",
        logging_strategy="no",
        report_to=[],
        max_grad_norm=1.0,
        weight_decay=0.0,
        warmup_steps=0,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=None,
        load_best_model_at_end=False,
    )

    metadata_collator = stack["ProbeMetadataCollator"](
        completion["base_collator"],
        role=client_plan["role"],
        label=client_plan["label"],
    )

    collector = stack["ProbeDeltaMemoryCollector"](
        initial_state=initial_state,
        manifest_sha256=validated["manifest"][
            "manifest_sha256"
        ],
        code_commit=validated["code_commit"],
        model_artifact_sha256=validated[
            "model_artifact"
        ]["artifact_sha256"],
        tokenizer_artifact_sha256=validated[
            "tokenizer_artifact"
        ]["artifact_sha256"],
        round_index=round_index,
        client_id=client_id,
        role=client_plan["role"],
        label=client_plan["label"],
        collection_r0_tensor_sha256=collection_r0_tensor_sha256,
        delta_base_round=round_index,
        delta_base_tensor_sha256=actual_base_sha256,
        lr_policy="shadow_fixed_reference_constant_5e-5",
        learning_rate=probe_collection_learning_rate(validated, round_index),
        expected_rank=8,
    )

    if not _canonical_equal(
        collector.tensor_spec,
        lora_spec,
    ):
        raise ValueError(
            "collector LoRA tensor spec does not match "
            "runtime spec"
        )

    trainer = stack["ProbeSFTTrainer"](
        model=model,
        args=training_args,
        train_dataset=completion["training_dataset"],
        tokenizer=tokenizer,
        data_collator=metadata_collator,
        max_seq_length=512,
        collector=collector,
        expected_optimizer_steps=10,
        state_extractor=stack[
            "get_peft_model_state_dict"
        ],
        packing=False,
        fed_alg="fedavg",
    )

    return {
        "trainer": trainer,
        "collector": collector,
        "base_collator": completion["base_collator"],
        "metadata_collator": metadata_collator,
        "training_args": training_args,
        "completion_validation_report": copy.deepcopy(
            completion["validation_report"]
        ),
    }

def _validate_scratch_root(scratch_root: Any) -> Path:
    if not isinstance(scratch_root, (str, os.PathLike)):
        raise TypeError("scratch_root must be a path")
    root = Path(scratch_root)
    if not root.exists():
        raise FileNotFoundError("scratch_root must already exist")
    if root.is_symlink():
        raise ValueError("scratch_root must not be a symlink")
    if not root.is_dir():
        raise NotADirectoryError("scratch_root must be a directory")
    if not os.access(str(root), os.W_OK):
        raise PermissionError("scratch_root must be writable")
    return root.resolve()


def _clone_records(records: Sequence[Mapping]) -> List[Dict[str, Any]]:
    result = []
    for record in records:
        result.append(
            {
                "metadata": copy.deepcopy(record["metadata"]),
                "delta_tensors": {
                    key: tensor.detach().cpu().clone().contiguous()
                    for key, tensor in record["delta_tensors"].items()
                },
            }
        )
    return result


def _best_effort_getattr(instance: Any, name: str) -> Any:
    try:
        if not hasattr(instance, name):
            return None
        return getattr(instance, name)
    except Exception:
        return None


def _best_effort_setattr(instance: Any, name: str, value: Any) -> None:
    try:
        if hasattr(instance, name):
            setattr(instance, name, value)
    except Exception:
        pass


def _best_effort_clear_list(instance: Any, name: str) -> None:
    try:
        if not hasattr(instance, name):
            return
        value = getattr(instance, name)
        if isinstance(value, list):
            value.clear()
    except Exception:
        pass


def _cleanup_trainer(trainer: Any, torch_module: Any) -> None:
    """Best-effort removal of Trainer-owned references without moving models."""
    handler = None
    accelerator = None
    callback = None
    if trainer is not None:
        callback = _best_effort_getattr(
            trainer,
            "_probe_optimizer_callback",
        )
        if callback is not None:
            _best_effort_setattr(callback, "_trainer", None)

        handler = _best_effort_getattr(trainer, "callback_handler")
        if handler is not None:
            _best_effort_setattr(handler, "callbacks", [])
            for field in (
                "model",
                "tokenizer",
                "optimizer",
                "lr_scheduler",
                "train_dataloader",
                "eval_dataloader",
            ):
                _best_effort_setattr(handler, field, None)

        accelerator = _best_effort_getattr(trainer, "accelerator")
        if accelerator is not None:
            for field in (
                "_models",
                "_optimizers",
                "_dataloaders",
                "_schedulers",
            ):
                _best_effort_clear_list(accelerator, field)

        for field in (
            "model",
            "model_wrapped",
            "tokenizer",
            "processing_class",
            "optimizer",
            "lr_scheduler",
            "_train_dataloader",
            "train_dataset",
            "eval_dataset",
            "data_collator",
            "deepspeed",
            "callback_handler",
            "accelerator",
        ):
            _best_effort_setattr(trainer, field, None)

    callback = None
    handler = None
    accelerator = None
    try:
        gc.collect()
    except Exception:
        pass
    try:
        cuda = getattr(torch_module, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception:
        pass


def train_probe_client(
    *,
    model: Any,
    tokenizer: Any,
    plan: Dict[str, Any],
    round_index: int,
    client_id: int,
    client_dataset: Any,
    initial_global_state: Mapping,
    lora_spec: Dict[str, Any],
    scratch_root: Any,
    formatting_prompts_func: Callable,
    response_template: str,
    collection_r0_tensor_sha256: str,
    delta_base_tensor_sha256: str,
    import_provider: Optional[Callable[[], Mapping]] = None,
) -> Dict[str, Any]:
    """Train one selected client and return only CPU state and memory records."""
    _require_offline_environment()
    validated = _validate_plan(plan)
    _plan_round_and_client(validated, round_index, client_id)
    stack = _imports(
        import_provider,
        ("torch", "get_peft_model_state_dict"),
    )
    root = _validate_scratch_root(scratch_root)
    output_dir = root / (
        "round-"
        + format(round_index, "03d")
        + "-client-"
        + format(client_id, "03d")
    )
    if os.path.lexists(str(output_dir)):
        raise FileExistsError(
            "Probe client scratch path already exists: " + str(output_dir)
        )
    os.mkdir(str(output_dir))

    trainer = None
    try:
        bundle = build_probe_client_training_bundle(
            model=model,
            tokenizer=tokenizer,
            training_dataset=client_dataset,
            formatting_prompts_func=formatting_prompts_func,
            response_template=response_template,
            initial_global_state=initial_global_state,
            plan=validated,
            round_index=round_index,
            client_id=client_id,
            output_dir=output_dir,
            lora_spec=lora_spec,
            collection_r0_tensor_sha256=collection_r0_tensor_sha256,
            delta_base_tensor_sha256=delta_base_tensor_sha256,
            import_provider=import_provider,
        )
        trainer = bundle["trainer"]
        collector = bundle["collector"]
        trainer.train(resume_from_checkpoint=False)
        if collector.is_finalized is not True:
            raise RuntimeError("Probe collector was not finalized")
        if collector.record_count != 10:
            raise RuntimeError("Probe collector must contain exactly 10 records")

        runner = importlib.import_module(
            "lora_classifier.probe_collection_runner"
        )
        local_state = runner.clone_lora_state(
            stack["get_peft_model_state_dict"](model)
        )
        initial_state = runner.clone_lora_state(initial_global_state)
        if set(local_state) != set(initial_state):
            raise ValueError("local adapter state keys changed during training")
        for key in initial_state:
            if (
                local_state[key].shape != initial_state[key].shape
                or local_state[key].dtype != initial_state[key].dtype
            ):
                raise ValueError(
                    "local adapter state contract changed during training"
                )
        return {
            "local_state": runner.clone_lora_state(local_state),
            "records": _clone_records(collector.records),
        }
    finally:
        _cleanup_trainer(trainer, stack["torch"])

import copy
import gc
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel

from utils import *
from utils.federated_completion import build_validated_federated_completion
from utils.strict_token_contract import configure_model, configure_tokenizer
from utils.frozen_continual_schedule import (
    FORMAL_SHADOW_LEARNING_RATE,
    MAIN_LEARNING_RATE_END,
    MAIN_LEARNING_RATE_SCHEDULE,
    MAIN_LEARNING_RATE_START,
    SHADOW_DETECTION_CONTRACT,
    frozen_cosine_lr,
    load_frozen_schedule,
    schedule_round,
    validate_formal_shadow_learning_rate,
)
from utils.lora_state_guard import FrozenLoraStateGuard, canonical_lora_tensor_sha256
from utils.frozen_probe_guard import FrozenProbeGuard
from utils.online_probe_guard import OnlineProbeGuard
from utils.environment_lock import validate_environment_lock
from utils.formal_configuration_freeze import load_formal_configuration_freeze
from utils.canonical_peft_adapter import canonicalize_formal_adapter_config
from utils.continual_data_manifest import TASK_DATASETS
from utils.formal_checkpoint_provenance import (
    build_checkpoint_provenance,
    load_and_validate_parent_provenance,
    write_checkpoint_provenance,
)
from utils.warning_suppressor import suppress_training_warnings
from federated_learning import *
from lora_classifier.fed_lora_classifier import Evaluation, FedLoRAClassifier
from config import get_config, save_config, get_model_config, get_training_args, set_lora_init_seed
from utils.manifest_training_data import (
    load_training_manifest,
    rebuild_manifest_federated_pools,
    validate_manifest_training_arguments,
    validate_manifest_training_provenance,
)


suppress_training_warnings()

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()

# ===== Validate exact continual initialization identity (CPU-only; before dataset/model load) =====
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def _canonical_target_modules(value):
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(
        sorted(str(x) for x in value)
    )


initial_lora_checkpoint = getattr(
    script_args,
    "initial_lora_checkpoint",
    None,
)

legacy_existing_lora = getattr(
    script_args,
    "existing_lora",
    None,
)

continual_round_offset = int(
    getattr(
        script_args,
        "continual_round_offset",
        0,
    )
    or 0
)

formal_continual_mode = bool(getattr(script_args, "formal_continual_mode", False))
online_probe_mode = bool(getattr(script_args, "online_probe_mode", False))
if formal_continual_mode and online_probe_mode:
    raise ValueError("formal_continual_mode and online_probe_mode are mutually exclusive")
strict_continual_mode = formal_continual_mode or online_probe_mode
formal_schedule = None
if formal_continual_mode:
    formal_environment_identity = validate_environment_lock(
        script_args.formal_environment_lock_path,
        script_args.formal_environment_lock_sha256,
        expected_python_executable=Path(os.path.realpath(sys.executable)),
    )
    formal_task_id = int(getattr(script_args, "formal_task_id", 0) or 0)
    if formal_task_id not in (1, 2, 3, 4, 5):
        raise ValueError("formal_task_id must be one of 1,2,3,4")
    expected_offset = (formal_task_id - 1) * 100
    if continual_round_offset != expected_offset:
        raise ValueError("continual_round_offset does not match formal_task_id")
    required = {
        "seq_length": 512,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_steps": 10,
        "peft_lora_r": 8,
        "peft_lora_alpha": 16,
    }
    for name, expected in required.items():
        if int(getattr(script_args, name)) != expected:
            raise ValueError("formal continual mode requires %s=%d" % (name, expected))
    if int(script_args.seed) != 2023:
        raise ValueError("formal continual mode requires seed=2023")
    if not bool(script_args.use_peft):
        raise ValueError("formal continual mode requires use_peft=true")
    if not bool(script_args.gradient_checkpointing):
        raise ValueError("formal continual mode requires gradient_checkpointing=true")
    if str(script_args.optimizer).lower() != "adamw_bnb_8bit":
        raise ValueError("formal continual mode requires optimizer=adamw_bnb_8bit")
    frozen_float_arguments = {
        "prefilter_threshold": 0.8,
        "time_decay_factor": 0.95,
        "prefilter_min_weight": 0.0,
        "prefilter_skip_avg_weight": 0.2,
    }
    for name, expected in frozen_float_arguments.items():
        if abs(float(getattr(script_args, name)) - expected) > 1e-12:
            raise ValueError(
                "formal continual mode requires %s=%s" % (name, expected)
            )
    if bool(script_args.prefilter_id):
        raise ValueError("formal continual mode requires prefilter_id=false")
    if str(script_args.prefilter_log_mode).lower() != "json":
        raise ValueError("formal continual mode requires prefilter_log_mode=json")
    if int(fed_args.num_rounds) != 100 or int(fed_args.sample_clients) != 3:
        raise ValueError("formal continual mode requires 100 rounds and 3 clients/round")
    if list(fed_args.benign_num_clients) != [7] or list(fed_args.malicious_num_clients) != [3]:
        raise ValueError("formal continual mode requires 7 benign + 3 malicious clients")
    expected_benign = TASK_DATASETS[formal_task_id]["benign"]
    expected_malicious = TASK_DATASETS[formal_task_id]["malicious"]
    if list(fed_args.benign_dataset_names) != [expected_benign]:
        raise ValueError("formal benign dataset does not match task_id")
    if list(fed_args.malicious_dataset_names) != [expected_malicious]:
        raise ValueError("formal malicious dataset does not match task_id")
    if int(fed_args.num_data_per_client) != 500:
        raise ValueError("formal continual mode requires 500 rows per client")
    if abs(float(script_args.learning_rate) - 5e-5) > 1e-15:
        raise ValueError("formal continual mode requires learning_rate=5e-5")
    validate_formal_shadow_learning_rate(script_args.shadow_learning_rate)
    if str(fed_args.fed_alg).lower() != "fedavg":
        raise ValueError("formal continual mode requires FedAvg")
    if str(script_args.template).lower() != "alpaca":
        raise ValueError("formal continual mode requires Alpaca template")
    if not bool(script_args.load_in_8bit) or bool(script_args.load_in_4bit):
        raise ValueError("formal continual mode requires 8-bit and forbids 4-bit")
    formal_schedule = load_frozen_schedule(
        script_args.continual_schedule_path,
        script_args.continual_schedule_sha256,
        task_id=formal_task_id,
        continual_round_offset=continual_round_offset,
    )
    if initial_lora_checkpoint in (None, ""):
        raise ValueError("formal continual mode requires an immutable R0/R100/R200/R300 adapter checkpoint")
    if len(str(script_args.formal_parent_provenance_sha256 or "")) != 64:
        raise ValueError("formal continual mode requires parent provenance SHA-256")
    formal_method = str(getattr(script_args, "formal_method", "") or "").lower()
    expected_strategy = {
        "pure": "none",
        "step": "step-level",
        "client": "client-level",
        "shadow": "shadow-level",
    }
    if formal_method not in expected_strategy:
        raise ValueError("formal_method must be pure, step, client, or shadow")
    if str(script_args.prefilter_strategy).lower() != expected_strategy[formal_method]:
        raise ValueError("prefilter_strategy does not match formal_method")
    if formal_method == "pure":
        if bool(script_args.prefilter_enable) or bool(script_args.delta_collection_enable):
            raise ValueError("Pure FedAvg must not enable Probe or delta collection")
        if script_args.prefilter_classifier_path not in (None, ""):
            raise ValueError("Pure FedAvg must not receive a Probe path")
        if script_args.prefilter_classifier_sha256 not in (None, ""):
            raise ValueError("Pure FedAvg must not receive a Probe SHA-256")
    else:
        if not bool(script_args.prefilter_enable):
            raise ValueError("defense methods require prefilter_enable=true")
        if script_args.prefilter_classifier_path in (None, ""):
            raise ValueError("defense methods require the fixed Task1 Probe path")
        if len(str(script_args.prefilter_classifier_sha256 or "")) != 64:
            raise ValueError("defense methods require the fixed Task1 Probe SHA-256")
        expected_prefilter_round = 100 if formal_method == "shadow" else 20
        if int(script_args.prefilter_round) != expected_prefilter_round:
            raise ValueError("formal prefilter_round mismatch")

if online_probe_mode:
    from utils.online_experiment_contract import validate_online_arguments
    online_task_id, online_method = validate_online_arguments(script_args, fed_args)
    formal_schedule = load_frozen_schedule(
        script_args.continual_schedule_path,
        script_args.continual_schedule_sha256,
        task_id=online_task_id,
        continual_round_offset=continual_round_offset,
    )

if continual_round_offset < 0:
    raise ValueError(
        "continual_round_offset must be >= 0"
    )

if initial_lora_checkpoint in (None, ""):
    if continual_round_offset != 0:
        raise ValueError(
            "continual_round_offset requires "
            "initial_lora_checkpoint"
        )

    if getattr(
        script_args,
        "initial_lora_config_sha256",
        None,
    ) not in (None, ""):
        raise ValueError(
            "initial_lora_config_sha256 requires "
            "initial_lora_checkpoint"
        )

    if getattr(
        script_args,
        "initial_lora_adapter_sha256",
        None,
    ) not in (None, ""):
        raise ValueError(
            "initial_lora_adapter_sha256 requires "
            "initial_lora_checkpoint"
        )

else:
    if continual_round_offset <= 0 and not strict_continual_mode:
        raise ValueError(
            "continual initialization requires "
            "continual_round_offset > 0"
        )

if (
    initial_lora_checkpoint not in (None, "")
    and
    legacy_existing_lora not in (None, "")
):
    raise ValueError(
        "Use only one of initial_lora_checkpoint "
        "or existing_lora"
    )

initial_lora_identity = None

if initial_lora_checkpoint not in (
    None,
    "",
):
    ckpt = (
        Path(initial_lora_checkpoint)
        .expanduser()
        .resolve()
    )

    adapter_config_path = (
        ckpt / "adapter_config.json"
    )

    adapter_model_path = (
        ckpt / "adapter_model.safetensors"
    )

    expected_config_sha = str(
        getattr(
            script_args,
            "initial_lora_config_sha256",
            "",
        )
        or ""
    ).lower()

    expected_adapter_sha = str(
        getattr(
            script_args,
            "initial_lora_adapter_sha256",
            "",
        )
        or ""
    ).lower()

    if (
        not adapter_config_path.is_file()
        or adapter_config_path.stat().st_size == 0
    ):
        raise FileNotFoundError(
            "Missing initial LoRA config: "
            f"{adapter_config_path}"
        )

    if (
        not adapter_model_path.is_file()
        or adapter_model_path.stat().st_size == 0
    ):
        raise FileNotFoundError(
            "Missing initial LoRA weights: "
            f"{adapter_model_path}"
        )

    if (
        len(expected_config_sha) != 64
        or len(expected_adapter_sha) != 64
    ):
        raise ValueError(
            "Formal continual initialization "
            "requires both expected SHA-256 values"
        )

    actual_config_sha = _sha256_file(
        adapter_config_path
    )

    actual_adapter_sha = _sha256_file(
        adapter_model_path
    )

    if actual_config_sha != expected_config_sha:
        raise ValueError(
            "Initial LoRA config SHA mismatch: "
            f"expected={expected_config_sha} "
            f"actual={actual_config_sha}"
        )

    if actual_adapter_sha != expected_adapter_sha:
        raise ValueError(
            "Initial LoRA adapter SHA mismatch: "
            f"expected={expected_adapter_sha} "
            f"actual={actual_adapter_sha}"
        )

    adapter_cfg = json.loads(
        adapter_config_path.read_text(
            encoding="utf-8"
        )
    )

    expected_base = str(
        Path(
            script_args.model_name_or_path
        )
        .expanduser()
        .resolve()
    )

    recorded_base_raw = adapter_cfg.get(
        "base_model_name_or_path"
    )

    if recorded_base_raw:
        recorded_base = str(
            Path(recorded_base_raw)
            .expanduser()
            .resolve()
        )

        if recorded_base != expected_base:
            raise ValueError(
                "Initial LoRA base model mismatch: "
                f"expected={expected_base} "
                f"recorded={recorded_base}"
            )

    if (
        str(
            adapter_cfg.get(
                "peft_type",
                "",
            )
        ).upper()
        != "LORA"
    ):
        raise ValueError(
            "Initial adapter peft_type "
            "must be LORA"
        )

    if (
        str(
            adapter_cfg.get(
                "task_type",
                "",
            )
        ).upper()
        != "CAUSAL_LM"
    ):
        raise ValueError(
            "Initial adapter task_type "
            "must be CAUSAL_LM"
        )

    if (
        int(adapter_cfg.get("r"))
        != int(script_args.peft_lora_r)
    ):
        raise ValueError(
            "Initial adapter LoRA rank "
            "does not match --peft_lora_r"
        )

    if (
        int(
            adapter_cfg.get(
                "lora_alpha"
            )
        )
        != int(
            script_args.peft_lora_alpha
        )
    ):
        raise ValueError(
            "Initial adapter LoRA alpha "
            "does not match --peft_lora_alpha"
        )

    if (
        abs(
            float(
                adapter_cfg.get(
                    "lora_dropout",
                    0.0,
                )
            )
            - 0.05
        )
        > 1e-12
    ):
        raise ValueError(
            "Initial adapter lora_dropout "
            "must be 0.05 for this frozen recipe"
        )

    if (
        str(
            adapter_cfg.get(
                "bias",
                "",
            )
        ).lower()
        != "none"
    ):
        raise ValueError(
            "Initial adapter bias must be 'none'"
        )

    if bool(
        adapter_cfg.get(
            "fan_in_fan_out",
            False,
        )
    ):
        raise ValueError(
            "Initial adapter fan_in_fan_out "
            "must be false"
        )

    if (
        adapter_cfg.get(
            "modules_to_save"
        )
        is not None
    ):
        raise ValueError(
            "Initial adapter modules_to_save "
            "must be null"
        )

    if bool(
        adapter_cfg.get(
            "use_rslora",
            False,
        )
    ):
        raise ValueError(
            "Initial adapter use_rslora "
            "must be false"
        )

    if bool(
        adapter_cfg.get(
            "use_dora",
            False,
        )
    ):
        raise ValueError(
            "Initial adapter use_dora "
            "must be false"
        )

    if (
        _canonical_target_modules(
            adapter_cfg.get(
                "target_modules"
            )
        )
        != (
            "q_proj",
            "v_proj",
        )
    ):
        raise ValueError(
            "Unexpected initial adapter "
            "target_modules="
            f"{adapter_cfg.get('target_modules')!r}"
        )

    script_args.initial_lora_checkpoint = (
        str(ckpt)
    )

    initial_lora_checkpoint = str(ckpt)

    initial_lora_identity = {
        "mode":
            "trainable_peft_adapter",
        "checkpoint":
            str(ckpt),
        "adapter_config_sha256":
            actual_config_sha,
        "adapter_model_sha256":
            actual_adapter_sha,
        "base_model_name_or_path":
            str(
                script_args.model_name_or_path
            ),
        "continual_round_offset":
            continual_round_offset,
    }

    print(
        "CONTINUAL_INITIAL_LORA="
        f"{ckpt}"
    )

    print(
        "CONTINUAL_INITIAL_CONFIG_SHA256="
        f"{actual_config_sha}"
    )

    print(
        "CONTINUAL_INITIAL_ADAPTER_SHA256="
        f"{actual_adapter_sha}"
    )

    print(
        "CONTINUAL_ROUND_OFFSET="
        f"{continual_round_offset}"
    )

    print(
        "CONTINUAL_INITIAL_LORA_IDENTITY=PASS"
    )

# ===== Load and validate the deterministic data manifest =====
data_manifest = load_training_manifest(
    script_args.data_manifest_path,
    script_args.data_manifest_sha256,
)
if (formal_continual_mode or online_probe_mode) and data_manifest is None:
    raise ValueError("continual mode requires a hashed data manifest")
if data_manifest is not None:
    if (
        (formal_continual_mode or online_probe_mode)
        and int(data_manifest.get("task_id", 0)) != int(script_args.formal_task_id)
    ):
        raise ValueError("task_id does not match data manifest task_id")
    validate_manifest_training_arguments(
        data_manifest,
        script_args,
        fed_args,
    )
    validate_manifest_training_provenance(
        data_manifest,
        Path(__file__).resolve().parent,
        allow_experiment_copy=online_probe_mode,
    )
    if formal_continual_mode:
        formal_configuration = load_formal_configuration_freeze(
            script_args.formal_configuration_freeze_path,
            script_args.formal_configuration_freeze_sha256,
            code_commit=data_manifest["code_commit"],
            task_id=int(script_args.formal_task_id),
            data_manifest_sha256=script_args.data_manifest_sha256,
            schedule_sha256=script_args.continual_schedule_sha256,
            environment_lock_sha256=script_args.formal_environment_lock_sha256,
            fixed_task1_probe_sha256=(
                None
                if formal_method == "pure"
                else script_args.prefilter_classifier_sha256
            ),
        )
        parent_provenance = load_and_validate_parent_provenance(
            initial_lora_checkpoint,
            script_args.formal_parent_provenance_sha256,
            task_id=int(script_args.formal_task_id),
            method=formal_method,
            code_commit=data_manifest["code_commit"],
            adapter_config_sha256=initial_lora_identity["adapter_config_sha256"],
            adapter_model_sha256=initial_lora_identity["adapter_model_sha256"],
            stage_base_tensor_sha256=script_args.formal_expected_stage_base_tensor_sha256,
            fixed_probe_sha256=(
                None
                if formal_method == "pure"
                else script_args.prefilter_classifier_sha256
            ),
            environment_lock_sha256=script_args.formal_environment_lock_sha256,
            configuration_freeze_sha256=script_args.formal_configuration_freeze_sha256,
            python_executable=formal_environment_identity[
                "locked_python_executable"
            ],
            shadow_learning_rate=script_args.shadow_learning_rate,
        )

# ===== Initialize classifier singleton if prefilter is enabled =====
enable = getattr(script_args, 'prefilter_enable', False)
delta_collection_enable = getattr(
    script_args, 'delta_collection_enable', False
)

probe_guard = None
if not enable:
    print("[prefilter] LoRA classifier disabled.")
else:
    try:
        clf = FedLoRAClassifier.instance(
            getattr(script_args, 'prefilter_classifier_path', None)
        )
        print("[prefilter] LoRA classifier enabled (mode=delta).")

        norm = str(getattr(clf, 'normalize', 'none')).lower().replace('-', '_')
        script_args.normalization_method = ('l2' if norm == 'l2' else 'none')
        print(f"[prefilter] Recorded classifier normalization: {script_args.normalization_method}")

        print(f"[prefilter] Recorded prefilter threshold: {script_args.prefilter_threshold}")

        if formal_continual_mode:
            probe_guard = FrozenProbeGuard(
                script_args.prefilter_classifier_path,
                script_args.prefilter_classifier_sha256,
                threshold=script_args.prefilter_threshold,
            )
        elif online_probe_mode:
            probe_guard = OnlineProbeGuard(
                script_args.prefilter_classifier_path,
                script_args.prefilter_classifier_sha256,
                threshold=script_args.prefilter_threshold,
                task_id=int(script_args.formal_task_id),
            )

    except Exception as e:
        raise RuntimeError(
            "Prefilter classifier initialization failed"
        ) from e

# ===== Load the dataset =====
if data_manifest is None:
    dataset_list, num_client_list = get_sft_datasets(script_args, fed_args)
else:
    dataset_list, num_client_list = rebuild_manifest_federated_pools(
        data_manifest,
        script_args.local_data_dir,
    )
print(dataset_list, num_client_list)

# ===== Split the dataset into clients =====
local_datasets = []
num_clients = sum(num_client_list)
for dataset, num_client in zip(dataset_list, num_client_list):
    splited_datasets = split_dataset(fed_args, script_args, dataset, num_client)
    local_datasets.extend(splited_datasets)
    
fed_args.num_clients = num_clients
save_config(script_args, fed_args)

if probe_guard is not None:
    probe_guard.write_provenance(
        Path(script_args.output_dir) / (
            "fixed_probe_identity.json"
            if formal_continual_mode
            else "online_probe_identity.json"
        )
    )

if initial_lora_identity is not None:
    initial_lora_identity.update({
        "stage_num_rounds":
            int(fed_args.num_rounds),
        "display_round_start":
            continual_round_offset + 1,
        "display_round_end":
            continual_round_offset
            + int(fed_args.num_rounds),
        "data_manifest_path":
            script_args.data_manifest_path,
        "data_manifest_sha256":
            script_args.data_manifest_sha256,
    })
    if formal_continual_mode:
        initial_lora_identity.update({
            "main_learning_rate_end": MAIN_LEARNING_RATE_END,
            "main_learning_rate_schedule": MAIN_LEARNING_RATE_SCHEDULE,
            "main_learning_rate_start": MAIN_LEARNING_RATE_START,
            "shadow_base_source_global_round": continual_round_offset,
            "shadow_detection_active": formal_method == "shadow",
            "shadow_detection_contract": SHADOW_DETECTION_CONTRACT,
            "shadow_learning_rate": FORMAL_SHADOW_LEARNING_RATE,
            "environment_lock_path": formal_environment_identity[
                "environment_lock_path"
            ],
            "environment_lock_sha256": formal_environment_identity[
                "environment_lock_sha256"
            ],
            "python_executable": formal_environment_identity[
                "locked_python_executable"
            ],
        })

    provenance_path = (
        Path(script_args.output_dir)
        / "continual_init_provenance.json"
    )

    provenance_path.write_text(
        json.dumps(
            initial_lora_identity,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    provenance_sha_path = (
        provenance_path.with_suffix(
            provenance_path.suffix
            + ".sha256"
        )
    )

    provenance_sha_path.write_text(
        f"{_sha256_file(provenance_path)}  "
        f"{provenance_path.name}\n",
        encoding="utf-8",
    )

    print(
        "CONTINUAL_INIT_PROVENANCE="
        f"{provenance_path}"
    )

print(script_args, fed_args)


# ===== Get model config =====
device_map, quantization_config, _ = get_model_config(script_args)
load_kwargs = dict(
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=True,
)

try:
    cfg = AutoConfig.from_pretrained(script_args.model_name_or_path)
    if hasattr(cfg, "attn_implementation"):
        load_kwargs["attn_implementation"] = "flash_attention_2"
except Exception:
    pass

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    **load_kwargs,
)

if hasattr(model.config, "attn_implementation"):
    model.config.attn_implementation = "flash_attention_2"

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=script_args.gradient_checkpointing
            )

if initial_lora_checkpoint not in (None, ""):
    model = PeftModel.from_pretrained(
        model,
        initial_lora_checkpoint,
        is_trainable=True,
    )

elif script_args.existing_lora is not None:
    model = PeftModel.from_pretrained(
        model,
        script_args.existing_lora
        + '/checkpoint-100',
        is_trainable=True,
    )

else:
    # Set fixed random seed for LoRA parameter initialization
    set_lora_init_seed(script_args.seed)
    model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)
shadow_lora_base = copy.deepcopy(global_dict)
shadow_base_guard = None
if online_probe_mode:
    online_stage_base_sha256 = canonical_lora_tensor_sha256(shadow_lora_base)
    supplied_stage_base_sha256 = str(
        getattr(script_args, "formal_expected_stage_base_tensor_sha256", "")
        or ""
    ).lower()
    if supplied_stage_base_sha256 and supplied_stage_base_sha256 != online_stage_base_sha256:
        raise ValueError("online stage base tensor SHA-256 mismatch")
    script_args.formal_expected_stage_base_tensor_sha256 = online_stage_base_sha256
if strict_continual_mode:
    shadow_base_guard = FrozenLoraStateGuard(
        shadow_lora_base,
        script_args.formal_expected_stage_base_tensor_sha256,
    )
    shadow_base_guard.write_provenance(
        Path(script_args.output_dir) / "stage_shadow_base.json",
        task_id=int(script_args.formal_task_id),
        source_round=continual_round_offset,
        formal_method=(formal_method if formal_continual_mode else online_method),
        shadow_learning_rate=script_args.shadow_learning_rate,
    )

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
token_contract_report = configure_tokenizer(tokenizer)
configure_model(model)
print("STRICT_TOKEN_CONTRACT=" + json.dumps(token_contract_report, sort_keys=True))

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)

from federated_learning.module_gate import ModuleGate
if not online_probe_mode or online_method != 'shadow':
    raise ValueError('This five-theme ModuleGate experiment requires online Shadow')
module_gate = ModuleGate(Path(__file__).resolve().parents[1], online_task_id,
                         initial_lora_checkpoint, global_dict, script_args.output_dir)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
prefilter_strategy = PrefilterStrategy(script_args, fed_args)
strategy_name = str(getattr(script_args, 'prefilter_strategy', 'step-level')).lower()

for stage_round in tqdm(range(fed_args.num_rounds)):
    display_round = (
        continual_round_offset
        + stage_round
        + 1
    )
    frozen_round = schedule_round(formal_schedule, stage_round) if formal_schedule else None
    clients_this_round = (
        list(frozen_round["clients"])
        if frozen_round is not None
        else get_clients_this_round(fed_args, stage_round)
    )
    client_actual_samples = {} 
    if shadow_base_guard is not None:
        shadow_base_guard.assert_unchanged(
            shadow_lora_base,
            context="before global round %d" % display_round,
        )
    if probe_guard is not None:
        probe_guard.assert_unchanged(context="before global round %d" % display_round)

    print(
        f">> ==================== "
        f"Round {display_round} "
        f"(stage {stage_round+1}/"
        f"{fed_args.num_rounds}) : "
        f"{clients_this_round} "
        f"===================="
    )
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
        if frozen_round is not None:
            if len(local_datasets[client]) != 500:
                raise ValueError("formal client pool must contain exactly 500 rows")
            sub_dataset = local_datasets[client].select(frozen_round["sample_positions"])
        else:
            sub_dataset = get_dataset_this_round(local_datasets[client], stage_round, fed_args, script_args)

        completion = build_validated_federated_completion(
            tokenizer=tokenizer,
            formatting_prompts_func=formatting_prompts_func,
            response_template=response_template,
            dataset=sub_dataset,
            max_seq_length=script_args.seq_length,
            require_sample_identity=strict_continual_mode,
        )
        training_sub_dataset = completion["training_dataset"]
        round_data_collator = completion["base_collator"]
        client_actual_samples[client] = len(sub_dataset)  # Record the actual number of training samples
        
        # ===== Train local model on the client side =====
        new_lr = (
            float(frozen_round["learning_rate"])
            if frozen_round is not None
            else cosine_learning_rate(stage_round, fed_args.num_rounds, script_args.learning_rate, 1e-6)
        )
        if frozen_round is not None and new_lr != frozen_cosine_lr(stage_round):
            raise RuntimeError("runtime learning rate differs from frozen schedule")
        training_args = get_training_args(script_args, new_lr)
        prefilter_round = getattr(script_args, 'prefilter_round', None)
        classifier_tracking_active = (
            enable
            and strategy_name not in ('shadow-level', 'evidence-level')
            and (
                prefilter_round is None
                or stage_round < int(prefilter_round)
            )
        )

        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=training_sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=round_data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
            current_round=stage_round,
            tracker_initial_state=None,
            tracker_enabled=(
                delta_collection_enable
                or classifier_tracking_active
            ),
            client_id=client,
            audit_phase="global",
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
        else:
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

        # SFTTrainer owns optimizer state and callback tensors that are no
        # longer needed after this client's update has been copied.  Releasing
        # them before constructing the independent Shadow trainer prevents
        # allocator growth across hundreds of sequential trainer instances.
        trainer = None
        results = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ===== Shadow branch: conditionally run with fixed initial LoRA for detection =====
        use_shadow = enable and prefilter_strategy.use_shadow(client, stage_round)
        if use_shadow:
            shadow_trainer = None
            try:
                if shadow_base_guard is not None:
                    shadow_base_guard.assert_unchanged(
                        shadow_lora_base,
                        context="before shadow client %d round %d" % (client, display_round),
                    )
                set_peft_model_state_dict(model, shadow_lora_base)
                shadow_lr = validate_formal_shadow_learning_rate(
                    script_args.shadow_learning_rate
                )
                shadow_training_args = get_training_args(script_args, shadow_lr)
                shadow_trainer = get_fed_local_sft_trainer(
                    model=model,
                    tokenizer=tokenizer,
                    training_args=shadow_training_args,
                    local_dataset=training_sub_dataset,
                    formatting_prompts_func=formatting_prompts_func,
                    data_collator=round_data_collator,
                    global_dict=shadow_lora_base,
                    fed_args=fed_args,
                    script_args=script_args,
                    local_auxiliary=auxiliary_model_list[client],
                    global_auxiliary=global_auxiliary,
                    current_round=stage_round,
                    tracker_initial_state=shadow_lora_base,
                    tracker_enabled=True,
                    client_id=client,
                    audit_phase="shadow",
                )
                _ = shadow_trainer.train()
                if shadow_base_guard is not None:
                    shadow_base_guard.assert_unchanged(
                        shadow_lora_base,
                        context="after shadow client %d round %d" % (client, display_round),
                    )
            except Exception as se:
                raise RuntimeError(
                    f"Shadow-level training failed for client {client} "
                    f"in round {display_round}"
                ) from se
            finally:
                shadow_trainer = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ===== Prefilter: Detect harmful samples and adjust weights based on malicious ratio =====
    client_harmful_mapping = {}
    eval_result = {}
    
    if enable:
        need_classifier = any(prefilter_strategy.use_shadow(c, stage_round) for c in clients_this_round)
        if not need_classifier and strategy_name in ('step-level', 'client-level'):
            pr = getattr(script_args, 'prefilter_round', None)
            need_classifier = (pr is None) or (int(stage_round) < int(pr))
        if need_classifier:
            params_dir = os.path.join(script_args.output_dir, 'fed_lora_params', f'round_{stage_round+1}')
            try:
                clf_local = FedLoRAClassifier.instance()
                client_harmful_mapping, eval_result = clf_local.get_harmful_mapping(
                    params_dir=params_dir,
                    clients_this_round=clients_this_round,
                    round_num=stage_round,
                    fed_args=fed_args,
                    script_args=script_args
                )

                try:
                    eval_engine = Evaluation(FedLoRAClassifier.instance())
                    _, current_precision = eval_engine.evaluate(
                        params_dir=params_dir,
                        output_dir=script_args.output_dir,
                        round_num=stage_round,
                        mode='prefilter',
                        print_stats=False,
                        existing_result=eval_result,
                        threshold=getattr(script_args, 'prefilter_threshold', None)
                    )
                    print(f">> Round {display_round} Classifier Precision: {current_precision*100:.2f}% ")
                except Exception as e:
                    raise RuntimeError(
                        "Failed to save prefilter classification results "
                        f"for round {display_round}"
                    ) from e
                # Delete LoRA params after evaluation
                if os.path.exists(params_dir) and not strict_continual_mode:
                    shutil.rmtree(params_dir)
            except Exception as e:
                raise RuntimeError(
                    f"Classifier evaluation failed for round {display_round}"
                ) from e

    filtered_clients_this_round, client_effective_samples, skip_round = prefilter_strategy.compute(
        stage_round,
        clients_this_round,
        client_actual_samples,
        client_harmful_mapping,
        eval_result
    )

    for c in clients_this_round:
        base_samples = client_actual_samples.get(c, 0)
        eff = float(client_effective_samples.get(c, 0))
        w = (eff / float(base_samples)) if base_samples > 0 else 0.0

        status, frozen_w = prefilter_strategy.get_frozen_info(c)
        if status == 'malicious':
            print(f"[prefilter] Client {c} frozen MALICIOUS | weight={w:.3f}")
        elif status == 'benign':
            print(f"[prefilter] Client {c} frozen BENIGN | weight={w:.3f}")
        else:
            harm_cnt = len(client_harmful_mapping.get(c, []))
            print(f"[prefilter] Client {c} has {harm_cnt} harmful steps | weight={w:.3f}")

    # =====  Server-side aggregation and checkpoint logic =====
    save_steps = 10 if formal_continual_mode else (10 if (
        initial_lora_checkpoint not in (None, "")
        or script_args.existing_lora is not None
    ) else 50)

    def _save_checkpoint(set_state=False):
        if not (
            ((stage_round + 1) % save_steps == 0)
            or (
                (stage_round + 1)
                == int(
                    getattr(
                        fed_args,
                        'num_rounds',
                        0,
                    )
                )
            )
            or ((stage_round + 1) == 10)
        ):
            return
        if set_state:
            set_peft_model_state_dict(model, global_dict)
        ckpt_dir = os.path.join(
            script_args.output_dir,
            f"checkpoint-{display_round}",
        )
        os.makedirs(ckpt_dir, exist_ok=False)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        if formal_continual_mode:
            adapter_config_sha256 = canonicalize_formal_adapter_config(
                os.path.join(ckpt_dir, "adapter_config.json")
            )
            checkpoint_provenance = build_checkpoint_provenance(
                code_commit=data_manifest["code_commit"],
                task_id=int(script_args.formal_task_id),
                formal_method=formal_method,
                global_round=display_round,
                adapter_config_sha256=adapter_config_sha256,
                adapter_model_sha256=_sha256_file(
                    os.path.join(ckpt_dir, "adapter_model.safetensors")
                ),
                endpoint_lora_tensor_sha256=canonical_lora_tensor_sha256(global_dict),
                stage_base_tensor_sha256=shadow_base_guard.expected_sha256,
                parent_provenance_sha256=parent_provenance["sha256"],
                data_manifest_sha256=str(script_args.data_manifest_sha256).lower(),
                schedule_sha256=str(script_args.continual_schedule_sha256).lower(),
                fixed_probe_sha256=(
                    None
                    if formal_method == "pure"
                    else str(script_args.prefilter_classifier_sha256).lower()
                ),
                shadow_learning_rate=script_args.shadow_learning_rate,
                environment_lock_sha256=script_args.formal_environment_lock_sha256,
                configuration_freeze_sha256=script_args.formal_configuration_freeze_sha256,
                python_executable=formal_environment_identity[
                    "locked_python_executable"
                ],
            )
            write_checkpoint_provenance(ckpt_dir, checkpoint_provenance)
        elif online_probe_mode:
            online_checkpoint_provenance = {
                "schema": "safe-fcl.continual-online-checkpoint.v1",
                "experiment_lock_sha256": _sha256_file(str(Path(__file__).resolve().parents[1] / "provenance" / "EXPERIMENT_LOCK.json")),
                "task_id": int(script_args.formal_task_id),
                "stage_id": int(script_args.formal_task_id),
                "original_task_id": (3,1,4,2,5)[int(script_args.formal_task_id)-1],
                "order_experiment_id": 'risk5_order_3_1_4_2_ood5_20261009',
                "order_method_key": 'mlp_shadow_gate',
                "method": online_method,
                "global_round": int(display_round),
                "adapter_config_sha256": _sha256_file(
                    os.path.join(ckpt_dir, "adapter_config.json")
                ),
                "adapter_model_sha256": _sha256_file(
                    os.path.join(ckpt_dir, "adapter_model.safetensors")
                ),
                "parent_adapter_sha256": initial_lora_identity[
                    "adapter_model_sha256"
                ],
                "data_manifest_sha256": str(script_args.data_manifest_sha256).lower(),
                "schedule_sha256": str(script_args.continual_schedule_sha256).lower(),
                "online_probe_sha256": (None if online_method == "pure" else str(script_args.prefilter_classifier_sha256).lower()),
                "probe_training_task_id": 4,
                "probe_weights_frozen": True,
                "task5_ood": True,
                "stage_base_tensor_sha256": shadow_base_guard.expected_sha256,
                "endpoint_lora_tensor_sha256": canonical_lora_tensor_sha256(global_dict),
                "module_gate": module_gate.save_checkpoint(ckpt_dir, global_dict, display_round),
            }
            with open(
                os.path.join(ckpt_dir, "ONLINE_CHECKPOINT_PROVENANCE.json"),
                "x",
                encoding="utf-8",
            ) as handle:
                json.dump(online_checkpoint_provenance, handle, indent=2, sort_keys=True)
                handle.write("\n")

    if skip_round:
        module_gate.skip(display_round)
        print(f">> Round {display_round} skipped. Global model unchanged.")
        try:
            if shadow_base_guard is not None:
                shadow_base_guard.assert_unchanged(
                    shadow_lora_base,
                    context="after skipped global round %d" % display_round,
                )
            if probe_guard is not None:
                probe_guard.assert_unchanged(
                    context="after skipped global round %d" % display_round
                )
            _save_checkpoint(set_state=True)
        except Exception as se:
            raise RuntimeError(
                f"Final checkpoint save failed for round {display_round}"
            ) from se
        continue

    module_gate_previous = copy.deepcopy(global_dict)
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, client_effective_samples,
        filtered_clients_this_round, stage_round, proxy_dict=proxy_dict,
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    global_dict = module_gate.aggregate(module_gate_previous, global_dict, local_dict_list,
        filtered_clients_this_round, client_effective_samples, client_actual_samples, new_lr, display_round)
    set_peft_model_state_dict(model, global_dict)   # Update global model
    if shadow_base_guard is not None:
        shadow_base_guard.assert_unchanged(
            shadow_lora_base,
            context="after global aggregation round %d" % display_round,
        )
    if probe_guard is not None:
        probe_guard.assert_unchanged(
            context="after global aggregation round %d" % display_round
        )

    _save_checkpoint()

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

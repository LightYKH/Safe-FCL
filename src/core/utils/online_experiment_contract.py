"""Frozen Step/Client/Shadow Task2-3 repair from audited fixed R100 parents."""
import json
import re
from pathlib import Path
from .continual_data_manifest import TASK_DATASETS
from .data_manifest import sha256_file
from .formal_method_chain import adapter_tensor_sha256


ROOT = Path(__file__).resolve().parents[2]
FIXED_P1_SHA256 = "c5de221242a81279009622db395048ad66571c2c3d8ca0a2eb8bb249866f4f8f"
STRATEGIES = {"step": "step-level", "client": "client-level", "shadow": "shadow-level"}


def _object(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required provenance must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required provenance must contain an object: {path}")
    return value


def _hashed_file(path, expected):
    path = Path(path)
    if re.fullmatch(r"[0-9a-f]{64}", str(expected or "")) is None:
        raise ValueError(f"required SHA256 is invalid: {path}")
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"required file is absent or its SHA256 differs: {path}")
    return expected


def _same_path(actual, expected, description):
    actual = Path(actual or "")
    if actual.is_symlink() or actual.resolve() != Path(expected).resolve():
        raise ValueError(description)
    return actual.resolve()


def _validate_parent(args, task, method):
    cfg = _object(ROOT / 'inputs/stage.json')
    parent = _same_path(args.initial_lora_checkpoint, cfg['parent_checkpoint'], 'Wrong branch or stage parent')
    if cfg['task'] != task or cfg['method'] != method:
        raise ValueError('Wrong stage or method')
    for name, digest in cfg['parent_files'].items():
        _hashed_file(parent / name, digest)
    if args.initial_lora_adapter_sha256 != cfg['parent_files']['adapter_model.safetensors']:
        raise ValueError('Wrong parent adapter bytes')
    if args.initial_lora_config_sha256 != cfg['parent_files']['adapter_config.json']:
        raise ValueError('Wrong parent config')
    if adapter_tensor_sha256(parent / 'adapter_model.safetensors') != cfg['parent_tensor_sha256']:
        raise ValueError('Wrong parent adapter tensor identity')
    if task > 1:
        name = 'ONLINE_CHECKPOINT_PROVENANCE.json' if (parent / 'ONLINE_CHECKPOINT_PROVENANCE.json').exists() else 'FORMAL_CHECKPOINT_PROVENANCE.json'
        identity = _object(parent / name)
        if (identity.get('task_id'), identity.get('global_round'), identity.get('method')) != (task-1,(task-1)*100,method):
            raise ValueError('Wrong parent task, round or method')
        if identity['adapter_model_sha256'] != args.initial_lora_adapter_sha256:
            raise ValueError('Parent provenance does not bind adapter')
    if cfg['gate'] and task > 1:
        _hashed_file(parent / 'MODULE_GATE_STATE.json', cfg['parent_files']['MODULE_GATE_STATE.json'])


def validate_online_arguments(args, fed):
    task = int(args.formal_task_id)
    method = str(args.formal_method).lower()
    if task not in (1,2,3,4,5) or method not in ('client','shadow'):
        raise ValueError('Task4 continuation requires Task4 and paper Client')
    expected = dict(seq_length=512, batch_size=4, gradient_accumulation_steps=4,
                    max_steps=10, peft_lora_r=8, peft_lora_alpha=16, seed=2023,
                    learning_rate=5e-5, shadow_learning_rate=5e-5,
                    prefilter_threshold=0.8, time_decay_factor=0.95,
                    prefilter_min_weight=0.0, prefilter_skip_avg_weight=0.2,
                    prefilter_round=100 if method == 'shadow' else 20,
                    optimizer='adamw_bnb_8bit', template='alpaca',
                    prefilter_log_mode='json', prefilter_id=False, use_peft=True,
                    gradient_checkpointing=True, load_in_8bit=True, load_in_4bit=False,
                    delta_collection_enable=False, continual_round_offset=(task-1)*100,
                    prefilter_strategy=STRATEGIES[method], prefilter_enable=True,
                    online_probe_mode=True, formal_continual_mode=False)
    for name, value in expected.items():
        if getattr(args, name, None) != value:
            raise ValueError(f'online contract requires {name}={value!r}')
    expected_fed = dict(num_rounds=100, sample_clients=3, benign_num_clients=[7],
                        malicious_num_clients=[3], num_data_per_client=500, fed_alg='fedavg',
                        benign_dataset_names=[TASK_DATASETS[task]['benign']],
                        malicious_dataset_names=[TASK_DATASETS[task]['malicious']])
    for name, value in expected_fed.items():
        if getattr(fed, name, None) != value:
            raise ValueError(f'online contract requires {name}={value!r}')
    expected_output = ROOT / 'training' / 'online_probe' / (method + '_level') / f'task{task}_r{(task-1)*100+1:03d}_r{task*100:03d}'
    _same_path(args.output_dir, expected_output, 'repair output must use the new root same-method task directory')
    for supplied_path, expected_path, supplied_sha in (
        (args.prefilter_classifier_path, ROOT / f'task{task}' / 'probe_model.pt', args.prefilter_classifier_sha256),
        (args.data_manifest_path, ROOT / 'manifests' / f'task{task}.data_manifest.json', args.data_manifest_sha256),
        (args.continual_schedule_path, ROOT / 'schedules' / f'task{task}.schedule.json', args.continual_schedule_sha256),
    ):
        _same_path(supplied_path, expected_path, 'repair inputs must use the frozen files inside the new root')
        _hashed_file(expected_path, supplied_sha)
    _validate_parent(args, task, method)
    return task, method

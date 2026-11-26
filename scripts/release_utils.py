"""Shared filesystem and integrity helpers for the public release."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAFE_FCL_METHOD = "safe_fcl"
_LEGACY_METHOD_SHA256 = "8e60df8c047d6585a8529001b77a2cb3c9ceb1d6bfefb8652d73926b7bf95072"


def normalize_method(method: str) -> str:
    """Resolve the current CLI method ID and the one used by earlier runs."""
    if method == SAFE_FCL_METHOD:
        return SAFE_FCL_METHOD
    if isinstance(method, str) and method.isascii():
        if hashlib.sha256(method.encode("ascii")).hexdigest() == _LEGACY_METHOD_SHA256:
            return SAFE_FCL_METHOD
    raise ValueError(f"Unknown method: {method}")


def method_config_path(method: str, group: str = "methods") -> Path:
    path = ROOT / "configs" / group / f"{normalize_method(method)}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def offline():
    for key in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','DATASETS_OFFLINE']:
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'

def assert_profile(args):
    for key, expected in dict(batch_size=4,gradient_accumulation_steps=4,seq_length=512,max_steps=10).items():
        if type(args.get(key)) is not int or args[key] != expected:
            raise ValueError(f'{key} must be {expected}; got {args.get(key)!r}')
    if args.get('online_probe_mode') is not True or args.get('formal_continual_mode') is not False:
        raise ValueError('Only the released continual online mode is supported')

def freeze(stage):
    paths=[]
    for folder in ['code','inputs','manifests','schedules','scripts']:
        paths.extend(p for p in (stage/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    cfg=load(stage/'inputs/stage.json')
    paths.extend(Path(cfg['parent_checkpoint'])/n for n in cfg['parent_files'])
    for p in stage.glob('task*/probe_model.pt'):paths.append(p)
    if (stage/'provenance/REPAIR_CONTRACT.json').exists(): paths.append(stage/'provenance/REPAIR_CONTRACT.json')
    write(stage/'provenance/EXPERIMENT_LOCK.json',dict(files={str(p.resolve()):sha(p) for p in paths}))

def verify(stage):
    for filename,digest in load(stage/'provenance/EXPERIMENT_LOCK.json')['files'].items():
        if sha(filename)!=digest:raise ValueError('Frozen input changed: '+filename)

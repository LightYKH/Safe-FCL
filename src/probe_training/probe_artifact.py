"""Initialize a fresh Probe1 or continue this branch's exact preceding Probe."""
from pathlib import Path
import hashlib
import torch


def initialize_probe(model, cfg, feature_metadata):
    task = int(cfg['task'])
    previous = cfg.get('previous_probe')
    if task == 1:
        if previous is not None or cfg.get('previous_probe_sha256') is not None:
            raise ValueError('Probe1 must be randomly initialized')
        return dict(feature_metadata, architecture='mlp1', normalization='l2',
                    task_id=1, seed=2023, early_stopping=True,
                    orthogonal_projection=True, compression=False,
                    probe_variant='mlp1_raw_orthogonal_shadow_modulegate_ten_round')
    if not previous:
        raise ValueError('Continual probe requires its preceding checkpoint')
    path = Path(previous)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != cfg['previous_probe_sha256']:
        raise ValueError('Preceding probe hash differs')
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if artifact['task_id'] != task - 1 or artifact['architecture'] != 'mlp1':
        raise ValueError('Preceding probe task or architecture differs')
    for name in ['ordered_keys', 'key_sizes', 'feature_dim', 'filter_policy']:
        if artifact[name] != feature_metadata[name]:
            raise ValueError('Preceding probe feature schema differs: ' + name)
    model.load_state_dict(artifact['model_state_dict'], strict=True)
    return artifact

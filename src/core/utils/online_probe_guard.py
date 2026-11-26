"""Fail-closed identity guard for a task-specific continual Probe."""

import hashlib
import json
import re
from pathlib import Path

import torch
from lora_classifier.neural_probe_models import make_model


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OnlineProbeGuard:
    def __init__(self, path: str, expected_sha256: str, *, threshold: float, task_id: int):
        unresolved = Path(path).expanduser()
        if unresolved.is_symlink():
            raise ValueError("online Probe path must not be a symlink")
        self.path = unresolved.resolve()
        if not self.path.is_file():
            raise FileNotFoundError("online Probe file does not exist")
        self.expected_sha256 = str(expected_sha256).lower()
        if re.fullmatch(r"[0-9a-f]{64}", self.expected_sha256) is None:
            raise ValueError("online Probe expected SHA-256 is required")
        self.assert_unchanged(context="initialization")
        try:
            artifact = torch.load(str(self.path), map_location="cpu", weights_only=True)
        except TypeError:
            artifact = torch.load(str(self.path), map_location="cpu")
        if not isinstance(artifact, dict):
            raise ValueError("online Probe artifact must be a dictionary")
        required = {
            "model_state_dict", "feature_dim", "normalization", "threshold",
            "task_id", "seed", "ordered_keys", "key_sizes", "filter_policy",
        }
        missing = sorted(required.difference(artifact))
        if missing:
            raise ValueError("online Probe missing fields: " + ", ".join(missing))
        if int(artifact["task_id"]) != int(task_id):
            raise ValueError("online Probe task_id does not match training task")
        if int(artifact["seed"]) != 2023:
            raise ValueError("online Probe seed must be 2023")
        if abs(float(artifact["threshold"]) - float(threshold)) > 1e-15:
            raise ValueError("online Probe threshold mismatch")
        if str(artifact["normalization"]).lower().replace("-", "_") != "l2":
            raise ValueError("online Probe normalization must be l2")
        if artifact["filter_policy"] != "first_B":
            raise ValueError("online Probe filter_policy must be first_B")
        ordered_keys = artifact["ordered_keys"]
        key_sizes = artifact["key_sizes"]
        if not isinstance(ordered_keys, list) or not ordered_keys:
            raise ValueError("online Probe ordered_keys are missing")
        if not isinstance(key_sizes, dict) or set(key_sizes) != set(ordered_keys):
            raise ValueError("online Probe key_sizes do not match ordered_keys")
        feature_dim = int(artifact["feature_dim"])
        if feature_dim != sum(int(key_sizes[key]) for key in ordered_keys):
            raise ValueError("online Probe feature_dim is inconsistent")
        state = artifact['model_state_dict']
        architecture = artifact.get('architecture')
        if architecture not in ('mlp1', 'mlp2') or feature_dim != 40960:
            raise ValueError('This experiment requires an uncompressed one/two hidden-layer MLP')
        root = Path(__file__).resolve().parents[2]
        contract = json.loads((root/'provenance/REPAIR_CONTRACT.json').read_text())
        recipe=artifact.get('probe_training_recipe')
        if not isinstance(recipe,dict) or recipe != contract['probe_training']:
            raise ValueError('Aligned MLP fitting recipe differs from frozen experiment')
        if recipe.get('positive_class_weight') is not True or recipe.get('extra_current_resampling') is not False:
            raise ValueError('MLP must be class weighted without extra resampling')
        if architecture != contract['probe_architecture']:
            raise ValueError('Probe architecture differs from the experiment contract')
        with torch.random.fork_rng(devices=[]):
            model = make_model(architecture)
        model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Probe has nonfinite parameters')
        if artifact.get('early_stopping') is not True or artifact.get('orthogonal_projection') is not True:
            raise ValueError('Expected an early-stopped orthogonal probe')
        if artifact.get('source_checkpoint_sha256') != contract['probe_checkpoints'][str(task_id)]['sha256']:
            raise ValueError('Unexpected source checkpoint')
        self.identity = {
            "schema": "safe-fcl.continual-online-probe-identity.v1",
            "probe_path": str(self.path),
            "probe_sha256": self.expected_sha256,
            "task_id": int(task_id),
            "feature_dim": feature_dim,
            "normalization": "l2",
            "threshold": float(threshold),
            "seed": 2023,
            "architecture": architecture,
            "source_checkpoint_sha256": artifact["source_checkpoint_sha256"],
        }

    def assert_unchanged(self, *, context: str) -> None:
        actual = _sha256_file(self.path)
        if actual != self.expected_sha256:
            raise RuntimeError(
                "online Probe changed at %s: expected=%s actual=%s"
                % (context, self.expected_sha256, actual)
            )

    def write_provenance(self, path) -> None:
        target = Path(path)
        if target.exists():
            raise FileExistsError("refusing to overwrite online Probe provenance")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

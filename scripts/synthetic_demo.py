"""Exercise CRA, Shadow weighting and ModuleGate with generated CPU tensors."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import nn

from release_utils import ROOT, SAFE_FCL_METHOD, load, method_config_path, offline, write


sys.path[:0] = [str(ROOT / "src/core"), str(ROOT / "src/probe_models")]
from federated_learning.fed_global import global_aggregate
from federated_learning.module_gate import ModuleGate
from lora_classifier.prefilter_strategy import PrefilterStrategy
from orthogonal import Projector, extend_bases, projected_adamw


def fit_probe(threshold, steps):
    """Fit a small probe on synthetic step features and project a later update."""
    generator = torch.Generator().manual_seed(2023)
    labels = torch.arange(16, dtype=torch.float64) % 2
    features = torch.randn(16, 4, generator=generator, dtype=torch.float64) * 0.1
    features += (2 * labels[:, None] - 1)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1)).double()
    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    for _ in range(40):
        optimizer.zero_grad()
        loss(model(features).squeeze(-1), labels).backward()
        optimizer.step()

    bases, _ = extend_bases(model, features, {}, completed_task=1)
    projector = Projector(model, bases)
    with projected_adamw(projector):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss(model(features.roll(1, dims=1)).squeeze(-1), labels).backward()
        optimizer.step()
    leakage = projector.summary()["maximum_update_leakage"]
    if projector.steps != 1 or leakage > 1e-10:
        raise RuntimeError("Synthetic CRA projection check failed")

    step_features = torch.stack((features[0].repeat(steps, 1), features[:steps]))
    with torch.no_grad():
        flagged = model(step_features).squeeze(-1).sigmoid() >= threshold
    if flagged[0].any() or not 0 < int(flagged[1].sum()) < steps:
        raise RuntimeError("Synthetic Shadow probes did not distinguish the clients")
    return {client: row.nonzero().flatten().tolist() for client, row in enumerate(flagged)}, leakage


def run_rounds(workspace, training, gate_config, flagged_steps):
    config = dict(gate_config, expected_modules=1)
    write(workspace / "inputs/module_gate_config.json", config)
    module = "demo.q_proj"
    a_key, b_key = f"{module}.lora_A.weight", f"{module}.lora_B.weight"
    state = {a_key: torch.ones(config["lora_rank"], 2), b_key: torch.zeros(2, config["lora_rank"])}
    output = workspace / "output"
    gate = ModuleGate(workspace, 1, None, state, output)
    clients = [0, 1]
    fed = SimpleNamespace(fed_alg="fedavg", num_clients=len(clients))
    prefilter = PrefilterStrategy(SimpleNamespace(**training, output_dir=str(output)), fed)
    samples = training["batch_size"] * training["gradient_accumulation_steps"] * training["max_steps"]
    actual = dict.fromkeys(clients, samples)
    last_round = config["warmup_observations"] + 1

    for round_number in range(1, last_round + 1):
        final = round_number == last_round
        selected, effective, skip = prefilter.compute(
            round_number - 1, clients, actual, flagged_steps if final else {}
        )
        if skip or selected != clients:
            raise RuntimeError("Synthetic federated round was skipped")
        deltas = (0.001, 0.1 if final else 0.001)
        local = [{a_key: state[a_key], b_key: state[b_key] + delta} for delta in deltas]
        baseline, _ = global_aggregate(fed, dict(state), local, effective, selected, round_number - 1)
        state = gate.aggregate(
            state, baseline, local, selected, effective, actual,
            training["shadow_learning_rate"], round_number,
        )

    record = load(output / "module_gate_latest.json")
    clean, risky = record["modules"][module]
    if not 0 < risky["security"] < clean["security"] or not 0 < risky["gate"] < clean["gate"]:
        raise RuntimeError("Synthetic ModuleGate did not attenuate the risky outlier")
    if torch.equal(state[b_key], baseline[b_key]):
        raise RuntimeError("Synthetic ModuleGate did not change the aggregated adapter")
    return risky["security"], risky["gate"]


def main():
    offline()
    torch.set_num_threads(1)
    torch.manual_seed(2023)
    training = load(method_config_path(SAFE_FCL_METHOD))["training"]
    gate_config = load(method_config_path(SAFE_FCL_METHOD, "gates"))
    flagged, leakage = fit_probe(training["prefilter_threshold"], training["max_steps"])
    with TemporaryDirectory(prefix="safe-fcl-demo-") as directory:
        security, retention = run_rounds(Path(directory), training, gate_config, flagged)
    print(f"Synthetic demo passed. CRA leakage={leakage:.2e}, "
          f"Shadow security={security:.4f}, ModuleGate retention={retention:.4f}.")


if __name__ == "__main__":
    main()

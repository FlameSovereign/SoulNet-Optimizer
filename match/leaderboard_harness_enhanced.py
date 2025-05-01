
import torch
import torch.nn as nn
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD, RMSprop

from extreme_scenarios import generate_batch_traces
from CollapseGrammarOptimizer_vGH1_0 import CollapseGrammarOptimizer_vGH1

def trace_integrity(losses):
    has_nan = any(np.isnan(losses))
    rebound = any(np.diff(losses) > 0.1)
    loss_range = round(max(losses) - min(losses), 4)
    return {
        "has_nan": has_nan,
        "rebound": rebound,
        "loss_range": loss_range
    }

def run_optimizer_trace(optimizer_cls, mode, name):
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    traces = generate_batch_traces(mode, batch=32)
    targets = torch.ones((32, 1))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizer_cls(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(5):
        output = model(traces)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    loss_drop = losses[0] - losses[-1]
    stability = sum(np.diff(losses) < 0) / len(losses)
    integrity = trace_integrity(losses)

    return {
        "name": name,
        "start_loss": round(losses[0], 4),
        "end_loss": round(losses[-1], 4),
        "loss_drop": round(loss_drop, 4),
        "stability": round(stability, 4),
        "trace_integrity": integrity,
        "losses": losses
    }

def leaderboard_run(mode="adversarial_spike"):
    results = []
    optimizers = {
        "CollapseGrammarGH": lambda p, lr=1e-3: CollapseGrammarOptimizer_vGH1(p, lr=lr),
        "Adam": lambda p, lr=1e-3: Adam(p, lr=lr),
        "SGD": lambda p, lr=1e-3: SGD(p, lr=lr),
        "RMSprop": lambda p, lr=1e-3: RMSprop(p, lr=lr)
    }

    for name, cls in optimizers.items():
        results.append(run_optimizer_trace(cls, mode, name))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"leaderboard_results_{mode}_{timestamp}.json"
    summaryfile = f"collapse_summary_{mode}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    summary = {}
    for entry in results:
        summary[entry["name"]] = {
            "loss_drop": entry["loss_drop"],
            "stability": entry["stability"],
            "collapse_integrity": entry["trace_integrity"]
        }

    with open(summaryfile, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Leaderboard run complete for mode '{mode}'")
    print(f"📄 Results saved to: {filename}")
    print(f"📊 Summary saved to: {summaryfile}")

    # Plot
    plt.figure(figsize=(10, 6))
    for entry in results:
        plt.plot(entry["losses"], label=entry["name"])
    plt.title(f"Trace: {mode}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plotfile = f"loss_curve_{mode}_{timestamp}.png"
    plt.savefig(plotfile)
    print(f"📈 Plot saved to: {plotfile}")

if __name__ == "__main__":
    for m in ["plateau_burst", "entropy_pulse"]:
        leaderboard_run(mode=m)

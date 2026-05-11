import json
import matplotlib.pyplot as plt

from os import path
from kqe.local_config import RESDIR

with open("../data/power_decay.json") as f:
    std = json.load(f)

with open("../data/power_decay_qmc.json") as f:
    qmc = json.load(f)

dims = std["dims"]
methods = std["distances"]

labels = {
    "ekqd_1":          "EKQD p=1",
    "ekqd_2":          "EKQD p=2",
    "ekqd_centered_1": "EKQD centered p=1",
    "ekqd_centered_2": "EKQD centered p=2",
    "supkqd_1":        "SupKQD p=1",
    "supkqd_2":        "SupKQD p=2",
    "esw":             "ESW",
    "esw_mu_normal":   "ESW (mu normal)",
}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
fig.suptitle("Power Decay — Rejection Rate vs Dimension (Standard vs QMC)", fontsize=14, fontweight="bold")

for ax, method in zip(axes, methods):
    y_std = [std["rejection_rate"][method][str(d)] for d in dims]
    y_qmc = [qmc["rejection_rate"][method][str(d)] for d in dims]

    ax.plot(dims, y_std, marker="o", color="#3266ad", linewidth=2, markersize=5, label="Standard")
    ax.plot(dims, y_qmc, marker="s", color="#e07b39", linewidth=2, markersize=5, linestyle="--", label="QMC")
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=0.9, alpha=0.7, label="α = 0.05")

    ax.set_title(labels[method], fontsize=11, fontweight="bold")
    ax.set_xlabel("Dimension", fontsize=9)
    ax.set_ylabel("Rejection rate", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(path.join(RESDIR, "power_decay_8plots.png"), dpi=150, bbox_inches="tight")
print("Saved.")

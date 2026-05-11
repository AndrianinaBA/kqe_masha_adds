import json
import matplotlib.pyplot as plt
from os import path

from kqe.local_config import RESDIR
# ── Load data ────────────────────────────────────────────────────────────────

with open("../data/laplace_vs_gaussian_qmc_type_1_normal.json") as f:
    data = json.load(f)

rejection_rate = data["rejection_rate"]
num_samples = [str(n) for n in data["num_samples_lst"]]
metrics = data["distances"]

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    values = [rejection_rate[metric][n] for n in num_samples]
    
    ax.plot(num_samples, values, marker="o", color="steelblue", linewidth=2)
    ax.axhline(y=0.05, color="red", linestyle="--", linewidth=1.5, label="α = 0.05")
    
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Rejection rate")
    ax.set_ylim(0, 0.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Type I Error per Method — Laplace vs Gaussian (QMC)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(path.join(RESDIR, "type1_error_normal.png"), dpi=150)
plt.show()

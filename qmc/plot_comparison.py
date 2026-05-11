import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from os import path
from kqe.local_config import RESDIR

# ── Load data ────────────────────────────────────────────────────────────────
with open("../data/laplace_vs_gaussian.json") as f:
    std_data = json.load(f)

with open("../data/laplace_vs_gaussian_qmc.json") as f:
    qmc_data = json.load(f)

std_rr = std_data["rejection_rate"]
qmc_rr = qmc_data["rejection_rate"]

num_samples = [str(n) for n in std_data["num_samples_lst"]]
metrics = std_data["distances"]

# ── Layout ───────────────────────────────────────────────────────────────────
n_cols = 3
n_rows = int(np.ceil(len(metrics) / n_cols))

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(n_cols * 4, n_rows * 3),
    sharey=True,
    constrained_layout=True,
)
axes_flat = axes.flatten()

# ── Style ─────────────────────────────────────────────────────────────────────
COLOR_STD = "#378ADD"
COLOR_QMC = "#1D9E75"
ALPHA_LEVEL = 0.05

x = np.array([int(n) for n in num_samples])

for ax, metric in zip(axes_flat, metrics):
    y_std = [std_rr[metric][n] for n in num_samples]
    y_qmc = [qmc_rr[metric][n] for n in num_samples]

    ax.plot(x, y_std, marker="o", color=COLOR_STD, linewidth=1.8,
            markersize=4, label="Standard")
    ax.plot(x, y_qmc, marker="s", color=COLOR_QMC, linewidth=1.8,
            markersize=4, linestyle="--", label="QMC")
    ax.axhline(ALPHA_LEVEL, color="gray", linewidth=0.8,
               linestyle=":", label="α = 0.05")

    ax.set_title(metric.replace("_", " "), fontsize=10, fontweight="500")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

# hide unused subplots
for ax in axes_flat[len(metrics):]:
    ax.set_visible(False)

# ── Shared legend ─────────────────────────────────────────────────────────────
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=3,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Rejection rate: Standard vs QMC — Laplace vs Gaussian (n=300 runs)",
    fontsize=12,
    fontweight="500",
    y=1.01,
)

# plt.savefig("rejection_rate_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: rejection_rate_comparison.png")
plt.savefig(path.join(RESDIR, "comparison.pdf"), dpi=150)

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from os import path
from kqe.local_config import RESDIR
# -----------------------------------------------------------------
# Load data
# -----------------------------------------------------------------
with open("../data/power_decay_qmc_type_1_mc.json", "r") as f:
    data = json.load(f)

rejection_rates = data["rejection_rate"]
dims = data["dims"]
alpha = 0.05

# -----------------------------------------------------------------
# Plot
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

colors = plt.cm.tab10.colors
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

for i, (metric, rates) in enumerate(rejection_rates.items()):
    x = dims
    y = [rates[str(d)] for d in dims]
    ax.plot(x, y, marker=markers[i % len(markers)], color=colors[i % len(colors)],
            label=metric, linewidth=1.8, markersize=6)

# Alpha reference line
ax.axhline(y=alpha, color="crimson", linestyle="--", linewidth=1.5, label=f"α = {alpha}")

ax.set_xscale("log", base=2)
ax.set_xticks(dims)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

ax.set_xlabel("Dimension", fontsize=12)
ax.set_ylabel("Rejection Rate (Type I Error)", fontsize=12)
ax.set_title("Type I Error by Metric and Dimension\n(H₀: X, Y ~ N(0, I) — both under null)", fontsize=13)

ax.set_ylim(0, 0.25)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(path.join(RESDIR, "type1_error_plot_power_decay_mc.png"), dpi=150)
print("Saved to type1_error_plot.png")

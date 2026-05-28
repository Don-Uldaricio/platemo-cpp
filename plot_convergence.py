import sys
import csv
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/20260526_115558/convergence.csv"

rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "algo": row["algo"],
            "dataset": int(row["dataset"]),
            "nhidden": int(row["nhidden"]),
            "fe": int(row["fe"]),
            "hv": float(row["hv"]),
        })

# group hv values by (algo, dataset, nhidden, fe)
groups = defaultdict(list)
for r in rows:
    key = (r["algo"], r["dataset"], r["nhidden"], r["fe"])
    groups[key].append(r["hv"])

# aggregate
agg = {}
for key, hvs in groups.items():
    agg[key] = (np.mean(hvs), np.std(hvs, ddof=1) if len(hvs) > 1 else 0.0)

datasets = sorted({k[1] for k in agg})
algos = sorted({k[0] for k in agg})
nhidden_vals = sorted({k[2] for k in agg})

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
linestyles = ["-", "--", "-.", ":"]
color_map = {nh: colors[i % len(colors)] for i, nh in enumerate(nhidden_vals)}
ls_map = {nh: linestyles[i % len(linestyles)] for i, nh in enumerate(nhidden_vals)}

nrows, ncols = len(datasets), len(algos)
fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5 * ncols, 4 * nrows),
    squeeze=False,
)

for r, ds in enumerate(datasets):
    for c, algo in enumerate(algos):
        ax = axes[r][c]
        for nh in nhidden_vals:
            fe_vals = sorted(fe for (a, d, n, fe) in agg if a == algo and d == ds and n == nh)
            if not fe_vals:
                continue
            means = np.array([agg[(algo, ds, nh, fe)][0] for fe in fe_vals])
            stds = np.array([agg[(algo, ds, nh, fe)][1] for fe in fe_vals])
            col = color_map[nh]
            ls = ls_map[nh]
            ax.plot(fe_vals, means, label=f"nh={nh}", color=col, linestyle=ls)
            ax.fill_between(fe_vals, means - stds, means + stds, alpha=0.2, color=col)
        ax.set_title(f"{algo} — Dataset {ds}")
        ax.set_xlabel("Function evaluations")
        ax.legend(fontsize=8)
        if c == 0:
            ax.set_ylabel("HV (mean ± std)")

plt.tight_layout()

out_path = os.path.join(os.path.dirname(csv_path), "convergence_hv.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

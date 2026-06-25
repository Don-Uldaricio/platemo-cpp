import sys
import csv
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MAX_TICKS = 10  # máximo de checkpoints de boxplot cuando hay muchas generaciones

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/20260526_115558/convergence.csv"

rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "algo":       row["algo"],
            "dataset":    int(row["dataset"]),
            "nhidden":    int(row["nhidden"]),
            "generation": int(row["generation"]),
            "hv":         float(row["hv"]),
        })

# group hv values by (algo, dataset, nhidden, generation)
groups = defaultdict(list)
for r in rows:
    key = (r["algo"], r["dataset"], r["nhidden"], r["generation"])
    groups[key].append(r["hv"])

datasets     = sorted({k[1] for k in groups})
algos        = sorted({k[0] for k in groups})
nhidden_vals = sorted({k[2] for k in groups})

colors     = plt.rcParams["axes.prop_cycle"].by_key()["color"]
linestyles = ["-", "--", "-.", ":"]
color_map  = {nh: colors[i % len(colors)] for i, nh in enumerate(nhidden_vals)}
ls_map     = {nh: linestyles[i % len(linestyles)] for i, nh in enumerate(nhidden_vals)}

nrows, ncols = len(datasets), len(algos)
fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(6 * ncols, 5 * nrows),
    squeeze=False,
)

n_nh = len(nhidden_vals)

for r, ds in enumerate(datasets):
    for c, algo in enumerate(algos):
        ax = axes[r][c]

        # collect all generations present for this (algo, ds) across all nhidden
        all_gens = sorted({g for (a, d, n, g) in groups if a == algo and d == ds})
        if not all_gens:
            ax.set_visible(False)
            continue

        max_gen = all_gens[-1]
        if len(all_gens) <= MAX_TICKS:
            ticks = all_gens
        else:
            # evenly spaced indices into all_gens, always including last
            indices = np.linspace(0, len(all_gens) - 1, MAX_TICKS, dtype=int)
            ticks = sorted({all_gens[i] for i in indices} | {max_gen})

        # width of each box within a tick group
        # tighter when generations are dense
        tick_spacing = (ticks[-1] - ticks[0]) / max(len(ticks) - 1, 1) if len(ticks) > 1 else 1
        box_width = min(tick_spacing * 0.6 / n_nh, tick_spacing * 0.55)

        legend_patches = []
        for i, nh in enumerate(nhidden_vals):
            col = color_map[nh]
            ls  = ls_map[nh]

            # offset so boxes for different nhidden don't overlap
            offset = (i - (n_nh - 1) / 2) * box_width * 1.15
            positions = [g + offset for g in ticks]

            # data for boxplots at each tick
            box_data = [groups.get((algo, ds, nh, g), []) for g in ticks]
            # skip nhidden if no data at all
            if all(len(d) == 0 for d in box_data):
                continue

            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                showmeans=True,
                boxprops=dict(facecolor=col, alpha=0.45, linewidth=1.2),
                medianprops=dict(color="black", linewidth=2),
                meanprops=dict(marker="D", markerfacecolor=col, markeredgecolor="black",
                               markersize=4),
                whiskerprops=dict(color=col, linewidth=1.2),
                capprops=dict(color=col, linewidth=1.5),
                flierprops=dict(marker="o", markerfacecolor=col, markeredgecolor="black",
                                markersize=3, alpha=0.6),
            )

            # median trend line through all available generations
            all_nh_gens = sorted(g for (a, d, n, g) in groups if a == algo and d == ds and n == nh)
            medians = [np.median(groups[(algo, ds, nh, g)]) for g in all_nh_gens]
            ax.plot(all_nh_gens, medians, color=col, linestyle=ls, linewidth=1.2, alpha=0.8)

            legend_patches.append(mpatches.Patch(facecolor=col, edgecolor="black",
                                                  alpha=0.7, label=f"nh={nh}"))

        ax.set_title(f"{algo} — Dataset {ds}")
        ax.set_xlabel("Generation")
        if c == 0:
            ax.set_ylabel("HV")
        ax.set_xlim(ticks[0] - tick_spacing * 0.6, ticks[-1] + tick_spacing * 0.6)
        if legend_patches:
            ax.legend(handles=legend_patches, fontsize=8)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(csv_path), "convergence_hv.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

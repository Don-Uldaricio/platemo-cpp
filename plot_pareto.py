#!/usr/bin/env python3
"""Plot Pareto fronts from experiment results.

Usage:
    python plot_pareto.py                        # latest results dir (runs only)
    python plot_pareto.py results/20260514_145135
    python plot_pareto.py --combined             # overlay combined Pareto front
    python plot_pareto.py --run 2                # show only run 2
    python plot_pareto.py --ds 1                 # show only dataset 1 (Iris)
    python plot_pareto.py --ds wine              # show only Wine dataset
    python plot_pareto.py --nh 40                # show only nh=40
    python plot_pareto.py --save                 # save as PNG
"""

import sys
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DS_NAMES = {1: "Statlog Australian", 2: "Climate", 3: "German", 4: "Sonar", 5: "Iris", 6: "Wine"}


def pareto_filter(points: np.ndarray) -> np.ndarray:
    """Return mask of non-dominated points (minimization on both objectives)."""
    n = len(points)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                dominated[i] = True
                break
    return ~dominated


def load_fronts(fronts_dir: Path) -> dict:
    """Return dict: (algo, ds, nh) -> list of (run_id, ndarray shape (N,2))."""
    pattern = re.compile(r"(.+)_ds(\d+)_nh(\d+)_run(\d+)_front\.csv")
    groups: dict = {}
    for f in sorted(fronts_dir.glob("*_front.csv")):
        m = pattern.match(f.name)
        if not m:
            continue
        algo, ds, nh, run = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        key = (algo, ds, nh)
        data = np.genfromtxt(f, delimiter=",", skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        groups.setdefault(key, []).append((run, data))
    return groups


def load_fitness_mode(fronts_dir: Path) -> str:
    """Read fitness_mode from any _meta.json in fronts_dir (all runs share the same mode)."""
    import json
    for mf in fronts_dir.glob("*_meta.json"):
        try:
            with open(mf) as fp:
                meta = json.load(fp)
            if "fitness_mode" in meta:
                return meta["fitness_mode"]
        except Exception:
            continue
    return "accuracy"


def plot_group(ax, key, runs, run_count, show_combined=False, fitness_mode="accuracy"):
    algo, ds, nh = key
    colors = cm.tab10(np.linspace(0, 0.9, run_count))

    all_points = []
    for run_id, pts in sorted(runs, key=lambda x: x[0]):
        sorted_pts = pts[pts[:, 0].argsort()]
        ax.scatter(
            sorted_pts[:, 0], sorted_pts[:, 1],
            color=colors[run_id % run_count],
            s=25, alpha=0.6, zorder=2,
            label=f"run {run_id}",
        )
        ax.plot(
            sorted_pts[:, 0], sorted_pts[:, 1],
            color=colors[run_id % run_count],
            linewidth=0.8, alpha=0.4, zorder=1,
        )
        all_points.append(sorted_pts)

    # Combined Pareto front across all runs
    if show_combined and len(runs) > 1:
        combined = np.vstack(all_points)
        mask = pareto_filter(combined)
        front = combined[mask]
        front = front[front[:, 0].argsort()]
        ax.scatter(front[:, 0], front[:, 1], color="black", s=50, zorder=4, marker="D",
                   label="combined front")
        ax.step(front[:, 0], front[:, 1], color="black", linewidth=1.2,
                where="post", zorder=3)

    ax.set_title(f"{algo} | {DS_NAMES.get(ds, f'ds{ds}')} | nh={nh}", fontsize=9)
    f2_label = "f2: 1 − AUC" if fitness_mode == "auc" else "f2: training error"
    ax.set_xlabel("f1: complexity (non-zero frac)", fontsize=8)
    ax.set_ylabel(f2_label, fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=6, loc="upper right")


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto fronts from experiment results")
    parser.add_argument("results_dir", nargs="?", help="Path to results run directory")
    parser.add_argument("--save", action="store_true", help="Save plot as PNG instead of showing")
    parser.add_argument("--combined", action="store_true", help="Overlay combined Pareto front on each subplot")
    parser.add_argument("--run", type=int, default=None, metavar="N", help="Show only run N")
    parser.add_argument("--ds", type=str, default=None, metavar="ID",
                        help="Show only dataset N (number or name: iris, wine, german, sonar)")
    parser.add_argument("--nh", type=int, default=None, metavar="N",
                        help="Show only results with N hidden neurons")
    args = parser.parse_args()

    ds_filter = None
    if args.ds is not None:
        try:
            ds_filter = int(args.ds)
        except ValueError:
            name_map = {v.lower(): k for k, v in DS_NAMES.items()}
            ds_filter = name_map.get(args.ds.lower())
            if ds_filter is None:
                print(f"Unknown dataset '{args.ds}'. Use a number (1–4) or name: iris, wine, german, sonar.")
                sys.exit(1)

    base = Path("results")
    if args.results_dir:
        run_dir = Path(args.results_dir)
    else:
        runs = sorted(base.glob("*/fronts"), key=lambda p: p.parent.name)
        if not runs:
            print("No results with fronts/ found. Run experiments first.")
            sys.exit(1)
        run_dir = runs[-1].parent
        print(f"Using latest run: {run_dir}")

    fronts_dir = run_dir / "fronts"
    if not fronts_dir.exists():
        print(f"No fronts/ directory in {run_dir}. Re-run experiments to generate front files.")
        sys.exit(1)

    groups = load_fronts(fronts_dir)
    if not groups:
        print(f"No *_front.csv files found in {fronts_dir}")
        sys.exit(1)

    if ds_filter is not None:
        groups = {k: v for k, v in groups.items() if k[1] == ds_filter}
    if args.nh is not None:
        groups = {k: v for k, v in groups.items() if k[2] == args.nh}
    if not groups:
        print("No results match the given filters.")
        sys.exit(1)

    n = len(groups)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    fitness_mode = load_fitness_mode(fronts_dir)
    run_count = max(len(v) for v in groups.values())
    for idx, (key, runs) in enumerate(sorted(groups.items())):
        ax = axes[idx // ncols][idx % ncols]
        filtered = [(rid, pts) for rid, pts in runs if args.run is None or rid == args.run]
        plot_group(ax, key, filtered, run_count, show_combined=args.combined,
                   fitness_mode=fitness_mode)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Pareto Fronts — {run_dir.name}", fontsize=11, y=1.01)
    plt.tight_layout()

    if args.save:
        out = run_dir / "pareto_fronts.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()



if __name__ == "__main__":
    main()

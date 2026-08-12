#!/usr/bin/env python3
"""Connectivity analysis restricted to the best-accuracy solution of each front.

For every archived run, selects the Pareto solution with minimum training
error (max accuracy) and computes two connectivity indicators on it:

  - any_out_disc: whether at least one output class has in-degree 0
    (no active hidden->output connection reaches it).
  - pct_useless_conn: percentage of that solution's active synapses that
    belong to a "useless" path, i.e. an input->hidden edge feeding a sink
    neuron (has input, no output) or a hidden->output edge coming from a
    source neuron (has output, no input). These edges are active weights
    that cannot contribute to any input-to-output path.

Aggregates by (dataset, algo) and by algo alone (pooling the 4 datasets).

Usage:
    python analyze_connectivity_best_acc.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_network import load_meta, load_front, reconstruct_matrices, _neuron_status
from analyze_connectivity import iter_front_files, ALGO_LABELS, DS_DISPLAY

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"
DEFAULT_OUT_DIR = HERE / "comparison" / "connectivity"


def best_acc_metrics(csv_path: Path):
    meta = load_meta(csv_path)
    objs, weights = load_front(csv_path, meta)
    nF, nH, nO = meta["nFeatures"], meta["nHidden"], meta["nOutputs"]
    idx = int(np.argmin(objs[:, 1]))
    w = weights[idx]
    accuracy = 1.0 - float(objs[idx, 1])

    W1, W2 = reconstruct_matrices(w, meta)
    status, has_in, has_out = _neuron_status(W1, W2, nH)

    out_indeg = (W2 > 0).sum(axis=1).astype(int)
    any_out_disc = bool((out_indeg == 0).any())

    sink_mask = status == 2     # has_in & ~has_out
    source_mask = status == 3   # ~has_in & has_out

    useless_in = int((W1[sink_mask, :] > 0).sum()) if sink_mask.any() else 0
    useless_out = int((W2[:, source_mask] > 0).sum()) if source_mask.any() else 0
    useless_edges = useless_in + useless_out

    active_edges = int((W1 > 0).sum() + (W2 > 0).sum())
    pct_useless = 100.0 * useless_edges / active_edges if active_edges > 0 else 0.0

    return {
        "accuracy": accuracy,
        "any_out_disc": any_out_disc,
        "out0_disc": bool(out_indeg[0] == 0),
        "out1_disc": bool(out_indeg[1] == 0),
        "active_edges": active_edges,
        "useless_edges": useless_edges,
        "pct_useless_conn": pct_useless,
    }


def main():
    files = iter_front_files(DEFAULT_RESULTS_DIR)
    rows = []
    for ds_no, algo, run, csv_path in files:
        m = best_acc_metrics(csv_path)
        rows.append({
            "dataset_no": ds_no,
            "dataset_name": DS_DISPLAY[ds_no],
            "algo": ALGO_LABELS[algo],
            "run": run,
            **m,
        })

    def summarize(group):
        n = len(group)
        pct_disc = 100.0 * sum(1 for r in group if r["any_out_disc"]) / n
        pct_out0 = 100.0 * sum(1 for r in group if r["out0_disc"]) / n
        pct_out1 = 100.0 * sum(1 for r in group if r["out1_disc"]) / n
        mean_useless = sum(r["pct_useless_conn"] for r in group) / n
        mean_acc = sum(r["accuracy"] for r in group) / n
        return {"n_runs": n, "mean_accuracy": mean_acc,
                "pct_any_out_disc": pct_disc, "pct_out0_disc": pct_out0,
                "pct_out1_disc": pct_out1, "mean_pct_useless_conn": mean_useless}

    by_combo = defaultdict(list)
    by_algo = defaultdict(list)
    for r in rows:
        by_combo[(r["dataset_no"], r["algo"])].append(r)
        by_algo[r["algo"]].append(r)

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "best_acc_by_combo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_no", "dataset_name", "algo", "n_runs", "mean_accuracy",
                    "pct_any_out_disc", "pct_out0_disc", "pct_out1_disc", "mean_pct_useless_conn"])
        for (ds_no, algo) in sorted(by_combo.keys()):
            s = summarize(by_combo[(ds_no, algo)])
            w.writerow([ds_no, DS_DISPLAY[ds_no], algo, s["n_runs"],
                        f"{s['mean_accuracy']:.6f}", f"{s['pct_any_out_disc']:.4f}",
                        f"{s['pct_out0_disc']:.4f}", f"{s['pct_out1_disc']:.4f}",
                        f"{s['mean_pct_useless_conn']:.4f}"])

    with open(out_dir / "best_acc_by_algo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "n_runs", "mean_accuracy", "pct_any_out_disc",
                    "pct_out0_disc", "pct_out1_disc", "mean_pct_useless_conn"])
        for algo in sorted(by_algo.keys()):
            s = summarize(by_algo[algo])
            w.writerow([algo, s["n_runs"], f"{s['mean_accuracy']:.6f}",
                        f"{s['pct_any_out_disc']:.4f}", f"{s['pct_out0_disc']:.4f}",
                        f"{s['pct_out1_disc']:.4f}", f"{s['mean_pct_useless_conn']:.4f}"])

    with open(out_dir / "best_acc_all_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_no", "dataset_name", "algo", "run", "accuracy",
                    "any_out_disc", "active_edges", "useless_edges", "pct_useless_conn"])
        for r in rows:
            w.writerow([r["dataset_no"], r["dataset_name"], r["algo"], r["run"],
                        f"{r['accuracy']:.6f}", r["any_out_disc"], r["active_edges"],
                        r["useless_edges"], f"{r['pct_useless_conn']:.4f}"])

    print("=== Por dataset x algoritmo (solo mejor exactitud) ===")
    print(f"{'ds':>4} {'algo':<8} {'runs':>5} {'acc%':>7} {'out_disc%':>10} "
          f"{'out0%':>7} {'out1%':>7} {'useless_conn%':>14}")
    for (ds_no, algo) in sorted(by_combo.keys()):
        s = summarize(by_combo[(ds_no, algo)])
        print(f"{ds_no:>4} {algo:<8} {s['n_runs']:>5} {s['mean_accuracy']*100:>6.1f} "
              f"{s['pct_any_out_disc']:>9.1f} {s['pct_out0_disc']:>6.1f} {s['pct_out1_disc']:>6.1f} "
              f"{s['mean_pct_useless_conn']:>13.1f}")

    print("\n=== Por algoritmo (4 datasets agrupados, solo mejor exactitud) ===")
    for algo in sorted(by_algo.keys()):
        s = summarize(by_algo[algo])
        print(f"{algo:<8} {s['n_runs']:>5} {s['mean_accuracy']*100:>6.1f} "
              f"{s['pct_any_out_disc']:>9.1f} {s['pct_out0_disc']:>6.1f} {s['pct_out1_disc']:>6.1f} "
              f"{s['mean_pct_useless_conn']:>13.1f}")

    print(f"\nEscrito {out_dir / 'best_acc_by_combo.csv'}")
    print(f"Escrito {out_dir / 'best_acc_by_algo.csv'}")
    print(f"Escrito {out_dir / 'best_acc_all_runs.csv'}")


if __name__ == "__main__":
    main()

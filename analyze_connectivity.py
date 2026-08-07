#!/usr/bin/env python3
"""Batch connectivity analysis across every archived Pareto front.

Scans all results/ds*_*/fronts/*_front.csv files, classifies hidden-neuron
and output-neuron connectivity for every Pareto solution (not just one
selected solution, unlike analyze_network.py), and aggregates the findings
per (dataset, algorithm) combination plus a repo-wide total.

"Active" synapse means weight > epsilon (default epsilon=0.0, which is
exactly the criterion SparseSNN::CalObj uses for f1_complexity in the C++
code, so results here are directly comparable to that objective).

Usage:
    python analyze_connectivity.py
    python analyze_connectivity.py --epsilon 1e-6   # sensitivity check
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_network import (
    load_meta, load_front, reconstruct_matrices, _neuron_status,
    plot_network_graph,
)

ALGO_LABELS = {"moeackf": "MOEACKF", "snsgaii": "SNSGAII"}
DS_DISPLAY = {1: "Statlog Australian", 2: "Climate", 3: "Statlog German", 4: "Sonar"}
LEGIBLE_DATASETS = {1, 2, 3}  # nFeatures <= 40 -> plot_network_graph draws a full diagram

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"
DEFAULT_OUT_DIR = HERE / "comparison" / "connectivity"

SUMMARY_FIELDS = [
    "dataset_no", "dataset_name", "algo", "n_runs", "n_solutions",
    "mean_dead_frac_pooled", "mean_dead_frac_run_avg",
    "mean_sink_frac_pooled", "mean_source_frac_pooled",
    "pct_sol_out_disc_pooled", "pct_sol_out_disc_run_avg", "pct_run_any_out_disc",
    "pct_sparsest_sol_out_disc", "pct_best_acc_sol_out_disc",
    "pct_out0_disc_pooled", "pct_out1_disc_pooled",
    "pct_sol_out_unreachable_real_pooled", "bias_only_frac",
]

ALL_SOLUTIONS_FIELDS = [
    "dataset_no", "dataset_name", "algo", "run", "sol_idx", "n_sol_in_front",
    "f1_complexity", "f2_train_error", "accuracy", "is_sparsest", "is_best_acc",
    "n_hidden", "n_live", "n_dead", "n_sink", "n_source",
    "dead_frac", "sink_frac", "source_frac",
    "out0_indeg", "out1_indeg", "out0_disc", "out1_disc", "any_out_disc", "n_out_disc",
    "bias_only_neurons", "bias_only_frac",
    "real_reach_out0", "real_reach_out1", "any_out_unreachable_real",
    "f1_check_err", "csv_path",
]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def iter_front_files(results_dir: Path):
    """Yield (dataset_no, algo, run, csv_path) for every *_front.csv, sorted."""
    combo_pattern = re.compile(r"^ds(\d+)_(moeackf|snsgaii)$")
    run_pattern = re.compile(r"_run(\d+)_front\.csv$")
    entries = []
    for combo_dir in sorted(results_dir.glob("ds*_*")):
        m = combo_pattern.match(combo_dir.name)
        if not m:
            continue
        ds_no, algo = int(m.group(1)), m.group(2)
        fronts_dir = combo_dir / "fronts"
        if not fronts_dir.is_dir():
            continue
        for csv_path in sorted(fronts_dir.glob("*_front.csv")):
            rm = run_pattern.search(csv_path.name)
            if not rm:
                continue
            assert ALGO_LABELS[algo] in csv_path.name, (
                f"{csv_path.name}: expected algo label {ALGO_LABELS[algo]!r} in filename"
            )
            entries.append((ds_no, algo, int(rm.group(1)), csv_path))
    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    return entries


# ---------------------------------------------------------------------------
# Per-solution connectivity metrics
# ---------------------------------------------------------------------------

def solution_metrics(W1: np.ndarray, W2: np.ndarray, nF: int, nH: int, nO: int,
                     epsilon: float = 0.0) -> dict:
    """Classify one solution's hidden/output connectivity.

    Reuses analyze_network._neuron_status unmodified: to support --epsilon we
    floor sub-threshold weights to exactly 0.0 first, so its internal ">0"
    check still does the right thing.
    """
    assert nO == 2, f"expected nOutputs==2 for all archived data, got {nO}"
    if epsilon > 0.0:
        W1 = np.where(W1 > epsilon, W1, 0.0)
        W2 = np.where(W2 > epsilon, W2, 0.0)

    status, has_in, _ = _neuron_status(W1, W2, nH)
    has_in_real = (W1[:, :nF] > 0).any(axis=1)   # excludes the bias-only column
    bias_only = has_in & ~has_in_real

    out_indeg = (W2 > 0).sum(axis=1).astype(int)                     # (2,)
    real_reach = ((W2 > 0) & has_in_real[None, :]).any(axis=1)        # (2,) bool

    n_dead = int((status == 1).sum())
    n_sink = int((status == 2).sum())
    n_source = int((status == 3).sum())

    return {
        "n_live": int((status == 0).sum()),
        "n_dead": n_dead,
        "n_sink": n_sink,
        "n_source": n_source,
        "dead_frac": n_dead / nH,
        "sink_frac": n_sink / nH,
        "source_frac": n_source / nH,
        "out0_indeg": int(out_indeg[0]),
        "out1_indeg": int(out_indeg[1]),
        "out0_disc": bool(out_indeg[0] == 0),
        "out1_disc": bool(out_indeg[1] == 0),
        "any_out_disc": bool((out_indeg == 0).any()),
        "n_out_disc": int((out_indeg == 0).sum()),
        "bias_only_neurons": int(bias_only.sum()),
        "bias_only_frac": float(bias_only.sum()) / nH,
        "real_reach_out0": bool(real_reach[0]),
        "real_reach_out1": bool(real_reach[1]),
        "any_out_unreachable_real": bool((~real_reach).any()),
    }


def scan_corpus(files, epsilon: float = 0.0):
    """Load every front, compute per-solution metrics for every row.

    Returns (rows, stats) where stats carries corpus-wide footnote numbers
    that don't fit the per-solution/per-combo shape.
    """
    rows = []
    tiny_weight_files = set()
    n_tiny_weights = 0
    max_f1_err = 0.0

    for ds_no, algo, run, csv_path in files:
        meta = load_meta(csv_path)
        objs, weights = load_front(csv_path, meta)
        nF, nH, nO, D = meta["nFeatures"], meta["nHidden"], meta["nOutputs"], meta["D"]
        n_sol = len(objs)

        idx_best_acc = int(np.argmin(objs[:, 1]))
        idx_sparsest = int(np.argmin(objs[:, 0]))

        tiny_mask = (weights > 0) & (weights < 1e-6)
        if tiny_mask.any():
            n_tiny_weights += int(tiny_mask.sum())
            tiny_weight_files.add(str(csv_path))

        for idx in range(n_sol):
            w = weights[idx]
            nz = int((w > 0).sum())
            f1_check_err = abs(float(objs[idx, 0]) - nz / D)
            max_f1_err = max(max_f1_err, f1_check_err)

            W1, W2 = reconstruct_matrices(w, meta)
            m = solution_metrics(W1, W2, nF, nH, nO, epsilon=epsilon)

            rows.append({
                "dataset_no": ds_no,
                "dataset_name": DS_DISPLAY[ds_no],
                "algo": ALGO_LABELS[algo],
                "run": run,
                "sol_idx": idx,
                "n_sol_in_front": n_sol,
                "f1_complexity": float(objs[idx, 0]),
                "f2_train_error": float(objs[idx, 1]),
                "accuracy": 1.0 - float(objs[idx, 1]),
                "is_sparsest": idx == idx_sparsest,
                "is_best_acc": idx == idx_best_acc,
                "n_hidden": nH,
                "f1_check_err": f1_check_err,
                "csv_path": str(csv_path),
                **m,
            })

    stats = {
        "n_tiny_weights": n_tiny_weights,
        "n_files_with_tiny_weights": len(tiny_weight_files),
        "max_f1_check_err": max_f1_err,
    }
    return rows, stats


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize_group(group_rows: list) -> dict:
    n_sol = len(group_rows)
    per_run = defaultdict(list)
    for r in group_rows:
        per_run[(r["dataset_no"], r["algo"], r["run"])].append(r)
    n_runs = len(per_run)

    def pooled_pct(key):
        return 100.0 * sum(1 for r in group_rows if r[key]) / n_sol

    def pooled_mean(key):
        return sum(r[key] for r in group_rows) / n_sol

    def run_avg_mean(key):
        means = [sum(r[key] for r in rs) / len(rs) for rs in per_run.values()]
        return sum(means) / len(means)

    def run_avg_pct(key):
        means = [100.0 * sum(1 for r in rs if r[key]) / len(rs) for rs in per_run.values()]
        return sum(means) / len(means)

    def subset_pct(pred, key):
        subset = [r for r in group_rows if pred(r)]
        if not subset:
            return float("nan")
        return 100.0 * sum(1 for r in subset if r[key]) / len(subset)

    pct_run_any_out_disc = 100.0 * sum(
        1 for rs in per_run.values() if any(r["any_out_disc"] for r in rs)
    ) / n_runs

    return {
        "n_runs": n_runs,
        "n_solutions": n_sol,
        "mean_dead_frac_pooled": pooled_mean("dead_frac"),
        "mean_dead_frac_run_avg": run_avg_mean("dead_frac"),
        "mean_sink_frac_pooled": pooled_mean("sink_frac"),
        "mean_source_frac_pooled": pooled_mean("source_frac"),
        "pct_sol_out_disc_pooled": pooled_pct("any_out_disc"),
        "pct_sol_out_disc_run_avg": run_avg_pct("any_out_disc"),
        "pct_run_any_out_disc": pct_run_any_out_disc,
        "pct_sparsest_sol_out_disc": subset_pct(lambda r: r["is_sparsest"], "any_out_disc"),
        "pct_best_acc_sol_out_disc": subset_pct(lambda r: r["is_best_acc"], "any_out_disc"),
        "pct_out0_disc_pooled": pooled_pct("out0_disc"),
        "pct_out1_disc_pooled": pooled_pct("out1_disc"),
        "pct_sol_out_unreachable_real_pooled": pooled_pct("any_out_unreachable_real"),
        "bias_only_frac": pooled_mean("bias_only_frac"),
    }


def build_summary_rows(rows: list) -> list:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset_no"], r["algo"])].append(r)

    summary_rows = []
    for ds_no, algo in sorted(groups.keys()):
        s = summarize_group(groups[(ds_no, algo)])
        s["dataset_no"], s["dataset_name"], s["algo"] = ds_no, DS_DISPLAY[ds_no], algo
        summary_rows.append(s)

    all_summary = summarize_group(rows)
    all_summary["dataset_no"], all_summary["dataset_name"], all_summary["algo"] = "ALL", "Todos", "ALL"
    summary_rows.append(all_summary)
    return summary_rows


# ---------------------------------------------------------------------------
# Curated example selection (deterministic rules, see plan doc §3)
# ---------------------------------------------------------------------------

def select_example_a(rows: list):
    """Structural extreme: fewest active weights among solutions with BOTH outputs disconnected."""
    candidates = [r for r in rows if r["out0_disc"] and r["out1_disc"]]
    if not candidates:
        return None
    candidates.sort(key=lambda r: (r["f1_complexity"], r["dataset_no"], r["algo"], r["run"], r["sol_idx"]))
    return candidates[0]


def select_example_b(rows: list):
    """Deployable case: combo with worst best-accuracy-solution disconnection rate."""
    best_rows = [r for r in rows if r["is_best_acc"] and r["dataset_no"] in LEGIBLE_DATASETS]
    by_combo = defaultdict(list)
    for r in best_rows:
        by_combo[(r["dataset_no"], r["algo"])].append(r)
    if not by_combo:
        return None

    def disc_rate(combo):
        rs = by_combo[combo]
        return sum(1 for r in rs if r["any_out_disc"]) / len(rs)

    worst_combo = max(by_combo, key=lambda c: (disc_rate(c), c))
    affected = [r for r in by_combo[worst_combo] if r["any_out_disc"]]
    pool = affected if affected else by_combo[worst_combo]
    median_acc = float(np.median([r["accuracy"] for r in pool]))
    pool = sorted(pool, key=lambda r: (abs(r["accuracy"] - median_acc), r["run"]))
    return pool[0]


def select_example_c(rows: list):
    """Interior/moderate case: exactly one output disconnected, not a front extreme."""
    interior = [
        r for r in rows
        if r["n_out_disc"] == 1 and not r["is_sparsest"] and not r["is_best_acc"]
        and r["n_sol_in_front"] >= 3 and r["dataset_no"] in LEGIBLE_DATASETS
    ]
    if not interior:
        return None
    dead_fracs = [r["dead_frac"] for r in interior]
    q1, q3 = np.percentile(dead_fracs, [25, 75])
    pool = [r for r in interior if q1 <= r["dead_frac"] <= q3] or interior
    median_dead = float(np.median([r["dead_frac"] for r in pool]))
    pool = sorted(pool, key=lambda r: (
        abs(r["dead_frac"] - median_dead), r["dataset_no"], r["algo"], r["run"], r["sol_idx"]
    ))
    return pool[0]


def render_example(row: dict, out_path: Path, epsilon: float = 0.0):
    csv_path = Path(row["csv_path"])
    meta = load_meta(csv_path)
    _, weights = load_front(csv_path, meta)
    w = weights[row["sol_idx"]]
    if epsilon > 0.0:
        w = np.where(w > epsilon, w, 0.0)
    W1, W2 = reconstruct_matrices(w, meta)
    plot_network_graph(W1, W2, meta, out_path)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_summary_csv(summary_rows: list, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row[k] for k in SUMMARY_FIELDS})


def write_all_solutions_csv(rows: list, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_SOLUTIONS_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in ALL_SOLUTIONS_FIELDS})


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

def run_self_checks(rows: list, stats: dict, summary_rows: list) -> bool:
    print("\n=== Autoverificación ===")
    ok = True

    if stats["max_f1_check_err"] < 1e-6:
        print(f"[PASS] f1_complexity (CSV) vs. recomputo (pesos>0)/D: "
              f"error máximo = {stats['max_f1_check_err']:.2e}")
    else:
        print(f"[FAIL] f1_complexity vs. recomputo: error máximo = "
              f"{stats['max_f1_check_err']:.2e} (> 1e-6)")
        ok = False

    golden = next((r for r in rows if r["dataset_no"] == 1 and r["algo"] == "MOEACKF"
                   and r["run"] == 0 and r["sol_idx"] == 0), None)
    if golden is None:
        print("[FAIL] Ejemplo dorado (ds1_moeackf run0 sol0) no encontrado")
        ok = False
    else:
        expected = {"n_dead": 39, "n_sink": 1, "n_live": 0, "n_source": 0,
                    "out0_indeg": 0, "out1_indeg": 0}
        for k, exp in expected.items():
            got = golden[k]
            status = "PASS" if got == exp else "FAIL"
            ok = ok and (status == "PASS")
            print(f"[{status}] ejemplo dorado {k}: esperado={exp} obtenido={got}")

    frac_keys = ("dead_frac", "sink_frac", "source_frac", "bias_only_frac")
    bad = [(r["csv_path"], r["sol_idx"], k, r[k])
           for r in rows for k in frac_keys if not (-1e-9 <= r[k] <= 1.0 + 1e-9)]
    if bad:
        print(f"[FAIL] {len(bad)} valores de fracción fuera de [0,1], p.ej. {bad[0]}")
        ok = False
    else:
        print("[PASS] Todas las fracciones por-solución dentro de [0,1]")

    pct_cols = [c for c in SUMMARY_FIELDS if c.startswith("pct_")]
    bad_pct = [(s["dataset_no"], s["algo"], c, s[c]) for s in summary_rows for c in pct_cols
               if s[c] == s[c] and not (-1e-9 <= s[c] <= 100.0 + 1e-9)]  # s[c]==s[c] filters NaN
    if bad_pct:
        print(f"[FAIL] {len(bad_pct)} porcentajes agregados fuera de [0,100], p.ej. {bad_pct[0]}")
        ok = False
    else:
        print("[PASS] Todas las columnas pct_* agregadas dentro de [0,100]")

    print(f"[INFO] pesos no-cero pero < 1e-6: {stats['n_tiny_weights']} "
          f"en {stats['n_files_with_tiny_weights']} de {len(set(r['csv_path'] for r in rows))} archivos")
    print(f"[INFO] soluciones analizadas: {len(rows)}")

    print("=== TODO PASS ===" if ok else "=== HAY FALLOS, revisar arriba antes de citar estos números ===")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Treat weights <= epsilon as inactive (default 0.0, matches "
                             "f1_complexity exactly); use e.g. 1e-6 for a sensitivity check")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = iter_front_files(args.results_dir)
    print(f"Encontrados {len(files)} archivos *_front.csv bajo {args.results_dir}")
    if not files:
        raise SystemExit(f"No se encontraron *_front.csv bajo {args.results_dir}")

    rows, stats = scan_corpus(files, epsilon=args.epsilon)
    print(f"Analizadas {len(rows)} soluciones (filas) en total")

    summary_rows = build_summary_rows(rows)
    write_summary_csv(summary_rows, args.out / "connectivity_summary.csv")
    write_all_solutions_csv(rows, args.out / "all_solutions.csv")
    print(f"Escrito {args.out / 'connectivity_summary.csv'}")
    print(f"Escrito {args.out / 'all_solutions.csv'}")

    ok = run_self_checks(rows, stats, summary_rows)

    print("\n=== Resumen por combinación ===")
    print(f"{'ds':>4} {'algo':<8} {'runs':>5} {'sols':>5} {'dead%(runavg)':>14} "
          f"{'out_disc%(bestacc)':>19} {'out_disc%(pooled)':>18}")
    for s in summary_rows:
        print(f"{str(s['dataset_no']):>4} {s['algo']:<8} {s['n_runs']:>5} {s['n_solutions']:>5} "
              f"{s['mean_dead_frac_run_avg'] * 100:>13.1f} {s['pct_best_acc_sol_out_disc']:>18.1f} "
              f"{s['pct_sol_out_disc_pooled']:>17.1f}")

    print("\n=== Ejemplos ilustrativos ===")
    examples = [
        ("A", select_example_a(rows), "example_a_extreme"),
        ("B", select_example_b(rows), "example_b_deployable"),
        ("C", select_example_c(rows), "example_c_interior"),
    ]
    for label, row, fname in examples:
        if row is None:
            print(f"[WARN] No se encontró candidato para el Ejemplo {label}")
            continue
        out_path = args.out / f"{fname}.png"
        render_example(row, out_path, epsilon=args.epsilon)
        print(f"Ejemplo {label}: ds{row['dataset_no']} {row['algo']} run{row['run']} "
              f"sol{row['sol_idx']}/{row['n_sol_in_front']}  acc={row['accuracy']:.3f} "
              f"f1={row['f1_complexity']:.4f}  dead={row['n_dead']}/{row['n_hidden']}  "
              f"out_indeg=[{row['out0_indeg']},{row['out1_indeg']}]  -> {out_path}")

    if args.epsilon == 0.0:
        print("\n[INFO] Para el chequeo de sensibilidad correr de nuevo con --epsilon 1e-6 "
              "y comparar connectivity_summary.csv (usar --out a una carpeta distinta para no "
              "sobrescribir los resultados con epsilon=0).")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

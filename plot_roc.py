#!/usr/bin/env python3
"""Grafica curvas ROC y calcula AUC para cada solución del frente de Pareto.

Requiere el CSV generado por --spikes-out y el JSON de metadatos (_meta.json).

Uso:
    python plot_roc.py <spikes_csv>
    python plot_roc.py <spikes_csv> --save
    python plot_roc.py <spikes_csv> --max-solutions 5
    python plot_roc.py <spikes_csv> --color-by sparsity
    python plot_roc.py <spikes_csv> --threshold 3          # evalúa todas las redes con k=3
    python plot_roc.py <spikes_csv> --solution 2           # solo la red con solution_id=2
    python plot_roc.py <spikes_csv> --solution 2 --threshold 3
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_meta(spikes_path: Path) -> dict:
    meta_path = Path(str(spikes_path).replace("_spikes.csv", "_meta.json"))
    if not meta_path.exists():
        # buscar .meta.json genérico
        meta_path = Path(str(spikes_path) + ".meta.json")
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def compute_roc(spike_counts: np.ndarray, labels: np.ndarray, T_max: int):
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    fprs = [0.0]
    tprs = [0.0]
    for k in range(T_max, -1, -1):
        pred_pos = spike_counts >= k
        tp = int((pred_pos & (labels == 1)).sum())
        fp = int((pred_pos & (labels == 0)).sum())
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)

    return np.array(fprs), np.array(tprs)


def compute_auc(fprs: np.ndarray, tprs: np.ndarray) -> float:
    return float(np.trapz(tprs, fprs))


def best_youden_threshold(fprs: np.ndarray, tprs: np.ndarray, T_max: int):
    """Retorna (k, tpr, fpr) del umbral que maximiza TPR-FPR (Youden's J).
    El índice 0 es el punto artificial (0,0); índice i → k = T_max-(i-1)."""
    j = tprs - fprs
    idx = int(np.argmax(j[1:]) + 1)  # ignorar índice 0
    k = T_max - (idx - 1)
    return k, float(tprs[idx]), float(fprs[idx])


def eval_at_threshold(counts: np.ndarray, labels: np.ndarray, k: int):
    """Evalúa la red con un umbral fijo k. Retorna (accuracy, tpr, fpr, J)."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    pred = (counts >= k).astype(int)
    correct = int((pred == labels).sum())
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tpr = tp / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    return correct / len(labels), tpr, fpr, tpr - fpr


def main():
    parser = argparse.ArgumentParser(description="Plot ROC curves from spike count CSV")
    parser.add_argument("spikes_csv", help="CSV generado con --spikes-out")
    parser.add_argument("--save", action="store_true", help="Guardar figura en PNG")
    parser.add_argument("--max-solutions", type=int, default=None,
                        help="Máximo de soluciones a graficar (default: todas)")
    parser.add_argument("--color-by", choices=["sparsity", "f2", "auc"], default="sparsity",
                        help="Variable para colorear las curvas (default: sparsity)")
    parser.add_argument("--T-max", type=int, default=None,
                        help="Umbral máximo de spikes (sobreescribe _meta.json)")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Evaluar todas las redes con este umbral fijo (en lugar de k*)")
    parser.add_argument("--solution", type=int, default=None,
                        help="Evaluar/graficar solo esta solución (solution_id)")
    args = parser.parse_args()

    spikes_path = Path(args.spikes_csv)
    if not spikes_path.exists():
        print(f"Error: {spikes_path} no existe.", file=sys.stderr)
        sys.exit(1)

    meta = load_meta(spikes_path)
    T_max = args.T_max or meta.get("T_max")
    if T_max is None:
        print("Error: T_max no encontrado. Especifícalo con --T-max.", file=sys.stderr)
        sys.exit(1)

    # Cargar CSV
    import csv
    solutions = {}  # solution_id → {"f1": ..., "f2": ..., "counts": [], "labels": []}
    with open(spikes_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["solution_id"])
            if sid not in solutions:
                solutions[sid] = {
                    "f1": float(row["f1_complexity"]),
                    "f2": float(row["f2_train_metric"]),
                    "counts": [],
                    "labels": [],
                }
            solutions[sid]["counts"].append(int(row["spike_count"]))
            solutions[sid]["labels"].append(int(row["true_label"]))

    if not solutions:
        print("Error: CSV vacío.", file=sys.stderr)
        sys.exit(1)

    sol_ids = sorted(solutions.keys())
    if args.solution is not None:
        if args.solution not in solutions:
            print(f"Error: solution_id={args.solution} no encontrada. "
                  f"IDs disponibles: {sol_ids}", file=sys.stderr)
            sys.exit(1)
        sol_ids = [args.solution]
    elif args.max_solutions:
        sol_ids = sol_ids[: args.max_solutions]

    # Calcular AUC para ordenar y colorear
    auc_values = {}
    for sid in sol_ids:
        counts = np.array(solutions[sid]["counts"])
        labels = np.array(solutions[sid]["labels"])
        fprs, tprs = compute_roc(counts, labels, T_max)
        auc_values[sid] = compute_auc(fprs, tprs)

    # Colorear según la variable elegida
    if args.color_by == "sparsity":
        color_vals = {sid: solutions[sid]["f1"] for sid in sol_ids}
        cbar_label = "Sparsity (f1)"
    elif args.color_by == "f2":
        color_vals = {sid: solutions[sid]["f2"] for sid in sol_ids}
        cbar_label = "Train metric (f2)"
    else:
        color_vals = {sid: auc_values[sid] for sid in sol_ids}
        cbar_label = "AUC"

    vals = list(color_vals.values())
    v_min, v_max = min(vals), max(vals)
    norm = plt.Normalize(v_min, v_max) if v_max > v_min else plt.Normalize(0, 1)
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(7, 6))

    for sid in sol_ids:
        counts = np.array(solutions[sid]["counts"])
        labels = np.array(solutions[sid]["labels"])
        fprs, tprs = compute_roc(counts, labels, T_max)
        auc = auc_values[sid]
        color = cmap(norm(color_vals[sid]))
        ax.plot(fprs, tprs, color=color, alpha=0.7, linewidth=1.2,
                label=f"sol {sid}  AUC={auc:.3f}")

        # Marcar el umbral elegido (fijo o k*) sobre la curva
        if args.threshold is not None:
            _, tpr_pt, fpr_pt, _ = eval_at_threshold(counts, labels, args.threshold)
        else:
            _, tpr_pt, fpr_pt = best_youden_threshold(fprs, tprs, T_max)
        ax.plot(fpr_pt, tpr_pt, "o", color=color, markersize=5, zorder=5)

    # Línea de referencia (clasificador aleatorio)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Aleatorio")

    # Delimitadores del área máxima (clasificador perfecto en (0,1))
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(y=1, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    threshold_label = f"k={args.threshold} (fijo)" if args.threshold is not None else "k* (Youden)"
    ax.set_title(f"Curvas ROC — Frente de Pareto\n"
                 f"(T_max={T_max}, {len(sol_ids)} soluciones, umbral: {threshold_label})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=cbar_label)

    # Leyenda solo si hay pocas soluciones
    if len(sol_ids) <= 10:
        ax.legend(fontsize=7, loc="lower right")

    auc_arr = np.array(list(auc_values.values()))
    print(f"AUC  media: {auc_arr.mean():.4f}  ±{auc_arr.std():.4f}")
    print(f"AUC  min:   {auc_arr.min():.4f}    max: {auc_arr.max():.4f}")

    use_fixed_threshold = args.threshold is not None
    k_label = f"k={args.threshold}" if use_fixed_threshold else "k*"
    print(f"\n{'sol':>4}  {'f1':>6}  {'f2':>6}  {'AUC':>6}  {k_label:>4}  {'Acc':>6}  {'TPR':>6}  {'FPR':>6}  {'J':>6}")
    print("-" * 66)
    best_ks = []
    for sid in sol_ids:
        counts = np.array(solutions[sid]["counts"])
        labels = np.array(solutions[sid]["labels"])
        fprs_s, tprs_s = compute_roc(counts, labels, T_max)
        auc = auc_values[sid]
        f1  = solutions[sid]["f1"]
        f2  = solutions[sid]["f2"]

        if use_fixed_threshold:
            k_used = args.threshold
            acc, tpr_k, fpr_k, j_k = eval_at_threshold(counts, labels, k_used)
        else:
            k_used, tpr_k, fpr_k = best_youden_threshold(fprs_s, tprs_s, T_max)
            acc, _, _, j_k = eval_at_threshold(counts, labels, k_used)

        print(f"{sid:>4}  {f1:>6.3f}  {f2:>6.3f}  {auc:>6.3f}  {k_used:>4}  {acc:>6.3f}  {tpr_k:>6.3f}  {fpr_k:>6.3f}  {j_k:>6.3f}")
        best_ks.append(k_used)

    if not use_fixed_threshold:
        best_ks_arr = np.array(best_ks)
        print(f"\nUmbral óptimo (Youden)  media: {best_ks_arr.mean():.1f}  min: {best_ks_arr.min()}  max: {best_ks_arr.max()}")

    plt.tight_layout()
    if args.save:
        out = spikes_path.with_name(spikes_path.stem + "_roc.png")
        plt.savefig(out, dpi=150)
        print(f"Figura guardada en: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

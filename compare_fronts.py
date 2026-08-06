#!/usr/bin/env python3
"""Compara los frentes de Pareto de MOEACKF vs SNSGAII en results/normal_results, con foco en
complejidad (f1) vs. error de entrenamiento (f2) — ángulo "frentes" de docs/metodologia_comparacion.md.

Tres técnicas complementarias, por dataset:
  1. Solución de mejor accuracy por corrida: en cada *_front.csv, la fila con f2_train_error
     mínimo. Da una muestra pareada por seed (complejidad, error, tamaño del frente) comparable
     con compare_platform.py (que usa la misma definición sobre MATLAB_RESULTS).
  2. Frente combinado por algoritmo (unión no-dominada de las 31 corridas) + C-metric de
     cobertura (Zitzler, dominancia débil) + gráfico superpuesto.
  3. Error mediano por banda (decil) de complejidad, sobre el pool de puntos crudos de ambos
     algoritmos — para ver en qué región del espacio domina cada uno.

Usage:
    python compare_fronts.py
    python compare_fronts.py --results-dir results/normal_results --out comparison/fronts
    python compare_fronts.py --ds 1 --show   # ventana interactiva (zoom/pan) solo para ds1
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import comparison_lib as lib

ANGLE = "fronts_best_accuracy"


def _raw_points(results_dir: Path, ds: int, algo: str) -> np.ndarray:
    fronts_dir = lib.combo_dir(results_dir, ds, algo) / "fronts"
    nh = lib.nhidden_of(results_dir, ds, algo)
    parts = []
    for f in sorted(fronts_dir.glob(f"{lib.ALGO_LABELS[algo]}_ds{ds}_nh{nh}_run*_front.csv")):
        df = lib.load_front_csv(f)
        parts.append(df[["f1_complexity", "f2_train_error"]].to_numpy())
    return np.vstack(parts)


def analyze_best_accuracy(results_dir: Path, ds: int, alpha: float) -> tuple[str, list[dict]]:
    best = {}
    for algo in lib.ALGOS:
        seeds = set(lib.load_results_csv(results_dir, ds, algo, columns=("seed", "hv"))["seed"])
        nh = lib.nhidden_of(results_dir, ds, algo)
        fronts_dir = lib.combo_dir(results_dir, ds, algo) / "fronts"
        best[algo] = lib.load_all_runs_best_accuracy(fronts_dir, algo, ds, nh, seeds)

    label_a, label_b = lib.ALGO_LABELS["moeackf"], lib.ALGO_LABELS["snsgaii"]
    md = [f"## Dataset {ds} ({lib.DS_NAMES.get(ds, ds)}) — solución de mejor accuracy por corrida\n"]
    csv_rows: list[dict] = []
    for metric, col, metric_label in (
        ("complexity", "f1_complexity", "Complejidad de la mejor red (frac. pesos activos)"),
        ("error", "f2_train_error", "Error de entrenamiento de la mejor red"),
        ("front_size", "front_size", "Tamaño del frente (# soluciones no dominadas)"),
    ):
        x, y, _ = lib.align_by_seed(best["moeackf"], best["snsgaii"], col,
                                     label_x="moeackf", label_y="snsgaii")
        res = lib.paired_comparison(x, y, label_a, label_b, alpha)
        md.append(f"### {metric_label}\n")
        md.append(lib.render_comparison_markdown(res, metric_label))
        md.append("")
        csv_rows.extend(lib.render_comparison_csv_row(res, ds, ANGLE, metric))
    return "\n".join(md), csv_rows


def analyze_combined_front(results_dir: Path, ds: int, out_dir: Path,
                            show: bool = False) -> tuple[str, list[dict]]:
    fronts = {algo: lib.combined_front(_raw_points(results_dir, ds, algo)) for algo in lib.ALGOS}
    c_ab, c_ba = lib.c_metric(fronts["moeackf"], fronts["snsgaii"])

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"moeackf": "tab:blue", "snsgaii": "tab:orange"}
    for algo in lib.ALGOS:
        front = fronts[algo]
        ax.scatter(front[:, 0], front[:, 1], label=lib.ALGO_LABELS[algo], s=45, zorder=3,
                   color=colors[algo])
        ax.step(front[:, 0], front[:, 1], where="post", color=colors[algo], alpha=0.6,
                linewidth=1.3, zorder=2)
    ax.set_xlabel("f1: complexity (non-zero frac)")
    ax.set_ylabel("f2: training error")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Frente combinado (31 corridas, no dominado) — {lib.DS_NAMES.get(ds, ds)}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    png_name = f"ds{ds}_combined_front.png"
    fig.savefig(out_dir / png_name, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    md = (
        f"## Dataset {ds} ({lib.DS_NAMES.get(ds, ds)}) — frente combinado y C-metric\n\n"
        f"Unión de las 31 corridas por algoritmo, deduplicada y filtrada a no-dominados "
        f"(dominancia estricta).\n\n"
        f"| Métrica | Valor |\n|---|---|\n"
        f"| Tamaño frente combinado MOEACKF | {len(fronts['moeackf'])} |\n"
        f"| Tamaño frente combinado SNSGAII | {len(fronts['snsgaii'])} |\n"
        f"| C(MOEACKF, SNSGAII) — fracción del frente SNSGAII cubierta (dominada o igualada) "
        f"por el frente MOEACKF | {c_ab:.3f} |\n"
        f"| C(SNSGAII, MOEACKF) — fracción del frente MOEACKF cubierta por el frente SNSGAII "
        f"| {c_ba:.3f} |\n\n"
        f"![Frente combinado ds{ds}]({png_name})\n"
    )
    csv_rows = [{
        "dataset": ds, "angle": "fronts_combined", "metric": "c_metric",
        "n_front_moeackf": len(fronts["moeackf"]), "n_front_snsgaii": len(fronts["snsgaii"]),
        "c_moeackf_covers_snsgaii": c_ab, "c_snsgaii_covers_moeackf": c_ba,
    }]
    return md, csv_rows


def analyze_complexity_bands(results_dir: Path, ds: int, out_dir: Path, n_bins: int = 10,
                              show: bool = False) -> str:
    parts = []
    for algo in lib.ALGOS:
        pts = _raw_points(results_dir, ds, algo)
        df = pd.DataFrame(pts, columns=["f1_complexity", "f2_train_error"])
        df["algo"] = lib.ALGO_LABELS[algo]
        parts.append(df)
    pooled = pd.concat(parts, ignore_index=True)
    pooled["band"] = pd.qcut(pooled["f1_complexity"], q=n_bins, duplicates="drop")

    median_err = pooled.groupby(["band", "algo"], observed=True)["f2_train_error"].median().unstack("algo")
    counts = pooled.groupby(["band", "algo"], observed=True).size().unstack("algo").fillna(0)
    bands = sorted(median_err.index, key=lambda iv: iv.left)

    label_a, label_b = lib.ALGO_LABELS["moeackf"], lib.ALGO_LABELS["snsgaii"]
    lines = [
        f"## Dataset {ds} ({lib.DS_NAMES.get(ds, ds)}) — error mediano por banda de complejidad\n",
        f"Deciles de complejidad (`f1_complexity`) sobre el pool de puntos crudos de ambos "
        f"algoritmos ({n_bins} bandas solicitadas; pueden colapsar menos por empates en los "
        f"bordes, frecuente porque {100*(pooled['f1_complexity']==0).mean():.1f}% de los puntos "
        f"tienen complejidad exactamente 0).\n",
        f"| Banda de complejidad | {label_a} error mediano (n) | {label_b} error mediano (n) |",
        "|---|---|---|",
    ]
    for band in bands:
        m = median_err.loc[band].get("MOEACKF", float("nan"))
        s = median_err.loc[band].get("SNSGAII", float("nan"))
        nm = int(counts.loc[band].get("MOEACKF", 0))
        ns = int(counts.loc[band].get("SNSGAII", 0))
        m_str = f"{m:.4f} ({nm})" if nm else "— (0)"
        s_str = f"{s:.4f} ({ns})" if ns else "— (0)"
        lines.append(f"| {band} | {m_str} | {s_str} |")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    mids = [iv.mid for iv in bands]
    for algo, color in (("MOEACKF", "tab:blue"), ("SNSGAII", "tab:orange")):
        ys = [median_err.loc[b].get(algo, np.nan) for b in bands]
        ax.plot(mids, ys, marker="o", label=algo, color=color)
    ax.set_xlabel("Complejidad (centro de banda, frac. pesos activos)")
    ax.set_ylabel("Error de entrenamiento mediano")
    ax.set_title(f"Error por banda de complejidad — {lib.DS_NAMES.get(ds, ds)}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"ds{ds}_complexity_bands.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    lines.append(f"\n![Error por banda ds{ds}](ds{ds}_complexity_bands.png)\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results/normal_results"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("comparison/fronts"))
    parser.add_argument("--ds", type=int, default=None,
                         help="Procesar un único dataset (default: todos los encontrados)")
    parser.add_argument("--show", action="store_true",
                         help="Mostrar cada gráfico en una ventana interactiva (zoom/pan) "
                              "además de guardarlo como PNG")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = lib.discover_datasets(args.results_dir)
    if args.ds is not None:
        if args.ds not in datasets:
            print(f"ds{args.ds} no tiene resultados de ambos algoritmos en {args.results_dir}")
            return
        datasets = [args.ds]
    if not datasets:
        print(f"No se encontraron pares ds{{N}}_moeackf / ds{{N}}_snsgaii en {args.results_dir}")
        return

    summary_parts = ["# Frentes de Pareto — MOEACKF vs SNSGAII (normal_results)\n"]
    best_rows, cmetric_rows = [], []

    for ds in datasets:
        md1, rows1 = analyze_best_accuracy(args.results_dir, ds, args.alpha)
        (args.out / f"ds{ds}_best_accuracy.md").write_text(md1)
        pd.DataFrame(rows1).to_csv(args.out / f"ds{ds}_best_accuracy.csv", index=False)
        best_rows.extend(rows1)

        md2, rows2 = analyze_combined_front(args.results_dir, ds, args.out, show=args.show)
        (args.out / f"ds{ds}_combined_front.md").write_text(md2)
        pd.DataFrame(rows2).to_csv(args.out / f"ds{ds}_combined_front.csv", index=False)
        cmetric_rows.extend(rows2)

        md3 = analyze_complexity_bands(args.results_dir, ds, args.out, show=args.show)
        (args.out / f"ds{ds}_complexity_bands.md").write_text(md3)

        summary_parts += [md1, md2, md3]
        print(f"ds{ds}: listo")

    pd.DataFrame(best_rows).to_csv(args.out / "best_accuracy_summary.csv", index=False)
    pd.DataFrame(cmetric_rows).to_csv(args.out / "combined_front_summary.csv", index=False)
    (args.out / "summary.md").write_text("\n\n".join(summary_parts))
    print(f"Tablas y figuras guardadas en {args.out}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compara la convergencia de MOEACKF vs SNSGAII en results/normal_results — ángulo
"convergencia" de docs/metodologia_comparacion.md.

convergence.csv no registra FE por fila (solo `generation`, que NO es comparable entre
algoritmos: consumen distinto número de evaluaciones por generación). comparison_lib.approximate_fe
aproxima fe_approx = generation / generation_final_de_la_corrida * maxfe, asumiendo consumo
~constante de FE por generación dentro de cada corrida — se declara como aproximación, no FE exacto.

Dos técnicas:
  (a) Curvas de convergencia: HV mediano + banda P10-P90 vs. FE aproximado, ambos algoritmos
      superpuestos, por dataset (interpolación lineal de cada corrida sobre una grilla común).
  (b) Velocidad: FE aproximado hasta alcanzar 95% (indicador principal) y 90% (sensibilidad) del
      HV final de cada corrida, comparado pareado por seed.

Usage:
    python compare_convergence.py
    python compare_convergence.py --results-dir results/normal_results --out comparison/convergence
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import comparison_lib as lib


def plot_convergence_curve(results_dir: Path, ds: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"moeackf": "tab:blue", "snsgaii": "tab:orange"}
    for algo in lib.ALGOS:
        conv = pd.read_csv(lib.combo_dir(results_dir, ds, algo) / "convergence.csv")
        conv = lib.approximate_fe(conv)
        maxfe = float(conv["maxfe"].iloc[0])
        grid = np.linspace(0, maxfe, 100)
        curves = []
        for _, g in conv.groupby("seed"):
            g = g.sort_values("fe_approx")
            curves.append(np.interp(grid, g["fe_approx"], g["hv"]))
        curves = np.array(curves)
        median = np.median(curves, axis=0)
        p10 = np.percentile(curves, 10, axis=0)
        p90 = np.percentile(curves, 90, axis=0)
        ax.plot(grid, median, color=colors[algo], label=lib.ALGO_LABELS[algo], linewidth=1.6)
        ax.fill_between(grid, p10, p90, color=colors[algo], alpha=0.18)
    ax.set_xlabel("FE aproximado")
    ax.set_ylabel("HV (mediana y banda P10–P90 de 31 corridas)")
    ax.set_title(f"Convergencia — {lib.DS_NAMES.get(ds, ds)}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"ds{ds}_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_convergence_speed(results_dir: Path, ds: int, alpha: float) -> tuple[str, list[dict]]:
    speeds = {}
    for algo in lib.ALGOS:
        conv = pd.read_csv(lib.combo_dir(results_dir, ds, algo) / "convergence.csv")
        conv = lib.approximate_fe(conv)
        rows = []
        for seed, g in conv.groupby("seed"):
            rows.append({
                "seed": seed,
                "fe95": lib.fe_to_threshold(g, 0.95),
                "fe90": lib.fe_to_threshold(g, 0.90),
            })
        speeds[algo] = pd.DataFrame(rows)

    label_a, label_b = lib.ALGO_LABELS["moeackf"], lib.ALGO_LABELS["snsgaii"]
    md = ["### Velocidad de convergencia\n"]
    csv_rows: list[dict] = []

    x, y, _ = lib.align_by_seed(speeds["moeackf"], speeds["snsgaii"], "fe95",
                                 label_x="moeackf", label_y="snsgaii")
    res95 = lib.paired_comparison(x, y, label_a, label_b, alpha)
    md.append("**FE hasta 95% del HV final (indicador principal, entra en la corrección "
               "Holm-Bonferroni de la familia `principal_snn`):**\n")
    md.append(lib.render_comparison_markdown(res95, "FE hasta 95% del HV final"))
    csv_rows.extend(lib.render_comparison_csv_row(res95, ds, "convergence_speed", "fe95"))

    x90, y90, _ = lib.align_by_seed(speeds["moeackf"], speeds["snsgaii"], "fe90",
                                     label_x="moeackf", label_y="snsgaii")
    res90 = lib.paired_comparison(x90, y90, label_a, label_b, alpha)
    md.append("\n**FE hasta 90% del HV final (chequeo de sensibilidad; no se corrige aparte "
               "por su redundancia con el de 95% -- misma trayectoria, mismo run):**\n")
    md.append(lib.render_comparison_markdown(res90, "FE hasta 90% del HV final"))
    csv_rows.extend(lib.render_comparison_csv_row(res90, ds, "convergence_speed_sensitivity", "fe90"))

    return "\n".join(md), csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results/normal_results"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("comparison/convergence"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = lib.discover_datasets(args.results_dir)
    if not datasets:
        print(f"No se encontraron pares ds{{N}}_moeackf / ds{{N}}_snsgaii en {args.results_dir}")
        return

    summary_parts = ["# Convergencia — MOEACKF vs SNSGAII (normal_results)\n"]
    all_rows = []

    for ds in datasets:
        plot_convergence_curve(args.results_dir, ds, args.out)
        speed_md, rows = analyze_convergence_speed(args.results_dir, ds, args.alpha)
        full_md = (f"## Dataset {ds} ({lib.DS_NAMES.get(ds, ds)})\n\n"
                   f"### Curva de convergencia\n\n"
                   f"![Convergencia ds{ds}](ds{ds}_convergence.png)\n\n"
                   f"{speed_md}")
        (args.out / f"ds{ds}_convergence.md").write_text(full_md)
        pd.DataFrame(rows).to_csv(args.out / f"ds{ds}_convergence_speed.csv", index=False)
        all_rows.extend(rows)
        summary_parts.append(full_md)
        print(f"ds{ds}: listo")

    pd.DataFrame(all_rows).to_csv(args.out / "convergence_speed_summary.csv", index=False)
    (args.out / "summary.md").write_text("\n\n".join(summary_parts))
    print(f"Tablas y figuras guardadas en {args.out}/")


if __name__ == "__main__":
    main()

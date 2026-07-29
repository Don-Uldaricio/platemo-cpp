#!/usr/bin/env python3
"""Compara SNN (results/normal_results, C++) vs. ANN (results/MATLAB_RESULTS, PlatEMO/MATLAB
original) por dataset y por algoritmo — ángulo "plataforma" de docs/metodologia_comparacion.md.

Granularidad por algoritmo (MOEACKF-ANN vs MOEACKF-SNN, SNSGAII-ANN vs SNSGAII-SNN), no
agrupada: aísla la plataforma como único factor distinto entre las muestras comparadas.

Los CSV de MATLAB_RESULTS no tienen columna seed/run -- no hay señal de emparejamiento entre una
corrida MATLAB y una corrida C++ (RNG distintos, implementaciones distintas), así que se tratan
como muestras INDEPENDIENTES: Mann-Whitney U, no Wilcoxon signed-rank.

Tres indicadores por (dataset, algoritmo): HV, complejidad de la red de mejor accuracy, accuracy
de esa misma red. La "red de mejor accuracy" en SNN se deriva de fronts/*_front.csv con la misma
definición que ya trae precalculada MATLAB_RESULTS (ver compare_fronts.py).

Usage:
    python compare_platform.py
    python compare_platform.py --results-dir results/normal_results --matlab-dir results/MATLAB_RESULTS
"""

import argparse
from pathlib import Path

import pandas as pd

import comparison_lib as lib

LABEL_SNN = "SNN"
LABEL_ANN = "ANN-MATLAB"


def compare_platform_algo(results_dir: Path, matlab_dir: Path, ds: int, algo: str,
                           alpha: float) -> tuple[str, list[dict]]:
    seeds = set(lib.load_results_csv(results_dir, ds, algo, columns=("seed", "hv"))["seed"])
    snn_hv = lib.load_results_csv(results_dir, ds, algo, columns=("seed", "hv"))["hv"].to_numpy()
    nh = lib.nhidden_of(results_dir, ds, algo)
    fronts_dir = lib.combo_dir(results_dir, ds, algo) / "fronts"
    snn_best = lib.load_all_runs_best_accuracy(fronts_dir, algo, ds, nh, seeds)

    matlab_df = lib.load_matlab_run_csv(matlab_dir, ds, algo)

    angle = f"platform_{algo}"
    md = [f"### {lib.ALGO_LABELS[algo]}\n"]
    csv_rows: list[dict] = []

    res_hv = lib.unpaired_comparison(snn_hv, matlab_df["hv"].to_numpy(), LABEL_SNN, LABEL_ANN, alpha)
    md.append("**Hipervolumen (HV):**\n")
    md.append(lib.render_comparison_markdown(res_hv, "HV"))
    csv_rows.extend(lib.render_comparison_csv_row(res_hv, ds, angle, "hv"))

    res_cx = lib.unpaired_comparison(snn_best["f1_complexity"].to_numpy(),
                                      matlab_df["complexity"].to_numpy(), LABEL_SNN, LABEL_ANN, alpha)
    md.append("\n**Complejidad de la red de mejor accuracy (frac. pesos activos):**\n")
    md.append(lib.render_comparison_markdown(res_cx, "Complejidad"))
    csv_rows.extend(lib.render_comparison_csv_row(res_cx, ds, angle, "complexity"))

    res_acc = lib.unpaired_comparison(snn_best["accuracy"].to_numpy(),
                                       matlab_df["accuracy"].to_numpy(), LABEL_SNN, LABEL_ANN, alpha)
    md.append("\n**Accuracy de la red de mejor accuracy:**\n")
    md.append(lib.render_comparison_markdown(res_acc, "Accuracy"))
    csv_rows.extend(lib.render_comparison_csv_row(res_acc, ds, angle, "accuracy"))

    return "\n".join(md), csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results/normal_results"))
    parser.add_argument("--matlab-dir", type=Path, default=Path("results/MATLAB_RESULTS"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("comparison/platform"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = lib.discover_datasets(args.matlab_dir, suffix="_matlab")
    if not datasets:
        print(f"No se encontraron pares ds{{N}}_moeackf_matlab / ds{{N}}_snsgaii_matlab en {args.matlab_dir}")
        return

    summary_parts = ["# Plataforma: SNN (C++) vs. ANN (MATLAB/PlatEMO original)\n",
                      "Mann-Whitney U (muestras independientes -- no hay seed compartida entre "
                      "MATLAB y C++), por dataset y por algoritmo.\n"]
    all_rows = []

    for ds in datasets:
        md_parts = [f"## Dataset {ds} ({lib.DS_NAMES.get(ds, ds)})\n"]
        ds_rows = []
        for algo in lib.ALGOS:
            md, rows = compare_platform_algo(args.results_dir, args.matlab_dir, ds, algo, args.alpha)
            md_parts.append(md)
            ds_rows.extend(rows)
        full_md = "\n".join(md_parts)
        (args.out / f"ds{ds}_platform.md").write_text(full_md)
        pd.DataFrame(ds_rows).to_csv(args.out / f"ds{ds}_platform.csv", index=False)
        all_rows.extend(ds_rows)
        summary_parts.append(full_md)
        print(f"ds{ds}: listo")

    pd.DataFrame(all_rows).to_csv(args.out / "summary.csv", index=False)
    (args.out / "summary.md").write_text("\n\n".join(summary_parts))
    print(f"Tablas guardadas en {args.out}/")


if __name__ == "__main__":
    main()

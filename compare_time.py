#!/usr/bin/env python3
"""Compara tiempo de ejecución (time_s) de MOEACKF vs SNSGAII en results/time_experiments --
ángulo "tiempo" de docs/metodologia_comparacion.md.

A diferencia de normal_results/bootstrap_results (donde time_s se descarta por paralelismo no
controlado entre tandas de ejecución), time_experiments se ejecutó con hiperparámetros
compartidos entre algoritmos (sin params_table.sh) y bajo condiciones de hardware controladas
(mismo JOBS/OMP_THREADS, sin otros procesos compitiendo por CPU) -- por eso acá SÍ se reporta
la comparación de time_s como válida y directa. El HV de esta carpeta NO se usa (hiperparámetros
no afinados por algoritmo, no comparable con normal_results).

Pareado por seed (N=15, seeds 100-114), por dataset.

Usage:
    python compare_time.py
    python compare_time.py --results-dir results/time_experiments --out comparison/time
"""

import argparse
from pathlib import Path

import pandas as pd

import comparison_lib as lib


def compare_dataset_time(results_dir: Path, ds: int, alpha: float) -> dict:
    a = lib.load_results_csv(results_dir, ds, "moeackf", suffix="_time", columns=("seed", "time_s"))
    b = lib.load_results_csv(results_dir, ds, "snsgaii", suffix="_time", columns=("seed", "time_s"))
    x, y, common = lib.align_by_seed(a, b, "time_s", label_x="moeackf", label_y="snsgaii")
    result = lib.paired_comparison(x, y, lib.ALGO_LABELS["moeackf"], lib.ALGO_LABELS["snsgaii"], alpha)
    result["dataset"] = ds
    return result


def render_markdown(result: dict) -> str:
    ds = result["dataset"]
    name = lib.DS_NAMES.get(ds, f"dataset {ds}")
    label_a, label_b = result["labels"]
    lines = [f"## Dataset {ds} ({name})", "",
             lib.render_comparison_markdown(result, "time_s", unit_note="Unidad: segundos de reloj."), ""]
    if result["significant"]:
        faster = label_a if result[label_a]["median"] < result[label_b]["median"] else label_b
        lines.append(f"Diferencia significativa (α=0.05): **{faster}** es más rápido (menor "
                     f"tiempo mediano).")
    else:
        lines.append("Sin diferencia estadísticamente significativa (α=0.05) en tiempo de ejecución.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results/time_experiments"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("comparison/time"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = lib.discover_datasets(args.results_dir, suffix="_time")
    if not datasets:
        print(f"No se encontraron pares ds{{N}}_moeackf_time / ds{{N}}_snsgaii_time en {args.results_dir}")
        return

    summary_parts = [
        "# Comparación de tiempo de ejecución — MOEACKF vs SNSGAII (time_experiments)\n",
        "Wilcoxon signed-rank pareado por seed sobre time_s. Condiciones controladas de "
        "hardware e hiperparámetros compartidos entre algoritmos (ver "
        "docs/metodologia_comparacion.md) -- a diferencia de normal_results/bootstrap_results, "
        "acá la comparación de tiempos es válida.\n",
    ]
    all_rows = []

    for ds in datasets:
        result = compare_dataset_time(args.results_dir, ds, args.alpha)
        md = render_markdown(result)
        (args.out / f"ds{ds}_time_comparison.md").write_text(md)
        rows = lib.render_comparison_csv_row(result, ds, "time", "time_s")
        pd.DataFrame(rows).to_csv(args.out / f"ds{ds}_time_comparison.csv", index=False)
        all_rows.extend(rows)
        summary_parts.append(md)
        print(f"ds{ds}: listo")

    (args.out / "summary.md").write_text("\n".join(summary_parts))
    pd.DataFrame(all_rows).to_csv(args.out / "summary.csv", index=False)
    print(f"Tablas guardadas en {args.out}/")


if __name__ == "__main__":
    main()

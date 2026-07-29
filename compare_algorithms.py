#!/usr/bin/env python3
"""Compara MOEACKF vs SNSGAII por dataset usando hipervolumen (HV) — ángulo "principal" de
docs/metodologia_comparacion.md. Es uno de varios scripts compare_*.py que comparten lógica
en comparison_lib.py; ver ese módulo para el resto de los ángulos (frentes, convergencia,
plataforma ANN/SNN, bootstrap, tiempo) y apply_corrections.py para la corrección conjunta.

Implementa la metodología descrita en docs/metodologia_comparacion.md:
  - Comparación siempre por dataset por separado (nunca se mezclan datasets
    en un mismo test).
  - Diseño pareado por `seed` (la misma seed produce una corrida de cada
    algoritmo, así que el pareo controla la varianza atribuible al azar).
  - Prueba principal: Wilcoxon signed-rank test (pareado) sobre HV.
  - Tamaño del efecto: Â₁₂ de Vargha-Delaney, calculado directamente sobre
    los pares (proporción de pares donde un algoritmo supera al otro,
    empates cuentan 0.5).
  - Shapiro-Wilk se reporta solo como información complementaria: la
    metodología ya decidió usar pruebas no paramétricas de todas formas.

Nota: accuracy de test queda fuera de este análisis — es una limitación estructural de los
datos archivados (100% de las corridas usan arquitectura de 2 salidas WTA, que no exporta
*_spikes.csv), no un dato pendiente. Ver compare_fronts.py para el análisis de complejidad/error
a través de todo el frente, que es el sustituto de este indicador.

Usage:
    python compare_algorithms.py                       # results/normal_results (default)
    python compare_algorithms.py --results-dir results/normal_results
    python compare_algorithms.py --alpha 0.05
    python compare_algorithms.py --out comparison/hv
"""

import argparse
from pathlib import Path

import pandas as pd

import comparison_lib as lib


def compare_dataset(results_dir: Path, ds: int, alpha: float) -> dict:
    a = lib.load_results_csv(results_dir, ds, "moeackf", columns=("seed", "hv"))
    b = lib.load_results_csv(results_dir, ds, "snsgaii", columns=("seed", "hv"))
    x, y, common = lib.align_by_seed(a, b, "hv", label_x="moeackf", label_y="snsgaii")
    result = lib.paired_comparison(x, y, lib.ALGO_LABELS["moeackf"], lib.ALGO_LABELS["snsgaii"], alpha)
    result["dataset"] = ds
    return result


def render_markdown(result: dict) -> str:
    ds = result["dataset"]
    name = lib.DS_NAMES.get(ds, f"dataset {ds}")
    label_a, label_b = result["labels"]
    lines = [f"## Dataset {ds} ({name})", "", lib.render_comparison_markdown(result, "HV"), ""]
    if result["significant"]:
        winner = label_a if result[label_a]["median"] > result[label_b]["median"] else label_b
        lines.append(f"Diferencia significativa (α=0.05): **{winner}** tiene mayor HV mediano.")
    else:
        lines.append("Sin diferencia estadísticamente significativa (α=0.05).")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results/normal_results"),
                         help="Directorio con subcarpetas ds{N}_{algo}/results.csv "
                              "(default: results/normal_results)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Umbral de significancia (default: 0.05)")
    parser.add_argument("--out", type=Path, default=Path("comparison/hv"),
                         help="Directorio de salida para tablas (default: comparison/hv)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    datasets = lib.discover_datasets(args.results_dir)
    if not datasets:
        print(f"No se encontraron pares ds{{N}}_moeackf / ds{{N}}_snsgaii en {args.results_dir}")
        return

    print(f"Datasets encontrados: {datasets}")
    print("Indicador: HV (hipervolumen). Accuracy de test no disponible como indicador "
          "(limitación estructural, ver docs/metodologia_comparacion.md).\n")

    summary_sections = [
        "# Comparación MOEACKF vs SNSGAII — HV, resumen por dataset\n",
        "Prueba: Wilcoxon signed-rank pareado por seed sobre HV. "
        "Cada dataset se analiza por separado (no se combinan en un mismo test), "
        "según lo indicado en docs/metodologia_comparacion.md.\n",
    ]
    all_csv_rows = []

    for ds in datasets:
        result = compare_dataset(args.results_dir, ds, args.alpha)

        md = render_markdown(result)
        print(md)

        (args.out / f"ds{ds}_hv_comparison.md").write_text(md)

        rows = lib.render_comparison_csv_row(result, ds, "hv_main", "hv")
        pd.DataFrame(rows).to_csv(args.out / f"ds{ds}_hv_comparison.csv", index=False)
        all_csv_rows.extend(rows)

        summary_sections.append(md)

    (args.out / "summary.md").write_text("\n".join(summary_sections))
    pd.DataFrame(all_csv_rows).to_csv(args.out / "summary.csv", index=False)

    print(f"Tablas guardadas en {args.out}/")


if __name__ == "__main__":
    main()

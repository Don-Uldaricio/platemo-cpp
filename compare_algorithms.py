#!/usr/bin/env python3
"""Compara MOEACKF vs SNSGAII por dataset usando hipervolumen (HV).

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

Nota: accuracy de test queda fuera de este análisis porque no existen
archivos *_spikes.csv en los resultados actuales (ver docs/metodologia_
comparacion.md, sección "Por qué revisar train vs. test"). Solo se compara
HV hasta que se regeneren esos datos.

Usage:
    python compare_algorithms.py                       # results/ (default)
    python compare_algorithms.py --results-dir results
    python compare_algorithms.py --alpha 0.05
    python compare_algorithms.py --out results/comparison
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALGOS = ("moeackf", "snsgaii")
ALGO_LABELS = {"moeackf": "MOEACKF", "snsgaii": "SNSGAII"}
DS_NAMES = {1: "Statlog Australian", 2: "Climate", 3: "German", 4: "Sonar", 5: "Iris", 6: "Wine"}


def discover_datasets(results_dir: Path) -> list[int]:
    """Devuelve los números de dataset que tienen resultados de ambos algoritmos."""
    found = []
    for d in sorted(results_dir.glob("ds*_moeackf")):
        m = d.name[len("ds"):-len("_moeackf")]
        if not m.isdigit():
            continue
        ds = int(m)
        if (results_dir / f"ds{ds}_snsgaii" / "results.csv").exists():
            found.append(ds)
    return found


def load_hv(results_dir: Path, ds: int, algo: str) -> pd.DataFrame:
    csv_path = results_dir / f"ds{ds}_{algo}" / "results.csv"
    df = pd.read_csv(csv_path)
    return df[["seed", "hv"]].drop_duplicates(subset="seed")


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """Â₁₂ pareado: proporción de pares donde x > y (empates cuentan 0.5)."""
    wins = np.sum(x > y) + 0.5 * np.sum(x == y)
    return wins / len(x)


def iqr(a: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(a, 25)), float(np.percentile(a, 75))


def compare_dataset(results_dir: Path, ds: int, alpha: float) -> dict:
    a = load_hv(results_dir, ds, "moeackf")
    b = load_hv(results_dir, ds, "snsgaii")

    seeds_a, seeds_b = set(a["seed"]), set(b["seed"])
    common = sorted(seeds_a & seeds_b)
    if seeds_a != seeds_b:
        print(f"[ds{ds}] AVISO: seeds no coinciden entre algoritmos "
              f"(moeackf={len(seeds_a)}, snsgaii={len(seeds_b)}); "
              f"usando intersección N={len(common)}")

    a = a.set_index("seed").loc[common]["hv"].to_numpy()
    b = b.set_index("seed").loc[common]["hv"].to_numpy()
    n = len(common)

    shapiro_a = stats.shapiro(a) if n >= 3 else None
    shapiro_b = stats.shapiro(b) if n >= 3 else None

    wilcoxon = stats.wilcoxon(a, b) if n >= 1 else None
    a12 = vargha_delaney_a12(a, b)

    return {
        "dataset": ds,
        "n": n,
        "moeackf": {"median": float(np.median(a)), "iqr": iqr(a),
                    "shapiro_p": float(shapiro_a.pvalue) if shapiro_a else float("nan")},
        "snsgaii": {"median": float(np.median(b)), "iqr": iqr(b),
                    "shapiro_p": float(shapiro_b.pvalue) if shapiro_b else float("nan")},
        "wilcoxon_p": float(wilcoxon.pvalue) if wilcoxon else float("nan"),
        "a12": a12,
        "significant": bool(wilcoxon and wilcoxon.pvalue < alpha),
    }


def format_median_iqr(stats_dict: dict) -> str:
    lo, hi = stats_dict["iqr"]
    return f"{stats_dict['median']:.4f} [{lo:.4f}, {hi:.4f}]"


def render_markdown(result: dict) -> str:
    ds = result["dataset"]
    name = DS_NAMES.get(ds, f"dataset {ds}")
    sig = "*" if result["significant"] else ""
    lines = [
        f"## Dataset {ds} ({name})",
        "",
        f"N = {result['n']} corridas pareadas por seed.",
        "",
        "| Algoritmo | HV (mediana [IQR]) | p-valor (Wilcoxon) | Â₁₂ |",
        "|---|---|---|---|",
        f"| {ALGO_LABELS['moeackf']} | {format_median_iqr(result['moeackf'])} | "
        f"— | — |",
        f"| {ALGO_LABELS['snsgaii']} | {format_median_iqr(result['snsgaii'])} | "
        f"{result['wilcoxon_p']:.4g}{sig} | {result['a12']:.3f} |",
        "",
        f"Shapiro-Wilk p (informativo): MOEACKF={result['moeackf']['shapiro_p']:.4g}, "
        f"SNSGAII={result['snsgaii']['shapiro_p']:.4g}.",
        "",
    ]
    if result["significant"]:
        winner = "MOEACKF" if result["moeackf"]["median"] > result["snsgaii"]["median"] else "SNSGAII"
        lines.append(f"Diferencia significativa (α=0.05): **{winner}** tiene mayor HV mediano.")
    else:
        lines.append("Sin diferencia estadísticamente significativa (α=0.05).")
    lines.append("")
    return "\n".join(lines)


def render_csv_row(result: dict) -> list[dict]:
    rows = []
    for algo_key in ("moeackf", "snsgaii"):
        s = result[algo_key]
        rows.append({
            "dataset": result["dataset"],
            "algo": ALGO_LABELS[algo_key],
            "n": result["n"],
            "hv_median": s["median"],
            "hv_iqr_low": s["iqr"][0],
            "hv_iqr_high": s["iqr"][1],
            "shapiro_p": s["shapiro_p"],
            "wilcoxon_p": result["wilcoxon_p"] if algo_key == "snsgaii" else "",
            "a12": result["a12"] if algo_key == "snsgaii" else "",
            "significant_alpha_0.05": result["significant"] if algo_key == "snsgaii" else "",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                         help="Directorio con subcarpetas ds{N}_{algo}/results.csv (default: results)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Umbral de significancia (default: 0.05)")
    parser.add_argument("--out", type=Path, default=None,
                         help="Directorio de salida para tablas (default: <results-dir>/comparison)")
    args = parser.parse_args()

    out_dir = args.out or (args.results_dir / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(args.results_dir)
    if not datasets:
        print(f"No se encontraron pares ds{{N}}_moeackf / ds{{N}}_snsgaii en {args.results_dir}")
        return

    print(f"Datasets encontrados: {datasets}")
    print("Indicador: HV (hipervolumen). Accuracy de test no disponible "
          "(no existen *_spikes.csv) — ver docs/metodologia_comparacion.md.\n")

    summary_sections = [
        "# Comparación MOEACKF vs SNSGAII — resumen por dataset\n",
        "Prueba: Wilcoxon signed-rank pareado por seed sobre HV. "
        "Cada dataset se analiza por separado (no se combinan en un mismo test), "
        "según lo indicado en docs/metodologia_comparacion.md.\n",
        "**Accuracy de test: pendiente.** No existen archivos `*_spikes.csv` en los "
        "resultados actuales, por lo que este análisis cubre únicamente HV.\n",
    ]
    all_csv_rows = []

    for ds in datasets:
        result = compare_dataset(args.results_dir, ds, args.alpha)

        md = render_markdown(result)
        print(md)

        (out_dir / f"ds{ds}_hv_comparison.md").write_text(md)

        rows = render_csv_row(result)
        pd.DataFrame(rows).to_csv(out_dir / f"ds{ds}_hv_comparison.csv", index=False)
        all_csv_rows.extend(rows)

        summary_sections.append(md)

    (out_dir / "summary.md").write_text("\n".join(summary_sections))
    pd.DataFrame(all_csv_rows).to_csv(out_dir / "summary.csv", index=False)

    print(f"Tablas guardadas en {out_dir}/")


if __name__ == "__main__":
    main()

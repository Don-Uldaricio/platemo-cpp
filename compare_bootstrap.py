#!/usr/bin/env python3
"""Compara results/bootstrap_results (fitness evaluado por remuestreo bootstrap: 50 resamples
del 30% de los datos de entrenamiento por generación) vs. results/normal_results (evaluación
completa) -- ángulo "bootstrap" de docs/metodologia_comparacion.md.

Pareado por seed (ambos comparten seeds 100-130, mismo pipeline C++), por dataset y por
algoritmo (8 comparaciones: 4 datasets x 2 algoritmos). Indicador principal: HV. El
multiplicador de time_s se reporta de forma descriptiva (sin test formal -- es un efecto
esperado de promediar sobre 50 remuestreos en vez de una sola pasada).

Usage:
    python compare_bootstrap.py
    python compare_bootstrap.py --normal-dir results/normal_results --bootstrap-dir results/bootstrap_results
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import comparison_lib as lib


def compare_bootstrap_algo(normal_dir: Path, bootstrap_dir: Path, ds: int, algo: str,
                            alpha: float) -> tuple[str, list[dict]]:
    normal = lib.load_results_csv(normal_dir, ds, algo, columns=("seed", "hv", "time_s"))
    boot = lib.load_results_csv(bootstrap_dir, ds, algo, suffix="_bs",
                                 columns=("seed", "hv", "time_s"))

    x, y, common = lib.align_by_seed(normal, boot, "hv", label_x="normal", label_y="bootstrap")
    res = lib.paired_comparison(x, y, "Normal", "Bootstrap", alpha)

    t_normal = normal.set_index("seed").loc[common, "time_s"].to_numpy()
    t_boot = boot.set_index("seed").loc[common, "time_s"].to_numpy()
    ratio = float(np.median(t_boot) / np.median(t_normal))

    md = (f"### {lib.ALGO_LABELS[algo]}\n\n"
          f"{lib.render_comparison_markdown(res, 'HV')}\n\n"
          f"Tiempo mediano: normal = {np.median(t_normal):.1f} s, bootstrap = "
          f"{np.median(t_boot):.1f} s (×{ratio:.1f}; descriptivo, no se testea formalmente -- "
          f"costo esperado de promediar 50 remuestreos por generación en vez de una sola "
          f"pasada completa).\n")
    csv_rows = lib.render_comparison_csv_row(res, ds, f"bootstrap_{algo}", "hv")
    for row in csv_rows:
        row["time_s_ratio_bootstrap_over_normal"] = ratio
    return md, csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normal-dir", type=Path, default=Path("results/normal_results"))
    parser.add_argument("--bootstrap-dir", type=Path, default=Path("results/bootstrap_results"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("comparison/bootstrap"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = lib.discover_datasets(args.bootstrap_dir, suffix="_bs")
    if not datasets:
        print(f"No se encontraron pares ds{{N}}_moeackf_bs / ds{{N}}_snsgaii_bs en {args.bootstrap_dir}")
        return

    summary_parts = ["# Bootstrap vs. evaluación completa (normal_results)\n",
                      "Wilcoxon signed-rank pareado por seed sobre HV, por dataset y por "
                      "algoritmo.\n"]
    all_rows = []

    for ds in datasets:
        md_parts = [f"## Dataset {ds} ({lib.DS_NAMES.get(ds, ds)})\n"]
        ds_rows = []
        for algo in lib.ALGOS:
            md, rows = compare_bootstrap_algo(args.normal_dir, args.bootstrap_dir, ds, algo, args.alpha)
            md_parts.append(md)
            ds_rows.extend(rows)
        full_md = "\n".join(md_parts)
        (args.out / f"ds{ds}_bootstrap.md").write_text(full_md)
        pd.DataFrame(ds_rows).to_csv(args.out / f"ds{ds}_bootstrap.csv", index=False)
        all_rows.extend(ds_rows)
        summary_parts.append(full_md)
        print(f"ds{ds}: listo")

    pd.DataFrame(all_rows).to_csv(args.out / "summary.csv", index=False)
    (args.out / "summary.md").write_text("\n\n".join(summary_parts))
    print(f"Tablas guardadas en {args.out}/")


if __name__ == "__main__":
    main()

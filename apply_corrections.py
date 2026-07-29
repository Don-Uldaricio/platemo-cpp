#!/usr/bin/env python3
"""Aplica corrección de Holm-Bonferroni dentro de cada "familia" de tests relacionados, agregando
los p-valores crudos que ya calcularon los demás scripts compare_*.py -- ver
docs/metodologia_comparacion.md, sección de corrección por comparaciones múltiples. No recalcula
ninguna estadística nueva: solo lee los .csv que ya escribió cada script y corrige en conjunto.

Familias (la corrección nunca cruza familias -- cada una responde una pregunta distinta):
  - principal_snn (por dataset): HV, complejidad/error/tamaño-de-frente de la mejor red,
    velocidad de convergencia a 95%. 5 miembros x 4 datasets = 20 tests.
  - plataforma (por dataset x algoritmo): HV, complejidad, accuracy (SNN vs ANN-MATLAB).
    3 miembros x 8 grupos (4 datasets x 2 algoritmos) = 24 tests.
  - bootstrap (por dataset): HV bootstrap-vs-normal de MOEACKF y de SNSGAII por separado.
    2 miembros x 4 datasets = 8 tests.
  - tiempo (por dataset): time_s. 1 miembro x 4 datasets = 4 tests (sin corrección real, m=1).

Total esperado: 56 tests en 20 familias.

FE a 90% (convergence_speed_sensitivity) y el C-metric de frentes combinados (sin p-valor, es
descriptivo) quedan fuera de toda familia -- no se corrigen.

Usage:
    python apply_corrections.py
    python apply_corrections.py --alpha 0.05 --out comparison
"""

import argparse
from pathlib import Path

import pandas as pd

import comparison_lib as lib

SOURCES = [
    "comparison/hv/summary.csv",
    "comparison/fronts/best_accuracy_summary.csv",
    "comparison/convergence/convergence_speed_summary.csv",
    "comparison/platform/summary.csv",
    "comparison/bootstrap/summary.csv",
    "comparison/time/summary.csv",
]

FAMILY_ANGLES_PRINCIPAL = {"hv_main", "fronts_best_accuracy", "convergence_speed"}


def family_key(row: pd.Series):
    angle = row["angle"]
    ds = row["dataset"]
    if angle in FAMILY_ANGLES_PRINCIPAL:
        return f"principal_snn|ds{ds}"
    if angle.startswith("platform_"):
        return f"plataforma|ds{ds}|{angle}"
    if angle.startswith("bootstrap_"):
        return f"bootstrap|ds{ds}"
    if angle == "time":
        return f"tiempo|ds{ds}"
    return None  # convergence_speed_sensitivity, fronts_combined: fuera de toda familia


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=Path("."),
                         help="Directorio raíz desde donde resolver comparison/... (default: .)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("comparison"))
    args = parser.parse_args()

    frames = []
    for rel_path in SOURCES:
        path = args.base / rel_path
        if not path.exists():
            print(f"AVISO: no se encontró {path}, se omite")
            continue
        df = pd.read_csv(path)
        df["source_file"] = rel_path
        frames.append(df)
    if not frames:
        print("No se encontró ningún summary.csv de compare_*.py. Correr esos scripts primero.")
        return
    all_df = pd.concat(frames, ignore_index=True)

    # Solo las filas con p_value real (la fila "label_y" de cada comparación pareada/no pareada
    # trae el resumen del test; la fila "label_x" solo trae estadística descriptiva).
    tested = all_df[pd.to_numeric(all_df["p_value"], errors="coerce").notna()].copy()
    tested["p_value"] = tested["p_value"].astype(float)
    tested["family"] = tested.apply(family_key, axis=1)

    in_family = tested[tested["family"].notna()].copy()
    out_family = tested[tested["family"].isna()].copy()

    adjusted_parts = []
    for _, group in in_family.groupby("family"):
        group = group.copy()
        group["p_holm"] = lib.holm_bonferroni(group["p_value"].tolist())
        group["significant_holm"] = group["p_holm"] < args.alpha
        adjusted_parts.append(group)
    corrected = pd.concat(adjusted_parts, ignore_index=True) if adjusted_parts else in_family.copy()

    out_family["p_holm"] = float("nan")
    out_family["significant_holm"] = out_family["significant_alpha_0.05"]

    final = pd.concat([corrected, out_family], ignore_index=True)
    final = final.sort_values(["family", "dataset", "angle", "metric"], na_position="last")

    cols = ["family", "dataset", "angle", "metric", "group", "n", "test", "p_value", "p_holm",
            "a12", "significant_alpha_0.05", "significant_holm", "source_file"]
    final = final[[c for c in cols if c in final.columns]]

    args.out.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out / "corrections_summary.csv", index=False)

    lines = [
        "# Corrección por comparaciones múltiples (Holm-Bonferroni) — veredicto final\n",
        "Corrección aplicada DENTRO de cada familia (nunca entre familias). Ver "
        "docs/metodologia_comparacion.md para la definición de cada familia. Las filas con "
        "familia vacía (FE a 90%, C-metric) son chequeos de sensibilidad/descriptivos fuera de "
        "toda familia y no llevan p_holm.\n",
        "| Familia | Dataset | Ángulo | Métrica | p crudo | p Holm | Â₁₂ | Sig. (crudo) | Sig. (Holm) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in final.iterrows():
        fam = row.get("family")
        fam_str = fam if isinstance(fam, str) else "—"
        p_holm = f"{row['p_holm']:.4g}" if pd.notna(row["p_holm"]) else "—"
        a12 = f"{row['a12']:.3f}" if pd.notna(row.get("a12")) else "—"
        lines.append(f"| {fam_str} | ds{row['dataset']} | {row['angle']} | {row['metric']} | "
                     f"{row['p_value']:.4g} | {p_holm} | {a12} | "
                     f"{row['significant_alpha_0.05']} | {row['significant_holm']} |")
    (args.out / "corrections_summary.md").write_text("\n".join(lines))

    n_families = in_family["family"].nunique()
    print(f"Familias: {n_families}. Tests corregidos: {len(in_family)}. "
          f"Tests fuera de familia (sensibilidad/descriptivo): {len(out_family)}.")
    print(f"Guardado en {args.out}/corrections_summary.{{md,csv}}")


if __name__ == "__main__":
    main()

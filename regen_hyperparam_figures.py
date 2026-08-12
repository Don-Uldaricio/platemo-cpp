#!/usr/bin/env python3
"""
Regenera las figuras de convergencia de hiperparametros
(seccion "Resultados de la busqueda de hiperparametros" de la tesis)
con tipografia mas grande, a partir de los estudios Optuna originales.

Estas figuras son deterministas (grafican los valores objetivo reales
de cada trial), asi que se recalculan directamente desde los .db.
Las figuras de IMPORTANCIA (fANOVA) son otra historia: el evaluador
de Optuna usa un RandomForest sin semilla fija, por lo que volver a
correrlo da valores distintos a los ya publicados. Esas se regeneran
aparte con regen_importancia_figures.py, reutilizando los valores ya
publicados en la tesis en vez de recalcular el fANOVA.

Uso:
    python regen_hyperparam_figures.py [--out DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import plot_optimization_history

optuna.logging.set_verbosity(optuna.logging.WARNING)

# db fuente de cada dataset (confirmado contra los best-HV citados en la tesis).
DB_SOURCES = {
    1: Path("/home/nez/Downloads/bo_results/20260713_130616"),
    2: Path("/home/nez/Downloads/bo_results/20260625_124245"),
    3: Path("/home/nez/Downloads/bo_results/20260714_081329"),
    4: Path("/home/nez/Downloads/bo_results/20260715_043758"),
}

ALGOS = ["MOEACKF", "SNSGAII"]

DEFAULT_OUT = Path(__file__).parent.parent / "tesis-latex" / "images"

# Tamano de fuente base (default de matplotlib/ggplot es 10; se sube ~30%).
plt.rcParams.update({
    "font.size": 13,
})


def load_study(db_path: Path):
    study_name = db_path.stem
    return optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for ds, src_dir in DB_SOURCES.items():
        for algo in ALGOS:
            db_path = src_dir / f"snn_bo_{algo}_ds{ds}.db"
            if not db_path.exists():
                print(f"  SKIP {db_path}: no existe")
                continue
            study = load_study(db_path)

            ax = plot_optimization_history(study)
            ax.set_title(f"Convergencia HV medio — ds{ds} — {algo}")
            fig = ax.get_figure()
            fig.tight_layout()
            out = args.out / f"convergencia_{algo}_ds{ds}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)
            print(f"  guardado: {out}")


if __name__ == "__main__":
    main()

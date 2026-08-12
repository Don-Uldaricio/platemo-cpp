#!/usr/bin/env python3
"""
Re-renderiza las figuras "importancia_<ALGO>_ds<N>.png" (fANOVA) con
tipografia mas grande, SIN recalcular el fANOVA (el evaluador de Optuna
usa un RandomForest sin semilla fija, asi que volver a correrlo da
valores distintos a los ya publicados en la tesis).

En su lugar, se reconstruye cada grafico usando los mismos valores que
ya estan impresos en las figuras actuales (leidos de las imagenes
existentes), reproduciendo el mismo codigo de dibujo que usa
optuna.visualization.matplotlib.plot_param_importances
(ggplot style, mismos colores/leyenda/formato de etiquetas), solo que
con fuente mas grande. No se altera ningun numero ya publicado.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent.parent / "tesis-latex" / "images"

plt.rcParams.update({
    "font.size": 13,
})

# Valores leidos de las figuras ya publicadas (orden = de mayor a menor
# importancia, igual que en las imagenes actuales). Los valores con
# etiqueta "<0.01" no tienen precision original; se usan placeholders
# pequenos y decrecientes solo para preservar el orden y el largo de
# barra casi nulo que ya se ve en las figuras actuales.

def below_001(n):
    return [0.0095 - 0.0008 * i for i in range(n)]


DATA = {
    ("MOEACKF", 1): [
        ("wScale", 0.94), ("disC", 0.02), ("disM", 0.01),
        *zip(["sLower", "dt", "extra_eval", "disSM", "sGap",
              "encoding_duration", "proSM", "proM", "architecture"],
             below_001(9)),
    ],
    ("SNSGAII", 1): [
        ("wScale", 0.48), ("sLower", 0.44), ("sGap", 0.02), ("dt", 0.01), ("proM", 0.01),
        *zip(["proSM", "disM", "disSM", "extra_eval", "encoding_duration",
              "disC", "architecture"],
             below_001(7)),
    ],
    ("MOEACKF", 2): [
        ("architecture", 0.95), ("wScale", 0.04),
        *zip(["disC", "disSM", "proSM", "extra_eval", "sLower", "proM",
              "disM", "dt", "encoding_duration", "sGap"],
             below_001(10)),
    ],
    ("SNSGAII", 2): [
        ("architecture", 0.71), ("wScale", 0.23), ("proSM", 0.02),
        *zip(["proM", "extra_eval", "dt", "sGap", "encoding_duration",
              "sLower", "disM", "disSM", "disC"],
             below_001(9)),
    ],
    ("MOEACKF", 3): [
        ("wScale", 0.76), ("disM", 0.10), ("disSM", 0.05), ("sLower", 0.05),
        *zip(["architecture", "extra_eval", "sGap", "proM", "proSM",
              "encoding_duration", "dt", "disC"],
             below_001(8)),
    ],
    ("SNSGAII", 3): [
        ("wScale", 0.86), ("sLower", 0.05), ("disC", 0.03), ("dt", 0.02), ("proSM", 0.01),
        *zip(["disSM", "architecture", "proM", "extra_eval", "disM",
              "sGap", "encoding_duration"],
             below_001(7)),
    ],
    ("MOEACKF", 4): [
        ("architecture", 0.38), ("wScale", 0.20), ("dt", 0.09), ("disM", 0.08),
        ("disSM", 0.07), ("encoding_duration", 0.04), ("disC", 0.04), ("sGap", 0.03),
        ("proM", 0.02), ("sLower", 0.02), ("extra_eval", 0.01), ("proSM", 0.01),
    ],
    ("SNSGAII", 4): [
        ("sLower", 0.42), ("wScale", 0.20), ("disC", 0.10), ("proM", 0.05),
        ("dt", 0.05), ("disM", 0.05), ("proSM", 0.03), ("architecture", 0.02),
        ("sGap", 0.02), ("disSM", 0.02), ("encoding_duration", 0.02), ("extra_eval", 0.01),
    ],
}


def label_for(val: float) -> str:
    return f"{val:.2f}" if val >= 0.01 else "<0.01"


def plot_importances(param_names, importance_values, title):
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.set_title(title, loc="left")
    ax.set_xlabel("Hyperparameter Importance")
    ax.set_ylabel("Hyperparameter")

    # optuna dibuja de abajo hacia arriba con pos creciente; como ya
    # tenemos el orden top-to-bottom (mayor a menor), lo invertimos.
    param_names = list(reversed(param_names))
    importance_values = list(reversed(importance_values))
    pos = np.arange(len(param_names))

    ax.barh(pos, importance_values, height=0.8, align="center",
            label="Objective Value", color=plt.get_cmap("tab20c")(0))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for idx, val in enumerate(importance_values):
        text = ax.text(val, idx, label_for(val), va="center")
        bbox = text.get_window_extent(renderer).transformed(ax.transData.inverted())
        _, plot_xmax = ax.get_xlim()
        if bbox.xmax > plot_xmax:
            ax.set_xlim(xmax=1.05 * bbox.xmax)

    ax.set_yticks(pos, param_names)
    ax.legend(loc="best")
    return fig


def main():
    for (algo, ds), rows in DATA.items():
        param_names = [r[0] for r in rows]
        importance_values = [r[1] for r in rows]
        title = f"Importancia de hiperparametros — ds{ds} — {algo}"
        fig = plot_importances(param_names, importance_values, title)
        fig.tight_layout()
        out = OUT_DIR / f"importancia_{algo}_ds{ds}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"  guardado: {out}")


if __name__ == "__main__":
    main()

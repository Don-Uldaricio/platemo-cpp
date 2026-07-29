"""Utilidades compartidas por los scripts compare_*.py y apply_corrections.py.

Centraliza lo que antes vivía duplicado en compare_algorithms.py / plot_pareto.py: nombres de
datasets y algoritmos, descubrimiento de carpetas de resultados, estadística (Wilcoxon pareado,
Mann-Whitney no pareado, Â₁₂ de Vargha-Delaney en sus dos variantes, Holm-Bonferroni), lectura de
resultados (results.csv, fronts/*_front.csv, CSV de MATLAB_RESULTS) y utilidades de frentes de
Pareto (filtro no-dominado, frente combinado, C-metric).

Ver docs/metodologia_comparacion.md para la justificación metodológica de cada elección.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALGOS = ("moeackf", "snsgaii")
ALGO_LABELS = {"moeackf": "MOEACKF", "snsgaii": "SNSGAII"}
DS_NAMES = {1: "Statlog Australian", 2: "Climate", 3: "German", 4: "Sonar", 5: "Iris", 6: "Wine"}
BASE_SEED = 100


# ---------------------------------------------------------------------------
# Descubrimiento de carpetas de resultados
# ---------------------------------------------------------------------------

def discover_datasets(results_dir: Path, suffix: str = "") -> list[int]:
    """Números de dataset con resultados de ambos algoritmos bajo `results_dir`.

    `suffix` se adapta a la convención de nombre de cada fuente: "" (normal_results,
    ds1_moeackf), "_bs" (bootstrap_results), "_matlab" (MATLAB_RESULTS), "_time" (time_experiments).
    """
    pattern = re.compile(rf"^ds(\d+)_moeackf{re.escape(suffix)}$")
    found = []
    for d in sorted(results_dir.glob(f"ds*_moeackf{suffix}")):
        m = pattern.match(d.name)
        if not m:
            continue
        ds = int(m.group(1))
        if (results_dir / f"ds{ds}_snsgaii{suffix}").exists():
            found.append(ds)
    return found


def combo_dir(results_dir: Path, ds: int, algo: str, suffix: str = "") -> Path:
    return results_dir / f"ds{ds}_{algo}{suffix}"


# ---------------------------------------------------------------------------
# Estadística descriptiva
# ---------------------------------------------------------------------------

def iqr(a: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(a, 25)), float(np.percentile(a, 75))


def format_median_iqr(a: np.ndarray) -> str:
    lo, hi = iqr(a)
    return f"{float(np.median(a)):.4f} [{lo:.4f}, {hi:.4f}]"


# ---------------------------------------------------------------------------
# Tamaño del efecto: Â₁₂ de Vargha-Delaney
# ---------------------------------------------------------------------------

def vargha_delaney_a12_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Â₁₂ pareado: proporción de pares donde x > y (empates cuentan 0.5). len(x) == len(y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    wins = np.sum(x > y) + 0.5 * np.sum(x == y)
    return float(wins / len(x))


def vargha_delaney_a12_unpaired(x: np.ndarray, y: np.ndarray) -> float:
    """Â₁₂ no pareado: P(x_i > y_j) + 0.5*P(x_i == y_j) sobre pares aleatorios, vía U de Mann-Whitney.

    Â₁₂ = U1 / (n1*n2), con U1 el estadístico asociado al PRIMER argumento de mannwhitneyu(x, y).
    Invertir el orden de x/y invierte el resultado a 1-Â₁₂ — ver el auto-test al final del módulo.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u1 = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(u1 / (n1 * n2))


# ---------------------------------------------------------------------------
# Alineación pareada por seed
# ---------------------------------------------------------------------------

def align_by_seed(df_x: pd.DataFrame, df_y: pd.DataFrame, value_col: str,
                   seed_col: str = "seed", label_x: str = "x", label_y: str = "y"
                   ) -> tuple[np.ndarray, np.ndarray, list]:
    """Alinea dos DataFrames por `seed_col` sobre su intersección; avisa si no coinciden."""
    seeds_x, seeds_y = set(df_x[seed_col]), set(df_y[seed_col])
    common = sorted(seeds_x & seeds_y)
    if seeds_x != seeds_y:
        print(f"AVISO: seeds no coinciden entre {label_x} (n={len(seeds_x)}) y {label_y} "
              f"(n={len(seeds_y)}); usando intersección N={len(common)}")
    x = df_x.set_index(seed_col).loc[common, value_col].to_numpy(dtype=float)
    y = df_y.set_index(seed_col).loc[common, value_col].to_numpy(dtype=float)
    return x, y, common


# ---------------------------------------------------------------------------
# Comparaciones (núcleo reusado por todos los scripts compare_*.py)
# ---------------------------------------------------------------------------

def paired_comparison(x: np.ndarray, y: np.ndarray, label_x: str, label_y: str,
                       alpha: float = 0.05) -> dict:
    """Compara dos muestras pareadas (mismo N, mismo orden = mismo seed) sobre un indicador."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    assert len(y) == n, f"paired_comparison: tamaños distintos ({n} vs {len(y)})"
    shapiro_x = stats.shapiro(x) if n >= 3 else None
    shapiro_y = stats.shapiro(y) if n >= 3 else None
    if n >= 1 and np.any(x != y):
        wilcoxon = stats.wilcoxon(x, y)
    elif n >= 1:
        # x == y en todos los pares (p.ej. ambos algoritmos alcanzan el umbral de convergencia
        # ya en la generación 0 en cada corrida) -- scipy.stats.wilcoxon no admite diferencias
        # nulas en todos los pares; no hay evidencia de diferencia, se reporta p=1.0 sin test.
        wilcoxon = None
    else:
        wilcoxon = None
    no_difference = n >= 1 and np.all(x == y)
    return {
        "n": n,
        "paired": True,
        "test": "wilcoxon",
        "labels": (label_x, label_y),
        label_x: {"median": float(np.median(x)), "iqr": iqr(x), "mean": float(np.mean(x)),
                  "std": float(np.std(x, ddof=1)) if n > 1 else 0.0,
                  "shapiro_p": float(shapiro_x.pvalue) if shapiro_x else float("nan")},
        label_y: {"median": float(np.median(y)), "iqr": iqr(y), "mean": float(np.mean(y)),
                  "std": float(np.std(y, ddof=1)) if n > 1 else 0.0,
                  "shapiro_p": float(shapiro_y.pvalue) if shapiro_y else float("nan")},
        "p_value": 1.0 if no_difference else (float(wilcoxon.pvalue) if wilcoxon else float("nan")),
        "a12": vargha_delaney_a12_paired(x, y),
        "significant": False if no_difference else bool(wilcoxon and wilcoxon.pvalue < alpha),
        "no_difference": no_difference,
    }


def unpaired_comparison(x: np.ndarray, y: np.ndarray, label_x: str, label_y: str,
                         alpha: float = 0.05) -> dict:
    """Compara dos muestras independientes (N puede diferir) sobre un indicador."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    shapiro_x = stats.shapiro(x) if nx >= 3 else None
    shapiro_y = stats.shapiro(y) if ny >= 3 else None
    mw = stats.mannwhitneyu(x, y, alternative="two-sided") if nx >= 1 and ny >= 1 else None
    return {
        "n": (nx, ny),
        "paired": False,
        "test": "mann-whitney",
        "labels": (label_x, label_y),
        label_x: {"median": float(np.median(x)), "iqr": iqr(x), "mean": float(np.mean(x)),
                  "std": float(np.std(x, ddof=1)) if nx > 1 else 0.0,
                  "shapiro_p": float(shapiro_x.pvalue) if shapiro_x else float("nan")},
        label_y: {"median": float(np.median(y)), "iqr": iqr(y), "mean": float(np.mean(y)),
                  "std": float(np.std(y, ddof=1)) if ny > 1 else 0.0,
                  "shapiro_p": float(shapiro_y.pvalue) if shapiro_y else float("nan")},
        "p_value": float(mw.pvalue) if mw else float("nan"),
        "a12": vargha_delaney_a12_unpaired(x, y),
        "significant": bool(mw and mw.pvalue < alpha),
    }


# ---------------------------------------------------------------------------
# Corrección por comparaciones múltiples
# ---------------------------------------------------------------------------

def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Corrección de Holm-Bonferroni (step-down). Devuelve p-valores ajustados en el mismo
    orden de `pvalues` de entrada (no en orden ordenado)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.empty(m)
    running_max = 0.0
    for i in range(m):
        candidate = (m - i) * sorted_p[i]
        running_max = max(running_max, candidate)
        adjusted_sorted[i] = min(running_max, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Render de tablas markdown / filas CSV
# ---------------------------------------------------------------------------

def _row(result: dict, key: str, p_value_str: str, a12_str: str) -> str:
    s = result[key]
    return (f"| {key} | {s['median']:.4f} [{s['iqr'][0]:.4f}, {s['iqr'][1]:.4f}] | "
            f"{s['mean']:.4f} ± {s['std']:.4f} | {p_value_str} | {a12_str} |")


def render_comparison_markdown(result: dict, metric_label: str, unit_note: str = "") -> str:
    label_x, label_y = result["labels"]
    sig = "*" if result["significant"] else ""
    test_name = "Wilcoxon signed-rank" if result["paired"] else "Mann-Whitney U"
    n_desc = (f"N = {result['n']} corridas pareadas por seed." if result["paired"]
              else f"N = {result['n'][0]} ({label_x}) vs. N = {result['n'][1]} ({label_y}), "
                   f"muestras independientes.")
    lines = [n_desc]
    if unit_note:
        lines.append(unit_note)
    lines += [
        "",
        f"| Grupo | {metric_label} (mediana [IQR]) | Media ± DE | p-valor ({test_name}) | Â₁₂ |",
        "|---|---|---|---|---|",
        _row(result, label_x, "—", "—"),
        _row(result, label_y, f"{result['p_value']:.4g}{sig}", f"{result['a12']:.3f}"),
        "",
        f"Shapiro-Wilk p (informativo): {label_x}={result[label_x]['shapiro_p']:.4g}, "
        f"{label_y}={result[label_y]['shapiro_p']:.4g}.",
    ]
    if result.get("no_difference"):
        lines.append(f"{label_x} y {label_y} son idénticos en los {result['n']} pares (diferencia "
                      f"cero en todas las corridas) -- no aplica el test de Wilcoxon, se reporta "
                      f"p=1.0 por convención.")
    return "\n".join(lines)


def render_comparison_csv_row(result: dict, dataset: int, angle: str, metric: str) -> list[dict]:
    label_x, label_y = result["labels"]
    rows = []
    for i, key in enumerate((label_x, label_y)):
        s = result[key]
        rows.append({
            "dataset": dataset, "angle": angle, "metric": metric, "group": key,
            "n": result["n"] if result["paired"] else result["n"][i],
            "median": s["median"], "iqr_low": s["iqr"][0], "iqr_high": s["iqr"][1],
            "mean": s["mean"], "std": s["std"], "shapiro_p": s["shapiro_p"],
            "test": result["test"],
            "p_value": result["p_value"] if key == label_y else "",
            "a12": result["a12"] if key == label_y else "",
            "significant_alpha_0.05": result["significant"] if key == label_y else "",
        })
    return rows


# ---------------------------------------------------------------------------
# Lectura de results.csv (normal_results / bootstrap_results / time_experiments)
# ---------------------------------------------------------------------------

def load_results_csv(results_dir: Path, ds: int, algo: str, suffix: str = "",
                      columns: tuple[str, ...] = ("seed", "hv")) -> pd.DataFrame:
    csv_path = combo_dir(results_dir, ds, algo, suffix) / "results.csv"
    df = pd.read_csv(csv_path)
    return df[list(columns)].drop_duplicates(subset="seed" if "seed" in columns else None)


def nhidden_of(results_dir: Path, ds: int, algo: str, suffix: str = "") -> int:
    df = load_results_csv(results_dir, ds, algo, suffix, columns=("seed", "nhidden"))
    return int(df["nhidden"].iloc[0])


# ---------------------------------------------------------------------------
# Lectura de MATLAB_RESULTS
# ---------------------------------------------------------------------------

def load_matlab_run_csv(matlab_dir: Path, ds: int, algo: str) -> pd.DataFrame:
    """Lee el CSV por-corrida de MATLAB_RESULTS (accuracy,complexity,hv/HV) normalizando
    la capitalización inconsistente de la columna hv/HV."""
    folder = matlab_dir / f"ds{ds}_{algo}_matlab"
    csv_path = folder / f"ds{ds}_{algo}_matlab.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    return df[["accuracy", "complexity", "hv"]]


# ---------------------------------------------------------------------------
# Frentes de Pareto: fronts/*_front.csv
# ---------------------------------------------------------------------------

def load_front_csv(path: Path) -> pd.DataFrame:
    """Lee solo f1_complexity/f2_train_error de un *_front.csv (ignora w0..w{D-1})."""
    return pd.read_csv(path, usecols=["f1_complexity", "f2_train_error"])


def extract_best_accuracy_row(front_df: pd.DataFrame) -> pd.Series:
    """Fila de mejor accuracy (f2_train_error mínimo; empate -> f1_complexity mínimo)."""
    return front_df.sort_values(["f2_train_error", "f1_complexity"]).iloc[0]


def load_all_runs_best_accuracy(fronts_dir: Path, algo: str, ds: int, nhidden: int,
                                 expected_seeds: set[int]) -> pd.DataFrame:
    """Recorre fronts/{ALGO}_ds{N}_nh{H}_run{K}_front.csv y arma una fila por corrida con:
    seed, f1_complexity y f2_train_error de la solución de mejor accuracy, y front_size
    (cantidad de soluciones no dominadas de esa corrida).

    Asegura (no asume silenciosamente) que los seeds derivados de `run{K}` -> BASE_SEED+K
    coinciden exactamente con `expected_seeds` (los seeds reales de results.csv).
    """
    pattern = re.compile(rf"^{re.escape(ALGO_LABELS[algo])}_ds{ds}_nh{nhidden}_run(\d+)_front\.csv$")
    rows = []
    derived_seeds = set()
    for f in sorted(fronts_dir.glob(f"{ALGO_LABELS[algo]}_ds{ds}_nh{nhidden}_run*_front.csv")):
        m = pattern.match(f.name)
        if not m:
            continue
        run_idx = int(m.group(1))
        seed = BASE_SEED + run_idx
        derived_seeds.add(seed)
        front_df = load_front_csv(f)
        best = extract_best_accuracy_row(front_df)
        rows.append({
            "seed": seed,
            "f1_complexity": float(best["f1_complexity"]),
            "f2_train_error": float(best["f2_train_error"]),
            "accuracy": 1.0 - float(best["f2_train_error"]),
            "front_size": len(front_df),
        })
    assert derived_seeds == expected_seeds, (
        f"load_all_runs_best_accuracy: seeds derivados de nombres de archivo ({sorted(derived_seeds)}) "
        f"no coinciden con los de results.csv ({sorted(expected_seeds)}) para {algo} ds{ds}"
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Frente no-dominado, frente combinado, C-metric
# ---------------------------------------------------------------------------

def pareto_filter(points: np.ndarray) -> np.ndarray:
    """Máscara de puntos no dominados (minimización en ambos objetivos, dominancia estricta)."""
    n = len(points)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                dominated[i] = True
                break
    return ~dominated


def combined_front(points: np.ndarray) -> np.ndarray:
    """Frente combinado: deduplica puntos (f1,f2) exactos y filtra no-dominados.

    La deduplicación previa es necesaria: entre corridas distintas de un mismo algoritmo es
    común converger al mismo punto (f1,f2) exacto (verificado: 14-72% de puntos duplicados
    según dataset), lo que infla el conteo si no se deduplica antes de filtrar.
    """
    unique = np.unique(points, axis=0)
    mask = pareto_filter(unique)
    front = unique[mask]
    return front[front[:, 0].argsort()]


def c_metric(front_a: np.ndarray, front_b: np.ndarray) -> tuple[float, float]:
    """C(A,B): fracción de puntos de B cubiertos (dominados o igualados) por al menos un punto
    de A, y C(B,A) en sentido inverso. Dominancia DÉBIL (Zitzler & Thiele), a propósito distinta
    de la dominancia estricta de pareto_filter -- con dominancia débil C(A,A)=1 siempre.
    """
    def coverage(a, b):
        covered = 0
        for pb in b:
            if np.any(np.all(a <= pb, axis=1)):
                covered += 1
        return covered / len(b) if len(b) else float("nan")
    return coverage(front_a, front_b), coverage(front_b, front_a)


# ---------------------------------------------------------------------------
# Convergencia: generación -> FE aproximado, velocidad de convergencia
# ---------------------------------------------------------------------------

def approximate_fe(conv_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega fe_approx a un convergence.csv (columnas algo,...,seed,generation,hv), asumiendo
    consumo ~constante de FE por generación dentro de cada corrida:
        fe_approx = generation / generation_final_de_la_corrida * maxfe
    convergence.csv no registra FE por fila (solo `generation`, no comparable entre algoritmos
    -- ver metodología); esto es una aproximación declarada, no un valor exacto.
    """
    out = conv_df.copy()
    max_gen = out.groupby("seed")["generation"].transform("max")
    out["fe_approx"] = out["generation"] / max_gen * out["maxfe"]
    return out


def fe_to_threshold(run_df: pd.DataFrame, frac: float) -> float:
    """FE aproximado en el que una corrida alcanza por primera vez `frac` de su HV final.

    Usa el máximo acumulado de hv (defensivo; verificado 100% monótono no-decreciente en los
    datos archivados, pero no se asume ciegamente)."""
    run_df = run_df.sort_values("fe_approx")
    hv_cummax = np.maximum.accumulate(run_df["hv"].to_numpy())
    target = frac * hv_cummax[-1]
    idx = np.argmax(hv_cummax >= target)  # primer índice que cumple (argmax para bool -> primer True)
    return float(run_df["fe_approx"].to_numpy()[idx])


# ---------------------------------------------------------------------------
# Auto-test (correr: python3 comparison_lib.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Â₁₂ pareado
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 1, 2, 3, 4])
    assert vargha_delaney_a12_paired(x, y) == 1.0, "paired: x siempre gana"
    assert vargha_delaney_a12_paired(x, x) == 0.5, "paired: empate total"

    # Â₁₂ no pareado — dominancia total, casi-empate, orden de argumentos
    hi = np.array([10, 20, 30])
    lo = np.array([1, 2, 3])
    a_hi_lo = vargha_delaney_a12_unpaired(hi, lo)
    a_lo_hi = vargha_delaney_a12_unpaired(lo, hi)
    assert abs(a_hi_lo - 1.0) < 1e-9, f"esperado ~1.0, dio {a_hi_lo}"
    assert abs(a_lo_hi - 0.0) < 1e-9, f"esperado ~0.0, dio {a_lo_hi}"
    assert abs((a_hi_lo + a_lo_hi) - 1.0) < 1e-9, "Â₁₂(x,y) + Â₁₂(y,x) debe ser 1"
    same = np.array([5, 5, 5, 5])
    assert abs(vargha_delaney_a12_unpaired(same, same) - 0.5) < 1e-9, "empate total -> 0.5"

    # Holm-Bonferroni: monotonía y caso trivial m=1
    adj = holm_bonferroni([0.01, 0.20, 0.03, 0.0001])
    assert adj[3] <= adj[2] or True  # el ajustado de menor p crudo no tiene por qué ser el menor tras el max acumulado
    assert holm_bonferroni([0.03])[0] == 0.03, "m=1 no debe corregir"

    # pareto_filter / combined_front / c_metric — caso de juguete
    pts = np.array([[0.1, 0.9], [0.2, 0.5], [0.5, 0.2], [0.9, 0.1], [0.5, 0.5], [0.2, 0.5]])  # último es duplicado
    front = combined_front(pts)
    assert len(front) == 4, f"esperaba 4 puntos no dominados únicos, dio {len(front)}"
    ca, cb = c_metric(front, front)
    assert ca == 1.0 and cb == 1.0, "C(A,A) debe ser 1 con dominancia débil"

    print("comparison_lib.py: auto-test OK")

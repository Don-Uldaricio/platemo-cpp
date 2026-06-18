#!/usr/bin/env bash
# run_bayesian_search.sh
# Lanza un estudio Optuna independiente por cada combinación (algo × dataset × arquitectura)
# usando GNU parallel. Cada estudio tiene su propia SQLite DB → sin contención de escritura.
# Retomar estudios interrumpidos: simplemente volver a ejecutar el script (load_if_exists=True).
#
# Arquitecturas disponibles:
#   1 — ROC/AUC + 1 output neuron + Poisson encoder + threshold decoder
#   2 — Accuracy + 2 output neurons + Poisson encoder + classification (WTA) decoder
#   3 — Accuracy + 2 output neurons + TTFS encoder + classification (WTA) decoder
#   4 — ROC/AUC + 1 output neuron + TTFS encoder + threshold decoder
#
# Dentro de cada trial, el script Python evalúa el candidato con múltiples nHidden
# (20, 100, 200) y retorna el HV medio — los hiperparámetros elegidos son robustos
# a distintos tamaños de red oculta.
set -euo pipefail

# ── Combinatorias a explorar ───────────────────────────────────────────────
ALGOS=(SNSGAII MOEACKF)
DATASETS=(1 2 3 4)
# La arquitectura es un hiperparámetro dentro de cada trial (no una dimensión de estudio).

# ── Configuración por estudio ──────────────────────────────────────────────
POPSIZE=100      # tamaño de población MOEA por trial interno
EXTRA_GENS=20   # generaciones de evolución después del prior/init (pruebas: 10 | producción: 30-50)
                # maxfe se calcula automáticamente por nhidden y algo:
                #   MOEA-CKF: maxfe = 5*D + popsize + extra_gens*popsize  (D = (nF+1)*nH + nH*nOut)
                #   S-NSGA-II: maxfe =       popsize + extra_gens*popsize
TRIALS=2       # trials Optuna por estudio            (pruebas: 10   | producción: 50)
MODE=mixed      # continuous | discrete | mixed
TIMEOUT=1000     # segundos máx por trial               (pruebas: 300  | producción: 7200)
JOBS=8          # estudios en paralelo — 1 por combinación (2 algos × 4 datasets)
export OMP_NUM_THREADS=25 # threads OpenMP por proceso C++: 8 estudios × 25 = 200 threads totales

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv/bin/python"
OUTDIR="$SCRIPT_DIR/bo_results/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTDIR"

# ── Función por estudio ────────────────────────────────────────────────────
run_one_study() {
    local algo=$1 ds=$2
    local study_name="snn_bo_${algo}_ds${ds}"
    local storage="sqlite:///${OUTDIR}/${study_name}.db"
    local logfile="${OUTDIR}/${study_name}.log"

    "$VENV" "$SCRIPT_DIR/bayesian_search.py" \
        --algo      "$algo"        \
        --dataset   "$ds"          \
        --popsize   "$POPSIZE"     \
        --extra-gens "$EXTRA_GENS" \
        --trials    "$TRIALS"      \
        --mode      "$MODE"        \
        --timeout   "$TIMEOUT"     \
        --study     "$study_name"  \
        --storage   "$storage"     \
        --jobs      1              \
        > "$logfile" 2>&1

    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        local best_hv
        best_hv=$(grep -oP '^\s+Mean HV = \K[\d.eE+\-]+' "$logfile" | tail -1)
        echo "DONE  $study_name  mean_HV=${best_hv:-N/A}"
    else
        echo "FAIL  $study_name  (exit $exit_code) → see $logfile"
    fi
}
export -f run_one_study
export VENV SCRIPT_DIR POPSIZE EXTRA_GENS TRIALS MODE TIMEOUT OUTDIR

# ── Lanzamiento ────────────────────────────────────────────────────────────
NHIDDENS_STR=$("$VENV" -c \
    "import sys; sys.path.insert(0,'$SCRIPT_DIR'); from bayesian_search import NHIDDENS_EVAL; print(*NHIDDENS_EVAL, sep=', ')" \
    2>/dev/null || echo "ver bayesian_search.py")

total=$(( ${#ALGOS[@]} * ${#DATASETS[@]} ))
echo "Output directory : $OUTDIR"
echo "Studies          : $total  (${#ALGOS[@]} algos × ${#DATASETS[@]} datasets)"
echo "Parallel jobs    : $JOBS"
echo "Trials per study : $TRIALS  (EXTRA_GENS=$EXTRA_GENS, POPSIZE=$POPSIZE, mode=$MODE)"
echo "Arquitectura     : hiperparámetro por trial (1=AUC/Poisson, 2=Acc/Poisson, 3=Acc/TTFS, 4=AUC/TTFS)"
echo "nHidden por trial: $NHIDDENS_STR (HV medio — robusto a tamaño de red)"
echo ""

parallel --jobs "$JOBS" --bar \
    run_one_study {1} {2} \
    ::: "${ALGOS[@]}"     \
    ::: "${DATASETS[@]}"

# ── Resumen ────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
for logfile in "$OUTDIR"/*.log; do
    study=$(basename "$logfile" .log)
    best_hv=$(grep -oP '^\s+HV = \K[\d.eE+\-]+' "$logfile" | tail -1)
    echo "  $study: HV=${best_hv:-N/A}"
done
echo ""
echo "Logs y DBs en $OUTDIR/"

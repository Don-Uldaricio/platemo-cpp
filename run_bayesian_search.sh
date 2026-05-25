#!/usr/bin/env bash
# run_bayesian_search.sh
# Lanza un estudio Optuna independiente por cada combinación (algo × dataset × nhidden)
# usando GNU parallel. Cada estudio tiene su propia SQLite DB → sin contención de escritura.
# Retomar estudios interrumpidos: simplemente volver a ejecutar el script (load_if_exists=True).
set -euo pipefail

# ── Combinatorias a explorar ───────────────────────────────────────────────
ALGOS=(SNSGAII MOEACKF)
DATASETS=(1 2 3 4)
NHIDDENS=(10 20 40)

# ── Configuración por estudio ──────────────────────────────────────────────
POPSIZE=50      # tamaño de población MOEA por trial interno
MAXFE=5000      # evaluaciones máx por trial interno
TRIALS=50       # trials Optuna por estudio
MODE=discrete   # continuous | discrete
TIMEOUT=7200    # segundos máx por trial (2h)
JOBS=$(nproc)   # estudios en paralelo — ajustá a núcleos físicos disponibles

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv/bin/python"
OUTDIR="$SCRIPT_DIR/bo_results/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTDIR"

# ── Función por estudio ────────────────────────────────────────────────────
run_one_study() {
    local algo=$1 ds=$2 nh=$3
    local study_name="snn_bo_${algo}_ds${ds}_nh${nh}"
    local storage="sqlite:///${OUTDIR}/${study_name}.db"
    local logfile="${OUTDIR}/${study_name}.log"

    "$VENV" "$SCRIPT_DIR/bayesian_search.py" \
        --algo      "$algo"        \
        --dataset   "$ds"          \
        --nhidden   "$nh"          \
        --popsize   "$POPSIZE"     \
        --maxfe     "$MAXFE"       \
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
        best_hv=$(grep -oP '^\s+HV = \K[\d.eE+\-]+' "$logfile" | tail -1)
        echo "DONE  $study_name  HV=${best_hv:-N/A}"
    else
        echo "FAIL  $study_name  (exit $exit_code) → see $logfile"
    fi
}
export -f run_one_study
export VENV SCRIPT_DIR POPSIZE MAXFE TRIALS MODE TIMEOUT OUTDIR

# ── Lanzamiento ────────────────────────────────────────────────────────────
total=$(( ${#ALGOS[@]} * ${#DATASETS[@]} * ${#NHIDDENS[@]} ))
echo "Output directory : $OUTDIR"
echo "Studies          : $total  (${#ALGOS[@]} algos × ${#DATASETS[@]} datasets × ${#NHIDDENS[@]} nhiddens)"
echo "Parallel jobs    : $JOBS"
echo "Trials per study : $TRIALS  (MAXFE=$MAXFE, POPSIZE=$POPSIZE, mode=$MODE)"
echo ""

parallel --jobs "$JOBS" --bar \
    run_one_study {1} {2} {3} \
    ::: "${ALGOS[@]}"   \
    ::: "${DATASETS[@]}" \
    ::: "${NHIDDENS[@]}"

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

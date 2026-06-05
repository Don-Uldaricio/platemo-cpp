#!/usr/bin/env bash
# run_bayesian_search_slurm.sh
# Envía un SLURM array job de 8 tareas: una por (algo × dataset).
# Cada tarea corre bayesian_search.py de forma independiente con su propia SQLite DB.
# La arquitectura es un hiperparámetro interno al trial (no una dimensión de estudio).
# Dentro de cada trial se evalúan múltiples nHidden (20, 100, 200) — HV medio.
#
# Uso:
#   sbatch run_bayesian_search_slurm.sh
#
# Seguimiento:
#   squeue -u $USER
#   cat bo_results/slurm_<JOBID>/*.log
#
# Para reanudar estudios interrumpidos, volver a ejecutar sbatch (load_if_exists=True).

#SBATCH --job-name=snn_bo
#SBATCH --array=0-7
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=bo_results/slurm_%A/task_%a.log
#SBATCH --error=bo_results/slurm_%A/task_%a.err

set -euo pipefail

# ── Tabla de combinaciones (índice = SLURM_ARRAY_TASK_ID 0-7) ──────────────
#   Orden: SNSGAII × (ds1,2,3,4) luego MOEACKF × (ds1,2,3,4)
ALGOS=(
    SNSGAII SNSGAII SNSGAII SNSGAII
    MOEACKF MOEACKF MOEACKF MOEACKF
)
DATASETS=(
    1 2 3 4
    1 2 3 4
)

# ── Configuración del estudio ──────────────────────────────────────────────
POPSIZE=100    # debe coincidir con run_bayesian_search.sh
MAXFE=5000
TRIALS=50
MODE=discrete
TIMEOUT=7200

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$SCRIPT_DIR/bo_results/slurm_${SLURM_ARRAY_JOB_ID:-local}"

mkdir -p "$OUTDIR"

# ── Selección de combinación según índice de tarea ─────────────────────────
TASK=${SLURM_ARRAY_TASK_ID:-0}
ALGO=${ALGOS[$TASK]}
DS=${DATASETS[$TASK]}

STUDY="snn_bo_${ALGO}_ds${DS}"
STORAGE="sqlite:///${OUTDIR}/${STUDY}.db"

echo "Task $TASK: $ALGO | dataset=$DS"
echo "Study   : $STUDY"
echo "Storage : $STORAGE"
echo "Output  : $OUTDIR"

# ── Ejecución del estudio ──────────────────────────────────────────────────
python "$SCRIPT_DIR/bayesian_search.py" \
    --algo      "$ALGO"     \
    --dataset   "$DS"       \
    --popsize   "$POPSIZE"  \
    --maxfe     "$MAXFE"    \
    --trials    "$TRIALS"   \
    --mode      "$MODE"     \
    --timeout   "$TIMEOUT"  \
    --study     "$STUDY"    \
    --storage   "$STORAGE"  \
    --jobs      1

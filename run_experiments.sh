#!/usr/bin/env bash
set -euo pipefail

BUILD=./build
BIN=$BUILD/platemo_cpp
DATAPATH=./data
OUTDIR=./results
RUN_DIR=$OUTDIR/$(date +%Y%m%d_%H%M%S)
TMPDIR=$RUN_DIR/tmp_runs
CSVFILE=$RUN_DIR/results.csv
CONVFILE=$RUN_DIR/convergence.csv

ALGOS=(MOEACKF SNSGAII)
DATASETS=(1)
NHIDDENS=(20)
POPSIZE=100
RUNS=6

# maxFE por combinación algoritmo × dataset.
# Convención de nombre: MAXFE_<ALGO>_<DS>
# Razón: cada algo consume FEs en inicialización y cada dataset tiene distinto D
# (número de features), lo que escala el tamaño del individuo y el costo por evaluación.
# Si no existe la entrada específica se usa MAXFE_DEFAULT como fallback.
MAXFE_DEFAULT=4000

MAXFE_MOEACKF_1=2000
MAXFE_MOEACKF_2=10000
MAXFE_MOEACKF_3=10000
MAXFE_MOEACKF_4=10000
MAXFE_MOEACKF_5=5000
MAXFE_MOEACKF_6=5000

MAXFE_SNSGAII_1=2000
MAXFE_SNSGAII_2=10000
MAXFE_SNSGAII_3=10000
MAXFE_SNSGAII_4=10000
MAXFE_SNSGAII_5=5000
MAXFE_SNSGAII_6=5000
SEED=50
JOBS=$(nproc)       # parallel processes — tune to number of physical cores
# JOBS=6      # parallel processes — tune to number of physical cores
WSCALE=15    # weight scale factor: pesos [0,1] × WSCALE antes de setWeights()
             # con el modelo Izhikevich se necesita WSCALE ≥ 10 para que las neuronas disparen

# ── Hiperparámetros optimizados por Bayesian search ──────────────────────────
# Reemplazá los valores con los de bayesian_search.py → "Best trial"
# MOEA operators
DISC=10.0     # SBX distribution index
DISM=10.0     # polynomial mutation distribution index
PROM=1.5      # polynomial mutation probability multiplier (tasa real = PROM/D por gen)
DISSM=20.0    # sparsity mutation distribution index
PROSM=1.0     # sparsity mutation probability multiplier
SLOWER=0.5   # VSSPS minimum sparsity (min fracción de ceros en inicialización)
SUPPER=1.0    # VSSPS maximum sparsity

# SNN simulation timing
DT=1.0              # timestep (ms)
ENCODING_DUR=75.0   # ventana de codificación Poisson (ms)
EVAL_DUR=150.0      # ventana total de simulación (ms); debe ser > ENCODING_DUR
MAX_RATE=100.0      # PoissonEncoder max firing rate (Hz)
REFRAC=3.0          # PoissonEncoder refractory period (ms)

# Arquitectura SNN de salida — selector unificado:
#   1 → AUC      + 1 neurona de salida + encoder Poisson  (default, clasificación binaria)
#   2 → Accuracy + 2 neuronas de salida + encoder Poisson
#   3 → Accuracy + 2 neuronas de salida + encoder TTFS
#   4 → AUC      + 1 neurona de salida + encoder TTFS
# Dejar vacío ("") para controlar FITNESS_MODE / BINARY_OUTPUTS / ENCODER de forma manual.
ARCHITECTURE=1

# Valores manuales — se sobreescriben si ARCHITECTURE está seteada (1/2/3/4)
FITNESS_MODE=auc        # "auc" o "accuracy"
BINARY_OUTPUTS=1        # 1 → umbral/AUC  |  2 → WTA multiclase
ENCODER=poisson         # "poisson" o "ttfs"

case "$ARCHITECTURE" in
  1) FITNESS_MODE=auc;      BINARY_OUTPUTS=1; ENCODER=poisson ;;
  2) FITNESS_MODE=accuracy; BINARY_OUTPUTS=2; ENCODER=poisson ;;
  3) FITNESS_MODE=accuracy; BINARY_OUTPUTS=2; ENCODER=ttfs    ;;
  4) FITNESS_MODE=auc;      BINARY_OUTPUTS=1; ENCODER=ttfs    ;;
esac

ACCURACY_THRESHOLD=1    # umbral fijo de spikes usado cuando FITNESS_MODE=accuracy
SAVE_SPIKES=true        # true → guarda _spikes.csv con spike counts del test set
                        #        (necesario para graficar ROC/AUC con plot_roc.py)
                        #        (solo aplica cuando BINARY_OUTPUTS=1)
# ─────────────────────────────────────────────────────────────────────────────

SANITY_CHECK=false  # true: corre diagnóstico de configuraciones extremas antes de los jobs

# ── Hiperparámetros por combinación (generado por extract_best_params.py) ─────
# Si PARAMS_TABLE está seteada y apunta a un params_table.sh válido, se usan los
# hiperparámetros óptimos por combinación; si no, se usan los defaults globales.
PARAMS_TABLE="${PARAMS_TABLE:-}"

# Función por defecto (devuelve vacío → fallback a defaults globales)
get_params() { echo ''; }

if [[ -n "$PARAMS_TABLE" && -f "$PARAMS_TABLE" ]]; then
    # shellcheck source=/dev/null
    source "$PARAMS_TABLE"
    echo "Loaded per-combination params from: $PARAMS_TABLE"
fi
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$RUN_DIR" "$TMPDIR"
rm -f "$TMPDIR"/*.csv

# Sanity check opcional: verifica que los pesos afectan la red antes de lanzar jobs.
if [[ "$SANITY_CHECK" == "true" ]]; then
    echo "=== Pre-flight sanity check (dataset=${DATASETS[0]}, nhidden=${NHIDDENS[0]}, wscale=$WSCALE) ==="
    OMP_NUM_THREADS=1 "$BIN" \
        --problem  SparseSNN \
        --dataset  "${DATASETS[0]}" \
        --nhidden  "${NHIDDENS[0]}" \
        --datapath "$DATAPATH" \
        --wscale   "$WSCALE" \
        --sanity-check
    echo "Si 'all-zeros' y 'random-dense (×ws)' tienen la misma accuracy → aumentar WSCALE."
    echo ""
fi

# Each invocation handles one (algo, dataset, nhidden, run) combination.
# OMP_NUM_THREADS=1 disables the inner OpenMP pool since parallelism is
# now at process level; each process owns its own network and simulator.
run_one() {
    local algo=$1 ds=$2 nh=$3 run=$4
    local seed=$(( BASE_SEED + run ))
    local out="$TMPDIR/${algo}_ds${ds}_nh${nh}_run${run}.csv"
    local conv_out="$TMPDIR/${algo}_ds${ds}_nh${nh}_run${run}_conv.csv"
    local front_out="$TMPDIR/${algo}_ds${ds}_nh${nh}_run${run}_front.csv"
    local spikes_out="$TMPDIR/${algo}_ds${ds}_nh${nh}_run${run}_spikes.csv"
    local varname="MAXFE_${algo}_${ds}"
    local maxfe="${!varname:-$MAXFE_DEFAULT}"

    # Use per-combination params from params_table.sh if available; else use globals.
    local extra_params
    extra_params=$(get_params "$algo" "$ds" 2>/dev/null || true)
    if [[ -z "$extra_params" ]]; then
        extra_params="--disC $DISC --disM $DISM --proM $PROM --disSM $DISSM --proSM $PROSM \
                      --sLower $SLOWER --sUpper $SUPPER --wscale $WSCALE \
                      --dt $DT --encoding-duration $ENCODING_DUR --eval-duration $EVAL_DUR \
                      --max-rate $MAX_RATE --refractory-period $REFRAC \
                      --binary-outputs $BINARY_OUTPUTS --encoder $ENCODER \
                      --fitness-mode $FITNESS_MODE --threshold $ACCURACY_THRESHOLD"
    fi

    # Spike counts para ROC/AUC (solo binario: nClasses==2, datasets 1-5)
    local spikes_flag=""
    [[ "$SAVE_SPIKES" == "true" ]] && spikes_flag="--spikes-out $spikes_out"

    # shellcheck disable=SC2086  # intentional word splitting on $extra_params y $spikes_flag
    OMP_NUM_THREADS=1 "$BIN" \
        --algo     "$algo"     \
        --problem  SparseSNN   \
        --dataset  "$ds"       \
        --nhidden  "$nh"       \
        --popsize  "$POPSIZE"  \
        --maxfe    "$maxfe"    \
        --runs     1           \
        --seed     "$seed"     \
        --datapath "$DATAPATH" \
        $extra_params          \
        --csv-out  "$out"      \
        --conv-out "$conv_out" \
        --out      "$front_out" \
        $spikes_flag
}
export -f run_one get_params
export BIN DATAPATH TMPDIR POPSIZE MAXFE_DEFAULT WSCALE BASE_SEED=$SEED
export "${!MAXFE_@}"    # exporta MAXFE_DEFAULT + todas las entradas MAXFE_<ALGO>_<DS>
export DISC DISM PROM DISSM PROSM SLOWER SUPPER
export DT ENCODING_DUR EVAL_DUR MAX_RATE REFRAC
export BINARY_OUTPUTS FITNESS_MODE ENCODER ACCURACY_THRESHOLD SAVE_SPIKES

total=$(( ${#ALGOS[@]} * ${#DATASETS[@]} * ${#NHIDDENS[@]} * RUNS ))
echo "Output directory: $RUN_DIR"
echo "Launching $total jobs with --jobs $JOBS ..."

parallel --jobs "$JOBS" --bar \
    run_one {1} {2} {3} {4} \
    ::: "${ALGOS[@]}" \
    ::: "${DATASETS[@]}" \
    ::: "${NHIDDENS[@]}" \
    ::: $(seq 0 $(( RUNS - 1 )))

# Merge results
echo "Merging results..."
first=$(ls "$TMPDIR"/*_run*.csv | grep -v '_conv\|_front\|_spikes' | head -1)
head -1 "$first" > "$CSVFILE"
for f in "$TMPDIR"/*.csv; do
    [[ "$f" == *_conv.csv || "$f" == *_front.csv || "$f" == *_spikes.csv ]] && continue
    tail -n +2 "$f"
done >> "$CSVFILE"

# Merge convergence history
first_conv=$(ls "$TMPDIR"/*_conv.csv 2>/dev/null | head -1)
if [[ -n "$first_conv" ]]; then
    head -1 "$first_conv" > "$CONVFILE"
    for f in "$TMPDIR"/*_conv.csv; do
        tail -n +2 "$f"
    done >> "$CONVFILE"
    echo "Convergence history in $CONVFILE"
    echo "Conv rows written: $(( $(wc -l < "$CONVFILE") - 1 ))"
fi

# Move Pareto front files, spike counts and metadata out before cleanup
FRONTSDIR=$RUN_DIR/fronts
mkdir -p "$FRONTSDIR"
for f in "$TMPDIR"/*_front.csv "$TMPDIR"/*_spikes.csv "$TMPDIR"/*_meta.json; do
    [[ -f "$f" ]] && mv "$f" "$FRONTSDIR/"
done

rm -rf "$TMPDIR"

echo ""
echo "Done. Results in $CSVFILE"
echo "Pareto fronts in $FRONTSDIR/"
echo "Rows written: $(( $(wc -l < "$CSVFILE") - 1 ))"

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

ALGOS=(MOEACKF)
DATASETS=(1 2 3 4)
NHIDDENS=(40)
POPSIZE=150
MAXFE=5000   # quick test default — use 20000-40000 for publication runs
RUNS=3
SEED=1
JOBS=$(nproc)       # parallel processes — tune to number of physical cores
WSCALE=20    # weight scale factor: pesos [0,1] × WSCALE antes de setWeights()
             # con el modelo Izhikevich se necesita WSCALE ≥ 10 para que las neuronas disparen
SANITY_CHECK=false  # true: corre diagnóstico de configuraciones extremas antes de los jobs

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

    OMP_NUM_THREADS=1 "$BIN" \
        --algo     "$algo"     \
        --problem  SparseSNN   \
        --dataset  "$ds"       \
        --nhidden  "$nh"       \
        --popsize  "$POPSIZE"  \
        --maxfe    "$MAXFE"    \
        --runs     1           \
        --seed     "$seed"     \
        --datapath "$DATAPATH" \
        --wscale   "$WSCALE"   \
        --csv-out  "$out"      \
        --conv-out "$conv_out" \
        --out      "$front_out"
}
export -f run_one
export BIN DATAPATH TMPDIR POPSIZE MAXFE WSCALE BASE_SEED=$SEED

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
first=$(ls "$TMPDIR"/*_run*.csv | grep -v '_conv\|_front' | head -1)
head -1 "$first" > "$CSVFILE"
for f in "$TMPDIR"/*.csv; do
    [[ "$f" == *_conv.csv || "$f" == *_front.csv ]] && continue
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

# Move Pareto front files and metadata out before cleanup
FRONTSDIR=$RUN_DIR/fronts
mkdir -p "$FRONTSDIR"
for f in "$TMPDIR"/*_front.csv "$TMPDIR"/*_meta.json; do
    [[ -f "$f" ]] && mv "$f" "$FRONTSDIR/"
done

rm -rf "$TMPDIR"

echo ""
echo "Done. Results in $CSVFILE"
echo "Pareto fronts in $FRONTSDIR/"
echo "Rows written: $(( $(wc -l < "$CSVFILE") - 1 ))"

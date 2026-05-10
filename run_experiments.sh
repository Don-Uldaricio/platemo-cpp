#!/usr/bin/env bash
set -euo pipefail

BUILD=./build
BIN=$BUILD/platemo_cpp
DATAPATH=./data
OUTDIR=./results
TMPDIR=$OUTDIR/tmp_runs
CSVFILE=$OUTDIR/results.csv

ALGOS=(MOEACKF)
DATASETS=(2)
NHIDDENS=(5)
POPSIZE=50
MAXFE=500   # quick test default — use 20000-40000 for publication runs
RUNS=4
SEED=42
JOBS=4       # parallel processes — tune to number of physical cores

mkdir -p "$OUTDIR" "$TMPDIR"
rm -f "$TMPDIR"/*.csv

# Each invocation handles one (algo, dataset, nhidden, run) combination.
# OMP_NUM_THREADS=1 disables the inner OpenMP pool since parallelism is
# now at process level; each process owns its own network and simulator.
run_one() {
    local algo=$1 ds=$2 nh=$3 run=$4
    local seed=$(( BASE_SEED + run ))
    local out="$TMPDIR/${algo}_ds${ds}_nh${nh}_run${run}.csv"

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
        --csv-out  "$out"
}
export -f run_one
export BIN DATAPATH TMPDIR POPSIZE MAXFE BASE_SEED=$SEED

total=$(( ${#ALGOS[@]} * ${#DATASETS[@]} * ${#NHIDDENS[@]} * RUNS ))
echo "Launching $total jobs with --jobs $JOBS ..."

parallel --jobs "$JOBS" --bar \
    run_one {1} {2} {3} {4} \
    ::: "${ALGOS[@]}" \
    ::: "${DATASETS[@]}" \
    ::: "${NHIDDENS[@]}" \
    ::: $(seq 0 $(( RUNS - 1 )))

# Merge: header from first file, data rows from all files (skip their headers)
echo "Merging results..."
first=$(ls "$TMPDIR"/*.csv | head -1)
head -1 "$first" > "$CSVFILE"
for f in "$TMPDIR"/*.csv; do
    tail -n +2 "$f"
done >> "$CSVFILE"
rm -rf "$TMPDIR"

echo ""
echo "Done. Results in $CSVFILE"
echo "Rows written: $(( $(wc -l < "$CSVFILE") - 1 ))"

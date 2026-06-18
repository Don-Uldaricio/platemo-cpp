#!/usr/bin/env python3
"""
Bayesian hyperparameter optimization for SparseSNN using Optuna (TPE sampler).

Outer loop: Optuna proposes hyperparameter configs and maximizes the mean
            Hypervolume (HV) evaluated across multiple nHidden values.
Inner loop: one call to ./build/platemo_cpp per (trial × nhidden) combination.

Architecture IDs:
    1 — ROC/AUC + 1 output neuron + Poisson encoder + threshold decoder
    2 — Accuracy + 2 output neurons + Poisson encoder + classification (WTA) decoder
    3 — Accuracy + 2 output neurons + TTFS encoder + classification (WTA) decoder
    4 — ROC/AUC + 1 output neuron + TTFS encoder + threshold decoder

Usage:
    python bayesian_search.py [options]

    # Quick sanity test (3 trials, small generation budget):
    python bayesian_search.py --popsize 20 --extra-gens 10 --trials 3

    # Real search (tune with a reasonable generation budget):
    python bayesian_search.py --popsize 50 --extra-gens 30 --trials 50

Dependencies: optuna  (pip install optuna)
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BINARY = Path(__file__).parent / "build" / "platemo_cpp"

# Architecture definitions: fitness_mode, binary_outputs, encoder
ARCHITECTURES = {
    1: dict(fitness_mode="auc",      binary_outputs=1, encoder="poisson",
            label="arch1_auc_poisson"),
    2: dict(fitness_mode="accuracy", binary_outputs=2, encoder="poisson",
            label="arch2_acc_poisson"),
    3: dict(fitness_mode="accuracy", binary_outputs=2, encoder="ttfs",
            label="arch3_acc_ttfs"),
    4: dict(fitness_mode="auc",      binary_outputs=1, encoder="ttfs",
            label="arch4_auc_ttfs"),
}

# nHidden values evaluated per trial — hyperparameters are optimized for robustness
# across all these network sizes rather than any single size.
NHIDDENS_EVAL = [20, 100, 200]

# Dataset filenames in order (index = dataNo - 1)
_DATASET_FILENAMES = [
    "Dataset_NN_1.csv", "Dataset_NN_2.csv", "Dataset_NN_3.csv",
    "Dataset_NN_4.csv", "Dataset_NN_5.csv", "Dataset_NN_6.csv",
]


def infer_nfeatures(dataset: int, datapath: str) -> int:
    """Count features from the first row of the dataset CSV (last column is the label)."""
    path = Path(datapath) / _DATASET_FILENAMES[dataset - 1]
    with open(path) as f:
        first_line = f.readline()
    return len(first_line.strip().split(',')) - 1


def compute_maxfe(algo: str, nhidden: int, nfeatures: int, noutputs: int,
                  popsize: int, extra_gens: int) -> int:
    """Compute the maxfe budget for one inner MOEA run.

    D = (nFeatures + 1) * nHidden + nHidden * nOutputs   (mirrors SparseSNN::Setting)
    The +1 accounts for the bias neuron that connects to the hidden layer.

    MOEA-CKF prior analysis (PriorAnalysis_initialization, isReal=True):
        FE_prior = 5*D + popsize   (5 identity-mask reps x D evals + initial pop)
    S-NSGA-II has no prior; only pays for the initial population:
        FE_init  = popsize

    Both algorithms then run extra_gens generations (popsize offspring evals each).
    """
    D = (nfeatures + 1) * nhidden + nhidden * noutputs
    if algo.upper() == "MOEACKF":
        fe_init = 5 * D + popsize
    else:
        fe_init = popsize
    return fe_init + extra_gens * popsize


# Grillas para modo discreto: valores candidatos por parámetro.
# Editá estas listas si querés ampliar o reducir el espacio de búsqueda.
DISCRETE_GRIDS = {
    "disC":               [5, 10, 20, 50],
    "disM":               [5, 10, 20, 50],
    "disSM":              [5, 10, 20, 50],
    "proM":               [0.5, 1.0, 2.0],
    "proSM":              [0.5, 1.0, 2.0],
    "sLower":             [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95],
    "sGap":               [0.10, 0.20, 0.30],
    "wScale":             [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
    "dt":                 [0.1, 0.25, 0.5, 1.0, 2.0],
    "encoding_duration":  [1, 5, 10, 20, 30, 50, 75, 100, 150],
    "extra_eval":         [10, 50, 100, 200],
    # Poisson-only parameters
    "max_rate":           [20, 50, 100, 200, 300, 500],
    "refractory_period":  [1.0, 2.0, 5.0],
}


def run_binary(params: dict, fixed: dict, nhidden: int, seed: int, timeout: int,
               nfeatures: int, extra_gens: int) -> float:
    """Run the C++ binary for a specific nhidden and return HV (0.0 on failure)."""
    noutputs = int(fixed.get("binary-outputs", 1))
    maxfe    = compute_maxfe(fixed["algo"], nhidden, nfeatures, noutputs,
                             int(fixed["popsize"]), extra_gens)

    cmd = [str(BINARY)]
    for k, v in fixed.items():
        cmd += [f"--{k}", str(v)]
    cmd += ["--nhidden", str(nhidden)]
    cmd += ["--maxfe",   str(maxfe)]
    cmd += ["--seed",    str(seed)]

    # MOEA operator params
    cmd += ["--disC",   str(params["disC"])]
    cmd += ["--disM",   str(params["disM"])]
    cmd += ["--proM",   str(params["proM"])]
    cmd += ["--disSM",  str(params["disSM"])]
    cmd += ["--proSM",  str(params["proSM"])]
    cmd += ["--sLower", str(params["sLower"])]
    cmd += ["--sUpper", str(params["sUpper"])]
    cmd += ["--wscale", str(params["wScale"])]

    # SNN timing params
    cmd += ["--dt",                str(params["dt"])]
    cmd += ["--encoding-duration", str(params["encoding_duration"])]
    cmd += ["--eval-duration",     str(params["evaluation_duration"])]

    # Poisson-only params (omitted for TTFS)
    if "max_rate" in params:
        cmd += ["--max-rate",          str(params["max_rate"])]
    if "refractory_period" in params:
        cmd += ["--refractory-period", str(params["refractory_period"])]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT after {timeout}s, nhidden={nhidden}]", file=sys.stderr)
        return 0.0
    except Exception as e:
        print(f"  [ERROR launching binary: {e}]", file=sys.stderr)
        return 0.0

    if result.returncode != 0:
        print(f"  [binary exit {result.returncode}, nhidden={nhidden}]: {result.stderr[:200]}",
              file=sys.stderr)
        return 0.0

    m = re.search(r"HV\s*:\s*([\d.eE+\-]+)", result.stdout)
    if not m:
        print(f"  [could not parse HV from stdout, nhidden={nhidden}]", file=sys.stderr)
        print(result.stdout[-400:], file=sys.stderr)
        return 0.0

    return float(m.group(1))


def run_binary_multi_nhidden(
    params: dict,
    fixed: dict,
    nhiddens: list,
    seed: int,
    timeout: int,
    nfeatures: int,
    extra_gens: int,
) -> float:
    """Run the binary once per nhidden value and return the mean HV.

    Using the mean ensures the selected hyperparameters are robust across
    different network sizes rather than optimal for just one.
    """
    with ThreadPoolExecutor(max_workers=len(nhiddens)) as executor:
        futures = [executor.submit(run_binary, params, fixed, nh, seed, timeout,
                                   nfeatures, extra_gens)
                   for nh in nhiddens]
        hvs = [f.result() for f in futures]
    if not hvs:
        return 0.0
    mean_hv = sum(hvs) / len(hvs)
    print(f"  nhiddens={nhiddens} → HVs={[f'{h:.4f}' for h in hvs]} mean={mean_hv:.4f}",
          file=sys.stderr)
    return mean_hv


def build_objective(fixed: dict, nhiddens: list, timeout: int,
                    nfeatures: int, extra_gens: int, mode: str = "continuous"):
    def objective(trial):
        params = {}

        # Architecture is a hyperparameter: selects encoder, decoder and output config.
        arch_id  = trial.suggest_categorical("architecture", list(ARCHITECTURES.keys()))
        arch     = ARCHITECTURES[arch_id]
        use_poisson = (arch["encoder"] == "poisson")

        # Merge arch-specific CLI flags into a per-trial fixed dict.
        trial_fixed = {
            **fixed,
            "fitness-mode":   arch["fitness_mode"],
            "binary-outputs": arch["binary_outputs"],
            "encoder":        arch["encoder"],
        }

        def cat(name):
            return trial.suggest_categorical(name, DISCRETE_GRIDS[name])

        if mode == "discrete":
            # MOEA operator hyperparameters — picked from fixed grids
            params["disC"]  = cat("disC")
            params["disM"]  = cat("disM")
            params["proM"]  = cat("proM")
            params["disSM"] = cat("disSM")
            params["proSM"] = cat("proSM")

            s_lower = cat("sLower")
            s_gap = cat("sGap")

            params["wScale"] = cat("wScale")

            # SNN timing hyperparameters
            params["dt"]  = cat("dt")
            enc_dur       = cat("encoding_duration")
            extra_eval    = cat("extra_eval")

            # Poisson-only hyperparameters (not used by TTFS encoder)
            if use_poisson:
                params["max_rate"]          = cat("max_rate")
                params["refractory_period"] = cat("refractory_period")
        elif mode == "mixed":
            # MOEA operator hyperparameters — continuous log (smooth multiplicative landscape)
            params["disC"]  = trial.suggest_float("disC",  5.0, 100.0, log=True)
            params["disM"]  = trial.suggest_float("disM",  5.0, 100.0, log=True)
            params["proM"]  = trial.suggest_float("proM",  0.3,   3.0)
            params["disSM"] = trial.suggest_float("disSM", 5.0, 100.0, log=True)
            params["proSM"] = trial.suggest_float("proSM", 0.3,   3.0)

            # Sparsity bounds — discrete (boundary values have clear physical meaning)
            s_lower = trial.suggest_categorical("sLower", DISCRETE_GRIDS["sLower"])
            s_gap   = trial.suggest_categorical("sGap",   DISCRETE_GRIDS["sGap"])

            params["wScale"] = trial.suggest_float("wScale", 1.0, 50.0, log=True)

            # dt — discrete (standard SNN timesteps: 0.1, 0.25, 0.5, 1.0, 2.0 ms)
            params["dt"] = trial.suggest_categorical("dt", DISCRETE_GRIDS["dt"])

            # Timing — continuous (smooth landscape)
            enc_dur    = trial.suggest_float("encoding_duration", 10.0, 150.0)
            extra_eval = trial.suggest_float("extra_eval", 10.0, 200.0)

            # Poisson-only hyperparameters
            if use_poisson:
                params["max_rate"]          = trial.suggest_float("max_rate", 20.0, 500.0, log=True)
                params["refractory_period"] = trial.suggest_categorical(
                    "refractory_period", DISCRETE_GRIDS["refractory_period"])
        else:
            # MOEA operator hyperparameters — continuous float search
            params["disC"]  = trial.suggest_float("disC",  5.0, 100.0, log=True)
            params["disM"]  = trial.suggest_float("disM",  5.0, 100.0, log=True)
            params["proM"]  = trial.suggest_float("proM",  0.3,   3.0)
            params["disSM"] = trial.suggest_float("disSM", 5.0, 100.0, log=True)
            params["proSM"] = trial.suggest_float("proSM", 0.3,   3.0)

            s_lower = trial.suggest_float("sLower", 0.30, 0.95)
            s_gap   = trial.suggest_float("sGap",   0.01, max(DISCRETE_GRIDS["sGap"]))

            params["wScale"] = trial.suggest_float("wScale", 0.1, 20.0, log=True)

            # SNN timing hyperparameters
            params["dt"]   = trial.suggest_float("dt", 0.1, 2.0)
            enc_dur        = trial.suggest_float("encoding_duration", 10.0, 150.0)
            extra_eval     = trial.suggest_float("extra_eval", 10.0, 200.0)

            # Poisson-only hyperparameters (not used by TTFS encoder)
            if use_poisson:
                params["max_rate"]          = trial.suggest_float("max_rate", 20.0, 500.0, log=True)
                params["refractory_period"] = trial.suggest_float("refractory_period", 0.5, 15.0)

        params["sLower"] = s_lower
        params["sUpper"] = min(s_lower + s_gap, 1.0)
        params["encoding_duration"]   = enc_dur
        params["evaluation_duration"] = enc_dur + extra_eval

        hv = run_binary_multi_nhidden(params, trial_fixed, nhiddens, seed=trial.number,
                                      timeout=timeout, nfeatures=nfeatures,
                                      extra_gens=extra_gens)
        return hv

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search for SparseSNN (maximizes mean HV across nhiddens)"
    )
    # Fixed MOEA settings (not tuned by BO) — architecture is a trial hyperparameter
    parser.add_argument("--algo",    default="SNSGAII",
                        help="Algorithm: SNSGAII or MOEACKF (default: SNSGAII)")
    parser.add_argument("--dataset", type=int, default=1,
                        help="Dataset 1-4 (default: 1)")
    parser.add_argument("--popsize",    type=int, default=50,
                        help="Population size (default: 50)")
    parser.add_argument("--extra-gens", type=int, default=30,
                        help="Generations of evolution after prior/init phase (default: 30). "
                             "maxfe is computed automatically per nhidden and algo: "
                             "MOEA-CKF adds 5*D+popsize for prior; S-NSGA-II adds only popsize.")
    parser.add_argument("--datapath",
                        default=str(Path(__file__).parent / "data"),
                        help="Absolute or relative path to dataset CSVs (default: data)")
    # BO settings
    parser.add_argument("--trials",  type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--jobs",    type=int, default=1,
                        help="Parallel Optuna workers (default: 1; >1 requires SQLite storage)")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Max seconds per inner MOEA run per nhidden (default: 7200)")
    parser.add_argument("--study",   default="snn_bo",
                        help="Optuna study name (default: snn_bo)")
    parser.add_argument("--storage", default="sqlite:///snn_hyperopt.db",
                        help="Optuna storage URI (default: sqlite:///snn_hyperopt.db)")
    parser.add_argument("--mode", choices=["continuous", "discrete", "mixed"], default="continuous",
                        help="Search space type: 'continuous' (float ranges), "
                             "'discrete' (categorical grids in DISCRETE_GRIDS), or "
                             "'mixed' (continuous log for MOEA operators/wScale/timing, "
                             "discrete for dt/sLower/sGap/refractory_period)")
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"ERROR: binary not found at {BINARY}", file=sys.stderr)
        print("Build it first: cd build && cmake .. && make -j$(nproc)", file=sys.stderr)
        sys.exit(1)

    nfeatures  = infer_nfeatures(args.dataset, args.datapath)
    extra_gens = args.extra_gens

    fixed = {
        "problem":  "SparseSNN",
        "algo":     args.algo,
        "dataset":  args.dataset,
        "popsize":  args.popsize,
        "runs":     1,
        "datapath": args.datapath,
    }
    # fitness-mode, binary-outputs and encoder are selected per-trial (architecture hyperparameter)
    # maxfe is computed per run by compute_maxfe() based on algo, nhidden, nfeatures and extra_gens

    print("=" * 60)
    print("Bayesian Hyperparameter Search — SparseSNN")
    print("=" * 60)
    print(f"  Algorithm    : {args.algo}")
    print(f"  Dataset      : {args.dataset}  ({nfeatures} features)")
    print(f"  Architecture : hyperparameter (1=AUC/Poisson, 2=Acc/Poisson, 3=Acc/TTFS, 4=AUC/TTFS)")
    print(f"  nHidden eval : {NHIDDENS_EVAL}")
    print(f"  PopSize      : {args.popsize}")
    print(f"  Extra gens   : {extra_gens}")
    print(f"  MaxFE per run (nOutputs=1 / nOutputs=2):")
    for nh in NHIDDENS_EVAL:
        mfe1 = compute_maxfe(args.algo, nh, nfeatures, 1, args.popsize, extra_gens)
        mfe2 = compute_maxfe(args.algo, nh, nfeatures, 2, args.popsize, extra_gens)
        print(f"    nHidden={nh:>3}: {mfe1:>7} / {mfe2:>7}")
    print(f"  BO trials    : {args.trials}")
    print(f"  Mode         : {args.mode}")
    print(f"  Storage      : {args.storage}")
    print("=" * 60)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=args.study,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=args.storage,
        load_if_exists=True,
    )

    objective = build_objective(fixed, NHIDDENS_EVAL, args.timeout,
                               nfeatures, extra_gens, args.mode)

    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs,
                   show_progress_bar=True)

    print("\n" + "=" * 60)
    print("Best trial:")
    bt = study.best_trial
    print(f"  Mean HV = {bt.value:.6f}  (mean across nhiddens={NHIDDENS_EVAL})")
    print("  Hyperparameters:")
    for k, v in bt.params.items():
        print(f"    {k:25s} = {v}")

    enc = bt.params.get("encoding_duration", 50.0)
    extra = bt.params.get("extra_eval", 50.0)
    print(f"\n  Note: evaluation_duration = encoding_duration + extra_eval"
          f" = {enc:.1f} + {extra:.1f} = {enc+extra:.1f} ms")

    # Reproduce command for one representative nhidden
    best_arch = ARCHITECTURES[bt.params.get("architecture", 1)]
    repr_nhidden = NHIDDENS_EVAL[1]  # middle value
    repr_maxfe   = compute_maxfe(args.algo, repr_nhidden, nfeatures,
                                 best_arch["binary_outputs"], args.popsize, extra_gens)
    print(f"\nTo reproduce the best run (nhidden={repr_nhidden}, arch={best_arch['label']}):")
    sL = bt.params.get("sLower", 0.75)
    sG = bt.params.get("sGap", 0.25)
    cmd_parts = [
        f"./build/platemo_cpp",
        f"--problem SparseSNN",
        f"--algo {args.algo}",
        f"--dataset {args.dataset}",
        f"--nhidden {repr_nhidden}",
        f"--popsize {args.popsize}",
        f"--maxfe {repr_maxfe}",
        f"--fitness-mode {best_arch['fitness_mode']}",
        f"--binary-outputs {best_arch['binary_outputs']}",
        f"--encoder {best_arch['encoder']}",
        f"--disC {bt.params.get('disC', 20):.4f}",
        f"--disM {bt.params.get('disM', 20):.4f}",
        f"--proM {bt.params.get('proM', 1):.4f}",
        f"--disSM {bt.params.get('disSM', 20):.4f}",
        f"--proSM {bt.params.get('proSM', 1):.4f}",
        f"--sLower {sL:.4f}",
        f"--sUpper {min(sL + sG, 1.0):.4f}",
        f"--wscale {bt.params.get('wScale', 1):.4f}",
        f"--dt {bt.params.get('dt', 1):.4f}",
        f"--encoding-duration {enc:.2f}",
        f"--eval-duration {enc+extra:.2f}",
    ]
    if best_arch["encoder"] == "poisson":
        cmd_parts += [
            f"--max-rate {bt.params.get('max_rate', 100):.2f}",
            f"--refractory-period {bt.params.get('refractory_period', 5):.4f}",
        ]
    cmd_parts.append(f"--datapath {args.datapath}")
    print("  " + " \\\n    ".join(cmd_parts))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Bayesian hyperparameter optimization for SparseSNN using Optuna (TPE sampler).

Outer loop: Optuna proposes hyperparameter configs and maximizes Hypervolume (HV)
            of the Pareto front produced by the inner MOEA run.
Inner loop: one call to ./build/platemo_cpp per trial.

Usage:
    python bayesian_search.py [options]

    # Quick sanity test (3 trials, tiny MOEA budget):
    python bayesian_search.py --nhidden 5 --popsize 20 --maxfe 300 --trials 3

    # Real search (tune with a reasonable MOEA budget):
    python bayesian_search.py --nhidden 10 --popsize 50 --maxfe 5000 --trials 50

Dependencies: optuna  (pip install optuna)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

BINARY = Path(__file__).parent / "build" / "platemo_cpp"

# Grillas para modo discreto: valores candidatos por parámetro.
# Editá estas listas si querés ampliar o reducir el espacio de búsqueda.
DISCRETE_GRIDS = {
    "disC":               [5, 10, 15, 20, 30, 50, 100],
    "disM":               [5, 10, 15, 20, 30, 50, 100],
    "disSM":              [5, 10, 15, 20, 30, 50, 100],
    "proM":               [0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
    "proSM":              [0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
    "sLower":             [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95],
    "sGap":               [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "wScale":             [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    "dt":                 [0.1, 0.25, 0.5, 1.0, 2.0],
    "encoding_duration":  [10, 20, 30, 50, 75, 100, 150],
    "extra_eval":         [10, 20, 50, 100, 150, 200],
    "max_rate":           [20, 50, 100, 200, 300, 500],
    "refractory_period":  [0.5, 1.0, 2.0, 5.0, 10.0, 15.0],
}


def run_binary(params: dict, fixed: dict, seed: int, timeout: int) -> float:
    """Run the C++ binary with given params and return HV (0.0 on failure)."""
    cmd = [str(BINARY)]
    for k, v in fixed.items():
        cmd += [f"--{k}", str(v)]
    cmd += ["--seed", str(seed)]

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
    cmd += ["--max-rate",          str(params["max_rate"])]
    cmd += ["--refractory-period", str(params["refractory_period"])]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT after {timeout}s]", file=sys.stderr)
        return 0.0
    except Exception as e:
        print(f"  [ERROR launching binary: {e}]", file=sys.stderr)
        return 0.0

    if result.returncode != 0:
        print(f"  [binary exit {result.returncode}]: {result.stderr[:200]}", file=sys.stderr)
        return 0.0

    m = re.search(r"HV\s*:\s*([\d.eE+\-]+)", result.stdout)
    if not m:
        print("  [could not parse HV from stdout]", file=sys.stderr)
        print(result.stdout[-400:], file=sys.stderr)
        return 0.0

    return float(m.group(1))


def build_objective(fixed: dict, timeout: int, mode: str = "continuous"):
    def objective(trial):
        params = {}

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
            valid_gaps = [g for g in DISCRETE_GRIDS["sGap"] if s_lower + g <= 1.0]
            if not valid_gaps:
                valid_gaps = [0.05]
            s_gap = trial.suggest_categorical("sGap", valid_gaps)

            params["wScale"] = cat("wScale")

            # SNN timing hyperparameters
            params["dt"]              = cat("dt")
            enc_dur                   = cat("encoding_duration")
            extra_eval                = cat("extra_eval")
            params["max_rate"]        = cat("max_rate")
            params["refractory_period"] = cat("refractory_period")
        else:
            # MOEA operator hyperparameters — continuous float search
            params["disC"]  = trial.suggest_float("disC",  5.0, 100.0, log=True)
            params["disM"]  = trial.suggest_float("disM",  5.0, 100.0, log=True)
            params["proM"]  = trial.suggest_float("proM",  0.3,   3.0)
            params["disSM"] = trial.suggest_float("disSM", 5.0, 100.0, log=True)
            params["proSM"] = trial.suggest_float("proSM", 0.3,   3.0)

            s_lower = trial.suggest_float("sLower", 0.30, 0.95)
            s_gap   = trial.suggest_float("sGap",   0.01, 1.0 - s_lower)

            params["wScale"] = trial.suggest_float("wScale", 0.1, 20.0, log=True)

            # SNN timing hyperparameters
            params["dt"]              = trial.suggest_float("dt", 0.1, 2.0)
            enc_dur                   = trial.suggest_float("encoding_duration", 10.0, 150.0)
            extra_eval                = trial.suggest_float("extra_eval", 10.0, 200.0)
            params["max_rate"]        = trial.suggest_float("max_rate", 20.0, 500.0, log=True)
            params["refractory_period"] = trial.suggest_float("refractory_period", 0.5, 15.0)

        params["sLower"] = s_lower
        params["sUpper"] = min(s_lower + s_gap, 1.0)
        params["encoding_duration"]   = enc_dur
        params["evaluation_duration"] = enc_dur + extra_eval  # always > encoding_duration

        hv = run_binary(params, fixed, seed=trial.number, timeout=timeout)
        return hv

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search for SparseSNN (maximizes HV)"
    )
    # Fixed MOEA settings (not tuned by BO)
    parser.add_argument("--algo",     default="SNSGAII",
                        help="Algorithm: SNSGAII or MOEACKF (default: SNSGAII)")
    parser.add_argument("--dataset",  type=int, default=1,
                        help="Dataset 1-4 (default: 1)")
    parser.add_argument("--nhidden",  type=int, default=10,
                        help="Hidden neurons (default: 10)")
    parser.add_argument("--popsize",  type=int, default=50,
                        help="Population size (default: 50)")
    parser.add_argument("--maxfe",    type=int, default=5000,
                        help="Max evaluations per inner MOEA run (default: 5000)")
    parser.add_argument("--datapath",
                        default=str(Path(__file__).parent / "data"),
                        help="Absolute or relative path to dataset CSVs (default: data)")
    # BO settings
    parser.add_argument("--trials",   type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--jobs",     type=int, default=1,
                        help="Parallel Optuna workers (default: 1; >1 requires SQLite storage)")
    parser.add_argument("--timeout",  type=int, default=7200,
                        help="Max seconds per inner MOEA run (default: 7200)")
    parser.add_argument("--study",    default="snn_bo",
                        help="Optuna study name (default: snn_bo)")
    parser.add_argument("--storage",  default="sqlite:///snn_hyperopt.db",
                        help="Optuna storage URI (default: sqlite:///snn_hyperopt.db)")
    parser.add_argument("--mode", choices=["continuous", "discrete"], default="continuous",
                        help="Search space type: 'continuous' (float ranges, default) or "
                             "'discrete' (categorical grids defined in DISCRETE_GRIDS)")
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"ERROR: binary not found at {BINARY}", file=sys.stderr)
        print("Build it first: cd build && cmake .. && make -j$(nproc)", file=sys.stderr)
        sys.exit(1)

    fixed = {
        "problem":  "SparseSNN",
        "algo":     args.algo,
        "dataset":  args.dataset,
        "nhidden":  args.nhidden,
        "popsize":  args.popsize,
        "maxfe":    args.maxfe,
        "runs":     1,
        "datapath": args.datapath,
    }

    print("=" * 60)
    print("Bayesian Hyperparameter Search — SparseSNN")
    print("=" * 60)
    print(f"  Algorithm  : {args.algo}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  nHidden    : {args.nhidden}")
    print(f"  PopSize    : {args.popsize}")
    print(f"  MaxFE      : {args.maxfe}")
    print(f"  BO trials  : {args.trials}")
    print(f"  Mode       : {args.mode}")
    print(f"  Storage    : {args.storage}")
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

    objective = build_objective(fixed, args.timeout, args.mode)

    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs,
                   show_progress_bar=True)

    print("\n" + "=" * 60)
    print("Best trial:")
    bt = study.best_trial
    print(f"  HV = {bt.value:.6f}")
    print("  Hyperparameters:")
    for k, v in bt.params.items():
        print(f"    {k:25s} = {v}")

    # Reconstruct actual evaluation_duration from stored params
    enc = bt.params.get("encoding_duration", 50.0)
    extra = bt.params.get("extra_eval", 50.0)
    print(f"\n  Note: evaluation_duration = encoding_duration + extra_eval"
          f" = {enc:.1f} + {extra:.1f} = {enc+extra:.1f} ms")

    print("\nTo reproduce the best run:")
    sL = bt.params.get("sLower", 0.75)
    sG = bt.params.get("sGap", 0.25)
    cmd_parts = [
        f"./build/platemo_cpp",
        f"--problem SparseSNN",
        f"--algo {args.algo}",
        f"--dataset {args.dataset}",
        f"--nhidden {args.nhidden}",
        f"--popsize {args.popsize}",
        f"--maxfe {args.maxfe}",
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
        f"--max-rate {bt.params.get('max_rate', 100):.2f}",
        f"--refractory-period {bt.params.get('refractory_period', 5):.4f}",
        f"--datapath {args.datapath}",
    ]
    print("  " + " \\\n    ".join(cmd_parts))


if __name__ == "__main__":
    main()

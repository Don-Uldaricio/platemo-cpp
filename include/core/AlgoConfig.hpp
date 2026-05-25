#pragma once

// Operator hyperparameters for MOEA algorithms (SNSGAII, MOEACKF).
// Passed from main() so they can be set via CLI or Bayesian optimization.
struct AlgoConfig {
    double disC   = 20.0;  // SBX distribution index (crossover spread)
    double disM   = 20.0;  // polynomial mutation distribution index (value mutation)
    double proM   = 1.0;   // polynomial mutation probability multiplier (rate = proM/D per gene)
    double disSM  = 20.0;  // sparsity mutation distribution index
    double proSM  = 1.0;   // sparsity mutation probability multiplier (rate = proSM/D per gene)
    double sLower = 0.75;  // VSSPS minimum sparsity (min fraction of zeros in initialization)
    double sUpper = 1.0;   // VSSPS maximum sparsity (max fraction of zeros in initialization)
};

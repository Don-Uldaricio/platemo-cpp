#pragma once
#include "../../utils/Random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// Mutación polinomial estándar (portada de Pymoo/PlatEMO).
// Perturba cada gen independientemente con una distribución polinomial sesgada hacia los bounds.
// genome: vector de genes a mutar.
// lb: bounds inferiores por gen.
// ub: bounds superiores por gen.
// eta: índice de distribución; mayor eta = perturbaciones más pequeñas (default típico: 20).
// Retorna: nuevo vector de genes mutados, garantizados dentro de [lb, ub].
inline std::vector<double> polyMutateCore(
    const std::vector<double>& genome,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    double eta)
{
    int n = genome.size();
    std::vector<double> result(n);
    double exp_ = 1.0 / (eta + 1.0);

    for (int i = 0; i < n; i++) {
        double g = genome[i];
        double lo = lb[i], hi = ub[i];
        double range = hi - lo;
        if (range < 1e-10) { result[i] = g; continue; }

        double delta1 = (g - lo) / range;
        double delta2 = (hi - g) / range;
        double ran = rng::uniform();
        double deltaq;

        if (ran < 0.5) {
            double xy  = 1.0 - delta1;
            double val = 2.0*ran + (1.0 - 2.0*ran) * std::pow(xy, eta+1.0);
            deltaq = std::pow(val, exp_) - 1.0;
        } else {
            double xy  = 1.0 - delta2;
            double val = 2.0*(1.0-ran) + 2.0*(ran-0.5) * std::pow(xy, eta+1.0);
            deltaq = 1.0 - std::pow(val, exp_);
        }

        result[i] = std::max(lo, std::min(hi, g + deltaq * range));
    }
    return result;
}

// Versión escalar de la mutación polinomial (muta un único gen).
// gene: valor actual del gen.
// lb: bound inferior.
// ub: bound superior.
// eta: índice de distribución.
// Retorna: nuevo valor del gen dentro de [lb, ub].
inline double polyMutateCoreScalar(double gene, double lb, double ub, double eta) {
    double range = ub - lb;
    if (range < 1e-10) return gene;
    double delta1 = (gene - lb) / range;
    double delta2 = (ub  - gene) / range;
    double ran = rng::uniform();
    double deltaq;
    double exp_ = 1.0 / (eta + 1.0);
    if (ran < 0.5) {
        double xy  = 1.0 - delta1;
        double val = 2.0*ran + (1.0 - 2.0*ran) * std::pow(xy, eta+1.0);
        deltaq = std::pow(val, exp_) - 1.0;
    } else {
        double xy  = 1.0 - delta2;
        double val = 2.0*(1.0-ran) + 2.0*(ran-0.5) * std::pow(xy, eta+1.0);
        deltaq = 1.0 - std::pow(val, exp_);
    }
    return std::max(lb, std::min(ub, gene + deltaq * range));
}

#pragma once
#include "polyMutateCore.hpp"
#include "sm2target.hpp"
#include "../../utils/Random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// Sparse Polynomial Mutation.
// Mutates both values (on non-zeros) and sparsity patterns.
inline Matrix spm(const Matrix& Pop,
                  const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                  double probMut   = 1.0, double distrMut  = 20.0,
                  double probSMut  = 1.0, double distrSMut = 20.0)
{
    int N = Pop.rows(), D = Pop.cols();
    Matrix newPop = Pop;

    // -- Value mutations on non-zero positions --
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            if (Pop(i,j) == 0.0) continue;
            if (rng::uniform() >= probMut / D) continue;
            newPop(i,j) = polyMutateCoreScalar(Pop(i,j), lb(j), ub(j), distrMut);
        }
    }

    // -- Sparsity mutations --
    std::vector<double> sparsities(N);
    for (int i = 0; i < N; i++) {
        int zCnt = 0;
        for (int j = 0; j < D; j++) if (Pop(i,j) == 0.0) zCnt++;
        sparsities[i] = (double)zCnt / D;
    }

    std::vector<double> newSparsities = sparsities;
    std::vector<bool> doMutate(N, false);
    for (int i = 0; i < N; i++) {
        if (rng::uniform() < probSMut / D) {
            doMutate[i] = true;
            newSparsities[i] = polyMutateCoreScalar(sparsities[i], 0.0, 1.0, distrSMut);
            newSparsities[i] = std::max(0.0, std::min(1.0, newSparsities[i]));
        }
    }

    // Check if any sparsity changed
    bool anyChanged = false;
    for (int i = 0; i < N; i++)
        if (doMutate[i] && std::abs(newSparsities[i] - sparsities[i]) > 1.0/D) {
            anyChanged = true; break;
        }

    if (anyChanged) {
        newPop = sm2target(newPop, lb, ub, newSparsities);
    }

    return newPop;
}

// Version with std::vector bounds
inline Matrix spm(const Matrix& Pop,
                  const std::vector<double>& lb_v, const std::vector<double>& ub_v,
                  double probMut = 1.0, double distrMut = 20.0,
                  double probSMut = 1.0, double distrSMut = 20.0)
{
    Eigen::VectorXd lb(lb_v.size()), ub(ub_v.size());
    for (int i = 0; i < (int)lb_v.size(); i++) { lb(i) = lb_v[i]; ub(i) = ub_v[i]; }
    return spm(Pop, lb, ub, probMut, distrMut, probSMut, distrSMut);
}

#pragma once
#include "../../core/Solution.hpp"
#include "../../utils/Random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// Ajusta la esparcidad de cada individuo de la población a un nivel objetivo.
// Para cada individuo, si necesita más ceros apaga pesos no-cero al azar;
// si necesita más no-ceros, asigna valores aleatorios en bounds a posiciones cero.
// Pop: Matrix N x D con los pesos actuales (cero = inactivo).
// lb: Vector D con bounds inferiores por variable.
// ub: Vector D con bounds superiores por variable.
// newSparsities: vector de N valores en [0,1] con la esparcidad objetivo por individuo
//               (fracción de ceros deseada).
// Retorna: Matrix N x D con los pesos ajustados al nivel de esparcidad objetivo.
inline Matrix sm2target(const Matrix& Pop,
                        const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                        const std::vector<double>& newSparsities)
{
    if (Pop.size() == 0) return Pop;

    int N = Pop.rows(), D = Pop.cols();
    Matrix newPop = Pop;

    // Current sparsities
    std::vector<double> sparsities(N);
    for (int i = 0; i < N; i++) {
        int zCnt = 0;
        for (int j = 0; j < D; j++) if (Pop(i,j) == 0.0) zCnt++;
        sparsities[i] = (double)zCnt / D;
    }

    for (int i = 0; i < N; i++) {
        if (std::abs(newSparsities[i] - sparsities[i]) < 1.0/D) continue;

        int nz2add = (int)std::round(D * (sparsities[i] - newSparsities[i]));

        if (nz2add > 0) {
            // Need more non-zeros: flip some zeros to non-zeros
            std::vector<int> zeroLocs;
            for (int j = 0; j < D; j++) if (Pop(i,j) == 0.0) zeroLocs.push_back(j);

            int numToFlip = std::min(nz2add, (int)zeroLocs.size());
            if (numToFlip <= 0) continue;

            auto toFlip = rng::randperm(zeroLocs.size(), numToFlip);
            for (int k : toFlip) {
                int j = zeroLocs[k];
                newPop(i,j) = lb(j) + rng::uniform() * (ub(j) - lb(j));
            }
        } else {
            // Need more zeros: flip some non-zeros to zeros
            std::vector<int> nzLocs;
            for (int j = 0; j < D; j++) if (Pop(i,j) != 0.0) nzLocs.push_back(j);

            int numToFlip = std::min(-nz2add, (int)nzLocs.size());
            if (numToFlip <= 0) continue;

            auto toFlip = rng::randperm(nzLocs.size(), numToFlip);
            for (int k : toFlip) {
                int j = nzLocs[k];
                newPop(i,j) = 0.0;
            }
        }
    }
    return newPop;
}

// Overload con esparcidad objetivo uniforme para todos los individuos.
// targetSparsity: fracción de ceros deseada (mismo valor para todas las filas).
inline Matrix sm2target(const Matrix& Pop,
                        const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                        double targetSparsity)
{
    int N = Pop.rows();
    std::vector<double> targets(N, targetSparsity);
    return sm2target(Pop, lb, ub, targets);
}

// Version with std::vector bounds
inline Matrix sm2target(const Matrix& Pop,
                        const std::vector<double>& lb_v, const std::vector<double>& ub_v,
                        const std::vector<double>& newSparsities)
{
    Eigen::VectorXd lb(lb_v.size()), ub(ub_v.size());
    for (int i = 0; i < (int)lb_v.size(); i++) { lb(i) = lb_v[i]; ub(i) = ub_v[i]; }
    return sm2target(Pop, lb, ub, newSparsities);
}

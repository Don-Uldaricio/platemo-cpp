#pragma once
#include "../../core/Solution.hpp"
#include "../../utils/Random.hpp"
#include <cmath>
#include <algorithm>

// Simulated Binary Crossover (SBX): operador de cruce para variables reales.
// Genera dos hijos interpolando entre dos padres con una distribución beta controlada por disC.
// Parent: Matrix 2N x D donde las primeras N filas son Parent1 y las siguientes N son Parent2.
// lb: Vector D con los bounds inferiores por variable.
// ub: Vector D con los bounds superiores por variable.
// proC: probabilidad de que ocurra crossover por individuo (default 1.0 = siempre).
// disC: índice de distribución SBX; mayor valor = hijos más cercanos a los padres (default 20).
// Retorna: Matrix 2N x D con los dos hijos por par (primeras N = Child1, siguientes N = Child2).
inline Matrix sbx(const Matrix& Parent,
                  const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                  double proC = 1.0, double disC = 20.0)
{
    int half = Parent.rows() / 2;
    Matrix P1 = Parent.topRows(half);
    Matrix P2 = Parent.bottomRows(half);
    int N = P1.rows(), D = P1.cols();

    Matrix beta(N, D);
    Matrix mu = rng::rand(N, D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++) {
            if (mu(i,j) <= 0.5)
                beta(i,j) = std::pow(2.0*mu(i,j), 1.0/(disC+1.0));
            else
                beta(i,j) = std::pow(2.0 - 2.0*mu(i,j), -1.0/(disC+1.0));
        }

    // Random sign flip
    auto signs = rng::randi(0, 1, N, D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            beta(i,j) *= (signs(i,j) == 0 ? -1.0 : 1.0);

    // Half of positions: beta = 1
    Matrix r1 = rng::rand(N, D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (r1(i,j) < 0.5) beta(i,j) = 1.0;

    // Per-individual crossover probability
    Matrix r2 = rng::rand(N, 1);
    for (int i = 0; i < N; i++)
        if (r2(i,0) > proC)
            for (int j = 0; j < D; j++) beta(i,j) = 1.0;

    Matrix mid  = (P1 + P2) / 2.0;
    Matrix diff = (P1 - P2) / 2.0;

    Matrix C1 = mid + beta.cwiseProduct(diff);
    Matrix C2 = mid - beta.cwiseProduct(diff);

    Matrix Offspring(2*N, D);
    Offspring.topRows(N)    = C1;
    Offspring.bottomRows(N) = C2;

    // Clamp to bounds
    for (int j = 0; j < D; j++) {
        Offspring.col(j) = Offspring.col(j).cwiseMax(lb(j)).cwiseMin(ub(j));
    }
    return Offspring;
}

// Overload de sbx con bounds como std::vector en lugar de Eigen::VectorXd.
inline Matrix sbx(const Matrix& Parent,
                  const std::vector<double>& lb, const std::vector<double>& ub,
                  double proC = 1.0, double disC = 20.0)
{
    Eigen::VectorXd lb_(lb.size()), ub_(ub.size());
    for (int i = 0; i < (int)lb.size(); i++) { lb_(i) = lb[i]; ub_(i) = ub[i]; }
    return sbx(Parent, lb_, ub_, proC, disC);
}

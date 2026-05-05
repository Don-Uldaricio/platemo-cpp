#pragma once
#include "../../core/Problem.hpp"
#include "BinaryCrossover.hpp"
#include "BinaryMutation.hpp"
#include "../../utils/Random.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace r2detail {

// SBX half crossover: produces N offspring from 2N parents (first N x second N)
inline Matrix SBXhalf(const Matrix& Parent1, const Matrix& Parent2,
                      double proC = 1.0, double disC = 20.0) {
    int N = Parent1.rows(), D = Parent1.cols();
    Matrix beta(N, D);
    Matrix mu = rng::rand(N, D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++) {
            if (mu(i,j) <= 0.5)
                beta(i,j) = std::pow(2.0 * mu(i,j), 1.0/(disC+1.0));
            else
                beta(i,j) = std::pow(2.0 - 2.0*mu(i,j), -1.0/(disC+1.0));
        }

    // Random sign
    auto signs = rng::randi(0, 1, N, D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            beta(i,j) *= (signs(i,j) == 0) ? -1.0 : 1.0;

    // Half of beta replaced with 1
    Matrix r1 = rng::rand(N, D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (r1(i,j) < 0.5) beta(i,j) = 1.0;

    Matrix r2 = rng::rand(N, 1);
    for (int i = 0; i < N; i++)
        if (r2(i,0) > proC)
            for (int j = 0; j < D; j++) beta(i,j) = 1.0;

    return (Parent1 + Parent2) / 2.0 + beta.cwiseProduct(Parent1 - Parent2) / 2.0;
}

// Polynomial mutation on subset of columns (indexSite is bool vector)
inline Matrix PM_subset(Problem& prob, Matrix Offspring, const std::vector<bool>& indexSite,
                        double proM = 1.0, double disM = 20.0) {
    int N = Offspring.rows(), D_sub = Offspring.cols();

    // Build lower/upper for the selected subset
    Matrix Lower(N, D_sub), Upper(N, D_sub);
    int col = 0;
    for (int j = 0; j < prob.D; j++) {
        if (!indexSite[j]) continue;
        for (int i = 0; i < N; i++) {
            Lower(i, col) = prob.lower(j);
            Upper(i, col) = prob.upper(j);
        }
        col++;
    }

    Offspring = Offspring.cwiseMax(Lower).cwiseMin(Upper);

    Matrix mu = rng::rand(N, D_sub);
    Matrix Site = rng::rand(N, D_sub);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < D_sub; j++) {
            if (Site(i,j) >= proM / D_sub) continue;
            double delta;
            if (mu(i,j) <= 0.5) {
                double xy = 1.0 - (Offspring(i,j) - Lower(i,j)) / (Upper(i,j) - Lower(i,j) + 1e-10);
                delta = std::pow(2.0*mu(i,j) + (1.0-2.0*mu(i,j))*std::pow(xy, disM+1.0), 1.0/(disM+1.0)) - 1.0;
            } else {
                double xy = 1.0 - (Upper(i,j) - Offspring(i,j)) / (Upper(i,j) - Lower(i,j) + 1e-10);
                delta = 1.0 - std::pow(2.0*(1.0-mu(i,j)) + 2.0*(mu(i,j)-0.5)*std::pow(xy, disM+1.0), 1.0/(disM+1.0));
            }
            Offspring(i,j) += (Upper(i,j) - Lower(i,j)) * delta;
        }
    return Offspring;
}

} // namespace r2detail

struct Reproduction2Result {
    Matrix  OffDec;
    MatrixB OffMask;
};

// Dual-grouping reproduction for MOEA-CKF
Reproduction2Result Reproduction2(
    Problem& prob,
    const Matrix& ParentDec, const MatrixB& ParentMask,
    const std::vector<bool>& NSV, const std::vector<bool>& SV,
    bool isReal)
{
    int half = ParentDec.rows() / 2;
    int N    = half;
    int D    = ParentDec.cols();

    MatrixB Parent1Mask = ParentMask.topRows(half);
    MatrixB Parent2Mask = ParentMask.bottomRows(half);

    // Binary variation on SV and NSV groups
    MatrixB OffMask = MatrixB::Zero(N, D);

    // Collect SV column indices
    std::vector<int> svIdx, nsvIdx;
    for (int j = 0; j < D; j++) {
        if (SV[j])  svIdx.push_back(j);
        if (NSV[j]) nsvIdx.push_back(j);
    }

    auto applyGroup = [&](const std::vector<int>& idx) {
        if (idx.empty()) return;
        MatrixB p1(N, idx.size()), p2(N, idx.size());
        for (int i = 0; i < N; i++)
            for (int k = 0; k < (int)idx.size(); k++) {
                p1(i,k) = Parent1Mask(i, idx[k]);
                p2(i,k) = Parent2Mask(i, idx[k]);
            }
        MatrixB off = BinaryCrossover(p1, p2);
        off = BinaryMutation(off);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < (int)idx.size(); k++)
                OffMask(i, idx[k]) = off(i,k);
    };

    applyGroup(svIdx);
    applyGroup(nsvIdx);

    Reproduction2Result res;
    res.OffMask = OffMask;

    if (!isReal) {
        res.OffDec = Matrix::Ones(N, D);
        return res;
    }

    Matrix Parent1Dec = ParentDec.topRows(half);
    Matrix Parent2Dec = ParentDec.bottomRows(half);
    res.OffDec = Matrix::Zero(N, D);

    for (int i = 0; i < N; i++) {
        // For each offspring, crossover values at NonZero and Zero positions separately
        std::vector<int> nzIdx, zIdx;
        for (int j = 0; j < D; j++) {
            if (OffMask(i,j)) nzIdx.push_back(j);
            else              zIdx.push_back(j);
        }

        auto crossoverGroup = [&](const std::vector<int>& gIdx) {
            if (gIdx.empty()) return;
            Matrix p1row(1, gIdx.size()), p2row(1, gIdx.size());
            std::vector<bool> site(D, false);
            for (int k : gIdx) site[k] = true;

            for (int k = 0; k < (int)gIdx.size(); k++) {
                p1row(0,k) = Parent1Dec(i, gIdx[k]);
                p2row(0,k) = Parent2Dec(i, gIdx[k]);
            }
            Matrix child = r2detail::SBXhalf(p1row, p2row);
            child = r2detail::PM_subset(prob, child, site);
            for (int k = 0; k < (int)gIdx.size(); k++)
                res.OffDec(i, gIdx[k]) = child(0,k);
        };

        crossoverGroup(nzIdx);
        crossoverGroup(zIdx);
    }

    return res;
}

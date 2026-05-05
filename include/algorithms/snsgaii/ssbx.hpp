#pragma once
#include "sbx.hpp"
#include "sm2target.hpp"
#include "../../utils/Random.hpp"

// Sparse Simulated Binary Crossover.
// Handles zero/non-zero patterns: applies SBX to matching positions,
// randomly swaps mismatched zero/non-zero positions.
inline Matrix ssbx(const Matrix& Parent,
                   const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
                   double proC = 1.0, double disC = 20.0)
{
    int half = Parent.rows() / 2;
    Matrix P1 = Parent.topRows(half);
    Matrix P2 = Parent.bottomRows(half);
    int N = P1.rows(), D = P1.cols();

    Matrix Off1 = Matrix::Constant(N, D, -99.0);
    Matrix Off2 = Matrix::Constant(N, D, -99.0);

    // Step 1: SBX on positions that are both zero or both non-zero
    for (int i = 0; i < N; i++) {
        std::vector<int> matching, notMatching;
        for (int j = 0; j < D; j++) {
            bool z1 = (P1(i,j) == 0.0), z2 = (P2(i,j) == 0.0);
            if (z1 == z2) matching.push_back(j);
            else          notMatching.push_back(j);
        }

        if (!matching.empty()) {
            Matrix subP(2, matching.size());
            Eigen::VectorXd subLb(matching.size()), subUb(matching.size());
            for (int k = 0; k < (int)matching.size(); k++) {
                subP(0,k) = P1(i, matching[k]);
                subP(1,k) = P2(i, matching[k]);
                subLb(k)  = lb(matching[k]);
                subUb(k)  = ub(matching[k]);
            }
            Matrix sbxOff = sbx(subP, subLb, subUb, proC, disC);
            for (int k = 0; k < (int)matching.size(); k++) {
                Off1(i, matching[k]) = sbxOff(0,k);
                Off2(i, matching[k]) = sbxOff(1,k);
            }
        }

        // Step 2: swap mismatch positions with random ratio
        if (!notMatching.empty()) {
            double z2nzRatio = rng::uniform();

            // Determine which mismatched positions to swap
            std::vector<double> swapMask(notMatching.size(), 0.0);
            int nSwap = (int)std::round(z2nzRatio * notMatching.size());
            auto toSwap = rng::randperm(notMatching.size(), nSwap);
            for (int k : toSwap) swapMask[k] = 1.0;

            for (int k = 0; k < (int)notMatching.size(); k++) {
                int j = notMatching[k];
                if (swapMask[k] == 1.0) {
                    Off1(i,j) = P2(i,j);
                    Off2(i,j) = P1(i,j);
                } else {
                    Off1(i,j) = P1(i,j);
                    Off2(i,j) = P2(i,j);
                }
            }
        }
    }

    Matrix result(2*N, D);
    result.topRows(N)    = Off1;
    result.bottomRows(N) = Off2;
    return result;
}

// Version with std::vector bounds
inline Matrix ssbx(const Matrix& Parent,
                   const std::vector<double>& lb_v, const std::vector<double>& ub_v,
                   double proC = 1.0, double disC = 20.0)
{
    Eigen::VectorXd lb(lb_v.size()), ub(ub_v.size());
    for (int i = 0; i < (int)lb_v.size(); i++) { lb(i) = lb_v[i]; ub(i) = ub_v[i]; }
    return ssbx(Parent, lb, ub, proC, disC);
}

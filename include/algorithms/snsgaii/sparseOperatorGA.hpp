#pragma once
#include "../../core/Problem.hpp"
#include "ssbx.hpp"
#include "spm.hpp"
#include <functional>

// Sparse GA operator: applies crossover then mutation.
// crossoverFn: (Parent, lb, ub, proC, disC) -> Offspring (2N x D)
// mutationFn:  (Pop, lb, ub, probMut, distrMut, probSMut, distrSMut) -> Pop (N x D)
inline Population sparseOperatorGA(
    Problem& prob,
    const Population& parent,
    double proC = 1.0, double disC = 20.0,
    double proM = 1.0, double disM = 20.0,
    double proSM = 1.0, double disSM = 20.0,
    bool useSsbx = true)
{
    Matrix ParentDec = getDecs(parent);
    int D = ParentDec.cols();

    // Extract bounds
    std::vector<double> lb(D), ub(D);
    for (int j = 0; j < D; j++) {
        lb[j] = prob.lower(j);
        ub[j] = prob.upper(j);
    }

    // Crossover
    Matrix offspring;
    if (useSsbx) {
        offspring = ssbx(ParentDec, lb, ub, proC, disC);
    } else {
        offspring = sbx(ParentDec, lb, ub, proC, disC);
    }

    // Mutation
    offspring = spm(offspring, lb, ub, proM, disM, proSM, disSM);

    return prob.Evaluation(offspring);
}

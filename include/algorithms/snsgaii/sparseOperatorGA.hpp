#pragma once
#include "../../core/Problem.hpp"
#include "ssbx.hpp"
#include "spm.hpp"
#include <functional>

// Operador genético completo para problemas esparcidos: aplica cruce y luego mutación.
// Combina ssbx (o sbx) + spm y evalúa el resultado con prob.Evaluation().
// prob: problema que provee los bounds y el método Evaluation.
// parent: mating pool de 2N individuos (los pares están en [0..N-1] y [N..2N-1]).
// proC: probabilidad de crossover (default 1.0).
// disC: índice de distribución del cruce (default 20).
// proM: probabilidad de mutación de valores (default 1.0).
// disM: índice de distribución de mutación de valores (default 20).
// proSM: probabilidad de mutación de esparcidad (default 1.0).
// disSM: índice de distribución de mutación de esparcidad (default 20).
// useSsbx: si true usa SSBX (sparse-aware), si false usa SBX estándar (default true).
// Retorna: Population de N hijos evaluados.
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

#pragma once
#include "../../core/Problem.hpp"
#include "../../core/AlgoConfig.hpp"
#include "../../utils/TournamentSelection.hpp"
#include "EnvironmentalSelection.hpp"
#include "sparseOperatorGA.hpp"
#include "vssps.hpp"
#include <iostream>

// S-NSGA-II: Sparse Non-dominated Sorting Genetic Algorithm II.
// Algoritmo evolutivo multi-objetivo diseñado para problemas esparcidos.
// Referencia: I. Kropp et al., IEEE TEVC 2024.
// Flujo por generación:
//   1. Selección por torneo 2-way con (frontNo, -crowdDis) como criterios.
//   2. Crossover esparcido (SSBX) + mutación esparcida (SPM).
//   3. Selección ambiental NSGA-II (merge padres+hijos → top N).
// prob: problema a optimizar (define N, D, maxFE, Evaluation, etc.).
// verbose: si true, imprime FE y tamaño de población en cada generación.
// Retorna: Population final de N soluciones al agotar el presupuesto de evaluaciones.
Population SNSGAII(Problem& prob, bool verbose = false,
                   const AlgoConfig& acfg = AlgoConfig{}) {
    // Initial population via VSSPS
    Population Population_ = vssps(prob, acfg.sLower, acfg.sUpper);

    auto [pop, FrontNo, CrowdDis] = EnvironmentalSelection_NSGA2(Population_, prob.N);
    Population_ = pop;

    while (prob.NotTerminated(Population_)) {
        // Tournament selection — negate CrowdDis so higher diversity is preferred (NSGA-II semantics)
        std::vector<double> negCrowdDis(CrowdDis.size());
        for (int i = 0; i < (int)CrowdDis.size(); i++) negCrowdDis[i] = -CrowdDis[i];
        auto matingIdx = TournamentSelection(2, prob.N, FrontNo, negCrowdDis);

        // Build mating pool
        Population matingPool;
        matingPool.reserve(prob.N);
        for (int idx : matingIdx) matingPool.push_back(Population_[idx]);

        // Sparse crossover + mutation
        Population offspring = sparseOperatorGA(prob, matingPool,
                                                1.0,        acfg.disC,
                                                acfg.proM,  acfg.disM,
                                                acfg.proSM, acfg.disSM,
                                                true); // use ssbx

        // Merge and select
        Population merged = concat(Population_, offspring);
        auto [newPop, newFront, newCrowd] = EnvironmentalSelection_NSGA2(merged, prob.N);
        Population_ = newPop;
        FrontNo    = newFront;
        CrowdDis   = newCrowd;

        if (verbose)
            std::cout << "FE=" << prob.FE << " popSize=" << Population_.size() << "\n";
        prob.MaybeLog(Population_);
    }
    return Population_;
}

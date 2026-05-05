#pragma once
#include "../../core/Problem.hpp"
#include "../../utils/TournamentSelection.hpp"
#include "EnvironmentalSelection.hpp"
#include "sparseOperatorGA.hpp"
#include "vssps.hpp"
#include <iostream>

// S-NSGA-II: Sparse Non-dominated Sorting Genetic Algorithm II
// Reference: I. Kropp et al., IEEE TEVC 2024
Population SNSGAII(Problem& prob, bool verbose = false) {
    // Initial population via VSSPS
    Population Population_ = vssps(prob, 0.75, 1.0);

    auto [pop, FrontNo, CrowdDis] = EnvironmentalSelection_NSGA2(Population_, prob.N);
    Population_ = pop;

    while (prob.NotTerminated(Population_)) {
        // Tournament selection
        auto matingIdx = TournamentSelection(2, prob.N, FrontNo, CrowdDis);

        // Build mating pool
        Population matingPool;
        matingPool.reserve(prob.N);
        for (int idx : matingIdx) matingPool.push_back(Population_[idx]);

        // Sparse crossover + mutation
        Population offspring = sparseOperatorGA(prob, matingPool,
                                                1.0, 20.0,
                                                1.0, 20.0,
                                                1.0, 20.0,
                                                true); // use ssbx

        // Merge and select
        Population merged = concat(Population_, offspring);
        auto [newPop, newFront, newCrowd] = EnvironmentalSelection_NSGA2(merged, prob.N);
        Population_ = newPop;
        FrontNo    = newFront;
        CrowdDis   = newCrowd;

        if (verbose)
            std::cout << "FE=" << prob.FE << " popSize=" << Population_.size() << "\n";
    }
    return Population_;
}

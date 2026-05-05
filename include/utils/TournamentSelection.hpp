#pragma once
#include "Random.hpp"
#include <vector>
#include <algorithm>
#include <numeric>

// K-tournament selection.
// Returns N winner indices (0-based).
// Selects the candidate with lexicographically smallest fitness tuple.
// fitness: vector of fitness arrays, one per criterion (lower is better).
// E.g., TournamentSelection(2, 100, {frontNo}, {-crowdDis})
inline std::vector<int> TournamentSelection(int K, int N, const std::vector<std::vector<double>>& fitness) {
    if (fitness.empty() || fitness[0].empty()) return {};

    int popSize = fitness[0].size();
    int nCrit   = fitness.size();

    // Build lexicographic rank: unique(fitness rows) -> rank
    std::vector<std::vector<double>> fitnessRows(popSize, std::vector<double>(nCrit));
    for (int i = 0; i < popSize; i++)
        for (int c = 0; c < nCrit; c++)
            fitnessRows[i][c] = fitness[c][i];

    // Sort indices by fitness rows lexicographically
    std::vector<int> sortedIdx(popSize);
    std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
    std::sort(sortedIdx.begin(), sortedIdx.end(), [&](int a, int b){
        return fitnessRows[a] < fitnessRows[b];
    });

    // Assign ranks (1-based, ties get same rank)
    std::vector<int> rank(popSize);
    rank[sortedIdx[0]] = 1;
    for (int i = 1; i < popSize; i++) {
        if (fitnessRows[sortedIdx[i]] == fitnessRows[sortedIdx[i-1]])
            rank[sortedIdx[i]] = rank[sortedIdx[i-1]];
        else
            rank[sortedIdx[i]] = i + 1;
    }

    std::vector<int> winners(N);
    for (int t = 0; t < N; t++) {
        int best = -1;
        for (int k = 0; k < K; k++) {
            int cand = rng::randint(0, popSize-1);
            if (best < 0 || rank[cand] < rank[best]) best = cand;
        }
        winners[t] = best;
    }
    return winners;
}

// Convenience: single fitness criterion
inline std::vector<int> TournamentSelection(int K, int N, const std::vector<double>& fit) {
    return TournamentSelection(K, N, std::vector<std::vector<double>>{fit});
}

// Two criteria
inline std::vector<int> TournamentSelection(int K, int N,
    const std::vector<double>& fit1, const std::vector<double>& fit2) {
    return TournamentSelection(K, N, std::vector<std::vector<double>>{fit1, fit2});
}

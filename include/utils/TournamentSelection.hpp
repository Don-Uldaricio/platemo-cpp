#pragma once
#include "Random.hpp"
#include <vector>
#include <algorithm>
#include <numeric>

// Selección por torneo de tamaño K que produce N ganadores.
// En cada torneo, se eligen K candidatos aleatorios y gana el de menor rank lexicográfico
// sobre los criterios de fitness dados. Menor valor = mejor en cada criterio.
// K: tamaño del torneo (típicamente 2).
// N: número de ganadores a producir (con reemplazo).
// fitness: vector de criterios; cada criterio es un vector de tamaño popSize.
//          Ejemplo: TournamentSelection(2, 100, {frontNo, negCrowdDis})
// Retorna: vector de N índices (0-based) de los ganadores de cada torneo.
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

// Overload para un único criterio de fitness.
// fit: vector de tamaño popSize con el fitness de cada solución (menor = mejor).
inline std::vector<int> TournamentSelection(int K, int N, const std::vector<double>& fit) {
    return TournamentSelection(K, N, std::vector<std::vector<double>>{fit});
}

// Overload para dos criterios de fitness (caso típico de NSGA-II: frente + crowding negado).
// fit1: primer criterio (ej. frontNo), fit2: segundo criterio (ej. -crowdDis).
inline std::vector<int> TournamentSelection(int K, int N,
    const std::vector<double>& fit1, const std::vector<double>& fit2) {
    return TournamentSelection(K, N, std::vector<std::vector<double>>{fit1, fit2});
}

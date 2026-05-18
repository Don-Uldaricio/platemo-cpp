#pragma once
#include "../../core/Solution.hpp"
#include "../../utils/NDSort.hpp"
#include "../../utils/CrowdingDistance.hpp"
#include <vector>
#include <algorithm>
#include <numeric>

struct EnvSelNSGA2Result {
    Population pop;
    std::vector<double> FrontNo;
    std::vector<double> CrowdDis;
};

// Selección ambiental de NSGA-II: reduce la población combinada (padres + hijos) a N individuos.
// Estrategia: incluye todos los frentes completos que quepan en N; del último frente
// rellena los lugares restantes eligiendo los de mayor distancia de crowding.
// pop: población combinada (padres + hijos), tamaño > N.
// N: tamaño objetivo de la siguiente generación.
// Retorna: EnvSelNSGA2Result con la población seleccionada, sus números de frente y distancias de crowding.
EnvSelNSGA2Result EnvironmentalSelection_NSGA2(const Population& pop, int N) {
    Matrix objs = getObjs(pop);
    Matrix cons = getCons(pop);

    auto [FrontNo, MaxFNo] = NDSort(objs, cons, N);
    auto CrowdDis = CrowdingDistance(objs, FrontNo);

    int totalN = pop.size();
    std::vector<bool> Next(totalN, false);

    // Select all solutions in fronts < MaxFNo
    for (int i = 0; i < totalN; i++)
        if (!std::isinf(FrontNo[i]) && (int)std::round(FrontNo[i]) < MaxFNo) Next[i] = true;

    int selected = std::count(Next.begin(), Next.end(), true);
    int remaining = N - selected;

    // Fill from last front by crowding distance (descending)
    std::vector<int> lastFront;
    for (int i = 0; i < totalN; i++)
        if ((int)FrontNo[i] == MaxFNo) lastFront.push_back(i);

    if (remaining > 0 && !lastFront.empty()) {
        std::sort(lastFront.begin(), lastFront.end(), [&](int a, int b){
            return CrowdDis[a] > CrowdDis[b];
        });
        for (int k = 0; k < remaining && k < (int)lastFront.size(); k++)
            Next[lastFront[k]] = true;
    }

    Population nextPop;
    std::vector<double> nextFront, nextCrowd;
    for (int i = 0; i < totalN; i++) {
        if (Next[i]) {
            nextPop.push_back(pop[i]);
            nextFront.push_back(FrontNo[i]);
            nextCrowd.push_back(CrowdDis[i]);
        }
    }

    return {nextPop, nextFront, nextCrowd};
}

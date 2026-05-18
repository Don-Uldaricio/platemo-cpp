#pragma once
#include "../core/Solution.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <set>

// Calcula la distancia de crowding de NSGA-II para cada solución, frente por frente.
// Mide qué tan aislada está una solución respecto a sus vecinas dentro de su frente.
// PopObj: Matrix N x M con los objetivos de cada solución.
// FrontNo: vector de tamaño N con el número de frente 1-indexado de cada solución (inf = no asignada).
// Retorna: vector de N doubles con la distancia de crowding de cada solución.
//          Las soluciones extremas (mínimo/máximo de algún objetivo) reciben distancia infinita.
inline std::vector<double> CrowdingDistance(const Matrix& PopObj, const std::vector<double>& FrontNo) {
    int N = PopObj.rows(), M = PopObj.cols();
    std::vector<double> CrowdDis(N, 0.0);

    // Collect unique finite front numbers
    std::set<double> frontSet;
    for (double f : FrontNo)
        if (!std::isinf(f)) frontSet.insert(f);

    for (double frontId : frontSet) {
        std::vector<int> Front;
        for (int i = 0; i < N; i++)
            if (FrontNo[i] == frontId) Front.push_back(i);

        int Fsize = Front.size();
        if (Fsize < 3) {
            for (int idx : Front) CrowdDis[idx] = std::numeric_limits<double>::infinity();
            continue;
        }

        for (int m = 0; m < M; m++) {
            // Sort front by objective m
            std::vector<int> rank(Fsize);
            std::iota(rank.begin(), rank.end(), 0);
            std::sort(rank.begin(), rank.end(), [&](int a, int b){
                return PopObj(Front[a], m) < PopObj(Front[b], m);
            });

            double fmax = PopObj(Front[rank.back()], m);
            double fmin = PopObj(Front[rank[0]], m);
            double range = fmax - fmin;

            CrowdDis[Front[rank[0]]]    = std::numeric_limits<double>::infinity();
            CrowdDis[Front[rank.back()]] = std::numeric_limits<double>::infinity();

            if (range < 1e-10) continue;

            for (int j = 1; j < Fsize-1; j++) {
                double span = PopObj(Front[rank[j+1]], m) - PopObj(Front[rank[j-1]], m);
                CrowdDis[Front[rank[j]]] += span / range;
            }
        }
    }

    return CrowdDis;
}

// Overload sin números de frente: trata toda la población como un único frente.
// Útil cuando ya se filtró la población a un solo frente.
// PopObj: Matrix N x M.
// Retorna: vector de N doubles con la distancia de crowding de cada solución.
inline std::vector<double> CrowdingDistance(const Matrix& PopObj) {
    int N = PopObj.rows();
    std::vector<double> FrontNo(N, 1.0);
    return CrowdingDistance(PopObj, FrontNo);
}

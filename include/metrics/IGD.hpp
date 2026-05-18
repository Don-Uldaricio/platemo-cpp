#pragma once
#include "../core/Solution.hpp"
#include "../utils/NDSort.hpp"
#include <limits>
#include <cmath>

// Calcula el Inverted Generational Distance (IGD): para cada punto de referencia del frente óptimo,
// encuentra la solución más cercana en el frente actual, y promedia todas esas distancias mínimas.
// Menor IGD = frente más cercano al óptimo real.
// pop: población de la que se extrae el frente de Pareto actual (solo soluciones factibles no-dominadas).
// optimum: Matrix nRef x M con los puntos de referencia del frente óptimo.
// Retorna: IGD como double (0 = perfecto, inf si no hay soluciones factibles).
inline double IGD(const Population& pop, const Matrix& optimum) {
    Population best = getBest(pop);
    if (best.empty()) return std::numeric_limits<double>::infinity();

    Matrix PopObj = getObjs(best);
    int M = PopObj.cols();

    if (M != optimum.cols()) return std::numeric_limits<double>::quiet_NaN();

    // pdist2(optimum, PopObj) -> min over PopObj rows for each optimum row
    int nRef = optimum.rows();
    double total = 0.0;
    for (int i = 0; i < nRef; i++) {
        double minDist = std::numeric_limits<double>::infinity();
        for (int j = 0; j < PopObj.rows(); j++) {
            double d = (optimum.row(i) - PopObj.row(j)).norm();
            if (d < minDist) minDist = d;
        }
        total += minDist;
    }
    return total / nRef;
}

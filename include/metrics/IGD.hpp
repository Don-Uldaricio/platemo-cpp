#pragma once
#include "../core/Solution.hpp"
#include "../utils/NDSort.hpp"
#include <limits>
#include <cmath>

// Inverted Generational Distance
// Measures mean distance from each reference point to the nearest solution
// Lower is better
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

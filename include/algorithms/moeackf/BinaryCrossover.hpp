#pragma once
#include "../../core/Solution.hpp"
#include "../../utils/Random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// Crossover binario no-balanceado sobre máscaras de esparcidad.
// En las posiciones donde los padres difieren, cada bit del hijo se invierte con una probabilidad
// calculada para preservar la densidad promedio de unos. Las posiciones iguales se heredan directamente.
// Parent1: MatrixB N x D con la máscara del primer padre.
// Parent2: MatrixB N x D con la máscara del segundo padre.
// Retorna: MatrixB N x D con la máscara del hijo (inicia como copia de Parent1).
inline MatrixB BinaryCrossover(const MatrixB& Parent1, const MatrixB& Parent2) {
    int N = Parent1.rows(), D = Parent1.cols();
    MatrixB Offspring = Parent1;

    for (int i = 0; i < N; i++) {
        // Find positions where parents differ
        std::vector<int> diff;
        for (int j = 0; j < D; j++)
            if (Parent1(i,j) != Parent2(i,j)) diff.push_back(j);

        if (diff.empty()) continue;

        // Count how many of the diff positions are 1 in offspring (from Parent1)
        int oneCount = 0;
        for (int j : diff) if (Offspring(i,j)) oneCount++;
        double MOne = (diff.empty()) ? 0.5 : (double)oneCount / diff.size();

        double r;
        if (MOne < 1e-10) r = 0.0;
        else if (1.0 - MOne < 1e-10) r = 0.0;
        else r = std::min(0.5, std::min(2.0*MOne, 2.0*(1.0-MOne)));

        for (int j : diff) {
            double rate;
            if (Offspring(i,j)) {
                rate = (MOne > 1e-10) ? r / 2.0 / MOne : 0.0;
            } else {
                rate = (1.0 - MOne > 1e-10) ? r / 2.0 / (1.0 - MOne) : 0.0;
            }
            if (rng::uniform() < rate)
                Offspring(i,j) = !Offspring(i,j);
        }
    }
    return Offspring;
}

// Overload for single row (1 x D)
inline MatrixB BinaryCrossoverRow(const MatrixB& p1row, const MatrixB& p2row) {
    return BinaryCrossover(p1row, p2row);
}

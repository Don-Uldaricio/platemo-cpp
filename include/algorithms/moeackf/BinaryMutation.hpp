#pragma once
#include "../../core/Solution.hpp"
#include "../../utils/Random.hpp"
#include <cmath>
#include <algorithm>

// Mutación binaria no-balanceada sobre máscaras de esparcidad.
// Para cada individuo, calcula tasas de mutación diferenciadas para unos y ceros
// de forma que la densidad promedio se preserve estadísticamente.
// La tasa efectiva por bit es r = min(1/D, 2·MOne, 2·(1-MOne)), donde MOne es la densidad actual.
// Offspring_in: MatrixB N x D con las máscaras a mutar.
// Retorna: MatrixB N x D con las máscaras mutadas.
inline MatrixB BinaryMutation(const MatrixB& Offspring_in) {
    int N = Offspring_in.rows(), D = Offspring_in.cols();
    MatrixB Offspring = Offspring_in;

    for (int i = 0; i < N; i++) {
        int oneCount = 0;
        for (int j = 0; j < D; j++) if (Offspring(i,j)) oneCount++;
        double MOne = (double)oneCount / D;

        double r;
        if (MOne < 1e-10 || (1.0 - MOne) < 1e-10)
            r = 1.0 / D;
        else
            r = std::min(1.0/D, std::min(2.0*MOne, 2.0*(1.0-MOne)));

        double rate1 = (MOne > 1e-10) ? r / 2.0 / MOne : 0.0;
        double rate0 = ((1.0-MOne) > 1e-10) ? r / 2.0 / (1.0-MOne) : 0.0;

        for (int j = 0; j < D; j++) {
            double rate = Offspring(i,j) ? rate1 : rate0;
            if (rng::uniform() < rate)
                Offspring(i,j) = !Offspring(i,j);
        }
    }
    return Offspring;
}

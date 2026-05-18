#pragma once
#include "../../core/Solution.hpp"
#include "../../utils/KMeans.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

struct SparsityKnowledge {
    std::vector<bool> allOne;   // variables that are always 1 in elite
    std::vector<bool> allZero;  // variables that are always 0 in elite
    std::vector<bool> other;    // the rest
};

struct SparsityResult {
    SparsityKnowledge Local;    // based on elite (front 1)
    SparsityKnowledge Global;   // based on whole population
    std::vector<double> Fitness;
    std::vector<bool> NSV;      // non-significant variables
    std::vector<bool> SV;       // significant variables
    double theta;
};

// Analiza la esparcidad de la población actual para clasificar variables en grupos.
// Se ejecuta en cada generación del MOEA-CKF para actualizar el conocimiento sobre qué variables
// son importantes (SV) y cuáles no (NSV).
// Pasos:
//   1. Extrae el "conocimiento local" del frente 1 (elite): detecta variables siempre activas,
//      siempre inactivas, o mixtas.
//   2. Extrae el "conocimiento global" de toda la población (mismos tres casos).
//   3. Normaliza el Fitness acumulado y aplica un ajuste por frecuencia de aparición en la elite.
//   4. Clasifica con k-means (k=2) en variables significativas (SV) y no-significativas (NSV).
// prob: problema actual (se usa prob.FE y prob.maxFE para el ajuste temporal).
// Mask: MatrixB N x D con las máscaras de la población actual (true = variable activa).
// FrontNo: vector de N doubles con el número de frente de cada solución.
// Fitness: vector de D doubles con la importancia acumulada de cada variable (se actualiza y retorna).
// NSV_prev: clasificación NSV de la generación anterior (usada si k-means falla).
// SV_prev: clasificación SV de la generación anterior.
// Retorna: SparsityResult con Local, Global (conocimiento de esparcidad), Fitness actualizado,
//          NSV, SV (clasificación binaria de variables), y theta (fracción de elite).
SparsityResult SparsityAnalysis(
    const Problem& prob,
    const MatrixB& Mask,
    const std::vector<double>& FrontNo,
    std::vector<double> Fitness,
    const std::vector<bool>& NSV_prev,
    const std::vector<bool>& SV_prev)
{
    int N = Mask.rows(), D = Mask.cols();

    // Collect elite mask (front == 1)
    std::vector<int> eliteIdx;
    for (int i = 0; i < N; i++)
        if (!std::isinf(FrontNo[i]) && (int)std::round(FrontNo[i]) == 1) eliteIdx.push_back(i);

    int nElite = eliteIdx.size();

    // Local knowledge from elite
    SparsityKnowledge Local, Global;
    Local.allOne.resize(D, true);
    Local.allZero.resize(D, true);
    Global.allOne.resize(D, true);
    Global.allZero.resize(D, true);

    for (int j = 0; j < D; j++) {
        for (int i : eliteIdx) {
            if (!Mask(i,j)) Local.allOne[j] = false;
            if ( Mask(i,j)) Local.allZero[j] = false;
        }
        for (int i = 0; i < N; i++) {
            if (!Mask(i,j)) Global.allOne[j] = false;
            if ( Mask(i,j)) Global.allZero[j] = false;
        }
    }

    Local.other.resize(D);
    Global.other.resize(D);
    for (int j = 0; j < D; j++) {
        Local.other[j]  = !Local.allOne[j]  && !Local.allZero[j];
        Global.other[j] = !Global.allOne[j] && !Global.allZero[j];
    }

    double theta = (double)nElite / N;

    // Normalize fitness
    double minF = *std::min_element(Fitness.begin(), Fitness.end());
    double maxF = *std::max_element(Fitness.begin(), Fitness.end());
    for (int j = 0; j < D; j++)
        Fitness[j] = (Fitness[j] - minF + 1e-6) / (maxF - minF + 1e-6) + 1e-6;

    std::vector<bool> NSV(D), SV(D);

    // Need at least 2 distinct fitness values for k-means
    bool allSame = true;
    for (int j = 1; j < D; j++)
        if (std::abs(Fitness[j] - Fitness[0]) > 1e-10) { allSame = false; break; }

    if (allSame || (!NSV_prev.empty() && (int)NSV_prev.size() == D)) {
        // Use previous grouping
        if (!NSV_prev.empty()) {
            NSV = NSV_prev;
            SV  = SV_prev;
        } else {
            // Default: split evenly
            std::fill(NSV.begin(), NSV.end(), false);
            std::fill(SV.begin(), SV.end(), false);
            for (int j = 0; j < D; j++) {
                if (j < D/2) NSV[j] = true;
                else         SV[j]  = true;
            }
        }
    } else {
        // Apply frequency adjustment
        std::vector<double> adjFitness = Fitness;
        for (int j = 0; j < D; j++) {
            double eliteMean = 0;
            for (int i : eliteIdx) eliteMean += (Mask(i,j) ? 1.0 : 0.0);
            eliteMean /= nElite;
            adjFitness[j] += (1.0 - eliteMean) *
                              (0.5*theta + 0.5*(prob.FE / (double)prob.maxFE)) * Fitness[j];
        }

        // K-means with k=2 on adjFitness
        std::vector<int> cluster = kmeans2(adjFitness);

        double mean0 = 0, mean1 = 0;
        int cnt0 = 0, cnt1 = 0;
        for (int j = 0; j < D; j++) {
            if (cluster[j] == 0) { mean0 += adjFitness[j]; cnt0++; }
            else                 { mean1 += adjFitness[j]; cnt1++; }
        }
        if (cnt0 > 0) mean0 /= cnt0;
        if (cnt1 > 0) mean1 /= cnt1;

        // Higher mean fitness = non-significant variables (NSV)
        for (int j = 0; j < D; j++) {
            if (mean0 > mean1) {
                NSV[j] = (cluster[j] == 1);
                SV[j]  = (cluster[j] == 0);
            } else {
                NSV[j] = (cluster[j] == 0);
                SV[j]  = (cluster[j] == 1);
            }
        }
    }

    return {Local, Global, Fitness, NSV, SV, theta};
}

#pragma once
#include "../../core/Solution.hpp"
#include "../../utils/NDSort.hpp"
#include "../../utils/Random.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

struct EnvSelResult {
    Population pop;
    Matrix dec;
    MatrixB mask;
    double sRatio;
};

// Selección ambiental de MOEA-CKF (truncación tipo SPEA2).
// Reduce la población combinada a N individuos manteniendo diversidad en el frente de Pareto.
// Pasos:
//   1. Elimina duplicados por objetivo (o por decisión si todos los objetivos son iguales).
//   2. Ordena por frente no-dominado.
//   3. Del último frente, elimina iterativamente el individuo con menor distancia al vecino más
//      cercano (el más "hacinado"), hasta ajustar al tamaño N.
//   4. Calcula sRatio: proporción de offspring1 vs offspring2 que sobrevivieron,
//      para retroalimentar la proporción rho del algoritmo principal.
// pop: población combinada (padres + offspring1 + offspring2).
// Dec: Matrix con las decisiones brutas (sin enmascarar), mismo orden que pop.
// Mask: MatrixB con las máscaras de esparcidad, mismo orden que pop.
// N: tamaño objetivo de la siguiente generación.
// len: índice donde comienza offspring1 dentro de pop (= tamaño de la población padre).
// num: número de individuos de offspring1.
// Retorna: EnvSelResult con la población seleccionada, sus Dec, Mask y el sRatio calculado.
EnvSelResult EnvironmentalSelection_CKF(
    const Population& pop, const Matrix& Dec, const MatrixB& Mask,
    int N, int len, int num)
{
    int totalN = pop.size();

    // Remove duplicated solutions (by objectives)
    Matrix allObjs = getObjs(pop);
    std::vector<bool> keep(totalN, true);
    std::vector<int> uni;

    // Find unique rows in objs
    {
        std::map<std::vector<double>, int> seen;
        for (int i = 0; i < totalN; i++) {
            std::vector<double> row(allObjs.cols());
            for (int j = 0; j < (int)allObjs.cols(); j++) row[j] = allObjs(i,j);
            if (seen.find(row) == seen.end()) {
                seen[row] = i;
                uni.push_back(i);
            }
        }
    }

    // If all objectives are identical, use unique decs
    if (uni.size() == 1) {
        Matrix allDecs = getDecs(pop);
        uni.clear();
        std::map<std::vector<double>, int> seen;
        for (int i = 0; i < totalN; i++) {
            std::vector<double> row(allDecs.cols());
            for (int j = 0; j < (int)allDecs.cols(); j++) row[j] = allDecs(i,j);
            if (seen.find(row) == seen.end()) {
                seen[row] = i;
                uni.push_back(i);
            }
        }
    }

    Population pop2    = subset(pop, uni);
    int        N2      = std::min(N, (int)pop2.size());
    Matrix     Dec2(uni.size(), Dec.cols());
    MatrixB    Mask2(uni.size(), Mask.cols());
    for (int i = 0; i < (int)uni.size(); i++) {
        Dec2.row(i)  = Dec.row(uni[i]);
        Mask2.row(i) = Mask.row(uni[i]);
    }

    // Non-dominated sorting
    Matrix objs2 = getObjs(pop2);
    auto [FrontNo, MaxFNo] = NDSort(objs2, N2);

    std::vector<bool> Next(pop2.size(), false);
    for (int i = 0; i < (int)FrontNo.size(); i++)
        if (!std::isinf(FrontNo[i]) && (int)std::round(FrontNo[i]) < MaxFNo) Next[i] = true;

    // Normalize objectives using first front range
    Eigen::RowVectorXd fmax(objs2.cols()), fmin(objs2.cols());
    fmax.setConstant(-1e18); fmin.setConstant(1e18);
    for (int i = 0; i < (int)objs2.rows(); i++) {
        if (!std::isinf(FrontNo[i]) && (int)std::round(FrontNo[i]) == 1) {
            fmax = fmax.cwiseMax(objs2.row(i));
            fmin = fmin.cwiseMin(objs2.row(i));
        }
    }
    Matrix normObjs = objs2;
    for (int i = 0; i < (int)objs2.rows(); i++) {
        for (int j = 0; j < objs2.cols(); j++) {
            double range = fmax(j) - fmin(j);
            normObjs(i,j) = (range > 1e-10) ? (objs2(i,j) - fmin(j)) / range : 0.0;
        }
    }

    // Truncate last front
    std::vector<int> Last;
    for (int i = 0; i < (int)FrontNo.size(); i++)
        if (!std::isinf(FrontNo[i]) && (int)std::round(FrontNo[i]) == MaxFNo) Last.push_back(i);

    int selected = (int)std::count(Next.begin(), Next.end(), true);
    int needed   = N2 - selected;
    int toDelete = (int)Last.size() - needed;
    // Clamp: can't delete more than available
    toDelete = std::max(0, std::min(toDelete, (int)Last.size()));

    if (toDelete > 0) {
        // Truncation: iteratively remove solution with smallest nearest neighbor distance
        Matrix lastObjs(Last.size(), objs2.cols());
        for (int i = 0; i < (int)Last.size(); i++)
            lastObjs.row(i) = normObjs.row(Last[i]);

        Matrix dist = pdist2(lastObjs, lastObjs);
        for (int r = 0; r < (int)Last.size(); r++) dist(r,r) = std::numeric_limits<double>::infinity();

        std::vector<bool> deleted(Last.size(), false);
        for (int k = 0; k < toDelete; k++) {
            // Find remaining solution with smallest nearest-neighbor distance
            std::vector<int> remain;
            for (int i = 0; i < (int)Last.size(); i++)
                if (!deleted[i]) remain.push_back(i);

            // Sort each remaining solution by distances to others (remain only)
            int bestIdx = -1;
            std::vector<double> bestRow;
            for (int ri : remain) {
                std::vector<double> dists;
                for (int rj : remain)
                    if (rj != ri) dists.push_back(dist(ri, rj));
                std::sort(dists.begin(), dists.end());
                if (bestIdx < 0 || dists < bestRow) {
                    bestIdx = ri;
                    bestRow = dists;
                }
            }
            if (bestIdx >= 0) deleted[bestIdx] = true;
            else break;
        }

        // Add non-deleted Last elements to Next
        for (int i = 0; i < (int)Last.size(); i++)
            if (!deleted[i]) Next[Last[i]] = true;
    } else {
        for (int i : Last) Next[i] = true;
    }

    // Compute sRatio based on which original members survived
    // Map uni indices back to original population indices
    std::vector<bool> successOrig(totalN, false);
    for (int i = 0; i < (int)uni.size(); i++)
        if (Next[i]) successOrig[uni[i]] = true;

    int s1 = 0, s2 = 0;
    for (int i = len; i < len + num && i < totalN; i++)
        if (successOrig[i]) s1++;
    for (int i = len + num; i < totalN; i++)
        if (successOrig[i]) s2++;

    double sRatio = (s1 + 1e-6) / (s1 + s2 + 1e-6);
    sRatio = std::max(0.1, std::min(0.9, sRatio));

    // Build output
    Population outPop;
    Matrix outDec(0, Dec.cols());
    MatrixB outMask(0, Mask.cols());
    std::vector<int> rows;
    for (int i = 0; i < (int)Next.size(); i++)
        if (Next[i]) rows.push_back(i);

    outPop = subset(pop2, rows);
    outDec.resize(rows.size(), Dec.cols());
    outMask.resize(rows.size(), Mask.cols());
    for (int i = 0; i < (int)rows.size(); i++) {
        outDec.row(i)  = Dec2.row(rows[i]);
        outMask.row(i) = Mask2.row(rows[i]);
    }

    return {outPop, outDec, outMask, sRatio};
}

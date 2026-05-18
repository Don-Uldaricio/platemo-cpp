#pragma once
#include "../core/Solution.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <limits>

// Encuentra las filas únicas de una matriz (equivalente a unique(...,'rows') de MATLAB).
// M: Matrix N x D de entrada.
// Retorna: par (matriz de filas únicas ordenadas, vector loc) donde loc[i] es el índice
//          de la fila única a la que corresponde la fila i original.
inline std::pair<Matrix, std::vector<int>> uniqueRows(const Matrix& M) {
    int N = M.rows(), D = M.cols();
    std::vector<std::vector<double>> rows(N, std::vector<double>(D));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            rows[i][j] = M(i,j);

    std::map<std::vector<double>, int> seen;
    std::vector<int> uniqueIdx;
    std::vector<int> loc(N);

    for (int i = 0; i < N; i++) {
        auto it = seen.find(rows[i]);
        if (it == seen.end()) {
            seen[rows[i]] = uniqueIdx.size();
            uniqueIdx.push_back(i);
        }
        loc[i] = seen[rows[i]];
    }

    int Nuniq = uniqueIdx.size();
    // Sort unique rows lexicographically (MATLAB unique returns sorted)
    std::vector<int> sortIdx(Nuniq);
    std::iota(sortIdx.begin(), sortIdx.end(), 0);
    // Build sorted order of unique rows
    std::vector<std::vector<double>> urows(Nuniq);
    for (int i = 0; i < Nuniq; i++) urows[i] = rows[uniqueIdx[i]];
    std::sort(sortIdx.begin(), sortIdx.end(), [&](int a, int b){ return urows[a] < urows[b]; });

    // Build reverse map: old unique idx -> new sorted position
    std::vector<int> rank(Nuniq);
    for (int i = 0; i < Nuniq; i++) rank[sortIdx[i]] = i;

    // Remap loc
    for (int i = 0; i < N; i++) loc[i] = rank[loc[i]];

    Matrix uniqM(Nuniq, D);
    for (int i = 0; i < Nuniq; i++)
        for (int j = 0; j < D; j++)
            uniqM(i,j) = urows[sortIdx[i]][j];

    return {uniqM, loc};
}

// Ordenamiento no-dominado ENS-SS (Efficient Non-dominated Sorting with Sequential Search).
// Eficiente para M < 3 o N < 500. Asigna número de frente a cada solución.
// PopObj: Matrix N x M con los objetivos de cada solución.
// nSort: mínimo de soluciones a ordenar (parar al llegar a este número).
// Retorna: par (frontNos, maxFront) donde frontNos[i] es el número de frente 1-indexado
//          de la solución i (inf si no fue procesada), y maxFront es el mayor frente asignado.
inline std::pair<std::vector<double>, int> ENS_SS(const Matrix& PopObj, int nSort) {
    auto [uObj, Loc] = uniqueRows(PopObj);
    int N = uObj.rows(), M = uObj.cols();

    // Count occurrences of each unique row in original
    std::vector<int> Table(N, 0);
    for (int l : Loc) Table[l]++;

    std::vector<double> FrontNo(N, std::numeric_limits<double>::infinity());
    int MaxFNo = 0;

    // Count sorted so far
    auto countSorted = [&]() {
        int cnt = 0;
        for (int i = 0; i < N; i++)
            if (!std::isinf(FrontNo[i])) cnt += Table[i];
        return cnt;
    };

    int target = std::min(nSort, (int)Loc.size());

    while (countSorted() < target) {
        MaxFNo++;
        for (int i = 0; i < N; i++) {
            if (!std::isinf(FrontNo[i])) continue;
            bool dominated = false;
            for (int j = i-1; j >= 0; j--) {
                if (std::isinf(FrontNo[j]) || (int)std::round(FrontNo[j]) != MaxFNo) continue;
                // Check if j dominates i: uObj(j,:) <= uObj(i,:) with at least one strict
                // Since rows are sorted by first obj, uObj(j,0) <= uObj(i,0)
                // Check remaining objectives
                int m = 1;
                while (m < M && uObj(i,m) >= uObj(j,m)) m++;
                dominated = (m >= M);
                if (dominated || M == 2) break;
            }
            if (!dominated) FrontNo[i] = MaxFNo;
        }
    }

    // Map back to original
    std::vector<double> result(Loc.size());
    for (int i = 0; i < (int)Loc.size(); i++) result[i] = FrontNo[Loc[i]];
    return {result, MaxFNo};
}

// Ordenamiento no-dominado sin restricciones. Wrapper sobre ENS_SS.
// PopObj: Matrix N x M de objetivos.
// nSort: mínimo de soluciones a ordenar (INT_MAX para ordenar todas).
// Retorna: par (frontNos, maxFront).
inline std::pair<std::vector<double>, int> NDSort(const Matrix& PopObj, int nSort = INT_MAX) {
    return ENS_SS(PopObj, nSort);
}

// Ordenamiento no-dominado con dominación restringida (constraint domination).
// Las soluciones infactibles (cualquier restricción > 0) reciben penalización en objetivos.
// PopObj: Matrix N x M de objetivos.
// PopCon: Matrix N x C de restricciones (valores > 0 indican violación).
// nSort: mínimo de soluciones a ordenar.
// Retorna: par (frontNos, maxFront).
inline std::pair<std::vector<double>, int> NDSort(const Matrix& PopObj, const Matrix& PopCon, int nSort = INT_MAX) {
    int N = PopObj.rows(), M = PopObj.cols();
    Matrix AugObj = PopObj;

    for (int i = 0; i < N; i++) {
        // Check if infeasible
        bool infeasible = false;
        double totalViol = 0.0;
        for (int c = 0; c < PopCon.cols(); c++) {
            if (PopCon(i,c) > 0) {
                infeasible = true;
                totalViol += PopCon(i,c);
            }
        }
        if (infeasible) {
            // Penalize: set objectives to max + sum of violations
            double maxV = PopObj.colwise().maxCoeff().maxCoeff();
            for (int m = 0; m < M; m++)
                AugObj(i,m) = maxV + totalViol;
        }
    }

    return ENS_SS(AugObj, nSort);
}

// Retorna las soluciones factibles del frente 1 (frente de Pareto) de la población.
// Si no hay soluciones factibles, retorna un vector vacío.
// Para problemas de un solo objetivo, retorna la solución con menor objetivo.
// pop: población completa.
// Retorna: Population con las soluciones no-dominadas factibles.
inline Population getBest(const Population& pop) {
    if (pop.empty()) return {};

    std::vector<int> feasible;
    for (int i = 0; i < (int)pop.size(); i++)
        if (pop[i].isFeasible()) feasible.push_back(i);

    if (feasible.empty()) return {};

    Population feasPop = subset(pop, feasible);
    int M = feasPop[0].obj.size();

    if (M > 1) {
        Matrix objs = getObjs(feasPop);
        auto [frontNo, _] = NDSort(objs, 1);
        std::vector<bool> mask(feasPop.size(), false);
        for (int i = 0; i < (int)frontNo.size(); i++)
            if ((int)frontNo[i] == 1) mask[i] = true;
        return subsetMask(feasPop, mask);
    } else {
        // Single objective: return solution with minimum objective
        int best = 0;
        for (int i = 1; i < (int)feasPop.size(); i++)
            if (feasPop[i].obj(0) < feasPop[best].obj(0)) best = i;
        return {feasPop[best]};
    }
}

#pragma once
#include "../../core/Problem.hpp"
#include "../../core/AlgoConfig.hpp"
#include "../../utils/NDSort.hpp"
#include "../../utils/CrowdingDistance.hpp"
#include "../../utils/TournamentSelection.hpp"
#include "PriorAnalysis.hpp"
#include "EnvironmentalSelection.hpp"
#include "SparsityAnalysis.hpp"
#include "Reproduction1.hpp"
#include "Reproduction2.hpp"
#include <iostream>
#include <algorithm>

// MOEA-CKF: Multi-objective Evolutionary Algorithm based on Cross-scale Knowledge Fusion.
// Algoritmo evolutivo multi-objetivo con fusión de conocimiento entre escalas para problemas esparcidos.
// Referencia: Z. Ding et al., IEEE TSMC 2024.
// Flujo por generación:
//   1. SparsityAnalysis: clasifica variables en SV (significativas) y NSV (no-significativas).
//   2. Divide la población en Grupo1 (fracción rho) y Grupo2 (fracción 1-rho).
//   3. Grupo1 → Reproduction1 (reducción de dimensionalidad con SVD, crossover en espacio reducido).
//   4. Grupo2 → Reproduction2 (crossover por grupos SV/NSV separados).
//   5. Selección CKF que actualiza rho según qué estrategia generó más supervivientes.
// prob: problema a optimizar.
// verbose: si true, imprime FE y tamaño de población en cada generación.
// Retorna: Population final de N soluciones al agotar el presupuesto de evaluaciones.
Population MOEACKF(Problem& prob, bool verbose = false,
                   const AlgoConfig& acfg = AlgoConfig{}) {
    int D = prob.D;

    bool isReal = true;
    for (int enc : prob.encoding)
        if (enc != 1) { isReal = false; break; }

    // Prior analysis initialization
    auto initRes = PriorAnalysis_initialization(prob, isReal);
    Matrix     TDec    = initRes.TDec;
    MatrixB    TMask   = initRes.TMask;
    Population TempPop = initRes.TempPop;
    std::vector<double> Fitness = initRes.Fitness;

    // Initial environmental selection
    auto selInit = EnvironmentalSelection_CKF(TempPop, TDec, TMask, prob.N, 0, 0);
    Population pop   = selInit.pop;
    Matrix     Dec   = selInit.dec;
    MatrixB    Mask  = selInit.mask;

    prob.LogInitial(pop);

    double rho = 0.5;
    std::vector<bool> NSV_prev, SV_prev;

    while (prob.NotTerminated(pop)) {
        Matrix objs = getObjs(pop);
        Matrix cons = getCons(pop);
        auto [FrontNo, maxFNo] = NDSort(objs, cons, INT_MAX);
        auto CrowdDis = CrowdingDistance(objs, FrontNo);

        int popN = (int)pop.size();

        // Divide population
        std::vector<bool> Pop1Site(popN);
        for (int i = 0; i < popN; i++)
            Pop1Site[i] = (rng::uniform() < rho);

        int n1 = (int)std::count(Pop1Site.begin(), Pop1Site.end(), true);

        std::vector<Population> pops(2);
        Matrix Dec1, Dec2;
        MatrixB Mask1, Mask2;
        {
            std::vector<int> idx1, idx2;
            for (int i = 0; i < popN; i++) {
                if (Pop1Site[i]) idx1.push_back(i);
                else             idx2.push_back(i);
            }
            pops[0] = subset(pop, idx1);
            pops[1] = subset(pop, idx2);
            Dec1.resize(idx1.size(), D); Mask1.resize(idx1.size(), D);
            Dec2.resize(idx2.size(), D); Mask2.resize(idx2.size(), D);
            for (int k = 0; k < (int)idx1.size(); k++) {
                Dec1.row(k)  = Dec.row(idx1[k]);
                Mask1.row(k) = Mask.row(idx1[k]);
            }
            for (int k = 0; k < (int)idx2.size(); k++) {
                Dec2.row(k)  = Dec.row(idx2[k]);
                Mask2.row(k) = Mask.row(idx2[k]);
            }
        }

        // Sparsity analysis
        auto spRes = SparsityAnalysis(prob, Mask, FrontNo, Fitness, NSV_prev, SV_prev);
        NSV_prev = spRes.NSV;
        SV_prev  = spRes.SV;
        Fitness  = spRes.Fitness;

        // Offspring1: dual dimension reduction
        Population off1;
        Matrix     OffDec1;
        MatrixB    OffMask1;
        int        len1 = 0;

        if (n1 >= 2) {
            auto r1 = Reproduction1(prob, pops[0], Dec1, Mask1,
                                    FrontNo, CrowdDis, Pop1Site,
                                    spRes.Local, spRes.Global,
                                    spRes.NSV, spRes.SV,
                                    spRes.theta, isReal, acfg);
            len1    = r1.len1;
            OffDec1 = r1.OffDec;
            OffMask1= r1.OffMask;

            Matrix evalDec1(len1, D);
            for (int i = 0; i < len1; i++)
                for (int j = 0; j < D; j++)
                    evalDec1(i,j) = OffMask1(i,j) ? OffDec1(i,j) : 0.0;
            off1 = prob.Evaluation(evalDec1);
        }

        // Offspring2: dual grouping
        Population off2;
        Matrix     OffDec2;
        MatrixB    OffMask2;

        if (!pops[1].empty()) {
            int n2pool = (prob.N - len1) * 2;
            if (n2pool < 2) n2pool = 2;
            std::vector<double> subFront2, subCrowd2;
            for (int i = 0; i < popN; i++) {
                if (!Pop1Site[i]) {
                    subFront2.push_back(FrontNo[i]);
                    subCrowd2.push_back(-CrowdDis[i]);
                }
            }
            if ((int)subFront2.size() >= 2) {
                auto mpool2 = TournamentSelection(2, n2pool, subFront2, subCrowd2);

                Matrix  mDec(n2pool, D);
                MatrixB mMask(n2pool, D);
                for (int k = 0; k < n2pool; k++) {
                    int mk = mpool2[k] % (int)pops[1].size();
                    mDec.row(k)  = Dec2.row(mk);
                    mMask.row(k) = Mask2.row(mk);
                }

                auto r2  = Reproduction2(prob, mDec, mMask, spRes.NSV, spRes.SV, isReal, acfg);
                OffDec2  = r2.OffDec;
                OffMask2 = r2.OffMask;

                int n2off = (int)OffDec2.rows();
                Matrix evalDec2(n2off, D);
                for (int i = 0; i < n2off; i++)
                    for (int j = 0; j < D; j++)
                        evalDec2(i,j) = OffMask2(i,j) ? OffDec2(i,j) : 0.0;
                off2 = prob.Evaluation(evalDec2);
            }
        }

        // Merge
        int totMerge = (int)pop.size() + (int)off1.size() + (int)off2.size();
        Population mergedPop = concat(concat(pop, off1), off2);
        Matrix     mergedDec(totMerge, D);
        MatrixB    mergedMask(totMerge, D);

        int base = 0;
        for (int i = 0; i < (int)pop.size(); i++) {
            mergedDec.row(base)  = Dec.row(i);
            mergedMask.row(base) = Mask.row(i);
            base++;
        }
        for (int i = 0; i < (int)off1.size(); i++) {
            mergedDec.row(base)  = OffDec1.row(i);
            mergedMask.row(base) = OffMask1.row(i);
            base++;
        }
        for (int i = 0; i < (int)off2.size(); i++) {
            mergedDec.row(base)  = OffDec2.row(i);
            mergedMask.row(base) = OffMask2.row(i);
            base++;
        }

        auto selRes = EnvironmentalSelection_CKF(
            mergedPop, mergedDec, mergedMask,
            prob.N, (int)pop.size(), len1);

        pop  = selRes.pop;
        Dec  = selRes.dec;
        Mask = selRes.mask;
        rho  = (rho + selRes.sRatio) / 2.0;

        if (verbose)
            std::cout << "FE=" << prob.FE << " popSize=" << pop.size() << "\n";
        prob.MaybeLog(pop);
    }
    return pop;
}

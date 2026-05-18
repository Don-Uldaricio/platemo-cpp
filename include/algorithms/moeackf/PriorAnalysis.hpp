#pragma once
#include "../../core/Problem.hpp"
#include "../../utils/NDSort.hpp"
#include "../../utils/TournamentSelection.hpp"
#include "../../utils/Random.hpp"
#include <vector>
#include <Eigen/Dense>

struct PriorAnalysisResult {
    Matrix  TDec;
    MatrixB TMask;
    Population TempPop;
    std::vector<double> Fitness;
};

// Fase de inicialización del análisis previo de MOEA-CKF.
// Estima la importancia de cada variable evaluando soluciones donde solo una variable está activa
// (máscara identidad). Las variables que producen soluciones con mejor ranking reciben mayor peso.
// Para problemas reales repite 5 veces (nRep=5) para reducir el ruido estadístico.
// Luego genera la población inicial con sesgo hacia las variables más importantes usando torneo.
// prob: problema a optimizar (se usarán prob.N, prob.D, prob.Evaluation, etc.).
// isReal: true si todas las variables son reales (activa las 5 repeticiones y muestreo uniforme).
// Retorna: PriorAnalysisResult con TDec (decisiones), TMask (máscaras), TempPop (soluciones evaluadas)
//          y Fitness (importancia acumulada de cada variable, menor = más importante).
PriorAnalysisResult PriorAnalysis_initialization(Problem& prob, bool isReal) {
    int D = prob.D;

    Matrix     TDec;
    MatrixB    TMask;
    Population TempPop;
    std::vector<double> Fitness(D, 0.0);

    int nRep = isReal ? 5 : 1;  // 1 + 4*REAL

    for (int rep = 0; rep < nRep; rep++) {
        // Each "solution" tests one variable at a time (identity mask)
        Matrix Dec(D, D);
        if (isReal) {
            Dec = rng::unifrndBounded(D, prob.lower, prob.upper);
        } else {
            Dec = Matrix::Ones(D, D);
        }

        // Mask = identity
        MatrixB Mask = MatrixB::Zero(D, D);
        for (int i = 0; i < D; i++) Mask(i,i) = true;

        // Evaluate: actual decision = Dec .* Mask
        Matrix evalDec(D, D);
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                evalDec(i,j) = Mask(i,j) ? Dec(i,j) : 0.0;

        Population pop = prob.Evaluation(evalDec);

        // Accumulate to TDec, TMask, TempPop
        if (TDec.rows() == 0) {
            TDec  = Dec;
            TMask = Mask;
        } else {
            Matrix tmp(TDec.rows() + D, D);
            tmp.topRows(TDec.rows()) = TDec;
            tmp.bottomRows(D)        = Dec;
            TDec = tmp;

            MatrixB tmpM(TMask.rows() + D, D);
            tmpM.topRows(TMask.rows()) = TMask;
            tmpM.bottomRows(D)         = Mask;
            TMask = tmpM;
        }
        TempPop.insert(TempPop.end(), pop.begin(), pop.end());

        // Compute NDSort on [objs, cons] combined
        Matrix objsCons(D, prob.M + (int)pop[0].con.size());
        for (int i = 0; i < D; i++) {
            objsCons.row(i).head(prob.M) = pop[i].obj.transpose();
            objsCons.row(i).tail(pop[i].con.size()) = pop[i].con.transpose();
        }
        auto [frontNos, _] = NDSort(objsCons, INT_MAX);

        for (int i = 0; i < D; i++) Fitness[i] += frontNos[i];
    }

    // Generate initial population with bias toward important variables
    Matrix Dec2(prob.N, D);
    if (isReal) {
        Dec2 = rng::unifrndBounded(prob.N, prob.lower, prob.upper);
    } else {
        Dec2 = Matrix::Ones(prob.N, D);
    }

    MatrixB Mask2 = MatrixB::Zero(prob.N, D);
    for (int i = 0; i < prob.N; i++) {
        int k = rng::randint(1, D);
        // Tournament selection based on Fitness (lower front = better = more likely selected)
        auto idx = TournamentSelection(2, k, Fitness);
        for (int j : idx) Mask2(i,j) = true;
    }

    Matrix evalDec2(prob.N, D);
    for (int i = 0; i < prob.N; i++)
        for (int j = 0; j < D; j++)
            evalDec2(i,j) = Mask2(i,j) ? Dec2(i,j) : 0.0;

    Population pop2 = prob.Evaluation(evalDec2);

    // Append
    {
        Matrix tmp(TDec.rows() + prob.N, D);
        tmp.topRows(TDec.rows()) = TDec;
        tmp.bottomRows(prob.N)   = Dec2;
        TDec = tmp;

        MatrixB tmpM(TMask.rows() + prob.N, D);
        tmpM.topRows(TMask.rows()) = TMask;
        tmpM.bottomRows(prob.N)    = Mask2;
        TMask = tmpM;
    }
    TempPop.insert(TempPop.end(), pop2.begin(), pop2.end());

    return {TDec, TMask, TempPop, Fitness};
}

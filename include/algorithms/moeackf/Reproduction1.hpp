#pragma once
#include "../../core/Problem.hpp"
#include "../../core/AlgoConfig.hpp"
#include "BinaryCrossover.hpp"
#include "BinaryMutation.hpp"
#include "../../utils/TournamentSelection.hpp"
#include "../../utils/Random.hpp"
#include "SparsityAnalysis.hpp"
#include <Eigen/SVD>
#include <vector>
#include <cmath>

namespace r1detail {

// SBX "half": produce N hijos (solo Child1) a partir de 2N padres emparejados.
// Parent: Matrix 2N x D (primeras N filas = Parent1, siguientes N = Parent2).
// proC: probabilidad de crossover por individuo.
// disC: índice de distribución SBX.
// Retorna: Matrix N x D con solo los primeros hijos.
inline Matrix GAhalfCross(const Matrix& Parent, double proC = 1.0, double disC = 20.0) {
    int half = Parent.rows() / 2;
    Matrix P1 = Parent.topRows(half);
    Matrix P2 = Parent.bottomRows(half);
    int N = P1.rows(), D = P1.cols();

    Matrix beta(N, D), mu = rng::rand(N, D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            beta(i,j) = (mu(i,j) <= 0.5)
                ? std::pow(2.0*mu(i,j), 1.0/(disC+1.0))
                : std::pow(2.0 - 2.0*mu(i,j), -1.0/(disC+1.0));

    auto signs = rng::randi(0, 1, N, D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (signs(i,j) == 0) beta(i,j) = -beta(i,j);

    Matrix r1 = rng::rand(N, D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (r1(i,j) < 0.5) beta(i,j) = 1.0;

    Matrix r2 = rng::rand(N, 1);
    for (int i = 0; i < N; i++)
        if (r2(i,0) > proC)
            for (int j = 0; j < D; j++) beta(i,j) = 1.0;

    return (P1 + P2) / 2.0 + beta.cwiseProduct(P1 - P2) / 2.0;
}

// Aplica mutación polinomial a una sub-matriz que representa un subconjunto de columnas del espacio original.
// Off: Matrix N x K con los valores de las K columnas seleccionadas.
// colMask: vector bool de tamaño prob.D; true en las posiciones correspondientes a las K columnas de Off.
//          Debe tener exactamente K entradas true para extraer los bounds correctos.
// proM: probabilidad de mutación; tasa real por gen = proM/K.
// disM: índice de distribución de la mutación polinomial.
// Retorna: Matrix N x K con los valores mutados y clampeados a bounds.
inline Matrix PM_cols(Problem& prob, Matrix Off, const std::vector<bool>& colMask,
                      double proM = 1.0, double disM = 20.0) {
    int N = Off.rows(), K = Off.cols();
    if (K == 0) return Off;

    // Build lower/upper arrays for the K selected columns
    std::vector<double> lo(K), hi(K);
    int col = 0;
    for (int j = 0; j < prob.D && col < K; j++) {
        if (!colMask[j]) continue;
        lo[col] = prob.lower(j);
        hi[col] = prob.upper(j);
        col++;
    }

    // Clamp
    for (int i = 0; i < N; i++)
        for (int k = 0; k < K; k++)
            Off(i,k) = std::max(lo[k], std::min(hi[k], Off(i,k)));

    // Polynomial mutation
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            if (rng::uniform() >= proM / K) continue;
            double range = hi[k] - lo[k];
            if (range < 1e-10) continue;
            double mu_ = rng::uniform();
            double delta;
            if (mu_ <= 0.5) {
                double xy = 1.0 - (Off(i,k) - lo[k]) / range;
                delta = std::pow(2.0*mu_ + (1.0-2.0*mu_)*std::pow(xy, disM+1.0), 1.0/(disM+1.0)) - 1.0;
            } else {
                double xy = 1.0 - (hi[k] - Off(i,k)) / range;
                delta = 1.0 - std::pow(2.0*(1.0-mu_) + 2.0*(mu_-0.5)*std::pow(xy, disM+1.0), 1.0/(disM+1.0));
            }
            Off(i,k) = std::max(lo[k], std::min(hi[k], Off(i,k) + range * delta));
        }
    }
    return Off;
}

// GAhalfCross seguido de mutación polinomial, para el subconjunto de columnas allZero.
// Combina GAhalfCross + PM_cols en un solo paso conveniente.
// Parent: Matrix 2N x K (K = número de columnas allZero).
// colMask: vector bool de tamaño prob.D que selecciona las K columnas.
// Retorna: Matrix N x K con los hijos cruzados y mutados.
inline Matrix GAhalf(Problem& prob, const Matrix& Parent, const std::vector<bool>& colMask,
                     double proC = 1.0, double disC = 20.0,
                     double proM = 1.0, double disM = 20.0) {
    Matrix Off = GAhalfCross(Parent, proC, disC);
    return PM_cols(prob, Off, colMask, proM, disM);
}

} // namespace r1detail

struct Reproduction1Result {
    Matrix  OffDec;
    MatrixB OffMask;
    int     len1;
};

Population MOEACKF(Problem&, bool);  // forward decl to allow Reproduction1 to be separate

// Reproducción con reducción de dimensionalidad (Dual Dimension Reduction) de MOEA-CKF.
// Genera offspring para el Grupo 1 (individuos seleccionados por Pop1Site).
// Para las máscaras: usa BinaryCrossover + BinaryMutation separadamente en grupos NSV y SV.
// Para los valores reales: aplica SVD sobre las variables no-siempre-cero (dimensión reducida),
//   cruza en el espacio reducido con GAhalfCross, y reprojecta. Las variables siempre-cero
//   reciben GAhalf directo.
// prob: problema actual.
// Pop1: subpoblación del Grupo 1 (individuos con Pop1Site[i] == true).
// Dec1: Matrix con las decisiones brutas del Grupo 1.
// Mask1: MatrixB con las máscaras del Grupo 1.
// FrontNo: números de frente de TODA la población (misma indexación que Pop1Site).
// CrowdDis: distancias de crowding de TODA la población.
// Pop1Site: vector bool de tamaño N que indica qué individuos pertenecen al Grupo 1.
// LocalK: conocimiento de esparcidad local (frente 1).
// GlobalK: conocimiento de esparcidad global (toda la población).
// NSV: vector bool D indicando variables no-significativas.
// SV: vector bool D indicando variables significativas.
// theta: fracción de elite (ratio de Grupo1 que está en el frente 1).
// isReal: true si el problema es de variables reales.
// Retorna: Reproduction1Result con OffDec (decisiones), OffMask (máscaras) y len1 (número de offspring).
Reproduction1Result Reproduction1(
    Problem& prob,
    const Population& Pop1, const Matrix& Dec1, const MatrixB& Mask1,
    const std::vector<double>& FrontNo, const std::vector<double>& CrowdDis,
    const std::vector<bool>& Pop1Site,
    const SparsityKnowledge& LocalK, const SparsityKnowledge& GlobalK,
    const std::vector<bool>& NSV, const std::vector<bool>& SV,
    double theta, bool isReal,
    const AlgoConfig& acfg = AlgoConfig{})
{
    int N  = (int)Pop1.size();
    int D  = prob.D;

    // Tournament selection from Pop1 subset
    std::vector<double> subFront, subCrowd;
    for (int i = 0; i < (int)FrontNo.size(); i++) {
        if (Pop1Site[i]) {
            subFront.push_back(FrontNo[i]);
            subCrowd.push_back(-CrowdDis[i]);
        }
    }
    if ((int)subFront.size() < 2) {
        // Not enough parents, return empty
        return {Matrix::Zero(0,D), MatrixB::Zero(0,D), 0};
    }

    auto matingIdx = TournamentSelection(2, N*2, subFront, subCrowd);

    // Build parent masks for N offspring (matingIdx has 2*N entries)
    MatrixB Parent1Mask(N, D), Parent2Mask(N, D);
    for (int i = 0; i < N; i++) {
        int i1 = std::min(matingIdx[i],   (int)Mask1.rows()-1);
        int i2 = std::min(matingIdx[N+i], (int)Mask1.rows()-1);
        Parent1Mask.row(i) = Mask1.row(i1);
        Parent2Mask.row(i) = Mask1.row(i2);
    }

    // Generate offspring masks
    MatrixB OffMask = MatrixB::Zero(N, D);
    for (int i = 0; i < N; i++) {
        const SparsityKnowledge& K = (rng::uniform() < theta) ? LocalK : GlobalK;

        std::vector<int> nsvOther, svOther;
        for (int j = 0; j < D; j++) {
            if (K.other[j] && NSV[j]) nsvOther.push_back(j);
            if (K.other[j] && SV[j])  svOther.push_back(j);
        }

        auto applyGroup = [&](const std::vector<int>& idx) {
            if (idx.empty()) return;
            MatrixB sub1(1, idx.size()), sub2(1, idx.size());
            for (int k = 0; k < (int)idx.size(); k++) {
                sub1(0,k) = Parent1Mask(i, idx[k]);
                sub2(0,k) = Parent2Mask(i, idx[k]);
            }
            MatrixB out = BinaryCrossover(sub1, sub2);
            out = BinaryMutation(out);
            for (int k = 0; k < (int)idx.size(); k++)
                OffMask(i, idx[k]) = out(0,k);
        };

        applyGroup(nsvOther);
        applyGroup(svOther);
        for (int j = 0; j < D; j++)
            if (K.allOne[j]) OffMask(i,j) = true;
    }

    Matrix OffDec;

    if (!isReal) {
        OffDec = Matrix::Ones(N, D);
        return {OffDec, OffMask, N};
    }

    // Global knowledge columns
    std::vector<int> notZeroCols, zeroCols;
    for (int j = 0; j < D; j++) {
        if (!GlobalK.allZero[j]) notZeroCols.push_back(j);
        else                     zeroCols.push_back(j);
    }
    int Dred = (int)notZeroCols.size();

    OffDec = Matrix::Zero(N, D);

    if (Dred > 0) {
        // Build notZero sub-matrix of Dec1
        Matrix T_Best(N, Dred);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < Dred; k++)
                T_Best(i,k) = Dec1(i, notZeroCols[k]);

        // Normalize
        Eigen::RowVectorXd varMean = T_Best.colwise().mean();
        Eigen::RowVectorXd varMax  = T_Best.colwise().maxCoeff();
        Eigen::RowVectorXd varMin  = T_Best.colwise().minCoeff();
        Eigen::RowVectorXd varNorm = (varMax - varMin).array() + 1e-6;

        Matrix NormDec(N, Dred);
        for (int i = 0; i < N; i++)
            NormDec.row(i) = (T_Best.row(i) - varMean).array() / varNorm.array();

        // SVD of covariance matrix
        Matrix cov = NormDec.transpose() * NormDec / N;
        Eigen::JacobiSVD<Matrix> svd(cov, Eigen::ComputeFullU);
        Matrix U = svd.matrixU();

        // Subspace size: sum of other[NSV] + sum of allOne
        int K_pca = 0;
        for (int j = 0; j < D; j++) {
            if (GlobalK.other[j] && NSV[j]) K_pca++;
            if (GlobalK.allOne[j]) K_pca++;
        }
        K_pca = std::max(1, std::min(K_pca, Dred));

        Matrix Ureduce = U.leftCols(K_pca);
        Matrix Pop_reduce = NormDec * Ureduce;  // N x K_pca

        // Build mating pool in reduced space
        Matrix mating(2*N, K_pca);
        for (int i = 0; i < N; i++) {
            int i1 = std::min(matingIdx[i],   N-1);
            int i2 = std::min(matingIdx[N+i], N-1);
            mating.row(i)   = Pop_reduce.row(i1);
            mating.row(N+i) = Pop_reduce.row(i2);
        }
        Matrix Off_reduce = r1detail::GAhalfCross(mating, 1.0, acfg.disC);  // N x K_pca

        // Restore to original space
        Matrix T_OffDec = Off_reduce * Ureduce.transpose();  // N x Dred
        for (int i = 0; i < N; i++)
            T_OffDec.row(i) = T_OffDec.row(i).array() * varNorm.array() + varMean.array();

        // Apply PM
        std::vector<bool> notZeroMask(D, false);
        for (int j : notZeroCols) notZeroMask[j] = true;
        T_OffDec = r1detail::PM_cols(prob, T_OffDec, notZeroMask, acfg.proM, acfg.disM);

        for (int i = 0; i < N; i++)
            for (int k = 0; k < Dred; k++)
                OffDec(i, notZeroCols[k]) = T_OffDec(i,k);
    }

    // allZero columns: GAhalf crossover + PM
    if (!zeroCols.empty()) {
        int K_zero = (int)zeroCols.size();
        Matrix zeroParents(2*N, K_zero);
        for (int i = 0; i < N; i++) {
            int i1 = std::min(matingIdx[i],   N-1);
            int i2 = std::min(matingIdx[N+i], N-1);
            for (int k = 0; k < K_zero; k++) {
                zeroParents(i,   k) = Dec1(i1, zeroCols[k]);
                zeroParents(N+i, k) = Dec1(i2, zeroCols[k]);
            }
        }
        std::vector<bool> zeroMask(D, false);
        for (int j : zeroCols) zeroMask[j] = true;
        Matrix zeroOff = r1detail::GAhalf(prob, zeroParents, zeroMask, 1.0, acfg.disC, acfg.proM, acfg.disM);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K_zero; k++)
                OffDec(i, zeroCols[k]) = zeroOff(i,k);
    }

    return {OffDec, OffMask, N};
}

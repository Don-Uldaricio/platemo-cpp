/*
 * Mide el costo puro del algoritmo evolutivo (MOEACKF / SNSGAII), separado del
 * costo de evaluar la SNN: usa un Problem con el mismo D que un caso real de
 * SparseSNN pero con CalObj trivial (sin simulación), corriendo una corrida
 * completa (N, maxFE) igual a la real.
 *
 * Uso: ./moea_overhead_bench <D> <N> <maxFE>
 * Stdout: "MOEA <algoritmo> <D> <N> <maxFE> <total_s> <FE_final>"
 */

#include "../include/core/Problem.hpp"
#include "../include/algorithms/moeackf/MOEACKF.hpp"
#include "../include/algorithms/snsgaii/SNSGAII.hpp"

#include <chrono>
#include <iostream>
#include <string>

// Problem de costo trivial: mismo D/M que un SparseSNN real, pero CalObj es
// una función esfera (sin simulación), para aislar el overhead del MOEA en sí
// (ordenamiento no dominado, cruce/mutación, y la SVD de Reproduction1 en MOEACKF).
class DummyFastProblem : public Problem {
public:
    explicit DummyFastProblem(int D_) { D = D_; }

    void Setting() override {
        M = 2;
        lower = Vector::Constant(D, 0.0);
        upper = Vector::Constant(D, 1.0);
        encoding.assign(D, 1);
    }

    Matrix CalObj(const Matrix& dec) override {
        int n = dec.rows();
        Matrix obj(n, 2);
        // f1: fracción de "pesos" activos (> 0), igual forma que SparseSNN.
        // f2: distancia cuadrática media a 0.5 (función esfera desplazada).
        for (int i = 0; i < n; i++) {
            int nz = 0;
            double s = 0.0;
            for (int j = 0; j < D; j++) {
                if (dec(i, j) > 0.0) nz++;
                double d = dec(i, j) - 0.5;
                s += d * d;
            }
            obj(i, 0) = static_cast<double>(nz) / D;
            obj(i, 1) = s / D;
        }
        return obj;
    }

    Matrix GetOptimum(int /*n*/) override { return Matrix::Ones(1, 2); }
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Uso: " << argv[0] << " <D> <N> <maxFE>\n";
        return 1;
    }
    int D     = std::stoi(argv[1]);
    int N     = std::stoi(argv[2]);
    int maxFE = std::stoi(argv[3]);

    {
        DummyFastProblem prob(D);
        prob.N = N;
        prob.maxFE = maxFE;
        prob.init();

        auto t0 = std::chrono::high_resolution_clock::now();
        Population finalPop = MOEACKF(prob, false, AlgoConfig{});
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "MOEA MOEACKF " << D << " " << N << " " << maxFE << " "
                  << total_s << " " << prob.FE << "\n";
    }
    {
        DummyFastProblem prob(D);
        prob.N = N;
        prob.maxFE = maxFE;
        prob.init();

        auto t0 = std::chrono::high_resolution_clock::now();
        Population finalPop = SNSGAII(prob, false, AlgoConfig{});
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "MOEA SNSGAII " << D << " " << N << " " << maxFE << " "
                  << total_s << " " << prob.FE << "\n";
    }

    return 0;
}

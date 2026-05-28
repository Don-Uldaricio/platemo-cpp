#pragma once
#include "../core/Problem.hpp"
#include "../utils/Random.hpp"

#include "core/network.hpp"
#include "core/simulator.hpp"
#include "encoding/poissonEncoder.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cmath>
#include <set>
#include <algorithm>
#include <memory>
#include <omp.h>

// Multi-objective optimization of a spiking neural network on classification tasks.
// Objectives (both minimized):
//   f1 = fraction of active (non-zero) synaptic weights  [sparsity]
//   f2 = training classification error                   [1 - accuracy]
//
// Decision variables: one weight per synapse in [0, 1].
// Zero weight = inactive synapse (contributes to sparsity).
//
// Topology: fully-connected, no recurrence.
//   (nFeatures+1) input+bias -> nHidden hidden neurons -> nClasses output neurons
//   D = (nFeatures+1)*nHidden + (nHidden+1)*nOutputs synapses (all excitatory).
//
// Input normalization: min-max to [0,1] so RateEncoder receives valid inputs.
class SparseSNN : public Problem {
public:
    int dataNo   = 1;
    int nHidden  = 20;
    std::string dataPath;

    int    nFeatures = 0;
    int    nClasses  = 0;
    int    nOutputs  = 0;
    int    nSamples  = 0;
    double wScale    = 1.0;   // multiplicador sobre pesos antes de setWeights(); sparsity se mide sin escala

    // SNN simulation timing — tuneable via CLI / Bayesian optimization
    double dt                  = 1.0;    // timestep (ms)
    double encoding_duration   = 50.0;  // Poisson encoding window (ms)
    double evaluation_duration = 100.0; // total simulation window (ms)
    double max_rate            = 100.0; // PoissonEncoder max firing rate (Hz)
    double refractory_period   = 5.0;   // PoissonEncoder refractory period (ms)

    // Training dataset for evaluateAccuracy(): (normalized_input, 0-indexed label)
    std::vector<std::pair<std::vector<double>, int>> trainDataset;

    SparseSNN(int dataNo_ = 1, int nHidden_ = 20,
              const std::string& dataPath_ = "data")
        : dataNo(dataNo_), nHidden(nHidden_), dataPath(dataPath_) {}

    // Carga los datos, define M=2, D=total pesos, bounds [0,1], y construye el pool de simuladores.
    // Los pesos en [0,1] representan la fuerza sináptica (0 = sinapsis inactiva).
    void Setting() override {
        loadData();

        M = 2;
        D = (nFeatures + 1) * nHidden + (nHidden + 1) * nOutputs;
        lower    = Vector::Constant(D, 0.0);
        upper    = Vector::Constant(D, 1.0);
        encoding.assign(D, 1);  // all real

        buildPool();
    }

    // Inicialización esparcida: cada peso es uniform(0,1) * Bernoulli(0.5) → ~50% ceros.
    // n: número de soluciones a generar (-1 usa prob.N).
    Population Initialization(int n = -1) override {
        if (n < 0) n = N;
        Matrix PopDec(n, D);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < D; j++)
                PopDec(i, j) = rng::uniform(0.0, 1.0) * rng::randint(0, 1);
        return Evaluation(PopDec);
    }

    // Sin fine-tuning (la SNN no tiene gradiente analítico): solo clampea a [0,1].
    // dec: Matrix n x D con las decisiones propuestas.
    // Retorna: Matrix n x D clampeada a [0,1].
    Matrix CalDec(const Matrix& dec) override {
        Matrix d = dec;
        for (int j = 0; j < D; j++)
            d.col(j) = d.col(j).cwiseMax(0.0).cwiseMin(1.0);
        return d;
    }

    // Calcula los dos objetivos para cada solución simulando la SNN en paralelo con OpenMP.
    //   f1 (obj 0) = fracción de pesos activos (> 0) [complejidad].
    //   f2 (obj 1) = 1 - accuracy sobre el training set [error de clasificación].
    // Cada thread usa su propio Simulator del pool para evitar condiciones de carrera.
    // dec: Matrix n x D con los pesos a evaluar.
    // Retorna: Matrix n x 2 con [complejidad, error] por fila.
    Matrix CalObj(const Matrix& dec) override {
        int n = dec.rows();
        Matrix obj(n, 2);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            int tid = omp_get_thread_num();

            std::vector<double> weights(D);
            int nzCount = 0;
            for (int j = 0; j < D; j++) {
                double raw = dec(i, j);
                if (raw > 0.0) nzCount++;
                weights[j] = raw * wScale;
            }
            obj(i, 0) = static_cast<double>(nzCount) / D;

            sim_pool[tid]->setWeights(weights);
            double acc = sim_pool[tid]->evaluateAccuracy(trainDataset);
            obj(i, 1) = 1.0 - acc;
        }
        return obj;
    }

    Matrix GetOptimum(int /*n*/) override {
        return Matrix::Ones(1, 2);  // ideal: (0 sparsity, 0 error) — unreachable in practice
    }

    // Evalúa configuraciones extremas de pesos y muestra accuracy + spikes por salida.
    // Útil para verificar que los pesos realmente afectan el comportamiento de la red.
    // Debe llamarse después de Setting() (init()).
    void runSanityCheck() {
        using namespace std;
        auto& sim = *sim_pool[0];

        // genera vector de pesos aleatorio denso
        auto makeRandom = [&](double density) {
            vector<double> w(D);
            for (int j = 0; j < D; j++)
                w[j] = (rng::uniform(0.0, 1.0) < density) ? rng::uniform(0.0, 1.0) : 0.0;
            return w;
        };

        // aplica escala a un vector de pesos
        auto scale = [&](vector<double> w, double s) {
            for (double& x : w) x *= s;
            return w;
        };

        struct Config { string name; vector<double> weights; };
        vector<Config> configs;
        configs.push_back({"all-zeros",              vector<double>(D, 0.0)});
        configs.push_back({"all-ones (×1)",          vector<double>(D, 1.0)});
        if (wScale != 1.0)
            configs.push_back({"all-ones (×wScale)", vector<double>(D, wScale)});
        configs.push_back({"random-dense (×1)",      makeRandom(1.0)});
        if (wScale != 1.0)
            configs.push_back({"random-dense (×ws)", scale(makeRandom(1.0), wScale)});
        configs.push_back({"random-sparse25 (×1)",   makeRandom(0.25)});
        if (wScale != 1.0)
            configs.push_back({"random-sp25 (×ws)",  scale(makeRandom(0.25), wScale)});

        cout << "\n=== Sanity Check (D=" << D << ", wScale=" << wScale
             << ", trainSamples=" << trainDataset.size() << ") ===\n";
        cout << left << setw(22) << "Config"
             << setw(10) << "Accuracy"
             << "  Output spikes (first sample) [per class]\n";
        cout << string(60, '-') << "\n";

        for (auto& cfg : configs) {
            sim.setWeights(cfg.weights);

            double acc = sim.evaluateAccuracy(trainDataset);

            // spike diagnostic on first sample
            auto result = sim.simulateEncoded(trainDataset[0].first);
            cout << left << setw(22) << cfg.name
                 << setw(10) << fixed << setprecision(4) << acc
                 << "  [";
            for (size_t k = 0; k < result.output_spike_times.size(); k++) {
                if (k) cout << ", ";
                cout << result.output_spike_times[k].size();
            }
            cout << "]  total=" << (int)result.total_spikes << "\n";
        }
        cout << "========================================\n\n";
    }

private:
    // One Network+Simulator per OpenMP thread — built once in buildPool().
    std::vector<std::unique_ptr<Network>>   net_pool;
    std::vector<std::unique_ptr<Simulator>> sim_pool;

    // Lee el CSV del dataset, normaliza con min-max a [0,1] (requerido por PoissonEncoder),
    // construye etiquetas 0-indexadas, divide 80% train / 20% test,
    // y agrega dos valores de bias (1.0) al final de cada vector de entrada
    // (uno para la capa oculta y otro para la de salida).
    void loadData() {
        static const char* filenames[] = {
            "Dataset_NN_1.csv",
            "Dataset_NN_2.csv",
            "Dataset_NN_3.csv",
            "Dataset_NN_4.csv",
            "Dataset_NN_5.csv",
        };
        if (dataNo < 1 || dataNo > 5)
            throw std::runtime_error("SparseSNN: dataNo must be 1-5");

        std::string fname = dataPath + "/" + filenames[dataNo - 1];
        std::ifstream file(fname);
        if (!file.is_open())
            throw std::runtime_error("SparseSNN: cannot open dataset: " + fname);

        std::vector<std::vector<double>> rows;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::vector<double> row;
            std::string token;
            while (std::getline(ss, token, ','))
                row.push_back(std::stod(token));
            rows.push_back(row);
        }

        int totalSamples = static_cast<int>(rows.size());
        int totalCols    = static_cast<int>(rows[0].size());
        nFeatures = totalCols - 1;

        // Build raw matrices
        std::vector<std::vector<double>> inputs(totalSamples, std::vector<double>(nFeatures));
        std::vector<double> labelRaw(totalSamples);
        for (int i = 0; i < totalSamples; i++) {
            for (int j = 0; j < nFeatures; j++) inputs[i][j] = rows[i][j];
            labelRaw[i] = rows[i][nFeatures];
        }

        // Min-max normalization to [0,1] per feature (RateEncoder expects [0,1])
        std::vector<double> featureMin(nFeatures,  1e18);
        std::vector<double> featureMax(nFeatures, -1e18);
        for (int i = 0; i < totalSamples; i++)
            for (int j = 0; j < nFeatures; j++) {
                featureMin[j] = std::min(featureMin[j], inputs[i][j]);
                featureMax[j] = std::max(featureMax[j], inputs[i][j]);
            }
        for (int i = 0; i < totalSamples; i++)
            for (int j = 0; j < nFeatures; j++) {
                double range = featureMax[j] - featureMin[j];
                inputs[i][j] = (range < 1e-10) ? 0.5
                                                : (inputs[i][j] - featureMin[j]) / range;
            }

        // Build 0-indexed integer labels
        std::set<double> catSet(labelRaw.begin(), labelRaw.end());
        std::vector<double> cats(catSet.begin(), catSet.end());
        nClasses = static_cast<int>(cats.size());
        nOutputs = nClasses;

        std::vector<int> labels(totalSamples);
        for (int i = 0; i < totalSamples; i++)
            labels[i] = static_cast<int>(
                std::find(cats.begin(), cats.end(), labelRaw[i]) - cats.begin());

        // 80/20 train/test split
        int trainSize = static_cast<int>(std::ceil(totalSamples * 0.8));
        nSamples = trainSize;

        trainDataset.clear();
        trainDataset.reserve(trainSize);
        for (int i = 0; i < trainSize; i++) {
            auto inp = inputs[i];
            inp.push_back(1.0);  // bias for hidden layer (always active)
            inp.push_back(1.0);  // bias for output layer (always active)
            trainDataset.emplace_back(std::move(inp), labels[i]);
        }
    }

    // Construye un par (Network, Simulator) con la topología de la SNN.
    // Topología: nFeatures neuronas de entrada + 2 bias (uno para capa oculta, otro para salida)
    //            → nHidden neuronas ocultas → nOutputs neuronas de salida.
    // Todas las sinapsis son excitatorias con peso inicial 0.0 (se asignan con setWeights()).
    // Configuración de simulación: dt=1ms, encoding_duration=50ms, evaluation_duration=100ms.
    // Encoder: PoissonEncoder con max 100 Hz.
    // Retorna: par (Network*, Simulator*) envueltos en unique_ptr.
    std::pair<std::unique_ptr<Network>, std::unique_ptr<Simulator>> makeNetSim() {
        SimulationConfig cfg;
        cfg.dt                  = dt;
        cfg.encoding_duration   = encoding_duration;
        cfg.evaluation_duration = evaluation_duration;
        cfg.verbose             = false;

        auto n = std::make_unique<Network>(cfg.dt, /*allow_recurrent=*/false);

        for (int i = 0; i < nFeatures; i++)
            n->addInputNeuron(NeuronType::REGULAR_SPIKING);
        int bias_h_id = n->addInputNeuron(NeuronType::REGULAR_SPIKING);  // ID nFeatures
        int bias_o_id = n->addInputNeuron(NeuronType::REGULAR_SPIKING);  // ID nFeatures+1

        for (int h = 0; h < nHidden; h++)
            n->addHiddenNeuron(NeuronType::REGULAR_SPIKING);
        for (int o = 0; o < nOutputs; o++)
            n->addOutputNeuron(NeuronType::REGULAR_SPIKING);

        int firstHidden = nFeatures + 2;
        int firstOutput = nFeatures + 2 + nHidden;

        // Layer 1: (nFeatures+1) inputs+bias → nHidden hidden
        for (int h = 0; h < nHidden; h++) {
            for (int i = 0; i < nFeatures; i++)
                n->addSynapse(i, firstHidden + h, true, 0.0);
            n->addSynapse(bias_h_id, firstHidden + h, true, 0.0);
        }
        // Layer 2: nHidden hidden + bias → nOutputs output
        for (int o = 0; o < nOutputs; o++) {
            for (int h = 0; h < nHidden; h++)
                n->addSynapse(firstHidden + h, firstOutput + o, true, 0.0);
            n->addSynapse(bias_o_id, firstOutput + o, true, 0.0);
        }

        auto s = std::make_unique<Simulator>(n.get(), cfg);
        s->setEncoder(std::make_unique<PoissonEncoder>(max_rate, true, refractory_period));
        return {std::move(n), std::move(s)};
    }

    // Crea un Network+Simulator independiente por cada thread OpenMP disponible.
    // Necesario porque Simulator/Network no son thread-safe: cada thread debe tener su propia instancia.
    // Se llama una sola vez en Setting() antes de cualquier evaluación.
    void buildPool() {
        int nThreads = omp_get_max_threads();
        net_pool.resize(nThreads);
        sim_pool.resize(nThreads);
        for (int t = 0; t < nThreads; t++) {
            auto [n, s] = makeNetSim();
            net_pool[t] = std::move(n);
            sim_pool[t] = std::move(s);
        }
    }
};

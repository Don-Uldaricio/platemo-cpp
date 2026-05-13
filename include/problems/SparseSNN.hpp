#pragma once
#include "../core/Problem.hpp"
#include "../utils/Random.hpp"

#include "core/network.hpp"
#include "core/simulator.hpp"
#include "encoding/rateEncoder.hpp"

#include <fstream>
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
//   (nFeatures+1) input+bias -> nHidden hidden neurons -> 1 output neuron (binary)
//   D = (nFeatures+1)*nHidden + (nHidden+1)*nOutputs synapses (all excitatory).
//
// Input normalization: min-max to [0,1] so RateEncoder receives valid inputs.
class SparseSNN : public Problem {
public:
    int dataNo   = 1;
    int nHidden  = 20;
    std::string dataPath;

    int nFeatures = 0;
    int nClasses  = 0;
    int nOutputs  = 0;
    int nSamples  = 0;

    // Training dataset for evaluateAccuracy(): (normalized_input, 0-indexed label)
    std::vector<std::pair<std::vector<double>, int>> trainDataset;

    SparseSNN(int dataNo_ = 1, int nHidden_ = 20,
              const std::string& dataPath_ = "data")
        : dataNo(dataNo_), nHidden(nHidden_), dataPath(dataPath_) {}

    void Setting() override {
        loadData();

        M = 2;
        D = (nFeatures + 1) * nHidden + (nHidden + 1) * nOutputs;
        lower    = Vector::Constant(D, 0.0);
        upper    = Vector::Constant(D, 1.0);
        encoding.assign(D, 1);  // all real

        buildPool();
    }

    Population Initialization(int n = -1) override {
        if (n < 0) n = N;
        // Sparse init: Bernoulli(0.5) mask over uniform [0,1] weights
        Matrix PopDec(n, D);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < D; j++)
                PopDec(i, j) = rng::uniform(0.0, 1.0) * rng::randint(0, 1);
        return Evaluation(PopDec);
    }

    // No gradient step — clamp only
    Matrix CalDec(const Matrix& dec) override {
        Matrix d = dec;
        for (int j = 0; j < D; j++)
            d.col(j) = d.col(j).cwiseMax(0.0).cwiseMin(1.0);
        return d;
    }

    Matrix CalObj(const Matrix& dec) override {
        int n = dec.rows();
        Matrix obj(n, 2);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            int tid = omp_get_thread_num();

            std::vector<double> weights(D);
            for (int j = 0; j < D; j++) weights[j] = dec(i, j);

            int nzCount = 0;
            for (double w : weights) if (w > 0.0) nzCount++;
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

private:
    // One Network+Simulator per OpenMP thread — built once in buildPool().
    std::vector<std::unique_ptr<Network>>   net_pool;
    std::vector<std::unique_ptr<Simulator>> sim_pool;

    void loadData() {
        static const char* filenames[] = {
            "Dataset_NN_1.csv",
            "Dataset_NN_2.csv",
            "Dataset_NN_3.csv",
            "Dataset_NN_4.csv",
        };
        if (dataNo < 1 || dataNo > 4)
            throw std::runtime_error("SparseSNN: dataNo must be 1-4");

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
        nOutputs = (nClasses <= 2) ? 1 : nClasses;

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

    std::pair<std::unique_ptr<Network>, std::unique_ptr<Simulator>> makeNetSim() {
        SimulationConfig cfg;
        cfg.dt                  = 1.0;
        cfg.encoding_duration   = 50.0;
        cfg.evaluation_duration = 100.0;
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
        s->setEncoder(std::make_unique<RateEncoder>(100.0));
        return {std::move(n), std::move(s)};
    }

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

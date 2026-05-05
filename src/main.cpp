#include "core/Problem.hpp"
#include "core/Solution.hpp"
#include "problems/SparseNN.hpp"
#include "problems/SparseSNN.hpp"
#include "algorithms/moeackf/MOEACKF.hpp"
#include "algorithms/snsgaii/SNSGAII.hpp"
#include "metrics/HV.hpp"
#include "metrics/IGD.hpp"
#include "utils/Random.hpp"
#include "utils/NDSort.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <chrono>
#include <numeric>
#include <cmath>

struct RunConfig {
    std::string algorithm;
    std::string problem;
    int dataNo;
    int nHidden;
    int N;
    int maxFE;
    int nRuns;
    unsigned seed;
    std::string dataPath;
    bool verbose;
    std::string outFile;
};

void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --algo      <MOEACKF|SNSGAII>  Algorithm (default: MOEACKF)\n"
              << "  --problem   <SparseNN|SparseSNN> Problem (default: SparseNN)\n"
              << "  --dataset   <1-4>              Dataset number (default: 1)\n"
              << "  --nhidden   <int>              Hidden layer size (default: 20)\n"
              << "  --popsize   <int>              Population size (default: 100)\n"
              << "  --maxfe     <int>              Max evaluations (default: 10000)\n"
              << "  --runs      <int>              Number of independent runs (default: 1)\n"
              << "  --seed      <uint>             Random seed (default: random)\n"
              << "  --datapath  <path>             Path to CSV data files (default: data)\n"
              << "  --out       <file>             Save Pareto front to CSV (optional)\n"
              << "  --verbose                      Print per-generation progress\n"
              << "\nProblems:\n"
              << "  SparseNN  : Multi-layer ANN with backprop fine-tuning (baseline)\n"
              << "  SparseSNN : Spiking neural network (Izhikevich) evaluated via spike decoding\n"
              << "\nDataset info:\n"
              << "  1: Statlog_Australian  (690 samples, 14 features, 2 classes)\n"
              << "  2: Climate             (540 samples, 18 features, 2 classes)\n"
              << "  3: Statlog_German      (1000 samples, 24 features, 2 classes)\n"
              << "  4: Connectionist Sonar (208 samples, 60 features, 2 classes)\n"
              << "\nObjectives (both minimized):\n"
              << "  f1: Network complexity (fraction of non-zero weights)\n"
              << "  f2: Training classification error\n"
              << "\nMetrics reported:\n"
              << "  HV  (Hypervolume indicator, higher is better)\n"
              << "  IGD (Inverted Generational Distance, lower is better)\n";
}

// Save Pareto front objectives to a CSV file
void saveParetoFront(const Population& pop, const std::string& fname) {
    Population best = getBest(pop);
    if (best.empty()) { std::cerr << "No feasible solutions to save.\n"; return; }

    std::ofstream f(fname);
    if (!f.is_open()) { std::cerr << "Cannot open " << fname << "\n"; return; }

    f << "f1_complexity,f2_train_error\n";
    for (const auto& s : best)
        f << s.obj(0) << "," << s.obj(1) << "\n";

    std::cout << "  Pareto front saved to: " << fname
              << " (" << best.size() << " solutions)\n";
}

int main(int argc, char* argv[]) {
    RunConfig cfg;
    cfg.algorithm = "MOEACKF";
    cfg.problem   = "SparseNN";
    cfg.dataNo    = 1;
    cfg.nHidden   = 20;
    cfg.N         = 100;
    cfg.maxFE     = 10000;
    cfg.nRuns     = 1;
    cfg.seed      = std::random_device{}();
    cfg.dataPath  = "data";
    cfg.verbose   = false;
    cfg.outFile   = "";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printHelp(argv[0]); return 0; }
        else if (arg == "--algo"     && i+1 < argc) cfg.algorithm = argv[++i];
        else if (arg == "--problem"  && i+1 < argc) cfg.problem   = argv[++i];
        else if (arg == "--dataset"  && i+1 < argc) cfg.dataNo    = std::stoi(argv[++i]);
        else if (arg == "--nhidden"  && i+1 < argc) cfg.nHidden   = std::stoi(argv[++i]);
        else if (arg == "--popsize"  && i+1 < argc) cfg.N         = std::stoi(argv[++i]);
        else if (arg == "--maxfe"    && i+1 < argc) cfg.maxFE     = std::stoi(argv[++i]);
        else if (arg == "--runs"     && i+1 < argc) cfg.nRuns     = std::stoi(argv[++i]);
        else if (arg == "--seed"     && i+1 < argc) cfg.seed      = std::stoul(argv[++i]);
        else if (arg == "--datapath" && i+1 < argc) cfg.dataPath  = argv[++i];
        else if (arg == "--out"      && i+1 < argc) cfg.outFile   = argv[++i];
        else if (arg == "--verbose")                cfg.verbose   = true;
        else { std::cerr << "Unknown argument: " << arg << "\n"; printHelp(argv[0]); return 1; }
    }

    std::cout << "============================================\n"
              << "PlatEMO-CPP: Sparse Multi-objective EA\n"
              << "============================================\n"
              << "Algorithm : " << cfg.algorithm << "\n"
              << "Problem   : " << cfg.problem << "\n"
              << "Dataset   : " << cfg.dataNo << "\n"
              << "nHidden   : " << cfg.nHidden << "\n"
              << "Pop size  : " << cfg.N << "\n"
              << "Max FE    : " << cfg.maxFE << "\n"
              << "Runs      : " << cfg.nRuns << "\n"
              << "Seed base : " << cfg.seed << "\n"
              << "Data path : " << cfg.dataPath << "\n"
              << "============================================\n\n";

    std::vector<double> hvVals, igdVals, timeVals;

    for (int run = 0; run < cfg.nRuns; run++) {
        unsigned runSeed = cfg.seed + run;
        rng::seed(runSeed);

        std::cout << "--- Run " << (run+1) << "/" << cfg.nRuns
                  << " (seed=" << runSeed << ") ---\n";

        // Create and initialize problem
        std::unique_ptr<Problem> probPtr;
        if (cfg.problem == "SparseSNN") {
            auto p = std::make_unique<SparseSNN>(cfg.dataNo, cfg.nHidden, cfg.dataPath);
            p->N     = cfg.N;
            p->maxFE = cfg.maxFE;
            probPtr  = std::move(p);
        } else if (cfg.problem == "SparseNN") {
            auto p = std::make_unique<SparseNN>(cfg.dataNo, cfg.nHidden, cfg.dataPath);
            p->N     = cfg.N;
            p->maxFE = cfg.maxFE;
            probPtr  = std::move(p);
        } else {
            std::cerr << "Unknown problem: " << cfg.problem << "\n";
            return 1;
        }
        Problem& prob = *probPtr;

        try {
            prob.init();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }

        int nSamples = (cfg.problem == "SparseSNN")
            ? static_cast<SparseSNN&>(prob).nSamples
            : static_cast<SparseNN&>(prob).nSamples;

        std::cout << "  D=" << prob.D << " (weights), M=2 (objectives)"
                  << ", train samples=" << nSamples << "\n";

        auto tStart = std::chrono::steady_clock::now();

        // Run algorithm
        Population finalPop;
        if (cfg.algorithm == "MOEACKF") {
            finalPop = MOEACKF(prob, cfg.verbose);
        } else if (cfg.algorithm == "SNSGAII") {
            finalPop = SNSGAII(prob, cfg.verbose);
        } else {
            std::cerr << "Unknown algorithm: " << cfg.algorithm << "\n";
            return 1;
        }

        auto tEnd  = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(tEnd - tStart).count();

        // Compute metrics
        double hvScore  = HV(finalPop, prob.optimum);
        double igdScore = IGD(finalPop, prob.optimum);

        hvVals.push_back(hvScore);
        igdVals.push_back(igdScore);
        timeVals.push_back(elapsed);

        // Report results
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  FE used : " << prob.FE << "\n"
                  << "  Time    : " << std::setprecision(2) << elapsed << " s\n"
                  << "  HV      : " << std::setprecision(6) << hvScore  << "\n"
                  << "  IGD     : " << igdScore  << "\n";

        // Print Pareto front summary
        Population best = getBest(finalPop);
        std::cout << "  Pareto front size: " << best.size() << "\n";
        if (!best.empty()) {
            double minErr = best[0].obj(1), maxErr = best[0].obj(1);
            double minCmp = best[0].obj(0), maxCmp = best[0].obj(0);
            for (const auto& s : best) {
                minErr = std::min(minErr, s.obj(1));
                maxErr = std::max(maxErr, s.obj(1));
                minCmp = std::min(minCmp, s.obj(0));
                maxCmp = std::max(maxCmp, s.obj(0));
            }
            std::cout << "  Complexity range: [" << minCmp << ", " << maxCmp << "]\n"
                      << "  Train error range: [" << minErr << ", " << maxErr << "]\n";
        }
        std::cout << "\n";

        // Save last run Pareto front if requested
        if (!cfg.outFile.empty() && run == cfg.nRuns - 1)
            saveParetoFront(finalPop, cfg.outFile);
    }

    // Summary statistics over multiple runs
    if (cfg.nRuns > 1) {
        auto stats = [](const std::vector<double>& v) -> std::pair<double,double> {
            double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            double var  = 0;
            for (double x : v) var += (x-mean)*(x-mean);
            return {mean, std::sqrt(var / v.size())};
        };

        auto [hvMean, hvStd]   = stats(hvVals);
        auto [igdMean, igdStd] = stats(igdVals);
        auto [tMean, tStd]     = stats(timeVals);

        std::cout << "============================================\n"
                  << "Summary over " << cfg.nRuns << " runs:\n"
                  << "  HV  mean±std: " << std::fixed << std::setprecision(6)
                  << hvMean << " ± " << hvStd << "\n"
                  << "  IGD mean±std: " << igdMean << " ± " << igdStd << "\n"
                  << "  Time mean   : " << std::setprecision(2) << tMean << " s\n"
                  << "============================================\n";
    }

    return 0;
}

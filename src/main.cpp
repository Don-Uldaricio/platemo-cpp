#include "core/Problem.hpp"
#include "core/Solution.hpp"
#include "core/AlgoConfig.hpp"
#include "problems/SparseNN.hpp"
#include "problems/SparseSNN.hpp"
#include "algorithms/moeackf/MOEACKF.hpp"
#include "algorithms/snsgaii/SNSGAII.hpp"
#include "metrics/HV.hpp"
#include "utils/Random.hpp"
#include "utils/NDSort.hpp"
#include "encoding/poissonEncoder.hpp"
#include "encoding/ttfsEncoder.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <set>
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
    bool   verbose     = false;
    bool   sanityCheck = false;
    double wScale      = 1.0;
    std::string outFile;
    std::string csvOutFile;
    std::string convOutFile;
    std::string spikesOutFile;  // CSV con spike counts del test set por solución del frente

    // MOEA operator hyperparameters (tuneable via CLI / Bayesian optimization)
    AlgoConfig acfg;

    // SNN simulation timing hyperparameters
    double dt                  = 1.0;
    double encoding_duration   = 50.0;
    double evaluation_duration = 100.0;
    double max_rate            = 100.0;
    double refractory_period   = 5.0;

    // SparseSNN binary fitness configuration
    FitnessMode fitnessMode   = FitnessMode::ACCURACY;
    int accuracyThreshold     = 1;
    // false (default) → 1 output neuron (threshold/AUC); true → 2 output neurons (WTA)
    bool twoOutputBinary      = false;

    // Encoder de spikes
    EncoderType encoderType   = EncoderType::POISSON;

    // Bootstrap resampling fitness evaluation
    bool bootstrapEval = false;
    int  bootstrapN    = 50;
    int  bootstrapSize = 50;
};

void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --algo      <MOEACKF|SNSGAII>  Algorithm (default: MOEACKF)\n"
              << "  --problem   <SparseNN|SparseSNN> Problem (default: SparseNN)\n"
              << "  --dataset   <1-6>              Dataset number (default: 1)\n"
              << "  --nhidden   <int>              Hidden layer size (default: 20)\n"
              << "  --popsize   <int>              Population size (default: 100)\n"
              << "  --maxfe     <int>              Max evaluations (default: 10000)\n"
              << "  --runs      <int>              Number of independent runs (default: 1)\n"
              << "  --seed      <uint>             Random seed (default: random)\n"
              << "  --datapath  <path>             Path to CSV data files (default: data)\n"
              << "  --out       <file>             Save Pareto front to CSV (optional)\n"
              << "  --csv-out   <file>             Append per-run metrics to CSV (optional)\n"
              << "  --conv-out  <file>             Append per-checkpoint HV history to CSV (optional)\n"
              << "  --conv-out logs HV at every generation (generation column in CSV)\n"
              << "  --wscale    <float>            Weight scale factor applied at evaluation (default: 1.0)\n"
              << "  --sanity-check                 Evaluate extreme weight configs and exit (no GA run)\n"
              << "  --verbose                      Print per-generation progress\n"
              << "\nMOEA operator hyperparameters:\n"
              << "  --disC      <float>            SBX distribution index (default: 20.0)\n"
              << "  --disM      <float>            Polynomial mutation distribution index (default: 20.0)\n"
              << "  --proM      <float>            Polynomial mutation probability multiplier (default: 1.0)\n"
              << "  --disSM     <float>            Sparsity mutation distribution index (default: 20.0)\n"
              << "  --proSM     <float>            Sparsity mutation probability multiplier (default: 1.0)\n"
              << "  --sLower    <float>            VSSPS minimum sparsity (default: 0.75)\n"
              << "  --sUpper    <float>            VSSPS maximum sparsity (default: 1.0)\n"
              << "\nSNN simulation timing hyperparameters:\n"
              << "  --dt                  <float>  Simulation timestep ms (default: 1.0)\n"
              << "  --encoding-duration   <float>  Poisson encoding window ms (default: 50.0)\n"
              << "  --eval-duration       <float>  Total simulation window ms (default: 100.0)\n"
              << "  --max-rate            <float>  PoissonEncoder max firing rate Hz (default: 100.0)\n"
              << "  --refractory-period   <float>  PoissonEncoder refractory period ms (default: 5.0)\n"
              << "  --encoder             <poisson|ttfs>  Spike encoder type (default: poisson)\n"
              << "\nSparseSNN binary classification options (nClasses==2):\n"
              << "  --binary-outputs <1|2>         Output neurons for binary tasks: 1=threshold/AUC (default), 2=WTA\n"
              << "  --fitness-mode <accuracy|auc>  f2 metric: accuracy with fixed threshold or AUC (default: accuracy)\n"
              << "                                 (only used when --binary-outputs 1)\n"
              << "  --threshold    <int>           Spike threshold for --fitness-mode accuracy (default: 1)\n"
              << "  --spikes-out   <file>          Save test-set spike counts per Pareto solution to CSV\n"
              << "                                 (enables ROC/AUC plotting with plot_roc.py; only with --binary-outputs 1)\n"
              << "  --bootstrap-eval               Enable bootstrap resampling fitness evaluation\n"
              << "  --bootstrap-n   <int>          Number of bootstrap subsets per evaluation (default: 50)\n"
              << "  --bootstrap-size <int>         Samples per bootstrap subset, with replacement (default: 50)\n"
              << "\nProblems:\n"
              << "  SparseNN  : Multi-layer ANN with backprop fine-tuning (baseline)\n"
              << "  SparseSNN : Spiking neural network (Izhikevich) evaluated via spike decoding\n"
              << "\nDataset info:\n"
              << "  1: Statlog_Australian  (690 samples, 14 features, 2 classes)\n"
              << "  2: Climate             (540 samples, 18 features, 2 classes)\n"
              << "  3: Statlog_German      (1000 samples, 24 features, 2 classes)\n"
              << "  4: Connectionist Sonar (208 samples, 60 features, 2 classes)\n"
              << "  6: Wine                (178 samples, 13 features, 3 classes)\n"
              << "\nObjectives (both minimized):\n"
              << "  f1: Network complexity (fraction of non-zero weights)\n"
              << "  f2: Training classification error\n"
              << "\nMetrics reported:\n"
              << "  HV  (Hypervolume indicator, higher is better)\n";
}

// Save Pareto front objectives + weight vectors to a CSV file.
// Also writes a companion _meta.json with topology info for the Python visualizer.
void saveParetoFront(const Population& pop, const std::string& fname,
                     const RunConfig& rcfg, int nFeatures, int nHidden, int nOutputs) {
    Population best = getBest(pop);
    if (best.empty()) { std::cerr << "No feasible solutions to save.\n"; return; }

    std::ofstream f(fname);
    if (!f.is_open()) { std::cerr << "Cannot open " << fname << "\n"; return; }

    int D = best.empty() ? 0 : static_cast<int>(best[0].dec.size());

    // Header: objectives + one column per weight
    f << "f1_complexity,f2_train_error";
    for (int j = 0; j < D; j++) f << ",w" << j;
    f << "\n";

    std::set<std::pair<double,double>> seen;
    int written = 0;
    for (const auto& s : best) {
        if (seen.emplace(s.obj(0), s.obj(1)).second) {
            f << std::setprecision(10) << s.obj(0) << "," << s.obj(1);
            for (int j = 0; j < D; j++) f << "," << s.dec(j);
            f << "\n";
            ++written;
        }
    }

    std::cout << "  Pareto front saved to: " << fname
              << " (" << written << " solutions, D=" << D << " weights)\n";

    // Write companion metadata JSON for the Python visualizer
    std::string metaFname = fname;
    auto pos = metaFname.rfind("_front.csv");
    if (pos != std::string::npos)
        metaFname.replace(pos, 10, "_meta.json");
    else
        metaFname += ".meta.json";

    std::ofstream mf(metaFname);
    if (mf.is_open()) {
        std::string fmStr = (rcfg.fitnessMode == FitnessMode::AUC) ? "auc" : "accuracy";
        mf << "{\n"
           << "  \"nFeatures\": "   << nFeatures        << ",\n"
           << "  \"nHidden\": "     << nHidden          << ",\n"
           << "  \"nOutputs\": "    << nOutputs         << ",\n"
           << "  \"wScale\": "      << rcfg.wScale      << ",\n"
           << "  \"dataNo\": "      << rcfg.dataNo      << ",\n"
           << "  \"D\": "           << D                << ",\n"
           << "  \"fitness_mode\": \"" << fmStr        << "\"\n"
           << "}\n";
        std::cout << "  Metadata saved to:    " << metaFname << "\n";
    }
}

// Evalúa el test set para cada solución del frente de Pareto y guarda spike counts a CSV.
// Columnas: solution_id, f1_complexity, f2_train_metric, sample_id, spike_count, true_label
// Solo para SparseSNN con nOutputs==1 (modo binario).
void exportSpikeCountsCSV(
    const Population& pop,
    SparseSNN& prob,
    const std::string& fname,
    const RunConfig& rcfg)
{
    Population best = getBest(pop);
    if (best.empty()) { std::cerr << "No feasible solutions for spikes export.\n"; return; }

    auto [n, s] = prob.makeNetSim();

    std::ofstream f(fname);
    if (!f.is_open()) { std::cerr << "Cannot open " << fname << "\n"; return; }

    f << "solution_id,f1_complexity,f2_train_metric,sample_id,spike_count,true_label\n";

    std::set<std::pair<double,double>> seen;
    int solId = 0;
    for (const auto& sol : best) {
        if (!seen.emplace(sol.obj(0), sol.obj(1)).second) continue;

        std::vector<double> weights(prob.D);
        for (int j = 0; j < prob.D; j++)
            weights[j] = sol.dec(j) * rcfg.wScale;
        s->setWeights(weights);

        auto counts = s->collectSpikeCounts(prob.testDataset);
        for (size_t i = 0; i < counts.size(); ++i) {
            f << solId << ","
              << std::setprecision(10) << sol.obj(0) << ","
              << sol.obj(1) << ","
              << i << ","
              << counts[i] << ","
              << prob.testDataset[i].second << "\n";
        }
        ++solId;
    }

    int T_max = static_cast<int>(
        (rcfg.evaluation_duration - rcfg.encoding_duration) / rcfg.dt);

    std::cout << "  Spike counts saved to: " << fname
              << " (" << solId << " solutions, T_max=" << T_max << ")\n";

    // Escribir T_max en el JSON de metadatos para que Python lo use
    std::string metaFname = fname;
    auto pos = metaFname.rfind("_spikes.csv");
    if (pos != std::string::npos)
        metaFname.replace(pos, 11, "_meta.json");
    else
        metaFname += ".meta.json";

    // Si ya existe el _meta.json de saveParetoFront, abrir y añadir T_max;
    // si no, crear uno nuevo con lo mínimo.
    std::ofstream mf(metaFname);
    if (mf.is_open()) {
        mf << "{\n"
           << "  \"nFeatures\": " << prob.nFeatures << ",\n"
           << "  \"nHidden\": "   << prob.nHidden   << ",\n"
           << "  \"nOutputs\": "  << prob.nOutputs  << ",\n"
           << "  \"nClasses\": "  << prob.nClasses  << ",\n"
           << "  \"wScale\": "    << rcfg.wScale    << ",\n"
           << "  \"dataNo\": "    << rcfg.dataNo    << ",\n"
           << "  \"T_max\": "     << T_max           << ",\n"
           << "  \"D\": "         << prob.D          << "\n"
           << "}\n";
        std::cout << "  Metadata saved to:     " << metaFname << "\n";
    }
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
    cfg.verbose     = false;
    cfg.outFile     = "";
    cfg.csvOutFile  = "";
    cfg.convOutFile = "";
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
        else if (arg == "--out"          && i+1 < argc) cfg.outFile     = argv[++i];
        else if (arg == "--csv-out"      && i+1 < argc) cfg.csvOutFile  = argv[++i];
        else if (arg == "--conv-out"     && i+1 < argc) cfg.convOutFile = argv[++i];
        else if (arg == "--wscale"       && i+1 < argc) cfg.wScale       = std::stod(argv[++i]);
        else if (arg == "--sanity-check")                cfg.sanityCheck  = true;
        else if (arg == "--verbose")                     cfg.verbose      = true;
        // MOEA operator hyperparameters
        else if (arg == "--disC"   && i+1 < argc) cfg.acfg.disC   = std::stod(argv[++i]);
        else if (arg == "--disM"   && i+1 < argc) cfg.acfg.disM   = std::stod(argv[++i]);
        else if (arg == "--proM"   && i+1 < argc) cfg.acfg.proM   = std::stod(argv[++i]);
        else if (arg == "--disSM"  && i+1 < argc) cfg.acfg.disSM  = std::stod(argv[++i]);
        else if (arg == "--proSM"  && i+1 < argc) cfg.acfg.proSM  = std::stod(argv[++i]);
        else if (arg == "--sLower" && i+1 < argc) cfg.acfg.sLower = std::stod(argv[++i]);
        else if (arg == "--sUpper" && i+1 < argc) cfg.acfg.sUpper = std::stod(argv[++i]);
        // SNN simulation timing hyperparameters
        else if (arg == "--dt"                 && i+1 < argc) cfg.dt                  = std::stod(argv[++i]);
        else if (arg == "--encoding-duration"  && i+1 < argc) cfg.encoding_duration   = std::stod(argv[++i]);
        else if (arg == "--eval-duration"      && i+1 < argc) cfg.evaluation_duration = std::stod(argv[++i]);
        else if (arg == "--max-rate"           && i+1 < argc) cfg.max_rate            = std::stod(argv[++i]);
        else if (arg == "--refractory-period"  && i+1 < argc) cfg.refractory_period   = std::stod(argv[++i]);
        else if (arg == "--encoder" && i+1 < argc) {
            std::string enc = argv[++i];
            if (enc == "poisson")       cfg.encoderType = EncoderType::POISSON;
            else if (enc == "ttfs")     cfg.encoderType = EncoderType::TTFS;
            else { std::cerr << "Unknown --encoder: " << enc << " (use poisson|ttfs)\n"; return 1; }
        }
        // SparseSNN binary fitness options
        else if (arg == "--fitness-mode" && i+1 < argc) {
            std::string fm = argv[++i];
            if (fm == "auc")           cfg.fitnessMode = FitnessMode::AUC;
            else if (fm == "accuracy") cfg.fitnessMode = FitnessMode::ACCURACY;
            else { std::cerr << "Unknown --fitness-mode: " << fm << " (use accuracy|auc)\n"; return 1; }
        }
        else if (arg == "--threshold"       && i+1 < argc) cfg.accuracyThreshold = std::stoi(argv[++i]);
        else if (arg == "--spikes-out"      && i+1 < argc) cfg.spikesOutFile     = argv[++i];
        else if (arg == "--binary-outputs"  && i+1 < argc) {
            int bo = std::stoi(argv[++i]);
            if (bo != 1 && bo != 2) {
                std::cerr << "--binary-outputs must be 1 or 2\n"; return 1;
            }
            cfg.twoOutputBinary = (bo == 2);
        }
        else if (arg == "--bootstrap-eval")               cfg.bootstrapEval = true;
        else if (arg == "--bootstrap-n"    && i+1 < argc) cfg.bootstrapN    = std::stoi(argv[++i]);
        else if (arg == "--bootstrap-size" && i+1 < argc) cfg.bootstrapSize = std::stoi(argv[++i]);
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
              << "Encoder   : " << (cfg.encoderType == EncoderType::TTFS ? "ttfs" : "poisson") << "\n"
              << "============================================\n\n";

    std::vector<double> hvVals, timeVals;

    for (int run = 0; run < cfg.nRuns; run++) {
        unsigned runSeed = cfg.seed + run;
        rng::seed(runSeed);

        std::cout << "--- Run " << (run+1) << "/" << cfg.nRuns
                  << " (seed=" << runSeed << ") ---\n";

        // Create and initialize problem
        std::unique_ptr<Problem> probPtr;
        if (cfg.problem == "SparseSNN") {
            auto p = std::make_unique<SparseSNN>(cfg.dataNo, cfg.nHidden, cfg.dataPath);
            p->N                  = cfg.N;
            p->maxFE              = cfg.maxFE;
            p->wScale             = cfg.wScale;
            p->dt                 = cfg.dt;
            p->encoding_duration  = cfg.encoding_duration;
            p->evaluation_duration= cfg.evaluation_duration;
            p->max_rate           = cfg.max_rate;
            p->refractory_period  = cfg.refractory_period;
            p->twoOutputBinary    = cfg.twoOutputBinary;
            p->fitnessMode        = cfg.fitnessMode;
            p->accuracyThreshold  = cfg.accuracyThreshold;
            p->encoderType        = cfg.encoderType;
            p->bootstrapEval      = cfg.bootstrapEval;
            p->bootstrapN         = cfg.bootstrapN;
            p->bootstrapSize      = cfg.bootstrapSize;
            probPtr   = std::move(p);
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

        if (cfg.sanityCheck) {
            if (cfg.problem != "SparseSNN") {
                std::cerr << "--sanity-check only supported for SparseSNN\n";
                return 1;
            }
            static_cast<SparseSNN&>(prob).runSanityCheck();
            return 0;
        }

        int nSamples = (cfg.problem == "SparseSNN")
            ? static_cast<SparseSNN&>(prob).nSamples
            : static_cast<SparseNN&>(prob).nSamples;

        std::cout << "  D=" << prob.D << " (weights), M=2 (objectives)"
                  << ", train samples=" << nSamples << "\n";

        // Setup convergence logging (HV por generación)
        if (!cfg.convOutFile.empty()) {
            prob.logGenInterval = 1;
            prob.onLog = [&](int gen, const Population& pop) {
                double hv = HV(pop, prob.optimum);
                std::ofstream f(cfg.convOutFile, std::ios::app);
                if (!f.is_open()) return;
                f.seekp(0, std::ios::end);
                if (f.tellp() == 0)
                    f << "algo,problem,dataset,nhidden,popsize,maxfe,run,seed,generation,hv\n";
                f << cfg.algorithm << ","
                  << cfg.problem   << ","
                  << cfg.dataNo    << ","
                  << cfg.nHidden   << ","
                  << cfg.N         << ","
                  << cfg.maxFE     << ","
                  << (run + 1)     << ","
                  << runSeed       << ","
                  << gen           << ","
                  << std::fixed << std::setprecision(6) << hv << "\n";
            };
        }

        auto tStart = std::chrono::steady_clock::now();

        // Run algorithm
        Population finalPop;
        if (cfg.algorithm == "MOEACKF") {
            finalPop = MOEACKF(prob, cfg.verbose, cfg.acfg);
        } else if (cfg.algorithm == "SNSGAII") {
            finalPop = SNSGAII(prob, cfg.verbose, cfg.acfg);
        } else {
            std::cerr << "Unknown algorithm: " << cfg.algorithm << "\n";
            return 1;
        }

        auto tEnd  = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(tEnd - tStart).count();

        // Compute metrics
        double hvScore = HV(finalPop, prob.optimum);

        hvVals.push_back(hvScore);
        timeVals.push_back(elapsed);

        if (!cfg.csvOutFile.empty()) {
            std::ofstream csvf(cfg.csvOutFile, std::ios::app);
            if (csvf.is_open()) {
                csvf.seekp(0, std::ios::end);
                if (csvf.tellp() == 0)
                    csvf << "algo,problem,dataset,nhidden,popsize,maxfe,run,seed,hv,time_s\n";
                csvf << cfg.algorithm << ","
                     << cfg.problem   << ","
                     << cfg.dataNo    << ","
                     << cfg.nHidden   << ","
                     << cfg.N         << ","
                     << cfg.maxFE     << ","
                     << (run + 1)     << ","
                     << runSeed       << ","
                     << std::fixed << std::setprecision(6) << hvScore << ","
                     << std::setprecision(4) << elapsed    << "\n";
            }
        }

        // Report results
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  FE used : " << prob.FE << "\n"
                  << "  Time    : " << std::setprecision(2) << elapsed << " s\n"
                  << "  HV      : " << std::setprecision(6) << hvScore  << "\n";

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
        if (!cfg.outFile.empty() && run == cfg.nRuns - 1) {
            int nF = 0, nH = cfg.nHidden, nO = 0;
            if (cfg.problem == "SparseSNN") {
                auto& sp = static_cast<SparseSNN&>(prob);
                nF = sp.nFeatures;
                nO = sp.nOutputs;
            }
            saveParetoFront(finalPop, cfg.outFile, cfg, nF, nH, nO);
        }

        // Export test-set spike counts for ROC/AUC analysis (SparseSNN binary only)
        if (!cfg.spikesOutFile.empty() && run == cfg.nRuns - 1
            && cfg.problem == "SparseSNN") {
            auto& sp = static_cast<SparseSNN&>(prob);
            if (sp.nOutputs == 1) {
                exportSpikeCountsCSV(finalPop, sp, cfg.spikesOutFile, cfg);
            } else {
                std::cerr << "  --spikes-out skipped: not a binary problem (nOutputs="
                          << sp.nOutputs << ")\n";
            }
        }
    }

    // Summary statistics over multiple runs
    if (cfg.nRuns > 1) {
        auto stats = [](const std::vector<double>& v) -> std::pair<double,double> {
            double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            double var  = 0;
            for (double x : v) var += (x-mean)*(x-mean);
            return {mean, std::sqrt(var / v.size())};
        };

        auto [hvMean, hvStd] = stats(hvVals);
        auto [tMean, tStd]   = stats(timeVals);

        std::cout << "============================================\n"
                  << "Summary over " << cfg.nRuns << " runs:\n"
                  << "  HV  mean±std: " << std::fixed << std::setprecision(6)
                  << hvMean << " ± " << hvStd << "\n"
                  << "  Time mean   : " << std::setprecision(2) << tMean << " s\n"
                  << "============================================\n";
    }

    return 0;
}

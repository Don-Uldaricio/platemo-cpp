#pragma once
#include "Solution.hpp"
#include <chrono>
#include <limits>
#include <random>

class Problem {
public:
    int    N         = 100;
    int    maxFE     = 10000;
    int    FE        = 0;
    int    M         = 0;
    int    D         = 0;
    double maxRuntime = std::numeric_limits<double>::infinity();
    Vector lower;
    Vector upper;
    std::vector<int> encoding;  // 1=real, 2=integer, 3=label, 4=binary, 5=permutation
    Matrix optimum;
    std::chrono::steady_clock::time_point startTime;

    virtual ~Problem() = default;

    virtual void Setting() = 0;

    virtual Population Initialization(int n = -1) {
        if (n < 0) n = N;
        Matrix PopDec(n, D);
        for (int j = 0; j < D; j++) {
            std::uniform_real_distribution<double> dist(lower(j), upper(j));
            for (int i = 0; i < n; i++)
                PopDec(i, j) = dist(rng_engine());
        }
        return Evaluation(PopDec);
    }

    virtual Population Evaluation(const Matrix& dec) {
        Matrix PopDec = CalDec(dec);
        Matrix PopObj = CalObj(PopDec);
        Matrix PopCon = CalCon(PopDec);
        Population pop = makePopulation(PopDec, PopObj, PopCon);
        FE += pop.size();
        return pop;
    }

    virtual Matrix CalDec(const Matrix& dec) {
        Matrix d = dec;
        for (int j = 0; j < D; j++) {
            d.col(j) = d.col(j).cwiseMax(lower(j)).cwiseMin(upper(j));
            if (j < (int)encoding.size() &&
                (encoding[j] == 2 || encoding[j] == 3 || encoding[j] == 4))
                d.col(j) = d.col(j).array().round();
        }
        return d;
    }

    virtual Matrix CalObj(const Matrix& dec) {
        return Matrix::Zero(dec.rows(), 1);
    }

    virtual Matrix CalCon(const Matrix& dec) {
        return Matrix::Zero(dec.rows(), 1);
    }

    virtual Matrix GetOptimum(int /*n*/) {
        if (M > 1) return Matrix::Ones(1, M);
        return Matrix::Zero(1, 1);
    }

    bool NotTerminated(const Population& /*pop*/) {
        return FE < maxFE;
    }

    void init() {
        startTime = std::chrono::steady_clock::now();
        Setting();
        optimum = GetOptimum(10000);
        FE = 0;
    }

protected:
    static std::mt19937& rng_engine() {
        static std::mt19937 e(std::random_device{}());
        return e;
    }
};

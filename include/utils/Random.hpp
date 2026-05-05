#pragma once
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

namespace rng {

inline std::mt19937& engine() {
    static std::mt19937 e(std::random_device{}());
    return e;
}

inline void seed(uint32_t s) { engine().seed(s); }

inline double uniform(double lo = 0.0, double hi = 1.0) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(engine());
}

inline int randint(int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    return d(engine());
}

// N x D matrix of uniform random values in [lo, hi]
inline Eigen::MatrixXd rand(int N, int D, double lo = 0.0, double hi = 1.0) {
    Eigen::MatrixXd M(N, D);
    std::uniform_real_distribution<double> dist(lo, hi);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            M(i,j) = dist(engine());
    return M;
}

// N x D matrix: each column j uniform in [lower(j), upper(j)]
inline Eigen::MatrixXd unifrndBounded(int N, const Eigen::VectorXd& lower, const Eigen::VectorXd& upper) {
    int D = lower.size();
    Eigen::MatrixXd M(N, D);
    for (int j = 0; j < D; j++) {
        std::uniform_real_distribution<double> dist(lower(j), upper(j));
        for (int i = 0; i < N; i++)
            M(i,j) = dist(engine());
    }
    return M;
}

// N x D matrix of random integers in [lo, hi]
inline Eigen::MatrixXi randi(int lo, int hi, int N, int D) {
    Eigen::MatrixXi M(N, D);
    std::uniform_int_distribution<int> dist(lo, hi);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            M(i,j) = dist(engine());
    return M;
}

// Random permutation of k elements from 0..n-1
inline std::vector<int> randperm(int n, int k = -1) {
    if (k < 0) k = n;
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::shuffle(v.begin(), v.end(), engine());
    return std::vector<int>(v.begin(), v.begin() + k);
}

} // namespace rng

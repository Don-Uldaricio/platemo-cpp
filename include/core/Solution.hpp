#pragma once
#include <vector>
#include <Eigen/Dense>
#include <numeric>
#include <algorithm>
#include <limits>

using Matrix   = Eigen::MatrixXd;
using MatrixB  = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
using Vector   = Eigen::VectorXd;
using RowVec   = Eigen::RowVectorXd;
using VectorI  = Eigen::VectorXi;

struct Solution {
    Vector dec;
    Vector obj;
    Vector con;
    Vector add;

    Solution() = default;
    Solution(const Vector& d, const Vector& o, const Vector& c)
        : dec(d), obj(o), con(c) {}

    bool isFeasible() const {
        return (con.array() <= 0.0).all();
    }
};

using Population = std::vector<Solution>;

inline Matrix getDecs(const Population& pop) {
    if (pop.empty()) return Matrix(0, 0);
    int N = pop.size(), D = pop[0].dec.size();
    Matrix M(N, D);
    for (int i = 0; i < N; i++) M.row(i) = pop[i].dec.transpose();
    return M;
}

inline Matrix getObjs(const Population& pop) {
    if (pop.empty()) return Matrix(0, 0);
    int N = pop.size(), M_ = pop[0].obj.size();
    Matrix R(N, M_);
    for (int i = 0; i < N; i++) R.row(i) = pop[i].obj.transpose();
    return R;
}

inline Matrix getCons(const Population& pop) {
    if (pop.empty()) return Matrix(0, 0);
    int N = pop.size(), C = pop[0].con.size();
    Matrix R(N, C);
    for (int i = 0; i < N; i++) R.row(i) = pop[i].con.transpose();
    return R;
}

inline Population makePopulation(const Matrix& decs, const Matrix& objs, const Matrix& cons) {
    int N = decs.rows();
    Population pop(N);
    for (int i = 0; i < N; i++) {
        pop[i].dec = decs.row(i).transpose();
        pop[i].obj = objs.row(i).transpose();
        pop[i].con = cons.row(i).transpose();
    }
    return pop;
}

inline Population concat(const Population& a, const Population& b) {
    Population result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

inline Population subset(const Population& pop, const std::vector<int>& idx) {
    Population result;
    result.reserve(idx.size());
    for (int i : idx) result.push_back(pop[i]);
    return result;
}

inline Population subsetMask(const Population& pop, const std::vector<bool>& mask) {
    Population result;
    for (int i = 0; i < (int)pop.size(); i++)
        if (mask[i]) result.push_back(pop[i]);
    return result;
}

// Pairwise squared Euclidean distance matrix (rows of A vs rows of B)
inline Matrix pdist2(const Matrix& A, const Matrix& B) {
    int NA = A.rows(), NB = B.rows();
    Matrix D(NA, NB);
    for (int i = 0; i < NA; i++)
        for (int j = 0; j < NB; j++)
            D(i,j) = (A.row(i) - B.row(j)).norm();
    return D;
}

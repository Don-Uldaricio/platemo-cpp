#pragma once
#include "../core/Solution.hpp"
#include "../utils/NDSort.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// ---- Exact HV via recursive slicing (for M < 4) ----
namespace hv_detail {

using Slice_t = std::vector<std::pair<double, std::vector<std::vector<double>>>>;

inline std::vector<double> head(const std::vector<std::vector<double>>& pl) {
    if (pl.empty()) return {};
    return pl[0];
}

inline std::vector<std::vector<double>> tail(const std::vector<std::vector<double>>& pl) {
    if (pl.size() < 2) return {};
    return std::vector<std::vector<double>>(pl.begin()+1, pl.end());
}

inline std::vector<std::vector<double>> insert_point(
    const std::vector<double>& p, int k, const std::vector<std::vector<double>>& pl)
{
    int m = p.size();
    std::vector<std::vector<double>> ql;
    auto pl_ = pl;
    auto hp = head(pl_);
    while (!pl_.empty() && !hp.empty() && hp[k] < p[k]) {
        ql.push_back(hp);
        pl_ = tail(pl_);
        hp = head(pl_);
    }
    ql.push_back(p);
    while (!pl_.empty()) {
        auto q = head(pl_);
        bool flag1 = false, flag2 = false;
        for (int i = k; i < m; i++) {
            if (p[i] < q[i]) flag1 = true;
            else if (p[i] > q[i]) flag2 = true;
        }
        if (!(flag1 && !flag2)) ql.push_back(q);
        pl_ = tail(pl_);
    }
    return ql;
}

inline Slice_t slice(const std::vector<std::vector<double>>& pl, int k, const std::vector<double>& ref) {
    auto p  = head(pl);
    auto pl_ = tail(pl);
    std::vector<std::vector<double>> ql;
    Slice_t S;

    while (!pl_.empty()) {
        ql = insert_point(p, k+1, ql);
        auto p_ = head(pl_);
        double width = std::abs(p[k] - p_[k]);
        S.push_back({width, ql});
        p   = p_;
        pl_ = tail(pl_);
    }
    ql = insert_point(p, k+1, ql);
    double width = std::abs(p[k] - ref[k]);
    S.push_back({width, ql});
    return S;
}

// Add entry to S (merge if same set of points)
inline void addToS(std::pair<double, std::vector<std::vector<double>>> entry, Slice_t& S) {
    for (auto& s : S) {
        if (s.second == entry.second) {
            s.first += entry.first;
            return;
        }
    }
    S.push_back(entry);
}

inline double computeHV(std::vector<std::vector<double>> PopObj_,
                        const std::vector<double>& RefPoint) {
    int M = RefPoint.size();
    // Sort by first objective
    std::sort(PopObj_.begin(), PopObj_.end());

    Slice_t S = {{1.0, PopObj_}};

    for (int k = 0; k < M-1; k++) {
        Slice_t Snew;
        for (auto& [coeff, pts] : S) {
            auto Stemp = slice(pts, k, RefPoint);
            for (auto& [w, ql] : Stemp)
                addToS({coeff * w, ql}, Snew);
        }
        S = Snew;
    }

    double score = 0.0;
    for (auto& [coeff, pts] : S) {
        auto p = head(pts);
        if (!p.empty())
            score += coeff * std::abs(p[M-1] - RefPoint[M-1]);
    }
    return score;
}

} // namespace hv_detail

// Hypervolume indicator
// Returns 0 if no feasible solutions, nan if dimension mismatch
inline double HV(const Population& pop, const Matrix& optimum) {
    Population best = getBest(pop);
    if (best.empty()) return 0.0;

    Matrix PopObj = getObjs(best);
    int N = PopObj.rows(), M = PopObj.cols();

    if (M != optimum.cols()) return std::numeric_limits<double>::quiet_NaN();

    // Normalize objectives to [0, 1.1] using fmin and fmax from optimum
    Eigen::RowVectorXd fmin = PopObj.colwise().minCoeff().cwiseMin(Eigen::RowVectorXd::Zero(M));
    Eigen::RowVectorXd fmax = optimum.colwise().maxCoeff();
    Eigen::RowVectorXd scale = (fmax - fmin) * 1.1;

    Matrix normObj(N, M);
    for (int i = 0; i < N; i++)
        normObj.row(i) = (PopObj.row(i) - fmin).array() / scale.array();

    // Remove solutions above reference point
    std::vector<int> valid;
    for (int i = 0; i < N; i++)
        if ((normObj.row(i).array() <= 1.0).all()) valid.push_back(i);

    if (valid.empty()) return 0.0;

    std::vector<double> RefPoint(M, 1.0);

    if (M < 4) {
        // Exact HV via recursive slicing
        std::vector<std::vector<double>> pts;
        for (int i : valid) {
            std::vector<double> row(M);
            for (int j = 0; j < M; j++) row[j] = normObj(i,j);
            pts.push_back(row);
        }
        return hv_detail::computeHV(pts, RefPoint);
    } else {
        // Monte Carlo estimation
        int SampleNum = 1000000;
        Eigen::RowVectorXd minVal = normObj.rowwise().minCoeff().transpose();
        // Actually: min over valid solutions per objective
        Eigen::RowVectorXd MinValue(M), MaxValue(M);
        for (int j = 0; j < M; j++) {
            MinValue(j) = normObj.col(j).minCoeff();
            MaxValue(j) = 1.0;
        }

        // Generate samples in [MinValue, MaxValue]
        int dominated = 0;
        for (int s = 0; s < SampleNum; s++) {
            std::vector<double> sample(M);
            for (int j = 0; j < M; j++)
                sample[j] = rng::uniform(MinValue(j), MaxValue(j));

            bool dominated_by_any = false;
            for (int i : valid) {
                bool dom = true;
                for (int j = 0; j < M; j++) {
                    if (normObj(i,j) > sample[j]) { dom = false; break; }
                }
                if (dom) { dominated_by_any = true; break; }
            }
            if (dominated_by_any) dominated++;
        }

        double vol = 1.0;
        for (int j = 0; j < M; j++) vol *= (MaxValue(j) - MinValue(j));
        return vol * (double)dominated / SampleNum;
    }
}

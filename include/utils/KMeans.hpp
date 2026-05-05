#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include "Random.hpp"

// Simple Lloyd's k-means for k=2, 1D data.
// Returns cluster labels 0 or 1 for each data point.
inline std::vector<int> kmeans2(const std::vector<double>& data, int maxIter = 100) {
    int N = data.size();
    if (N == 0) return {};
    if (N == 1) return {0};

    // Init: pick two random distinct points as centroids
    double c0 = data[0], c1 = data[N > 1 ? N-1 : 0];
    if (c0 == c1) {
        // All points equal: assign alternately
        std::vector<int> labels(N, 0);
        for (int i = 1; i < N; i += 2) labels[i] = 1;
        return labels;
    }

    std::vector<int> labels(N, 0);
    for (int iter = 0; iter < maxIter; iter++) {
        // Assignment step
        std::vector<int> newLabels(N);
        for (int i = 0; i < N; i++) {
            double d0 = std::abs(data[i] - c0);
            double d1 = std::abs(data[i] - c1);
            newLabels[i] = (d1 < d0) ? 1 : 0;
        }

        // Update step
        double sum0 = 0, sum1 = 0;
        int cnt0 = 0, cnt1 = 0;
        for (int i = 0; i < N; i++) {
            if (newLabels[i] == 0) { sum0 += data[i]; cnt0++; }
            else                   { sum1 += data[i]; cnt1++; }
        }
        if (cnt0 == 0 || cnt1 == 0) {
            // Degenerate: split at median
            std::vector<double> sorted = data;
            std::sort(sorted.begin(), sorted.end());
            double median = sorted[N/2];
            for (int i = 0; i < N; i++) newLabels[i] = (data[i] > median) ? 1 : 0;
            return newLabels;
        }

        double nc0 = sum0 / cnt0;
        double nc1 = sum1 / cnt1;

        if (newLabels == labels && std::abs(nc0-c0) < 1e-12 && std::abs(nc1-c1) < 1e-12)
            break;

        labels = newLabels;
        c0 = nc0;
        c1 = nc1;
    }

    return labels;
}

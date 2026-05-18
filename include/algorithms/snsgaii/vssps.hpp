#pragma once
#include "../../core/Problem.hpp"
#include "../../utils/Random.hpp"

// Variable Stripe Sparse Population Sampling (VSSPS): inicialización especializada para problemas esparcidos.
// Genera N individuos con diferentes niveles de esparcidad usando un patrón de "franjas":
// cada individuo tiene una franja contigua de posiciones activas (no-cero), y el resto son ceros.
// El ancho de la franja varía desde alta densidad hasta baja densidad a través de la población,
// garantizando diversidad en el eje de complejidad desde la primera generación.
// prob: problema que define N, D y Evaluation().
// sLower: esparcidad mínima (fracción de ceros del individuo más denso), default 0.75.
// sUpper: esparcidad máxima (fracción de ceros del individuo más esparcido), default 1.0.
// Retorna: Population de N individuos con patrones de esparcidad estructurados y evaluados.
inline Population vssps(Problem& prob, double sLower = 0.75, double sUpper = 1.0) {
    int N = prob.N;
    int D = prob.D;

    // Generate base population (uniform random)
    Population basePop = prob.Initialization(N);
    Matrix baseDecMat = getDecs(basePop);

    // Density vector: from high density to low density
    std::vector<double> densityVector(N);
    for (int i = 0; i < N; i++)
        densityVector[i] = 1.0 - (sLower + (sUpper - sLower) * i / (N - 1 > 0 ? N-1 : 1));

    // Width of non-zero stripe per individual
    std::vector<int> widthVector(N);
    for (int i = 0; i < N; i++)
        widthVector[i] = (int)std::round(densityVector[i] * D);

    // Clamp widths
    int maxWidth = (int)std::floor((1.0 - sLower) * D);
    for (int i = 0; i < N; i++)
        if (widthVector[i] > maxWidth) widthVector[i] = maxWidth;

    // Cumulative widths for cycle computation
    std::vector<int> cumulativeWidths(N);
    cumulativeWidths[0] = widthVector[0];
    for (int i = 1; i < N; i++)
        cumulativeWidths[i] = cumulativeWidths[i-1] + widthVector[i];

    // Build density mask
    MatrixB mask = MatrixB::Zero(N, D);

    // Count individuals with zero width
    int zeroWidthCount = 0;
    for (int w : widthVector) if (w == 0) zeroWidthCount++;

    int processedIndvs = (zeroWidthCount == N) ? N : 0;

    // Cycle-based stripe assignment
    std::vector<int> currentCumWidths = cumulativeWidths;
    int cycle_count = 0;
    int currentIndv = 0;

    std::vector<std::vector<int>> cycles;

    while (processedIndvs < N) {
        cycle_count++;
        // Find individuals that fit in this cycle
        std::vector<int> spotsThatFit;
        int largestFit = 0;
        for (int i = 0; i < N; i++) {
            if (currentCumWidths[i] <= D && currentCumWidths[i] != 0) {
                spotsThatFit.push_back(i);
                largestFit = std::max(largestFit, currentCumWidths[i]);
            }
        }

        if (spotsThatFit.empty()) break;

        cycles.push_back(spotsThatFit);
        processedIndvs += spotsThatFit.size();

        // Subtract largestFit from all cumulative widths
        for (int i = 0; i < N; i++) {
            currentCumWidths[i] -= largestFit;
            if (currentCumWidths[i] < 0) currentCumWidths[i] = 0;
        }
    }

    // Assign stripes
    currentIndv = 0;
    for (int c = 0; c < (int)cycles.size(); c++) {
        const auto& cycle = cycles[c];

        std::vector<int> widths;
        for (int idx : cycle) widths.push_back(widthVector[idx]);

        int totalWidth = 0;
        for (int w : widths) totalWidth += w;
        int gapToFill = D - totalWidth;
        int gapSize = (!widths.empty()) ? (int)std::ceil((double)gapToFill / widths.size()) : 0;

        int position = 0;
        for (int i = 0; i < (int)widths.size() && currentIndv < N; i++) {
            int width = widths[i];
            int gapWidth = 0;
            if (gapToFill > 0) {
                gapWidth = std::min(gapSize, gapToFill);
                gapToFill -= gapWidth;
            }

            int startPoint = position;
            int endPoint;
            if (c == (int)cycles.size() - 1) {
                endPoint = position + width - 1;
            } else {
                endPoint = position + width - 1 + gapWidth;
            }
            endPoint = std::min(endPoint, D-1);

            for (int j = startPoint; j <= endPoint && j < D; j++)
                mask(currentIndv, j) = true;

            position += width + gapWidth;
            currentIndv++;
        }
    }

    // Apply mask to base decisions
    Matrix sparseDec = baseDecMat;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            if (!mask(i,j)) sparseDec(i,j) = 0.0;

    // Evaluate with sparse decisions
    Matrix calDec  = prob.CalDec(sparseDec);
    Matrix calObj  = prob.CalObj(sparseDec);
    Matrix calCon  = prob.CalCon(sparseDec);
    return makePopulation(calDec, calObj, calCon);
}

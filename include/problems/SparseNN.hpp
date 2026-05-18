#pragma once
#include "../core/Problem.hpp"
#include "../utils/Random.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cmath>
#include <set>
#include <algorithm>

// Neural network training problem (Sparse_NN from PlatEMO)
// Two objectives: (1) fraction of non-zero weights, (2) training error
class SparseNN : public Problem {
public:
    int dataNo  = 1;   // Dataset 1-4
    int nHidden = 20;  // Hidden layer size

    Matrix TrainIn, TrainOut, TestIn, TestOut;
    Matrix TrainLabel, TestLabel;
    int nFeatures = 0, nOutputs = 0, nSamples = 0;
    std::string dataPath;

    SparseNN(int dataNo_ = 1, int nHidden_ = 20,
             const std::string& dataPath_ = "data")
        : dataNo(dataNo_), nHidden(nHidden_), dataPath(dataPath_) {}

    // Carga los datos y define los parámetros del problema.
    // D = (nFeatures+1)*nHidden + (nHidden+1)*nOutputs (total de pesos incluyendo biases).
    // Todos los pesos en [-1, 1].
    void Setting() override {
        loadData();
        M = 2;
        D = (nFeatures + 1) * nHidden + (nHidden + 1) * nOutputs;
        lower = Vector::Constant(D, -1.0);
        upper = Vector::Constant(D,  1.0);
        encoding.assign(D, 1);  // All real
    }

    // Inicialización esparcida: cada peso es uniform(-1,1) * Bernoulli(0.5),
    // produciendo ~50% de pesos en cero desde el inicio.
    // n: número de soluciones a generar (-1 usa prob.N).
    Population Initialization(int n = -1) override {
        if (n < 0) n = N;
        Matrix PopDec(n, D);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < D; j++) {
                double val = rng::uniform(-1.0, 1.0);
                int mask_bit = rng::randint(0, 1);
                PopDec(i, j) = val * mask_bit;
            }
        return Evaluation(PopDec);
    }

    // Fine-tuning: aplica 1 época de backpropagation a cada solución antes de evaluarla.
    // Esto mejora el error de clasificación sin cambiar el patrón de esparcidad (los ceros se mantienen).
    // dec: Matrix n x D con los pesos propuestos por el algoritmo evolutivo.
    // Retorna: Matrix n x D con los pesos ajustados por gradiente.
    Matrix CalDec(const Matrix& dec) override {
        int n = dec.rows();
        Matrix result = dec;
        for (int i = 0; i < n; i++) {
            auto [W1, W2] = decodeWeights(result.row(i));
            trainStep(W1, W2, 1);
            result.row(i) = encodeWeights(W1, W2);
        }
        return result;
    }

    // Calcula los dos objetivos para cada solución:
    //   f1 (obj 0) = fracción de pesos no-cero (complejidad de la red, en [0,1]).
    //   f2 (obj 1) = tasa de error de clasificación sobre el training set (en [0,1]).
    // dec: Matrix n x D con los pesos ya fine-tuneados.
    // Retorna: Matrix n x 2 con [complejidad, error] por fila.
    Matrix CalObj(const Matrix& dec) override {
        int n = dec.rows();
        Matrix obj(n, 2);
        for (int i = 0; i < n; i++) {
            auto [W1, W2] = decodeWeights(dec.row(i));
            auto [Z, Y] = predict(TrainIn, W1, W2);

            // Objective 1: fraction of non-zero weights
            int nzCount = 0;
            for (int j = 0; j < D; j++)
                if (dec(i,j) != 0.0) nzCount++;
            obj(i, 0) = (double)nzCount / D;

            // Objective 2: training error (misclassification rate)
            Matrix Zpred = Z;
            if (nOutputs == 1) {
                // Binary: round to 0/1
                double err = 0;
                for (int s = 0; s < nSamples; s++) {
                    int pred = (Z(s,0) >= 0.5) ? 1 : 0;
                    int label = (int)TrainLabel(s,0);
                    if (pred != label) err++;
                }
                obj(i, 1) = err / nSamples;
            } else {
                double err = 0;
                for (int s = 0; s < nSamples; s++) {
                    int predClass = 0;
                    for (int c = 1; c < nOutputs; c++)
                        if (Z(s,c) > Z(s,predClass)) predClass = c;
                    if (predClass != (int)TrainLabel(s,0)-1) err++;
                }
                obj(i, 1) = err / nSamples;
            }
        }
        return obj;
    }

    Matrix GetOptimum(int /*n*/) override {
        return Matrix::Ones(1, 2);
    }

private:
    // Lee el CSV del dataset, normaliza features con z-score, construye etiquetas,
    // y divide en 80% train / 20% test.
    // Formato CSV: cada fila es una muestra, última columna es la etiqueta de clase.
    // Para 2 clases: nOutputs=1, salida binaria {0,1}.
    // Para >2 clases: nOutputs=nClases, salida one-hot.
    void loadData() {
        std::string fname = dataPath + "/Dataset_NN_" + std::to_string(dataNo) + ".csv";
        std::ifstream file(fname);
        if (!file.is_open())
            throw std::runtime_error("Cannot open dataset: " + fname);

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

        int totalSamples = rows.size();
        int totalCols    = rows[0].size();
        nFeatures = totalCols - 1;

        // Build data matrix
        Matrix Data(totalSamples, totalCols);
        for (int i = 0; i < totalSamples; i++)
            for (int j = 0; j < totalCols; j++)
                Data(i,j) = rows[i][j];

        // Normalize features
        Matrix Input = Data.leftCols(nFeatures);
        Eigen::RowVectorXd Mean = Input.colwise().mean();
        Eigen::RowVectorXd Std(nFeatures);
        for (int j = 0; j < nFeatures; j++) {
            double var = 0;
            for (int i = 0; i < totalSamples; i++)
                var += std::pow(Input(i,j) - Mean(j), 2);
            Std(j) = std::sqrt(var / totalSamples);
            if (Std(j) < 1e-10) Std(j) = 1.0;
        }
        for (int i = 0; i < totalSamples; i++)
            Input.row(i) = (Input.row(i) - Mean).array() / Std.array();

        // Labels (last column)
        Eigen::VectorXd labelCol = Data.col(nFeatures);
        std::set<double> catSet(labelCol.data(), labelCol.data() + totalSamples);
        std::vector<double> cats(catSet.begin(), catSet.end());
        int nCat = cats.size();

        Matrix Output;
        if (nCat <= 2) {
            nOutputs = 1;
            Output = Matrix(totalSamples, 1);
            for (int i = 0; i < totalSamples; i++)
                Output(i,0) = (labelCol(i) == cats[0]) ? 1.0 : 0.0;
        } else {
            nOutputs = nCat;
            Output = Matrix::Zero(totalSamples, nCat);
            for (int i = 0; i < totalSamples; i++) {
                int c = std::find(cats.begin(), cats.end(), labelCol(i)) - cats.begin();
                Output(i, c) = 1.0;
            }
        }

        // Split 80/20
        int trainSize = (int)std::ceil(totalSamples * 0.8);
        TrainIn    = Input.topRows(trainSize);
        TrainOut   = Output.topRows(trainSize);
        TestIn     = Input.bottomRows(totalSamples - trainSize);
        TestOut    = Output.bottomRows(totalSamples - trainSize);
        nSamples   = trainSize;

        if (nCat <= 2) {
            TrainLabel = TrainOut;
            TestLabel  = TestOut;
        } else {
            TrainLabel = Matrix(trainSize, 1);
            TestLabel  = Matrix(totalSamples - trainSize, 1);
            for (int i = 0; i < trainSize; i++) {
                int best = 0;
                for (int c = 1; c < nOutputs; c++)
                    if (TrainOut(i,c) > TrainOut(i,best)) best = c;
                TrainLabel(i,0) = best + 1;
            }
            for (int i = 0; i < totalSamples - trainSize; i++) {
                int best = 0;
                for (int c = 1; c < nOutputs; c++)
                    if (TestOut(i,c) > TestOut(i,best)) best = c;
                TestLabel(i,0) = best + 1;
            }
        }
    }

    // Convierte un vector plano de D pesos en las matrices de la red.
    // row: RowVector de tamaño D con todos los pesos en el orden del encoding.
    // Retorna: par (W1, W2) donde W1 es (nFeatures+1) x nHidden y W2 es (nHidden+1) x nOutputs.
    // La fila 0 de W1 y W2 corresponde al bias de cada capa.
    std::pair<Matrix, Matrix> decodeWeights(const Eigen::RowVectorXd& row) const {
        // total size = (nFeatures+1)*nHidden + (nHidden+1)*nOutputs
        Matrix W1 = Matrix(nFeatures+1, nHidden);
        Matrix W2 = Matrix(nHidden+1, nOutputs);
        int idx = 0;
        for (int j = 0; j < nHidden; j++)
            for (int i = 0; i <= nFeatures; i++)
                W1(i,j) = row(idx++);
        for (int j = 0; j < nOutputs; j++)
            for (int i = 0; i <= nHidden; i++)
                W2(i,j) = row(idx++);
        return {W1, W2};
    }

    // Operación inversa a decodeWeights: aplana W1 y W2 en un vector plano de tamaño D.
    // W1: Matrix (nFeatures+1) x nHidden.
    // W2: Matrix (nHidden+1) x nOutputs.
    // Retorna: RowVector de tamaño D con los pesos en el mismo orden que decodeWeights.
    Eigen::RowVectorXd encodeWeights(const Matrix& W1, const Matrix& W2) const {
        Eigen::RowVectorXd row(D);
        int idx = 0;
        for (int j = 0; j < nHidden; j++)
            for (int i = 0; i <= nFeatures; i++)
                row(idx++) = W1(i,j);
        for (int j = 0; j < nOutputs; j++)
            for (int i = 0; i <= nHidden; i++)
                row(idx++) = W2(i,j);
        return row;
    }

    // Forward pass de la red neuronal.
    // Capa oculta: activación tanh.  Capa de salida: activación sigmoid.
    // Agrega automáticamente la columna de bias (valor 1) a la entrada y a la capa oculta.
    // X: Matrix nSamples x nFeatures con las entradas (sin bias).
    // W1: Matrix (nFeatures+1) x nHidden con los pesos de la primera capa (fila 0 = bias).
    // W2: Matrix (nHidden+1) x nOutputs con los pesos de la segunda capa (fila 0 = bias).
    // Retorna: par (Z, Y) donde Z es la salida (nSamples x nOutputs) e Y es la capa oculta (nSamples x nHidden).
    std::pair<Matrix, Matrix> predict(const Matrix& X, const Matrix& W1, const Matrix& W2) const {
        int n = X.rows();
        // Add bias column to X
        Matrix Xb(n, nFeatures+1);
        Xb.col(0).setOnes();
        Xb.rightCols(nFeatures) = X;

        // Hidden: tanh activation Y = tanh(Xb * W1)
        Matrix netH = Xb * W1;
        Matrix Y = netH.unaryExpr([](double v){ return std::tanh(v); });

        // Add bias to Y
        Matrix Yb(n, nHidden+1);
        Yb.col(0).setOnes();
        Yb.rightCols(nHidden) = Y;

        // Output: sigmoid
        Matrix netO = Yb * W2;
        Matrix Z = netO.unaryExpr([](double v){ return 1.0 / (1.0 + std::exp(-v)); });

        return {Z, Y};
    }

    // Aplica nEpoch épocas de gradient descent con backpropagation sobre el training set completo.
    // Usa batch gradient descent (un único update por época).
    // Regla de actualización: W -= dL/dW / n (sin momentum ni learning rate especial, lr=1/n).
    // W1: pesos de la primera capa (modificados in-place).
    // W2: pesos de la segunda capa (modificados in-place).
    // nEpoch: número de épocas de entrenamiento.
    void trainStep(Matrix& W1, Matrix& W2, int nEpoch) const {
        int n = TrainIn.rows();
        Matrix Xb(n, nFeatures+1);
        Xb.col(0).setOnes();
        Xb.rightCols(nFeatures) = TrainIn;

        for (int epoch = 0; epoch < nEpoch; epoch++) {
            auto [Z, Y] = predict(TrainIn, W1, W2);
            // P = (Z - T) * Z * (1-Z): output delta
            Matrix P = (Z - TrainOut).array() * Z.array() * (1.0 - Z.array());
            // Q = P * W2[1:end,:]' * (1 - Y^2): hidden delta (skip bias row)
            Matrix W2noB = W2.bottomRows(nHidden);  // nHidden x nOutputs
            Matrix Q = (P * W2noB.transpose()).array() * (1.0 - Y.array().square());

            // Gradients
            Matrix Yb(n, nHidden+1);
            Yb.col(0).setOnes();
            Yb.rightCols(nHidden) = Y;

            Matrix dW2 = Yb.transpose() * P;
            Matrix dW1 = Xb.transpose() * Q;

            W1 -= dW1 / n;
            W2 -= dW2 / n;
        }
    }
};

#pragma once
#include <vector>
#include <Eigen/Dense>
#include <numeric>
#include <algorithm>
#include <limits>

// Alias de tipos Eigen usados en todo el proyecto
using Matrix   = Eigen::MatrixXd;                                      // Matriz de doubles (NxD genérica)
using MatrixB  = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;  // Matriz booleana (para máscaras de esparcidad)
using Vector   = Eigen::VectorXd;                                      // Vector columna de doubles
using RowVec   = Eigen::RowVectorXd;                                   // Vector fila de doubles
using VectorI  = Eigen::VectorXi;                                      // Vector de enteros

// Representa una solución individual dentro de la población.
struct Solution {
    Vector dec;  // Variables de decisión (los pesos de la red), tamaño D
    Vector obj;  // Valores de los objetivos evaluados, tamaño M (f1=complejidad, f2=error)
    Vector con;  // Valores de restricciones (≤ 0 = factible), tamaño C
    Vector add;  // Campo auxiliar libre para datos extra (no siempre usado)

    Solution() = default;
    Solution(const Vector& d, const Vector& o, const Vector& c)
        : dec(d), obj(o), con(c) {}

    // Retorna true si todas las restricciones son <= 0 (solución factible).
    bool isFeasible() const {
        return (con.array() <= 0.0).all();
    }
};

// Una población es simplemente un vector de soluciones.
using Population = std::vector<Solution>;

// Extrae las variables de decisión de toda la población como una matriz N x D.
// pop: vector de N soluciones, cada una con dec de tamaño D.
// Retorna: Matrix N x D donde la fila i corresponde a pop[i].dec.
inline Matrix getDecs(const Population& pop) {
    if (pop.empty()) return Matrix(0, 0);
    int N = pop.size(), D = pop[0].dec.size();
    Matrix M(N, D);
    for (int i = 0; i < N; i++) M.row(i) = pop[i].dec.transpose();
    return M;
}

// Extrae los valores de objetivos de toda la población como una matriz N x M.
// pop: vector de N soluciones, cada una con obj de tamaño M.
// Retorna: Matrix N x M donde la fila i corresponde a pop[i].obj.
inline Matrix getObjs(const Population& pop) {
    if (pop.empty()) return Matrix(0, 0);
    int N = pop.size(), M_ = pop[0].obj.size();
    Matrix R(N, M_);
    for (int i = 0; i < N; i++) R.row(i) = pop[i].obj.transpose();
    return R;
}

// Extrae los valores de restricciones de toda la población como una matriz N x C.
// pop: vector de N soluciones, cada una con con de tamaño C.
// Retorna: Matrix N x C donde la fila i corresponde a pop[i].con.
inline Matrix getCons(const Population& pop) {
    if (pop.empty()) return Matrix(0, 0);
    int N = pop.size(), C = pop[0].con.size();
    Matrix R(N, C);
    for (int i = 0; i < N; i++) R.row(i) = pop[i].con.transpose();
    return R;
}

// Construye una Population a partir de matrices separadas de decisiones, objetivos y restricciones.
// decs: Matrix N x D de decisiones.
// objs: Matrix N x M de objetivos.
// cons: Matrix N x C de restricciones.
// Retorna: vector de N soluciones inicializadas.
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

// Concatena dos poblaciones en una sola.
// Retorna: vector con todos los elementos de `a` seguidos de los de `b`.
inline Population concat(const Population& a, const Population& b) {
    Population result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

// Selecciona un subconjunto de la población por índices.
// pop: población de origen.
// idx: vector con los índices a extraer (pueden repetirse).
// Retorna: Population con las soluciones en el orden dado por idx.
inline Population subset(const Population& pop, const std::vector<int>& idx) {
    Population result;
    result.reserve(idx.size());
    for (int i : idx) result.push_back(pop[i]);
    return result;
}

// Selecciona un subconjunto de la población usando una máscara booleana.
// pop: población de origen.
// mask: vector de bool de tamaño pop.size(); true = incluir esa solución.
// Retorna: Population con las soluciones donde mask[i] == true.
inline Population subsetMask(const Population& pop, const std::vector<bool>& mask) {
    Population result;
    for (int i = 0; i < (int)pop.size(); i++)
        if (mask[i]) result.push_back(pop[i]);
    return result;
}

// Calcula la matriz de distancias Euclidianas entre todas las filas de A y todas las filas de B.
// A: Matrix NA x K.
// B: Matrix NB x K.
// Retorna: Matrix NA x NB donde D(i,j) = ||A[i] - B[j]||_2.
inline Matrix pdist2(const Matrix& A, const Matrix& B) {
    int NA = A.rows(), NB = B.rows();
    Matrix D(NA, NB);
    for (int i = 0; i < NA; i++)
        for (int j = 0; j < NB; j++)
            D(i,j) = (A.row(i) - B.row(j)).norm();
    return D;
}

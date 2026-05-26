#pragma once
#include "Solution.hpp"
#include <chrono>
#include <functional>
#include <limits>
#include <random>

// Clase base abstracta para todos los problemas de optimización.
// Cada problema concreto (SparseNN, SparseSNN) debe heredar de esta clase
// e implementar al menos Setting() y CalObj().
class Problem {
public:
    int    N         = 100;    // Tamaño de la población
    int    maxFE     = 10000;  // Máximo número de evaluaciones permitidas
    int    FE        = 0;      // Contador de evaluaciones usadas hasta ahora
    int    M         = 0;      // Número de objetivos (se fija en Setting())
    int    D         = 0;      // Dimensión del espacio de decisión (número de pesos de la red)
    double maxRuntime = std::numeric_limits<double>::infinity();  // Límite de tiempo (no usado actualmente)
    Vector lower;              // Bounds inferiores por variable, tamaño D
    Vector upper;              // Bounds superiores por variable, tamaño D
    std::vector<int> encoding; // Tipo de cada variable: 1=real, 2=entero, 3=label, 4=binario, 5=permutación
    Matrix optimum;            // Puntos de referencia del frente óptimo, usado para calcular HV e IGD
    std::chrono::steady_clock::time_point startTime;  // Tiempo de inicio, registrado en init()

    std::function<void(int, const Population&)> onLog;  // Callback opcional para loggear convergencia
    int logInterval = 0;   // Cada cuántos FE llamar a onLog (0 = desactivado)
    int nextLogFE   = 0;   // Próximo FE en que se disparará onLog

    virtual ~Problem() = default;

    // Configura el problema: debe asignar M, D, lower, upper y encoding.
    // Es llamado por init() antes de cualquier evaluación.
    virtual void Setting() = 0;

    // Genera n soluciones iniciales aleatorias (uniformes en [lower, upper]) y las evalúa.
    // n: número de soluciones a generar; si es -1 se usa N.
    // Retorna: Population de n soluciones con dec, obj y con calculados.
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

    // Pipeline completo de evaluación de una matriz de decisiones.
    // dec: Matrix n x D con las decisiones a evaluar (sin aplicar bounds aún).
    // Aplica CalDec (clamp/redondeo) → CalObj → CalCon y acumula FE.
    // Retorna: Population de n soluciones con todos los campos calculados.
    virtual Population Evaluation(const Matrix& dec) {
        Matrix PopDec = CalDec(dec);
        Matrix PopObj = CalObj(PopDec);
        Matrix PopCon = CalCon(PopDec);
        Population pop = makePopulation(PopDec, PopObj, PopCon);
        FE += pop.size();
        return pop;
    }

    // Proyecta las decisiones a los bounds y redondea variables discretas.
    // dec: Matrix n x D con decisiones posiblemente fuera de rango.
    // Retorna: Matrix n x D con decisiones válidas según encoding.
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

    // Calcula los objetivos para cada solución. Debe ser sobreescrita por subclases.
    // dec: Matrix n x D de decisiones ya proyectadas a bounds.
    // Retorna: Matrix n x M con los valores de cada objetivo por fila.
    virtual Matrix CalObj(const Matrix& dec) {
        return Matrix::Zero(dec.rows(), 1);
    }

    // Calcula las restricciones para cada solución. Valor > 0 indica violación.
    // dec: Matrix n x D de decisiones ya proyectadas a bounds.
    // Retorna: Matrix n x C (por defecto todos ceros = sin restricciones).
    virtual Matrix CalCon(const Matrix& dec) {
        return Matrix::Zero(dec.rows(), 1);
    }

    // Retorna los puntos de referencia del frente de Pareto óptimo o ideal.
    // n: número de puntos de referencia a generar.
    // Retorna: Matrix de referencia usada para calcular HV e IGD.
    virtual Matrix GetOptimum(int /*n*/) {
        if (M > 1) return Matrix::Ones(1, M);
        return Matrix::Zero(1, 1);
    }

    // Condición de parada: retorna true mientras queden evaluaciones disponibles.
    // Llamado al inicio de cada generación del algoritmo evolutivo.
    bool NotTerminated(const Population& /*pop*/) {
        return FE < maxFE;
    }

    // Registra el estado de la población inicial al finalizar la inicialización del algoritmo.
    // Usa prob.FE real para que el eje X del CSV refleje el costo total de inicialización
    // (p.ej. PriorAnalysis de MOEACKF consume 5D+N FEs antes de la primera generación).
    // Avanza nextLogFE al primer checkpoint posterior a FE para evitar filas fuera de orden.
    void LogInitial(const Population& pop) {
        if (logInterval <= 0 || !onLog) return;
        onLog(FE, pop);
        while (nextLogFE <= FE) nextLogFE += logInterval;
    }

    // Dispara el callback de logging si corresponde según el intervalo configurado.
    // Usa el FE del checkpoint (nextLogFE) como valor registrado, no el FE real actual,
    // para evitar filas duplicadas cuando varios checkpoints se saltan en una misma generación.
    void MaybeLog(const Population& pop) {
        if (logInterval <= 0 || !onLog) return;
        while (nextLogFE > 0 && FE >= nextLogFE && nextLogFE <= maxFE) {
            onLog(nextLogFE, pop);
            nextLogFE += logInterval;
        }
    }

    // Inicializa el problema: llama a Setting(), obtiene el óptimo de referencia y resetea FE.
    // Debe llamarse una vez antes de comenzar el algoritmo evolutivo.
    void init() {
        startTime = std::chrono::steady_clock::now();
        Setting();
        optimum = GetOptimum(10000);
        FE = 0;
        nextLogFE = logInterval;
    }

protected:
    // Generador de números aleatorios interno de la clase base (singleton estático).
    static std::mt19937& rng_engine() {
        static std::mt19937 e(std::random_device{}());
        return e;
    }
};

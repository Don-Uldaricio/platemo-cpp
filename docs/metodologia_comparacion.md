# Metodología de comparación estadística entre algoritmos

> Ámbito: comparación de **MOEACKF vs. SNSGAII** sobre un mismo dataset (tarea). Toda
> comparación se realiza **por separado para cada dataset** — nunca se agregan datos de
> distintos datasets en una misma prueba estadística. Las conclusiones agregadas entre
> datasets, si se presentan, son un resumen cualitativo posterior y no un test adicional.

## 1. Diseño experimental

Ambos algoritmos se ejecutan sobre el mismo problema bi-objetivo `SparseSNN`, que minimiza
simultáneamente:

- **f1 (dispersión / complejidad)**: fracción de pesos sinápticos activos (no nulos) en la red.
- **f2 (error)**: `1 − accuracy` en el conjunto de entrenamiento (`FITNESS_MODE=accuracy`).

El diseño experimental tiene tres propiedades que hay que explicitar en la tesis porque
sustentan la validez de la comparación:

1. **Presupuesto de evaluaciones igualado por dataset**: `MAXFE_<ALGO>_<DS>` en
   `run_experiments.sh` se define de forma que, para un mismo dataset, ambos algoritmos
   consumen el mismo número máximo de evaluaciones de fitness. Sin esto, una diferencia en
   HV podría deberse simplemente a que un algoritmo tuvo más presupuesto, no a que sea mejor.
2. **Hiperparámetros fijos por combinación**: los hiperparámetros de los operadores
   (`--disC`, `--disM`, `--proM`, etc.) provienen de una búsqueda Bayesiana
   (`bayesian_search.py` → `params_table.sh`) específica por combinación algoritmo×dataset, y
   se mantienen constantes entre las corridas repetidas de esa combinación.
3. **Semillas pareadas por índice de corrida**: en `run_one()` del script, `seed = BASE_SEED +
   run`, y esta fórmula **no depende del algoritmo**. Es decir, la corrida `run=i` de MOEACKF y
   la corrida `run=i` de SNSGAII parten de la misma semilla de aleatoriedad (mismo split de
   datos si el split depende del seed, misma inicialización de generador aleatorio subyacente).
   Esto convierte la comparación en un **diseño pareado**, y debe aprovecharse eligiendo
   pruebas estadísticas pareadas (ver §4), que tienen mayor potencia que las no pareadas al
   controlar la varianza atribuible a la semilla.

**Número de corridas independientes**: el script trae por defecto `RUNS=6`, pero los
resultados ya archivados (`results/ds1..ds4`) contienen ~30–31 corridas por combinación. Para
potencia estadística adecuada en pruebas no paramétricas se recomienda **N ≥ 30 corridas
independientes** por combinación algoritmo×dataset; con `RUNS=6` los tests igual pueden
ejecutarse pero el resultado será menos confiable y más sensible a outliers. Confirmar qué
conjunto de corridas se va a usar antes de fijar la tabla final de resultados.

## 2. Indicadores a comparar por dataset

Para cada dataset, cada indicador se calcula por corrida y algoritmo, generando dos muestras
(una por algoritmo) de tamaño N (número de corridas) que se comparan estadísticamente.

| Indicador | Qué mide | Fuente en el repo |
|---|---|---|
| **Hipervolumen (HV)** | Calidad global del frente de Pareto (convergencia + diversidad) | Columna `hv` en `results.csv` |
| **Accuracy de test** | Generalización real del clasificador (no solo la métrica optimizada en entrenamiento) | Se reconstruye desde `fronts/*_spikes.csv` (requiere `SAVE_SPIKES=true`) |
| **Velocidad de convergencia** | Qué tan rápido se acerca cada algoritmo a su HV final | `convergence.csv` (columna `generation`, `hv`) |

**Nota importante — el HV ya es comparable sin trabajo adicional.** En
`include/metrics/HV.hpp`, el hipervolumen se calcula normalizando `f1,f2` contra un punto de
referencia **fijo e independiente del algoritmo, la corrida o el dataset**
(`SparseSNN::GetOptimum()` retorna `(1,1)`, es decir, el ideal `(0,0)` sin normalizar por
dataset). Esto es una fortaleza metodológica a mencionar explícitamente: el HV no depende de
un frente de referencia estimado empíricamente por dataset, por lo que las comparaciones entre
algoritmos (e incluso, con cautela, entre datasets) son directas.

### 2.1 Por qué revisar train vs. test

El hipervolumen y `f2` en `*_front.csv` están calculados sobre el **error de entrenamiento**.
Un algoritmo puede lograr mejor HV simplemente por sobreajustar más al training set. Por eso se
recomienda reportar, además del HV, el accuracy de **test** reconstruido desde `*_spikes.csv`
(columnas `sample_id, spike_count, true_label` por solución del frente), y comparar la brecha
train–test entre algoritmos como indicador de robustez/generalización.

### 2.2 Sobre el tiempo de ejecución (`time_s`)

`results.csv` incluye una columna `time_s`, pero **se descarta deliberadamente como indicador
de comparación entre algoritmos**: las corridas fueron ejecutadas en distintas tandas con
distinta cantidad de procesos en paralelo y/o hilos OMP (`JOBS`, `OMP_THREADS` en
`run_experiments.sh`), por lo que el tiempo de reloj no refleja el costo computacional real de
cada algoritmo de forma comparable. Si se quisiera reportar costo computacional habría que
re-ejecutar ambos algoritmos bajo condiciones de hardware/paralelismo idénticas y controladas
(mismo `JOBS`, mismo `OMP_THREADS`, sin otros procesos compitiendo por CPU), lo cual queda fuera
del alcance de los resultados ya recolectados.

## 3. Verificación de supuestos

Antes de elegir la prueba estadística, para cada indicador y dataset:

1. **Normalidad**: Shapiro-Wilk sobre la distribución de cada algoritmo por separado (N
   corridas). Con N moderado (6–30) el test tiene poca potencia para detectar desviaciones
   pequeñas de normalidad, así que no debe ser el único criterio.
2. **Aun si no se rechaza normalidad**, se recomienda usar de todas formas pruebas **no
   paramétricas** como criterio principal, porque:
   - El hipervolumen (HV) suele tener distribución sesgada o acotada, no gaussiana.
   - Es el estándar de facto en la literatura de computación evolutiva para comparar
     algoritmos estocásticos (Derrac, García, Molina & Herrera, 2011 — *"A practical tutorial
     on the use of nonparametric statistical tests..."*).
   - Son robustas a outliers, frecuentes en optimización estocástica.

## 4. Pruebas estadísticas

### 4.1 Comparación principal (2 algoritmos, por dataset)

Dado el diseño **pareado por semilla** (§1), la prueba recomendada es:

- **Wilcoxon signed-rank test** (pareado), aplicado a los pares `(HV_MOEACKF[i], HV_SNSGAII[i])`
  para `i = 0..N-1` (mismo índice de corrida = mismo seed). Repetir para cada indicador (HV y
  accuracy de test) **por separado, por dataset**.
- **Alternativa** (Mann-Whitney U, para muestras independientes): usar solo si el pareo se
  rompe — por ejemplo, si alguna corrida de un algoritmo falló/no convergió y no hay el mismo
  número de corridas válidas para ambos algoritmos en algún índice.

### 4.2 Tamaño del efecto

El p-valor por sí solo no basta: con N alto, diferencias mínimas pueden ser "significativas"
sin ser relevantes en la práctica. Reportar siempre junto al p-valor:

- **Â₁₂ de Vargha-Delaney**: probabilidad de que una corrida aleatoria de un algoritmo supere a
  una corrida aleatoria del otro. Interpretación estándar: ~0.5 = sin diferencia práctica, ≥0.71
  o ≤0.29 = efecto grande.
- Alternativa: `r = Z / √N` (tamaño de efecto derivado del estadístico Z del Wilcoxon).

### 4.3 Corrección por comparaciones múltiples

Dentro de un mismo dataset se testean dos indicadores (HV y accuracy de test) — aplicar
**corrección de Holm-Bonferroni** sobre ese conjunto de p-valores para controlar la tasa de
error familiar (family-wise error rate), evitando falsos positivos por múltiples pruebas.

### 4.4 Extensión a más de 2 algoritmos (referencia futura)

Si en el futuro se agregan más algoritmos a comparar, la prueba pareada de a pares deja de ser
apropiada (inflación del error tipo I). En ese caso, por dataset:

1. **Friedman test** (no paramétrico, para >2 muestras relacionadas) como ómnibus.
2. Si es significativo, **post-hoc** de Nemenyi o Dunn con corrección (Holm/Bonferroni) para
   comparaciones por pares.

No aplica al caso actual (2 algoritmos), se documenta como referencia.

## 5. Otras consideraciones importantes para las conclusiones

- **No mezclar datasets en un mismo test** (requisito explícito del profesor). Al final, se
  puede hacer una síntesis **descriptiva** (no un test estadístico nuevo) del tipo "MOEACKF
  obtuvo HV significativamente mayor en 4 de 6 datasets", inspirada en la práctica de Demšar
  (2006) para comparar clasificadores sobre múltiples datasets — dejando claro que es un
  resumen cualitativo, no una prueba estadística conjunta.
- **Varianza/consistencia entre corridas**: reportar IQR o desviación estándar, no solo
  mediana/media. Un algoritmo con media similar pero menor varianza es más confiable — vale la
  pena mencionarlo aunque el test de medianas no sea significativo.
- **Sobreajuste (train vs. test)**: comparar la brecha entre HV de entrenamiento y accuracy de
  test entre algoritmos (§2.1) — un algoritmo puede "ganar" en HV pero generalizar peor.
- **Análisis por región del espacio objetivo**: el HV es un resumen agregado y puede esconder
  que un algoritmo domina en la zona de alta dispersión (redes muy ralas) y pierde en la zona de
  alta precisión, o viceversa. Un indicador de *set coverage* (C-metric) entre pares de frentes,
  o simplemente graficar los frentes superpuestos, ayuda a matizar esta conclusión.
- **Significancia estadística ≠ relevancia práctica**: siempre acompañar el p-valor con el
  tamaño del efecto (Â₁₂) y, cuando sea posible, con la magnitud real de la diferencia (p.ej.
  "0.03 de HV" no dice nada sin contexto de la escala del indicador).
- **Reproducibilidad**: documentar en la tesis los hiperparámetros usados por combinación
  (tabla de `params_table.sh`), el número de corridas efectivas, y las semillas base.
- **Visualizaciones complementarias recomendadas**:
  - Boxplots o violin plots de HV (y demás indicadores) por algoritmo, uno por dataset.
  - Curvas de convergencia (`convergence.csv`) con banda de intervalo de confianza (o
    percentiles) sobre las corridas, en vez de una sola curva.
  - Frentes de Pareto superpuestos de varias corridas por algoritmo, o *empirical attainment
    function* (EAF) si se quiere una vista probabilística de la región del espacio objetivo
    cubierta por cada algoritmo.

## 6. Plantilla de tabla de resultados (por dataset)

| Algoritmo | HV (mediana [IQR]) | Accuracy test (mediana [IQR]) | p-valor (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | ... | ... | — | — |
| SNSGAII | ... | ... | ... | ... |

Repetir una tabla de este tipo por cada dataset (1 a 6), sin combinar filas de distintos
datasets en un mismo test. Marcar con `*` las diferencias significativas tras corrección por
comparaciones múltiples (p.ej. `α = 0.05` ajustado con Holm-Bonferroni).

---

**Nota**: actualmente no existe en el repositorio un script que automatice estos cálculos
(búsqueda en `bayesian_search.py`, `extract_best_params.py`, `plot_*.py`,
`analyze_network.py` no encontró ninguno). Si se desea, el siguiente paso natural es
implementar un script en Python (`pandas` + `scipy.stats`) que lea `results.csv` y
`fronts/*_front.csv` / `*_spikes.csv` por dataset y genere automáticamente la tabla de §6.

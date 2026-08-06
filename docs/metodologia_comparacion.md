# Metodología de comparación estadística entre algoritmos

> Ámbito: comparación de **MOEACKF vs. SNSGAII** sobre un mismo dataset (tarea), y de cada uno de
> estos algoritmos consigo mismo bajo distintas condiciones experimentales (plataforma SNN/ANN,
> evaluación bootstrap/completa, tiempo de cómputo). Toda comparación se realiza **por separado
> para cada dataset** — nunca se agregan datos de distintos datasets en una misma prueba
> estadística. Las conclusiones agregadas entre datasets, si se presentan, son un resumen
> cualitativo posterior y no un test adicional.

## 1. Diseño experimental

Ambos algoritmos se ejecutan sobre el mismo problema bi-objetivo `SparseSNN`, que minimiza
simultáneamente:

- **f1 (dispersión / complejidad)**: fracción de pesos sinápticos activos (no nulos) en la red.
- **f2 (error)**: `1 − accuracy` en el conjunto de entrenamiento (`FITNESS_MODE=accuracy`).

El diseño experimental tiene tres propiedades que hay que explicitar en la tesis porque
sustentan la validez de la comparación:

1. **Presupuesto de evaluaciones igualado por dataset**: para cada dataset, ambos algoritmos
   consumen el mismo `maxfe` (verificado directamente en la columna `maxfe` de `results.csv`:
   ds1=20000, ds2=20000, ds3=40000, ds4=40000, igual para MOEACKF y SNSGAII en cada caso). Sin
   esto, una diferencia en HV podría deberse simplemente a que un algoritmo tuvo más presupuesto,
   no a que sea mejor.
2. **Hiperparámetros fijos por combinación**: en `results/normal_results` y
   `results/bootstrap_results`, los hiperparámetros de los operadores (`--disC`, `--disM`,
   `--proM`, etc.) provienen de una búsqueda Bayesiana (`bayesian_search.py` →
   `params_table.sh`) específica por combinación algoritmo×dataset, y se mantienen constantes
   entre las corridas repetidas de esa combinación. `results/time_experiments` es la excepción
   deliberada: usa hiperparámetros **compartidos** (no afinados por algoritmo) precisamente para
   que la comparación de tiempo de cómputo no esté confundida por una diferencia de
   hiperparámetros — ver §3.6.
3. **Semillas pareadas por índice de corrida**: en `run_one()` de `run_experiments.sh`,
   `seed = BASE_SEED + run`, y esta fórmula **no depende del algoritmo**. La corrida `run=i` de
   MOEACKF y la corrida `run=i` de SNSGAII parten de la misma semilla. Esto convierte la
   comparación en un **diseño pareado** dentro del pipeline C++ (ver §1.4 sobre qué fuentes son
   pareables y cuáles no), y debe aprovecharse eligiendo pruebas estadísticas pareadas (§6.1),
   que tienen mayor potencia que las no pareadas al controlar la varianza atribuible a la semilla.

**Número de corridas independientes**: `results/normal_results`, `results/bootstrap_results` y
`results/MATLAB_RESULTS` tienen **N=31 corridas** por combinación (seeds 100–130 en las dos
primeras); `results/time_experiments` tiene **N=15** (seeds 100–114). Para potencia estadística
adecuada en pruebas no paramétricas se recomienda N≥30, por lo que las conclusiones basadas en
`time_experiments` deben leerse con esa salvedad de N menor.

### 1.4 Las cuatro fuentes de datos archivadas

| Fuente | N | Pareo | Qué mide | Limitación clave |
|---|---|---|---|---|
| `results/normal_results/` | 31 | Por seed (100–130) | HV, frentes, convergencia — SNN con hiperparámetros afinados por algoritmo | Es el experimento principal; el resto de las fuentes se comparan contra esta |
| `results/bootstrap_results/` | 31 | Por seed (100–130), también pareable contra normal_results | Igual que arriba, pero fitness evaluado por remuestreo bootstrap (§3.5) | Usa los mismos hiperparámetros afinados que normal_results (misma búsqueda Bayesiana), solo cambia el régimen de evaluación — por eso sí es pareable contra normal_results, a diferencia de time_experiments |
| `results/MATLAB_RESULTS/` | 31 | **Sin seed — no pareable** | HV, complejidad y accuracy de la red de mejor accuracy, de la implementación original PlatEMO/MATLAB sobre ANN (no spiking) | RNG e implementación distintos al pipeline C++; evidencia direccional, no un experimento controlado en el mismo sentido que MOEACKF-vs-SNSGAII |
| `results/time_experiments/` | 15 | Por seed (100–114) | `time_s` de MOEACKF vs. SNSGAII bajo hiperparámetros compartidos y hardware controlado | HV de esta fuente **no** se usa (hiperparámetros no afinados, no comparable con normal_results) |

### 1.5 Fuente de verdad de los parámetros experimentales

`run_experiments.sh` es un script vivo que se ha reeditado muchas veces para corridas ad-hoc
distintas desde que se generaron los datos archivados (`git log -p` muestra decenas de cambios en
`RUNS`, `SEED`, `DATASETS`, `MAXFE_*`). Por ejemplo, el archivo tiene hoy `MAXFE_MOEACKF_1=2000` y
`SEED=50` como defaults, pero los datos realmente archivados en `results/normal_results/ds1_*`
usan `maxfe=20000` y seeds 100–130. **La metodología y el análisis toman como fuente de verdad las
columnas de `results.csv` / `convergence.csv` de cada corrida archivada**, no el estado actual del
script. Esto es importante para la sección de reproducibilidad de la tesis: los hiperparámetros de
operadores sí deben documentarse aparte desde `params_table.sh` (o los estudios de
`bo_results/`, ver `docs/resultados_hiperparametros.md`), pero `maxfe`, `nhidden` y las seeds
usadas se leen directamente de los datos, no del script.

## 2. Los ángulos de comparación — visión general

| # | Ángulo | Pregunta que responde | Fuente | Pareo | Script |
|---|---|---|---|---|---|
| 1 | HV principal | ¿Qué algoritmo tiene mejor frente de Pareto (SNN)? | `normal_results` | Por seed | `compare_algorithms.py` |
| 2 | Frentes de Pareto | ¿Cómo se reparten complejidad y error las soluciones de cada algoritmo? | `normal_results/*/fronts` | Por seed / conjuntos | `compare_fronts.py` |
| 3 | Convergencia | ¿Qué tan rápido converge cada algoritmo al mismo presupuesto de FE? | `normal_results/*/convergence.csv` | Por seed | `compare_convergence.py` |
| 4 | Plataforma | ¿HV/complejidad mejor en SNN (C++) o ANN (MATLAB original)? | `normal_results` vs. `MATLAB_RESULTS` | No pareado | `compare_platform.py` |
| 5 | Bootstrap | ¿El remuestreo bootstrap mejora o empeora el HV? | `normal_results` vs. `bootstrap_results` | Por seed | `compare_bootstrap.py` |
| 6 | Tiempo | ¿Qué algoritmo es más rápido en tiempo de reloj? | `time_experiments` | Por seed | `compare_time.py` |

Los ángulos 1–3 comparan MOEACKF vs. SNSGAII dentro de la misma fuente (`normal_results`); los
ángulos 4–6 comparan cada algoritmo consigo mismo bajo dos condiciones distintas (plataforma,
régimen de evaluación, tiempo). `apply_corrections.py` agrega los p-valores de los 6 scripts y
aplica la corrección por comparaciones múltiples (§6.5).

## 3. Indicadores por ángulo

### 3.1 Hipervolumen (HV)

**El HV ya es comparable sin trabajo adicional.** En `include/metrics/HV.hpp`, el hipervolumen se
calcula normalizando `f1,f2` contra un punto de referencia **fijo e independiente del algoritmo,
la corrida, el dataset y — por construcción de la métrica — la plataforma**
(`SparseSNN::GetOptimum()` retorna `(1,1)`, es decir, el ideal `(0,0)` sin normalizar por
dataset). Esto es una fortaleza metodológica a mencionar explícitamente: el HV no depende de un
frente de referencia estimado empíricamente por dataset, por lo que las comparaciones entre
algoritmos, entre datasets, y (con las salvedades de §3.4) entre plataformas SNN/ANN son directas,
porque tanto `f1_complexity`/`f2_train_error` (SNN) como `complexity`/`accuracy` (ANN, en
`MATLAB_RESULTS`) son fracciones en `[0,1]` con la misma definición (fracción de pesos activos;
1−accuracy).

### 3.2 Frentes de Pareto (complejidad vs. error)

El HV es un resumen agregado del frente; para entender **de qué está hecha** la diferencia de HV
se complementan tres técnicas sobre `fronts/*_front.csv` (columnas `f1_complexity`,
`f2_train_error`, tamaño de frente variable por corrida — típicamente 4–14 soluciones, no fijo):

1. **Solución de mejor accuracy por corrida**: en cada frente, la fila con `f2_train_error`
   mínimo (desempate por `f1_complexity` mínimo). Da una muestra pareada por seed de
   (complejidad, error, tamaño del frente) — la misma definición que trae precalculada
   `MATLAB_RESULTS` (columnas `accuracy,complexity,hv`), lo que habilita directamente la
   comparación de plataforma (§3.4) sin tener que reconstruir nada adicional.
2. **Frente combinado y C-metric**: unión de los puntos `(f1,f2)` de las 31 corridas de un
   algoritmo, deduplicada (14%–72% de los puntos crudos son duplicados exactos entre corridas
   distintas, según dataset) y filtrada a no-dominados (dominancia estricta) → "frente
   combinado". Se compara con la **C-metric de Zitzler & Thiele** (§6.4) entre los frentes
   combinados de ambos algoritmos, más un gráfico de dispersión/escalera superpuesto.
3. **Bandas de complejidad**: deciles de `f1_complexity` sobre el pool de puntos crudos de ambos
   algoritmos, comparando el error mediano por (algoritmo, banda) — para ver si un algoritmo
   domina en la zona rala (baja complejidad) y el otro en la zona precisa (alta complejidad), en
   vez de asumir que el HV agregado cuenta toda la historia.

### 3.3 Convergencia

`convergence.csv` registra `generation` y `hv` por corrida, **no FE**. Esto importa porque
`generation` **no es comparable entre algoritmos**: verificado directamente en los datos, para
ds1 (maxfe=20000 ambos) MOEACKF llega a la generación final 165 mientras SNSGAII llega a 199 —
MOEACKF consume más FE por generación (consistente con que su mecanismo de fusión de
conocimiento entre escalas evalúa puntos adicionales por generación). Se aproxima
`fe_approx = generation / generation_final_de_la_corrida × maxfe`, asumiendo consumo ~constante
de FE por generación dentro de cada corrida — es una aproximación declarada, no un valor exacto
(ver `comparison_lib.approximate_fe`).

Con `fe_approx`, dos técnicas:

- **Curvas de convergencia**: HV mediano + banda P10–P90 (31 corridas) vs. `fe_approx`, ambos
  algoritmos superpuestos, por dataset.
- **Velocidad**: `fe_approx` en el que cada corrida alcanza por primera vez el 95% de su propio
  HV final (indicador principal, entra en la corrección de §6.5) y, como chequeo de
  sensibilidad, el 90% (no entra en la corrección — ver §6.5 sobre por qué). Se usa el máximo
  acumulado de HV antes de buscar el umbral (defensivo: se verificó que HV es 100% monótona
  no-decreciente en las 24 combinaciones archivadas revisadas, pero no se asume ciegamente).

### 3.4 Plataforma: ANN (MATLAB) vs. SNN (C++)

`results/MATLAB_RESULTS/ds{N}_{algo}_matlab/` trae, por corrida: la accuracy de la red de mejor
accuracy del frente, la complejidad de esa misma red, y el HV del frente completo — de la
implementación original de PlatEMO/MATLAB, sobre una ANN estándar (no spiking) con el mismo
problema biobjetivo. Comparación **por algoritmo** (MOEACKF-ANN vs. MOEACKF-SNN, SNSGAII-ANN vs.
SNSGAII-SNN), no agrupada: así se aísla la plataforma como único factor distinto entre las
muestras, en vez de mezclar el efecto de la plataforma con el del algoritmo.

Estos CSV **no tienen columna seed** — no hay forma de emparejar una corrida MATLAB con una
corrida C++ (implementaciones y generadores de números aleatorios distintos, aunque el índice de
fila nominal pudiera sugerir una correspondencia, esa correspondencia no tiene ningún significado
estadístico real). Por eso esta comparación usa **Mann-Whitney U** (muestras independientes, ver
§6.2), no Wilcoxon.

### 3.5 Bootstrap vs. evaluación completa

`results/bootstrap_results/` usa los mismos hiperparámetros afinados que `results/normal_results/`
(misma búsqueda Bayesiana por combinación algoritmo×dataset) pero con `BOOTSTRAP_EVAL=true` en
`run_experiments.sh`: el fitness de cada individuo se calcula promediando sobre 50 remuestreos
bootstrap del 30% del conjunto de entrenamiento por generación, en vez de una sola pasada
completa (siguiendo Loyola-Jara et al., 2026), buscando fitness más robusto a la generalización.
Comparte exactamente el mismo conjunto de seeds (100–130) que `normal_results`, así que es
pareable **contra normal_results** (no solo entre algoritmos): la pregunta no es "MOEACKF vs.
SNSGAII" sino "¿el régimen bootstrap mejora o empeora el HV de este mismo algoritmo en este mismo
dataset?", por separado para MOEACKF y para SNSGAII (8 comparaciones: 4 datasets × 2 algoritmos).
El costo en `time_s` (bootstrap es sistemáticamente varias veces más lento — se evalúa fitness 50
veces por generación en vez de una) se reporta de forma descriptiva junto a cada tabla, sin test
formal: es un efecto esperado del propio diseño del remuestreo, no una pregunta de investigación.

### 3.6 Tiempo de cómputo (`time_s`)

`results/normal_results` y `results/bootstrap_results` **siguen sin comparación válida de
tiempo**: esas corridas se ejecutaron en distintas tandas con distinta cantidad de procesos en
paralelo y/o hilos OMP (`JOBS`, `OMP_THREADS` en `run_experiments.sh`), por lo que el tiempo de
reloj no refleja el costo computacional real de cada algoritmo de forma comparable.

`results/time_experiments/` es la excepción: se ejecutó específicamente para permitir esta
comparación, con (a) hiperparámetros **compartidos** entre algoritmos (sin `params_table.sh`) y
(b) condiciones de hardware controladas — mismo `JOBS`/`OMP_THREADS`, sin otros procesos
compitiendo por CPU. Por eso, y solo para esta fuente, la comparación de `time_s` (pareada por
seed, Wilcoxon) se reporta como válida. El HV de `time_experiments` no se usa como indicador de
calidad — con hiperparámetros compartidos (no afinados por algoritmo) no es comparable con el HV
de `normal_results`.

## 4. Por qué ya no se reporta "accuracy de test"

Es una limitación **estructural** de los datos archivados, no una tarea pendiente. El código
(`src/main.cpp`) solo exporta `*_spikes.csv` (necesarios para reconstruir accuracy de test) cuando
`nOutputs == 1`; las 31×16 corridas archivadas en `normal_results` y `bootstrap_results` usan
todas arquitectura de 2 salidas (WTA, `meta.json: "nOutputs": 2`). No existe ningún archivo
`*_spikes.csv` en ninguna de las cuatro fuentes — confirmado por búsqueda exhaustiva. Volver a
generarlos requeriría re-ejecutar todos los experimentos con una arquitectura de 1 salida, lo cual
queda fuera del alcance de este análisis sobre datos ya recolectados.

El análisis de frentes de Pareto (§3.2) es el sustituto práctico: en vez de un accuracy de test
puntual, mira la relación complejidad–error de **todo el frente** de cada corrida sobre el
conjunto de entrenamiento, incluyendo la solución de mejor accuracy. No mide generalización, pero
sí evita el punto débil original de mirar solo el HV agregado (que puede esconder en qué región
del espacio de objetivos ocurre la diferencia).

## 5. Verificación de supuestos

Antes de elegir la prueba estadística, para cada indicador y dataset:

1. **Normalidad**: Shapiro-Wilk sobre la distribución de cada grupo por separado (N corridas).
   Con N moderado (15–31) el test tiene poca potencia para detectar desviaciones pequeñas de
   normalidad, así que no debe ser el único criterio.
2. **Aun si no se rechaza normalidad**, se recomienda usar de todas formas pruebas **no
   paramétricas** como criterio principal, porque:
   - El hipervolumen (HV) y las demás métricas usadas suelen tener distribución sesgada o
     acotada, no gaussiana.
   - Es el estándar de facto en la literatura de computación evolutiva para comparar algoritmos
     estocásticos (Derrac, García, Molina & Herrera, 2011 — *"A practical tutorial on the use of
     nonparametric statistical tests..."*).
   - Son robustas a outliers, frecuentes en optimización estocástica.

## 6. Pruebas estadísticas

### 6.1 Comparación principal (pareada)

Cuando existe un seed compartido entre las dos muestras (ángulos 1, 2a, 3, 5, 6 — ver §2), la
prueba es **Wilcoxon signed-rank** (pareada), aplicada a los pares `(x[i], y[i])` con mismo
índice de seed. Caso degenerado: si `x[i] == y[i]` para **todos** los pares (ocurre, p.ej., en
ds2/Climate para el umbral de convergencia al 90%: ambos algoritmos ya alcanzan el 90% de su HV
final en la generación 0 en las 31 corridas), `scipy.stats.wilcoxon` no admite diferencias nulas
en todos los pares — se reporta `p=1.0` sin evidencia de diferencia, en vez de fallar (ver
`comparison_lib.paired_comparison`).

**Alternativa** (Mann-Whitney U, para muestras independientes): usar solo si el pareo se rompe —
por ejemplo, si alguna corrida de un algoritmo falló/no convergió y no hay el mismo número de
corridas válidas para ambos algoritmos en algún índice.

### 6.2 Comparación de plataforma (no pareada)

Para el ángulo 4 (§3.4), sin seed compartida entre MATLAB y C++, la prueba es **Mann-Whitney U**
(`scipy.stats.mannwhitneyu`, muestras independientes). Se usa aunque `N_x == N_y == 31`
coincidan en tamaño: el tamaño igual no implica que haya pareo real.

### 6.3 Tamaño del efecto

El p-valor por sí solo no basta: con N alto, diferencias mínimas pueden ser "significativas" sin
ser relevantes en la práctica. Reportar siempre junto al p-valor el **Â₁₂ de Vargha-Delaney**:
probabilidad de que una corrida aleatoria de un grupo supere a una corrida aleatoria del otro.
Interpretación estándar sobre `|Â₁₂ − 0.5|`: `< 0.06` insignificante, `0.06–0.14` pequeño,
`0.14–0.21` mediano, `≥ 0.21` grande.

- **Pareado**: proporción de pares donde `x > y` (empates cuentan 0.5) —
  `comparison_lib.vargha_delaney_a12_paired`.
- **No pareado**: `Â₁₂ = U₁ / (n₁·n₂)`, con `U₁` el estadístico U de Mann-Whitney asociado al
  primer argumento — `comparison_lib.vargha_delaney_a12_unpaired`. Invertir el orden de los
  argumentos invierte el resultado a `1 − Â₁₂`; el módulo incluye un auto-test
  (`python3 comparison_lib.py`) que verifica esto antes de confiar en los resultados de
  plataforma.

### 6.4 C-metric (cobertura de Zitzler)

Para comparar dos frentes combinados (§3.2, técnica 2), `C(A,B)` = fracción de puntos del frente
`B` que están dominados o igualados por al menos un punto del frente `A` — **dominancia débil**
(a propósito distinta de la dominancia estricta usada para filtrar el frente no-dominado en
primer lugar; con dominancia débil, `C(A,A) = 1` siempre, por definición). No produce p-valor: es
una estadística descriptiva sobre dos conjuntos, no un test de hipótesis sobre muestras — se
reporta junto al gráfico superpuesto, no se le busca significancia. Con frentes combinados
pequeños (4–14 puntos en los datos archivados), la resolución de la métrica es limitada;
interpretarla como una magnitud aproximada, no como una proporción de alta precisión.

### 6.5 Corrección por comparaciones múltiples

Se usa **Holm-Bonferroni** (step-down), aplicada **dentro de cada familia** de tests relacionados
— nunca entre familias, porque cada familia responde una pregunta distinta y corregir entre ellas
penalizaría el poder estadístico sin justificación. `apply_corrections.py` agrupa los p-valores
crudos que ya calculó cada script `compare_*.py`:

| Familia | Agrupada por | Miembros | m | # familias | Tests |
|---|---|---|---|---|---|
| `principal_snn` | dataset | HV, complejidad/error/tamaño-de-frente de la mejor red, velocidad de convergencia (FE a 95%) | 5 | 4 | 20 |
| `plataforma` | dataset × algoritmo | HV, complejidad, accuracy de la mejor red (SNN vs. ANN) | 3 | 8 | 24 |
| `bootstrap` | dataset | HV bootstrap-vs-normal de MOEACKF, ídem de SNSGAII | 2 | 4 | 8 |
| `tiempo` | dataset | `time_s` | 1 (sin corrección real) | 4 | 4 |

Total: 20 familias, 56 tests corregidos. **Fuera de toda familia** (no se corrigen, se reportan
como chequeos de sensibilidad o descriptivos): FE hasta 90% del HV final (redundante con el de
95% — misma trayectoria, mismo run, criterio apenas distinto) y la C-metric (no produce p-valor).
El veredicto final citable en la tesis es `comparison/corrections_summary.csv`
(`p_holm`, `significant_holm`); las tablas individuales de cada ángulo muestran el p crudo por
transparencia, pero no son el resultado corregido.

### 6.6 Extensión a más de 2 algoritmos (referencia futura)

Si en el futuro se agregan más algoritmos a comparar, la prueba pareada de a pares deja de ser
apropiada (inflación del error tipo I). En ese caso, por dataset:

1. **Friedman test** (no paramétrico, para >2 muestras relacionadas) como ómnibus.
2. Si es significativo, **post-hoc** de Nemenyi o Dunn con corrección (Holm/Bonferroni) para
   comparaciones por pares.

No aplica al caso actual (2 algoritmos), se documenta como referencia.

## 7. Otras consideraciones importantes para las conclusiones

- **No mezclar datasets en un mismo test** (requisito explícito del profesor). Al final, se puede
  hacer una síntesis **descriptiva** (no un test estadístico nuevo) del tipo "MOEACKF obtuvo HV
  significativamente mayor en N de 4 datasets", inspirada en la práctica de Demšar (2006) para
  comparar clasificadores sobre múltiples datasets — dejando claro que es un resumen cualitativo,
  no una prueba estadística conjunta.
- **Varianza/consistencia entre corridas**: reportar IQR o desviación estándar, no solo
  mediana/media. Un algoritmo con media similar pero menor varianza es más confiable — vale la
  pena mencionarlo aunque el test de medianas no sea significativo.
- **Significancia estadística ≠ relevancia práctica**: siempre acompañar el p-valor con el
  tamaño del efecto (Â₁₂) y, cuando sea posible, con la magnitud real de la diferencia.
- **Reproducibilidad**: documentar en la tesis los hiperparámetros usados por combinación (tabla
  de `params_table.sh` / `docs/resultados_hiperparametros.md`), el número de corridas efectivas
  por fuente (§1.4), y que `maxfe`/`nhidden`/seeds se toman de los datos archivados, no del
  estado actual de `run_experiments.sh` (§1.5).
- **Datasets 5 y 6** (Iris, Wine) están definidos en el código (`DS_NAMES` en
  `comparison_lib.py`) pero no tienen resultados corridos en ninguna de las cuatro fuentes — las
  conclusiones de este documento y del análisis cubren únicamente ds1–ds4.

## 8. Plantillas de tabla de resultados

Una tabla por ángulo, por dataset (repetida para cada dataset con datos, sin combinar filas de
distintos datasets en un mismo test):

**Ángulos 1, 2a, 3, 5, 6 (pareados):**

| Grupo | Indicador (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| A | ... | ... | — | — |
| B | ... | ... | ... | ... |

**Ángulo 4 (plataforma, no pareado):**

| Grupo | Indicador (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | ... | ... | — | — |
| ANN-MATLAB | ... | ... | ... | ... |

Marcar con `*` las diferencias significativas antes de corrección; el veredicto final tras
Holm-Bonferroni (§6.5) va en una columna aparte (`p_holm`, `significant_holm`), no reemplaza al
p crudo en la tabla individual.

## Scripts

No hace falta calcular nada de este documento a mano: cada ángulo tiene un script en la raíz de
`platemo-cpp/` que lee los datos archivados y escribe tablas markdown/CSV + figuras a
`comparison/<ángulo>/`.

| Ángulo | Script | Salida |
|---|---|---|
| HV principal | `compare_algorithms.py` | `comparison/hv/` |
| Frentes de Pareto | `compare_fronts.py` | `comparison/fronts/` |
| Convergencia | `compare_convergence.py` | `comparison/convergence/` |
| Plataforma | `compare_platform.py` | `comparison/platform/` |
| Bootstrap | `compare_bootstrap.py` | `comparison/bootstrap/` |
| Tiempo | `compare_time.py` | `comparison/time/` |
| Corrección conjunta | `apply_corrections.py` | `comparison/corrections_summary.{md,csv}` |

Toda la estadística compartida (Wilcoxon/Mann-Whitney, Â₁₂ pareado y no pareado, Holm-Bonferroni,
lectura de frentes, filtro no-dominado, C-metric, aproximación de FE) vive en `comparison_lib.py`,
con un auto-test ejecutable (`python3 comparison_lib.py`) que valida los casos borde antes de
confiar en los resultados.

# Resultados de la búsqueda de hiperparámetros

## Resumen general

Siguiendo la metodología descrita (búsqueda bayesiana con Optuna/TPE en un bucle externo sobre el SMOEA correspondiente), se ejecutaron **8 estudios independientes**, uno por cada combinación algoritmo × dataset (S-NSGA-II y MOEA-CKF, sobre ds1–ds4). Cada estudio constó de **60 trials**, con un presupuesto `maxFE` fijo por dataset y compartido entre ambos algoritmos (ds1 = 9900, ds2 = 11500, ds3 = 13900, ds4 = 28300), y cada trial se evaluó promediando el HV sobre tres tamaños de capa oculta (20, 40 y 80 neuronas), tal como se especifica en el protocolo de evaluación por trial.

La Tabla 1 resume el mejor resultado obtenido por cada estudio.

**Tabla 1. Mejor HV medio (sobre 20/40/80 neuronas) por algoritmo y dataset.**

| Dataset | Features | maxFE | Algoritmo | Mejor HV medio | Trial ganador | Arquitectura ganadora |
|---|---|---|---|---|---|---|
| ds1 | 14 | 9 900 | MOEA-CKF | 0.8704 | #55 | Accuracy / 2 salidas WTA / Poisson |
| ds1 | 14 | 9 900 | S-NSGA-II | 0.8725 | #10 | Accuracy / 2 salidas WTA / TTFS |
| ds2 | 18 | 11 500 | MOEA-CKF | 0.9279 | #42 | Accuracy / 2 salidas WTA / TTFS |
| ds2 | 18 | 11 500 | S-NSGA-II | 0.9276 | #58 | Accuracy / 2 salidas WTA / Poisson |
| ds3 | 24 | 13 900 | MOEA-CKF | 0.7540 | #2  | Accuracy / 2 salidas WTA / Poisson |
| ds3 | 24 | 13 900 | S-NSGA-II | 0.7704 | #22 | Accuracy / 2 salidas WTA / TTFS |
| ds4 | 60 | 28 300 | MOEA-CKF | 0.8051 | #53 | Accuracy / 2 salidas WTA / TTFS |
| ds4 | 60 | 28 300 | S-NSGA-II | 0.8134 | #28 | Accuracy / 2 salidas WTA / TTFS |

En los ocho estudios, la configuración ganadora final corresponde siempre a una arquitectura de **dos neuronas de salida con decodificación WTA y métrica de exactitud**, ya sea con encoder Poisson o TTFS; ninguna arquitectura de una sola neurona de salida evaluada por AUC resultó ganadora en ningún estudio (ver más abajo el detalle de ds2, el único dataset donde esa alternativa formó parte del espacio de búsqueda).

## Detalle de hiperparámetros ganadores por estudio

**Tabla 2. Hiperparámetros de la mejor configuración por estudio.**

| Parámetro | ds1‑CKF | ds1‑NSGA | ds2‑CKF | ds2‑NSGA | ds3‑CKF | ds3‑NSGA | ds4‑CKF | ds4‑NSGA |
|---|---|---|---|---|---|---|---|---|
| disC | 78.64 | 5.09 | 10.00 | 7.21 | 38.83 | 74.99 | 98.25 | 97.99 |
| disM | 7.74 | 91.28 | 15.46 | 60.12 | 18.69 | 45.30 | 76.13 | 68.53 |
| proM | 0.86 | 2.11 | 2.04 | 0.64 | 0.63 | 2.34 | 2.20 | 1.68 |
| disSM | 16.24 | 92.32 | 64.03 | 9.83 | 22.04 | 56.14 | 89.36 | 74.65 |
| proSM | 1.63 | 2.75 | 0.40 | 2.36 | 0.39 | 1.76 | 1.84 | 2.37 |
| sLower | 0.515 | 0.565 | 0.778 | 0.844 | 0.909 | 0.606 | 0.617 | 0.587 |
| sGap | 0.123 | 0.054 | 0.062 | 0.280 | 0.115 | 0.158 | 0.193 | 0.265 |
| sUpper (derivado) | 0.638 | 0.619 | 0.840 | 1.000 | 1.000 | 0.765 | 0.810 | 0.852 |
| wScale | 41.29 | 41.02 | 6.74 | 29.33 | 13.35 | 29.29 | 28.47 | 32.70 |
| dt (ms) | 1.0 | 0.1 | 0.1 | 1.0 | 2.0 | 0.1 | 0.1 | 0.1 |
| encoding_duration | 80 | 150 | 100 | 60 | 120 | 30 | 70 | 110 |
| extra_eval | 150 | 120 | 100 | 50 | 190 | 10 | 70 | 90 |
| evaluation_duration (derivado) | 230 | 270 | 200 | 110 | 310 | 40 | 140 | 200 |
| encoder ganador | Poisson | TTFS | TTFS | Poisson | Poisson | TTFS | TTFS | TTFS |
| max_rate (solo Poisson) | 57.30 | — | — | 55.09 | 156.98 | — | — | — |
| refractory_period (solo Poisson) | 1 | — | — | 4 | 3 | — | — | — |

*(CKF = MOEA-CKF, NSGA = S-NSGA-II)*

## Caso particular: ds2 y el espacio de arquitecturas de 4 vías

El estudio de ds2 se ejecutó antes de que el pipeline experimental redujera la variable categórica `architecture` a solo las dos alternativas de dos neuronas de salida. En consecuencia, para ds2 el optimizador TPE exploró efectivamente **4 arquitecturas**: (1) AUC/1 salida/Poisson, (2) Accuracy/2 salidas WTA/Poisson, (3) Accuracy/2 salidas WTA/TTFS y (4) AUC/1 salida/TTFS.

La Tabla 3 muestra cuántos de los 60 trials cayeron en cada arquitectura, para cada algoritmo.

**Tabla 3. Distribución de trials por arquitectura en ds2 (60 trials totales por algoritmo).**

| Arquitectura | MOEA-CKF | S-NSGA-II |
|---|---|---|
| 1 — AUC / 1 salida / Poisson | 6 | 6 |
| 2 — Accuracy / 2 salidas WTA / Poisson | 20 | 40 |
| 3 — Accuracy / 2 salidas WTA / TTFS | 26 | 6 |
| 4 — AUC / 1 salida / TTFS | 8 | 8 |

A pesar de que las arquitecturas de una sola salida (1 y 4) sí fueron muestreadas por TPE en ambos estudios, **ninguna de las dos produjo nunca el mejor trial**: el ganador de MOEA-CKF fue la arquitectura 3 (HV = 0.9279) y el de S-NSGA-II la arquitectura 2 (HV = 0.9276). Esto se puede apreciar en las Figuras 1 y 2: los trials con arquitectura de una salida quedan agrupados en la banda baja (HV ≈ 0.64–0.83), mientras que los trials con arquitectura de dos salidas se concentran en una banda alta y estrecha (HV ≈ 0.91–0.93), con una separación clara entre ambos grupos.

![Figura 1. Convergencia del HV medio — ds2 — MOEA-CKF](../bo_results/figures_tesis/convergencia_MOEACKF_ds2.png)
*Figura 1. Convergencia del HV medio por trial — ds2, MOEA-CKF.*

![Figura 2. Convergencia del HV medio — ds2 — S-NSGA-II](../bo_results/figures_tesis/convergencia_SNSGAII_ds2.png)
*Figura 2. Convergencia del HV medio por trial — ds2, S-NSGA-II.*

Los resultados mencionados se pueden observar también en las Figuras 3 y 4, donde `architecture` explica entre el 72 % y el 94 % de la varianza del HV en ds2 — muy por encima de cualquier otro hiperparámetro.

![Figura 3. Importancia de hiperparámetros — ds2 — MOEA-CKF](../bo_results/figures_tesis/importancia_MOEACKF_ds2.png)
*Figura 3. Importancia de hiperparámetros (fANOVA) — ds2, MOEA-CKF.*

![Figura 4. Importancia de hiperparámetros — ds2 — S-NSGA-II](../bo_results/figures_tesis/importancia_SNSGAII_ds2.png)
*Figura 4. Importancia de hiperparámetros (fANOVA) — ds2, S-NSGA-II.*

Este resultado empírico en ds2 es, en la práctica, el que motivó restringir el espacio de búsqueda a únicamente las dos arquitecturas de dos salidas para los estudios posteriores de ds1, ds3 y ds4: dado que las variantes de una sola salida evaluadas por AUC no resultaron competitivas en ningún caso, mantenerlas en el espacio de búsqueda solo diluye el presupuesto de trials sin aportar candidatos viables.

---

# Análisis y discusión de resultados

## HV alcanzado por dataset

El HV medio alcanzado por la mejor configuración varía considerablemente entre datasets: ds2 (≈0.93) > ds1 (≈0.87) > ds4 (≈0.81) > ds3 (≈0.75). Esta ordenación no sigue una relación monótona con el número de atributos de entrada (14, 18, 24 y 60 respectivamente para ds1–ds4): ds2, con 18 atributos, obtiene el HV más alto de los cuatro, mientras que ds3, con 24, obtiene el más bajo, y ds4, con 60 atributos (el más grande), se ubica en un punto intermedio. Esto sugiere que la dificultad efectiva del problema de clasificación de cada dataset domina por sobre la sola dimensionalidad de entrada a la hora de explicar el HV alcanzable, algo esperable dado que el HV combina exactitud (o AUC) y esparsidad, y no solo la primera. En cualquier caso, la equidad de la comparación entre datasets está resguardada porque, para cada dataset, ambos algoritmos comparten exactamente el mismo `maxFE` y el mismo protocolo de evaluación (Tabla 1), de modo que las diferencias de HV entre datasets reflejan la dificultad relativa del problema de clasificación subyacente y no una asimetría del presupuesto de búsqueda.

## MOEA-CKF vs. S-NSGA-II

En 3 de los 4 datasets (ds1, ds3 y ds4), S-NSGA-II obtuvo un HV medio ligeramente superior al de MOEA-CKF en la configuración ganadora (diferencias de 0.002 a 0.016 en HV), mientras que en ds2 la relación se invierte levemente a favor de MOEA-CKF (0.9279 vs. 0.9276, una diferencia de apenas 0.0003 que no es relevante en términos prácticos, ver Figuras 1 y 2). Cabe subrayar que estas cifras corresponden al **mejor trial de un único estudio de búsqueda de hiperparámetros** — no a una comparación estadística entre múltiples corridas independientes de los algoritmos con esa configuración fija, que es lo que se reserva para la fase experimental posterior. Por lo tanto, estas diferencias no deben interpretarse todavía como evidencia de que un algoritmo domine sistemáticamente al otro, sino únicamente como que, bajo el mismo presupuesto de búsqueda de hiperparámetros, ambos alcanzan un HV comparable, lo que valida que el espacio de búsqueda y el presupuesto definidos son igualmente aprovechables por ambos SMOEA.

En ds1, la diferencia entre algoritmos (0.8704 vs. 0.8725) es mínima y ambos estudios convergen con estabilidad hacia el final de los 60 trials. Esto se puede apreciar en las Figuras 5 y 6.

![Figura 5. Convergencia del HV medio — ds1 — MOEA-CKF](../bo_results/figures_tesis/convergencia_MOEACKF_ds1.png)
*Figura 5. Convergencia del HV medio por trial — ds1, MOEA-CKF.*

![Figura 6. Convergencia del HV medio — ds1 — S-NSGA-II](../bo_results/figures_tesis/convergencia_SNSGAII_ds1.png)
*Figura 6. Convergencia del HV medio por trial — ds1, S-NSGA-II.*

En ds3, S-NSGA-II supera a MOEA-CKF por el margen más amplio de los cuatro datasets (0.7704 vs. 0.7540). Los resultados mencionados se pueden observar en las Figuras 7 y 8.

![Figura 7. Convergencia del HV medio — ds3 — MOEA-CKF](../bo_results/figures_tesis/convergencia_MOEACKF_ds3.png)
*Figura 7. Convergencia del HV medio por trial — ds3, MOEA-CKF.*

![Figura 8. Convergencia del HV medio — ds3 — S-NSGA-II](../bo_results/figures_tesis/convergencia_SNSGAII_ds3.png)
*Figura 8. Convergencia del HV medio por trial — ds3, S-NSGA-II.*

En ds4, ambos algoritmos alcanzan un HV cercano (0.8051 vs. 0.8134), con trayectorias de convergencia similares entre sí, tal como se observa en las Figuras 9 y 10.

![Figura 9. Convergencia del HV medio — ds4 — MOEA-CKF](../bo_results/figures_tesis/convergencia_MOEACKF_ds4.png)
*Figura 9. Convergencia del HV medio por trial — ds4, MOEA-CKF.*

![Figura 10. Convergencia del HV medio — ds4 — S-NSGA-II](../bo_results/figures_tesis/convergencia_SNSGAII_ds4.png)
*Figura 10. Convergencia del HV medio por trial — ds4, S-NSGA-II.*

## Tendencias en los hiperparámetros ganadores

Algunas regularidades emergen al comparar la Tabla 2 entre estudios:

- **Paso de integración (`dt`)**: en 6 de los 8 estudios la mejor configuración adoptó el valor más fino disponible, `dt = 0.1` ms; solo ds1-CKF (`dt=1.0`) y ds3-CKF (`dt=2.0`) se alejan de ese valor. Esto sugiere que, en general, una mayor resolución temporal de la simulación de Izhikevich favorece el HV, aunque no de forma absoluta — hay al menos un caso por algoritmo donde un `dt` más grueso resultó igualmente competitivo, posiblemente porque MOEA-CKF explota su análisis previo (Prior Analysis) para compensar la pérdida de resolución con otros ajustes (p. ej. `wScale`, `sLower`).
- **Escala de pesos (`wScale`)**: los valores ganadores son considerablemente más dispersos, desde 6.7 (ds2-CKF) hasta 41.3 (ds1-CKF). `wScale` es, junto con `architecture`, el hiperparámetro más importante según el análisis fANOVA en 6 de los 8 estudios (ver más abajo), lo que es consistente con el rol que se le atribuye en la metodología: amplificar pesos codificados en [0,1] hasta una escala que efectivamente provoque disparos en la red — su valor óptimo depende fuertemente de la escala de entrada y de la arquitectura de cada dataset, por lo que no se observa un valor único "universal".
- **Banda de esparsidad (`sLower`/`sUpper`)**: las bandas ganadoras cubren un rango amplio, desde redes relativamente densas orientadas hacia baja esparsidad (`sLower≈0.52–0.61` en ds1 y ds4) hasta redes muy esparsas (`sLower≈0.78–0.91` en ds2 y ds3-CKF). No se observa una banda común entre datasets, lo que respalda la decisión metodológica de tratar `sLower`/`sGap` como hiperparámetros a optimizar en vez de fijarlos a priori.
- **Duración de evaluación (`evaluation_duration`)**: varía de 40 (ds3-NSGA) a 310 pasos (ds3-CKF) — el rango más amplio entre todos los hiperparámetros continuos/discretos, y sin relación aparente con el tamaño del dataset. Esto indica que la ventana temporal necesaria para que la evidencia se propague hasta las neuronas de salida es altamente dependiente tanto del dataset como del algoritmo evolutivo empleado, y confirma la pertinencia de haberla incluido en el espacio de búsqueda en lugar de fijarla.
- **Encoder ganador**: de los 8 estudios, 5 seleccionaron TTFS y 3 seleccionaron Poisson como configuración final; no hay una preferencia dominante de un encoder sobre el otro a nivel agregado, lo que sugiere que su conveniencia es específica de cada combinación algoritmo-dataset y refuerza la necesidad de tratar la arquitectura como una decisión estructural no relajable, tal como se planteó en la metodología.

## Importancia de hiperparámetros

El análisis fANOVA muestra un patrón claro y consistente con la sección anterior:

- En **ds2**, `architecture` domina absolutamente la varianza del HV (72–94 % según el algoritmo), reflejo directo de la brecha de rendimiento entre arquitecturas de una y dos salidas discutida más arriba (ver Figuras 3 y 4).
- En **ds1 y ds3**, `wScale` es el hiperparámetro más influyente en 3 de los 4 estudios (73–96 % de importancia); la excepción es S-NSGA-II en ds1, donde `wScale` (52 %) y `sLower` (39 %) se reparten casi toda la varianza explicada. Esto es consistente con el rol de `wScale` como amplificador necesario para que las conexiones codificadas en [0,1] logren provocar disparos. Esto se puede apreciar en las Figuras 11 a 14.

![Figura 11. Importancia de hiperparámetros — ds1 — MOEA-CKF](../bo_results/figures_tesis/importancia_MOEACKF_ds1.png)
*Figura 11. Importancia de hiperparámetros (fANOVA) — ds1, MOEA-CKF.*

![Figura 12. Importancia de hiperparámetros — ds1 — S-NSGA-II](../bo_results/figures_tesis/importancia_SNSGAII_ds1.png)
*Figura 12. Importancia de hiperparámetros (fANOVA) — ds1, S-NSGA-II.*

![Figura 13. Importancia de hiperparámetros — ds3 — MOEA-CKF](../bo_results/figures_tesis/importancia_MOEACKF_ds3.png)
*Figura 13. Importancia de hiperparámetros (fANOVA) — ds3, MOEA-CKF.*

![Figura 14. Importancia de hiperparámetros — ds3 — S-NSGA-II](../bo_results/figures_tesis/importancia_SNSGAII_ds3.png)
*Figura 14. Importancia de hiperparámetros (fANOVA) — ds3, S-NSGA-II.*

- En **ds4**, la importancia está más repartida y además difiere entre algoritmos: en MOEA-CKF domina `architecture` (34 %) seguido de `wScale` (22 %) y `dt` (10 %), mientras que en S-NSGA-II domina `sLower` (41 %) seguido de `wScale` (17 %) y `disC` (12 %). Es el dataset con más atributos de entrada (60) y también el que exhibe el espacio de decisión más "plano" y menos consistente entre algoritmos, donde ningún hiperparámetro concentra por sí solo la mayor parte de la mejora de HV. Los resultados mencionados se pueden observar en las Figuras 15 y 16.

![Figura 15. Importancia de hiperparámetros — ds4 — MOEA-CKF](../bo_results/figures_tesis/importancia_MOEACKF_ds4.png)
*Figura 15. Importancia de hiperparámetros (fANOVA) — ds4, MOEA-CKF.*

![Figura 16. Importancia de hiperparámetros — ds4 — S-NSGA-II](../bo_results/figures_tesis/importancia_SNSGAII_ds4.png)
*Figura 16. Importancia de hiperparámetros (fANOVA) — ds4, S-NSGA-II.*

En conjunto, estos resultados indican que la elección de arquitectura (cuando forma parte del espacio, como en ds2) y la escala de pesos (`wScale`) son sistemáticamente los factores de mayor impacto sobre el HV, mientras que los operadores de variación del SMOEA (`disC`, `disM`, `proM`, `disSM`, `proSM`) tienen, de manera individual, un efecto marginal frente a estos dos — aunque su ajuste conjunto sigue siendo necesario para alcanzar el óptimo, como lo evidencia que ninguno de los 8 estudios convergiera a los mismos valores para estos operadores.

## Limitación metodológica: asimetría del espacio de búsqueda en ds2

Un punto a señalar explícitamente para preservar la transparencia de la comparación entre datasets es que, a diferencia de ds1, ds3 y ds4 (donde los 60 trials se repartieron íntegramente entre 2 arquitecturas), en ds2 el mismo presupuesto de 60 trials se repartió entre **4** arquitecturas. En la práctica esto significa que el número efectivo de trials invertidos en la región competitiva del espacio (arquitecturas de 2 salidas) fue menor en ds2 que en los demás datasets — por ejemplo, S-NSGA-II en ds2 invirtió 46 de 60 trials en arquitecturas de 2 salidas (40+6), frente a los 60 de 60 en ds1, ds3 y ds4. Esto no invalida el resultado de ds2, ya que la configuración ganadora final sigue siendo comparable en naturaleza a la de los demás datasets (arquitectura de 2 salidas, exactitud, WTA); sin embargo, es una asimetría real en la eficiencia de la búsqueda que conviene explicitar como limitación al comparar el HV de ds2 con el de los demás datasets, más que como una diferencia atribuible únicamente a la dificultad intrínseca del dataset.

# Resultados y discusión — comparación MOEACKF vs. SNSGAII

> Metodología aplicada: la descrita en `docs/metodologia_comparacion.md`. Seis ángulos de
> comparación sobre cuatro fuentes de datos (ver §1.4 de la metodología): (1) hipervolumen (HV)
> principal, (2) frentes de Pareto (complejidad vs. error), (3) convergencia, (4) plataforma
> SNN (C++) vs. ANN (MATLAB/PlatEMO original), (5) evaluación bootstrap vs. completa, y
> (6) tiempo de cómputo. Los ángulos 1–3 y 5 están pareados por `seed` (Wilcoxon signed-rank +
> Â₁₂ pareado); el ángulo 4 (plataforma) no tiene seed compartida y usa Mann-Whitney U + Â₁₂ no
> pareado. Todo se analiza **por dataset por separado** (nunca se combinan datasets en un mismo
> test). Los p-valores mostrados en cada tabla son crudos; el veredicto final tras corrección de
> Holm-Bonferroni por familias (`docs/metodologia_comparacion.md` §6.5) está en
> `comparison/corrections_summary.csv` y se cita en el texto como "tras corrección".
> Accuracy de test no se incluye: es una limitación estructural de los datos archivados (ver
> §6 y la metodología §4), no un dato pendiente.
>
> Interpretación de Â₁₂ (Vargha & Delaney, 1998), sobre `|Â₁₂ − 0.5|`: `< 0.06` insignificante,
> `0.06–0.14` pequeño, `0.14–0.21` mediano, `≥ 0.21` grande.

## 1. Dataset 1 — Statlog Australian

### 1.1 Hipervolumen (HV)

#### Resultados

| Algoritmo | N | Media ± DE | Mediana [IQR] | Rango |
|---|---|---|---|---|
| MOEACKF | 31 | 0.8717 ± 0.0015 | 0.8721 [0.8704, 0.8721] | [0.8688, 0.8753] |
| SNSGAII | 31 | 0.8781 ± 0.0090 | 0.8767 [0.8702, 0.8856] | [0.8626, 0.8957] |

Wilcoxon signed-rank: **W = 96.0, p = 0.0022** (significativo, α=0.05; tras corrección
Holm-Bonferroni dentro de la familia `principal_snn`, p=0.0043, **sigue significativo**).
SNSGAII superó a MOEACKF en 21/31 corridas pareadas (67.7 %), sin empates. Â₁₂ = 0.323 → efecto
**grande** (|0.323−0.5| = 0.177, cercano al umbral de 0.21) a favor de SNSGAII.

#### Discusión

SNSGAII obtiene un HV mediano y medio estadísticamente superior a MOEACKF en este dataset, con
un tamaño de efecto grande — no es una diferencia marginal ni producto del azar de una muestra
chica. Sin embargo, la dispersión cuenta una segunda historia relevante para la tesis: MOEACKF
es notablemente más **consistente** (DE = 0.0015, IQR de apenas 0.0017) mientras que SNSGAII
varía casi seis veces más (DE = 0.0090) y su corrida más baja (0.8626) queda por debajo del
mínimo histórico de MOEACKF (0.8688). Es decir, SNSGAII gana en promedio y en la mayoría de las
corridas, pero a costa de mayor variabilidad entre ejecuciones — un trade-off exploración/
consistencia que conviene mencionar explícitamente en la tesis en vez de reportar solo la
mediana. Las secciones siguientes (frentes, convergencia) muestran de dónde sale esta diferencia.

### 1.2 Frentes de Pareto

**Solución de mejor accuracy por corrida** (fila de `f2_train_error` mínimo en cada frente):

| Indicador | MOEACKF | SNSGAII | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| Complejidad (frac. pesos activos) | 0.0107 ± 0.0044, mediana 0.0088 [0.0074, 0.0140] | 0.0288 ± 0.0214, mediana 0.0221 [0.0132, 0.0375] | 7.78e-05* | 0.226 |
| Error de entrenamiento | 0.1399 ± 0.0017, mediana 0.1395 [0.1395, 0.1413] | 0.1313 ± 0.0100, mediana 0.1322 [0.1232, 0.1395] | 3.73e-04* | 0.710 |
| Tamaño del frente | 5.74 ± 0.97, mediana 6 [5, 6] | 7.35 ± 3.05, mediana 7 [5, 9] | 0.0145* | 0.323 |

Las tres diferencias son significativas y se mantienen tras corrección Holm-Bonferroni (familia
`principal_snn`: p=3.11e-4, 1.12e-3 y 0.0145 respectivamente). SNSGAII paga ~2.5× más complejidad
en su mejor red (0.022 vs. 0.009) a cambio de un error meliente menor (0.132 vs. 0.140,
Â₁₂=0.710 grande) y retiene un frente algo más grande. Esto explica directamente la ventaja de
HV de §1.1: no es una mejora uniforme, sino un desplazamiento hacia soluciones más densas y más
precisas en el extremo de alta precisión del frente.

**Frente combinado y C-metric**: tamaño MOEACKF=6, SNSGAII=9. C(MOEACKF,SNSGAII)=0.556 (el 55.6%
de los puntos del frente combinado de SNSGAII están dominados por el de MOEACKF), C(SNSGAII,
MOEACKF)=0.167 (solo 16.7% en sentido inverso).

![Frente combinado ds1](fronts/ds1_combined_front.png)

Este resultado matiza el de HV: pese a tener menos puntos, el frente combinado de MOEACKF
domina más del frente combinado de SNSGAII que al revés — sus 6 soluciones son individualmente
difíciles de superar. La ventaja de SNSGAII en HV viene de cubrir una región más amplia del
espacio (9 puntos, incluida la solución de mejor accuracy de la tabla anterior), no de que cada
punto individual de SNSGAII sea mejor que su contraparte de MOEACKF.

**Error por banda de complejidad** (deciles sobre el pool de puntos crudos de ambos algoritmos):

| Banda de complejidad | MOEACKF error mediano (n) | SNSGAII error mediano (n) |
|---|---|---|
| (-0.001, 0.00147] | 0.4275 (58) | 0.4438 (50) |
| (0.00147, 0.00294] | 0.3043 (31) | 0.4393 (10) |
| (0.00294, 0.00441] | 0.3043 (18) | 0.4375 (6) |
| (0.00441, 0.00588] | 0.1431 (31) | 0.3152 (9) |
| (0.00588, 0.00735] | 0.1413 (14) | 0.2699 (21) |
| (0.00735, 0.0103] | 0.1413 (14) | 0.2319 (44) |
| (0.0103, 0.0132] | 0.1386 (4) | 0.1440 (28) |
| (0.0132, 0.0191] | 0.1395 (7) | 0.1377 (23) |
| (0.0191, 0.0882] | 0.1395 (1) | 0.1268 (37) |

![Error por banda ds1](fronts/ds1_complexity_bands.png)

MOEACKF domina claramente en las bandas de complejidad baja-media (hasta ~0.0132), con errores
notablemente menores que SNSGAII en el mismo rango (p.ej. banda 0.0044–0.0059: 0.143 vs. 0.315).
SNSGAII recién iguala y supera levemente a MOEACKF en la banda más alta (0.0191–0.0882), pero es
justamente ahí donde concentra la mayoría de sus puntos (37 de 189 puntos crudos, contra 1 solo
punto de MOEACKF). El HV agregado de SNSGAII se explica por operar consistentemente en esa
región de mayor complejidad y precisión, no por dominar en todo el espectro.

### 1.3 Convergencia

**Curva de convergencia** (HV mediano + banda P10–P90 vs. FE aproximado):

![Convergencia ds1](convergence/ds1_convergence.png)

MOEACKF converge mucho más rápido al principio (sube abruptamente y se estabiliza ya cerca de
FE≈3000–4000 en HV≈0.87), mientras SNSGAII arranca más bajo, sube de forma más gradual, y recién
supera el plateau de MOEACKF alrededor de FE≈11000–13000, terminando levemente más alto.

**Velocidad de convergencia** (FE hasta alcanzar el 95% del HV final de cada corrida):

| Grupo | Mediana [IQR] | Media ± DE | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 1212.1 [969.7, 1515.2] | 1380.3 ± 651.0 | — | — |
| SNSGAII | 5226.1 [5025.1, 6733.7] | 5874.5 ± 1377.7 | 9.31e-10* | 0.000 |

Diferencia significativa tras corrección (p=4.66e-9). Â₁₂=0.000: en las 31 corridas, MOEACKF
alcanza el 95% de su propio HV final antes que SNSGAII, en promedio ~4.3× más rápido. Esto
cuantifica lo visible en la curva: MOEACKF es sistemáticamente más rápido para llegar cerca de
su techo, pero ese techo es más bajo que el de SNSGAII.

### 1.4 Plataforma: SNN (C++) vs. ANN (MATLAB original)

| Indicador | MOEACKF: SNN | MOEACKF: ANN-MATLAB | p (Mann-Whitney) | Â₁₂ |
|---|---|---|---|---|
| HV | 0.8717 ± 0.0015, mediana 0.8721 | 0.8901 ± 0.0018, mediana 0.8891 | 1.37e-11* | 0.000 |
| Complejidad | 0.0107 ± 0.0044, mediana 0.0088 | 0.0302 ± 0.0244, mediana 0.0203 | 2.58e-07* | 0.120 |
| Accuracy | 0.8601 ± 0.0017, mediana 0.8605 | 0.8817 ± 0.0021, mediana 0.8804 | 5.06e-12* | 0.000 |

| Indicador | SNSGAII: SNN | SNSGAII: ANN-MATLAB | p (Mann-Whitney) | Â₁₂ |
|---|---|---|---|---|
| HV | 0.8781 ± 0.0090, mediana 0.8767 | 0.8967 ± 0.0025, mediana 0.8972 | 5.89e-11* | 0.016 |
| Complejidad | 0.0288 ± 0.0214, mediana 0.0221 | 0.1473 ± 0.2572, mediana 0.0265 | 1.33e-04* | 0.221 |
| Accuracy | 0.8687 ± 0.0100, mediana 0.8678 | 0.8892 ± 0.0024, mediana 0.8895 | 3.12e-11* | 0.011 |

Las seis comparaciones son significativas y se mantienen tras corrección Holm-Bonferroni (familia
`plataforma`, p ajustados entre 2.6e-11 y 9.4e-11 para HV/accuracy, y 2.58e-07/1.33e-04 para
complejidad). La implementación original en ANN (MATLAB) supera de forma consistente a la SNN en
HV y accuracy para ambos algoritmos, con una brecha de accuracy de ~0.02 en los dos casos — un
patrón esperable dado que la dinámica de disparo añade una capa de aproximación que una ANN
estándar no enfrenta. La red de mejor accuracy en ANN también tiende a ser más densa (0.030 vs.
0.011 para MOEACKF; para SNSGAII la mediana es similar pero la media y el rango se disparan por
algunas corridas MATLAB que llegan a redes casi densas).

### 1.5 Bootstrap vs. evaluación completa

| Algoritmo | Normal | Bootstrap | p (Wilcoxon) | Â₁₂ | Tiempo mediano (×) |
|---|---|---|---|---|---|
| MOEACKF | 0.8717 ± 0.0015, mediana 0.8721 | 0.8832 ± 0.0025, mediana 0.8830 | 9.31e-10* | 0.000 | 209 s → 1734 s (×8.3) |
| SNSGAII | 0.8781 ± 0.0090, mediana 0.8767 | 0.8870 ± 0.0076, mediana 0.8879 | 8.66e-05* | 0.258 | 940 s → 14195 s (×15.1) |

Ambas diferencias significativas tras corrección (familia `bootstrap`, p=1.86e-9 y 8.66e-5). El
remuestreo bootstrap mejora el HV para los dos algoritmos en este dataset, de forma casi
absoluta para MOEACKF (Â₁₂=0.000: bootstrap gana las 31 corridas) y con efecto grande también
para SNSGAII, a un costo computacional considerable (8.3× y 15.1× respectivamente).

### 1.6 Tiempo de cómputo

| Grupo | Mediana [IQR] | Media ± DE | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 285.6 [263.7, 288.2] s | 277.4 ± 15.4 s | — | — |
| SNSGAII | 259.6 [258.4, 260.5] s | 259.5 ± 2.5 s | 0.00153* | 0.867 |

Diferencia significativa (familia `tiempo`, m=1, sin corrección adicional). SNSGAII es más
rápido, con efecto grande (Â₁₂=0.867) y mucha menos variabilidad entre corridas (DE=2.5s vs.
15.4s de MOEACKF).

## 2. Dataset 2 — Climate

### 2.1 Hipervolumen (HV)

#### Resultados

| Algoritmo | N | Media ± DE | Mediana [IQR] | Rango |
|---|---|---|---|---|
| MOEACKF | 31 | 0.9276 ± 0.0036 | 0.9256 [0.9255, 0.9285] | [0.9235, 0.9361] |
| SNSGAII | 31 | 0.9273 ± 0.0015 | 0.9278 [0.9257, 0.9278] | [0.9256, 0.9299] |

Wilcoxon signed-rank: **W = 199.0, p = 0.347** (no significativo, α=0.05; tras corrección,
p=0.694, sigue sin serlo). SNSGAII superó a MOEACKF en 23/31 corridas (74.2 %), pero Â₁₂ = 0.258.

#### Discusión

Este dataset es el único caso donde el conteo de "victorias" por corrida (Â₁₂) y el resultado
de significancia del Wilcoxon **divergen**: SNSGAII gana más seguido, pero por márgenes mínimos
(diferencia mediana de solo 0.00016), mientras que las pocas corridas donde gana MOEACKF lo hace
por márgenes más grandes (su máximo, 0.9361, supera ampliamente el máximo de SNSGAII, 0.9299).
No hay evidencia suficiente de una diferencia real entre ambos algoritmos en HV para este
dataset: ambos convergen a una región muy similar y estrecha. Las secciones siguientes muestran
que, aun sin diferencia en HV, los frentes de cada algoritmo llegan ahí por caminos muy distintos.

### 2.2 Frentes de Pareto

**Solución de mejor accuracy por corrida**:

| Indicador | MOEACKF | SNSGAII | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| Complejidad (frac. pesos activos) | 0.0752 ± 0.1386, mediana 0.0190 [0.0083, 0.0363] | 0.0032 ± 0.0008, mediana 0.0036 [0.0024, 0.0036] | 9.31e-10* | 1.000 |
| Error de entrenamiento | 0.0786 ± 0.0038, mediana 0.0810 [0.0775, 0.0810] | 0.0792 ± 0.0017, mediana 0.0787 [0.0787, 0.0810] | 0.5345 | 0.565 |
| Tamaño del frente | 6.61 ± 1.17, mediana 7 [6, 7] | 3.42 ± 0.67, mediana 3 [3, 4] | 1.42e-06* | 0.984 |

Complejidad y tamaño del frente siguen siendo significativos tras corrección (p=4.66e-9 y
5.69e-6); el error de la mejor red no lo es (p=0.694, igual que el HV agregado). El resultado más
llamativo es la complejidad: SNSGAII encuentra una red de mejor accuracy **extremadamente rala**
(mediana 0.0036 — apenas ~3 pesos activos de 840) con Â₁₂=1.000 (gana las 31 corridas), logrando
un error estadísticamente indistinguible del de MOEACKF, que necesita una red bastante más densa
y variable (mediana 0.019, pero con corridas que llegan mucho más alto — DE=0.14, casi el doble
de la propia mediana). SNSGAII también retiene un frente mucho más chico (mediana 3 vs. 7).

**Frente combinado y C-metric**: tamaño MOEACKF=8, SNSGAII=4. C(MOEACKF,SNSGAII)=0.500,
C(SNSGAII,MOEACKF)=0.500 — cobertura perfectamente simétrica pese a la diferencia de tamaño.

![Frente combinado ds2](fronts/ds2_combined_front.png)

**Error por banda de complejidad**:

| Banda de complejidad | MOEACKF error mediano (n) | SNSGAII error mediano (n) |
|---|---|---|
| (-0.001, 0.00119] | 0.4167 (56) | 0.4213 (53) |
| (0.00119, 0.00238] | 0.1343 (31) | 0.0810 (31) |
| (0.00238, 0.00357] | 0.1319 (19) | 0.0787 (19) |
| (0.00357, 0.00476] | 0.0856 (30) | 0.0764 (3) |
| (0.00476, 0.00595] | 0.0833 (12) | — (0) |
| (0.00595, 0.0119] | 0.0810 (28) | — (0) |
| (0.0119, 0.557] | 0.0787 (29) | — (0) |

![Error por banda ds2](fronts/ds2_complexity_bands.png)

SNSGAII prácticamente no tiene puntos más allá de la banda 4 (0.0036–0.0048): sus 31 corridas
concentran casi toda su masa en las bandas de menor complejidad, y ahí su error es
consistentemente **menor** que el de MOEACKF en las mismas bandas (p.ej. banda 2:
0.081 vs. 0.134). MOEACKF, en cambio, reparte sus puntos en un rango de complejidad mucho más
amplio sin ganar precisión por ello en la zona rala. Climate parece ser un dataset donde
SNSGAII "se especializa" en soluciones ultra-ralas y lo hace bien, mientras MOEACKF explora más
sin una ganancia proporcional — coherente con que el HV agregado no distinga a los algoritmos
(§2.1): son perfiles de frente muy distintos que terminan en una región de calidad similar.

### 2.3 Convergencia

![Convergencia ds2](convergence/ds2_convergence.png)

**Velocidad de convergencia** (FE hasta 95% y 90% del HV final):

| Umbral | MOEACKF | SNSGAII | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| 95% del HV final | 73.97 ± 85.61, mediana 0 [0, 127.4] | 9.73 ± 39.82, mediana 0 [0, 0] | 4.75e-04* | 0.694 |
| 90% del HV final | 0 ± 0, mediana 0 [0, 0] | 0 ± 0, mediana 0 [0, 0] | 1.0 (sin diferencia) | 0.500 |

Climate es un caso límite: en las 31 corridas de **ambos** algoritmos, la población inicial
(generación 0) ya alcanza el 90% del HV final — no hay evidencia de diferencia alguna en ese
umbral (`x == y` en los 31 pares). Al 95% sí hay una diferencia significativa (tras corrección,
p=1.43e-3), con SNSGAII llegando más seguido ya en FE≈0. Esto confirma cuantitativamente lo que
sugiere la curva: Climate es un problema comparativamente "saturado" donde el margen de mejora
sobre la población inicial es pequeño para los dos algoritmos, así que "velocidad de convergencia"
aquí mide más bien qué tan rápido se estabiliza la búsqueda que una carrera real de optimización.

### 2.4 Plataforma: SNN (C++) vs. ANN (MATLAB original)

| Indicador | MOEACKF: SNN | MOEACKF: ANN-MATLAB | p (Mann-Whitney) | Â₁₂ |
|---|---|---|---|---|
| HV | 0.9276 ± 0.0036, mediana 0.9256 | 0.9760 ± 0.0034, mediana 0.9767 | 1.40e-11* | 0.000 |
| Complejidad | 0.0752 ± 0.1386, mediana 0.0190 | 0.0565 ± 0.0255, mediana 0.0487 | 2.37e-04* | 0.228 |
| Accuracy | 0.9214 ± 0.0038, mediana 0.9190 | 0.9760 ± 0.0039, mediana 0.9768 | 6.06e-12* | 0.000 |

| Indicador | SNSGAII: SNN | SNSGAII: ANN-MATLAB | p (Mann-Whitney) | Â₁₂ |
|---|---|---|---|---|
| HV | 0.9273 ± 0.0015, mediana 0.9278 | 0.9493 ± 0.0064, mediana 0.9477 | 1.39e-11* | 0.000 |
| Complejidad | 0.0032 ± 0.0008, mediana 0.0036 | 0.8367 ± 0.1867, mediana 0.9114 | 7.72e-12* | 0.000 |
| Accuracy | 0.9208 ± 0.0017, mediana 0.9213 | 0.9657 ± 0.0060, mediana 0.9653 | 8.20e-12* | 0.000 |

Todo significativo tras corrección. La brecha ANN-SNN en HV es la más grande de los cuatro
datasets para MOEACKF (0.976 vs. 0.928, ~5 puntos). La complejidad diverge fuertemente por
algoritmo: para MOEACKF la red ANN es apenas algo más rala que la SNN (0.056 vs. 0.075); para
SNSGAII, en cambio, la red ANN de mejor accuracy es **casi completamente densa** (mediana 0.911,
más del 90% de los pesos activos) frente a la red SNN ultra-rala de §2.2 (mediana 0.0036). Es la
divergencia de plataforma más extrema del estudio: para llegar a su mejor accuracy, SNSGAII-ANN
necesita casi todos los pesos activos, mientras que SNSGAII-SNN logra un error estadísticamente
similar (§2.2) con una fracción minúscula de la red. Sujeto a las salvedades de §6 (RNG e
implementación distintos), esto sugiere que el sustrato spiking podría favorecer soluciones
ralas para este algoritmo-dataset en particular — una hipótesis a confirmar, no una conclusión
causal con el diseño actual.

### 2.5 Bootstrap vs. evaluación completa

| Algoritmo | Normal | Bootstrap | p (Wilcoxon) | Â₁₂ | Tiempo mediano (×) |
|---|---|---|---|---|---|
| MOEACKF | 0.9276 ± 0.0036, mediana 0.9256 | 0.9404 ± 0.0015, mediana 0.9401 | 9.31e-10* | 0.000 | 1805 s → 8749 s (×4.8) |
| SNSGAII | 0.9273 ± 0.0015, mediana 0.9278 | 0.9400 ± 0.0013, mediana 0.9399 | 9.31e-10* | 0.000 | 202 s → 998 s (×4.9) |

Ambas diferencias significativas tras corrección (p=1.86e-9 en ambos casos). A diferencia del
HV entre algoritmos (sin diferencia en este dataset, §2.1), el bootstrap sí produce una mejora
grande y perfectamente consistente (Â₁₂=0.000 en los dos algoritmos: bootstrap gana las 31
corridas) con el menor costo relativo de tiempo de los cuatro datasets (×4.8–4.9, frente a
×5–15 en el resto).

### 2.6 Tiempo de cómputo

| Grupo | Mediana [IQR] | Media ± DE | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 230.5 [224.0, 235.2] s | 233.0 ± 20.3 s | — | — |
| SNSGAII | 216.7 [216.0, 217.0] s | 216.2 ± 1.7 s | 8.54e-04* | 0.867 |

Diferencia significativa, SNSGAII más rápido (efecto grande) y con mucha menor variabilidad.

## 3. Dataset 3 — German

### 3.1 Hipervolumen (HV)

#### Resultados

| Algoritmo | N | Media ± DE | Mediana [IQR] | Rango |
|---|---|---|---|---|
| MOEACKF | 31 | 0.7511 ± 0.0060 | 0.7520 [0.7452, 0.7560] | [0.7429, 0.7622] |
| SNSGAII | 31 | 0.7645 ± 0.0160 | 0.7643 [0.7547, 0.7754] | [0.7349, 0.7999] |

Wilcoxon signed-rank: **W = 50.0, p = 2.8 × 10⁻⁵** (altamente significativo; tras corrección,
p=8.47e-5, sigue significativo). SNSGAII superó a MOEACKF en 24/31 corridas (77.4 %). Â₁₂ = 0.226
→ efecto **grande** a favor de SNSGAII.

#### Discusión

Resultado contundente a favor de SNSGAII: la diferencia de medianas (0.0133) y de medias
(0.0134) es la segunda mayor de los cuatro datasets, y a diferencia del dataset 2, acá la
dirección del conteo de victorias y la magnitud de las diferencias coinciden. SNSGAII vuelve a
mostrar mayor dispersión que MOEACKF (DE=0.0160 vs. 0.0060), pero incluso su corrida más baja
(0.7349) queda cerca de la mediana de MOEACKF, por lo que la mayor variabilidad no compromete
tanto la fiabilidad del resultado.

### 3.2 Frentes de Pareto

**Solución de mejor accuracy por corrida**:

| Indicador | MOEACKF | SNSGAII | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| Complejidad (frac. pesos activos) | 0.0182 ± 0.0216, mediana 0.0065 [0.0042, 0.0269] | 0.0414 ± 0.0360, mediana 0.0296 [0.0204, 0.0468] | 0.00915* | 0.226 |
| Error de entrenamiento | 0.2734 ± 0.0068, mediana 0.2725 [0.2681, 0.2800] | 0.2583 ± 0.0179, mediana 0.2587 [0.2462, 0.2694] | 8.90e-05* | 0.790 |
| Tamaño del frente | 5.52 ± 2.11, mediana 5 [4, 7] | 9.13 ± 2.51, mediana 9 [8, 10.5] | 1.85e-05* | 0.161 |

Los tres indicadores significativos tras corrección (p=9.15e-3, 1.78e-4, 7.41e-5). Mismo patrón
cualitativo que el dataset 1: SNSGAII paga más complejidad en su mejor red (0.030 vs. 0.007) a
cambio de menor error (0.259 vs. 0.273, efecto grande) y retiene un frente sustancialmente más
grande (9 vs. 5) — de los cuatro datasets, este es el más consistente con la narrativa "SNSGAII
gana explorando redes más densas y numerosas".

**Frente combinado y C-metric**: tamaño MOEACKF=6, SNSGAII=13. C(MOEACKF,SNSGAII)=0.308,
C(SNSGAII,MOEACKF)=0.333 — cobertura mutua similar pese a que el frente de SNSGAII es más del
doble de grande.

![Frente combinado ds3](fronts/ds3_combined_front.png)

**Error por banda de complejidad**:

| Banda de complejidad | MOEACKF error mediano (n) | SNSGAII error mediano (n) |
|---|---|---|
| (-0.001, 0.000926] | 0.4350 (55) | 0.4456 (56) |
| (0.000926, 0.00185] | 0.2825 (31) | 0.2988 (31) |
| (0.00185, 0.00278] | 0.2800 (20) | — (0) |
| (0.00278, 0.00556] | 0.2800 (21) | 0.2863 (31) |
| (0.00556, 0.00741] | 0.2712 (4) | 0.2775 (26) |
| (0.00741, 0.0111] | 0.2712 (7) | 0.2700 (41) |
| (0.0111, 0.0176] | 0.2750 (5) | 0.2662 (39) |
| (0.0176, 0.0315] | 0.2712 (11) | 0.2562 (31) |
| (0.0315, 0.138] | 0.2700 (17) | 0.2575 (28) |

![Error por banda ds3](fronts/ds3_complexity_bands.png)

En las bandas de menor complejidad ambos algoritmos son similares (incluso MOEACKF algo mejor);
a partir de la banda 5 (complejidad ≥0.0056) SNSGAII pasa a tener error consistentemente menor,
y la brecha se ensancha hacia las bandas más altas (0.0176–0.0315: 0.256 vs. 0.271). SNSGAII
aprovecha mejor la complejidad adicional en este dataset, lo que es coherente con su frente
combinado más grande (13 vs. 6 puntos) y con la ventaja de HV de §3.1.

### 3.3 Convergencia

![Convergencia ds3](convergence/ds3_convergence.png)

La curva muestra a SNSGAII arrancando mucho más bajo (~0.654) y subiendo con pendiente
pronunciada, cruzando a MOEACKF alrededor de FE≈4500–5000 y terminando por encima (~0.764 vs.
~0.752 de MOEACKF, que se aplana muy temprano).

**Velocidad de convergencia** (FE hasta 95% del HV final):

| Grupo | Mediana [IQR] | Media ± DE | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0 [0, 0] | 37.4 ± 81.4 | — | — |
| SNSGAII | 3408.5 [2456.1, 3909.8] | 3250.1 ± 953.8 | 9.31e-10* | 0.000 |

Diferencia significativa tras corrección (p=4.66e-9). Resultado paradójico pero coherente con la
curva: aunque SNSGAII logra el HV **final** más alto (§3.1), es MOEACKF quien llega casi
instantáneamente (mediana FE≈0) al 95% de su propio techo — más bajo, pero alcanzado de
inmediato — mientras SNSGAII necesita ~3400 FE (8.5% del presupuesto total de 40000) para llegar
al 95% de su techo, más alto. Es un ejemplo claro del trade-off velocidad/calidad: MOEACKF se
estabiliza rápido en un techo menor, SNSGAII tarda más en llegar a uno mayor.

### 3.4 Plataforma: SNN (C++) vs. ANN (MATLAB original)

| Indicador | MOEACKF: SNN | MOEACKF: ANN-MATLAB | p (Mann-Whitney) | Â₁₂ |
|---|---|---|---|---|
| HV | 0.7511 ± 0.0060, mediana 0.7520 | 0.8100 ± 0.0024, mediana 0.8098 | 1.40e-11* | 0.000 |
| Complejidad | 0.0182 ± 0.0216, mediana 0.0065 | 0.0509 ± 0.0221, mediana 0.0413 | 1.26e-06* | 0.142 |
| Accuracy | 0.7266 ± 0.0068, mediana 0.7275 | 0.7927 ± 0.0027, mediana 0.7925 | 1.20e-11* | 0.000 |

| Indicador | SNSGAII: SNN | SNSGAII: ANN-MATLAB | p (Mann-Whitney) | Â₁₂ |
|---|---|---|---|---|
| HV | 0.7645 ± 0.0160, mediana 0.7643 | 0.8118 ± 0.0025, mediana 0.8119 | 1.40e-11* | 0.000 |
| Complejidad | 0.0414 ± 0.0360, mediana 0.0296 | 0.6659 ± 0.3744, mediana 0.8751 | 6.89e-10* | 0.044 |
| Accuracy | 0.7417 ± 0.0179, mediana 0.7412 | 0.7987 ± 0.0036, mediana 0.7975 | 1.33e-11* | 0.000 |

Todo significativo tras corrección. Mismo patrón que el dataset 2: la red ANN de MOEACKF es
moderadamente más densa que la SNN (0.051 vs. 0.018), mientras que la red ANN de SNSGAII vuelve
a ser casi totalmente densa (mediana 0.875) frente a la SNN mucho más rala (0.030). El gap de
complejidad ANN-vs-SNN de SNSGAII es sistemáticamente mucho mayor que el de MOEACKF en los
cuatro datasets (ver síntesis, §5) — un patrón que se repite, no un caso aislado de Climate.

### 3.5 Bootstrap vs. evaluación completa

| Algoritmo | Normal | Bootstrap | p (Wilcoxon) | Â₁₂ | Tiempo mediano (×) |
|---|---|---|---|---|---|
| MOEACKF | 0.7511 ± 0.0060, mediana 0.7520 | 0.7568 ± 0.0031, mediana 0.7567 | 1.95e-04* | 0.226 | 1207 s → 6222 s (×5.2) |
| SNSGAII | 0.7645 ± 0.0160, mediana 0.7643 | 0.7580 ± 0.0090, mediana 0.7549 | 0.0664 | 0.710 | 914 s → 6854 s (×7.5) |

MOEACKF mejora significativamente con bootstrap (tras corrección, p=3.91e-4), pero SNSGAII
**no** (p=0.066, no sobrevive ni al umbral crudo de 0.05): German es el primer dataset donde el
bootstrap no ayuda de forma uniforme. Más aún, la dirección de Â₁₂=0.710 para SNSGAII sugiere
que la evaluación normal iguala o supera a la bootstrap más seguido que al revés, aunque sin
significancia estadística — no se puede afirmar que el bootstrap perjudique a SNSGAII en este
dataset, pero tampoco que lo ayude, pese a un costo de tiempo considerable (×7.5).

### 3.6 Tiempo de cómputo

| Grupo | Mediana [IQR] | Media ± DE | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 936.6 [895.7, 956.1] s | 928.7 ± 37.1 s | — | — |
| SNSGAII | 871.3 [858.9, 883.0] s | 867.7 ± 20.7 s | 4.27e-04* | 0.933 |

Diferencia significativa, SNSGAII más rápido, efecto grande.

## 4. Dataset 4 — Sonar

### 4.1 Hipervolumen (HV)

#### Resultados

| Algoritmo | N | Media ± DE | Mediana [IQR] | Rango |
|---|---|---|---|---|
| MOEACKF | 31 | 0.8224 ± 0.0302 | 0.8274 [0.8141, 0.8432] | [0.7494, 0.8618] |
| SNSGAII | 31 | 0.7808 ± 0.0374 | 0.7812 [0.7601, 0.8059] | [0.6840, 0.8435] |

Wilcoxon signed-rank: **W = 60.0, p = 8.7 × 10⁻⁵** (altamente significativo; tras corrección,
p=2.60e-4, sigue significativo). MOEACKF superó a SNSGAII en 22/31 corridas (71.0 %). Â₁₂ = 0.710
→ efecto **grande** a favor de MOEACKF.

#### Discusión

Sonar es el único dataset donde se invierte el patrón de los tres anteriores: MOEACKF supera a
SNSGAII de forma significativa y con la mayor diferencia absoluta de HV observada en todo el
estudio (Δmedia = 0.0417). También es el dataset con mayor varianza para ambos algoritmos
(DE≈0.03–0.04), consistente con que Sonar es, de los cuatro, el de mayor dimensionalidad (60
features) y por lo tanto el de paisaje de optimización más difícil. Las secciones siguientes
muestran que acá MOEACKF no gana solo en el agregado: domina en casi todos los indicadores
individuales.

### 4.2 Frentes de Pareto

**Solución de mejor accuracy por corrida**:

| Indicador | MOEACKF | SNSGAII | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| Complejidad (frac. pesos activos) | 0.0792 ± 0.0953, mediana 0.0429 [0.0208, 0.0847] | 0.0188 ± 0.0141, mediana 0.0159 [0.0077, 0.0270] | 1.07e-04* | 0.742 |
| Error de entrenamiento | 0.1876 ± 0.0327, mediana 0.1796 [0.1677, 0.1976] | 0.2403 ± 0.0417, mediana 0.2395 [0.2126, 0.2635] | 4.77e-05* | 0.194 |
| Tamaño del frente | 11.26 ± 3.28, mediana 12 [9, 13] | 8.71 ± 3.44, mediana 8 [7, 10.5] | 0.0166* | 0.645 |

Todo significativo tras corrección (p=2.60e-4, 1.91e-4, 0.0166). A diferencia de los datasets 1
y 3, acá MOEACKF **no** cede complejidad por precisión: su red de mejor accuracy es a la vez más
densa (0.043 vs. 0.016) y más precisa (error 0.180 vs. 0.240) que la de SNSGAII, y además retiene
un frente más grande (12 vs. 8). Es el único dataset donde un algoritmo domina de forma tan
directa en los tres indicadores del frente a la vez.

**Frente combinado y C-metric**: tamaño MOEACKF=14, SNSGAII=14 (iguales). C(MOEACKF,SNSGAII)=
**0.857**, C(SNSGAII,MOEACKF)=0.143 — el resultado más asimétrico de los cuatro datasets.

![Frente combinado ds4](fronts/ds4_combined_front.png)

El 85.7% de los puntos del frente combinado de SNSGAII están dominados por el de MOEACKF,
mientras que solo el 14.3% ocurre en sentido inverso, pese a que ambos frentes combinados tienen
el mismo tamaño (14 puntos). Es la evidencia más contundente del estudio de que un algoritmo
domina "punto a punto", no solo en el resumen agregado de HV.

**Error por banda de complejidad**:

| Banda de complejidad | MOEACKF error mediano (n) | SNSGAII error mediano (n) |
|---|---|---|
| (-0.001, 0.000397] | 0.3593 (59) | 0.3862 (38) |
| (0.000397, 0.00159] | 0.3503 (2) | 0.3743 (28) |
| (0.00159, 0.00317] | 0.2934 (23) | 0.3234 (51) |
| (0.00317, 0.00437] | 0.2754 (22) | 0.2874 (39) |
| (0.00437, 0.00595] | 0.2425 (26) | 0.2695 (33) |
| (0.00595, 0.00913] | 0.2216 (23) | 0.2455 (32) |
| (0.00913, 0.0163] | 0.2156 (29) | 0.2275 (29) |
| (0.0163, 0.0329] | 0.2096 (48) | 0.2635 (13) |
| (0.0329, 0.0787] | 0.2156 (55) | 0.2036 (7) |
| (0.0787, 0.386] | 0.2216 (62) | — (0) |

![Error por banda ds4](fronts/ds4_complexity_bands.png)

MOEACKF tiene error menor que SNSGAII en casi todo el espectro de complejidad (bandas 2 a 8),
con SNSGAII apenas adelante en una banda alta angosta (0.0329–0.0787, con solo 7 puntos de
SNSGAII, muestra chica) y MOEACKF siendo el único que alcanza la banda más alta (0.0787–0.386,
62 puntos). La ventaja de MOEACKF en Sonar es amplia y no se concentra en una sola región del
espacio de complejidad, a diferencia de lo que ocurre con SNSGAII en los datasets 1 y 3.

### 4.3 Convergencia

![Convergencia ds4](convergence/ds4_convergence.png)

MOEACKF se mantiene por encima de SNSGAII durante todo el presupuesto, y todavía sube lentamente
cerca de FE=40000 (no está del todo aplanado), mientras SNSGAII se estabiliza antes
(FE≈20000–25000) en un nivel claramente inferior.

**Velocidad de convergencia** (FE hasta 95% del HV final):

| Grupo | Mediana [IQR] | Media ± DE | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 13626.4 [10622.7, 18901.1] | 14363.7 ± 5291.3 | — | — |
| SNSGAII | 5614.0 [4110.3, 6416.0] | 6089.4 ± 3580.2 | 9.16e-06* | 0.903 |

Diferencia significativa tras corrección (p=4.58e-5). Acá se invierte la relación velocidad/
calidad vista en el dataset 3: el algoritmo con **menor** HV final (SNSGAII) converge más rápido
a su propio techo, mientras MOEACKF (mayor HV final) tarda ~2.4× más (mediana FE 13626 vs. 5614)
en llegar al 95% del suyo, y según la curva todavía no se estabiliza del todo al final del
presupuesto. Esto sugiere que MOEACKF podría beneficiarse de un presupuesto de FE aún mayor en
problemas de alta dimensionalidad como Sonar — un experimento natural para trabajo futuro.

### 4.4 Plataforma: SNN (C++) vs. ANN (MATLAB original)

| Indicador | MOEACKF: SNN | MOEACKF: ANN-MATLAB | p (Mann-Whitney) | Â₁₂ |
|---|---|---|---|---|
| HV | 0.8224 ± 0.0302, mediana 0.8274 | 0.8744 ± 0.0141, mediana 0.8740 | 5.89e-11* | 0.016 |
| Complejidad | 0.0792 ± 0.0953, mediana 0.0429 | 0.0141 ± 0.0131, mediana 0.0097 | 2.40e-06* | 0.849 |
| Accuracy | 0.8124 ± 0.0327, mediana 0.8204 | 0.8629 ± 0.0158, mediana 0.8623 | 4.93e-10* | 0.041 |

| Indicador | SNSGAII: SNN | SNSGAII: ANN-MATLAB | p (Mann-Whitney) | Â₁₂ |
|---|---|---|---|---|
| HV | 0.7808 ± 0.0374, mediana 0.7812 | 0.9086 ± 0.0132, mediana 0.9143 | 1.40e-11* | 0.000 |
| Complejidad | 0.0188 ± 0.0141, mediana 0.0159 | 0.4621 ± 0.4705, mediana 0.1503 | 5.74e-07* | 0.132 |
| Accuracy | 0.7597 ± 0.0417, mediana 0.7605 | 0.9150 ± 0.0163, mediana 0.9102 | 1.32e-11* | 0.000 |

Todo significativo tras corrección. Para SNSGAII, la brecha ANN-SNN de HV (0.909 vs. 0.781,
~0.13) y de accuracy (0.915 vs. 0.760, ~0.155) es la más grande de los cuatro datasets — la alta
dimensionalidad de Sonar parece penalizar más al sustrato SNN que al ANN, al menos para este
algoritmo. La complejidad de MOEACKF es la **única reversión** de las ocho combinaciones
plataforma×algoritmo del estudio: acá la red SNN de mejor accuracy (0.079) es más densa que la
ANN (0.014) — en el resto de los casos donde hay una brecha clara de complejidad, es la ANN la
más densa. Vale la pena señalar esta excepción explícitamente en la tesis en vez de darla por
parte del patrón general.

### 4.5 Bootstrap vs. evaluación completa

| Algoritmo | Normal | Bootstrap | p (Wilcoxon) | Â₁₂ | Tiempo mediano (×) |
|---|---|---|---|---|---|
| MOEACKF | 0.8224 ± 0.0302, mediana 0.8274 | 0.8334 ± 0.0233, mediana 0.8356 | 0.2241 | 0.484 | 1832 s → 9202 s (×5.0) |
| SNSGAII | 0.7808 ± 0.0374, mediana 0.7812 | 0.8020 ± 0.0315, mediana 0.8066 | 0.0132* | 0.290 | 1730 s → 11884 s (×6.9) |

Patrón inverso al del dataset 3: acá el bootstrap ayuda significativamente a SNSGAII (tras
corrección, p=0.0264) pero no muestra efecto en MOEACKF (p=0.224, Â₁₂=0.484, prácticamente una
moneda al aire). Entre los datasets 3 y 4 queda claro que el beneficio del bootstrap depende de
la combinación dataset×algoritmo, no de una regla simple ("siempre ayuda" o "ayuda siempre al
mismo algoritmo") — se documenta como pregunta abierta en las limitaciones (§6).

### 4.6 Tiempo de cómputo

| Grupo | Mediana [IQR] | Media ± DE | p (Wilcoxon) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 715.1 [525.9, 865.2] s | 747.4 ± 259.5 s | — | — |
| SNSGAII | 295.5 [294.0, 298.2] s | 295.7 ± 4.1 s | 6.10e-05* | 1.000 |

Diferencia significativa y la más extrema del estudio: Â₁₂=1.000, SNSGAII más rápido en las 15
corridas pareadas, sin una sola excepción. MOEACKF es aquí ~2.4× más lento (715 vs. 296 s
medianos) y con una variabilidad de tiempo mucho mayor (DE=259.5s, muy por encima del resto de
los datasets) — consistente con que su mecanismo de fusión de conocimiento entre escalas
(SVD, análisis de dispersión) escala peor con la dimensionalidad (60 features) que los
operadores más simples de SNSGAII.

## 5. Síntesis entre datasets (descriptiva, no un test adicional)

Siguiendo el requisito de no mezclar datasets en una misma prueba estadística, esta sección es
puramente **descriptiva** (en el espíritu de Demšar, 2006).

**HV principal**: SNSGAII gana en 2 de 4 datasets (ds1, ds3), MOEACKF en 1 (ds4), y en 1 no hay
diferencia significativa (ds2) — sin cambios respecto al análisis original. Se mantiene la
hipótesis (no probada estadísticamente con solo 4 puntos) de que SNSGAII rinde mejor en datasets
de menor dimensionalidad (14 y 24 features) y MOEACKF en el de mayor dimensionalidad (60
features), con el de dimensionalidad intermedia (18 features) sin diferencia. Los ángulos nuevos
refuerzan esta lectura con más textura:

- **Frentes**: en ds1 y ds3 (los que gana SNSGAII), su solución de mejor accuracy es más densa
  y más precisa que la de MOEACKF; en ds4 (que gana MOEACKF) el patrón es idéntico pero con los
  roles invertidos — MOEACKF es el que gana en densidad y precisión a la vez. El C-metric del
  frente combinado es más parejo en ds2/ds3 (0.50/0.50, 0.31/0.33) y muy asimétrico en ds4
  (0.86/0.14 a favor de MOEACKF) — el dataset con la victoria de HV más contundente es también
  el de dominancia punto-a-punto más contundente.
- **Convergencia**: en los tres datasets donde hay una diferencia clara de HV (ds1, ds3, ds4), el
  algoritmo que **pierde** en HV final es sistemáticamente el que converge más rápido a su propio
  techo (Â₁₂ de velocidad ≈0.00 para MOEACKF en ds1/ds3, ≈0.90 para MOEACKF en ds4, siempre a
  favor del algoritmo de menor HV). Plateau rápido en un techo bajo vs. ascenso lento a un techo
  alto parece ser un trade-off recurrente en este estudio, no ruido de una sola corrida.
- **Tiempo**: SNSGAII es más rápido en los cuatro datasets, pero la brecha crece marcadamente con
  la dimensionalidad — de ~6–10% en ds1–ds3 (14–24 features) a ~2.4× en ds4 (60 features,
  Â₁₂=1.000). Esto es consistente con que MOEACKF consume más FE por generación que SNSGAII (ver
  metodología §3.3) y que ese costo adicional escala con el tamaño del problema.
- **Plataforma**: el hallazgo más uniforme de todo el análisis — la ANN original (MATLAB) supera
  a la SNN en HV y accuracy en las **8 de 8** combinaciones dataset×algoritmo, siempre
  significativo tras corrección. La brecha de accuracy es más chica en ds1 (~0.02) y más grande
  en ds4 para SNSGAII (~0.155). El gap de complejidad ANN-vs-SNN de SNSGAII (extremo en ds2/ds3,
  moderado en ds1/ds4) es sistemáticamente mayor que el de MOEACKF (siempre moderado) — un patrón
  algoritmo-dependiente que se repite entre datasets y podría merecer investigación aparte.
- **Bootstrap**: mejora el HV de forma clara y consistente en ds1 y ds2 (ambos algoritmos, efecto
  grande en los dos), pero es mixto en ds3 (solo ayuda a MOEACKF) y ds4 (solo ayuda a SNSGAII) —
  no hay una regla simple, y el costo de tiempo (×4.8 a ×15.1) es siempre sustancial
  independientemente de si el HV mejora o no.

También se mantiene la observación original: en 3 de los 4 datasets (ds1, ds2, ds3) MOEACKF
muestra menor varianza entre corridas que SNSGAII en HV, incluso cuando pierde en mediana — es
decir, MOEACKF tiende a ser más consistente/predecible. Si el criterio de la tesis valora
reproducibilidad además de desempeño puntual, esto sigue siendo relevante más allá de qué
algoritmo "gana" cada test.

## 6. Limitaciones de este análisis

- **Accuracy de test**: limitación **estructural**, no pendiente. El 100% de las corridas
  archivadas en `normal_results` y `bootstrap_results` usan arquitectura de 2 salidas (WTA), y el
  código solo exporta `*_spikes.csv` (necesarios para reconstruir accuracy de test) cuando
  `nOutputs == 1`. No se puede, con los datos actuales, medir generalización directamente; el
  análisis de frentes (§1.2–§4.2 de este documento) es el sustituto que sí es posible, mirando la
  relación complejidad-error de todo el frente en vez de un accuracy de test puntual.
- **Datasets 5 y 6 (Iris, Wine) no tienen resultados corridos** en ninguna de las cuatro fuentes
  — las conclusiones de este documento cubren únicamente Statlog Australian, Climate, German y
  Sonar.
- **Tiempo de cómputo**: `normal_results` y `bootstrap_results` siguen sin comparación válida de
  `time_s` (paralelismo/hilos no controlados entre tandas de ejecución). Solo `time_experiments`
  (§1.6–§4.6) permite una comparación de tiempo válida, pero corrió con hiperparámetros
  compartidos (no afinados por algoritmo) — sus conclusiones de tiempo no deben mezclarse con las
  conclusiones de calidad (HV) del resto del análisis, que sí usa hiperparámetros afinados.
- **Comparación de plataforma (SNN vs. ANN)**: MATLAB/PlatEMO original y el port en C++ difieren
  en más que el modelo de neurona — implementación y generador de números aleatorios distintos, y
  sin columna de seed en los datos de MATLAB no hay forma de aislar esa diferencia. Los resultados
  de §1.4–§4.4 deben leerse como evidencia direccional (la ANN supera consistentemente a la SNN),
  no como una atribución causal limpia al hecho de ser spiking.
- **Bootstrap**: el régimen de evaluación cambia el ruido de la señal de fitness; el efecto sobre
  el HV final es mixto entre dataset y algoritmo (§3.5, §4.5) sin un patrón simple identificado —
  queda como pregunta abierta si existe alguna interacción entre el remuestreo y los mecanismos
  específicos de cada algoritmo (p.ej. el filtro de Kalman de MOEACKF).
- **C-metric sobre frentes combinados pequeños**: los frentes combinados de 31 corridas quedan en
  4–14 puntos tras deduplicar y filtrar no-dominados — la resolución de la métrica de cobertura es
  limitada con conjuntos tan chicos; se reporta como magnitud aproximada, no como proporción de
  alta precisión.
- **FE aproximado en convergencia**: `convergence.csv` no registra FE por fila, solo `generation`,
  que no es comparable entre algoritmos (MOEACKF consume más FE por generación que SNSGAII). El
  eje FE de las curvas y umbrales de velocidad (§1.3–§4.3) es una aproximación por interpolación
  lineal dentro de cada corrida, no un valor exacto por fila.

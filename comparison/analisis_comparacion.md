# Resultados y discusión — comparación MOEACKF vs. SNSGAII

> Metodología aplicada: la descrita en `docs/metodologia_comparacion.md`. Diseño pareado por
> `seed` (N=31 corridas por algoritmo y dataset, seeds 100–130 compartidas entre algoritmos),
> prueba de Wilcoxon signed-rank sobre el hipervolumen (HV) **por dataset por separado**
> (nunca se combinan datasets en un mismo test), y tamaño del efecto Â₁₂ de Vargha-Delaney.
> Accuracy de test no se incluye: no existen archivos `*_spikes.csv` en los resultados
> disponibles (ver limitaciones, §6).
>
> Interpretación de Â₁₂ (Vargha & Delaney, 1998), sobre `|Â₁₂ − 0.5|`: `< 0.06` insignificante,
> `0.06–0.14` pequeño, `0.14–0.21` mediano, `≥ 0.21` grande.

## 1. Dataset 1 — Statlog Australian

### Resultados

| Algoritmo | N | Media ± DE | Mediana [IQR] | Rango |
|---|---|---|---|---|
| MOEACKF | 31 | 0.8717 ± 0.0015 | 0.8721 [0.8704, 0.8721] | [0.8688, 0.8753] |
| SNSGAII | 31 | 0.8781 ± 0.0090 | 0.8767 [0.8702, 0.8856] | [0.8626, 0.8957] |

Wilcoxon signed-rank: **W = 96.0, p = 0.0022** (significativo, α=0.05). SNSGAII superó a
MOEACKF en 21/31 corridas pareadas (67.7 %), sin empates. Â₁₂ = 0.323 → efecto **grande**
(|0.323−0.5| = 0.177, cercano al umbral de 0.21) a favor de SNSGAII.

### Discusión

SNSGAII obtiene un HV mediano y medio estadísticamente superior a MOEACKF en este dataset, con
un tamaño de efecto grande — no es una diferencia marginal ni producto del azar de una muestra
chica. Sin embargo, la dispersión cuenta una segunda historia relevante para la tesis: MOEACKF
es notablemente más **consistente** (DE = 0.0015, IQR de apenas 0.0017) mientras que SNSGAII
varía casi seis veces más (DE = 0.0090) y su corrida más baja (0.8626) queda por debajo del
mínimo histórico de MOEACKF (0.8688). Es decir, SNSGAII gana en promedio y en la mayoría de las
corridas, pero a costa de mayor variabilidad entre ejecuciones — un trade-off exploración/
consistencia que conviene mencionar explícitamente en la tesis en vez de reportar solo la
mediana.

## 2. Dataset 2 — Climate

### Resultados

| Algoritmo | N | Media ± DE | Mediana [IQR] | Rango |
|---|---|---|---|---|
| MOEACKF | 31 | 0.9276 ± 0.0036 | 0.9256 [0.9255, 0.9285] | [0.9235, 0.9361] |
| SNSGAII | 31 | 0.9273 ± 0.0015 | 0.9278 [0.9257, 0.9278] | [0.9256, 0.9299] |

Wilcoxon signed-rank: **W = 199.0, p = 0.347** (no significativo, α=0.05). SNSGAII superó a
MOEACKF en 23/31 corridas (74.2 %), pero Â₁₂ = 0.258, que por magnitud correspondería a un
efecto grande si se lo mirara aislado.

### Discusión

Este dataset es el único caso donde el conteo de "victorias" por corrida (Â₁₂) y el resultado
de significancia del Wilcoxon **divergen**, y vale la pena explicarlo porque es un punto
metodológico importante: SNSGAII gana más seguido, pero por márgenes mínimos (diferencia
mediana de solo 0.00016; diferencia de medias de 0.00027, prácticamente cero), mientras que las
pocas corridas donde gana MOEACKF lo hace por márgenes más grandes (su máximo, 0.9361, supera
ampliamente el máximo de SNSGAII, 0.9299). El test de Wilcoxon pondera los rangos de las
diferencias por **magnitud**, no solo por signo — por eso un algoritmo puede "ganar" en
proporción de corridas sin que eso alcance significancia estadística. La conclusión correcta
aquí no es "SNSGAII es mejor porque ganó más corridas", sino que **no hay evidencia suficiente
de una diferencia real** entre ambos algoritmos en este dataset: ambos convergen a una región
de HV muy similar y estrecha (rangos de apenas ~0.01 para ambos), sugiriendo que Climate es un
problema comparativamente “saturado” donde el margen de mejora entre algoritmos es pequeño.

## 3. Dataset 3 — German

### Resultados

| Algoritmo | N | Media ± DE | Mediana [IQR] | Rango |
|---|---|---|---|---|
| MOEACKF | 31 | 0.7511 ± 0.0060 | 0.7520 [0.7452, 0.7560] | [0.7429, 0.7622] |
| SNSGAII | 31 | 0.7645 ± 0.0160 | 0.7643 [0.7547, 0.7754] | [0.7349, 0.7999] |

Wilcoxon signed-rank: **W = 50.0, p = 2.8 × 10⁻⁵** (altamente significativo, α=0.05). SNSGAII
superó a MOEACKF en 24/31 corridas (77.4 %). Â₁₂ = 0.226 → efecto **grande**
(|0.226−0.5| = 0.274) a favor de SNSGAII.

### Discusión

Este es el resultado más contundente a favor de SNSGAII entre los cuatro datasets: la
diferencia es significativa, el tamaño del efecto es grande, y —a diferencia del dataset 2— acá
la dirección del conteo de victorias y la magnitud de las diferencias **coinciden**: la
diferencia de medianas (0.0133) y de medias (0.0134) es la segunda mayor de los cuatro
datasets. SNSGAII vuelve a mostrar mayor dispersión que MOEACKF (DE = 0.0160 vs. 0.0060), pero
a diferencia del dataset 1, acá incluso su corrida más baja (0.7349) queda cerca de la mediana
de MOEACKF, por lo que la mayor variabilidad no compromete tanto la fiabilidad del resultado:
SNSGAII es sistemáticamente mejor en este dataset, con margen amplio y consistente.

## 4. Dataset 4 — Sonar

### Resultados

| Algoritmo | N | Media ± DE | Mediana [IQR] | Rango |
|---|---|---|---|---|
| MOEACKF | 31 | 0.8224 ± 0.0302 | 0.8274 [0.8141, 0.8432] | [0.7494, 0.8618] |
| SNSGAII | 31 | 0.7808 ± 0.0374 | 0.7812 [0.7601, 0.8059] | [0.6840, 0.8435] |

Wilcoxon signed-rank: **W = 60.0, p = 8.7 × 10⁻⁵** (altamente significativo, α=0.05). MOEACKF
superó a SNSGAII en 22/31 corridas (71.0 %). Â₁₂ = 0.710 → efecto **grande** — justo en el
umbral que Vargha & Delaney consideran "grande" — a favor de MOEACKF.

### Discusión

Sonar es el único dataset donde **se invierte** el patrón de los tres anteriores: MOEACKF
supera a SNSGAII de forma significativa y con la mayor diferencia absoluta de HV observada en
todo el estudio (Δmedia = 0.0417, más del triple que en el dataset 3). También es el dataset
con mayor varianza para ambos algoritmos (DE ≈ 0.03–0.04, muy por encima del resto), consistente
con que Sonar es, de los cuatro, el de mayor dimensionalidad (60 features) y por lo tanto
probablemente el de paisaje de optimización más difícil — ambos algoritmos exploran con más
incertidumbre, pero MOEACKF logra hacerlo con mejor resultado promedio.

## 5. Síntesis entre datasets (descriptiva, no un test adicional)

Siguiendo el requisito de no mezclar datasets en una misma prueba estadística, esta sección es
puramente **descriptiva** (en el espíritu de Demšar, 2006, para resumir resultados por dataset
sin combinarlos en un test conjunto):

| Dataset | Ganador (α=0.05) | p-valor | Â₁₂ | Magnitud del efecto |
|---|---|---|---|---|
| 1 — Statlog Australian | SNSGAII | 0.0022 | 0.323 | Grande |
| 2 — Climate | *(sin diferencia)* | 0.347 | 0.258 | — (no significativo) |
| 3 — German | SNSGAII | <0.0001 | 0.226 | Grande |
| 4 — Sonar | MOEACKF | <0.0001 | 0.710 | Grande |

SNSGAII gana en 2 de 4 datasets, MOEACKF en 1, y en 1 no hay diferencia significativa. No hay un
algoritmo universalmente dominante en HV sobre estos cuatro datasets. Un patrón que emerge —
**a título de hipótesis, no de conclusión estadísticamente probada** dado que son solo 4
datasets — es que SNSGAII rinde mejor en los datasets de menor dimensionalidad (14 y 24
features, ds1 y ds3) mientras que MOEACKF rinde mejor en el de mayor dimensionalidad (60
features, ds4), y en el de dimensionalidad intermedia (18 features, ds2) no hay diferencia. Si
esta hipótesis interesa a la tesis, habría que confirmarla corriendo los datasets 5 y 6 (4 y 13
features) y viendo si el patrón se sostiene, no darla por válida con solo cuatro puntos.

También es consistente en 3 de los 4 datasets (ds1, ds2, ds3) que **MOEACKF muestra menor
varianza entre corridas** que SNSGAII, incluso cuando pierde en mediana — es decir, MOEACKF
tiende a ser más consistente/predecible, mientras que SNSGAII explora con mayor variabilidad.
Esto es relevante más allá de qué algoritmo "gana": si el criterio de la tesis valora
reproducibilidad además de desempeño puntual, MOEACKF podría preferirse incluso en datasets
donde no gana en mediana.

## 6. Limitaciones de este análisis

- **Solo se comparó HV.** No hay accuracy de test porque no existen archivos `*_spikes.csv` en
  los resultados actuales (ver `docs/metodologia_comparacion.md`, sección "Por qué revisar
  train vs. test"). No se puede, con los datos actuales, distinguir si las diferencias de HV
  observadas reflejan mejor generalización o simplemente mejor ajuste al error de entrenamiento
  (`f2`, que es parte del propio HV).
- **Datasets 5 y 6 no tienen resultados corridos todavía** — las conclusiones de este documento
  cubren únicamente Statlog Australian, Climate, German y Sonar.
- **No se comparó tiempo de cómputo** (`time_s` fue descartado deliberadamente, ver
  metodología, por paralelismo/hilos inconsistente entre tandas de ejecución).
- **El HV es un indicador agregado**: no distingue si un algoritmo domina en la zona de alta
  dispersión (redes más ralas) o en la de mayor precisión del frente. Antes de una conclusión
  final en la tesis, conviene complementar con un análisis de *set coverage* o graficar los
  frentes de Pareto superpuestos por dataset (sugerido en la metodología, §5), especialmente
  para el dataset 1 y el 2, donde el patrón de "quién gana" no es tan claro como en 3 y 4.

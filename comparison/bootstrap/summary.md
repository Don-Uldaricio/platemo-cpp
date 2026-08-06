# Bootstrap vs. evaluación completa (normal_results)


Wilcoxon signed-rank pareado por seed sobre HV, por dataset y por algoritmo.


## Dataset 1 (Statlog Australian)

### MOEACKF

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| Normal | 0.8721 [0.8704, 0.8721] | 0.8717 ± 0.0015 | — | — |
| Bootstrap | 0.8830 [0.8814, 0.8854] | 0.8832 ± 0.0025 | 9.313e-10* | 0.000 |

Shapiro-Wilk p (informativo): Normal=0.001086, Bootstrap=0.6257.

Tiempo mediano: normal = 209.0 s, bootstrap = 1734.2 s (×8.3; descriptivo, no se testea formalmente -- costo esperado de promediar 50 remuestreos por generación en vez de una sola pasada completa).

### SNSGAII

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| Normal | 0.8767 [0.8702, 0.8856] | 0.8781 ± 0.0090 | — | — |
| Bootstrap | 0.8879 [0.8829, 0.8909] | 0.8870 ± 0.0076 | 8.657e-05* | 0.258 |

Shapiro-Wilk p (informativo): Normal=0.4135, Bootstrap=0.3039.

Tiempo mediano: normal = 940.4 s, bootstrap = 14195.3 s (×15.1; descriptivo, no se testea formalmente -- costo esperado de promediar 50 remuestreos por generación en vez de una sola pasada completa).


## Dataset 2 (Climate)

### MOEACKF

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| Normal | 0.9256 [0.9255, 0.9285] | 0.9276 ± 0.0036 | — | — |
| Bootstrap | 0.9401 [0.9393, 0.9415] | 0.9404 ± 0.0015 | 9.313e-10* | 0.000 |

Shapiro-Wilk p (informativo): Normal=8.944e-05, Bootstrap=0.8816.

Tiempo mediano: normal = 1804.9 s, bootstrap = 8749.3 s (×4.8; descriptivo, no se testea formalmente -- costo esperado de promediar 50 remuestreos por generación en vez de una sola pasada completa).

### SNSGAII

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| Normal | 0.9278 [0.9257, 0.9278] | 0.9273 ± 0.0015 | — | — |
| Bootstrap | 0.9399 [0.9390, 0.9408] | 0.9400 ± 0.0013 | 9.313e-10* | 0.000 |

Shapiro-Wilk p (informativo): Normal=7.355e-05, Bootstrap=0.04974.

Tiempo mediano: normal = 201.9 s, bootstrap = 998.0 s (×4.9; descriptivo, no se testea formalmente -- costo esperado de promediar 50 remuestreos por generación en vez de una sola pasada completa).


## Dataset 3 (German)

### MOEACKF

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| Normal | 0.7520 [0.7452, 0.7560] | 0.7511 ± 0.0060 | — | — |
| Bootstrap | 0.7567 [0.7551, 0.7588] | 0.7568 ± 0.0031 | 0.0001954* | 0.226 |

Shapiro-Wilk p (informativo): Normal=0.005376, Bootstrap=0.9739.

Tiempo mediano: normal = 1207.2 s, bootstrap = 6221.9 s (×5.2; descriptivo, no se testea formalmente -- costo esperado de promediar 50 remuestreos por generación en vez de una sola pasada completa).

### SNSGAII

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| Normal | 0.7643 [0.7547, 0.7754] | 0.7645 ± 0.0160 | — | — |
| Bootstrap | 0.7549 [0.7540, 0.7568] | 0.7580 ± 0.0090 | 0.06642 | 0.710 |

Shapiro-Wilk p (informativo): Normal=0.7761, Bootstrap=8.504e-09.

Tiempo mediano: normal = 913.7 s, bootstrap = 6854.2 s (×7.5; descriptivo, no se testea formalmente -- costo esperado de promediar 50 remuestreos por generación en vez de una sola pasada completa).


## Dataset 4 (Sonar)

### MOEACKF

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| Normal | 0.8274 [0.8141, 0.8432] | 0.8224 ± 0.0302 | — | — |
| Bootstrap | 0.8356 [0.8157, 0.8424] | 0.8334 ± 0.0233 | 0.2241 | 0.484 |

Shapiro-Wilk p (informativo): Normal=0.001954, Bootstrap=0.628.

Tiempo mediano: normal = 1832.0 s, bootstrap = 9202.3 s (×5.0; descriptivo, no se testea formalmente -- costo esperado de promediar 50 remuestreos por generación en vez de una sola pasada completa).

### SNSGAII

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| Normal | 0.7812 [0.7601, 0.8059] | 0.7808 ± 0.0374 | — | — |
| Bootstrap | 0.8066 [0.7753, 0.8272] | 0.8020 ± 0.0315 | 0.0132* | 0.290 |

Shapiro-Wilk p (informativo): Normal=0.1431, Bootstrap=0.4388.

Tiempo mediano: normal = 1729.8 s, bootstrap = 11884.4 s (×6.9; descriptivo, no se testea formalmente -- costo esperado de promediar 50 remuestreos por generación en vez de una sola pasada completa).

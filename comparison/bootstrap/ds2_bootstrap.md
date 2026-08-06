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

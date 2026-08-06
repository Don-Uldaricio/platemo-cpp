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

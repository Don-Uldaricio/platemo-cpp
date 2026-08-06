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

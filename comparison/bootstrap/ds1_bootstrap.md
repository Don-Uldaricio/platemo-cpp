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

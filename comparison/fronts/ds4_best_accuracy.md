## Dataset 4 (Sonar) — solución de mejor accuracy por corrida

### Complejidad de la mejor red (frac. pesos activos)

N = 31 corridas pareadas por seed.

| Grupo | Complejidad de la mejor red (frac. pesos activos) (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.0429 [0.0208, 0.0847] | 0.0792 ± 0.0953 | — | — |
| SNSGAII | 0.0159 [0.0077, 0.0270] | 0.0188 ± 0.0141 | 0.0002673* | 0.742 |

Shapiro-Wilk p (informativo): MOEACKF=2.584e-06, SNSGAII=0.01256.

### Error de entrenamiento de la mejor red

N = 31 corridas pareadas por seed.

| Grupo | Error de entrenamiento de la mejor red (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.1796 [0.1677, 0.1976] | 0.1876 ± 0.0327 | — | — |
| SNSGAII | 0.2395 [0.2126, 0.2635] | 0.2403 ± 0.0417 | 4.766e-05* | 0.194 |

Shapiro-Wilk p (informativo): MOEACKF=0.0002955, SNSGAII=0.1685.

### Tamaño del frente (# soluciones no dominadas)

N = 31 corridas pareadas por seed.

| Grupo | Tamaño del frente (# soluciones no dominadas) (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 12.0000 [9.0000, 13.0000] | 11.2581 ± 3.2758 | — | — |
| SNSGAII | 8.0000 [7.0000, 10.5000] | 8.7097 ± 3.4370 | 0.01656* | 0.645 |

Shapiro-Wilk p (informativo): MOEACKF=0.7447, SNSGAII=0.8233.

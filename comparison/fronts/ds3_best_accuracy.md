## Dataset 3 (German) — solución de mejor accuracy por corrida

### Complejidad de la mejor red (frac. pesos activos)

N = 31 corridas pareadas por seed.

| Grupo | Complejidad de la mejor red (frac. pesos activos) (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.0065 [0.0042, 0.0269] | 0.0182 ± 0.0216 | — | — |
| SNSGAII | 0.0296 [0.0204, 0.0468] | 0.0414 ± 0.0360 | 0.009148* | 0.226 |

Shapiro-Wilk p (informativo): MOEACKF=2.672e-06, SNSGAII=3.169e-05.

### Error de entrenamiento de la mejor red

N = 31 corridas pareadas por seed.

| Grupo | Error de entrenamiento de la mejor red (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.2725 [0.2681, 0.2800] | 0.2734 ± 0.0068 | — | — |
| SNSGAII | 0.2587 [0.2462, 0.2694] | 0.2583 ± 0.0179 | 8.896e-05* | 0.790 |

Shapiro-Wilk p (informativo): MOEACKF=0.005023, SNSGAII=0.7736.

### Tamaño del frente (# soluciones no dominadas)

N = 31 corridas pareadas por seed.

| Grupo | Tamaño del frente (# soluciones no dominadas) (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 5.0000 [4.0000, 7.0000] | 5.5161 ± 2.1114 | — | — |
| SNSGAII | 9.0000 [8.0000, 10.5000] | 9.1290 ± 2.5132 | 1.852e-05* | 0.161 |

Shapiro-Wilk p (informativo): MOEACKF=0.005948, SNSGAII=0.554.

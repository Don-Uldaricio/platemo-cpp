## Dataset 2 (Climate) — solución de mejor accuracy por corrida

### Complejidad de la mejor red (frac. pesos activos)

N = 31 corridas pareadas por seed.

| Grupo | Complejidad de la mejor red (frac. pesos activos) (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.0190 [0.0083, 0.0363] | 0.0752 ± 0.1386 | — | — |
| SNSGAII | 0.0036 [0.0024, 0.0036] | 0.0032 ± 0.0008 | 1.17e-06* | 1.000 |

Shapiro-Wilk p (informativo): MOEACKF=1.301e-08, SNSGAII=1.698e-05.

### Error de entrenamiento de la mejor red

N = 31 corridas pareadas por seed.

| Grupo | Error de entrenamiento de la mejor red (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.0810 [0.0775, 0.0810] | 0.0786 ± 0.0038 | — | — |
| SNSGAII | 0.0787 [0.0787, 0.0810] | 0.0792 ± 0.0017 | 0.5345 | 0.565 |

Shapiro-Wilk p (informativo): MOEACKF=2.304e-05, SNSGAII=4.087e-05.

### Tamaño del frente (# soluciones no dominadas)

N = 31 corridas pareadas por seed.

| Grupo | Tamaño del frente (# soluciones no dominadas) (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 7.0000 [6.0000, 7.0000] | 6.6129 ± 1.1741 | — | — |
| SNSGAII | 3.0000 [3.0000, 4.0000] | 3.4194 ± 0.6720 | 1.422e-06* | 0.984 |

Shapiro-Wilk p (informativo): MOEACKF=0.05199, SNSGAII=0.0001161.

## Dataset 1 (Statlog Australian) — solución de mejor accuracy por corrida

### Complejidad de la mejor red (frac. pesos activos)

N = 31 corridas pareadas por seed.

| Grupo | Complejidad de la mejor red (frac. pesos activos) (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.0088 [0.0074, 0.0140] | 0.0107 ± 0.0044 | — | — |
| SNSGAII | 0.0221 [0.0132, 0.0375] | 0.0288 ± 0.0214 | 0.0002121* | 0.226 |

Shapiro-Wilk p (informativo): MOEACKF=0.00264, SNSGAII=0.0004447.

### Error de entrenamiento de la mejor red

N = 31 corridas pareadas por seed.

| Grupo | Error de entrenamiento de la mejor red (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.1395 [0.1395, 0.1413] | 0.1399 ± 0.0017 | — | — |
| SNSGAII | 0.1322 [0.1232, 0.1395] | 0.1313 ± 0.0100 | 0.000373* | 0.710 |

Shapiro-Wilk p (informativo): MOEACKF=0.0007094, SNSGAII=0.4026.

### Tamaño del frente (# soluciones no dominadas)

N = 31 corridas pareadas por seed.

| Grupo | Tamaño del frente (# soluciones no dominadas) (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 6.0000 [5.0000, 6.0000] | 5.7419 ± 0.9650 | — | — |
| SNSGAII | 7.0000 [5.0000, 9.0000] | 7.3548 ± 3.0501 | 0.0145* | 0.323 |

Shapiro-Wilk p (informativo): MOEACKF=0.01193, SNSGAII=0.1849.

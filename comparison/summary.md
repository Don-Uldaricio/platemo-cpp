# Comparación MOEACKF vs SNSGAII — resumen por dataset

Prueba: Wilcoxon signed-rank pareado por seed sobre HV. Cada dataset se analiza por separado (no se combinan en un mismo test), según lo indicado en docs/metodologia_comparacion.md.

**Accuracy de test: pendiente.** No existen archivos `*_spikes.csv` en los resultados actuales, por lo que este análisis cubre únicamente HV.

## Dataset 1 (Statlog Australian)

N = 31 corridas pareadas por seed.

| Algoritmo | HV (mediana [IQR]) | p-valor (Wilcoxon) | Â₁₂ |
|---|---|---|---|
| MOEACKF | 0.8721 [0.8704, 0.8721] | — | — |
| SNSGAII | 0.8767 [0.8702, 0.8856] | 0.002158* | 0.323 |

Shapiro-Wilk p (informativo): MOEACKF=0.001086, SNSGAII=0.4135.

Diferencia significativa (α=0.05): **SNSGAII** tiene mayor HV mediano.

## Dataset 2 (Climate)

N = 31 corridas pareadas por seed.

| Algoritmo | HV (mediana [IQR]) | p-valor (Wilcoxon) | Â₁₂ |
|---|---|---|---|
| MOEACKF | 0.9256 [0.9255, 0.9285] | — | — |
| SNSGAII | 0.9278 [0.9257, 0.9278] | 0.3468 | 0.258 |

Shapiro-Wilk p (informativo): MOEACKF=8.944e-05, SNSGAII=7.355e-05.

Sin diferencia estadísticamente significativa (α=0.05).

## Dataset 3 (German)

N = 31 corridas pareadas por seed.

| Algoritmo | HV (mediana [IQR]) | p-valor (Wilcoxon) | Â₁₂ |
|---|---|---|---|
| MOEACKF | 0.7520 [0.7452, 0.7560] | — | — |
| SNSGAII | 0.7643 [0.7547, 0.7754] | 2.825e-05* | 0.226 |

Shapiro-Wilk p (informativo): MOEACKF=0.005376, SNSGAII=0.7761.

Diferencia significativa (α=0.05): **SNSGAII** tiene mayor HV mediano.

## Dataset 4 (Sonar)

N = 31 corridas pareadas por seed.

| Algoritmo | HV (mediana [IQR]) | p-valor (Wilcoxon) | Â₁₂ |
|---|---|---|---|
| MOEACKF | 0.8274 [0.8141, 0.8432] | — | — |
| SNSGAII | 0.7812 [0.7601, 0.8059] | 8.657e-05* | 0.710 |

Shapiro-Wilk p (informativo): MOEACKF=0.001954, SNSGAII=0.1431.

Diferencia significativa (α=0.05): **MOEACKF** tiene mayor HV mediano.

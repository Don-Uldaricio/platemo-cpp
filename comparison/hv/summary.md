# Comparación MOEACKF vs SNSGAII — HV, resumen por dataset

Prueba: Wilcoxon signed-rank pareado por seed sobre HV. Cada dataset se analiza por separado (no se combinan en un mismo test), según lo indicado en docs/metodologia_comparacion.md.

## Dataset 1 (Statlog Australian)

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.8721 [0.8704, 0.8721] | 0.8717 ± 0.0015 | — | — |
| SNSGAII | 0.8767 [0.8702, 0.8856] | 0.8781 ± 0.0090 | 0.002158* | 0.323 |

Shapiro-Wilk p (informativo): MOEACKF=0.001086, SNSGAII=0.4135.

Diferencia significativa (α=0.05): **SNSGAII** tiene mayor HV mediano.

## Dataset 2 (Climate)

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.9256 [0.9255, 0.9285] | 0.9276 ± 0.0036 | — | — |
| SNSGAII | 0.9278 [0.9257, 0.9278] | 0.9273 ± 0.0015 | 0.3468 | 0.258 |

Shapiro-Wilk p (informativo): MOEACKF=8.944e-05, SNSGAII=7.355e-05.

Sin diferencia estadísticamente significativa (α=0.05).

## Dataset 3 (German)

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.7520 [0.7452, 0.7560] | 0.7511 ± 0.0060 | — | — |
| SNSGAII | 0.7643 [0.7547, 0.7754] | 0.7645 ± 0.0160 | 2.825e-05* | 0.226 |

Shapiro-Wilk p (informativo): MOEACKF=0.005376, SNSGAII=0.7761.

Diferencia significativa (α=0.05): **SNSGAII** tiene mayor HV mediano.

## Dataset 4 (Sonar)

N = 31 corridas pareadas por seed.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.8274 [0.8141, 0.8432] | 0.8224 ± 0.0302 | — | — |
| SNSGAII | 0.7812 [0.7601, 0.8059] | 0.7808 ± 0.0374 | 8.657e-05* | 0.710 |

Shapiro-Wilk p (informativo): MOEACKF=0.001954, SNSGAII=0.1431.

Diferencia significativa (α=0.05): **MOEACKF** tiene mayor HV mediano.

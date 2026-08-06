# Comparación de tiempo de ejecución — MOEACKF vs SNSGAII (time_experiments)

Wilcoxon signed-rank pareado por seed sobre time_s. Condiciones controladas de hardware e hiperparámetros compartidos entre algoritmos (ver docs/metodologia_comparacion.md) -- a diferencia de normal_results/bootstrap_results, acá la comparación de tiempos es válida.

## Dataset 1 (Statlog Australian)

N = 15 corridas pareadas por seed.
Unidad: segundos de reloj.

| Grupo | time_s (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 285.5552 [263.7128, 288.2233] | 277.3700 ± 15.4023 | — | — |
| SNSGAII | 259.5570 [258.4131, 260.5212] | 259.4892 ± 2.5043 | 0.001526* | 0.867 |

Shapiro-Wilk p (informativo): MOEACKF=0.02308, SNSGAII=0.3729.

Diferencia significativa (α=0.05): **SNSGAII** es más rápido (menor tiempo mediano).

## Dataset 2 (Climate)

N = 15 corridas pareadas por seed.
Unidad: segundos de reloj.

| Grupo | time_s (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 230.5385 [223.9658, 235.2065] | 232.9828 ± 20.2885 | — | — |
| SNSGAII | 216.7385 [215.9756, 217.0307] | 216.2164 ± 1.6709 | 0.0008545* | 0.867 |

Shapiro-Wilk p (informativo): MOEACKF=0.003215, SNSGAII=0.00975.

Diferencia significativa (α=0.05): **SNSGAII** es más rápido (menor tiempo mediano).

## Dataset 3 (German)

N = 15 corridas pareadas por seed.
Unidad: segundos de reloj.

| Grupo | time_s (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 936.5808 [895.6752, 956.0685] | 928.6555 ± 37.1301 | — | — |
| SNSGAII | 871.2661 [858.8802, 883.0376] | 867.7059 ± 20.6982 | 0.0004272* | 0.933 |

Shapiro-Wilk p (informativo): MOEACKF=0.4157, SNSGAII=0.09118.

Diferencia significativa (α=0.05): **SNSGAII** es más rápido (menor tiempo mediano).

## Dataset 4 (Sonar)

N = 15 corridas pareadas por seed.
Unidad: segundos de reloj.

| Grupo | time_s (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 715.0597 [525.9377, 865.2364] | 747.4226 ± 259.5051 | — | — |
| SNSGAII | 295.4588 [293.9731, 298.1651] | 295.6773 ± 4.0805 | 6.104e-05* | 1.000 |

Shapiro-Wilk p (informativo): MOEACKF=0.07146, SNSGAII=0.5855.

Diferencia significativa (α=0.05): **SNSGAII** es más rápido (menor tiempo mediano).

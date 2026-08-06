## Dataset 3 (German)

### Curva de convergencia

![Convergencia ds3](ds3_convergence.png)

### Velocidad de convergencia

**FE hasta 95% del HV final (indicador principal, entra en la corrección Holm-Bonferroni de la familia `principal_snn`):**

N = 31 corridas pareadas por seed.

| Grupo | FE hasta 95% del HV final (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.0000 [0.0000, 0.0000] | 37.3898 ± 81.3532 | — | — |
| SNSGAII | 3408.5213 [2456.1404, 3909.7744] | 3250.0606 ± 953.7954 | 9.313e-10* | 0.000 |

Shapiro-Wilk p (informativo): MOEACKF=7.935e-09, SNSGAII=0.6517.

**FE hasta 90% del HV final (chequeo de sensibilidad; no se corrige aparte por su redundancia con el de 95% -- misma trayectoria, mismo run):**

N = 31 corridas pareadas por seed.

| Grupo | FE hasta 90% del HV final (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 0.0000 [0.0000, 0.0000] | 0.0000 ± 0.0000 | — | — |
| SNSGAII | 1203.0075 [902.2556, 1503.7594] | 1190.0720 ± 479.9077 | 9.313e-10* | 0.000 |

Shapiro-Wilk p (informativo): MOEACKF=1, SNSGAII=0.4843.
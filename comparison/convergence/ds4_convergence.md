## Dataset 4 (Sonar)

### Curva de convergencia

![Convergencia ds4](ds4_convergence.png)

### Velocidad de convergencia

**FE hasta 95% del HV final (indicador principal, entra en la corrección Holm-Bonferroni de la familia `principal_snn`):**

N = 31 corridas pareadas por seed.

| Grupo | FE hasta 95% del HV final (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 13626.3736 [10622.7106, 18901.0989] | 14363.7008 ± 5291.3162 | — | — |
| SNSGAII | 5614.0351 [4110.2757, 6416.0401] | 6089.4171 ± 3580.2218 | 9.162e-06* | 0.903 |

Shapiro-Wilk p (informativo): MOEACKF=0.2127, SNSGAII=6.059e-06.

**FE hasta 90% del HV final (chequeo de sensibilidad; no se corrige aparte por su redundancia con el de 95% -- misma trayectoria, mismo run):**

N = 31 corridas pareadas por seed.

| Grupo | FE hasta 90% del HV final (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 5128.2051 [4029.3040, 6739.9267] | 5260.5459 ± 2587.3303 | — | — |
| SNSGAII | 3208.0201 [2406.0150, 4461.1529] | 3366.4807 ± 1557.9808 | 0.003597* | 0.774 |

Shapiro-Wilk p (informativo): MOEACKF=0.726, SNSGAII=0.02152.
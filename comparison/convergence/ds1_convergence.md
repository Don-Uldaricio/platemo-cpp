## Dataset 1 (Statlog Australian)

### Curva de convergencia

![Convergencia ds1](ds1_convergence.png)

### Velocidad de convergencia

**FE hasta 95% del HV final (indicador principal, entra en la corrección Holm-Bonferroni de la familia `principal_snn`):**

N = 31 corridas pareadas por seed.

| Grupo | FE hasta 95% del HV final (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 1212.1212 [969.6970, 1515.1515] | 1380.2542 ± 651.0023 | — | — |
| SNSGAII | 5226.1307 [5025.1256, 6733.6683] | 5874.5340 ± 1377.7373 | 9.313e-10* | 0.000 |

Shapiro-Wilk p (informativo): MOEACKF=0.008569, SNSGAII=0.001079.

**FE hasta 90% del HV final (chequeo de sensibilidad; no se corrige aparte por su redundancia con el de 95% -- misma trayectoria, mismo run):**

N = 31 corridas pareadas por seed.

| Grupo | FE hasta 90% del HV final (mediana [IQR]) | Media ± DE | p-valor (Wilcoxon signed-rank) | Â₁₂ |
|---|---|---|---|---|
| MOEACKF | 727.2727 [606.0606, 1090.9091] | 922.7761 ± 581.8833 | — | — |
| SNSGAII | 3417.0854 [3015.0754, 4020.1005] | 3663.4787 ± 783.1628 | 1.863e-09* | 0.032 |

Shapiro-Wilk p (informativo): MOEACKF=5.153e-06, SNSGAII=0.02541.
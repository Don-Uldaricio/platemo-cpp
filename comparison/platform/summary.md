# Plataforma: SNN (C++) vs. ANN (MATLAB/PlatEMO original)


Mann-Whitney U (muestras independientes -- no hay seed compartida entre MATLAB y C++), por dataset y por algoritmo.


## Dataset 1 (Statlog Australian)

### MOEACKF

**Hipervolumen (HV):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.8721 [0.8704, 0.8721] | 0.8717 ± 0.0015 | — | — |
| ANN-MATLAB | 0.8891 [0.8891, 0.8907] | 0.8901 ± 0.0018 | 1.37e-11* | 0.000 |

Shapiro-Wilk p (informativo): SNN=0.001086, ANN-MATLAB=2.1e-05.

**Complejidad de la red de mejor accuracy (frac. pesos activos):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Complejidad (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.0088 [0.0074, 0.0140] | 0.0107 ± 0.0044 | — | — |
| ANN-MATLAB | 0.0203 [0.0140, 0.0390] | 0.0302 ± 0.0244 | 2.584e-07* | 0.120 |

Shapiro-Wilk p (informativo): SNN=0.00264, ANN-MATLAB=3.201e-06.

**Accuracy de la red de mejor accuracy:**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Accuracy (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.8605 [0.8587, 0.8605] | 0.8601 ± 0.0017 | — | — |
| ANN-MATLAB | 0.8804 [0.8804, 0.8822] | 0.8817 ± 0.0021 | 5.059e-12* | 0.000 |

Shapiro-Wilk p (informativo): SNN=0.0007094, ANN-MATLAB=1.31e-05.
### SNSGAII

**Hipervolumen (HV):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.8767 [0.8702, 0.8856] | 0.8781 ± 0.0090 | — | — |
| ANN-MATLAB | 0.8972 [0.8948, 0.8979] | 0.8967 ± 0.0025 | 5.889e-11* | 0.016 |

Shapiro-Wilk p (informativo): SNN=0.4135, ANN-MATLAB=0.4768.

**Complejidad de la red de mejor accuracy (frac. pesos activos):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Complejidad (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.0221 [0.0132, 0.0375] | 0.0288 ± 0.0214 | — | — |
| ANN-MATLAB | 0.0265 [0.0265, 0.1256] | 0.1473 ± 0.2572 | 0.0001329* | 0.221 |

Shapiro-Wilk p (informativo): SNN=0.0004447, ANN-MATLAB=8.194e-09.

**Accuracy de la red de mejor accuracy:**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Accuracy (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.8678 [0.8605, 0.8768] | 0.8687 ± 0.0100 | — | — |
| ANN-MATLAB | 0.8895 [0.8877, 0.8904] | 0.8892 ± 0.0024 | 3.124e-11* | 0.011 |

Shapiro-Wilk p (informativo): SNN=0.4026, ANN-MATLAB=0.02121.

## Dataset 2 (Climate)

### MOEACKF

**Hipervolumen (HV):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.9256 [0.9255, 0.9285] | 0.9276 ± 0.0036 | — | — |
| ANN-MATLAB | 0.9767 [0.9747, 0.9770] | 0.9760 ± 0.0034 | 1.398e-11* | 0.000 |

Shapiro-Wilk p (informativo): SNN=8.944e-05, ANN-MATLAB=0.00123.

**Complejidad de la red de mejor accuracy (frac. pesos activos):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Complejidad (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.0190 [0.0083, 0.0363] | 0.0752 ± 0.1386 | — | — |
| ANN-MATLAB | 0.0487 [0.0399, 0.0662] | 0.0565 ± 0.0255 | 0.0002366* | 0.228 |

Shapiro-Wilk p (informativo): SNN=1.301e-08, ANN-MATLAB=0.0001116.

**Accuracy de la red de mejor accuracy:**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Accuracy (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.9190 [0.9190, 0.9225] | 0.9214 ± 0.0038 | — | — |
| ANN-MATLAB | 0.9768 [0.9745, 0.9768] | 0.9760 ± 0.0039 | 6.062e-12* | 0.000 |

Shapiro-Wilk p (informativo): SNN=2.304e-05, ANN-MATLAB=0.0002053.
### SNSGAII

**Hipervolumen (HV):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.9278 [0.9257, 0.9278] | 0.9273 ± 0.0015 | — | — |
| ANN-MATLAB | 0.9477 [0.9440, 0.9539] | 0.9493 ± 0.0064 | 1.394e-11* | 0.000 |

Shapiro-Wilk p (informativo): SNN=7.355e-05, ANN-MATLAB=0.1046.

**Complejidad de la red de mejor accuracy (frac. pesos activos):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Complejidad (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.0036 [0.0024, 0.0036] | 0.0032 ± 0.0008 | — | — |
| ANN-MATLAB | 0.9114 [0.7316, 0.9601] | 0.8367 ± 0.1867 | 7.72e-12* | 0.000 |

Shapiro-Wilk p (informativo): SNN=1.698e-05, ANN-MATLAB=9.104e-05.

**Accuracy de la red de mejor accuracy:**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Accuracy (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.9213 [0.9190, 0.9213] | 0.9208 ± 0.0017 | — | — |
| ANN-MATLAB | 0.9653 [0.9630, 0.9699] | 0.9657 ± 0.0060 | 8.2e-12* | 0.000 |

Shapiro-Wilk p (informativo): SNN=4.087e-05, ANN-MATLAB=0.009751.

## Dataset 3 (German)

### MOEACKF

**Hipervolumen (HV):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.7520 [0.7452, 0.7560] | 0.7511 ± 0.0060 | — | — |
| ANN-MATLAB | 0.8098 [0.8082, 0.8111] | 0.8100 ± 0.0024 | 1.401e-11* | 0.000 |

Shapiro-Wilk p (informativo): SNN=0.005376, ANN-MATLAB=0.3002.

**Complejidad de la red de mejor accuracy (frac. pesos activos):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Complejidad (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.0065 [0.0042, 0.0269] | 0.0182 ± 0.0216 | — | — |
| ANN-MATLAB | 0.0413 [0.0375, 0.0644] | 0.0509 ± 0.0221 | 1.262e-06* | 0.142 |

Shapiro-Wilk p (informativo): SNN=2.672e-06, ANN-MATLAB=5.421e-06.

**Accuracy de la red de mejor accuracy:**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Accuracy (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.7275 [0.7200, 0.7319] | 0.7266 ± 0.0068 | — | — |
| ANN-MATLAB | 0.7925 [0.7906, 0.7937] | 0.7927 ± 0.0027 | 1.201e-11* | 0.000 |

Shapiro-Wilk p (informativo): SNN=0.005023, ANN-MATLAB=0.0983.
### SNSGAII

**Hipervolumen (HV):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.7643 [0.7547, 0.7754] | 0.7645 ± 0.0160 | — | — |
| ANN-MATLAB | 0.8119 [0.8098, 0.8132] | 0.8118 ± 0.0025 | 1.402e-11* | 0.000 |

Shapiro-Wilk p (informativo): SNN=0.7761, ANN-MATLAB=0.517.

**Complejidad de la red de mejor accuracy (frac. pesos activos):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Complejidad (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.0296 [0.0204, 0.0468] | 0.0414 ± 0.0360 | — | — |
| ANN-MATLAB | 0.8751 [0.3506, 0.9942] | 0.6659 ± 0.3744 | 6.887e-10* | 0.044 |

Shapiro-Wilk p (informativo): SNN=3.169e-05, ANN-MATLAB=4.75e-05.

**Accuracy de la red de mejor accuracy:**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Accuracy (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.7412 [0.7306, 0.7538] | 0.7417 ± 0.0179 | — | — |
| ANN-MATLAB | 0.7975 [0.7963, 0.8006] | 0.7987 ± 0.0036 | 1.326e-11* | 0.000 |

Shapiro-Wilk p (informativo): SNN=0.7736, ANN-MATLAB=0.09416.

## Dataset 4 (Sonar)

### MOEACKF

**Hipervolumen (HV):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.8274 [0.8141, 0.8432] | 0.8224 ± 0.0302 | — | — |
| ANN-MATLAB | 0.8740 [0.8659, 0.8849] | 0.8744 ± 0.0141 | 5.885e-11* | 0.016 |

Shapiro-Wilk p (informativo): SNN=0.001954, ANN-MATLAB=0.5379.

**Complejidad de la red de mejor accuracy (frac. pesos activos):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Complejidad (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.0429 [0.0208, 0.0847] | 0.0792 ± 0.0953 | — | — |
| ANN-MATLAB | 0.0097 [0.0063, 0.0151] | 0.0141 ± 0.0131 | 2.395e-06* | 0.849 |

Shapiro-Wilk p (informativo): SNN=2.584e-06, ANN-MATLAB=1.467e-06.

**Accuracy de la red de mejor accuracy:**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Accuracy (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.8204 [0.8024, 0.8323] | 0.8124 ± 0.0327 | — | — |
| ANN-MATLAB | 0.8623 [0.8533, 0.8742] | 0.8629 ± 0.0158 | 4.926e-10* | 0.041 |

Shapiro-Wilk p (informativo): SNN=0.0002954, ANN-MATLAB=0.4852.
### SNSGAII

**Hipervolumen (HV):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | HV (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.7812 [0.7601, 0.8059] | 0.7808 ± 0.0374 | — | — |
| ANN-MATLAB | 0.9143 [0.8994, 0.9158] | 0.9086 ± 0.0132 | 1.402e-11* | 0.000 |

Shapiro-Wilk p (informativo): SNN=0.1431, ANN-MATLAB=0.008106.

**Complejidad de la red de mejor accuracy (frac. pesos activos):**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Complejidad (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.0159 [0.0077, 0.0270] | 0.0188 ± 0.0141 | — | — |
| ANN-MATLAB | 0.1503 [0.0254, 0.9942] | 0.4621 ± 0.4705 | 5.735e-07* | 0.132 |

Shapiro-Wilk p (informativo): SNN=0.01256, ANN-MATLAB=7.493e-07.

**Accuracy de la red de mejor accuracy:**

N = 31 (SNN) vs. N = 31 (ANN-MATLAB), muestras independientes.

| Grupo | Accuracy (mediana [IQR]) | Media ± DE | p-valor (Mann-Whitney U) | Â₁₂ |
|---|---|---|---|---|
| SNN | 0.7605 [0.7365, 0.7874] | 0.7597 ± 0.0417 | — | — |
| ANN-MATLAB | 0.9102 [0.9042, 0.9281] | 0.9150 ± 0.0163 | 1.318e-11* | 0.000 |

Shapiro-Wilk p (informativo): SNN=0.1685, ANN-MATLAB=0.289.
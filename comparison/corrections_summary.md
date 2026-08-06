# Corrección por comparaciones múltiples (Holm-Bonferroni) — veredicto final

Corrección aplicada DENTRO de cada familia (nunca entre familias). Ver docs/metodologia_comparacion.md para la definición de cada familia. Las filas con familia vacía (FE a 90%, C-metric) son chequeos de sensibilidad/descriptivos fuera de toda familia y no llevan p_holm.

| Familia | Dataset | Ángulo | Métrica | p crudo | p Holm | Â₁₂ | Sig. (crudo) | Sig. (Holm) |
|---|---|---|---|---|---|---|---|---|
| bootstrap|ds1 | ds1 | bootstrap_moeackf | hv | 9.313e-10 | 1.863e-09 | 0.000 | True | True |
| bootstrap|ds1 | ds1 | bootstrap_snsgaii | hv | 8.657e-05 | 8.657e-05 | 0.258 | True | True |
| bootstrap|ds2 | ds2 | bootstrap_moeackf | hv | 9.313e-10 | 1.863e-09 | 0.000 | True | True |
| bootstrap|ds2 | ds2 | bootstrap_snsgaii | hv | 9.313e-10 | 1.863e-09 | 0.000 | True | True |
| bootstrap|ds3 | ds3 | bootstrap_moeackf | hv | 0.0001954 | 0.0003907 | 0.226 | True | True |
| bootstrap|ds3 | ds3 | bootstrap_snsgaii | hv | 0.06642 | 0.06642 | 0.710 | False | False |
| bootstrap|ds4 | ds4 | bootstrap_moeackf | hv | 0.2241 | 0.2241 | 0.484 | False | False |
| bootstrap|ds4 | ds4 | bootstrap_snsgaii | hv | 0.0132 | 0.0264 | 0.290 | True | True |
| plataforma|ds1|platform_moeackf | ds1 | platform_moeackf | accuracy | 5.059e-12 | 1.518e-11 | 0.000 | True | True |
| plataforma|ds1|platform_moeackf | ds1 | platform_moeackf | complexity | 2.584e-07 | 2.584e-07 | 0.120 | True | True |
| plataforma|ds1|platform_moeackf | ds1 | platform_moeackf | hv | 1.37e-11 | 2.74e-11 | 0.000 | True | True |
| plataforma|ds1|platform_snsgaii | ds1 | platform_snsgaii | accuracy | 3.124e-11 | 9.373e-11 | 0.011 | True | True |
| plataforma|ds1|platform_snsgaii | ds1 | platform_snsgaii | complexity | 0.0001329 | 0.0001329 | 0.221 | True | True |
| plataforma|ds1|platform_snsgaii | ds1 | platform_snsgaii | hv | 5.889e-11 | 1.178e-10 | 0.016 | True | True |
| plataforma|ds2|platform_moeackf | ds2 | platform_moeackf | accuracy | 6.062e-12 | 1.818e-11 | 0.000 | True | True |
| plataforma|ds2|platform_moeackf | ds2 | platform_moeackf | complexity | 0.0002366 | 0.0002366 | 0.228 | True | True |
| plataforma|ds2|platform_moeackf | ds2 | platform_moeackf | hv | 1.398e-11 | 2.795e-11 | 0.000 | True | True |
| plataforma|ds2|platform_snsgaii | ds2 | platform_snsgaii | accuracy | 8.2e-12 | 2.316e-11 | 0.000 | True | True |
| plataforma|ds2|platform_snsgaii | ds2 | platform_snsgaii | complexity | 7.72e-12 | 2.316e-11 | 0.000 | True | True |
| plataforma|ds2|platform_snsgaii | ds2 | platform_snsgaii | hv | 1.394e-11 | 2.316e-11 | 0.000 | True | True |
| plataforma|ds3|platform_moeackf | ds3 | platform_moeackf | accuracy | 1.201e-11 | 3.604e-11 | 0.000 | True | True |
| plataforma|ds3|platform_moeackf | ds3 | platform_moeackf | complexity | 1.262e-06 | 1.262e-06 | 0.142 | True | True |
| plataforma|ds3|platform_moeackf | ds3 | platform_moeackf | hv | 1.401e-11 | 3.604e-11 | 0.000 | True | True |
| plataforma|ds3|platform_snsgaii | ds3 | platform_snsgaii | accuracy | 1.326e-11 | 3.979e-11 | 0.000 | True | True |
| plataforma|ds3|platform_snsgaii | ds3 | platform_snsgaii | complexity | 6.887e-10 | 6.887e-10 | 0.044 | True | True |
| plataforma|ds3|platform_snsgaii | ds3 | platform_snsgaii | hv | 1.402e-11 | 3.979e-11 | 0.000 | True | True |
| plataforma|ds4|platform_moeackf | ds4 | platform_moeackf | accuracy | 4.926e-10 | 9.853e-10 | 0.041 | True | True |
| plataforma|ds4|platform_moeackf | ds4 | platform_moeackf | complexity | 2.395e-06 | 2.395e-06 | 0.849 | True | True |
| plataforma|ds4|platform_moeackf | ds4 | platform_moeackf | hv | 5.885e-11 | 1.766e-10 | 0.016 | True | True |
| plataforma|ds4|platform_snsgaii | ds4 | platform_snsgaii | accuracy | 1.318e-11 | 3.953e-11 | 0.000 | True | True |
| plataforma|ds4|platform_snsgaii | ds4 | platform_snsgaii | complexity | 5.735e-07 | 5.735e-07 | 0.132 | True | True |
| plataforma|ds4|platform_snsgaii | ds4 | platform_snsgaii | hv | 1.402e-11 | 3.953e-11 | 0.000 | True | True |
| principal_snn|ds1 | ds1 | convergence_speed | fe95 | 9.313e-10 | 4.657e-09 | 0.000 | True | True |
| principal_snn|ds1 | ds1 | fronts_best_accuracy | complexity | 7.782e-05 | 0.0003113 | 0.226 | True | True |
| principal_snn|ds1 | ds1 | fronts_best_accuracy | error | 0.000373 | 0.001119 | 0.710 | True | True |
| principal_snn|ds1 | ds1 | fronts_best_accuracy | front_size | 0.0145 | 0.0145 | 0.323 | True | True |
| principal_snn|ds1 | ds1 | hv_main | hv | 0.002158 | 0.004315 | 0.323 | True | True |
| principal_snn|ds2 | ds2 | convergence_speed | fe95 | 0.0004751 | 0.001425 | 0.694 | True | True |
| principal_snn|ds2 | ds2 | fronts_best_accuracy | complexity | 9.313e-10 | 4.657e-09 | 1.000 | True | True |
| principal_snn|ds2 | ds2 | fronts_best_accuracy | error | 0.5345 | 0.6937 | 0.565 | False | False |
| principal_snn|ds2 | ds2 | fronts_best_accuracy | front_size | 1.422e-06 | 5.686e-06 | 0.984 | True | True |
| principal_snn|ds2 | ds2 | hv_main | hv | 0.3468 | 0.6937 | 0.258 | False | False |
| principal_snn|ds3 | ds3 | convergence_speed | fe95 | 9.313e-10 | 4.657e-09 | 0.000 | True | True |
| principal_snn|ds3 | ds3 | fronts_best_accuracy | complexity | 0.009148 | 0.009148 | 0.226 | True | True |
| principal_snn|ds3 | ds3 | fronts_best_accuracy | error | 8.896e-05 | 0.0001779 | 0.790 | True | True |
| principal_snn|ds3 | ds3 | fronts_best_accuracy | front_size | 1.852e-05 | 7.409e-05 | 0.161 | True | True |
| principal_snn|ds3 | ds3 | hv_main | hv | 2.825e-05 | 8.475e-05 | 0.226 | True | True |
| principal_snn|ds4 | ds4 | convergence_speed | fe95 | 9.162e-06 | 4.581e-05 | 0.903 | True | True |
| principal_snn|ds4 | ds4 | fronts_best_accuracy | complexity | 0.0001068 | 0.0002597 | 0.742 | True | True |
| principal_snn|ds4 | ds4 | fronts_best_accuracy | error | 4.766e-05 | 0.0001906 | 0.194 | True | True |
| principal_snn|ds4 | ds4 | fronts_best_accuracy | front_size | 0.01656 | 0.01656 | 0.645 | True | True |
| principal_snn|ds4 | ds4 | hv_main | hv | 8.657e-05 | 0.0002597 | 0.710 | True | True |
| tiempo|ds1 | ds1 | time | time_s | 0.001526 | 0.001526 | 0.867 | True | True |
| tiempo|ds2 | ds2 | time | time_s | 0.0008545 | 0.0008545 | 0.867 | True | True |
| tiempo|ds3 | ds3 | time | time_s | 0.0004272 | 0.0004272 | 0.933 | True | True |
| tiempo|ds4 | ds4 | time | time_s | 6.104e-05 | 6.104e-05 | 1.000 | True | True |
| — | ds1 | convergence_speed_sensitivity | fe90 | 1.863e-09 | — | 0.032 | True | True |
| — | ds2 | convergence_speed_sensitivity | fe90 | 1 | — | 0.500 | False | False |
| — | ds3 | convergence_speed_sensitivity | fe90 | 9.313e-10 | — | 0.000 | True | True |
| — | ds4 | convergence_speed_sensitivity | fe90 | 0.003597 | — | 0.774 | True | True |
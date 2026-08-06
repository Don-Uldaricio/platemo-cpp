## Dataset 4 (Sonar) — error mediano por banda de complejidad

Deciles de complejidad (`f1_complexity`) sobre el pool de puntos crudos de ambos algoritmos (10 bandas solicitadas; pueden colapsar menos por empates en los bordes, frecuente porque 8.1% de los puntos tienen complejidad exactamente 0).

| Banda de complejidad | MOEACKF error mediano (n) | SNSGAII error mediano (n) |
|---|---|---|
| (-0.001, 0.000397] | 0.3593 (59) | 0.3862 (38) |
| (0.000397, 0.00159] | 0.3503 (2) | 0.3743 (28) |
| (0.00159, 0.00317] | 0.2934 (23) | 0.3234 (51) |
| (0.00317, 0.00437] | 0.2754 (22) | 0.2874 (39) |
| (0.00437, 0.00595] | 0.2425 (26) | 0.2695 (33) |
| (0.00595, 0.00913] | 0.2216 (23) | 0.2455 (32) |
| (0.00913, 0.0163] | 0.2156 (29) | 0.2275 (29) |
| (0.0163, 0.0329] | 0.2096 (48) | 0.2635 (13) |
| (0.0329, 0.0787] | 0.2156 (55) | 0.2036 (7) |
| (0.0787, 0.386] | 0.2216 (62) | — (0) |

![Error por banda ds4](ds4_complexity_bands.png)

| arm | constraints | search space | objective | full evals | scored | best log Z/box | Δ vs T47 | Δ/σ_fold | χ²_ν | L_MAP |
|---|---|---|---|---|---|---|---|---|---|---|
| Agent loop (T01-T60) | Gate 0, p<=6, 1-2 nl | open (any sympy expr) | log Z_cross feedback | 60 | 60 | 35.35 | 0.00 | 0.0 | 8.99 | 17.9 |
| PySR U, MSE 'best' (Plan 04) | none | PySR U operators | weighted MSE | 1 | 1 | 30.19 | -5.15 | -13.8 | 21.39 | 41.4 |
| PySR C, MSE 'best' (Plan 04) | G0_1-3 by template | W(k e^-f) g template | weighted MSE | 1 | 1 | 32.23 | -3.12 | -8.3 | 7.00 | 24.4 |
| A1-U: PySR U hall of fame, log Z selection | none (Gate 0 at scoring) | PySR U operators | MSE search, log Z selection | 23 | 2 | 30.19 | -5.15 | -13.8 | 21.39 | 41.4 |
| A1-C: PySR C hall of fame, log Z selection | G0_1-3 by template | W(k e^-f) g template | MSE search, log Z selection | 19 | 19 | 34.33 | -1.01 | -2.7 | 6.60 | 23.3 |
| A2: PySR two-kernel template, log Z selection | G0_1-3 by template | sum of two W2 kernels | MSE search, log Z selection | 18 | 18 | 32.08 | -3.27 | -8.8 | 15.87 | 36.0 |
| A3: seed-grammar enumeration (top 60 of prescreen) | Gate 0, p<=6, 1-2 nl | seed grammar (292 forms) | log Z_cross (R={} prescreen, then full) | 60 | 60 | 34.23 | -1.12 | -3.0 | 10.61 | 24.8 |
| Best seed form (paper rank 1, P1) | Gate 0 | - | - | 1 | 1 | 32.89 | -2.46 | -6.6 | 8.14 | – |

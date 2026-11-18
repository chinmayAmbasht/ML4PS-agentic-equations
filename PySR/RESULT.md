# PySR benchmark: result (2026-11-14). Plan: `Plans/04_pysr.md`

## Setup
- **PySR:** 2.7.2.
- **Features:** k, D, q = f_b/(Ω_b/Ω_m), Ω_m, σ_8, Ω_b/Ω_m.
- **Rows:** one per (box, k) on the 16 bins with k ≤ 9.92.
- **Loss:** weighted MSE, weight = h / diag C_fv,j of the row's suite.
- **Search settings:** maxsize 30, 40 populations, batch 2000, 2700 s per search, seed 0, selection "best".
- **Searches:** 10 in total, 2 variants × (4 LOSO folds + pooled). All exited with code 0. Logs are in `logs/`; each search's outputs are in `runs/<v>_<fold>/`.
- **U (unconstrained):** binary +, −, ×, ÷; unary exp, log, square; no nested exp or log.
- **C (template):** S = W(k e^{−f(D,q,Ω_m,σ_8,ω_b)}) · g(k/(1+k), D, q, …), with f and g built from +, −, × and exp.
  - In f, `#2` is q; in g, `#2` is D.

## LOSO zero-shot (correlated L, equation and constants from 3 suites)
| held-out | U equation | L | C equation | L |
|---|---|---|---|---|
| IllustrisTNG | D/(−9.95 (q/k)² − 0.427) | 40.4 | f = 2.78(q − 0.219); g = −2.46 D | 41.1 |
| SIMBA | −2.87 D exp((−2.76 q − 0.280)/k) | 13.5 | f = q e^{1.89 − ω_b − q} − σ_8; g = −2.52 D | 16.3 |
| Astrid | −3.39 D e^{−1.94/(k+D)} | 743.9 | f = q; g = −2.23 D | 526.4 |
| Swift-EAGLE | −2.70 D exp(−4.12 q/k) | 91.8 | f = 2.84 q − 0.619; g = −2.49 D | 64.7 |
| **mean** | | **222.4** | | **162.1** |

For comparison:
- RF zero-shot: 294.5.
- T47 Stage-2 prior mean: 466.4.
- T47 least-squares population fit: 214.8.

## Pooled equations → pipeline (`harness/trial_state.yaml`)
**U:** the pooled search returned S = −0.193 D [k/(k/4.565 + q)]².
- Submission: −A₁ D k²/(q + k/b₀)².
- Result: Gate 0 passes, log Z_cross = 30.19 nats/box (σ_fold 0.70). It would rank 64 of 64 (Δ −5.15 vs T47).
- Marginalized: A₁. χ²_ν mean 21.4.

**C:** the pooled search returned f = 1.44 (q − e^{−8.15 q}), g = −2.49 D.
- Submission: −A₀ D W(k e^{−b₁[q − e^{−b₂ q}]}).
- Result: Gate 0 passes, log Z = 32.23 nats/box (σ_fold 0.31). It would rank 61 of 64 (Δ −3.12).
- Marginalized: b₂. χ²_ν mean 7.0.

Like the agents' forms, both pooled equations were selected on all 4 suites.

## Benchmark (mean correlated L; `PaperUpdate/Benchmark/bench_corr.json`)
| model | within | pooled | zero-shot, LOSO search | zero-shot, pipeline prior mean | adapted MAP |
|---|---|---|---|---|---|
| PySR U | 94.4 | 98.9 | 222.4 | 446.9 | 103.7 |
| PySR C | 82.2 | 84.5 | 162.1 | 113.3 | 31.5 |
| T47 | 88.4 | 103.1 | — | 466.4 | 31.1 |
| RF | 88.9 | 176.6 | 294.5 | — | — |

**Reading:**
- Template PySR matches or beats the top loop forms on L everywhere, and ties them once adapted (31.5 vs 31.1). Its zero-shot is the best of any model.
- It still sits about 3 nats/box below T47 in evidence.
- The loss and the evidence disagree on which form is best.

## Fix during the run
`to_form` first converted constants after sympy had simplified the template. The squared exp split one PySR constant into two atoms, which left a coefficient missing. Now each literal becomes its own symbol before parsing, and lost coefficients are asserted against.

The U submission scored before the fix is an exact reparametrization of the fixed one (b₀ ↔ 1/b₀ over the same log range), so U was not re-run.

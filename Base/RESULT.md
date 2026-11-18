# The paper's base forms on the main-loop grid (z = 0, 16 bins with k ≤ 9.92; 2026-11-11)

## How they were scored
- **Harness:** `Loop/evaluate.py` in base mode (`PK_ML4PS_BASE=1`, which allows 3–4 nonlinear coefficients; p ≤ 6). Plan:
  `Plans/01_ml4ps_k10.md`.
- **Grid:** the paper's 13 bins with k ≤ 10, plus 1.619, 3.388 and 8.668.
- **Nonlinear grid:** the **full 41 nodes per nonlinear coefficient** (21 for paper rank 1, which has 3 nonlinear coefficients).
- **Pipeline:**
  - Gate 0 → LOSO folds → G0_U, the end-step band S(9.92) − S(8.67) ∈ [−0.040, +0.036];
  - **log Z_cross with the forward promotion ladder** (σ_fold rule);
  - χ²_ν per held-out suite (C_fv, per-box MAP);
  - L_zero (zero-shot) and L_map (per-box MAP, prior widened ×1.30).
- **State:** trial ids are local to `Base/` (`Base/trial_state.yaml`, `Base/residuals/T0n.md`), and the forms are in
  `Base/sub/`.
- **Coefficient ranges:** chosen here; the paper does not list them.
- **Comparability:** these numbers are not comparable with the paper's, because the k range and the engine differ.

## Results
log Z_cross is in nats (higher is better), and Δ is relative to the best form.

| paper rank | form | Base id | p | log Z_cross | Δ | marginalized here (paper) | σ_fold |
|---|---|---|---|---|---|---|---|
| 1 | −A D [w W₂(k1 D^β) + (1−w) W₂(k2 D^β)], w = Ω_b/Ω_m | T05 | 4 | **122,536** | 0 | k2 (paper: log k₂) | 1,715 |
| 2 | −A D W₂(k00/(1+αD)) | T02 | 3 | 121,196 | −1,341 | k00 (paper: α) | 1,940 |
| 3 | −A D W₂(k00 (Ω_m/0.3)^(1/3)) | T01 | 2 | 117,232 | −5,304 | k00 (paper: A) | 2,048 |
| 12 | −D [A1 W₂(k1) + A2 W₂(k1 e^δ)] | T03 | 4 | 117,058 | −5,478 | A2 (paper: none) | 1,699 |
| 32 | van Daalen 2020 | T04 | 5 | — | — | Gate 0 fail (G0_2, G0_3) | — |

| form | χ²_ν mean / std | TNG | SIMBA | Astrid | Swift | L_zero | L_map | G0_U |
|---|---|---|---|---|---|---|---|---|
| rank 1 (T05) | 8.14 / 7.78 | 5.40 | 1.63 | 21.41 | 4.13 | 90.3 | **13.2** | pass |
| rank 2 (T02) | 7.01 / 6.05 | 4.10 | 2.66 | 17.44 | 3.84 | 86.3 | 32.9 | pass |
| rank 3 (T01) | **6.57 / 6.09** | 4.20 | 1.64 | 17.00 | 3.46 | 98.6 | 18.4 | pass |
| rank 12 (T03) | 14.40 / 12.22 | 16.34 | 2.00 | 33.50 | 5.76 | 95.1 | 25.2 | pass |

## Against the paper
- **Ordering:** the paper's order is 1 > 2 > 3 > 12. It holds here: rank 1 leads by about 0.7 σ_fold, and ranks 3 and 12 are
  tied (Δ 174 nats).
- **van Daalen** fails the physical checks, as the paper states.
- **Marginalized set:** it agrees for rank 1 (k2) and differs for ranks 2, 3 and 12.
- **Astrid** is the worst held-out suite for every form, with χ²_ν between 17 and 34.
- **No form fits best on every metric:** rank 3, the simplest form (p = 2), has the lowest χ²_ν, while rank 1 has the lowest
  L_map.

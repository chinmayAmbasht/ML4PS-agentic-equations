# ML4PS main loop at z = 0, k ≤ 9.92 (16 bins): result (2026-11-11)

This is an AgenticTuner-inspired loop: 50 trials, p ≤ 6 with 1–2 nonlinear coefficients, 3 lanes, starting fresh from the
paper's base forms (`Base/`). The plan is in `Plans/01_ml4ps_k10.md`.

- **Scoring:** every trial is scored on the full 41-node grid with the promotion ladder.
- **Units:** log Z_cross is quoted **per box**, as the total over the 3726 held-out boxes divided by 3726, in nats/box (owner
  2026-11-11). Dividing by this constant changes neither the ranking nor the ladder.
- **Audit:** clean (`audit_adjudication.md`).
- **Interruption:** a login expiry at about 13:24 IST stopped the run at 43 trials. It resumed at 16:00 from cache; 3 harness
  errors were not counted.

The **evidence pass** (`Evidence/RESULT.md`) re-scores the leaders with Σ ×0.5 and ×2 and on the 11-node grid. Read it before
quoting a single winner.

## Ranking (top 10 by log Z_cross, plus the best-χ² form and the paper's base forms)
χ²₁₃ is the summed χ² (C_fv) rescaled to a common 13 degrees of freedom per box. χ²_ν divides by 16 − p, which penalizes
larger p (see the χ² section).

**Notation for the equations in the tables:**
- W(u) = u²/(1+u²);
- q = f_b/(Ω_b/Ω_m), the retained baryon fraction;
- u0 = k√D/k0 and u1 = k(D/0.05)^0.75/k1.

The bracket gives the form's rank by log Z_cross among all 54 scored forms (50 loop trials and the 4 paper base forms). It also
gives the trial id that the text below refers to.

| form [rank by log Z_cross; trial id] | log Z_cross / box | σ_fold / box | marginalized | p | χ²_ν mean | χ²₁₃ | χ²_ν Astrid | L_zero † | L_map † |
|---|---|---|---|---|---|---|---|---|---|
| **S = −D [A q W(k√D/0.68) + B (1−q) W(u0)] + (E f_b − C D) W(u1)** [rank 1; T47] | **35.345** | 0.37 | k0, k1, B | 6 | 8.99 | 6.92 | 25.0 | 97.1 | 6.42 |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 2; T36] | 35.186 | 0.47 | k1, C, k0 | 5 | 8.85 | 7.49 | 22.5 | 95.2 | 6.70 |
| S = −A D [q W(k√D/0.8) + (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 3; T43] | 35.167 | 0.45 | k0, E | 5 | 8.42 | 7.13 | 22.5 | 86.9 | 7.79 |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) W(u1) − G D (k/0.4)²/[1+(k/0.4)²]² [rank 4; T46] | 35.161 | 0.45 | k0, k1, C | 6 | 9.14 | 7.03 | 23.2 | 97.2 | 5.17 |
| S = −A D [q W(k√D/1.0) + (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 5; T40] | 35.098 | 0.44 | k0, E | 5 | 8.66 | 7.33 | 22.8 | 71.2 | 6.42 |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) W(u1)(1+2u1)/(1+u1) [rank 6; T44] | 35.081 | 0.45 | k0, k1 | 5 | 8.44 | 7.14 | 23.8 | 94.8 | 7.40 |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D/(Ω_b/Ω_m)) W(u1) [rank 7; T45] | 34.992 | 0.47 | k1 | 5 | 14.39 | 12.17 | 34.8 | 74.1 | 35.39 |
| S = −D [A q W(u1) + B (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 8; T49] | 34.884 | 0.47 | k0 | 6 | 12.99 | 9.99 | 33.9 | 74.1 | 38.66 |
| S = −A D [√q W(k√D/0.68) + (1−√q) W(u0)] + (E f_b − C D) W(u1) [rank 9; T42] | 34.878 | 0.54 | k0 | 5 | 10.88 | 9.21 | 30.4 | 69.4 | 6.12 |
| **S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) u1³/(1+u1²)^1.5** [rank 10; T39] | 34.752 | 0.48 | k0, E | 5 | **7.25** | **6.13** | **17.2** | 92.6 | **4.86** |
| S = −D W(u0) [A + B u0/(1+u0)] [rank 44; T07] | 33.102 | 0.48 | k0 | 3 | 6.62 | 6.62 | 17.4 | 83.2 | 18.77 |
| S = −A D [(Ω_b/Ω_m) W(k/(k1 D^β)) + (1−Ω_b/Ω_m) W(k/(k2 D^β))] [rank 45; paper rank 1] | 32.887 | 0.46 | k2 | 4 | 8.14 | 7.52 | 21.4 | 90.3 | 13.18 |
| S = −A D W(k(1+αD)/k00) [rank 49; paper rank 2] | 32.527 | 0.52 | k00 | 3 | 7.01 | 7.01 | 17.4 | 86.3 | 32.86 |
| S = −A D W(k/(k00 (Ω_m/0.3)^(1/3))) [rank 53; paper rank 3] | 31.463 | 0.55 | k00 | 2 | 6.57 | 7.08 | 17.0 | 98.6 | 18.38 |
| S = −D [A1 W(k/k1) + A2 W(k/(k1 e^dl))] [rank 54; paper rank 12] | 31.417 | 0.46 | A2 | 4 | 14.40 | 13.29 | 33.5 | 95.1 | 25.21 |

† L in this table uses the paper's σ_k; see the section below for the over-estimated σ_k.

- **Best form:** T47 is +2.458 nats/box over paper rank 1 (+9,160 nats in total).
- **Near-ties:** the next five forms (T36, T43, T46, T40, T44) lie within 0.27 nats/box of T47. That gap is less than σ_fold
  (0.37–0.47 per box), so the loop does **not** separate them.
- **Base ordering:** the paper's base ordering (1 > 2 > 3 ≈ 12) holds on this grid.

Running best: T01 (32.27) → T03 (33.28) → T08 → T09 → T10 → T11 (33.55) → T17 (33.86) → T18 (34.07) → T23 → T26 → T29 (34.35)
→ T35 (34.46) → **T36 (35.19)** → **T47 (35.35)**.

## L with the over-estimated σ_k (owner 2026-11-11)
**The new σ_k:** σ_k = max(σ_k^paper, max_j √(diag C_fv,j + diag C_disc,j)), with C_disc taken from paper rank 1. This is
3.0× the paper's value at k = 0.36, 1.03× at k = 6.15, and equal to it at k ≥ 7.66 (`Artifacts/sigma_k_inflated.json`).

**Re-scoring:** all 50 trials and the 4 base forms were re-scored by `rescore_L.py` (`L_inflated.jsonl`). With the paper's σ_k,
all 54 reproduce the stored L, so the re-scoring is consistent.

| form [rank by log Z_cross; trial id] | log Z_cross / box | L_zero (paper σ_k) | L_zero (inflated) | L_map (paper σ_k) | L_map (inflated) | L_map inflated per suite (TNG / SIMBA / Astrid / Swift) |
|---|---|---|---|---|---|---|
| S = −D [A q W(k√D/0.68) + B (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 1; T47] | 35.345 | 97.1 | 25.6 | 6.42 | 2.92 | 2.86 / 3.87 / 3.38 / 1.58 |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 2; T36] | 35.186 | 95.2 | 24.5 | 6.70 | 3.37 | 3.10 / 3.48 / 2.47 / 4.43 |
| S = −A D [q W(k√D/0.8) + (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 3; T43] | 35.167 | 86.9 | 22.4 | 7.79 | 3.09 | 1.33 / 4.78 / 2.57 / 3.69 |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) W(u1) − G D (k/0.4)²/[1+(k/0.4)²]² [rank 4; T46] | 35.161 | 97.2 | 25.0 | 5.17 | 3.04 | 1.59 / 2.30 / 3.44 / 4.84 |
| S = −A D [q W(k√D/1.0) + (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 5; T40] | 35.098 | 71.2 | 18.7 | 6.42 | 2.77 | 1.16 / 4.05 / 2.64 / 3.24 |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) W(u1)(1+2u1)/(1+u1) [rank 6; T44] | 35.081 | 94.8 | 24.8 | 7.40 | 2.89 | 1.66 / 4.79 / 3.32 / 1.80 |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) u1³/(1+u1²)^1.5 [rank 10; T39] | 34.752 | 92.6 | 23.7 | 4.86 | 2.83 | 1.66 / 2.46 / 2.42 / 4.77 |
| S = −A D W(u0) + (E f_b − C D) W(k(D/0.05)^0.5/k1) [rank 28; T20] | 33.994 | 100.2 | 24.3 | 4.91 | 2.32 | 0.58 / 2.03 / 4.08 / 2.57 |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) u1²/(1+u1)² [rank 12; T41] | 34.435 | 63.3 | 15.9 | 10.98 | 4.03 | 4.58 / 6.07 / 4.20 / 1.28 |
| S = −D W(u0) [A + B u0/(1+u0)] [rank 44; T07] | 33.102 | 83.2 | 20.6 | 18.77 | 5.62 | 3.36 / 12.58 / 3.61 / 2.93 |
| S = −A D [(Ω_b/Ω_m) W(k/(k1 D^β)) + (1−Ω_b/Ω_m) W(k/(k2 D^β))] [rank 45; paper rank 1] | 32.887 | 90.3 | 22.5 | 13.18 | 5.55 | 6.03 / 8.99 / 3.26 / 3.92 |
| S = −A D W(k(1+αD)/k00) [rank 49; paper rank 2] | 32.527 | 86.3 | 21.9 | 32.86 | 9.21 | 5.12 / 23.49 / 4.46 / 3.77 |
| S = −A D W(k/(k00 (Ω_m/0.3)^(1/3))) [rank 53; paper rank 3] | 31.463 | 98.6 | 24.1 | 18.38 | 6.78 | 6.48 / 11.65 / 4.81 / 4.16 |
| S = −D [A1 W(k/k1) + A2 W(k/(k1 e^dl))] [rank 54; paper rank 12] | 31.417 | 95.1 | 23.6 | 25.21 | 8.32 | 9.24 / 15.66 / 5.32 / 3.07 |

- **L_map:** the T36 family is about 2× better than the paper's forms (2.8–3.4 against 5.6–9.2).
  - T20 (log Z 33.99) has the lowest L_map of any trial, 2.32.
  - The L_map ranking changes little: Spearman correlation 0.95 between the paper and the new σ_k, over 54 forms.
- **L_zero is not better for the loop forms.**
  - With either σ_k, the leaders' zero-shot loss (18.7–25.6 with the new σ_k) is about the same as the paper's forms (21.9–24.1).
  - The evidence gain therefore comes from how well a form adapts to each box under the fold prior, not from a better
    population-mean prediction.
  - The lowest L_zero of any trial is T41 (15.9); among the leaders it is T40 (18.7).

## What the loop found
All six leaders belong to one family, built in lane A at T36 from lane C's T29 two-scale knee:

    S(k) = −D [ q·W(k√D / 0.68) + (1 − q)·W(k√D / k0) ]  +  (E f_b − C D)·W(k (D/0.05)^0.75 / k1),
    W(y) = y² / (1 + y²),   q = f_b / (Ω_b/Ω_m)   (retained baryon fraction)

- **The step to 35.19 (T36):** the large-scale knee is weighted by the **retained** fraction q, not the cosmic fraction
  Ω_b/Ω_m. In T36, A multiplies the whole first bracket.
- **T47:** gives the two knees separate amplitudes (A on the retained-gas knee, B on the expelled-gas knee), so the ladder can
  marginalize B per box.
- **T43 / T40:** move the fixed knee scale from 0.68 to 0.8 and to 1.0.
- **T46:** adds a fixed low-k bump, −G·D·W₂(k/0.4).
- **T39:** replaces the core onset with y³/(1 + y²)^1.5. This costs 0.43–0.59 nats/box against T36 and T47, but it gives the
  best χ²₁₃ and L_map of any form, and an Astrid χ²_ν (17.2) on par with the paper's forms.

**Disclosure:** the fixed constants (0.68, 0.75, 0.05, 0.4 and the √D scalings) were tuned by the agents against all four
suites, so they are hidden degrees of freedom outside p.

## χ²
- **Degrees of freedom:** the leaders look worse on χ²_ν than T07 and paper rank 3 partly because χ²_ν divides by 16 − p. At a
  common 13 degrees of freedom, T47 (6.92) and T39 (6.13) beat paper rank 1 (7.52) and paper rank 3 (7.08).
- **Where the χ² comes from:** most of it is in Astrid's two largest-scale bins (k = 0.36 and 0.60), and some in the Swift and
  TNG high-k end, where the fitted curves level off while the data keep falling.
- **C_fv vs C_disc:** C_fv is up to about 540 times smaller than C_disc at k = 0.36. See `Figures/diagnostics/`
  (`fig_cov_ratio*`, `fig_cov_diff`).

## Files
- **Trials:** `trial_state.yaml`, which holds raw totals in nats; `residuals/T*.md`; and `submissions/`.
- **Lanes:** `lanes.json` maps trial → lane. `lanes_results.json` holds both workflow runs' raw returns.
- **Audit:** `audit.json` and `audit_adjudication.md`. The agents' working notes are in `lab_notebook.md`.
- **Figure:** `Figures/fig_ml4ps_loop.{png,pdf}` (`python -m Loop.plot_loop`).

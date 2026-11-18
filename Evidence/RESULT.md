# Evidence pass: ML4PS main loop at k ≤ 9.92 (2026-11-11)

**Forms re-scored:** the loop's top five by log Z_cross (T47, T36, T43, T46, T40), T39 (best L_map, best χ² at 13
degrees of freedom), T07 (best χ²_ν), and the paper's base forms ranks 1, 2, 3 and 12.

**Settings per form:**
- the promotion ladder on the full grid: 41 nodes per nonlinear coefficient, or 21 for paper rank 1, which has 3;
- the full-grid ladder with every fold's fitted Σ_suite scaled by ×0.5 and ×2;
- the coarse 11-node grid, as a check.

The code is `evidence_pass.py`; the raw records are in `results.jsonl` (44 rows) (run logs not included). Units are
log Z_cross per box (÷ 3726).

## Results

**Notation for the equations in the tables:**
- W(u) = u²/(1+u²);
- q = f_b/(Ω_b/Ω_m), the retained baryon fraction;
- u0 = k√D/k0 and u1 = k(D/0.05)^0.75/k1.

The bracket gives the form's rank by log Z_cross among all 54 scored forms (50 loop trials and the 4 paper base forms). It also
gives the trial id that the text below refers to.

| form [rank by log Z_cross; trial id] | full grid | Σ ×0.5 | Σ ×2 | σ_fold | marginalized (full) | 11 nodes | marginalized (11) |
|---|---|---|---|---|---|---|---|
| **S = −D [A q W(k√D/0.68) + B (1−q) W(u0)] + (E f_b − C D) W(u1)** [rank 1; T47] | **35.345** | 35.313 | 35.362 | 0.37 | k0, k1, B | 35.154 | k0, E |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 2; T36] | 35.186 | 35.177 | 35.190 | 0.47 | k1, C, k0 | 34.654 | k0, E |
| S = −A D [q W(k√D/0.8) + (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 3; T43] | 35.167 † | 35.158 | 35.171 | 0.45 | k0, E | 35.075 | k0, C |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) W(u1) − G D (k/0.4)²/[1+(k/0.4)²]² [rank 4; T46] | 35.161 | 35.157 | 35.163 | 0.45 | k0, k1, C | 34.825 | k0, E |
| S = −A D [q W(k√D/1.0) + (1−q) W(u0)] + (E f_b − C D) W(u1) [rank 5; T40] | 35.098 | 35.095 | 35.100 | 0.44 | k0, E | 35.009 | k0, E |
| S = −A D [q W(k√D/0.68) + (1−q) W(u0)] + (E f_b − C D) u1³/(1+u1²)^1.5 [rank 10; T39] | 34.752 | 34.745 | 34.755 | 0.48 | k0, E | 35.043 | k0, k1 |
| S = −D W(u0) [A + B u0/(1+u0)] [rank 44; T07] | 33.102 | 33.098 | 33.103 | 0.48 | k0 | 33.075 | k0 |
| S = −A D [(Ω_b/Ω_m) W(k/(k1 D^β)) + (1−Ω_b/Ω_m) W(k/(k2 D^β))] [rank 45; paper rank 1] | 32.887 | 32.883 | 32.889 | 0.46 | k2 | 33.509 | k2, beta, A |
| S = −A D W(k(1+αD)/k00) [rank 49; paper rank 2] | 32.527 | 32.523 | 32.529 | 0.52 | k00 | 32.478 | k00 |
| S = −A D W(k/(k00 (Ω_m/0.3)^(1/3))) [rank 53; paper rank 3] | 31.463 | 31.461 | 31.464 | 0.55 | k00 | 31.356 | k00 |
| S = −D [A1 W(k/k1) + A2 W(k/(k1 e^dl))] [rank 54; paper rank 12] | 31.417 | 31.408 | 31.421 | 0.46 | A2 | 31.273 | k1 |

## Findings
1. **The loop scores reproduce exactly.** The full-grid ×1 values equal `Loop/trial_state.yaml` for every form.
2. **Prior width doesn't matter.**
   - Scaling Σ_suite by ×0.5 or ×2 moves T47 by 0.032 (×0.5) and 0.017 (×2) nats/box, and every other form by ≤ 0.009.
   - The full ordering of all 11 forms is the same under ×0.5, ×1 and ×2.
   - The marginalized sets don't change.
3. **The leaders are still not separated.**
   - T47 leads in every Σ setting, but its margin over T36 is 0.14–0.17 nats/box, against σ_fold ≈ 0.37–0.47.
   - T36, T43, T46 and T40 are within 0.09 nats/box of each other.
   - The evidence supports the **family** (retained-fraction two-scale knee plus core term) over the paper's forms, by
     +2.2–2.5 nats/box, about 5σ_fold. It does not pick one member.
4. **The 11-node grid misranks forms again.**
   - On 11 nodes, T39 ranks 3rd and T36 6th, and paper rank 1 rises above T07.
   - The selected marginalized sets change for 7 of the 11 forms.
   - Ranking on the full grid, as the loop did, was necessary.
5. **† T43 convergence:** one ladder step fails the convergence check.
   - The failing step is the **non-promoted** third step, [k0, E, k1]: in the TNG fold, 41 % of the k1 posterior sits in the
     grid tail.
   - The selected set [k0, E] and its score are converged.
   - A wider k1 range could change whether that third step is promoted, so T43's score is a lower bound on what its ladder
     might reach. All other forms converge at every step and setting.

## Recommendation for the paper (owner to decide)
- **Lead form:** quote the family, with **T47** as the highest-evidence member.
- **Alternative:** **T39** is 0.59 nats/box lower in evidence but has the best held-out fit quality of any form: χ²₁₃ 6.13 and
  L_map 4.86. Its Astrid χ²_ν (17.2) matches the paper's forms (17.0–17.4) rather than the leaders' 22–25 (see `Loop/RESULT.md`). If the paper wants one form that is good on both axes, T39 is the candidate.
- **Hidden constants:** both carry 4–6 fixed constants (the knee scale 0.68, the D^0.5 and (D/0.05)^0.75 scalings, and T39's
  onset exponents) tuned by the agents on all four suites, and that should be disclosed.

L with the over-estimated σ_k: `Loop/RESULT.md` (section "L with the over-estimated σ_k"); log Z_cross and the
ladder do not use σ_k, so this pass is unchanged by it.

Fit plots: `Figures/fits/` (`python -m Loop.plot_fits`).

# ML4PS main-loop lab notebook (z = 0, 16 bins with k <= 9.92, p <= 6, full 41-node scoring; Plans/01_ml4ps_k10.md)

Shared by every trial agent; append only.

- **Start point:** the paper's Table 1 forms, scored with this harness in `Base/RESULT.md`.
  - Best: paper rank 1, the two-scale baryon-weighted W2 form, with log Z_cross **122,536** and k2 marginalized.
  - Rank 2 scores 121,196; ranks 3 and 12 score about 117.1k.
  - Astrid is the worst held-out suite for every base form (chi2_nu 17-34).
- **Residuals:** the base forms' residual tables are `Base/residuals/T0n.md` (ids in `Base/RESULT.md`).
- **Score:** log Z_cross is primary (higher is better; sigma_fold is about 1,700-2,050 nats).
- **Gate 0 end step:** G0_U needs S(9.92) - S(8.67) in [-0.040, +0.036] for >= 90 % of boxes.


## A 2026-11-11 — next trial
- **Hypothesis:** in every base form TNG and Swift residuals are positive at k ~ 1-4 and negative at k > 6 (data keep
  falling after W2 has saturated). W2 = x^2/(1+x^2) turns over too sharply. Free the transition sharpness:
  W_n(x) = x^2/(1+x^n)^(2/n) (n = 2 is W2; n < 2 gives a softer, slower approach to the plateau), keeping rank 3's
  Omega_m-scaled scale.
- **Expression:** `-A*D*x**2/(1 + x**n)**(2/n)`, x = k/(k0*(Omega_m/0.3)**(1/3)); lin A; nl k0 (log 0.1-30), n (lin 0.4-4).
- **Prediction:** log Z_cross ~ 119,500; marginalized k0; worst suite Astrid.
- **Outcome (T01, SCORED):** log Z_cross **120,230** (predicted 119,500; +3.0k over base rank 3, still -2.3k below rank 1);
  marginalized k0 (as predicted; adding n gave +484 < sigma_fold 1,890). chi2_nu mean/std **17.59 / 16.23** (much worse
  than predicted 6.2): TNG 15.3, SIMBA 1.9, Astrid 44.5 (worst, as predicted), Swift 8.7. L_zero 96.6, L_map 21.4. G0_U pass.
- **Lesson:** the fits pick n ~ 1.1-1.3 with tiny k0 (0.05-0.28): the soft tail does help k > 6, but the long x^2 -> x^1
  crossover then over-suppresses the largest scales (k = 0.36 residual +7.6 sigma TNG, +6.1 Astrid, +3.5 Swift). The
  high-k deficit needs a second, larger scale (rank 1's route), not a softer single knee; keep the low-k end sharp.

## C 2026-11-11 07:25 — next trial
- **Data probe (z=0):** at k ~ 10 the ratio S/D is ~ -3 in every suite and nearly D-independent, while at low k |S| ~ D^2
  (TNG, SIMBA) -- the base forms already model this with a D-dependent scale. New signal: Astrid's lowest-D quartile
  (D ~ 0.008) has S/D = +1.8 at k = 3.9 and +7.8 at k = 9.9, i.e. a small-scale *enhancement* in boxes that keep their
  baryons; Astrid's S/D spread grows to 8 at high k. A pure -A D W(k) term cannot produce a sign flip.
- **Hypothesis:** add a second, retained-baryon (condensation) component proportional to f_b at its own scale kc:
  expulsion ~ -A D W2 at rank 3's Omega_m-scaled k0, plus B f_b W2(k/kc). Different from lane A (shape of W).
- **Expression:** `-A*D*W2(k/(k0*(Omega_m/0.3)**(1/3))) + B*f_b*W2(k/kc)`, W2(x) = x^2/(1+x^2); lin A, B; nl k0 (log
  0.1-30), kc (log 1-100).
- **Prediction:** log Z_cross ~ 120,500; marginalized k0 (possibly + B); worst suite Astrid.

## B 2026-11-11 — next trial (lane B, agent 1)
- **Hypothesis:** the paper's rank-1 two-scale W2 form (best base, 122,536) needs 3 nl coefficients, over the loop limit. A
  quick suite-level probe (global and per-box-amplitude fits, <scratch>/ml4ps_loop) finds the D^beta scale exponent sits
  at beta ~ -1/2 in every suite, and fixing beta = -1/2 (k_eff = k_i / sqrt(D): larger deficit pushes the redistribution
  to larger scales) fits as well as or better than free beta. So: rank 1 with beta fixed, 1 lin + 2 nl.
- **Expression:** `-A*D*(w*W2(k*sqrt(D)/k1) + (1-w)*W2(k*sqrt(D)/k2))`, w = Omega_b_over_Omega_m, W2(x) = x^2/(1+x^2).
- **Prediction:** log Z_cross ~ 123,000 (k2 marginalized, maybe k1); worst suite Astrid (chi2_nu ~ 19); chi2 mean ~ 7.5.
- **Outcome (T02, SCORED):** log Z_cross **119,223** (predicted 120,500; +2.0k over base rank 3, -1.0k below lane A's T01,
  -3.3k below rank 1); marginalized **k0, kc** (both promoted; + A gave +406 < sigma_fold 2,277). chi2_nu mean/std
  **7.55 / 6.95** (predicted 6.0): TNG 6.37, SIMBA 1.45 (best so far), Astrid 19.2 (worst), Swift 3.18; L_zero 113.8 (worse),
  **L_map 10.21 (best so far, rank 1 13.2)**. G0_U pass.
- **Lesson:** the condensation reading failed: every fold fits B < 0 (-0.9 to -2.1) with kc ~ 0.6-1.05, so B f_b W2 is a
  second *suppression* at a large scale, not an upturn; TNG residuals stay positive at all k (+4.6 at k = 0.36). The
  Astrid low-D upturn is too rare to drive the fit. A large-scale term in f_b (or Omega_b/Omega_m) beyond D helps L_map
  and SIMBA; try it with rank 1's D-scaled two-scale structure rather than a separate condensation scale.

## C 2026-11-11 07:40 — next trial (lane C, agent 2)
- **Data probe (z=0, <scratch>/ml4ps_loop/c20):** log-log slope of -S(k=0.36) against D is suite-dependent: TNG 2.02,
  SIMBA 1.85, Astrid 0.95, Swift 0.99, while at k = 9.9 it is ~1 everywhere (0.89-1.17). A single fixed D-scaling of the
  scale (B's sqrt(D), i.e. low-k S ~ D^2) suits TNG/SIMBA but not Astrid/Swift, whose k = 0.36 bin dominates chi2
  (mean sq 62 / 11 in T02). Partial correlations with f_b_M_i, Omega_m, sigma_8 at fixed D are weak (<0.6, most <0.3).
- **Hypothesis:** single W2 knee whose scale scales as a free power of D, k_eff = k (D/0.05)^b / k0: low-k S ~ D^(1+2b),
  high-k S ~ D. With b nonlinear and marginalized, Stage 2 can give each suite its own exponent (b ~ 0.5 TNG/SIMBA, ~0
  Astrid/Swift). Different from B (fixed b = 1/2, two scales): one scale, free exponent.
- **Expression:** `-A*D*x**2/(1+x**2)`, x = k*(D/0.05)**b/k0; lin A; nl k0 (log 0.05-30), b (lin -0.5..1.5).
- **Prediction:** log Z_cross ~ 122,000; marginalized b (then maybe k0); worst suite Astrid (chi2 ~ 15); chi2 mean ~ 6.5.
- **Outcome (T03, SCORED):** log Z_cross **123,993** (predicted 123,000; +1,457 over base rank 1 with one fewer nl coeff);
  marginalized k2 (as predicted). chi2_nu mean/std 7.79 / 8.15 (predicted 7.5): TNG 4.01, SIMBA 1.48, Astrid 21.80
  (worst, as predicted), Swift 3.85. L_zero 86.2, L_map 15.1. G0_U pass. p = 3.
- **Lesson:** beta = -1/2 (k_eff ∝ 1/sqrt(D)) can be fixed with no loss; the freed nl slot is available for something
  else (e.g. a shape or Omega_m term). Astrid low-k (k <= 1.1) remains the dominant chi2 sink.
- **Outcome (T04, SCORED):** log Z_cross **121,706** (predicted 122,000; +4.5k over base rank 3, +0.5k over rank 2, -2.3k
  below B's T03 at 123,993); marginalized **k0 only** (adding b gave +479 < sigma_fold 2,035; predicted b). chi2_nu mean/std
  **7.00 / 6.10** (predicted 6.5): TNG 3.99 (best so far), SIMBA 2.63, Astrid 17.53 (worst, as predicted), Swift 3.84.
  L_zero **82.6 (best so far)**, L_map 32.4 (poor). G0_U pass.
- **Lesson:** the free exponent settles at b ~ 0.45-0.62 in every fold (i.e. ~B's sqrt(D)); the suite-dependent low-k D
  slope is not absorbed by b -- Astrid now has a negative k = 0.6-1.1 residual (-2.6 sigma, over-suppressed at large
  scales). Fix b = 1/2 and spend the freed slot on large-scale shape (B's two-scale route wins); one knee is not enough.

## B 2026-11-11 07:30 — next trial (lane B, agent 2)
- **Data probe (<scratch>/ml4ps_loop/b_t2, C_fv-diag weighted, per-box free k2 to mimic marginalization):** on T03's
  structure, swapping knee shapes (W3, sharper n=4), dropping/raising the D-exponent on either component, or freeing the
  two weights linearly changes the per-box chi2 by < 10 %; per-box log k2 correlates weakly with every feature (|r| < 0.3)
  except SIMBA's high-mass f_b_M4,5 (r ~ 0.7, not universal). The one linear add-on that lowers per-box chi2 in TNG
  (2.3 -> 1.7), Astrid (14 -> 9.5) and Swift (2.3 -> 2.05) is a squared large-scale knee, B*D*W2(x2)^2: it sharpens the
  k2 transition (more suppression at k > 6, less at k ~ 1-4, the TNG/Swift residual pattern) without a new nl coefficient.
- **Hypothesis:** T03 + linear knee-sharpening term: tail-dominated extra suppression growing as W2(x2)^2 (leading k^4).
- **Expression:** `-D*(A*(w*W2(x1) + (1-w)*W2(x2)) + B*W2(x2)**2)`, x_i = k*sqrt(D)/k_i, w = Omega_b_over_Omega_m;
  lin A, B; nl k1 (log 0.05-10), k2 (log 0.05-20).
- **Prediction:** log Z_cross ~ 125,000; marginalized k2; worst suite Astrid (chi2 ~ 18); chi2 mean ~ 6.8.
- **Outcome (T05, SCORED):** log Z_cross **123,833** (predicted 125,000; -160 vs T03, a tie well inside sigma_fold 1,892);
  marginalized k2 (as predicted; + k1 gave +1,083 < sigma_fold). chi2_nu mean/std **7.63 / 8.14** (predicted 6.8): TNG 4.20,
  SIMBA **0.96 (best so far)**, Astrid 21.56 (worst, as predicted), Swift 3.80. L_zero 85.5, **L_map 7.31 (best so far;
  T03 15.1)**. G0_U pass. p = 4.
- **Lesson:** the fold fits of B are small and change sign across folds (-0.23 Swift ... +0.32 Astrid), so the sharpened
  knee is not a shared population feature: it buys L_map and SIMBA but no evidence. Knee shape is not the bottleneck;
  Astrid's k <= 1.1 scatter needs a second per-box degree of freedom or a feature that predicts it.

## C 2026-11-11 07:50 — next trial (lane C, agent 3)
- **Data probe (<scratch>/ml4ps_loop/c3_ag3):** per-suite global fits of T03 are all good within-suite (mean sq
  1.0-4.0 sigma^2), so the LOSO loss is transfer: the non-marginalized A and k1 differ across suites (A 2.44-3.23, rel.
  spread 0.107; log k1 spread 0.127). Scanning fixed rescalings of the T03 scale argument, sqrt(D)*(Omega_m/0.3)^(-1/3)
  (rank 3's Omega_m scaling grafted onto rank 1's sqrt(D)) gives the most suite-universal set: A spread 0.096, log k1
  0.110, log k2 0.191 (vs 0.243), at a small within-suite cost. f_b_M_4/5 are 80-94 % NaN (unusable); f_b, s8, D/w worse.
- **Hypothesis:** physical halo scale k ~ Omega_m^(1/3) (R ~ (M/Omega_m)^(1/3)) plus deficit reach k ~ 1/sqrt(D): a more
  universal population, so better transfer (esp. A, k1 to Astrid). Different from B2 (knee shape) and C2 (free exponent).
- **Expression:** T03 with x_i = k*sqrt(D)/(k_i*(Omega_m/0.3)**(1/3)); lin A; nl k1 (log 0.05-10), k2 (log 0.05-20).
- **Prediction:** log Z_cross ~ 124,800; marginalized k2; worst suite Astrid (chi2 ~ 19); chi2 mean ~ 7.2.

## A 2026-11-11 07:58 — next trial (lane A, agent 2)
- **Dry runs (<scratch>/ml4ps_loop/A2_dry_*, not counted):** rank 2 x rank 3 scale (k00 (Om/0.3)^(1/3)/(1+alpha D))
  119,285 (Om scaling hurts here); rank 2 x linear tail factor (A + B x/(1+x)) 121,977 (+0.8k over rank 2); same tail
  factor on a single sqrt(D) knee 123,336 with chi2 mean 6.62; adding a separate nl tail scale kt gives 123,984 but is
  degenerate (kt ~ k0, B < 0) with L_zero 186, so not worth the extra nl.
- **Hypothesis:** the residual pattern (TNG/Swift + at k ~ 1-4, - at k > 6) is a knee that saturates too fast. A one-scale
  knee whose plateau keeps growing slowly, W2(x) (A + B x/(1+x)) (leading k^2, bounded, plateau A+B), gives the slow
  high-k fall with one nl coefficient; scale k_eff = k0/sqrt(D) (B's fixed beta = -1/2). Different from B's two scales and
  B2's W2^2 add-on (sharpening, not softening), and from C's Om grafting.
- **Expression:** `-D*W2(x)*(A + B*x/(1+x))`, x = k*sqrt(D)/k0; lin A (amplitude), B (other); nl k0 (log 0.01-10). p = 3.
- **Prediction:** log Z_cross ~ 123,300; marginalized k0; worst suite Astrid (chi2 ~ 17.4); chi2 mean ~ 6.6.
- **Outcome (T06, SCORED):** log Z_cross **123,375** (predicted 124,800; -618 vs T03, a tie inside sigma_fold 1,877);
  marginalized k2 (as predicted; + A gave +373 < sigma_fold). chi2_nu mean/std **9.56 / 9.29** (predicted 7.2; worse than
  T03's 7.79): TNG 5.59, SIMBA 2.21, Astrid 25.51 (worst, as predicted), Swift 4.93. L_zero **84.2** (T03 86.2), L_map 27.4
  (T03 15.1). G0_U pass. p = 3. Per fold vs T03: Astrid +385, TNG -35, Swift -85, SIMBA -883.
- **Lesson:** the Omega_m^(1/3) factor on rank 1's sqrt(D) scale buys Astrid transfer and L_zero but costs SIMBA and
  per-box chi2; within-suite parameter universality (global-fit spread) is a poor proxy for log Z_cross. Do not stack
  Omega_m on the sqrt(D) scale again; Astrid's k = 0.6-1.1 under-suppression (-2.2 sigma) is untouched by scale rescalings.
- **Outcome (T07, SCORED):** log Z_cross **123,336** (predicted 123,300; -656 vs T03, a tie inside sigma_fold 1,790, with
  one fewer nl coefficient: p = 3, 1 nl); marginalized k0 (as predicted; + A gave +635 < sigma_fold). chi2_nu mean/std
  **6.62 / 6.24 (lowest mean so far)**: TNG 3.29 (best so far), SIMBA 1.94, Astrid 17.36 (worst, as predicted), Swift 3.88.
  L_zero 83.2, L_map 18.8. G0_U pass.
- **Lesson:** folds fit A ~ 1.0-2.0, B ~ 1.1-1.9 (both positive, universal sign), k0 ~ 0.37-0.61, so the soft tail is a
  real shared feature. But the TNG/Swift pattern (+0.6 sigma at k ~ 1-3, -0.8..-1.0 at k > 8) is only halved, and Astrid
  k = 0.6-1.1 stays at -2 sigma. Next: keep this 1-nl knee and spend the free nl slot on something per-box (Astrid scatter).

## B 2026-11-11 08:15 — next trial (lane B, agent 3)
- **Dry runs (<scratch>/ml4ps_loop/b3_ag3, 21 nodes, not counted; T03 itself = 124,189 at 21 nodes):** T03 + linear
  -C*D*W2(k/k2) (D^1 low-k component, unbounded-plateau shift) 124,384, chi2 mean 11.1; baryon component without sqrt(D)
  (x1 = k/k1) 123,396; T03 + bump C*D*x2^2/(1+x2^2)^2 at x2 = k sqrt(D)/k2 123,769 (C < 0, chi2 7.94, a tie); T03 + same
  bump at y = k/k2 (no sqrt(D)) **124,699** (+510) but chi2 mean 16.6 (C > 0 in all folds 0.18-0.46). Om-scaled T03
  skipped (= C's T06). In every variant the second coefficient (k1, A or C) gains only +500..+1,000 < sigma_fold.
- **Hypothesis:** C's probe shows the low-k slope of -S vs D is ~2 (TNG/SIMBA) or ~1 (Astrid/Swift); T03's sqrt(D) scale
  forces D^2. A bounded linear bump C*D*y^2/(1+y^2)^2, y = k/k2 (leading k^2, -> 0 at high k, plateau -A D kept) mixes a
  D^1 low-k term into the D^2 one at no nl cost.
- **Expression:** `-D*(A*(w*W2(k sqrt(D)/k1) + (1-w)*W2(k sqrt(D)/k2)) + C*y^2/(1+y^2)^2)`, y = k/k2, w = Omega_b/Omega_m;
  lin A, C; nl k1 (log 0.05-10), k2 (log 0.05-20). p = 4.
- **Prediction:** log Z_cross ~ 124,500 (+0.5k over T03, a tie in sigma_fold); marginalized k2; worst suite Astrid (chi2 ~ 35);
  chi2 mean ~ 15 (expected to be worse than T03: evidence and per-box MAP chi2 disagree on this term).
- **Outcome (T08, SCORED):** log Z_cross **124,430** (predicted 124,500; **+437 over T03, new best but a tie inside
  sigma_fold 1,756**); marginalized k2 (as predicted; + k1 gave +680 < sigma_fold). Per fold vs T03: TNG +67, SIMBA -5,
  Astrid +225, Swift +150. chi2_nu mean/std **12.52 / 10.63** (predicted 15; T03 7.79): TNG 13.34, SIMBA 2.11, Astrid
  29.51 (worst, as predicted), Swift 5.13. L_zero 82.7 (best so far, ~T04), L_map 20.9. G0_U pass. p = 4.
- **Lesson:** C is positive and universal in every fold (0.21-0.31), so a D^1 low-k component is a real shared feature and
  helps evidence/L_zero slightly, but the per-box MAP chi2 gets much worse (TNG 4.0 -> 13.3): with only k2 per box, the
  fixed-scale bump fights the D-scaled knee box by box. Evidence and per-box chi2 now disagree; a second per-box DOF
  (still never promoted: every second coefficient gains +0.5..1.0k < sigma_fold) remains the open bottleneck.

## A 2026-11-11 08:30 — next trial (lane A, agent 3)
- **Dry runs (<scratch>/ml4ps_loop/A3_dry*, 41 nodes, not counted):** T07 + -C*D*W2(k/kc) (D^1 large-scale component,
  own nl scale) 122,772 with k0 only; [k0, kc] reaches 124,713 but +1,941 < sigma_fold so not promoted; folds turn B
  negative (-0.5..-2.2), C ~ 1.6-2.1, kc ~ 0.58 (Astrid k = 0.36 worsens to +1.2). T07 with fractional-deficit scale
  x = k sqrt(D/w)/k0: 119,080 (-4.3k) although chi2 mean 5.92 (best ever), L_map 12.1, L_zero 78.6: evidence and per-box
  chi2 disagree again; the sqrt(D) scale without w is what the evidence likes.
- **Hypothesis:** combine the two best structures: T03's baryon-weighted two-scale sqrt(D) knee (best logZ family) times
  T07's soft-tail factor (A + B x2/(1+x2)) on the large-k2 scale (B was positive and universal in every T07 fold), so the
  plateau keeps falling past the k2 knee (TNG/Swift -0.6..-1.0 sigma at k > 8 in T03) with no new nl coefficient.
  Different from B2's W2^2 sharpening and B3's fixed-scale bump.
- **Expression:** `-D*(w*W2(x1) + (1-w)*W2(x2))*(A + B*x2/(1+x2))`, x_i = k sqrt(D)/k_i, w = Omega_b_over_Omega_m;
  lin A, B; nl k1 (log 0.05-10), k2 (log 0.05-20). p = 4.
- **Prediction:** log Z_cross ~ 124,300; marginalized k2; worst suite Astrid (chi2 ~ 19); chi2 mean ~ 7.0.

## C 2026-11-11 08:40 — next trial (lane C, agent 4)
- **Dry runs (<scratch>/ml4ps_loop/c4_ag4, 41 nodes, not counted):** -D[A W2(k sqrt(D)/k2) + B W2(k/k1)] 123,508 (k2;
  k1 ~ 4-13, L_zero 78.2); rank-1 with baryon component unscaled (x1 = k/k1) 122,813 (A marg.); T03 with sqrt(D/w) scale
  123,013; T03 with extra sigma_8/0.8 factor 122,493; T07 knee with its tail on a D-free scale k1 124,784 (k0 only);
  **T07 + C*D*W2(k/k1) with k1 free over 0.05-30: 124,995, ladder promotes [k0, C]** (first time a 2nd coefficient beats
  sigma_fold). A3's similar dry run found kc ~ 0.58 and no promotion; here the folds put k1 ~ 4.5 (small scales).
- **Hypothesis:** two populations of S: a deficit-reach knee (scale ~ 1/sqrt(D), low-k S ~ D^2) with T07's soft tail, plus
  a D^1 small-scale core term at a D-independent scale (k1 ~ 4-5 h/Mpc, inner-halo redistribution). Marginalizing its
  amplitude C gives the per-box second DOF that every earlier ladder missed.
- **Expression:** `-D*(W2(x)*(A + B*x/(1+x)) + C*W2(k/k1))`, x = k sqrt(D)/k0; lin A, B, C; nl k0 (log 0.01-10), k1 (log
  0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 125,000; marginalized [k0, C]; worst suite Astrid (chi2 ~ 20); chi2 mean ~ 7.8.
- **Outcome (T09, SCORED):** log Z_cross **124,777** (predicted 124,300; **new best, +347 over T08 and +784 over T03**, a tie
  inside sigma_fold 1,728); marginalized k2 (as predicted; + k1 gave +845 < sigma_fold). Per fold vs T03: TNG +157, SIMBA
  +140, Astrid +315, Swift +173 (all up). chi2_nu mean/std **7.67 / 8.19** (predicted 7.0; T03 7.79): TNG 3.47, SIMBA 1.41,
  Astrid 21.76 (worst, as predicted), Swift 4.05. L_zero 85.8, **L_map 10.72** (T03 15.1). G0_U pass. p = 4.
- **Lesson:** the soft tail transfers to the two-scale form: B > 0 in every fold (1.16-2.24, larger in TNG/Swift), A
  0.75-2.02, so a slowly growing plateau is a real shared feature and it improves every fold without hurting chi2 (unlike
  B3's bump). Astrid's k <= 1.1 scatter is still the chi2 sink and k1 still misses promotion by ~0.9k.

## B 2026-11-11 08:55 — next trial (lane B, agent 4)
- **Dry run (<scratch>/ml4ps_loop/b4_ag4/dry, 41 nodes, not counted):** this exact form scored 125,020 with ladder
  promoting [k0, kb], chi2 mean 7.02 / std 5.86, L_zero 76.5, L_map 19.0, G0_U pass.
- **Hypothesis:** T07's 1-nl soft-tail sqrt(D) knee (B > 0 universal, lowest chi2) plus T08's universal D^1 low-k component
  as a *bounded bump* C*D*y^2/(1+y^2)^2, y = k/kb, at its own nl scale kb (T08 tied it to k2). The bump vanishes at high k,
  so the plateau/soft tail is untouched; it only mixes a D^1 large-scale term into the D^2 low-k scaling (Astrid/Swift).
  Different from C4's in-progress trial: C4 adds a step W2(k/k1) that shifts the small-scale plateau (k1 ~ 4.5); this is a
  transient bump that is zero at both ends.
- **Expression:** `-D*(W2(x)*(A + B*x/(1+x)) + C*y^2/(1+y^2)^2)`, x = k sqrt(D)/k0, y = k/kb; lin A, B, C; nl k0 (log
  0.01-10), kb (log 0.05-20). p = 5.
- **Prediction:** log Z_cross ~ 125,000; marginalized [k0, kb]; worst suite Astrid (chi2 ~ 17); chi2 mean ~ 7.0.
- **Outcome (T10, SCORED):** log Z_cross **124,995** (predicted 125,000; **new best**, +218 over T09, +1,002 over T03; still a
  tie inside sigma_fold 1,959); marginalized **[k0, C]** (as predicted; first trial with a promoted 2nd coefficient, +2,413
  > sigma_fold; + B gave +328, not promoted). chi2_nu mean/std **7.81 / 7.37** (predicted 7.8): TNG 4.93, SIMBA 1.82,
  Astrid 20.43 (worst, as predicted), Swift 4.08. **L_zero 74.8 (best so far)**, L_map 16.3 (SIMBA 53.7). G0_U pass. p = 5.
- **Lesson:** folds put the D^1 core at k1 ~ 4.3 h/Mpc (SIMBA 13.5), C ~ 1.8-2.7, and flip T07's tail to B < 0 (-0.3..-2.0):
  the core term takes over the high-k suppression, so B could likely be dropped (p = 4) for a simpler form. A per-box
  amplitude of a D-independent-scale core is the second DOF the ladder accepts; SIMBA L_map is the weak spot.

## C 2026-11-11 09:10 — next trial (lane C, agent 5)
- **Data probes (<scratch>/ml4ps_loop/c5_ag5, not counted):** at fixed D, an extra f_b (or Omega_b/Omega_m) term
  explains < 3 % of the k <= 1.1 variance in every suite and its sign flips across suites (TNG/Swift -, SIMBA/Astrid +),
  so a baryon-fraction add-on is not a shared feature. Low-k log slopes of S between 0.36 and 0.6 are < 2 (1.0-1.7) in
  low-D Astrid/Swift and all SIMBA: the data approach their small-k asymptote softly.
- **Hypothesis:** keep T10 (best, [k0, C] marginalized) but give the D^1 core a *soft* knee, y^2/(1+y)^2 (leading k^2,
  plateau 1, approached as 1 - 2/y): less core suppression at k ~ 1-4 relative to k ~ 10 (TNG/Swift +0.5..+0.6 sigma there,
  ~0 at k > 7), and SIMBA's large k1 (13.5) core still bites inside the grid. Different from B4's bump (zero at both ends)
  and from A's knee-tail factor (on the sqrt(D) knee, not the core).
- **Expression:** `-D*(W2(x)*(A + B*x/(1+x)) + C*(k/k1)**2/(1 + k/k1)**2)`, x = k sqrt(D)/k0; lin A, B, C; nl k0 (log
  0.01-10), k1 (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 125,300; marginalized [k0, C]; worst suite Astrid (chi2 ~ 20); chi2 mean ~ 7.5.
- **Outcome (T11, SCORED):** log Z_cross **125,020** (predicted 125,000; **new best, +25 over C4's T10 at 124,995 and +243
  over T09**, all ties inside sigma_fold 1,991); marginalized **[k0, kb]** (as predicted; kb promoted at +4,443 over k0
  alone; + C gave +413 < sigma_fold). Per fold: TNG 33,285, SIMBA 29,237, Astrid 29,889, Swift 32,608. chi2_nu mean/std
  **7.02 / 5.86 (lowest std so far)**: TNG 4.88, SIMBA 1.90, Astrid 16.98 (worst, as predicted; lowest Astrid so far),
  Swift 4.31. **L_zero 76.5 (best so far)**, L_map 19.0. G0_U pass. p = 5.
- **Lesson:** the bump did not land at large scales: folds put kb ~ 2.2-6.2 h/Mpc (log10 0.35-0.79) and C > 0 except Swift
  (-0.73), and with it B turns negative in 3 folds (-0.6..-2.0; Swift +0.40), so the bump + knee reshape the k ~ 2-6 region
  rather than adding a D^1 low-k term. T10 and T11 agree: a D^1 component at k ~ 4-5 is the per-box second DOF the ladder
  promotes. Next: try the same small-scale D^1 component on T09's two-scale soft-tail form, or drop B (now unstable).
- **Outcome (T12, SCORED but G0_U FAIL, 0.873 < 0.90 of boxes in the end-step band):** log Z_cross **124,627** (predicted
  125,300; -368 vs T10, -392 vs T11, a tie inside sigma_fold 1,666); marginalized **[k0, k1]** (predicted [k0, C]; + A gave
  +709, not promoted). Per fold vs T10: TNG -313, SIMBA +226, Astrid +111, Swift -391. chi2_nu mean/std **7.57 / 7.63**
  (predicted 7.5): TNG 4.65, SIMBA 1.84, Astrid 20.68 (worst, as predicted), Swift **3.13**. L_zero 92.3 (worse), L_map 15.3.
- **Lesson:** the soft core does flatten the TNG/Swift mid-k residuals (k 1-4: +0.5..0.6 -> ~+0.1 sigma), but the folds then
  put k1 at 6-22 with C ~ 6-10, so the slow 1 - 2k1/k approach keeps falling past k = 8.67 and the end step leaves the
  simulation band in > 10 % of boxes (Astrid high-k residual grows to +0.96). A soft core needs a bounded end slope (e.g.
  W2 core, or soft core with a fixed k1 <= 3); the k = 0.36 bin (TNG +3.0, Astrid +2.1) remains the chi2 sink.

## B 2026-11-11 10:05 — next trial (lane B, agent 5)
- **Hypothesis:** T10 (W2 step core at k1 ~ 4.3, [k0, C] promoted) and T11 (bump core at kb ~ 2-6, [k0, kb] promoted) are
  two shapes of the same D^1 small-scale component and tie at ~125.0k. Give that component a linear shape basis at one
  shared nl scale k1: step C*W2(y) + bump E*y^2/(1+y^2)^2 (y = k/k1; both leading k^2, bounded; E sets how fast the core
  saturates, so it can emulate T11 or T10 or a sharper/softer knee box-by-box without a 3rd nl and without the unbounded
  soft-core G0_U failure of C5). Nests T10 (E = 0). Dry runs aborted (machine congestion), so not pre-tested.
- **Expression:** `-D*(W2(x)*(A + B*x/(1+x)) + C*W2(y) + E*y^2/(1+y^2)^2)`, x = k sqrt(D)/k0, y = k/k1; lin A, B, C, E; nl
  k0 (log 0.01-10), k1 (log 0.05-30). p = 6.
- **Prediction:** log Z_cross ~ 125,300; marginalized [k0, C] (maybe + E); worst suite Astrid (chi2 ~ 18); chi2 mean ~ 7.0.
- **Outcome (T13, SCORED):** log Z_cross **124,830** (predicted 125,300; -190 vs T11's 125,020, a tie inside sigma_fold 1,800);
  marginalized **[k1, k0]** (k1 promoted first; + A gave +1,437 < sigma_fold; predicted [k0, C]). chi2_nu mean/std
  **9.92 / 11.08** (predicted 7.0; worse than T10/T11): TNG 3.81, SIMBA 2.69, Astrid 29.10 (worst, as predicted), Swift
  4.11. L_zero 88.6, L_map 25.7 (TNG 1.05 / Swift 1.18 best ever, SIMBA 94.6). G0_U pass. p = 6.
- **Lesson:** folds fit E ~ -3.0..-3.5 (universal) and C ~ 0.8-1.3 at k1 ~ 1.3-4.4, i.e. a core that is negative (an
  enhancement-then-suppression wiggle at k ~ k1) with B < 0 everywhere: the shape basis gives TNG/Swift near-perfect
  per-box fits but Astrid low-k and SIMBA transfer collapse. Extra linear shape freedom on the core does not buy evidence;
  the ~125.0k plateau of T10/T11/T13 suggests the knee+core family is saturated -- look for a feature, not a shape.

## A 2026-11-11 10:20 — next trial (lane A, agent 4)
- **Dry runs (<scratch>/ml4ps_loop/A4_ag4, 41 nodes, not counted):** T09 + D-free bump -C*y^2/(1+y^2)^2 at y = k/k1
  124,753 (k2; C ~ 0.02, Astrid k = 0.36 over-suppressed +7.8; chi2 10.7); T09 + D^1 bump at y = k/k2 (B3's add-on on the
  two-scale soft tail) 124,276 (k2; chi2 13.8, TNG 16.0); T10 with an Omega_m^(1/3)-scaled core 121,759 ([C]; Omega_m
  scaling hurts again, now on the D-free scale too). Probe: low-D Astrid/Swift keep a small large-scale floor
  (S(0.36) ~ -0.001..-0.002) with log slope ~1 between k = 0.36 and 0.6, but no add-on at those scales beats T09/T10.
- **Hypothesis:** in T10 (D^1 W2 core at k1 ~ 4.3, [k0, C] promoted) the core leaks a k^2 tail to large scales
  (-D C k^2/k1^2), which is where the chi2 sink is: k = 0.36 over-suppressed (TNG +2.95, Astrid +1.76 sigma) and TNG/Swift
  +0.5..0.6 at k ~ 1-4. A *sharper* core W2(y)^2 = y^4/(1+y^2)^2 (leading k^4, bounded plateau 1) confines the D^1 inner-halo
  redistribution to small scales and leaves the large scales to the sqrt(D) knee. Opposite direction to C5's soft core
  (which failed G0_U), different from B5's step+bump basis; no new coefficient.
- **Expression:** `-D*(W2(x)*(A + B*x/(1+x)) + C*(k/k1)**4/(1+(k/k1)**2)**2)`, x = k sqrt(D)/k0; lin A, B, C; nl k0 (log
  0.01-10), k1 (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 125,200; marginalized [k0, C]; worst suite Astrid (chi2 ~ 19); chi2 mean ~ 7.3.

## C 2026-11-11 10:35 — next trial (lane C, agent 6)
- **Data probes (<scratch>/ml4ps_loop/c6_ag6, not counted):** S(0.36)/S(0.6) medians: TNG 0.24 (steeper than k^2's
  0.36; 30 % of TNG boxes have S(0.36) > 0), SIMBA 0.43, Astrid 0.40, Swift 0.46, and low-D Astrid/Swift 0.48/0.61: a
  shallow D^1 large-scale floor with its own knee at k ~ 0.5-1, independent of the sqrt(D) knee. Partial correlations of
  S/D with Om, s8, w, f_b at fixed D are |r| < 0.4 with suite-flipping signs (Astrid vs the rest); f_b_M_i have >= 4 %
  NaN in bin 0 and 50-95 % in bins 3-5 (unusable). So no feature route; a p = 6 dry run was aborted (too slow).
- **Hypothesis:** T10's fold fits put the D^1 core at k1 ~ 4.3 in 3 of 4 suites; fix it there (k1 = 4.5, one fewer nl)
  and drop the unstable soft-tail B (sign flips across folds in T10/T11/T13). Spend the freed nl slot on a D^1 transient
  bump E*y^2/(1+y^2)^2 at its own free scale kb, so the large-scale floor is a separate per-box DOF from the core.
  Different from B5 (step + bump sharing one scale) and A4 (sharper W2^2 core, free k1).
- **Expression:** `-D*(A*W2(x) + C*W2(k/4.5) + E*y^2/(1+y^2)^2)`, x = k sqrt(D)/k0, y = k/kb; lin A, C, E; nl k0 (log
  0.01-10), kb (log 0.05-20). p = 5.
- **Prediction:** log Z_cross ~ 124,600; marginalized [k0, kb]; worst suite Astrid (chi2 ~ 18); chi2 mean ~ 7.5.
- **Outcome (T14, SCORED):** log Z_cross **122,238** (predicted 125,200; -2.76k vs T10, -2.78k vs best T11; sigma_fold 2,190);
  marginalized **k0 only** (predicted [k0, C]; + k1 gave +2,188, just under sigma_fold). Per fold vs T11: TNG -508, SIMBA
  +101, Astrid -1,761, Swift -613. chi2_nu mean/std **9.08 / 10.37** (predicted 7.3): TNG 3.77, SIMBA 2.05, Astrid 27.01
  (worst, as predicted), Swift 3.51. L_zero 78.5, L_map 20.7 (SIMBA 70.6). G0_U pass. p = 5.
- **Lesson:** the k^4 core sits at k1 ~ 2.2 h/Mpc (SIMBA 5.7), C ~ 1.3-1.8, and forces B < 0 in every fold (-0.7..-2.8)
  with A ~ 2.6-4.4: the sqrt(D) knee then over-compensates at large scales (Astrid k <= 1.1 now -1.6..-3.1, Swift k = 0.36
  -1.8). The D^1 core's k^2 large-scale tail in T10 is load-bearing (it is the D^1 low-k term), not a leak: keep W2 cores.

## A 2026-11-11 10:50 — next trial (lane A, agent 5)
- **Hypothesis:** T10/T11 promote [k0, C/kb] but the knee scale k0 and the core scale k1 are marginalized/learned
  independently, and their fold means co-move (T10: SIMBA has the largest k0 and k1). Reparametrize so one per-box halo
  scale sets both: core scale k1 = r*k0 (r a D-free ratio, nl), so marginalizing k0 shifts knee and core together box by
  box, and r (the core-to-knee ratio) is a shared population constant. Same function space as T10 (p = 5) but a different
  marginalization geometry; different from C6 (fixed core, bump at free kb) and from all shape changes (B5, C5, A4).
- **Expression:** `-D*(W2(x)*(A + B*x/(1+x)) + C*W2(k/(r*k0)))`, x = k sqrt(D)/k0; lin A, B, C; nl k0 (log 0.01-10), r (log
  1-300). p = 5.
- **Prediction:** log Z_cross ~ 125,300; marginalized [k0, C]; worst suite Astrid (chi2 ~ 19); chi2 mean ~ 7.5.
- **Outcome (T16, SCORED):** log Z_cross **121,245** (predicted 125,300; -3.75k vs T10, -3.78k vs best T11); marginalized
  **[C] only** (predicted [k0, C]; + k0 gave 123,664, +2,420 but under the inflated sigma_fold 2,942). Per fold vs T10: TNG
  -57, **SIMBA -2,569**, Astrid -720, Swift -404. chi2_nu mean/std **8.24 / 8.01** (predicted 7.5): TNG 5.00, SIMBA 1.82,
  Astrid 21.96 (worst, as predicted), Swift 4.16. L_zero 76.0, L_map 17.9. G0_U pass. p = 5.
- **Lesson:** fold fits reproduce T10 (r ~ 5.8-19, SIMBA largest) so the function space is the same, but tying k1 = r k0
  makes per-box k0 drag the core and the ladder loses k0 (SIMBA transfer collapses, sigma_fold grows). Knee and core scales
  are independent per box; keep k1 a separate D-free constant and do not co-marginalize them through a ratio.

## B 2026-11-11 11:00 — next trial (lane B, agent 6)
- **Data probes / dry runs (<scratch>/ml4ps_loop/b6_ag6, 41 nodes, not counted):** f_b_M_3..5 are NaN in 55-95 % of
  boxes (unusable as features). Low-D Astrid boxes (D < 0.01) have S(9.92) ~ +0.07 and S(3.39) ~ +0.01 while every form
  so far is -D*(positive). Dry runs: T10 without B (-D*(A W2(x) + C W2(k/k1))) 123,911 (Astrid chi2 54); + sigma_8 tilt on
  the core 123,834 (no help); T10 + E*f_b*W2(k/k1) (p = 6) 125,803 ([E, k0]); **same without B (p = 5) 126,149 ([k0, k1])**.
- **Hypothesis:** feature, not shape (B5's lesson). Split the baryons into expelled (D) and retained (f_b): the expelled
  fraction gives the sqrt(D)-scaled knee (large-scale reach, low-k S ~ D^2), the retained fraction sets a small-scale core
  term at a D-independent scale k1, so the core amplitude (E f_b - C D) is affine in D with a cosmology (Omega_b/Omega_m)
  floor, and need not vanish at low D. Drops T10's soft-tail B (unstable sign). Equivalent span to (E' w - C' D).
- **Expression:** `-A*D*W2(k sqrt(D)/k0) + (E*f_b - C*D)*W2(k/k1)`; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 126,100; marginalized [k0, k1]; worst suite Astrid (chi2 ~ 50); chi2 mean ~ 15.7.
- **Outcome (T15, SCORED but G0_U FAIL, 0.875 < 0.90):** log Z_cross **123,806** (predicted 124,600; -1,214 vs T11, a tie
  inside sigma_fold 1,938); marginalized **[k0, A]** (predicted [k0, kb]; + C gave +1,147, not promoted). Per fold: TNG
  33,073, SIMBA 29,307, Astrid 29,309, Swift 32,117. chi2_nu mean/std **12.65 / 15.10** (predicted 7.5): TNG 4.95, SIMBA
  2.60, Astrid 38.77 (worst, as predicted; worst since T01), Swift 4.27. L_zero 103.6, L_map 27.2. p = 5.
- **Lesson:** the folds do not use the fixed k = 4.5 core as a suppression: C < 0 (-0.5..-1.8) with a large positive bump
  E ~ 3.3-7.8 at kb ~ 3-7 h/Mpc, i.e. bump and core cancel into a wiggle (same failure as B5's T13), and the residual
  core step pushes the end slope out of band. A free D^1 bump always migrates to k ~ 3-7, never to the large-scale floor;
  stop adding D^1 bumps. Without B the TNG/Swift mid-k + / high-k - pattern returns (TNG +0.9 at k 1-3, -0.7 at 9.9).
- **Outcome (T17, SCORED):** log Z_cross **126,149** (predicted 126,100; **new best, +1,129 over T11's 125,020**, still inside
  sigma_fold ~1.9k); marginalized **[k0, k1]** (as predicted; + E gave +272, not promoted). Per fold: TNG 33,722, SIMBA
  29,787, Astrid 29,725. chi2_nu mean/std **15.74 / 19.64** (worse than T10/T11): TNG 6.54, SIMBA 3.11, Astrid 49.69
  (worst, as predicted), Swift 3.63. L_zero 91.5, L_map 31.2. G0_U pass. p = 5 (one fewer linear coeff than T10/T11).
- **Lesson:** folds fit E < 0 everywhere (-0.42..-0.96) with C 0.3-0.8 at k1 ~ 4-11: the core is a *suppression* with a
  w-like floor -|E| f_b, not condensation, and it lets B go. Evidence rises but Astrid per-box chi2 doubles (the dry run
  with B kept: 125,803 / chi2 8.5). Next: keep the f_b core and find a 1-coeff fix for Astrid's low-k chi2 sink.

## C 2026-11-11 09:20 — next trial (lane C, agent 7)
- **Dry runs (<scratch>/ml4ps_loop/c7_ag7, 41 nodes, not counted):** this exact form 126,937, ladder promotes
  [k2, B, A], chi2 mean/std 8.12 / 8.45 (Astrid 22.6), L_zero 103.3, **L_map 7.78**, G0_U pass; the same without B
  (-D*(A*knee + C*W2(k/kc)), p = 4) 124,124 (k2 only, chi2 8.29).
- **Hypothesis:** T09's two-scale soft-tail form had the baryon scale k1 ~ 0.6-0.75 in every fold (shared, never
  promoted), so fix k1 = 0.68 and spend the freed nl slot on T10's D^1 W2 core at a D-independent scale kc (folds:
  kc ~ 3.8-7.4). Feature-free, different from B6's f_b core (T17) and the earlier single-knee + core forms.
- **Expression:** `-D*((w*W2(k sqrt(D)/0.68) + (1-w)*W2(x2))*(A + B*x2/(1+x2)) + C*W2(k/kc))`, x2 = k sqrt(D)/k2,
  w = Omega_b_over_Omega_m; lin A, B, C; nl k2 (log 0.05-20), kc (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 126,900; marginalized [k2, B, A]; worst suite Astrid (chi2 ~ 22.6); chi2 mean ~ 8.1.

## B 2026-11-11 11:40 — next trial (lane B, agent 7)
- **Dry runs (<scratch>/ml4ps_loop/b7_ag7, 41 nodes, not counted; ladder final in brackets):** T17's core with a pure
  cosmic-fraction amplitude -E w W2(k/k1) (p = 4): [k0] 123,798, chi2 8.21 ([k0,k1] 125,300 not promoted); same + soft
  tail B (p = 5): [k0] 124,029, chi2 8.16; T17 + G D^2 core term (p = 6): [k0] 123,853. T17 with a D-scaled core scale
  y = k (D/0.05)^e / k1: e = 0.125 [k0] 123,811; e = 0.25 [k0] 125,142; **e = 0.5 [k0] 126,660, chi2 8.42** ([k0,k1]
  127,722, +1,062 < sigma_fold 1,924). Note: T17's [k0,k1] promotion is fragile; neighbours land on [k0] only.
- **Hypothesis:** the retained/expelled-baryon core is reached at the same deficit-set scale as the knee: both scales
  ~ 1/sqrt(D) (rank 1's two-scale structure), with T17's affine core amplitude (E f_b - C D). The D-free core scale of T17
  forces the per-box k1 to absorb the D-dependence (Astrid low-k chi2 50); D-scaling it removes that.
- **Expression:** `-A*D*W2(k sqrt(D)/k0) + (E*f_b - C*D)*W2(k (D/0.05)^0.5/k1)`; lin A, C, E; nl k0 (log 0.01-10), k1 (log
  0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 126,660; marginalized [k0]; worst suite Astrid (chi2 ~ 25); chi2 mean ~ 8.4.

## A 2026-11-11 12:00 — next trial (lane A, agent 6)
- **Dry runs (<scratch>/ml4ps_loop/a6_ag6, 41 nodes, not counted):** T10 without B 123,911 / T11 without B 124,044
  (both [k0] only: dropping B loses the 2nd promotion); T10 with free knee D-exponent b and core fixed at 4.3 124,618 ([k0]);
  soft tail moved from the knee to the D-free core (-D(A W2(x) + W2(y)(C + B y/(1+y)))) 125,058 [k0,k1] but Astrid chi2 105;
  T17 with a sharper k^4 knee W2(x)^2 124,183 [k0,E], chi2 7.22, L_map 4.62; **T17 + retained-baryon amplitude on the knee,
  (F f_b - A D) W2(x), 126,023 [k0, E], chi2 mean 11.60 (Astrid 36.5 vs 49.7), L_map 5.85**.
- **Hypothesis:** T17 split the core amplitude into expelled (-C D) and retained (E f_b) baryons; apply the same split to
  the sqrt(D) knee, so each component's amplitude is affine in (D, f_b) = (expelled, retained). The f_b knee term is a
  large-scale D^1-like floor (Astrid/Swift low-D, TNG k = 0.36 over-suppression) that is not a new shape or scale.
  Different from B7 (D-scaled core scale) and C7 (fixed-k1 two-scale + core).
- **Expression:** `(F*f_b - A*D)*W2(k sqrt(D)/k0) + (E*f_b - C*D)*W2(k/k1)`; lin A, C, E, F; nl k0 (log 0.01-10), k1 (log
  0.05-30). p = 6.
- **Prediction:** log Z_cross ~ 126,000 (a tie with T17); marginalized [k0, E]; worst suite Astrid (chi2 ~ 36);
  chi2 mean ~ 11.6 (lower than T17's 15.7), L_map ~ 5.9.
- **Outcome (T18, SCORED):** log Z_cross **126,937** (predicted 126,900; **new best, +788 over B6's T17 at 126,149 and +1,917
  over T11**, ~1.1 sigma_fold 1,698); marginalized **[k2, B, A]** (as predicted; first 3-coefficient promotion; + C gave +265,
  not promoted). Per fold: TNG 33,510, SIMBA 30,098, Astrid 30,480, Swift 32,849 (all above T17). chi2_nu mean/std
  **8.12 / 8.45** (predicted 8.1): TNG 4.92, SIMBA 1.18, Astrid 22.58 (worst), Swift 3.82. L_zero 103.3 (poor), **L_map 7.78**.
  G0_U pass. p = 5, no feature beyond D and w.
- **Lesson:** fixing T09's shared baryon scale (k1 = 0.68) and spending the nl slot on a D^1 W2 core (kc ~ 2.7-7.4) works;
  the per-box DOFs are now k2 and the knee amplitudes A, B (B sign flips across folds, -3.0..+1.3), the core is shared.
  Astrid k = 0.36 (+1.6) / 0.6-1.1 (-1.0..-1.4) remains the chi2 sink; L_zero worsened, so the population mean transfers poorly.

## C 2026-11-11 12:20 — next trial (lane C, agent 8)
- **Probes (<scratch>/ml4ps_loop/c8_ag8, per-box lstsq at T18 fold-mean scales, not counted):** a soft (x^2/(1+x)^2)
  (1-w) knee on T18 raises per-box chi2 in TNG/SIMBA/Astrid (only Swift improves), so keep W2; per-box core amplitudes are
  too poorly determined to test a feature split box by box, so the f_b core can only be judged at population level.
- **Hypothesis:** combine the two best lines. T18 (C7) keeps the per-box DOFs on the two-scale soft-tail knee [k2, B, A]
  and leaves the D^1 core C*W2(k/kc) shared; T17 (B6) showed the shared core transfers better with an affine
  expelled/retained amplitude (E f_b - C D) (+800 even with B kept in B6's T10 dry run). Give T18's shared core that
  amplitude, so the population-mean core carries a retained-baryon floor (L_zero 103 is T18's weak spot). Different from
  A6 (f_b on the knee of T17) and B7 (D-scaled core scale on T17).
- **Expression:** `-D*(w*W2(k sqrt(D)/0.68) + (1-w)*W2(x2))*(A + B*x2/(1+x2)) + (E*f_b - C*D)*W2(k/kc)`, x2 = k sqrt(D)/k2,
  w = Omega_b_over_Omega_m; lin A, B, C, E; nl k2 (log 0.05-20), kc (log 0.05-30). p = 6.
- **Prediction:** log Z_cross ~ 127,300; marginalized [k2, B, A]; worst suite Astrid (chi2 ~ 22); chi2 mean ~ 8.3.
- **Outcome (T19, SCORED):** log Z_cross **126,023** (predicted 126,000; -126 vs T17, -914 vs C7's T18 at 126,937; ties inside
  sigma_fold 1,730); marginalized **[k0, E]** (as predicted; + k1 gave +960, not promoted). Per fold vs T17: TNG -135, SIMBA
  -58, Astrid **+779**, Swift -712. chi2_nu mean/std **11.60 / 14.46** (predicted 11.6; T17 15.74): TNG 4.77, SIMBA 0.93,
  Astrid 36.51 (worst, as predicted), Swift 4.19. L_zero 92.4, **L_map 5.85 (best so far; SIMBA 7.2 vs T17's 105.8)**.
  G0_U pass. p = 6.
- **Lesson:** the knee's f_b term F flips sign across folds (TNG +0.01, SIMBA +0.16, Astrid -0.51, Swift -0.35), so it is not
  a shared population feature; it buys Astrid transfer, per-box chi2 and especially L_map (SIMBA), but no evidence. The
  retained-baryon (E f_b) core is the promoted per-box DOF. For the paper, T19 is the L_map/chi2 compromise of T17.

## A 2026-11-11 — next trial (lane A, agent 7)
- **Hypothesis:** T18 (best, [k2, B, A]) multiplies the whole knee, including the fixed large-scale baryon component
  w*W2(k sqrt(D)/0.68), by (A + B x2/(1+x2)), so every per-box DOF moves k <= 1.1 and k > 3 together; the k <= 1.1 bins
  remain the chi2 sink (TNG k=0.36 +3.2; Astrid +1.6 at 0.36, -1.0..-1.4 at 0.6-1.1). Give the large-scale baryon knee its
  own linear amplitude G and keep the soft tail on the (1-w) k2 component only, so G can be marginalized as a per-box
  large-scale amplitude. No new scale or feature; different from C8 (f_b core on T18) and B7 (D-scaled core scale on T17).
- **Expression:** `-D*(G*w*W2(k sqrt(D)/0.68) + (1-w)*W2(x2)*(A + B*x2/(1+x2)) + C*W2(k/kc))`, x2 = k sqrt(D)/k2,
  w = Omega_b_over_Omega_m; lin G, A, B, C; nl k2 (log 0.05-20), kc (log 0.05-30). p = 6. Not dry-run (machine congested).
- **Prediction:** log Z_cross ~ 127,300; marginalized [k2, B, A] + maybe G; worst suite Astrid (chi2 ~ 19); chi2 mean ~ 7.5.
- **Outcome (T20, SCORED):** log Z_cross **126,660** (predicted 126,660, exactly the dry run; +511 over T17, -277 vs T18's
  126,937, all ties inside sigma_fold 1,924); marginalized **[k0] only** (as predicted; + k1 gave +1,062, not promoted).
  Per fold: TNG 33,846, SIMBA 30,162, Astrid 29,937, Swift 32,716. chi2_nu mean/std **8.42 / 9.55** (T17 15.74 / 19.64):
  TNG 3.94, SIMBA 0.83, Astrid 24.81 (worst, as predicted; T17 49.7), Swift 4.11. **L_map 4.91 (best so far)**, L_zero
  100.2 (poor). G0_U pass (0.998). p = 5.
- **Lesson:** D-scaling the f_b core's scale (k1/sqrt(D), like the knee) halves Astrid's per-box chi2 and lifts evidence with
  only one marginalized coefficient; folds keep E < 0 (-0.4..-1.05), C 0.4-1.3. The remaining Astrid sink is now k = 0.36
  (+2.1 sigma, over-suppressed) and k > 3 (+0.5..0.7); L_zero is the weak metric (population mean transfer).
- **Outcome (T21, SCORED):** log Z_cross **125,224** (predicted 127,300; -1,712 vs best T18 at 126,937, a tie inside the
  inflated sigma_fold 2,522); marginalized **[k2] only** (predicted [k2, B, A]; + E gave 127,401, +2,177 but under
  sigma_fold, so not promoted). Per fold vs T18: TNG +164, SIMBA +22, **Astrid -2,148**, Swift +250. chi2_nu mean/std
  **9.05 / 9.71** (predicted 8.3): TNG 4.80, SIMBA 1.47, Astrid 25.73 (worst, as predicted), Swift 4.22. L_zero 90.7 (better
  than T18's 103.3), L_map 10.25. G0_U pass. p = 6.
- **Lesson:** folds fit E ~ -0.57..-1.0 and C <= 0 (except Astrid +0.15): the core becomes a pure retained-baryon
  suppression -|E| f_b W2(k/kc), and B shrinks/flips; per-box E then competes with A/B, sigma_fold inflates and the ladder
  stops at [k2]. The f_b core helps L_zero and 3 folds but costs Astrid; on T18 the D^1 core is better (T18's shared C
  ladder is fragile to added linear coefficients -- prefer p = 5).

## B 2026-11-11 13:10 — next trial (lane B, agent 8)
- **Dry run (<scratch>/ml4ps_loop/b8_ag8, 41 nodes, not counted, ~10 min on the congested machine):** this exact form
  125,236, ladder [k2, kc], chi2 mean/std 6.96 / 7.34 (TNG 3.43, SIMBA 1.16, Astrid 19.56, Swift 3.70), L_zero 98.0,
  L_map 7.94, G0_U pass.
- **Hypothesis:** apply B7's lesson (T20: D-scaling the core scale halves Astrid chi2 on T17) to C7's best T18: the D^1 core
  is reached at a deficit-set scale, W2(k (D/0.05)^0.5/kc), so all three components (fixed baryon knee 0.68, free k2 knee,
  core) scale as 1/sqrt(D). Keeps T18's p = 5, no f_b (C8's T21 showed f_b on T18's core costs Astrid). Different from A7
  (separate amplitude G on the 0.68 knee). Expect lower chi2 than T18 (Astrid k > 3 tail) but the ladder may change.
- **Expression:** `-D*((w*W2(k sqrt(D)/0.68) + (1-w)*W2(x2))*(A + B*x2/(1+x2)) + C*W2(k (D/0.05)^0.5/kc))`,
  x2 = k sqrt(D)/k2, w = Omega_b_over_Omega_m; lin A, B, C; nl k2 (log 0.05-20), kc (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 125,240 (the dry run; below T18, a tie inside sigma_fold); marginalized [k2, kc]; worst suite
  Astrid (chi2 ~ 19.6); chi2 mean ~ 7.0 (lowest of the T18 line).

## C 2026-11-11 09:40 — next trial (lane C, agent 9)
- **Dry runs (<scratch>/ml4ps_loop/c9_ag9, 41 nodes, not counted):** extending B7's D-exponent scan of T17's f_b/D core
  scale y = k (D/0.05)^e / k1 (B7: e = 0.125/0.25/0.5 -> 123.8k/125.1k/126.7k): **e = 0.75 127,309 [k1, A]** (+ C 127,706
  not promoted; chi2 mean 13.35, Astrid 41.9); e = 1.0 124,744 [k0] (chi2 9.96). Peak near e ~ 0.75.
- **Hypothesis:** the retained/expelled-baryon core reach shrinks faster with the deficit than the sqrt(D) knee (core scale
  ~ D^-0.75): high-D boxes push the core to large scales, low-D boxes confine it to small scales. Different from A7 (G
  split amplitude on T18) and B8 (D-scaled core on T18): this stays on the single-knee T20 family, one number changed.
- **Expression:** `-A*D*W2(k sqrt(D)/k0) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`; lin A, C, E; nl k0 (log 0.01-10), k1
  (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 127,300; marginalized [k1, A]; worst suite Astrid (chi2 ~ 42); chi2 mean ~ 13.4.
- **Outcome (T22, SCORED):** log Z_cross **120,979** (predicted 127,300; -5.96k vs T18); marginalized **[C] only** (predicted
  [k2, B, A]; + k2 gave 124,029, +3,050 but under the inflated sigma_fold 3,557). Per fold: TNG 33,692, SIMBA 26,713 (-3.4k
  vs T18), Astrid 27,679 (-2.8k), Swift 32,894. chi2_nu mean/std **9.21 / 7.87** (predicted 7.5): TNG 7.90, SIMBA 1.32,
  Astrid 22.22 (worst, as predicted), Swift 5.39. L_zero 84.2 (T18 103.3), L_map 9.97. G0_U pass. p = 6.
- **Lesson:** untying the large-scale knee from (A + B x2/(1+x2)) lets k2 collapse onto the baryon knee (k2 ~ 0.5-1 vs T18's
  larger k2) with G ~ 3-4, A ~ 4-7, B ~ -4..-10: the two knees become degenerate, sigma_fold doubles and the ladder stalls
  at [C]. T18's product structure (tail factor on both knees) is load-bearing regularization; do not split its amplitudes.
- **Outcome (T23, SCORED):** log Z_cross **127,309** (predicted 127,300, the dry run; **new best, +372 over C7's T18 at 126,937
  and +649 over T20**, a tie inside sigma_fold 2,014); marginalized **[k1, A]** (as predicted; + C gave +396, not promoted).
  Per fold: TNG 33,932, SIMBA 29,625, Astrid 30,677 (best Astrid fold so far), Swift 33,076. chi2_nu mean/std **13.35 /
  16.50** (predicted 13.4): TNG 5.04, SIMBA 2.22, Astrid 41.88 (worst), Swift 4.26. L_zero 76.6 (best of the f_b-core
  line), L_map 20.83 (SIMBA 62.8). G0_U pass. p = 5.
- **Lesson:** the D-exponent of the core scale peaks near 0.75 for evidence (and moves the promoted DOF from k0 to the core
  scale k1), but per-box chi2 pays: Astrid k = 0.36 is now +4.3 sigma over-suppressed (T20: +2.1). e = 0.5 (T20) is the
  chi2/L_map compromise; e = 0.75 the evidence winner. Next: a fix for Astrid's k = 0.36 bin on T23 that does not touch e.
- **Outcome (T24, SCORED):** log Z_cross **125,236** (predicted 125,240 = the dry run; -1,701 vs T18, below current best T23
  127,309; sigma_fold 1,576); marginalized **[k2, kc]** (as predicted; + A gave +467, not promoted). Per fold: TNG 33,054,
  SIMBA 30,008, Astrid 29,945, Swift 32,230. chi2_nu mean/std **6.96 / 7.34** (predicted 7.0; lowest of the T18 line):
  TNG 3.43, SIMBA 1.16, Astrid 19.56 (worst, as predicted), Swift 3.70. L_zero 98.0, L_map 7.94 (SIMBA 25.3). G0_U pass. p = 5.
- **Lesson:** D-scaling T18's core makes kc a per-box DOF (fold kc ~ 1.8-5 in (D/0.05)^0.5 units) and lowers chi2 in every
  suite, but the ladder loses T18's [B, A] promotion and evidence drops ~1.7k; Swift's fold kills the core (C ~ 0.09, B
  flips +1.85). On T18 keep the core D-free; D-scaling the core helps per-box fits, not cross-suite evidence.

## C 2026-11-11 09:55 — next trial (lane C, agent 10)
- **Dry runs (<scratch>/ml4ps_loop/c10_ag10, 41 nodes, not counted), all on T23's D^0.75 f_b/D core:** knee scale
  (D/0.05)^0.25 126,726 [k1, A]; knee scale (D/0.05)^0.75 (same as the core) 125,060 [k0] (+k1 127,588 not promoted),
  chi2 12.38; **sharp k^4 knee W2(x)^2, x = k sqrt(D)/k0: 127,095 [k1, A], chi2 mean 9.77 (TNG 2.34, SIMBA 2.32, Astrid
  29.58, Swift 4.83)** vs T23 13.35 (Astrid 41.9).
- **Hypothesis:** T23's chi2 sink is k = 0.36 over-suppression (TNG +3.4, Astrid +4.3 sigma) with Astrid under-suppressed at
  k 0.86-1.4: the data rise from k = 0.36 to 1 faster than k^2 (C6 probe: TNG S(0.36)/S(0.6) = 0.24 < 0.36). On T23 the
  D^1-ish k^2 large-scale term is carried by the core, so the sqrt(D) knee can be sharp (leading k^4, plateau 1) and stop
  leaking k^2 to the largest scales. Different from B8 (D-scaled core on T18); A6's k^4 knee was on T17's D-free core.
- **Expression:** `-A*D*W2(x)^2 + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`, x = k sqrt(D)/k0, W2(x)^2 = x^4/(1+x^2)^2; lin A,
  C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 127,100 (the dry run; a tie with T23); marginalized [k1, A]; worst suite Astrid (chi2 ~ 29.6);
  chi2 mean ~ 9.8 (much lower than T23's 13.35).

## A 2026-11-11 — next trial (lane A, agent 8)
- **Dry runs (<scratch>/ml4ps_loop/a8_ag8, 41 nodes, not counted):** T23 with knee fixed at k0 = 0.46 and the core
  D-exponent e free (nl, lin 0-1.5): 124,807 [k1, E], chi2 9.69 (folds e ~ 0.50-0.99, Astrid 0.50): a per-box core exponent
  does not pay. **T23 + T07's soft tail on the knee (p = 6): 127,407 [k0, k1]** (+ E 127,995 not promoted), chi2 mean 12.47
  (TNG 4.27, SIMBA 1.85, Astrid 39.4, Swift 4.32); B > 0 in every fold (0.25-1.02), unlike T10/T11 where the core flipped it.
- **Hypothesis:** T23's TNG/Swift residuals show the classic soft-tail pattern (+0.6..0.8 sigma at k ~ 1-4, -0.3..-0.4 at
  k > 8); with the f_b/D core at a D^0.75-scaled scale the soft tail B x/(1+x) on the sqrt(D) knee is again a shared,
  positive feature, and it moves the promoted per-box DOF back to [k0, k1]. Different from B8 (T18 line) and C9 (exponent).
- **Expression:** `-D*W2(x)*(A + B*x/(1+x)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`, x = k sqrt(D)/k0; lin A, B, C, E;
  nl k0 (log 0.01-10), k1 (log 0.05-30). p = 6.
- **Prediction:** log Z_cross ~ 127,400 (the dry run; a tie with T23); marginalized [k0, k1]; worst suite Astrid (chi2 ~ 39);
  chi2 mean ~ 12.5.
- **Outcome (T25, SCORED):** log Z_cross **127,095** (predicted 127,100, the dry run; -214 vs best T23 at 127,309, a tie inside
  sigma_fold 2,009); marginalized **[k1, A]** (as predicted; + E gave +193, not promoted). Per fold vs T23: TNG -117, SIMBA
  -35, Astrid -88, Swift +26. chi2_nu mean/std **9.77 / 11.48** (predicted 9.8; T23 13.35 / 16.50): TNG **2.34 (best so
  far)**, SIMBA 2.32, Astrid 29.58 (worst, as predicted; T23 41.9), Swift 4.83. L_zero 83.3, L_map 30.45 (SIMBA 104.1, poor).
  G0_U pass (1.0). p = 5.
- **Lesson:** the k^4 knee fixes the k = 0.36 over-suppression in TNG (+3.4 -> +0.3) and Astrid (+4.3 -> +0.7) at equal
  evidence, but over-corrects SIMBA (-2.0 at k <= 1.1) and Swift (-2.9 at k = 0.36), and SIMBA L_map collapses. The knee
  sharpness is suite-dependent (k^2 for SIMBA/Swift, k^4 for TNG/Astrid): a linear mix A W2(x)^2 + B W2(x) is the next try.

## B 2026-11-11 14:10 — next trial (lane B, agent 9)
- **Dry runs (<scratch>/ml4ps_loop/b9_ag9, 41 nodes, not counted; knee/core shape on T23, ladder final in brackets):**
  sharp knee x^4/(1+x^2)^2: 127,095 [k1, A], chi2 9.77 (TNG 2.34, Astrid 29.6); sharp core y^4/(1+y^2)^2: 127,124 [k1, C],
  chi2 12.76; both sharp: 127,047 [k0, k1], chi2 10.17 (Swift 6.65); **k^3 knee x^3/(1+x^2)^1.5: 127,301 [k1, A], chi2 9.93**.
- **Hypothesis:** T23's chi2 sink is k = 0.36 over-suppression (TNG +3.4, Astrid +4.3) with Astrid under-suppressed at
  k ~ 0.9-1.4: the data rise faster than k^2 from 0.36 to 1. Sharpen only the sqrt(D) knee to leading k^3 (plateau 1),
  keeping the D-scaled f_b/D core's k^2 tail as the low-k floor (A4/T14: core k^2 tail is load-bearing). No new coefficient.
- **Expression:** `-A*D*x^3/(1+x^2)^1.5 + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`, x = k sqrt(D)/k0; lin A, C, E; nl k0 (log
  0.01-10), k1 (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 127,300 (a tie with T23); marginalized [k1, A]; worst suite Astrid (chi2 ~ 31); chi2 mean ~ 9.9.
- **Outcome (T26, SCORED):** log Z_cross **127,407** (predicted 127,400 = the dry run; **new best, +98 over T23's 127,309**, a
  tie inside sigma_fold 1,757); marginalized **[k0, k1]** (as predicted; + E gave +588, not promoted). Per fold vs T23: TNG
  -67, SIMBA +348, Astrid +197, Swift -380. chi2_nu mean/std **12.47 / 15.60** (predicted 12.5; T23 13.35 / 16.50): TNG 4.27,
  SIMBA 1.85, Astrid 39.43 (worst, as predicted), Swift 4.32. L_zero 76.8 (~T23), **L_map 13.59 (T23 20.83)**. G0_U pass. p = 6.
- **Lesson:** with the D^0.75-scaled f_b/D core the soft tail is a shared positive feature (B 0.25-1.02) and improves chi2,
  L_map and evidence slightly at once; a free per-box core exponent (dry run) loses 2.5k. Astrid k = 0.36 is still the sink;
  the T23 line is at p = 6 now, so further gains must come from swapping a coefficient, not adding one.

## A 2026-11-11 — next trial (lane A, agent 9)
- **Probe (<scratch>/ml4ps_loop/a9_ag9, not counted):** data S(0.36)/S(0.86) medians TNG 0.10, Astrid 0.23, Swift 0.26,
  SIMBA 0.30 (pure k^2: 0.175); the k = 0.36 excess over k^2 extrapolation correlates only weakly and with suite-dependent sign
  with D, f_b, Omega_m, sigma_8 (|r| <= 0.6, mostly D), so no clean feature fix: the knee sharpness is a per-box/suite DOF.
  f_b_M high-mass bins are 80-95 % NaN (unusable).
- **Hypothesis:** C10's lesson (T25): the knee sharpness is suite-dependent (k^4 helps TNG/Astrid at k = 0.36, hurts
  SIMBA/Swift). Use a linear mix of the soft and sharp sqrt(D) knee, -D (A W2(x) + B W2(x)^2), on T23's D^0.75 f_b/D core, so
  the ladder can marginalize A (the k^2 low-k amplitude) per box while B sets the sharp plateau. Different from B9 (fixed k^3
  knee, p = 5) and T26 (x/(1+x) soft tail, which changes the k^3 term, not the k^2/k^4 balance).
- **Expression:** `-D*(A*W2(x) + B*W2(x)^2) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`, x = k sqrt(D)/k0; lin A, B, C, E;
  nl k0 (log 0.01-10), k1 (log 0.05-30). p = 6. Not dry-run.
- **Prediction:** log Z_cross ~ 127,500; marginalized [k1, A] (or [k0, k1]); worst suite Astrid (chi2 ~ 30); chi2 mean ~ 9.5.

## C 2026-11-11 10:15 — next trial (lane C, agent 11)
- **Dry runs (<scratch>/ml4ps_loop/c11_ag11, 41 nodes, not counted), both on T23's D^0.75 f_b/D core:** rank-1's
  two-scale baryon-weighted knee (w W2(k sqrt(D)/0.68) + (1-w) W2(k sqrt(D)/k0)): **127,993 [k1, k0]** (+ E 128,340 not
  promoted), chi2 13.18 (TNG 6.94, SIMBA 1.10, Astrid 38.87, Swift 5.79); w-weighted sharpness mix
  (w W2(x) + (1-w) W2(x)^2, one k0): 125,798 [k1], chi2 11.64 (TNG 2.86, Astrid 37.9).
- **Hypothesis:** T23 has the single sqrt(D) knee; the paper's rank-1 structure says the cosmic-baryon fraction w of the knee
  sits at a fixed shared baryon scale (0.68, C7's T18) and the rest at a free halo scale. On T23's f_b/D core this frees k0
  to become a per-box DOF together with the core scale (ladder [k1, k0] vs T23's [k1, A]). No new coefficient (p = 5).
  Different from B9 (k^3 knee) and A8 (soft tail).
- **Expression:** `-A*D*(w*W2(k sqrt(D)/0.68) + (1-w)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  w = Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 128,000 (the dry run); marginalized [k1, k0]; worst suite Astrid (chi2 ~ 39); chi2 mean ~ 13.2.
- **Outcome (T27, SCORED):** log Z_cross **127,301** (predicted 127,300 = the dry run; -8 vs T23, +206 vs T25's k^4 knee, -106
  vs A8's T26 at 127,407; all ties inside sigma_fold 2,009); marginalized **[k1, A]** (as predicted; + E gave +281, not
  promoted). Per fold: TNG 33,888, SIMBA 29,659, Astrid 30,622, Swift 33,133. chi2_nu mean/std **9.93 / 12.21** (predicted
  9.9; T23 13.35 / 16.50): TNG 2.42, SIMBA 2.11, Astrid 31.04 (worst, as predicted), Swift 4.14. **L_zero 73.67 (best so
  far)**, L_map 19.65 (SIMBA 60.6; T25 30.45). G0_U pass (1.0). p = 5.
- **Lesson:** the k^3 knee keeps T23's evidence and most of T25's chi2 gain (k = 0.36: TNG +3.4 -> +1.1, Astrid +4.3 -> +1.8)
  while over-correcting Swift less (-2.0 at 0.36 vs -2.9) and halving T25's SIMBA L_map loss; SIMBA is still under-suppressed
  at k <= 1.4 (-1.5..-1.8). Knee sharpness ~k^3 is the shared compromise; T26's soft tail on this k^3 knee is the obvious combo.

## B 2026-11-11 15:00 — next trial (lane B, agent 10)
- **Dry run (<scratch>/ml4ps_loop/b10_ag10, 41 nodes, not counted, ~8 min):** this exact form 127,176, ladder [k1, C]
  (+ B 127,794 not promoted, sigma_fold 1,994), chi2 mean 9.86 (TNG 3.39, SIMBA 1.79, Astrid 29.22, Swift 5.03). Fold B:
  TNG 0.70, SIMBA 1.05, Astrid 0.01, Swift 1.31 (Astrid does not want the soft tail).
- **Hypothesis:** the "obvious combo" from T27's lesson: B9's k^3 sqrt(D) knee (T27; fixes k = 0.36 over-suppression, best
  L_zero) times A8's soft tail (A + B x/(1+x)) (T26; best evidence, TNG/Swift mid-k + / high-k - pattern), on T23's
  D^0.75-scaled f_b/D core. Different from A9 (k^2/k^4 sharpness mix) and C11 (two-scale baryon-weighted knee).
- **Expression:** `-D*x^3/(1+x^2)^1.5*(A + B*x/(1+x)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`, x = k sqrt(D)/k0; lin A, B,
  C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 6.
- **Prediction:** log Z_cross ~ 127,180 (the dry run; a tie with T26/T27, the two gains do not stack); marginalized [k1, C];
  worst suite Astrid (chi2 ~ 29); chi2 mean ~ 9.9.
- **Outcome (T29, SCORED):** log Z_cross **127,993** (predicted 128,000 = the dry run; **new best, +586 over T26's 127,407 and
  +684 over T23**, a tie inside sigma_fold 2,024); marginalized **[k1, k0]** (as predicted; + E gave +347, not promoted). Per
  fold vs T23: TNG +134, SIMBA +120, Astrid +206, Swift +223 (all up). chi2_nu mean/std **13.18 / 14.99** (predicted 13.2):
  TNG 6.94, SIMBA 1.10, Astrid 38.87 (worst, as predicted), Swift 5.79. L_zero 73.2 (best of the f_b-core line), **L_map 8.57**
  (T23 20.8). G0_U pass (1.0). p = 5. Fold mu: log k0 ~ -0.47..-0.35, k1 ~ 0.57-0.83 (log10), E -0.45..-0.92.
- **Lesson:** the rank-1 w/(1-w) two-scale knee transfers better than the single sqrt(D) knee on the D^0.75 f_b/D core (all
  4 folds up, L_map and L_zero improve) but TNG k = 0.36 gets worse (+4.5 sigma, +2.2 at 0.6: the fixed 0.68 baryon knee
  over-suppresses large scales). Next: sharpen only the fixed 0.68 component (k^3/k^4) or the B9 k^3 shape on the (1-w) part.

## C 2026-11-11 10:31 — next trial (lane C, agent 12)
- **Hypothesis:** T29 (best, 127,993 [k1, k0]) over-suppresses k = 0.36 in TNG (+4.5 sigma), Astrid (+4.1) and Swift
  (+1.7), only SIMBA is slightly under (-0.3): the fixed 0.68 baryon component of the rank-1 two-scale knee leaks k^2 to the
  largest scales. Sharpen only that fixed component to B9's k^3 shape, u^3/(1+u^2)^1.5 with u = k sqrt(D)/0.68 (plateau 1),
  keeping the free (1-w) W2 knee (per-box k0) and T23's D^0.75 f_b/D core (per-box k1) untouched. No new coefficient
  (p = 5). Different from B10 (k^3 single knee x soft tail on T23) and A9 (k^2/k^4 amplitude mix).
- **Expression:** `-A*D*(w*u^3/(1+u^2)^1.5 + (1-w)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  u = k sqrt(D)/0.68, w = Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 5. Not dry-run.
- **Prediction:** log Z_cross ~ 128,200; marginalized [k1, k0]; worst suite Astrid (chi2 ~ 32); chi2 mean ~ 10.5.
- **Outcome (T28, lane A agent 9, SCORED):** log Z_cross **127,097** (predicted 127,500; -310 vs T26 127,407, -896 vs T29
  127,993, ties inside sigma_fold 1,996); marginalized **[k1, C]** (predicted [k1, A]; + k0 gave +676, not promoted). Per fold:
  TNG 33,914, SIMBA 29,581, Astrid 30,668, Swift 32,934. chi2_nu mean/std **12.91 / 15.67** (predicted 9.5): TNG 5.60, SIMBA
  2.02, Astrid 39.96 (worst, as predicted), Swift 4.08. L_zero **69.8 (best so far)**, L_map 14.49. G0_U pass. p = 6.
- **Lesson:** the fold-mean sharp-knee weight B is ~0 (TNG -0.34, SIMBA +0.29, Astrid 0.00, Swift +0.03): the population does
  not want a W2^2 component on top of W2, so the soft/sharp linear mix collapses to T23 and the extra coefficient only costs
  evidence. Knee sharpness should be set by a fixed exponent (B9's k^3 / T29) not a free linear mix.

## A 2026-11-11 — next trial (lane A, agent 10)
- **Hypothesis:** T29 (best, 127,993 [k1, k0]) still shows the soft-tail pattern in TNG (+0.7..1.0 sigma at k 1.6-3.4,
  -0.17..-0.27 at k >= 8.7) and Swift (+0.5..0.8 at k 1.6-2.9, -0.34..-0.43 at k >= 8.7). A8's T26 showed the soft tail
  (A + B x/(1+x)) is a shared positive feature on the D^0.75 f_b/D core; T18/T22 showed the tail factor multiplying the whole
  two-scale knee is load-bearing. Combine: T29's rank-1 two-scale knee times (A + B x0/(1+x0)), x0 = k sqrt(D)/k0. p = 6.
  Different from C12 (k^3 on the fixed 0.68 component) and B10 (k^3 single knee x soft tail on T23). Not dry-run.
- **Expression:** `-D*(w*W2(k sqrt(D)/0.68) + (1-w)*W2(x0))*(A + B*x0/(1+x0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  w = Omega_b_over_Omega_m; lin A, B, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30).
- **Prediction:** log Z_cross ~ 128,300; marginalized [k1, k0]; worst suite Astrid (chi2 ~ 36); chi2 mean ~ 11.5.
- **Outcome (T30, SCORED):** log Z_cross **127,176** (predicted 127,180 = the dry run; -231 vs T26's 127,407, -125 vs T27, ties
  inside sigma_fold 2,117); marginalized **[k1, C]** (as predicted; + B gave +618, not promoted). chi2_nu mean/std **9.86 /
  11.24** (predicted 9.9): TNG 3.39, SIMBA 1.79, Astrid 29.22 (worst, as predicted; lowest Astrid of the T23 line), Swift 5.03.
  L_zero 77.95, L_map 15.03. G0_U pass. p = 6.
- **Lesson:** the k^3 knee and the soft tail do not stack: once the knee is k^3, the soft tail is suite-dependent (Astrid B ~ 0,
  Swift 1.3), so it spends a coefficient for chi2/L_map trade-offs only. On the T23 line the knee shape is saturated; the
  Astrid k = 0.36 sink needs a new physical ingredient, not another knee-shape tweak.
- **Outcome (T31, lane A agent 10, SCORED):** log Z_cross **127,927** (predicted 128,300; -66 vs best T29 127,993, a tie inside
  sigma_fold 1,975); marginalized **[E, k0]** (predicted [k1, k0]; + k1 gave 129,146, +1,219 but under sigma_fold, not
  promoted). Per fold: TNG 34,024, SIMBA 30,317, Astrid 30,262, Swift 33,325. chi2_nu mean/std **12.47 / 12.91** (predicted
  11.5; T29 13.18 / 14.99): TNG 8.03, SIMBA 1.87, Astrid 34.50 (worst, as predicted; T29 38.9), Swift 5.45. L_zero 75.9,
  L_map 14.33 (SIMBA 41.1; T29 8.57). G0_U pass. p = 6.
- **Lesson:** fold B is suite-split (SIMBA/Swift ~1.1, TNG 0.33, Astrid -0.17), so the soft tail on T29's two-scale knee does
  not stack: evidence ties, Astrid chi2 drops but TNG k = 0.36 worsens (+4.6 sigma) and SIMBA L_map doubles. T29's p = 5 stays
  preferable; the TNG/Astrid k = 0.36 over-suppression from the fixed 0.68 knee (C12's target) is the real lever.

## A 2026-11-11 — next trial (lane A, agent 11)
- **Hypothesis:** T29 (best, 127,993 [k1, k0]) over-suppresses k = 0.36 (TNG +4.5, Astrid +4.1, Swift +1.7 sigma) with
  Astrid under-suppressed at k 0.86-1.4: the low-k rise is steeper than k^2. T27 showed B9's k^3 single knee fixes k = 0.36
  (TNG +3.4 -> +1.1, Astrid +4.3 -> +1.8) at equal evidence. Combine: keep T29's rank-1 two-scale structure but put the k^3
  shape on the dominant free (1-w) halo knee (per-box k0), leaving the fixed w 0.68 baryon knee W2 and T23's D^0.75 f_b/D
  core unchanged. p = 5. Mirror of C12 (who sharpens only the small-weight fixed 0.68 component, still in progress).
- **Expression:** `-A*D*(w*W2(k sqrt(D)/0.68) + (1-w)*x^3/(1+x^2)^1.5) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  x = k sqrt(D)/k0, w = Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). Not dry-run.
- **Prediction:** log Z_cross ~ 128,300; marginalized [k1, k0]; worst suite Astrid (chi2 ~ 31); chi2 mean ~ 10.5.

## C 2026-11-11 — outcome of lane C, agent 12 (k^3 fixed 0.68 baryon knee on T29)
- **Outcome (T32, SCORED):** log Z_cross **127,387** (predicted 128,200; -606 vs best T29 at 127,993, a tie inside sigma_fold
  1,808); marginalized **[k1, k0]** (as predicted; + E gave 128,147, +760, not promoted). Per fold vs T29: TNG -574, SIMBA
  +371, Astrid -425, Swift +22. chi2_nu mean/std **14.12 / 17.11** (predicted 10.5; T29 13.18): TNG 5.26 (T29 6.94), SIMBA
  1.94, Astrid 43.66 (worst, as predicted; T29 38.9), Swift 5.63. L_zero 73.4, L_map 14.99 (SIMBA 43.2; T29 8.57). G0_U pass.
  p = 5. Fold mu: log k0 -0.52..-0.36, k1 0.56-0.79 (log10), E -0.43..-0.85.
- **Lesson:** sharpening the fixed 0.68 component does not fix k = 0.36: TNG eases (+4.5 -> +3.7) but Astrid gets worse
  (+4.1 -> +4.6) and SIMBA flips to under-suppressed at k <= 1.1 (-1.3..-1.5), Swift unchanged (+1.7). The k = 0.36 excess is
  carried by the D^0.75 core's k^2 tail, not the baryon knee; leave the 0.68 component at W2 and target the core's low-k tail.

## C 2026-11-11 — next trial (lane C, agent 13)
- **Hypothesis:** C12's lesson (T32): T29's k = 0.36 over-suppression (TNG +4.5, Astrid +4.1, Swift +1.7 sigma; Astrid
  mean sq 150) is carried by the D^0.75-scaled f_b/D core's k^2 tail, not the 0.68 baryon knee. Sharpen only the core's
  low-k onset (k^3: y^3/(1+y^2)^1.5, or k^4: W2(y)^2; plateau 1, end slope unchanged) on T29, leaving both W2 knees as the
  k^2 large-scale carriers. Different from A11 (k^3 on the free (1-w) knee) and C12 (k^3 on the fixed 0.68 knee). p = 5.
  Dry-running k^3 vs k^4 core first (<scratch>/ml4ps_loop/c13_ag13); expression and prediction to follow.

## B 2026-11-11 16:00 — next trial (lane B, agent 11)
- **Dry runs (<scratch>/ml4ps_loop/b11_ag11, 41 nodes, not counted), all on T29:** fixed baryon knee scale 1.0:
  127,024 [k1, k0] (chi2 14.35); 0.45: 125,430 [k0] (+k1 127,933 not promoted; chi2 14.88) -> 0.68 is the optimum, keep it.
  **k^3 on the free (1-w) knee (= A11's form): 127,613 [k0]** (+A 127,783 not promoted), chi2 13.09 (TNG 9.34, SIMBA 3.04,
  Astrid 32.5, Swift 7.51): a tie, not a gain. **Core D-exponent 0.6 (vs 0.75): 127,985 [k1, k0]** (+A 128,459 not
  promoted, sigma_fold 1,985), chi2 mean 9.22 (TNG 4.15, SIMBA 1.93, Astrid 26.58, Swift 4.24) vs T29 13.18.
- **Hypothesis:** C9 tuned the core-scale D-exponent (0.75) on T23's single sqrt(D) knee; on T29's two-scale knee the
  large-scale k^2 is carried by the knee, so a milder deficit scaling of the f_b/D core reach (k1 ~ D^-0.6) keeps T29's
  evidence while halving per-box chi2 (Astrid 38.9 -> 26.6, TNG 6.9 -> 4.2). One number changed, p = 5. Different from
  A11 (k^3 free knee) and C13 (core low-k shape): this changes the core's D-scaling, not any shape.
- **Expression:** `-A*D*(w*W2(k sqrt(D)/0.68) + (1-w)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.6/k1)`,
  w = Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 127,985 (the dry run; a tie with T29); marginalized [k1, k0]; worst suite Astrid (chi2 ~ 26.6);
  chi2 mean ~ 9.2.
- **Dry runs (41 nodes, not counted):** k^3 core 128,396 [k1, k0] (+ A 128,559 not promoted), chi2 mean 7.98 (TNG 5.10,
  SIMBA 2.05, Astrid 20.88, Swift 3.91); k^4 core 127,226 [k0] (+ A 128,846 not promoted, sf 1,806), chi2 8.82 (TNG 9.97).
  Submitting the k^3 core.
- **Expression:** `-A*D*(w*W2(k sqrt(D)/0.68) + (1-w)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*y^3/(1+y^2)^1.5`, y = k (D/0.05)^0.75/k1,
  w = Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 128,400 (the dry run); marginalized [k1, k0]; worst suite Astrid (chi2 ~ 21); chi2 mean ~ 8.0.
- **Outcome (T34, SCORED):** log Z_cross **127,985** (predicted 127,985 = the dry run; -8 vs best T29 127,993, a tie inside
  sigma_fold 1,985); marginalized **[k1, k0]** (as predicted; + A gave +474, not promoted). Per fold: TNG 34,054, SIMBA 29,930,
  Astrid 30,712, Swift 33,289. chi2_nu mean/std **9.22 / 10.06** (predicted 9.2; T29 13.18 / 14.99): TNG 4.15, SIMBA 1.93,
  Astrid 26.58 (worst, as predicted; T29 38.9), Swift 4.24. L_zero 79.2 (T29 73.2), L_map 15.48 (SIMBA 48.4; T29 8.57).
  G0_U pass. p = 5. Fold C drops to 0.26-0.33 (Swift 0.89), E -0.42..-1.05: the core becomes mostly retained-baryon f_b.
- **Lesson:** the core D-exponent trades chi2 vs L_map at fixed evidence: e = 0.6 halves k = 0.36 over-suppression (TNG
  +4.5 -> +2.4, Astrid +4.1 -> +2.2, Swift +1.7 -> +0.2) but SIMBA turns under-suppressed at k <= 1.1 (-1.1..-1.5) and its
  L_map blows up. T34 = T29's chi2 compromise; SIMBA's low-k needs a suite-robust fix (fixed knee 0.45/1.0 both lose).
- **Outcome (T33, SCORED):** log Z_cross **127,613** (predicted 128,300; -380 vs best T29 127,993, +226 vs C12's T32 127,387,
  ties inside sigma_fold 2,130); marginalized **[k0] only** (predicted [k1, k0]; + A gave +170, not promoted). Per fold vs
  T29: TNG +168, SIMBA -423, Astrid -136, Swift +191. chi2_nu mean/std **13.09 / 11.42** (predicted 10.5): TNG 9.34, SIMBA
  3.04, Astrid 32.46 (worst, as predicted; T29 38.9), Swift 7.51. L_zero 79.7, L_map 39.2 (SIMBA 133, poor). G0_U pass. p = 5.
- **Lesson:** the k^3 free knee halves Astrid k = 0.36 (+4.1 -> +2.1) and Swift (+1.7 -> +0.9) but leaves TNG at +4.6 and
  turns SIMBA under-suppressed at all k <= 3 (-2.3..-0.6): k0 shrinks (log -0.7) and the 0.68 W2 knee carries TNG's low k.
  Sharpening either knee component (T32, T33) only trades suites; T29's W2/W2 stays best. TNG k = 0.36 needs another lever.
- **Outcome (T35, SCORED):** log Z_cross **128,396** (predicted 128,400 = the dry run; **new best, +403 over T29's 127,993**, a
  tie inside sigma_fold 1,941); marginalized **[k1, k0]** (as predicted; + A gave +164, not promoted). Per fold vs T29: TNG
  +206, SIMBA +377, Astrid +273, Swift +238 (all up). chi2_nu mean/std **7.98 / 7.52** (predicted 8.0; T29 13.18 / 14.99):
  TNG 5.10, SIMBA 2.05, Astrid 20.88 (worst, as predicted; T29 38.9), Swift 3.91. **L_zero 67.8 (best so far)**, L_map 17.7
  (T29 8.57). G0_U pass. p = 5.
- **Lesson:** the core's k^2 tail was the k = 0.36 leak: a k^3 core onset cuts Astrid k = 0.36 from +4.1 to +1.2 and
  halves Astrid chi2 while raising evidence in every fold (k^4 over-corrects: TNG chi2 10, ladder [k0] only). Fold C now
  flips sign (TNG/SIMBA +0.6, Astrid/Swift -0.6), so C*D may be droppable (p = 4); TNG k = 0.36 (+3.8) and SIMBA k <= 1.1
  under-suppression (-1.0..-1.7) remain.

## C 2026-11-11 11:15 — next trial (lane C, agent 14)
- **Plan:** T35's lesson: fold C flips sign (TNG/SIMBA +0.6, Astrid/Swift -0.6), so the core's -C*D is not a shared feature.
  Dry-running on T35 (k^3 D^0.75 f_b core): (a) drop C*D (p = 4, core amplitude E*f_b only); (b) core D-exponent 0.6 on the
  k^3 core (B11's T34 move, now on T35). Different from A12 (knee weight D or f_b over Omega_b/Omega_m) and B12 (cosmology /
  D scaling of the knee scales). Dirs <scratch>/ml4ps_loop/c14_ag14; expression and prediction to follow.

## B 2026-11-11 17:00 — next trial (lane B, agent 12)
- **Dry runs (<scratch>/ml4ps_loop/b12_ag12, 41 nodes, not counted), all on T29:** fixed 0.68 knee scaled by
  (Omega_m/0.3)^(1/3): 127,092 [k1, k0]; free knee D-exponent 0.35 (vs 0.5): 124,384 [E]; **sigma_8 scaling of the knee
  argument k sqrt(D) (sigma_8/0.8)^a:** free knee only a = 1: [C] 124,761 ([C, k1] 128,006 not promoted, gain 3,245 vs sf
  3,260); a = 0.5: [C] 125,102 ([C, k1] 128,042); a = -1: 126,308; **both knees a = 1: [k0] 127,125 ([k0, k1] 128,269 not
  promoted)**; both knees a = 2: 127,648-ish ([k0] 126,818); both knees + core a = 1: [C] 123,079. sigma_8 on the knees
  lifts the 2-DOF evidence (Astrid fold 31.1k, best ever) but makes the greedy ladder stall one step early.
- **Plan:** T35 (k^3 core) landed meanwhile; dry-running sigma_8 on both knees of T35 before submitting. Different from
  C14 (core amplitude / core D-exponent) and A12 (knee weight).

## A 2026-11-11 11:30 — next trial (lane A, agent 12)
- **Dry run (<scratch>/ml4ps_loop/a12_ag12, 41 nodes, not counted):** weight on the 0.68 knee = expelled fraction D/w:
  **120,984 [C]** (+k0 123,611 not promoted, sf 4,035), chi2 10.43 (TNG 5.27, SIMBA 3.65, Astrid 27.8, Swift 4.95); folds
  A 2.1-4.9, C flips sign (Astrid +0.12, others -0.9..-2.6). Bad: it inflates sigma_fold. The retained-fraction dry run was
  killed unfinished (machine load ~47).
- **Hypothesis:** T29's two-scale knee weights the fixed 0.68 baryon knee by the cosmic fraction w = Omega_b/Omega_m (really
  an Omega_m dependence). Weight it instead by the retained fraction q = f_b/w (the expelled part 1-q = D/w goes on the free
  k0 knee): baryons that halos kept redistribute at the fixed inner scale, expelled ones at the per-box outer scale. Different
  from C13 (k^3 core on T29): this changes the knee weights, not any shape or the core.
- **Expression:** `-A*D*(q*W2(k sqrt(D)/0.68) + (1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`, q = f_b/w;
  lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 5. Not dry-run.
- **Prediction:** log Z_cross ~ 127,000 (likely below T29; the D/w variant lost 7k); marginalized [k1, k0]; worst suite
  Astrid (chi2 ~ 30); chi2 mean ~ 12.

## C 2026-11-11 11:35 — next trial (lane C, agent 14)
- **Dry runs (<scratch>/ml4ps_loop/c14_ag14, 41 nodes, not counted), on T35:** (a) **drop -C*D (p = 4): 127,952 [k0]**
  (+ k1 128,321 not promoted, sf 1,961), chi2 12.47 (TNG 6.20, SIMBA 1.40, Astrid 37.45, Swift 4.83); fold A ~2.03-2.15,
  E -0.74..-1.25. (b) core D-exponent 0.6 on the k^3 core: 128,342 [k0, k1] (+ A 128,623 not promoted), chi2 7.93 (TNG
  3.47, SIMBA 2.51, Astrid 21.78, Swift 3.96): a T35 clone (fold C flips sign again), not worth a trial.
- **Hypothesis:** fold C flips sign on T35, so -C*D is not shared; dropping it gives the simplest form of the T29/T35 line
  (p = 4: A, E, k0, k1) and the ladder stops at a single marginalized coefficient k0 (reviewer question: does one per-box
  DOF suffice?). Evidence should tie T35 inside sigma_fold; per-box chi2 rises (Astrid k = 0.36-0.86 loses its D lever).
- **Expression:** `-A*D*(w*W2(k sqrt(D)/0.68) + (1-w)*W2(k sqrt(D)/k0)) + E*f_b*y^3/(1+y^2)^1.5`, y = k (D/0.05)^0.75/k1,
  w = Omega_b_over_Omega_m; lin A, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 4.
- **Prediction:** log Z_cross ~ 127,950 (the dry run; -444 vs T35, a tie); marginalized [k0]; worst suite Astrid (chi2 ~ 37);
  chi2 mean ~ 12.5.
- **Dry runs on T35 (k^3 core), sigma_8 on both knees:** a = 1: [E] 125,730 ([E, k1] 128,332 not promoted), chi2 15.78
  (TNG 13.4, Astrid 41.2); a = 0.5: [C] 125,865 ([C, k0] 128,574 not promoted, gain 2,709 < sf 3,061), chi2 14.85. On T35
  sigma_8 hurts the ladder (first DOF changes to an amplitude and SIMBA's 1-DOF fold drops ~2k) and worsens chi2.
- **Hypothesis (submitted, on T29):** the halo-scale argument of the sqrt(D) knee carries sigma_8: at fixed k, higher
  sigma_8 means more massive halos, so the deficit-driven redistribution reaches larger scales, x = k sqrt(D) (sigma_8/0.8)/k_i
  on both knee components (fixed 0.68 baryon knee and free (1-w) k0 knee); T23's D^0.75 W2 f_b/D core unchanged. p = 5.
- **Expression:** `-A*D*(w*W2(k sqrt(D) s/0.68) + (1-w)*W2(k sqrt(D) s/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  s = sigma_8/0.8, w = Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30).
- **Prediction:** log Z_cross ~ 127,125 (the dry run; below T35's 128,396, ladder stalls at [k0], [k0, k1] 128,269 not
  promoted); marginalized [k0]; worst suite Astrid (chi2 ~ 30); chi2 mean ~ 13. Submitted as the clean test of sigma_8.
- **Outcome (T36, SCORED):** log Z_cross **131,103** (predicted 127,000; **new best, +2,707 over C13's T35 at 128,396 and +3,110
  over T29**, more than sigma_fold 1,753); marginalized **[k1, C, k0]** (three promoted; + A gave +48, not promoted). Per fold
  vs T29: TNG +593, SIMBA +528, Astrid +1,803, Swift +586 (all up). chi2_nu mean/std **8.85 / 8.12** (predicted 12; T29
  13.18 / 14.99): TNG 5.50, SIMBA 1.09, Astrid 22.48 (worst, as predicted; T29 38.9), Swift 6.31. **L_map 6.70 (best so
  far)**, L_zero 95.2 (worse than T29's 73.2). G0_U pass. p = 5.
- **Lesson:** weighting the fixed 0.68 knee by the retained fraction q = f_b/w (not the cosmic w, not the expelled D/w, which
  lost 7k) is the first change since T23 to beat sigma_fold, and k = 0.36 improves sharply (TNG +4.5 -> +1.05, Astrid +4.1 ->
  +2.1). Swift flips to -2.2 at 0.36, and TNG/Swift now go negative at k >= 6 (-0.5..-1.3). Next: put C13's k^3 core on T36,
  or work on the high-k tail; the population-mean transfer (L_zero) is the weak metric.

## A 2026-11-11 — next trial (lane A, agent 13)
- **Hypothesis:** T36 (best, 131,103 [k1, C, k0]) fixed the knee weight (retained fraction q = f_b/w on the 0.68 knee) and
  C13's T35 fixed the core's k^2 leak (k^3 core onset, +403 over T29 in all folds, Astrid chi2 38.9 -> 20.9). The two touch
  different terms (knee weights vs core shape), so they should stack: put T35's k^3 core y^3/(1+y^2)^1.5 on T36. Targets
  T36's residual k = 0.36 excess (TNG +1.05, Astrid +2.14) and Astrid's k 0.6-1.4 dip. Swift's -2.2 at 0.36 may worsen.
  No new coefficient (p = 5). Different from C14 (drop C*D on T35) and B12 (sigma_8 on the knees).
- **Expression:** `-A*D*(q*W2(k sqrt(D)/0.68) + (1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*y^3/(1+y^2)^1.5`, q = f_b/w,
  y = k (D/0.05)^0.75/k1; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). Not dry-run.
- **Prediction:** log Z_cross ~ 131,500; marginalized [k1, C, k0]; worst suite Astrid (chi2 ~ 16); chi2 mean ~ 7.
- **Outcome (T37, SCORED):** log Z_cross **127,952** (predicted 127,950 = the dry run; -444 vs T35 128,396, a tie inside
  sigma_fold 1,961; T36 at 131,103 is now best); marginalized **[k0] only** (as predicted; + k1 gave +369, not promoted). Per
  fold: TNG 33,901, SIMBA 29,971, Astrid 30,660, Swift 33,420. chi2_nu mean/std **12.47 / 14.53** (predicted 12.5; T35 7.98):
  TNG 6.20, SIMBA 1.40, Astrid 37.46 (worst, as predicted), Swift 4.83. L_zero 75.05, **L_map 13.50** (SIMBA 27.7; T35 17.7).
  G0_U pass. p = 4.
- **Lesson:** p = 4 with one marginalized coefficient (k0) keeps T35's evidence within sigma_fold and improves L_map, so one
  per-box DOF suffices for log Z_cross; but the -C*D lever is what fixes the k = 0.36 leak per box (Astrid +1.2 -> +3.1,
  mean sq 55 -> 145; TNG +3.8 -> +4.2), so chi2 doubles. Keep C for chi2; T37 is the reviewer-facing minimal variant.

## C 2026-11-11 — next trial (lane C, agent 15)
- **Plan:** on T36 (best, 131,103 [k1, C, k0]; q = f_b/w on the 0.68 knee), tune one number rather than a shape: dry-running
  (a) the core D-exponent 0.6 (B11's chi2 lever, untested on T36) and (b) the fixed baryon knee scale 1.0 instead of 0.68
  (its optimum was set under the old w weighting; q now gives that knee most of the weight). Different from A13 (k^3 core on
  T36) and B12 (sigma_8 on the knees). Dirs <scratch>/ml4ps_loop/c15_ag15; expression and prediction to follow.
- **Outcome (T38, SCORED):** log Z_cross **127,125** (predicted 127,125 = the dry run; -1,271 vs best T35 128,396, -868 vs
  T29, ties inside sigma_fold 2,414); marginalized **[k0] only** (as predicted; + k1 gave 128,269, +1,143, not promoted).
  Per fold: TNG 34,069, SIMBA 28,864, Astrid 30,765, Swift 33,428. chi2_nu mean/std **15.77 / 14.44** (predicted 13; T29
  13.18): TNG 12.35, SIMBA 1.89, Astrid 39.92 (worst, as predicted), Swift 8.93. **L_zero 65.2 (best so far; T35 67.8)**,
  L_map 17.83 (SIMBA 43.5). G0_U pass. p = 5.
- **Lesson:** sigma_8 in the knee scale improves population-mean transfer (best L_zero; TNG/Astrid 1-DOF folds up) but
  raises sigma_fold and per-box chi2 (TNG 6.9 -> 12.3), so the greedy ladder stops at one DOF; on T35 it costs ~2.5k. The
  knee's sigma_8/Omega_m dependence is not a route to evidence; leave cosmology out of the knee scales.

## B 2026-11-11 18:00 — next trial (lane B, agent 13)
- **Plan:** on T36 (best, 131,103 [k1, C, k0]), tune the knee-weight law rather than a shape or a scale: the retained
  fraction q = f_b/w was the first change to beat sigma_fold, so test how sharply the split follows it, weight q^p on the
  fixed 0.68 knee and 1 - q^p on the free k0 knee, p = 2 and p = 0.5 (one number, p = 5 coefficients). Different from A13
  (k^3 core on T36) and C15 (core D-exponent / fixed knee scale on T36). Dry runs in <scratch>/ml4ps_loop/b13_ag13;
  expression and prediction to follow.

## C 2026-11-11 12:00 — next trial (lane C, agent 15, continued)
- **Dry runs (<scratch>/ml4ps_loop/c15_ag15, 41 nodes, not counted), on T36:** (a) core D-exponent 0.6: 130,204 [k0, k1]
  (+ A 130,687 not promoted), chi2 8.80 (TNG 3.46, SIMBA 1.35, Astrid 24.47, Swift 5.94): -900 vs T36, fold C again flips
  (Astrid +0.09). (b) **fixed baryon knee scale 1.0: 130,777 [k0, E]** (+ k1 131,985 not promoted, gain 1,208 < sf 1,480),
  chi2 8.66 (TNG 3.96, SIMBA 1.07, Astrid 22.76, Swift 6.87); 0-DOF 110,519 (T36 92,989), 1-DOF 128,308 (T36 124,719).
- **Hypothesis:** under q = f_b/w weighting the fixed baryon knee carries most of the knee, so its scale optimum moves from
  0.68 (set under w weighting) to ~1.0 h/Mpc: retained gas redistributes on smaller scales. Population-mean transfer improves
  strongly (0- and 1-DOF evidence up 17k / 3.6k) and TNG chi2 drops 5.5 -> 4.0, but the ladder promotes E before k1 and stops
  at 2 DOF. One number changed vs T36, p = 5. Submitted as the clean test.
- **Expression:** T36 with 0.68 -> 1.0: `-A*D*(q*W2(k sqrt(D)/1.0) + (1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  q = f_b/Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30).
- **Prediction:** log Z_cross ~ 130,780 (the dry run; a tie with T36, -330); marginalized [k0, E]; worst suite Astrid (chi2 ~ 23);
  chi2 mean ~ 8.7.
- **Outcome (T39, SCORED):** log Z_cross **129,485** (predicted 131,500; -1,617 vs best T36 131,103, inside sigma_fold 1,772;
  +1,090 vs T35); marginalized **[k0, E]** (predicted [k1, C, k0]; + k1 gave 130,464, +978, not promoted). Per fold vs T36:
  TNG -606, SIMBA -152, Astrid -841, Swift -18. chi2_nu mean/std **7.25 / 6.04** (predicted 7.0; T36 8.85 / 8.12): TNG 5.01,
  SIMBA 0.93, Astrid 17.21 (worst, as predicted), Swift 5.84. **L_map 4.86 (best so far)**, L_zero 92.6. G0_U pass. p = 5.
- **Lesson:** the k^3 core and the q-weighted knee do not stack in evidence: chi2/L_map improve (Swift k = 0.36 -2.2 -> +0.2,
  Astrid +2.1 -> +1.4) but TNG k = 0.36 regresses (+1.05 -> +2.97) and the ladder swaps C for E and stalls at 2 DOF. On T36
  the core's k^2 tail is what fits TNG's large scales; keep T36's W2 core and look at the high-k tail (TNG/Swift -1.1..-1.3).

## A 2026-11-11 12:10 — next trial (lane A, agent 14)
- **Hypothesis:** T36 (best, 131,103 [k1, C, k0]) leaves a suite-shaped high-k tail: TNG and Swift are under-suppressed at
  k 1-3 (+0.5..+0.6 sigma) and over-predicted at k >= 6 (-0.5..-1.3, data more suppressed), Astrid the reverse. The W2 core
  (y^2/(1+y^2)) saturates too fast after its knee, so the core reach cannot keep rising into k ~ 10. Give the core a softer
  saturation y^2/(1+y)^2 (still k^2 at small k, plateau 1, approach 1 - 2/y), leaving T36's q-weighted two-scale knee
  untouched. One shape change, p = 5. Different from B13 (q^p knee weights) and C15 (fixed knee scale 1.0 / core D-exponent),
  and from A13/T39 (k^3 core onset: that sharpened the low-k end, this softens the high-k end). Not dry-run.
- **Expression:** `-A*D*(q*W2(k sqrt(D)/0.68) + (1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*y^2/(1+y)^2`, q = f_b/w,
  y = k (D/0.05)^0.75/k1; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30).
- **Prediction:** log Z_cross ~ 131,000 (a tie with T36); marginalized [k1, C, k0]; worst suite Astrid (chi2 ~ 22); chi2 mean ~ 8.5.
- **Outcome (T40, SCORED):** log Z_cross **130,777** (predicted 130,780 = the dry run; -326 vs best T36 131,103, a tie inside
  sigma_fold); marginalized **[k0, E]** (as predicted; + k1 gave 131,985, +1,208 < sf 1,480, not promoted). chi2_nu mean/std
  **8.66 / 8.39** (T36 8.85 / 8.12): TNG 3.96, SIMBA 1.07, Astrid 22.76 (worst, as predicted), Swift 6.87. **L_zero 71.2**
  (T36 95.2), **L_map 6.42 (best so far)**. G0_U pass. p = 5.
- **Lesson:** with q weighting the fixed knee at 1.0 ties T36's evidence but improves both losses (L_zero 95 -> 71, L_map 6.7 ->
  6.4) and TNG chi2; the 3-DOF set [k0, E, k1] reaches 131,985 but the greedy ladder stops. A scale between 0.68 and 1.0
  (~0.85) may keep T36's ladder path and T40's transfer; core D-exponent 0.6 on T36 loses ~900.

## C 2026-11-11 12:20 — next trial (lane C, agent 16)
- **Plan:** follow T40's lesson: on T36 (best, 131,103 [k1, C, k0]) the fixed q-weighted baryon knee scale 0.68 keeps the
  3-DOF ladder, 1.0 (T40) improves transfer (L_zero 95 -> 71) and TNG chi2 but stalls at [k0, E]. Dry-running the
  intermediate scales 0.8 and 0.9 (one number, p = 5) to find one that keeps both. Different from B13 (q^p knee weights /
  soft knee) and A14 (soft core saturation). Dirs <scratch>/ml4ps_loop/c16_ag16; expression and prediction to follow.

## B 2026-11-11 18:40 — next trial (lane B, agent 13, continued)
- **Dry runs (<scratch>/ml4ps_loop/b13_ag13, 41 nodes, not counted), on T36:** weight q^2: 128,659 [k0] (+ k1 129,010 not
  promoted), chi2 14.25 (TNG 10.6, Astrid 36.7); **weight sqrt(q): 129,956 [k0]** (+ E 131,046 not promoted, gain 1,090 < sf
  1,995), chi2 10.89 (TNG 6.53, SIMBA 0.91, Astrid 30.4, Swift 5.66); 0-DOF 112,260 / 1-DOF 129,956 (T36 92,989 / 124,719).
  High-k tail tries: soft-saturating free knee x^2/(1+x)^2: 128,251 [k0, C] (chi2 14.85, Astrid 43.9); soft core
  y^2/(1+y)^2: 128,304 [k0] (chi2 10.73). All below T36.
- **Hypothesis:** the knee-weight law is near-optimal at linear q; submit sqrt(q) as the clean bracket (the better of the two
  exponents): stronger 1-DOF evidence (+5.2k vs T36) but the ladder stalls one step after k0. One number changed, p = 5.
- **Expression:** `-A*D*(sqrt(q)*W2(k sqrt(D)/0.68) + (1-sqrt(q))*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  q = f_b/Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30).
- **Prediction:** log Z_cross ~ 129,950 (the dry run; -1,150 vs T36, a tie); marginalized [k0]; worst suite Astrid (chi2 ~ 30);
  chi2 mean ~ 10.9.
- **Outcome (T41, SCORED):** log Z_cross **128,304** (predicted 131,000; -2,799 vs best T36 131,103, beyond sigma_fold 1,575);
  marginalized **[k0] only** (predicted [k1, C, k0]; + E gave +201, not promoted; 0-DOF 118,434 vs T36 92,989). Per fold:
  TNG 33,620, SIMBA 30,144, Astrid 31,479, Swift 33,060. chi2_nu mean/std **10.73 / 10.33** (predicted 8.5): TNG 8.62,
  SIMBA 1.56, Astrid 28.09 (worst, as predicted), Swift 4.66. **L_zero 63.3 (best so far)**, L_map 10.98. G0_U pass. p = 5.
- **Lesson:** the soft y^2/(1+y)^2 core does fix the high-k tail (TNG/Swift k >= 6 go from -0.5..-1.3 to ~0..+0.2) and gives
  the best population-mean transfer, but its slow approach leaks k^2-like amplitude to large scales: TNG k = 0.36 back to +4.3
  (T36 +1.05) and the ladder stops at [k0]. High-k tail and k = 0.36 are coupled through the core shape; a fix needs a core
  that is sharp at onset but soft at saturation (e.g. y^2/(1+y^2) times a slow high-k factor), not a single softer Lorentzian.

## A 2026-11-11 12:15 — next trial (lane A, agent 15)
- **Plan:** follow T41's lesson (core must be sharp at onset but soft at saturation): on T36 (best, 131,103 [k1, C, k0]) keep
  the W2 core's k^2 onset and multiply it by a slow, bounded high-k rise so the core keeps deepening into k ~ 10 (targets
  TNG/Swift k >= 6 at -0.5..-1.3) without the large-scale leak of y^2/(1+y)^2. Dry-running (a) W2(y)(1+2y)/(1+y) and
  (b) W2(y)(1+W2(y/3)), no new coefficient (p = 5). Different from C16 (fixed knee scale 0.8/0.9) and B13 (q^p weights).
  Dirs <scratch>/ml4ps_loop/a15_ag15; expression and prediction to follow.
- **Dry runs (<scratch>/ml4ps_loop/c16_ag16, 41 nodes, not counted), on T36:** fixed knee scale **0.9: 129,092 [k0]**
  (+ k1 130,739 not promoted, gain 1,647 < sf 1,982); **0.8: 131,032 [k0, E]** (+ k1 131,693 not promoted, sf 1,554);
  0-DOF 100,583 / 103,703. Knee-scale curve under q weighting is flat-to-noisy: 0.68 131,103, 0.8 131,032, 0.9 129,092,
  1.0 130,777, with the ladder path flipping between [k1, C, k0], [k0, E] and [k0].
- **Hypothesis:** 0.8 sits between T36 and T40, keeping T36's evidence while taking part of T40's transfer and chi2 gain.
  One number changed vs T36, p = 5. Submitted as the bracket to close the knee-scale line.
- **Expression:** T36 with 0.68 -> 0.8: `-A*D*(q*W2(k sqrt(D)/0.8) + (1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  q = f_b/Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30).
- **Prediction:** log Z_cross ~ 131,030 (the dry run; a tie with T36); marginalized [k0, E]; worst suite Astrid (chi2 ~ 22);
  chi2 mean ~ 8.7.
- **Outcome (T42, lane B agent 13, SCORED):** log Z_cross **129,956** (predicted 129,950 = the dry run; -1,147 vs best T36
  131,103, a tie inside sigma_fold ~2,000); marginalized **[k0] only** (as predicted; + E gave 131,046, not promoted).
  chi2_nu mean/std **10.89 / 11.49** (predicted 10.9): TNG 6.53, SIMBA 0.91, Astrid 30.44 (worst, as predicted), Swift 5.66.
  L_zero 69.4, **L_map 6.12** (T36 6.70). G0_U pass. p = 5.
- **Lesson:** the knee-weight law is bracketed: q^2 and sqrt(q) both lose to linear q (T36); sqrt(q) lifts 0/1-DOF evidence
  (+19k / +5k) but the greedy ladder stalls after k0. Soft-saturating knee or core shapes (x^2/(1+x)^2) do not fix the
  TNG/Swift high-k tail (both ~128.3k). Leave the T36 weight at linear q; the high-k tail needs a different ingredient.
- **Outcome (T43, SCORED):** log Z_cross **131,032** (predicted 131,030 = the dry run; -70 vs best T36 131,103, a tie inside
  sigma_fold 1,680); marginalized **[k0, E]** (as predicted; + k1 gave 131,693, +660, not promoted). Per fold: TNG 34,450,
  SIMBA 31,128, Astrid 31,515, Swift 33,939. chi2_nu mean/std **8.42 / 8.36** (predicted 8.7; T36 8.85 / 8.12): TNG 3.43,
  SIMBA 1.22, Astrid 22.54 (worst, as predicted), Swift 6.49. L_zero 86.9 (T36 95.2, T40 71.2), L_map 7.79 (T36 6.70). G0_U pass.
- **Lesson:** the fixed knee scale is exhausted: 0.68-1.0 all tie inside sigma_fold (0.9 loses 2k by stalling at [k0]), and only
  the ladder path changes. Scale tuning moves TNG k 0.6-1.1 (+0.6 -> +0.2) but not the k = 0.36 split (Astrid +2.1, Swift -2.1)
  or the TNG/Swift k >= 6 tail (-0.7..-1.0); those need a new term, not a knee-scale number. Note: dry runs took ~40 min at load 60.
- **Dry runs (<scratch>/ml4ps_loop/a15_ag15, 41 nodes, not counted), on T36:** (a) core W2(y)(1+2y)/(1+y): **130,713
  [k0, k1]** (+ E 131,010 not promoted, sf 1,594), chi2 8.44 (TNG 3.61, SIMBA 1.30, Astrid 23.83, Swift 5.02); 0-DOF 92,953,
  1-DOF 122,631. (b) W2(y)(1+W2(y/3)) not finished (machine load ~42; stopped).
- **Expression:** `-A*D*(q*W2(k sqrt(D)/0.68) + (1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(y)*(1+2y)/(1+y)`, q = f_b/w,
  y = k (D/0.05)^0.75/k1; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 5. Submission A15_T36_core_slowrise.json.
- **Prediction:** log Z_cross ~ 130,700 (the dry run; a tie with T36, -390); marginalized [k0, k1]; worst suite Astrid
  (chi2 ~ 24); chi2 mean ~ 8.4.

## B 2026-11-11 13:20 — next trial (lane B, agent 14)
- **Hypothesis:** T36 (best, 131,103 [k1, C, k0]) splits its knee by the retained fraction q = f_b/w (fixed 0.68 knee) and
  the expelled fraction 1 - q = D/w (free k0 knee), but the free knee's reach still scales with the absolute deficit sqrt(D),
  which carries a hidden Omega_b/Omega_m dependence. Scale the expelled-gas knee by the expelled *fraction*,
  x = k sqrt(D/w)/k0, so weight and reach use the same variable. Targets the k = 0.36 suite split (Astrid +2.1, Swift -2.2,
  TNG +1.05; mean sq 72/17/16). One argument changed, p = 5. Different from A15 (core high-k rise) and C16 (knee scale
  0.8/0.9). Dry runs started in <scratch>/ml4ps_loop/b14_ag14 but stopped (machine load ~42); not dry-run.
- **Expression:** `-A*D*(q*W2(k sqrt(D)/0.68) + (1-q)*W2(k sqrt(D/w)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  q = f_b/w, w = Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30).
- **Prediction:** log Z_cross ~ 130,500 (a tie with T36); marginalized [k1, C, k0]; worst suite Astrid (chi2 ~ 22); chi2 mean ~ 9.

## C 2026-11-11 13:40 — next trial (lane C, agent 17)
- **Hypothesis:** T36 (best, 131,103 [k1, C, k0]) leaves the k = 0.36 suite split as the chi2 sink (Astrid +2.1, Swift -2.2,
  TNG +1.05 sigma; mean sq 72 / 17 / 16, i.e. large per-box scatter, not only a mean offset). Every low-k term of T36 scales as
  D^2 or steeper (sqrt(D)-scaled knees), but C2's probe found -S(0.36) ~ D^1 in Astrid/Swift (D^2 in TNG/SIMBA). Add a
  localized large-scale D^1 bump, -G*D*u^2/(1+u^2)^2 with u = k/0.4 (fixed scale, peak at k = 0.4, k^2 onset, decays as k^-2
  so it leaves k >= 3 alone), as a per-box large-scale amplitude the ladder can marginalize. One linear coefficient, p = 6.
  Different from A15 (core high-k rise) and B14 (sqrt(D/w) free-knee argument). Not dry-run (load ~57).
- **Expression:** T36 + `- G*D*(k/0.4)**2/(1 + (k/0.4)**2)**2`; lin A, C, E, G; nl k0 (log 0.01-10), k1 (log 0.05-30).
- **Prediction:** log Z_cross ~ 131,800; marginalized [k1, C, k0] or with G; worst suite Astrid (chi2 ~ 18); chi2 mean ~ 7.5.

## B 2026-11-11 16:00 — next trial (lane B, agent 14, resumed)
- The B14 section above (13:20) was written but never scored (no T-id; no evaluate process running at 16:00). Submitting
  Loop/submissions/B14_ag14_freeknee_Dw.json unchanged, with the same hypothesis, expression and prediction (~130,500;
  [k1, C, k0]; worst Astrid). Still distinct from A15 (core slow rise) and C17 (D^1 low-k bump).
- **Outcome (no trial id, HARNESS ERROR, not counted):** `Loop.evaluate` raised in `REG.freeze`: "T44 already frozen with a
  different expression; use a new form_id". The registry already holds a T44 (probably an earlier, interrupted evaluation by
  another agent) while trial_state.yaml ends at T43 (n_used 43), so the next id collides. Stopped per the brief; the B14 form
  (sqrt(D/w) free-knee argument) is still untested. The harness's id allocation needs a fix before the next submit.

## C 2026-11-11 16:00 — next trial (lane C, agent 18)
- **Plan:** on T36 (best, 131,103 [k1, C, k0]), carry T36's winning variable (retained fraction q = f_b/w) into the *core
  amplitude*, not the knee: dry-running (a) core (E*q - C*D) and (b) core (E*f_b - C*D/w) (expelled fraction), p = 5, one
  variable swap each. Different from A15 (core high-k shape), B14 (free-knee argument sqrt(D/w)) and C17 (low-k D^1 bump).
  Dirs <scratch>/ml4ps_loop/c18_ag18; expression and prediction to follow.

## A 2026-11-11 16:05 — next trial (lane A, agent 16)
- **Plan:** C2's probe: -S(0.36) ~ D^1 in Astrid/Swift but every T36 knee is sqrt(D)-scaled (low-k ~ D^2). Instead of adding a
  bump (C17) or changing the free knee's argument (B14), make the *retained-fraction* fixed knee D-free: q*W2(k/kq) with kq
  fixed (0.68/sqrt(0.05) ~ 3.0, and 1.5), so the retained gas redistributes on a fixed halo scale (low-k ~ q D k^2) while the
  expelled gas keeps the sqrt(D) reach. p = 5, no new coefficient. Dry-running both in <scratch>/ml4ps_loop/a16_ag16.

## B 2026-11-11 16:05 IST — next trial (lane B, agent 15)
- **Hypothesis:** T36 (best, 131,103 [k1, C, k0]) leaves a per-box TNG/Swift high-k tail (k >= 6 at -0.5..-1.3 sigma, mean sq
  1-4) after per-box MAP; shape changes to the core (T41 soft core, A15 slow rise) trade it against k = 0.36. Instead give the
  core (inner-halo) scale a cosmology dependence: halo radii at fixed mass scale as (M/rho_m)^(1/3), so the core reach in
  h/Mpc moves as (Omega_m/0.3)^(1/3). Omega_m has been tried only on knees (T01, T06), never on the T36 core. One factor
  added, p = 5. Different from A15 (core slow rise), B14 (free-knee sqrt(D/w)), C17 (D^1 low-k bump), C18 (core amplitude in q).
  Not dry-run. f_b_M_i probed again: 4-94 % NaN per bin, unusable.
- **Expression:** T36 with core y = k (D/0.05)^0.75 / (k1 (Omega_m/0.3)^(1/3)); lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30).
  Submission Loop/submissions/B15_T36_core_Om_scale.json.
- **Prediction:** log Z_cross ~ 130,500 (a tie with T36); marginalized [k1, C, k0]; worst suite Astrid (chi2 ~ 22); chi2 mean ~ 9.
- **Harness note:** Loop/candidates_frozen.jsonl holds an orphaned T44 (A15's core slow-rise form, frozen 07:54 UTC, never
  recorded; no evaluate process or lock alive at 10:30 UTC), so any other form will collide at T44 as B14 found.
- **Outcome (no trial id, HARNESS ERROR, not counted):** same collision as B14: `REG.freeze` raised "T44 already frozen with a
  different expression". n_used stays 43. The B15 form (Omega_m-scaled core reach) is untested; the submission file is ready.
  The loop is blocked until the owner clears the orphaned T44 freeze row or re-runs A15_T36_core_slowrise.json (the only form
  that matches it).
- **Dry runs (<scratch>/ml4ps_loop/a16_ag16, 41 nodes, not counted):** D-free retained knee q*W2(k/kq): kq = 1.5 125,348
  [k0]; **kq = 3.0 130,165 [k0, E]** (+ C 130,391 not promoted), chi2 11.94 (TNG 9.28, SIMBA 1.40, Astrid 27.0, Swift 10.09);
  kq = 4.5 127,145 [k0]; kq = 7.0 127,956 [k0]. Peaks at kq ~ 3 (= 0.68/sqrt(0.05)), i.e. no better than T36's sqrt(D) knee.
- **Expression:** `-A*D*(q*W2(k/3.0) + (1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`, q = f_b/w; lin A, C, E;
  nl k0 (log 0.01-10), k1 (log 0.05-30). p = 5. Submission A16_T36_Dfree_retained_knee.json (the bracket).
- **Prediction:** log Z_cross ~ 130,170 (the dry run; a tie with T36, -940); marginalized [k0, E]; worst suite Astrid
  (chi2 ~ 27); chi2 mean ~ 11.9.
- **Outcome: HARNESS ERROR (not counted).** Loop.evaluate failed at REG.freeze: "T44 already frozen with a different
  expression; use a new form_id" (trial-id race with a parallel agent). Per the brief, reported and stopped; the submission
  A16_T36_Dfree_retained_knee.json is unscored. Lesson from the dry runs: the D-free retained knee peaks at kq ~ 3 and ties
  T36 (-940) at best; low-k D^1 via the retained knee does not fix the k = 0.36 split. (Lane A agent #15 of this run; the
  heading's "agent 16" avoids clashing with the earlier A15 section.)

## A 2026-11-11 16:20 — next trial (lane A, agent 16 of this run; heading "A16b" to avoid the earlier "agent 16" section)
- **Situation:** the loop is blocked: candidates_frozen.jsonl holds an orphaned T44 (A15's core slow-rise form, frozen 13:24
  local, never recorded; no evaluate process or lock alive at 16:17), so every other form hits HARNESS ERROR at REG.freeze
  (B14, B15, the earlier "A agent 16"). Agents cannot edit the registry. The only submission that can score is the exact
  frozen T44 form, so I re-submit A15's file unchanged (copy: Loop/submissions/A16b_rerun_T44_core_slowrise.json). This
  scores A15's untested idea and unblocks T45+ for B14/B15/C17/C18. Not a copy of any in-progress idea (A15 is not running).
- **Hypothesis (A15's):** T36 with the core multiplied by a slow bounded high-k rise (1+2y)/(1+y) keeps the k^2 core onset
  but lets the core keep deepening to k ~ 10 (TNG/Swift k >= 6 residuals -0.5..-1.3).
- **Expression:** `-A*D*(q*W2(k sqrt(D)/0.68) + (1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(y)*(1+2y)/(1+y)`, q = f_b/w,
  y = k (D/0.05)^0.75/k1; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 5.
- **Prediction:** log Z_cross ~ 130,700 (A15's dry run; a tie with T36); marginalized [k0, k1]; worst suite Astrid
  (chi2 ~ 24); chi2 mean ~ 8.4.
- **Dry runs (<scratch>/ml4ps_loop/c18_ag18, 41 nodes, not counted), on T36:** (a) core (E*q - C*D), q = f_b/w: 127,390
  [k0, k1, C] (+ A 128,772 not promoted), chi2 7.63 (TNG 6.36, SIMBA 1.13, Astrid 18.59, Swift 4.46); 0-DOF 95,636; fold C
  still all negative but TNG/Swift ~-1 vs SIMBA/Astrid -0.14, E ~ -0.1 (core q-term almost unused). (b) **core
  (E*f_b - C*D/w): 130,381 [k1]** (+ A 130,891 not promoted, sf 1,739), chi2 14.39 (TNG 11.58, SIMBA 2.95, Astrid 34.84,
  Swift 8.19); 0-DOF 120,879 / 1-DOF 130,381 (T36 92,989 / 124,719); **fold C universal positive 0.11-0.16**, E -0.83..-1.10,
  A 1.58-1.82, k1 ~ 10^0.91-0.99 (T36 C flipped -1.2..+0.2).
- **Hypothesis:** the core's -C*D lever flips sign across folds on T36, so it is not a shared feature; put it on the expelled
  *fraction* D/w (same variable as T36's knee split 1 - q). The dry run makes C universal and lifts the 0/1-DOF evidence by
  28k / 5.7k, but the ladder stalls at [k1] and per-box chi2 rises (C no longer a per-box lever). One variable swap, p = 5.
- **Expression:** `-A*D*(q*W2(k sqrt(D)/0.68) + (1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D/w)*W2(k (D/0.05)^0.75/k1)`, q = f_b/w,
  w = Omega_b_over_Omega_m; lin A, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). Submission C18_T36_core_expelledfrac.json.
- **Prediction:** log Z_cross ~ 130,380 (the dry run; -720 vs T36, a tie); marginalized [k1]; worst suite Astrid (chi2 ~ 35);
  chi2 mean ~ 14.4.
- **Outcome (T44, SCORED):** log Z_cross **130,712** (predicted 130,700; -390 vs best T36 131,103, a tie inside sigma_fold
  1,680); marginalized **[k0, k1]** (as predicted; + E 131,010 not promoted). Per fold: TNG 34,427, SIMBA 31,038, Astrid
  31,460, Swift 33,788. chi2_nu mean/std **8.44 / 8.99** (predicted 8.4): TNG 3.61, SIMBA 1.30, Astrid 23.83 (worst, as
  predicted), Swift 5.02. L_zero 94.8, L_map 7.40. G0_U pass. The orphaned T44 freeze is now consumed: T45+ submits work again.
- **Lesson:** the slow core rise does not fix the TNG k >= 6 tail (still -0.41..-1.15) and trades Swift's chi2 (6.3 -> 5.0)
  against Astrid (22.5 -> 23.8); core high-k shape is exhausted. Pending untested forms (B14 sqrt(D/w) knee, B15 Omega_m core,
  C17 D^1 bump, A16 D-free knee ~130.2k dry) can now be submitted.

## A 2026-11-11 16:27 — next trial (lane A, agent 17)
- **Plan:** on T36 (best, 131,103 [k1, C, k0]) give the two knee pieces *separate* amplitudes: -D*(A*q*W2(u) + B*(1-q)*W2(x))
  (retained-gas knee vs expelled-gas knee), so the ladder can marginalize the expelled-gas amplitude per box (targets the
  k = 0.36 per-box scatter, Astrid/Swift mean sq 72/17, and the mid-k sign split). Dry-running (a) + core (E*f_b - C*D) p = 6,
  (b) core -C*D only p = 5. Different from B14 (knee argument), B15 (Omega_m core), C17 (D^1 bump), C18 (core amplitude D/w).
  Dir <scratch>/ml4ps_loop/a17_ag17; expression and prediction to follow.
- **Outcome (T45, SCORED):** log Z_cross **130,381** (predicted 130,380 = the dry run; -722 vs best T36 131,103, a tie inside
  sigma_fold 1,763); marginalized **[k1] only** (as predicted; + A gave 130,891, not promoted). chi2_nu mean/std **14.39 /
  12.20** (predicted 14.4; T36 8.85): TNG 11.58, SIMBA 2.95, Astrid 34.84 (worst, as predicted), Swift 8.19. L_zero 74.1
  (T36 95.2), L_map 35.4 (T36 6.7). G0_U pass. p = 5.
- **Lesson:** a D/w core lever makes every core coefficient a shared universal-sign feature (C 0.11-0.16, E ~ -0.9) and lifts
  0/1-DOF evidence (+28k / +5.7k vs T36), i.e. better population transfer, but removes the per-box lever, so the ladder stops at
  one DOF and per-box chi2 / L_map blow up. The core's -C*D is valuable *because* it is per-box; q in the core amplitude (E*q) loses 3.7k.

## C 2026-11-11 16:30 — next trial (lane C, agent 15 of this run; heading "C15r" to avoid the earlier "agent 15" section)
- **Situation:** C17's D^1 low-k bump (13:40) was never scored (blocked by the orphaned T44 freeze; no T-id, no evaluate
  process alive at 16:28). It is lane C's own pending idea and differs from the in-progress A17 (separate knee amplitudes),
  and from B14/B15 (free-knee sqrt(D/w), Omega_m core). Submitting it unchanged as Loop/submissions/C15r_T36_D1_lowk_bump.json.
- **Hypothesis (C17's):** T36 (best, 131,103) still has the k = 0.36 split (Astrid +2.14, Swift -2.24, TNG +1.05; mean sq
  72 / 17 / 16, per-box scatter). All T36 low-k terms scale as D^2 or steeper; a fixed-scale per-box D^1 bump
  -G*D*u^2/(1+u^2)^2 (u = k/0.4, k^2 onset, ~k^-2 decay, negligible at k >= 3 and for G0_U) gives the ladder a large-scale lever.
- **Expression:** T36 + `- G*D*(k/0.4)**2/(1 + (k/0.4)**2)**2`; lin A, C, E, G; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 6.
- **Prediction:** log Z_cross ~ 131,500; marginalized [k1, C, k0] (G possibly promoted); worst suite Astrid (chi2 ~ 19);
  chi2 mean ~ 8.
- **Outcome (T46, SCORED):** log Z_cross **131,012** (predicted 131,500; -91 vs best T36 131,103, a tie inside sigma_fold
  1,686); marginalized **[k0, k1, C]** (T36's set; + G gave 131,239, +228, not promoted). chi2_nu mean/std **9.14 / 8.62**
  (predicted 8; T36 8.85 / 8.12): TNG 3.34, SIMBA 0.99, Astrid 23.19 (worst, as predicted), Swift 9.05. L_zero 97.2,
  **L_map 5.17 (best so far; T36 6.70)**. G0_U pass. p = 6.
- **Lesson:** the fixed-scale D^1 bump is used with a universal sign (fold G -0.07..-0.22, i.e. a small large-scale *enhancement*
  term, not suppression) but adds no evidence and does not split Astrid vs Swift at k = 0.36; the k = 0.36 split is not a
  missing D^1 amplitude. Low-k D-scaling via extra terms (C17 bump, A16 D-free knee) is closed; spend p = 6 elsewhere.

## C 2026-11-11 16:45 — next trial (lane C, agent 16 of this run; heading "C19")
- **Plan:** T36 (best, 131,103 [k1, C, k0]) per-box residuals show a suite-dependent *onset steepness* of the free knee:
  Astrid +2.14 at k = 0.36 but -0.8..-1.3 at k 0.6-1.1 (data steeper than the W2 onset; k^3 core T39 helped Astrid), Swift
  the reverse (-2.24 at 0.36, +0.3..+0.5 above). Add a localized, linear knee-sharpness lever on the expelled-gas knee:
  -B*D*(1-q)*(W3(x) - W2(x)), W3 = x^3/(1+x^2)^1.5, x = k sqrt(D)/k0 (same k0). It vanishes at k -> 0 (leading -x^2 term,
  k^2 OK) and k -> inf, so it only reshapes the knee; B can be marginalized per box. p = 6. Different from A17 (separate knee
  amplitudes), B14 (sqrt(D/w) argument), B15 (Omega_m core). Dry-running in <scratch>/ml4ps_loop/c19_ag16.
- **Dry runs (<scratch>/ml4ps_loop/a17_ag17, 41 nodes, not counted):** (a) split knee amplitudes + core (E*f_b - C*D),
  p = 6: **131,696 [k0, k1, B]** (+ C 132,193 not promoted, gain 497 < sf 1,537); 0-DOF 91,947; fold B 2.4-3.8 vs A 1.6-3.6
  (Astrid A 1.64 lowest). (b) split amplitudes, core -C*D only (no E), p = 5: 127,717 [k0] (+ k1 128,893 not promoted):
  the f_b core term is needed.
- **Expression:** `-D*(A*q*W2(k sqrt(D)/0.68) + B*(1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  q = f_b/Omega_b_over_Omega_m; lin A, B, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 6.
  Submission Loop/submissions/A17_T36_split_knee_amps.json.
- **Prediction:** log Z_cross ~ 131,700 (the dry run; +590 vs T36, a tie inside sigma_fold); marginalized [k0, k1, B]; worst
  suite Astrid (chi2 ~ 20); chi2 mean ~ 8.
- **Outcome (T47, SCORED):** log Z_cross **131,696** (predicted 131,700 = the dry run; **+593 vs T36 131,103, new nominal best
  but a tie inside sigma_fold 1,392**); marginalized **[k0, k1, B]** (as predicted; + C 132,193 not promoted). Per fold: TNG
  34,270, SIMBA 31,281, Astrid 32,281 (T36-family high), Swift 33,865. chi2_nu mean/std **8.99 / 9.34** (predicted 8): TNG 5.15,
  SIMBA 1.26, Astrid 24.97 (worst, as predicted), Swift 4.59. L_zero 97.1, L_map 6.42 (T36 6.70). G0_U pass. p = 6 (cap).
- **Lesson:** a separate, per-box-marginalized expelled-gas knee amplitude B buys a small evidence gain (mainly Astrid +) and
  Swift chi2, but the k = 0.36 split (TNG +1.29, Astrid +1.90) and the TNG/Swift k >= 6 tail (-0.65..-1.05) are unchanged;
  p = 6 is now full, so further gains must come from swapping a term, not adding one.
- **Dry run (<scratch>/ml4ps_loop/c19_ag16, 41 nodes, not counted):** 0-DOF 93,478; **[k0] 127,465** (+ B 128,896, +1,431 <
  sf 2,084, not promoted); chi2 7.43 (TNG 5.23, SIMBA 1.14, **Astrid 17.63**, Swift 5.73), the best chi2 mean on the T36 family.
  Fold B not universal (TNG 0.28, SIMBA -0.79, Astrid 0.39, Swift 2.07). A core-sharpness variant (-B*D*(W3(y)-W2(y))) was
  started but stopped (exceeded the foreground time limit); untested.
- **Expression:** `-A*D*(q*W2(k sqrt(D)/0.68) + (1-q)*W2(x)) - B*D*(1-q)*(W3(x) - W2(x)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`,
  x = k sqrt(D)/k0, q = f_b/w; lin A, B, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). p = 6.
  Submission Loop/submissions/C19_T36_knee_sharpness_lever.json.
- **Prediction:** log Z_cross ~ 127,470 (the dry run; ~-3.6k vs T36, ladder stalls at [k0]); marginalized [k0]; worst suite
  Astrid (chi2 ~ 17.6); chi2 mean ~ 7.4. Submitted for the chi2 / per-suite information, not to beat T36.

## A 2026-11-11 17:00 — next trial (lane A, agent 18)
- **Hypothesis:** T47 (best, 131,696 [k0, k1, B]) still has the TNG/Swift k >= 6 tail (-0.65..-1.05) vs Astrid +0.6 and fold C
  tracks that split (TNG/Swift -1.2..-1.4, Astrid -0.07). The retained-gas knee sits on a *fixed* scale (0.68/sqrt(D) ~ k 3),
  close to the core's k1 (fold 10^0.26-0.6 at D = 0.05). Move the retained-gas knee onto the core scale y (same k1), so the
  per-box k1 lever moves the retained-gas redistribution and the core together. Swap, not add: p = 6. Different from C19
  (knee-sharpness lever), B14 (sqrt(D/w) knee), B15 (Omega_m core). No dry run (dry runs exceed the 10-min foreground limit).
- **Expression:** `-D*(A*q*W2(y) + B*(1-q)*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(y)`, y = k (D/0.05)^0.75/k1, q = f_b/w;
  lin A, B, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). Submission Loop/submissions/A18_T47_retained_on_core_scale.json.
- **Prediction:** log Z_cross ~ 130,500 (slightly below T47); marginalized [k0, k1, B]; worst suite Astrid (chi2 ~ 24);
  chi2 mean ~ 9.
- **Outcome (T48, SCORED):** log Z_cross **127,465** (predicted 127,470 = the dry run; -3,638 vs best T36 131,103, beyond
  sigma_fold ~2,100); marginalized **[k0] only** (as predicted; + B 128,896 not promoted). chi2_nu mean/std **7.43 / 6.15**
  (T36 8.85 / 8.12; best chi2 mean of the T36 family): TNG 5.23, SIMBA 1.14, Astrid 17.63 (worst, as predicted), Swift 5.73.
  L_zero 92.6, L_map 5.39. G0_U pass. p = 6.
- **Lesson:** a per-box knee-sharpness lever is real (Astrid chi2 22.5 -> 17.6, every suite's chi2 down vs T36) but its sign
  differs by suite, so it adds per-box fit and no cross-suite evidence; with B in the form the ladder takes k0 first and stops.
  The Astrid/Swift onset split is suite-specific sharpness, not a shared feature-driven term.

## C 2026-11-11 17:10 — next trial (lane C, agent 17 of this run; heading "C20")
- **Hypothesis:** T47 (best, 131,696 [k0, k1, B]) gives the expelled-gas knee its own amplitude B, which the ladder marginalizes
  per box. With B per-box, the (1-q) = D/w weight on that knee (amplitude ~ D^2/w) may be redundant; drop it (simplification,
  swap not add, p = 6). Tests whether the expelled-knee amplitude needs the D^2/w scaling or plain D. Different from A18
  (retained knee on the core scale), B14 (sqrt(D/w) knee argument), B15 (Omega_m core). No dry run (trials take 8-15 min).
- **Expression:** `-D*(A*q*W2(k sqrt(D)/0.68) + B*W2(k sqrt(D)/k0)) + (E*f_b - C*D)*W2(k (D/0.05)^0.75/k1)`, q = f_b/w;
  lin A, B, C, E; nl k0 (log 0.01-10), k1 (log 0.05-30). Submission Loop/submissions/C20_T47_expelled_knee_no_q.json.
- **Prediction:** log Z_cross ~ 130,500 (slightly below T47; 0/1-DOF evidence loses the D/w lever); marginalized [k0, k1, B];
  worst suite Astrid (chi2 ~ 25); chi2 mean ~ 9.
- **Outcome (T49, SCORED):** log Z_cross **129,978** (predicted 130,500; -1,718 vs best T47 131,696, about -1 sigma_fold 1,747);
  marginalized **[k0] only** (predicted [k0, k1, B]; + B 130,566 not promoted). Per fold: TNG 34,090, SIMBA 30,335, Astrid
  31,831, Swift 33,722. chi2_nu mean/std **12.99 / 12.19** (predicted 9): TNG 6.45, SIMBA 3.41, Astrid 33.90 (worst, as
  predicted), Swift 8.19. L_zero 74.1, L_map 38.7. G0_U pass. p = 6. Fold k1 shifts out to 10^0.78-0.88 and C shrinks to -0.1..-0.46.
- **Lesson:** putting the retained-gas knee on the core scale makes k1 a shared (not per-box) scale, so the ladder stalls at one
  DOF and per-box chi2 / L_map blow up (same failure as T45). The retained-gas knee must keep its own fixed sqrt(D) scale;
  the core scale k1 is valuable only as an independent per-box lever.

## A 2026-11-11 17:15 — next trial (lane A, agent 19)
- **Situation:** n_used = 49 and C20 (T50 candidate) holds the lock, so this submission will most likely hit "budget
  exhausted". Submitting anyway per the brief.
- **Hypothesis:** T47 (best, 131,696 [k0, k1, B]) keeps a fixed retained-gas knee 0.68/sqrt(D); scale it by
  (Omega_m/0.3)^(1/3) (halo-radius cosmology scaling, as in T01) so the knee moves with cosmology at no coefficient cost. Swap,
  not add: p = 6. Different from C20 (dropping the (1-q) weight on the expelled knee).
- **Expression:** T47 with `k*D**0.5/0.68` -> `k*D**0.5/(0.68*(Omega_m/0.3)**(1/3))`; lin A, B, C, E; nl k0, k1.
  Submission Loop/submissions/A19_T47_Om_retained_knee.json.
- **Prediction:** log Z_cross ~ 131,500 (a tie with T47); marginalized [k0, k1, B]; worst suite Astrid (chi2 ~ 25); chi2 mean ~ 9.
- **Outcome (no trial id, BUDGET EXHAUSTED, not scored):** Loop.evaluate waited for C20's lock, then printed "budget
  exhausted (50/50 trials)". The form A19_T47_Om_retained_knee.json is unscored. Lesson: the loop's 50-trial budget is spent;
  T47 (131,696 [k0, k1, B]) stands as the nominal best.
- **Outcome (T50, SCORED):** log Z_cross **124,756** (predicted 130,500; -6,940 vs best T47 131,696, beyond sigma_fold 2,445);
  marginalized **[k0] only** (predicted [k0, k1, B]; + k1 126,866, +2,110 < sf, not promoted). 0-DOF 77,142 (T47 91,947).
  chi2_nu mean/std **9.52 / 8.11** (predicted 9): TNG 6.05, SIMBA 3.20, Astrid 23.45 (worst, as predicted), Swift 5.40.
  L_zero 151.2 (T47 97.1), L_map 35.9 (T47 6.4). G0_U pass. p = 6. Budget now 50/50.
- **Lesson:** the (1-q) = D/w weight on the expelled-gas knee is load-bearing even with a separate amplitude B: without it, the
  retained-gas amplitude A collapses (fold A -2.2..+0.4, sign flips) and 0-DOF evidence drops ~15k. The expelled knee's
  amplitude must scale as D^2/w (expelled fraction times deficit); keep T47's q / (1-q) split as is.

# ML4PS loop agent brief (SANDBOX-ML4PS, z = 0 main loop at k ≤ 9.92, trial agent)

You are one trial agent in an AgenticTuner-inspired loop. It searches for a compact, interpretable functional form for the
baryonic matter-power suppression S(k) = P_hydro/P_dmo − 1 **at z = 0**, for the rewrite of a workshop paper. You are
stateless. Everything earlier agents learned is on disk. **You submit exactly one form, record what you learned, and stop.**

## Where you may read and write
- **Root:** `<repo>` (call it M).
  Work only inside M.
- **Read freely:**
  - `M/Paper/ml4ps.tex`: the paper being rewritten. Its Table 1 forms, the physical-interpretation appendix and the
    reviewer comments (`\RevA`, `\RevB`, `\RevC`) are all fair game. The paper's base forms are your starting point;
  - `M/Base/RESULT.md`, `M/Base/trial_state.yaml` and `M/Base/residuals/*.md`: the paper's forms scored with this harness;
  - `M/Loop/lab_notebook.md`, `M/Loop/trial_state.yaml` and `M/Loop/residuals/*.md` (this loop);
  - the data in `M/Artifacts/data/*.h5`, which you may probe with Python via `Scripts.data.suppression_io.load`;
  - the code in `M/Scripts/` and `M/Loop/evaluate.py`.
- **Write only:**
  - `M/Loop/submissions/<a-name-you-choose>.json`;
  - your entries appended to `M/Loop/lab_notebook.md`;
  - scratch files under `<scratch>/ml4ps_loop/`.
- **Never** read or list anything outside M, and never read or list `M/Archive_k20/` (an earlier run; off limits). That
  includes the parent folder, any other SANDBOX, `~/.claude`, the web and any paper repository other than `M/Paper/`. Every tool call is audited, and a trial whose agent breaks this is discarded.

## The data and the test
- **Suites:** 4 CAMELS suites (IllustrisTNG, SIMBA, Astrid, Swift-EAGLE), 918–953 LH boxes each, **z = 0 only**.
- **Grid:** 16 k-bins from 0.36 to **9.92** h/Mpc: the paper's 13 bins with k ≤ 10 plus 1.62, 3.39 and 8.67. The paper's
  bins above 10 (12.4, 15.7, 20.0) are **not** used, so the paper's scores are not directly comparable; `Base/RESULT.md` has the
  paper's forms on this grid.
- **Features per box:** `D` (= Ω_b/Ω_m − f_b, the halo baryon deficit), `f_b`, `f_b_M_0..5` (f_b in 6 halo-mass bins),
  `Omega_b_over_Omega_m`, `Omega_m`, `sigma_8`, and the symbol `k`. Subgrid parameters are forbidden. Do not use `z`, since
  there is a single redshift.
- **The test is leave-one-suite-out, as in the paper:**
  - Stage 1 fits each training suite.
  - Stage 2 learns the population over suites.
  - The held-out suite is scored by its cross-suite predictive evidence.
  - The forward promotion ladder decides which coefficients are marginalized: a richer set is kept only when log Z_cross
    gains more than σ_fold.
- **Primary score: log Z_cross**, summed over the 4 held-out suites. **Higher is better.**
- **Also reported:**
  - the per-suite χ²_ν (C_fv, per-box MAP under the fold prior);
  - L_zero, the benchmark loss with the held-out suite predicted at the population mean and no held-out data;
  - L_map, the paper's adapted loss, using a per-box MAP under the ×1.30 widened prior.

  A form that raises log Z_cross while also lowering χ²_ν and L across suites is the strongest result.
- **Preferences:** prefer simpler, physically interpretable forms. Reviewer concerns are fair targets, for example the SIMBA
  upturn (mostly above k = 10, so only its onset is on this grid), or whether one marginalized coefficient suffices.

## Rules for a form
- S = expression in `k`, the features and the coefficients. It is linear in the `lin` coefficients and may be nonlinear in
  the `nl` coefficients.
- **Limits:** at most 6 coefficients in total, and 1–2 nonlinear ones. Each nonlinear coefficient gets a range:
  `{"lo", "hi", "spacing": "log" | "lin"}`. Use `log` only for strictly positive scales.
- **Roles:** every coefficient gets one of `amplitude`, `scale`, `exponent`, `redshift-evolution`, `other`.
- **Gate 0** (a failed trial still uses up budget):
  - **G0_1:** S → 0 as k → 0.
  - **G0_2:** the leading small-k term is k² or steeper.
  - **G0_3:** S stays bounded as k → ∞.
  - **G0_U, end-slope consistency.** At the fitted coefficients, for ≥ 90 % of boxes, the end step S(9.92) − S(8.67) must
    lie inside the range the simulations themselves span: **−0.040 … +0.036**. An upturn is allowed but not required.

## Other agents run in parallel
- Up to 3 trial agents work at the same time and share the notebook and the trial budget (50).
- A notebook section `## ... — next trial` with no outcome yet is another agent's trial **in progress**. Read those sections
  and pick an idea that is clearly **different** from theirs, not a near copy.
- Append to the notebook with a shell append (`cat >> Loop/lab_notebook.md <<'EOF' ... EOF`), never by rewriting the file.
  Start your section heading with your lane tag.
- `Loop.evaluate` may print that it is waiting for the lock. That is normal: wait, do not resubmit.
- The laptop's 8 cores are shared. Keep any probe or dry run to **at most 2 Python processes**, each with at most 2 threads.
- **Run everything in the foreground**: no background jobs and no monitors.

## What to do, in order
1. **Read** `Loop/lab_notebook.md` and `Loop/trial_state.yaml`, plus the residual tables of the latest trial and of the best
   trial so far (highest log Z_cross). If this loop has no trials yet, start from `Base/RESULT.md` and the paper's Table 1.
2. **Decide one change or one new idea.** Base it on the residuals: which suite, which k, and what sign.
3. **Before submitting**, append to `Loop/lab_notebook.md` a section `## <lane> <date time> — next trial` with:
   - your hypothesis;
   - the expression;
   - your **predicted log Z_cross, the coefficient(s) you expect to be marginalized, and the worst suite**.
4. **Write the submission** JSON in `Loop/submissions/`:
   ```json
   {"expr": "...", "lin": ["A"], "nl": {"k0": {"lo": 0.1, "hi": 30, "spacing": "log"}},
    "roles": {"A": "amplitude", "k0": "scale"}, "features": ["D"],
    "hypothesis": "...", "predicted_logZ_cross": 107500, "predicted_chi2_mean": 6.0}
   ```
5. **Submit once:** `cd M && python -m Loop.evaluate Loop/submissions/<file>.json`. It scores on the full grid (41 nodes per nonlinear coefficient) and takes about 2–6 minutes; it prints
   the trial id, log Z_cross, the marginalized set, χ² per suite, L and the Gate-0 result.
   - If it prints `REJECTED` because of the limits or a compile error, you may fix the JSON and resubmit **once**. The
     rejection already counted as a trial.
   - `HARNESS ERROR` does not count; report it and stop.
6. **Append the outcome** to the same notebook section: the trial id, the actual numbers against your prediction, and the
   lesson for the next agent (one to three lines).
7. **Stop.** Your final reply is one line: `<trial id>: <status> logZ_cross <x> chi2 mean <y> std <z>`.

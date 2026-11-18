# Ablation: the agent vs non-agentic search (2026-11-17). Plan: `Plans/05_ablation.md`

Responds to:
- **RevA:** does the advantage over SR come from the agent, or from differences in constraints, search space or objective?
- **RevC W6:** compare with a fixed grammar, an enumerative search, or SR under the same constraints.

## Setup
**One harness for every arm:** the unchanged `Loop.evaluate`, run through `Ablation/run_arm.py` with each arm's own `PK_ML4PS_LOOP_DIR=Ablation/<arm>/harness`.
- **Checks:** Gate 0 G0_1–3; p ≤ 6 with 1–2 nl (loop rule, no base mode); Stage 1/2; held-out log Z_cross with the promotion ladder on the 41-node grid; χ²_ν. The allowed features are the same as the agents' and the subgrid parameters are forbidden.
- **Isolation:** `Loop/trial_state.yaml` is unchanged (checksum verified before and after).
- **Shim:** `run_arm.py` wraps the harness in a small shim that registers a yaml representer for np.float64. Without it, a form with no linear coefficient crashes the yaml dump *after* scoring, which left the trial id frozen; the first attempt's logs are in `A*/run_attempt1.log`. No number is affected.

| arm | changes vs. the Plan 04 PySR baseline | candidates | full evaluations |
|---|---|---|---|
| A1-U / A1-C | **Objective.** Every hall-of-fame equation of the Plan 04 pooled U / C searches is selected by log Z_cross rather than PySR's MSE "best" | 24 / 19 equations (U_c06 has no constants and is skipped) | 23 / 19 |
| A2 | **Search space + constraints.** A new PySR template a₁(x)W(k e^{−f₁(x)}) + a₂(x)W(k e^{−f₂(x)}), with W(u) = u²/(1+u²) and operators {+, −, ×, exp}. It is a sum of two W kernels, as in the agents' forms, with S ∝ k² and bounded by construction. One pooled 2700 s search; otherwise Plan 04 settings | 18 equations | 18 |
| A3 | **Fixed grammar, enumerative.** −D Σ_{m≤2} A_m w_m W(k/(k_m s_m)). Weights w ∈ {1, ω_b, 1−ω_b, f_b}; scale modifiers s ∈ {1, D^β, (1+αD)^{−1}, (Ω_m/0.3)^{1/3}}; scales and amplitudes shared or separate. **Seed (Table 1) primitives only**, with every agent-discovered motif excluded: the q weights, the √D and (D/0.05)^0.75 scales, and the knee kernels | 292 forms after deduplication. The prescreen is log Z_cross at R = {} on 41 nodes (about 15 s per form); all 292 pass Gate 0 | top 60 by prescreen |

**Conversion of PySR equations** (`Ablation/sr_hof.py`, using `PySR.to_form.to_submission` with `max_coeff=6, max_nl=2`):
- **The rest are frozen:** the A1-C winner keeps 9 frozen constants, comparable to the ~10 frozen constants in the agent forms.
- **Forms with no nonlinear constant:** the harness needs 1–2 nl, so the most influential constant is declared nl (range ×/÷10) and flagged `nl_promoted` in the manifest.
- **Linearity check:** sympy's `simplify` hung for more than 2.5 h on C_c16, so the linearity check uses a numeric zero test at 40 random points, patched inside sr_hof only. The resulting submissions are field-for-field identical to the symbolic ones for all 33 equations converted before the hang.

## Results (`ablation.json`, `RESULT_table.md`, figure `PaperUpdate/figs/out/ablation_v4`)
T47 is 35.35 nats/box with σ_fold = 0.37.

| search | full evals | best log Z/box | Δ vs T47 | Δ/σ_fold | χ²_ν | L_MAP (C_fv/h) |
|---|---|---|---|---|---|---|
| Agent loop (T47) | 60 | **35.35** | 0 | 0 | 8.99 | 17.9 |
| PySR U, MSE pick (Plan 04) | 1 | 30.19 | −5.15 | −13.8 | 21.4 | 41.4 |
| PySR C, MSE pick (Plan 04) | 1 | 32.23 | −3.12 | −8.3 | 7.00 | 24.4 |
| A1-U, log Z selection | 23 | 30.19 | −5.15 | −13.8 | 21.4 | 41.4 |
| A1-C, log Z selection | 19 | 34.33 | −1.01 | −2.7 | 6.60 | 23.3 |
| A2, two-kernel template | 18 | 32.08 | −3.27 | −8.8 | 15.9 | 36.0 |
| A3, seed grammar | 60 | 34.23 | −1.12 | −3.0 | 10.6 | 24.8 |
| Seed form, paper rank 1 (P1, 3 nl) | 1 | 32.89 | −2.46 | −6.6 | 8.14 | – |

- **The objective matters most for SR.** Selecting the C hall of fame by log Z instead of MSE gains 2.1 nats/box (32.23 → 34.33).
  - The winner is C_c25: complexity 25, p = 3, marginalized b1 and A0.
  - Under the loop rule, C_c11 (the MSE pick) reproduces Plan 04 exactly: 120,080.5 total, 32.23/box.
- **Unconstrained SR does not benefit.** 21 of the 23 U equations fail Gate 0, mostly G0_1/G0_2 (no k² onset). The two that pass include the MSE pick itself.
- **The matched template (A2) did worse than the simple C template.** Its best equation is C2_c29 at 32.08, with χ²_ν 15.9.
  - Four sub-expressions in one 45-min search give a harder MSE problem. Most of its hall of fame does not marginalize anything.
  - This is one search at one seed; no claim is made beyond that budget.
- **The seed-grammar enumeration reaches 34.23** with prescreen rank 1 (G262), and the full scoring of 59 more forms never beat it.
  - G262 is −D[A₁(1−ω_b)W(k(1+αD)/k₁) + A₂ f_b W(k/(k₁(Ω_m/0.3)^{1/3}))], with p = 4 and marginalized α and k₁.
  - Under the loop rule it beats every seed form (P1, 32.89).
- **The agent is still ahead.** Every controlled arm stays ≥ 2.7 σ_fold below T47 (1.0–1.1 nats/box for the best two).
  - **Cost to match the best non-agentic arm:** the agent's best-so-far passes 34.23 at trial 29 and 34.33 at trial 29 (T29 = 34.35).
  - **Its lead comes later:** it reaches the T47 band at trial 36 (T36) and T47 itself at trial 47, through motifs absent from the grammar and from both PySR templates (q weights, √D scale, the (D/0.05)^0.75 knee).
- **Fit quality of the best non-agentic forms:** L_MAP is 23–25, against 17.9 for T47.

## Disclosures
- The agents also ran off-budget dry runs (≥ 82 mentioned in the notebook). In A3 the prescreen plays that role; A1 and A2 had no prescreen.
- The PySR pooled equations, like the agents' forms, are selected on all 4 suites.
- The A3 prescreen at R = {} is only an ordering; it puts the eventual winner first.
- The A2 search used 2700 s, the same as each Plan 04 search. No more seeds or time were tried.

## Files
- **Scripts:** `grammar.py`, `prescreen.py`, `run_search_c2.py`, `sr_hof.py`, `run_arm.py`, `winners_map.py`, `analyze.py`, `patch_appendix.py`.
- **Per-arm outputs:** `A*/harness/trial_state.yaml`, `A*/subs/`, `A1|A2/manifest.jsonl`, `A3/grammar.jsonl`, `A3/prescreen.jsonl`.

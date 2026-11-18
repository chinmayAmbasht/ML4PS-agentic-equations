# Bayesian agentic equation discovery with discrepant simulators

Code, processed data and agent-search records for the ML4PS workshop paper *Bayesian Agentic Equation Discovery for Cosmological Inference with Discrepant Simulators* (Ambasht, Wadekar & Pandey; NeurIPS 2026 workshop, Machine Learning for Physical Sciences).

We search for a compact analytic equation for the baryonic suppression of the matter power spectrum,
$\mathcal{S}(k)=P_\mathrm{hydro}/P_\mathrm{DMO}-1$, that transfers across four CAMELS hydrodynamical suites
(IllustrisTNG, SIMBA, Astrid and Swift-EAGLE) at $z=0$. LLM agents propose candidate equations, and every candidate goes through the same scoring harness:

1. **Physical checks.** $\mathcal{S}\to0$ and $\mathcal{S}\propto k^{\ge2}$ at low $k$, bounded at high $k$, and an end-slope band.
2. **Stage 1.** A within-suite fit with linear coefficients integrated analytically on a 41-node nonlinear grid.
3. **Stage 2.** Hierarchical pooling across suites: the population mean $\boldsymbol\mu$ and the between-suite covariance $\Sigma_\mathrm{suite}$.
4. **Held-out evidence.** Leave-one-suite-out predictive evidence $\log Z_\mathrm{cross}$, with a promotion ladder that selects which coefficients vary between suites.

The repository reproduces every number, table and figure in the paper.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./reproduce.sh quick        # ~1 min: Table 1, the full pool table, the ranking and Fig. 3 from the stored results
```

Score your own equation with exactly the harness used for the paper (about 10 minutes per form on a laptop):

```bash
python score_form.py Loop/submissions/A17_T36_split_knee_amps.json   # paper rank 1; expect log Z_cross = 131,696 (35.345 per box)
```

A submission is a small JSON file; see the docstring of `score_form.py` and any file in `Loop/submissions/`.

## What reproduces what

| Paper item | Command | Output | Time |
|---|---|---|---|
| Table 1 (main text) | `python -m PaperUpdate.table1_main` | `outputs/tables/table1_rows.tex` | s |
| Table 2 (all 64 forms) | `python -m PaperUpdate.make_tex` | `outputs/tables/pool_table.tex` | s |
| Ranking of all 64 forms | `python -m PaperUpdate.rank_all` | `PaperUpdate/ranking.{json,csv}`, `RANKING.md` | s |
| Fig. 1 (data) | `python -m PaperUpdate.figs.fig1_oldstyle` | `PaperUpdate/figs/out/pk_suppression_cv`, `fb_contours_k1p1` | min |
| Fig. 2 (pipeline) | TikZ source | `PaperUpdate/figs/agentic_loop_fig.tex` | – |
| Fig. 3 (benchmark) | `python -m PaperUpdate.figs.fig3_benchmark T47 T36 T43 T39 T51` | `PaperUpdate/figs/out/pooled_comparison_v4` | s |
| Held-out fits (appendix) | `python -m PaperUpdate.figs.fig_fits_appendix T47 T36 T43 T39 T51` | `.../fit_overlay_v4` | tens of min |
| Finite-volume covariance (appendix) | `python -m PaperUpdate.figs.fig_cfv_appendix` | `.../cfv_matrices_v4` | s |
| Non-agentic ablation (appendix figure and table) | `python -m Ablation.analyze` | `PaperUpdate/figs/out/ablation_v4`, `Ablation/ablation_table.tex` | s |
| Ablation overview (extra, not in the paper) | `python -m Ablation.summary_fig` | `Ablation/figs/ablation_summary` | s |

`./reproduce.sh figures` runs the figure rows above. `./reproduce.sh full` re-runs every computation from the processed data: all 64 forms, the benchmark, the van Daalen pipeline fit and the MAP sensitivity. It takes many CPU-hours. The PySR searches and the ablation arms are listed in `reproduce.sh` but are not run by default.

## Repository layout

```
Scripts/            scoring engine
  engine/             forms, Gate 0 physical checks, Stage 1 (stage1.py), Stage 2 (stage2.py), evidence + ladder (scoring.py, evidence.py), chi2
  data/               build the processed data from raw CAMELS (suppression, f_b, deficit D)
  common/config.py    every registered constant (k grid, covariance shrinkage, priors, grid sizes)
Loop/               the agent search (60 trials)
  evaluate.py         the harness each agent submitted to
  brief.md            the brief every trial agent received
  lab_notebook.md     the shared notebook: each agent's hypothesis, prediction, outcome and lesson
  submissions/        the 60 submitted forms (JSON)
  trial_state.yaml    the scores of every trial, in order
  residuals/          per-suite residual tables that later agents read
Base/               the four seed forms (the previous version of the paper) and van Daalen+20, scored by the same harness
PaperUpdate/        ranking, benchmark (random forest, van Daalen, PySR), losses, paper figures and tables
PySR/               symbolic-regression baseline: search settings, hall-of-fame equations, conversion to harness forms
Ablation/           controlled non-agentic searches (PySR selected by log Z, a two-kernel PySR template, seed-grammar enumeration)
Evidence/           evidence re-pass of the leading forms: full-grid ladder, Σ_suite scaled ×0.5 and ×2, coarse-grid check
Artifacts/data/     processed CAMELS data (about 25 MB, see below)
score_form.py       score any candidate equation
reproduce.sh        regenerate the paper outputs
```

## Data

`Artifacts/data/` ships the processed inputs:
- `pk_suppression_z000_allk.h5`: $\mathcal{S}(k)$ for every LH and CV box at $z=0$, on all native $k$ bins;
- `observables_zfb.h5` and `observables_z000.h5`: the per-box halo baryon fraction $f_{b,200c}$ (halos with $5\times10^{12}<M_{200c}<10^{14}\,M_\odot/h$), $f_b$ in six mass bins, the deficit $D=\Omega_b/\Omega_m-f_b$, $\Omega_m$ and $\sigma_8$.

All of it is derived from the public [CAMELS](https://camels.readthedocs.io) data. To rebuild it, download the CAMELS power spectra and halo profiles, point `CAMELS_ROOT` at them, and run:

```bash
export CAMELS_ROOT=/path/to/camels
python -m Scripts.data.build_suppression --z 0.00
python -m Scripts.data.observables_zfb
```

The expected directory layout is given by `SUITE_DIRS`, `PROFILE_DIR` and `FB_Z_DIR` in `Scripts/common/config.py`.

## The agent search

Each of the 60 trials was a fresh Claude Opus 5.5 agent. It had default sampling settings and no memory of earlier trials other than the files on disk. Up to three trials ran in parallel (lanes A, B and C).

What each agent did:
- read `Loop/brief.md`, the lab notebook, `trial_state.yaml` and the residual tables;
- could run Python on the data (it was allowed dry runs);
- submitted exactly one form through `Loop/evaluate.py`;
- appended its hypothesis, prediction and lesson to `Loop/lab_notebook.md`.

The search started from the four seed forms in `Base/` and stopped when the budget of 60 submissions was spent.

Two points to note:
- **Agent tag.** Agents named their submission files with their own tags (e.g. `A17_T36_...`), and trial ids (`T47`) were assigned at submission.
- **Trials T51–T60.** These are ten forms aimed at reducing $\chi^2$. They were resubmitted to the main loop with identical scores.

The loop design borrows the stateless-agent, shared-notebook and fixed-budget structure of AgenticTuner, but it is not a reimplementation of that method.

Paths in the records have been replaced by `<repo>` (the repository root) and `<scratch>` (a temporary directory). The brief refers to a copy of the earlier paper draft under `Paper/`, which agents could read. That draft is not included here.

**Disclosure.** The leading forms contain a few fixed constants (0.68, 0.05, 0.75 and the $\sqrt{D}$ scalings) that the agents chose while seeing scores on all four suites. The same four suites rank the forms and select the varying coefficients, so the winner's $\log Z_\mathrm{cross}$ is optimistically biased, as for any best-of-many selection.

## Notes on the code

- Run every command from the repository root. Fold caches are written to `$ML4PS_CACHE` (default `./cache`), which is safe to delete.
- Code comments refer to dated design decisions ("owner 2026-11-11", `Plans/...`). These record when and why a choice was made during the project. The plans themselves are not part of this release.
- `Scripts/common/config.py` is the single source of every registered constant. Changing a value there changes all scores.
- The harness also prints a quick diagonal-σ_k loss (`L_zero`, `L_map`) for the agents. It is not the paper's loss $L$. The paper's $L$ uses the full correlated covariance $C_{\mathrm{fv},j}/h$ and is computed by `PaperUpdate/Lcorr_all.py` and `PaperUpdate/Benchmark/bench.py`. Stored trials scored before the σ_k convention was finalized carry the older diagonal values; their $\log Z_\mathrm{cross}$ is unaffected.
- Verified: re-running `score_form.py` on paper rank 1 in a fresh checkout reproduces the stored $\log Z_\mathrm{cross}$, every promotion-ladder step and the per-fold values exactly. `./reproduce.sh quick` and the Fig. 1, covariance and ablation scripts reproduce the paper's tables and figures pixel for pixel.

## Citation

See `CITATION.cff`. Please also cite CAMELS (Villaescusa-Navarro et al. 2021) when using the data.

## License

MIT (see `LICENSE`).

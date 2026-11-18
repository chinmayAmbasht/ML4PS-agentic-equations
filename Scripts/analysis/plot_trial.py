"""Fit one loop trial's form and plot it against the data (held-out predictions, leave-one-suite-out).

Same fit as Loop.evaluate (coarse folds, 11 nodes per nonlinear coefficient): for each held-out suite j, the fold prior
from the other 3 suites; each box of j gets its per-box MAP (prior x1.30, C_fv + C_disc likelihood). Also the zero-shot
prediction at the fold mean mu_j. Plotting only: no score is written.

Figures follow v1's scheme (Scripts/figures/zfb6_fits.py there; style in Scripts/figures/panelspec.py), in Figures/trial_<id>/:
  fig_zfb6_fit_<id>            4 suites x 6 z; three boxes per suite (valid at every z) at the 10/50/90th percentiles of
                               S(k ~ 10, z = 0), the same boxes in every column; data +- sqrt(diag C_fv), adapted solid,
                               zero-shot dashed
  fig_zfb6_fit_residuals_<id>  (adapted - data)/sigma_k over all held-out boxes: median and 16-84%; grey band |dS| < sigma_k
  fig_chi2_<id>                pooled chi^2_nu (C_fv) per held-out suite, and the cumulative per-box chi^2_nu
plus fit.npz (the predictions) and a README.

Usage:  python -m Scripts.analysis.plot_trial --trial T07 [--refit]
        python -m Scripts.analysis.plot_trial --trial best        (lowest mean chi^2 passing Gate 0)
"""
import argparse
import json
import os
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[2]
os.environ.setdefault("PK_FROZEN", str(V2 / "Loop" / "candidates_frozen.jsonl"))
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "v2plots", f"ns"))       # fold caches stay out of the loop's folders
os.environ.setdefault("PK_NPAR", "4")

import numpy as np
import yaml
from scipy.optimize import minimize

from Scripts.common import config as C
from Scripts.engine import chi2 as X2
from Scripts.engine import evidence as E
from Scripts.engine import registry as REG
from Scripts.engine import scoring as SC
from Scripts.engine import stage1 as S1
from Scripts.engine.par import pmap
from Scripts.figures import panelspec as PS
from Scripts.run.score_forms import build_blocks, form_from_frozen

NZ, NK = len(C.REDSHIFTS), C.N_K
K = np.asarray(C.K_GRID)


def _boxes(args):
    """Per-box MAP and zero-shot predictions for one held-out suite (the chi2._suite fit, keeping the curves)."""
    form, f, b = args
    mu, Si = np.asarray(f["mu"], float), np.linalg.inv(C.PRIOR_WIDENING_FOR_MAP_DIAGNOSTIC * np.asarray(f["Sigma"], float))
    cd = np.diag(f["cdisc_diag"])
    zs = form.predict(b.obs, C.REDSHIFTS, E.to_native(form, f["order"], mu))
    mp = np.full_like(b.y, np.nan); chi = np.zeros(len(b.y))
    for i in range(len(b.y)):
        o = {k: v[[i]] for k, v in b.obs.items()}; mi = b.mask[[i]]
        wt = S1.BlockWhitener(b.cfv + cd, NK, mi); wf = S1.BlockWhitener(b.cfv, NK, mi)
        yw = wt.w(b.y[[i]])

        def neg(c):
            p = form.predict(o, C.REDSHIFTS, E.to_native(form, f["order"], c))
            if not np.all(np.isfinite(p[mi])):
                return 1e30
            r = yw - wt.w(p); d = c - mu
            return 0.5 * float(np.sum(r * r)) + 0.5 * float(d @ Si @ d)
        c = minimize(neg, mu.copy(), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000}).x
        p = form.predict(o, C.REDSHIFTS, E.to_native(form, f["order"], c))
        mp[i] = p[0]
        chi[i] = float(np.sum((wf.w(b.y[[i]]) - wf.w(p)) ** 2)) / max(int(mi.sum()) - form.n_coeff, 1)
    return {"y": np.where(b.mask, b.y, np.nan), "map": np.where(b.mask, mp, np.nan), "zs": np.where(b.mask, zs, np.nan),
            "sig": np.sqrt(np.diag(b.cfv)), "chi2_box": chi}


def fit(tid):
    row = next(r for r in REG.frozen_rows() if r["form_id"] == tid)
    form = form_from_frozen(row, n_nodes=C.LOOP_NL_NODES)
    blocks, _, _ = build_blocks(form, C.REDSHIFTS)
    fp = X2.fold_params(SC.cross_suite_folds(form, blocks, C.REDSHIFTS))
    res = dict(zip(C.SUITES, pmap(_boxes, [(form, fp[j], blocks[j]) for j in C.SUITES])))
    return form, fp, res



PCT = (10, 50, 90)
TEX_SYM = {"sigma_8": r"\sigma_8", "Omega_m": r"\Omega_m", "Omega_b_over_Omega_m": r"\Omega_b/\Omega_m", "f_b": "f_b",
           "k0": "k_0", "k1": "k_1", "k2": "k_2"}


def _tex(expr):
    """The expression as LaTeX (sympy), symbols renamed for print."""
    import sympy as sp
    e = sp.sympify(expr)
    names = {s: sp.Symbol(TEX_SYM.get(s.name, s.name)) for s in e.free_symbols}
    return sp.latex(e.xreplace(names), long_frac_ratio=2.5, mul_symbol="dot").replace(r"\cdot", r"\,")


def _reps(y):
    """Three boxes valid at every z, at the PCT percentiles of S(k ~ 10, z = 0)."""
    ok = np.where(np.isfinite(y.reshape(len(y), NZ, NK)).all(axis=(1, 2)))[0]
    ik = int(np.argmin(np.abs(K - 10.0)))
    v = y[ok, ik]
    return [int(ok[np.argmin(np.abs(v - np.percentile(v, q)))]) for q in PCT]


def _legend_fit():
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="0.3", marker="o", ls="", ms=3, label=r"data $\pm\sqrt{\mathrm{diag}\,C_\mathrm{fv}}$"),
         Line2D([], [], color="0.3", ls="-", label=r"adapted (per-box MAP, prior $\times1.3$)"),
         Line2D([], [], color="0.3", ls="--", label="zero-shot (fold prior mean)")]
    return h + [Line2D([], [], color=s, lw=3, label=rf"box at {q}th pct of $\mathcal{{S}}(k\!\approx\!10, z\!=\!0)$")
                for s, q in zip(PS.BOX_SHADES, PCT)]


def fig_fit(tid, eq, res, out):
    import matplotlib.pyplot as plt
    PS.style()
    fig, axes = plt.subplots(len(C.SUITES), NZ, figsize=(2.05 * NZ, 1.75 * len(C.SUITES)), sharex=True, sharey="row", squeeze=False)
    for r, j in enumerate(C.SUITES):
        d = res[j]; reps = _reps(d["y"])
        for c, z in enumerate(C.REDSHIFTS):
            ax = axes[r][c]; sl = slice(c * NK, (c + 1) * NK)
            ax.axhline(0, color="0.75", lw=0.5)
            for i, col in zip(reps, PS.BOX_SHADES):
                ax.errorbar(K, d["y"][i, sl], yerr=d["sig"][sl], fmt="o", ms=2.2, color=col, elinewidth=0.6, capsize=0, zorder=3)
                ax.plot(K, d["map"][i, sl], "-", color=col, lw=1.1, zorder=2)
                ax.plot(K, d["zs"][i, sl], "--", color=col, lw=0.7, alpha=0.7, zorder=1)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(rf"$z={z:g}$", fontsize=PS.FS)
            if c == 0:
                ax.set_ylabel(PS.SUITE_TEX[j] + "\n" + r"$\mathcal{S}(k,z)$", fontsize=PS.FS - 1)
            if r == len(C.SUITES) - 1:
                ax.set_xlabel(PS.K_LABEL, fontsize=PS.FS - 1)
    fig.legend(handles=_legend_fit(), loc="lower center", ncol=3, frameon=False, fontsize=PS.FS - 1, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(rf"{tid} (zfb6): $\mathcal{{S}}={eq}$" + "\n(held-out suite; prior from the other three)", fontsize=PS.FS)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    PS.save(fig, out / f"fig_zfb6_fit_{tid}")


def fig_resid(tid, res, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sk = np.tile(np.asarray(C.SIGMA_K), NZ)
    PS.style()
    fig, axes = plt.subplots(len(C.SUITES), NZ, figsize=(2.05 * NZ, 1.6 * len(C.SUITES)), sharex=True, sharey=True, squeeze=False)
    col = PS.MODEL_PALETTE[0]; stats = {}
    for r, j in enumerate(C.SUITES):
        d = res[j]
        for c, z in enumerate(C.REDSHIFTS):
            ax = axes[r][c]; sl = slice(c * NK, (c + 1) * NK)
            ax.axhspan(-1, 1, color="0.92", zorder=0); ax.axhline(0, color="0.6", lw=0.5)
            m = np.isfinite(d["y"][:, sl]).all(axis=1)
            e = (d["map"][m, sl] - d["y"][m, sl]) / sk[sl]
            lo, med, hi = np.percentile(e, [16, 50, 84], axis=0)
            ax.fill_between(K, lo, hi, color=col, alpha=0.15, lw=0); ax.plot(K, med, "-", color=col, lw=1.0)
            stats.setdefault(j, []).append(float(np.median(np.abs(e))))
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(rf"$z={z:g}$", fontsize=PS.FS)
            if c == 0:
                ax.set_ylabel(PS.SUITE_TEX[j] + "\n" + r"$\Delta\mathcal{S}/\sigma_k$", fontsize=PS.FS - 1)
            if r == len(C.SUITES) - 1:
                ax.set_xlabel(PS.K_LABEL, fontsize=PS.FS - 1)
    axes[0][0].set_ylim(-6, 6)
    h = [Line2D([], [], color=col, lw=2, label=tid), Line2D([], [], color="0.85", lw=6, label=r"$|\Delta\mathcal{S}| < \sigma_k$")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=PS.FS - 1, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(rf"{tid} (zfb6): adapted prediction $-$ data, in units of $\sigma_k$: "
                 r"median (line) and 16--84\% of held-out boxes (band)", fontsize=PS.FS)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    PS.save(fig, out / f"fig_zfb6_fit_residuals_{tid}")
    return stats


def fig_chi2(tid, res, rec, out):
    import matplotlib.pyplot as plt
    PS.style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    v = [rec["chi2_per_suite"][j] for j in C.SUITES]; x = np.arange(len(C.SUITES))
    a1.bar(x, v, 0.6, color=[PS.SUITE_COLOR[j] for j in C.SUITES], edgecolor="white", lw=0.3)
    for xi, vi in zip(x, v):
        a1.text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=PS.FS - 1)
    a1.axhline(C.CHI2_MEAN_MAX, color="0.3", lw=0.8, ls="--",
               label=rf"presentable: mean $\le{C.CHI2_MEAN_MAX:g}$, std $\le{C.CHI2_STD_MAX:g}$")
    a1.legend(frameon=False, fontsize=PS.FS - 1, loc="upper left")
    a1.set_xticks(x, [PS.SUITE_TEX[j] for j in C.SUITES])
    a1.set_ylabel(r"pooled $\chi^2_\nu\ (C_\mathrm{fv})$, held-out suite")
    a1.set_title(rf"mean {rec['chi2_mean']:.2f}, std {rec['chi2_std']:.2f}", fontsize=PS.FS)
    for j in C.SUITES:
        pb = np.sort(res[j]["chi2_box"])
        a2.plot(pb, np.arange(1, len(pb) + 1) / len(pb), color=PS.SUITE_COLOR[j], lw=1.2,
                label=rf"{PS.SUITE_TEX[j]} (median {np.median(pb):.2f})")
    a2.set_xscale("log"); a2.axvline(C.CHI2_MEAN_MAX, color="0.5", lw=0.6, ls=":"); a2.set_xlim(0.3, 300)
    a2.set_xlabel(r"per-box $\chi^2_\nu\ (C_\mathrm{fv})$"); a2.set_ylabel("cumulative fraction of boxes")
    a2.legend(frameon=False, fontsize=PS.FS - 1, loc="lower right")
    fig.suptitle(rf"{tid}: reduced $\chi^2$ at the per-box MAP (prior $\times1.3$, fit over $C_\mathrm{{fv}}+C_\mathrm{{disc}}$), "
                 "each suite held out", fontsize=PS.FS)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    PS.save(fig, out / f"fig_chi2_{tid}")


def readme(tid, rec, res, stats, out):
    zs = list(C.REDSHIFTS)
    L = [f"# {tid}: model-vs-data figures (diagnostic only; feeds nothing)", "",
         f"Form: `S = {rec['expr']}`", "",
         "- Data: S(k, z) of the held-out suite's boxes; error bars sqrt(diag C_fv).",
         "- Predictions: the coarse cross-suite fold of the held-out suite (prior from the other three; the loop's fit).",
         "  Solid: per-box MAP, prior x1.30 over C_fv + C_disc. Dashed: zero-shot fold prior mean.",
         f"- `fig_zfb6_fit_{tid}`: three boxes per suite (valid at every z) at the {PCT} percentiles of S at the k nearest",
         "  10 h/Mpc at z = 0; the same three boxes in every z column.",
         f"- `fig_zfb6_fit_residuals_{tid}`: (adapted - data)/sigma_k over all held-out boxes valid at that z; grey = |dS| < sigma_k.",
         f"- `fig_chi2_{tid}`: pooled chi^2_nu (C_fv) per held-out suite (the loop's numbers) and per-box chi^2_nu.", "",
         "Median |dS|/sigma_k over boxes and k (adapted):", "",
         "| suite | " + " | ".join(f"z={z:g}" for z in zs) + " |", "|---|" + "---|" * len(zs)]
    L += [f"| {j} | " + " | ".join(f"{x:.2f}" for x in v) + " |" for j, v in stats.items()]
    L += ["", "| suite | pooled chi2_nu (C_fv) | median per box | 16-84% per box | frac boxes < 2 |", "|---|---|---|---|---|"]
    for j in C.SUITES:
        pb = res[j]["chi2_box"]
        L.append(f"| {j} | {rec['chi2_per_suite'][j]:.2f} | {np.median(pb):.2f} | {np.percentile(pb, 16):.2f}-"
                 f"{np.percentile(pb, 84):.2f} | {np.mean(pb < 2):.2f} |")
    (out / "README.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True, help="a trial id such as T07, or best")
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    trials = yaml.safe_load((V2 / "Loop" / "trial_state.yaml").read_text())["trials"]
    scored = [t for t in trials if t.get("chi2_mean") is not None]
    if a.trial == "best":           # lowest mean chi^2 among trials that pass all of Gate 0
        ok = [t for t in scored if t["gate0"].get("G0_U")]
        a.trial = min(ok, key=lambda t: t["chi2_mean"])["trial"]
        print(f"best trial so far: {a.trial}")
    rec = next((t for t in scored if t["trial"] == a.trial), None)
    if rec is None:
        sys.exit(f"no scored trial {a.trial!r}; scored trials: {' '.join(t['trial'] for t in scored)} (or use --trial best)")
    out = V2 / "Figures" / f"trial_{a.trial}"; out.mkdir(parents=True, exist_ok=True)
    npz = out / "fit.npz"
    if a.refit or not npz.exists():
        form, fp, res = fit(a.trial)
        np.savez_compressed(npz, fp=json.dumps(fp), **{f"{j}|{k}": v for j, r in res.items() for k, v in r.items()})
    z = np.load(npz)
    res = {j: {k: z[f"{j}|{k}"] for k in ("y", "map", "zs", "sig", "chi2_box")} for j in C.SUITES}
    eq = _tex(rec["expr"])
    fig_fit(a.trial, eq, res, out)
    stats = fig_resid(a.trial, res, out)
    fig_chi2(a.trial, res, rec, out)
    readme(a.trial, rec, res, stats, out)
    print("figures:", out)


if __name__ == "__main__":
    sys.path.insert(0, str(V2))
    main()

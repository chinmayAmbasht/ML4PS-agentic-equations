"""z = 0 fit plots for the ML4PS best forms (held-out predictions, leave-one-suite-out). Plotting only: no score is written.

For each held-out suite j, the fold prior comes from the other 3 suites (registered Stage 2). Each box of j gets its per-box MAP
(prior x1.30, C_fv + C_disc likelihood; the same fit as Scripts.analysis.plot_trial._boxes, reused unchanged), plus the
zero-shot prediction at the fold mean. Default grid: the full one (41 nodes per nonlinear coefficient, 21 for forms with 3+),
matching Evidence/RESULT.md; --nodes 11 reproduces the loop's coarse numbers.

Figures in Figures/fits/:
  fig_fit_<label>      top: 3 boxes per held-out suite (10/50/90th pct of S(k~10)), data +- sqrt(diag C_fv), adapted solid,
                       zero-shot dashed; bottom: (adapted - data)/sigma_k over all held-out boxes, median and 16-84%
  fig_fit_compare      median (adapted - data)/sigma_k per suite for every form, and per-suite chi^2_nu bars
plus <label>.npz (the predictions) and chi2.json.

Usage:  python -m Loop.plot_fits label=path.json ... [--nodes N]
        python -m Loop.plot_fits --replot label ...      (redraw from the saved .npz, no refit)
"""
import json
import os
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns"))
os.environ.setdefault("PK_FROZEN", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"frozen.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.path.insert(0, str(M))

import numpy as np

from Loop import form_labels as FL
from Scripts.analysis.plot_trial import _boxes
from Scripts.common import config as C
from Scripts.engine import chi2 as X2
from Scripts.engine import scoring as SC
from Scripts.engine.forms import Form
from Scripts.engine.par import pmap
from Scripts.figures import panelspec as PS
from Scripts.run.score_forms import build_blocks

K = np.asarray(C.K_GRID)
OUT = M / "Figures" / "fits"
PCT = (10, 50, 90)
FORM_COLOR = ["#009E73", "#0072B2", "#D55E00", "#7F7F7F", "#CC79A7"]


def form_of(label, sub, n_nodes):
    nl = {k: {"lo": float(v["lo"]), "hi": float(v["hi"]), "spacing": v["spacing"], "n_nodes": n_nodes} for k, v in sub["nl"].items()}
    return Form(label, sub["expr"], list(sub["lin"]), nl, dict(sub["roles"]), list(sub["features"]))


def fit(label, sub, n_nodes):
    form = form_of(label, sub, n_nodes)
    blocks, _, _ = build_blocks(form, C.REDSHIFTS)
    fp = X2.fold_params(SC.cross_suite_folds(form, blocks, C.REDSHIFTS))
    res = dict(zip(C.SUITES, pmap(_boxes, [(form, fp[j], blocks[j]) for j in C.SUITES])))
    return form, res


def reps(y):
    ok = np.where(np.isfinite(y).all(axis=1))[0]
    v = y[ok, int(np.argmin(np.abs(K - 10.0)))]
    return [int(ok[np.argmin(np.abs(v - np.percentile(v, q)))]) for q in PCT]


def resid(d):
    m = np.isfinite(d["y"]).all(axis=1)
    return (d["map"][m] - d["y"][m]) / np.asarray(C.SIGMA_K)


def fig_form(label, title, res, chi, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    PS.style()
    fig, ax = plt.subplots(2, 4, figsize=(12.5, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    for c, j in enumerate(C.SUITES):
        d = res[j]; a = ax[0][c]
        a.axhline(0, color="0.75", lw=0.5)
        for i, col in zip(reps(d["y"]), PS.BOX_SHADES):
            a.errorbar(K, d["y"][i], yerr=d["sig"], fmt="o", ms=2.6, color=col, elinewidth=0.6, capsize=0, zorder=3)
            a.plot(K, d["map"][i], "-", color=col, lw=1.2, zorder=2)
            a.plot(K, d["zs"][i], "--", color=col, lw=0.8, alpha=0.75, zorder=1)
        a.set_xscale("log"); a.set_title(rf"{PS.SUITE_TEX[j]} held out: $\chi^2_\nu$ {chi[j]:.2f}", fontsize=PS.FS)
        b = ax[1][c]; e = resid(d)
        b.axhspan(-1, 1, color="0.92", zorder=0); b.axhline(0, color="0.6", lw=0.5)
        lo, med, hi = np.percentile(e, [16, 50, 84], axis=0)
        b.fill_between(K, lo, hi, color=PS.SUITE_COLOR[j], alpha=0.2, lw=0); b.plot(K, med, "-", color=PS.SUITE_COLOR[j], lw=1.2)
        b.set_ylim(-5, 5); b.set_xlabel(PS.K_LABEL, fontsize=PS.FS - 1)
        if c == 0:
            a.set_ylabel(r"$\mathcal{S}(k)=P_\mathrm{hydro}/P_\mathrm{dmo}-1$", fontsize=PS.FS - 1)
            b.set_ylabel(r"(adapted $-$ data)$/\sigma_k$" + (" (inflated)" if C.SIGMA_K_SOURCE == "inflated" else ""), fontsize=PS.FS - 1)
    h = [Line2D([], [], color="0.3", marker="o", ls="", ms=3, label=r"data $\pm\sqrt{\mathrm{diag}\,C_\mathrm{fv}}$"),
         Line2D([], [], color="0.3", ls="-", label=r"adapted (per-box MAP, prior $\times1.3$)"),
         Line2D([], [], color="0.3", ls="--", label="zero-shot (fold prior mean)")]
    h += [Line2D([], [], color=s, lw=3, label=rf"box at {q}th pct of $\mathcal{{S}}(k\!\approx\!10)$") for s, q in zip(PS.BOX_SHADES, PCT)]
    h += [Line2D([], [], color="0.5", lw=6, alpha=0.3, label=r"residual 16--84\% of boxes (median line); grey $|\Delta|<\sigma_k$")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=PS.FS - 1, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.5, 0.003, FL.NOTATION, ha="center", va="bottom", fontsize=PS.FS - 2)
    fig.suptitle(title, fontsize=PS.FS + 1)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    PS.save(fig, out / f"fig_fit_{label}")


def fig_compare(rows, out):
    import matplotlib.pyplot as plt
    PS.style()
    res_h = 0.3 * len(rows) + 0.45; H = 6.2 + res_h
    fig = plt.figure(figsize=(12.5, H))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9])
    for c, j in enumerate(C.SUITES):
        a = fig.add_subplot(gs[0, c])
        a.axhspan(-1, 1, color="0.92", zorder=0); a.axhline(0, color="0.6", lw=0.5)
        for (label, title, res, chi), col in zip(rows, FORM_COLOR):
            a.plot(K, np.median(resid(res[j]), axis=0), "-o", ms=2.5, color=col, lw=1.2, label=FL.rank_tag(label))
        a.set_xscale("log"); a.set_ylim(-3, 3); a.set_xlabel(PS.K_LABEL, fontsize=PS.FS - 1)
        a.set_title(f"{PS.SUITE_TEX[j]} held out", fontsize=PS.FS)
        if c == 0:
            a.set_ylabel(r"median (adapted $-$ data)$/\sigma_k$" + (" (inflated)" if C.SIGMA_K_SOURCE == "inflated" else ""), fontsize=PS.FS - 1)
            a.legend(frameon=False, fontsize=PS.FS - 2, loc="lower left")
    b = fig.add_subplot(gs[1, :])
    w = 0.8 / len(rows)
    for i, ((label, title, res, chi), col) in enumerate(zip(rows, FORM_COLOR)):
        v = [chi[j] for j in C.SUITES]
        b.bar(np.arange(4) + (i - (len(rows) - 1) / 2) * w, v, width=w - 0.02, color=col, label=FL.label(label) + rf"; mean $\chi^2_\nu$ {np.mean(v):.2f}")
        for x, y in zip(np.arange(4) + (i - (len(rows) - 1) / 2) * w, v):
            b.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=PS.FS - 3)
    b.set_xticks(range(4), [PS.SUITE_TEX[j] for j in C.SUITES]); b.set_yscale("log"); b.minorticks_off()
    b.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"]); b.set_ylim(0.6, 80)
    b.set_ylabel(r"$\chi^2_\nu$ of held-out suite ($C_\mathrm{fv}$)")
    h, l = b.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, frameon=False, fontsize=PS.FS - 1, bbox_to_anchor=(0.5, 0.25 / H), borderaxespad=0)
    fig.text(0.5, 0.005, FL.NOTATION, ha="center", va="bottom", fontsize=PS.FS - 2)
    fig.suptitle(r"Best forms at $z=0$, full grid: held-out residuals per suite and $\chi^2_\nu$", fontsize=PS.FS + 1)
    fig.tight_layout(rect=(0, (res_h + 0.1) / H, 1, 0.95))
    PS.save(fig, out / "fig_fit_compare")


def replot(out, labels):
    """Redraw from the saved <label>.npz and chi2.json (no refit), e.g. after a sigma_k change: the residual panels use C.SIGMA_K."""
    summary = json.loads((out / "chi2.json").read_text())
    rows = []
    for label in labels:
        z = np.load(out / f"{label}.npz")
        res = {j: {k.split("_", 1)[1]: z[k] for k in z.files if k.split("_", 1)[0] == j} for j in C.SUITES}
        chi, n = summary[label]["chi2_per_suite"], summary[label]["nodes"]
        title = FL.label(label) + "\n" + rf"held-out fits at $z=0$ ({n} nodes per nonlinear coefficient); mean $\chi^2_\nu$ {np.mean(list(chi.values())):.2f}"
        fig_form(label, title, res, chi, out)
        rows.append((label, title, res, chi))
    fig_compare(rows, out)


def main(argv):
    if argv and argv[0] == "--replot":
        return replot(OUT, argv[1:])
    nodes = None
    if "--nodes" in argv:
        i = argv.index("--nodes"); nodes = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    out = OUT if nodes is None else OUT / f"nodes{nodes}"
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    for it in argv:
        label, path = it.split("=", 1)
        sub = json.loads((M / path).read_text())
        n = nodes or (C.NL_GRID_NODES if len(sub["nl"]) <= 2 else 21)
        form, res = fit(label, sub, n)
        chi = {j: float(np.mean(res[j]["chi2_box"])) for j in C.SUITES}
        np.savez(out / f"{label}.npz", **{f"{j}_{k}": v for j, d in res.items() for k, v in d.items()})
        title = FL.label(label) + "\n" + rf"held-out fits at $z=0$ ({n} nodes per nonlinear coefficient); mean $\chi^2_\nu$ {np.mean(list(chi.values())):.2f}"
        fig_form(label, title, res, chi, out)
        rows.append((label, title, res, chi)); summary[label] = {"nodes": n, "chi2_per_suite": chi, "chi2_mean": float(np.mean(list(chi.values())))}
        print(label, n, {j: round(v, 2) for j, v in chi.items()}, flush=True)
    (out / "chi2.json").write_text(json.dumps(summary, indent=1))
    fig_compare(rows, out)


if __name__ == "__main__":
    main(sys.argv[1:])

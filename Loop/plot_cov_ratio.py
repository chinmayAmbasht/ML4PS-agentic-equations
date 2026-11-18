"""Diagnostic: how big is the model-discrepancy covariance C_disc next to the finite-volume covariance C_fv? Plotting only.

For each held-out suite j: C_fv is suite j's own CV covariance (b.cfv), C_disc is the fold's diagonal discrepancy, moment-matched
from the 3 training suites' Stage-1 residuals of the given form (Scripts.engine.cov.c_disc). C_disc is form dependent, C_fv is not.

Figures in Figures/diagnostics/:
  fig_cov_ratio          matshow of (C_fv - C_disc) / C_fv, element-wise, rows = forms, columns = held-out suite.
                         C_disc is diagonal, so every off-diagonal element is exactly 1; the diagonal is 1 - C_disc/C_fv.
  fig_cov_diff           matshow of C_fv - C_disc, not normalized (symmetric log)
  fig_cov_ratio_diag     C_disc/C_fv on the diagonal vs k (log), plus sqrt(diag C_fv) and sqrt(diag C_disc) against sigma_k
plus cov_ratio.json.

Usage:  python -m Loop.plot_cov_ratio label=path.json ...    (or --replot: redraw from cov_ratio.json)
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
from Loop.plot_fits import form_of
from Scripts.common import config as C
from Scripts.engine import chi2 as X2
from Scripts.engine import scoring as SC
from Scripts.figures import panelspec as PS
from Scripts.run.score_forms import build_blocks

K = np.asarray(C.K_GRID)
OUT = M / "Figures" / "diagnostics"


def covs(label, sub):
    n = C.NL_GRID_NODES if len(sub["nl"]) <= 2 else 21
    form = form_of(label, sub, n)
    blocks, _, _ = build_blocks(form, C.REDSHIFTS)
    fp = X2.fold_params(SC.cross_suite_folds(form, blocks, C.REDSHIFTS))
    return {j: (np.asarray(blocks[j].cfv), np.diag(fp[j]["cdisc_diag"])) for j in C.SUITES}


def caption(fig, rows, h):
    """Equations of the forms in the rows plus the shorthand, in the bottom h inches of the figure."""
    H = fig.get_figheight()
    lines = [FL.label(label) for label, _ in rows] + [FL.NOTATION]
    for i, t in enumerate(reversed(lines)):
        fig.text(0.5, (0.08 + 0.36 * i) / H, t, ha="center", va="bottom", fontsize=PS.FS - (2 if i == 0 else 1))


def cap_h(rows):
    return 0.36 * (len(rows) + 1) + 0.15


def fig_matrix(rows, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    PS.style()
    vals = [1 - np.diag(cd) / np.diag(cf) for _, cv in rows for cf, cd in cv.values()]
    lo = min(float(np.min(v)) for v in vals)
    # signed log scale: |values| span 1e-2 .. 6e2; linear within +-0.1 of 0 (C_disc ~ C_fv)
    norm = SymLogNorm(linthresh=0.1, vmin=-10 ** np.ceil(np.log10(max(-lo, 1.0))), vmax=1.0, base=10)
    cmap = plt.get_cmap("RdBu").copy(); cmap.set_bad("0.88")
    H = 3.0 * len(rows) + 0.8 + cap_h(rows)
    fig, ax = plt.subplots(len(rows), 4, figsize=(12.5, H), squeeze=False, layout="constrained")
    fig.get_layout_engine().set(rect=(0, cap_h(rows) / H, 1, 1 - cap_h(rows) / H))
    caption(fig, rows, cap_h(rows))
    tick = [0, 4, 8, 12, 15]
    off = ~np.eye(len(K), dtype=bool)
    for r, (label, cv) in enumerate(rows):
        for c, j in enumerate(C.SUITES):
            cf, cd = cv[j]; R = np.diag(1 - np.diag(cd) / np.diag(cf)); a = ax[r][c]
            im = a.matshow(np.where(off, np.nan, R), cmap=cmap, norm=norm)   # off-diagonal is exactly 1: shown grey
            a.set_xticks(tick, [f"{K[i]:.2g}" for i in tick], fontsize=PS.FS - 3)
            a.set_yticks(tick, [f"{K[i]:.2g}" for i in tick], fontsize=PS.FS - 3)
            a.xaxis.set_ticks_position("bottom")
            if r == 0:
                a.set_title(f"{PS.SUITE_TEX[j]} held out", fontsize=PS.FS)
            if c == 0:
                a.set_ylabel(FL.rank_tag(label).replace(", ", "\n"), fontsize=PS.FS - 1)
            if r == len(rows) - 1:
                a.set_xlabel(r"$k$ [$h$/Mpc]", fontsize=PS.FS - 1)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label(r"$(C_\mathrm{fv}-C_\mathrm{disc})/C_\mathrm{fv}$ (symmetric log)")
    fig.suptitle(r"$(C_\mathrm{fv}-C_\mathrm{disc})/C_\mathrm{fv}$ per held-out suite at $z=0$. Blue: $C_\mathrm{disc}<C_\mathrm{fv}$; "
                 r"white: equal; red: $C_\mathrm{disc}>C_\mathrm{fv}$. $C_\mathrm{disc}$ is diagonal, so off-diagonal elements are exactly 1 (grey)",
                 fontsize=PS.FS)
    PS.save(fig, out / "fig_cov_ratio")


def fig_diag(rows, out):
    import matplotlib.pyplot as plt
    PS.style()
    H = 5.8 + cap_h(rows)
    fig, ax = plt.subplots(2, 4, figsize=(13, H), sharex=True)
    caption(fig, rows, cap_h(rows))
    ls = ["-", "--", ":", "-."]
    sk = np.asarray(C.SIGMA_K_PAPER)            # the paper's sigma_k: this figure motivated the inflated one
    ski = np.asarray(C.SIGMA_K)
    for c, j in enumerate(C.SUITES):
        a, b = ax[0][c], ax[1][c]
        cf = np.diag(rows[0][1][j][0])
        b.plot(K, np.sqrt(cf) / sk, "k-", lw=1.4, label=r"$\sqrt{C_\mathrm{fv}}$")
        if C.SIGMA_K_SOURCE == "inflated":
            b.plot(K, ski / sk, color="0.45", ls=(0, (4, 2)), lw=1.1, label=r"inflated $\sigma_k$")
        for (label, cv), s in zip(rows, ls):
            cf, cd = np.diag(cv[j][0]), np.diag(cv[j][1])
            a.plot(K, cd / cf, ls=s, color=PS.SUITE_COLOR[j], lw=1.3, marker="o", ms=2.5, label=FL.rank_tag(label))
            b.plot(K, np.sqrt(cd) / sk, ls=s, color=PS.SUITE_COLOR[j], lw=1.2, label=rf"$\sqrt{{C_\mathrm{{disc}}}}$ {FL.rank_tag(label)}")
        a.axhline(1, color="0.6", lw=0.6); a.set_yscale("log"); a.set_xscale("log")
        a.set_title(f"{PS.SUITE_TEX[j]} held out", fontsize=PS.FS)
        b.axhline(1, color="0.6", lw=0.6); b.set_yscale("log"); b.set_xlabel(PS.K_LABEL, fontsize=PS.FS - 1)
        if c == 0:
            a.set_ylabel(r"$C_\mathrm{disc}/C_\mathrm{fv}$ (diagonal)", fontsize=PS.FS - 1)
            b.set_ylabel(r"error $/\sigma_k^\mathrm{paper}$", fontsize=PS.FS - 1)
            a.legend(frameon=False, fontsize=PS.FS - 2); b.legend(frameon=False, fontsize=PS.FS - 3)
    fig.suptitle(r"Diagonal of $C_\mathrm{disc}$ vs $C_\mathrm{fv}$ at $z=0$ (top), and both errors against the paper's $\sigma_k$ (bottom)",
                 fontsize=PS.FS + 1)
    fig.tight_layout(rect=(0, cap_h(rows) / H, 1, 1 - 0.3 / H))
    PS.save(fig, out / "fig_cov_ratio_diag")


def fig_diff(rows, out):
    """C_fv - C_disc, not normalized (units of S^2). Off-diagonal = C_fv's own off-diagonal (C_disc is diagonal)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    PS.style()
    D = [cf - cd for _, cv in rows for cf, cd in cv.values()]
    top = 10 ** np.ceil(np.log10(max(float(np.max(np.abs(d))) for d in D)))
    norm = SymLogNorm(linthresh=1e-6, vmin=-top, vmax=top, base=10)
    H = 3.0 * len(rows) + 0.8 + cap_h(rows)
    fig, ax = plt.subplots(len(rows), 4, figsize=(12.5, H), squeeze=False, layout="constrained")
    fig.get_layout_engine().set(rect=(0, cap_h(rows) / H, 1, 1 - cap_h(rows) / H))
    caption(fig, rows, cap_h(rows))
    tick = [0, 4, 8, 12, 15]
    for r, (label, cv) in enumerate(rows):
        for c, j in enumerate(C.SUITES):
            cf, cd = cv[j]; a = ax[r][c]
            im = a.matshow(cf - cd, cmap="RdBu", norm=norm)
            a.set_xticks(tick, [f"{K[i]:.2g}" for i in tick], fontsize=PS.FS - 3)
            a.set_yticks(tick, [f"{K[i]:.2g}" for i in tick], fontsize=PS.FS - 3)
            a.xaxis.set_ticks_position("bottom")
            if r == 0:
                a.set_title(f"{PS.SUITE_TEX[j]} held out", fontsize=PS.FS)
            if c == 0:
                a.set_ylabel(FL.rank_tag(label).replace(", ", "\n"), fontsize=PS.FS - 1)
            if r == len(rows) - 1:
                a.set_xlabel(r"$k$ [$h$/Mpc]", fontsize=PS.FS - 1)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label(r"$C_\mathrm{fv}-C_\mathrm{disc}$ (symmetric log; linear for $|x|<10^{-6}$)")
    fig.suptitle(r"$C_\mathrm{fv}-C_\mathrm{disc}$ per held-out suite at $z=0$, not normalized. Blue: $C_\mathrm{fv}$ larger; red: "
                 r"$C_\mathrm{disc}$ larger. Off-diagonal is $C_\mathrm{fv}$ alone ($C_\mathrm{disc}$ is diagonal)", fontsize=PS.FS)
    PS.save(fig, out / "fig_cov_diff")


def replot():
    """Redraw from cov_ratio.json without refitting. C_disc is diagonal (stored); C_fv is form independent, so the full matrix is
    rebuilt from the data blocks of any form (paper rank 1)."""
    d = json.loads((OUT / "cov_ratio.json").read_text())
    blocks, _, _ = build_blocks(form_of("cfv", json.loads((M / "Base/sub/R01_two_scale.json").read_text()), 11), C.REDSHIFTS)
    rows = [(label, {j: (np.asarray(blocks[j].cfv), np.diag(np.square(v["sqrt_cdisc_diag"]))) for j, v in f.items()})
            for label, f in d["forms"].items()]
    for label, cv in rows:
        for j, (cf, _) in cv.items():
            assert np.allclose(np.sqrt(np.diag(cf)), d["forms"][label][j]["sqrt_cfv_diag"]), "C_fv rebuild mismatch"
    fig_matrix(rows, OUT)
    fig_diag(rows, OUT)
    fig_diff(rows, OUT)


def main(argv):
    if argv == ["--replot"]:
        return replot()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    for it in argv:
        label, path = it.split("=", 1)
        cv = covs(label, json.loads((M / path).read_text()))
        rows.append((label, cv))
        summary[label] = {j: {"cdisc_over_cfv_diag": (np.diag(cd) / np.diag(cf)).tolist(),
                              "sqrt_cfv_diag": np.sqrt(np.diag(cf)).tolist(), "sqrt_cdisc_diag": np.sqrt(np.diag(cd)).tolist()}
                          for j, (cf, cd) in cv.items()}
        print(label, {j: [round(float(x), 2) for x in np.diag(cd) / np.diag(cf)] for j, (cf, cd) in cv.items()}, flush=True)
    (OUT / "cov_ratio.json").write_text(json.dumps({"k": K.tolist(), "forms": summary}, indent=1))
    fig_matrix(rows, OUT)
    fig_diag(rows, OUT)
    fig_diff(rows, OUT)


if __name__ == "__main__":
    main(sys.argv[1:])

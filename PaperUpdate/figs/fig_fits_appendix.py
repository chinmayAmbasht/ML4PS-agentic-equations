"""Appendix figure (v4): held-out (leave-one-suite-out) fits, as a grid: rows = equations, columns = held-out suite
(owner 2026-11-13: one panel per equation x suite instead of overlays; 2026-11-19: common y-axis, square panels). From PaperUpdate/preds/<key>.npz.
Each panel: 3 held-out boxes of that suite (10th / 50th / 90th percentile of S at k ~ 10), data +- sqrt(diag C_fv),
adapted per-box MAP (solid; Fig. 3 setting: prior x1.00, likelihood C_fv/h, from PaperUpdate/preds_sens/<key>__fvh.npz) and zero-shot at the Stage-2 fold prior mean (dashed); the panel's correlated L (L_corr.json,
own-suite C_fv/h) over all held-out boxes of that suite is printed in the corner.
Usage: python -m PaperUpdate.figs.fig_fits_appendix T47 T36 T43 T39 T51
"""
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(M))

import numpy as np

from Scripts.common import config as C
from Scripts.figures import panelspec as PS

OUT = M / "PaperUpdate" / "figs" / "out"
PCT = (10, 50, 90)


def reps(y):
    ok = np.where(np.isfinite(y).all(axis=1))[0]
    v = y[ok, -1]
    return [int(ok[np.argmin(np.abs(v - np.percentile(v, q)))]) for q in PCT]


def main(keys):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    PS.style()
    H = json.loads((M / "PaperUpdate" / "sigma_k_hartlap.json").read_text()); h = H["hartlap_h"]
    Lc = json.loads((M / "PaperUpdate" / "L_corr.json").read_text())
    SN = json.loads((M / "PaperUpdate" / "map_sensitivity.json").read_text())   # Fig. 3 MAP setting (fvh)
    R = {r["key"]: r for r in json.loads((M / "PaperUpdate" / "ranking.json").read_text())}
    K = np.asarray(C.K_GRID)
    nr, nc = len(keys), len(C.SUITES)
    fig, ax = plt.subplots(nr, nc, figsize=(6.6, 1.62 * nr + 0.75), sharex=True, sharey=True, squeeze=False)
    for r, key in enumerate(keys):
        z = np.load(M / "PaperUpdate" / "preds" / f"{key}.npz")
        zm = np.load(M / "PaperUpdate" / "preds_sens" / f"{key}__fvh.npz")
        for c, s in enumerate(C.SUITES):
            a = ax[r, c]; y = z[f"{s}__y"]
            e = np.asarray(H["per_suite"][s]) * np.sqrt(h)
            for b, col in zip(reps(y), PS.BOX_SHADES):
                a.errorbar(K, y[b], yerr=e, fmt="o", ms=1.8, lw=0.6, color=col, zorder=3)
                a.plot(K, zm[f"{s}__map"][b], color=col, lw=1.1, zorder=4)
                a.plot(K, z[f"{s}__zs"][b], color=col, lw=0.9, ls="--", zorder=2)
            a.text(0.04, 0.05, rf"$L_\mathrm{{MAP}}={SN[key]['fvh']['per_suite'][s]:.0f}$" "\n"
                   rf"$L_0={Lc[key]['L_zero_per_suite'][s]:.0f}$", transform=a.transAxes, fontsize=PS.FS - 2,
                   color="0.25", va="bottom")
            a.set_xscale("log"); a.axhline(0, color="0.6", lw=0.4, ls=":"); a.set_box_aspect(1)
            a.set_xticks([0.5, 1, 2, 5, 10]); a.set_xticklabels(["0.5", "1", "2", "5", "10"]); a.minorticks_off()
            if r == 0:
                a.set_title(PS.SUITE_TEX[s], fontsize=PS.FS)
            if r == nr - 1:
                a.set_xlabel(PS.K_LABEL)
        ax[r, 0].set_ylim(-0.56, 0.13); ax[r, 0].set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
        ax[r, 0].set_ylabel(f"{key} (rank {R[key]['rank']})\n" + r"$\mathcal{S}(k)$", fontsize=PS.FS - 1)
    hs = [Line2D([], [], color=col, lw=1.1, marker="o", ms=2.5) for col in PS.BOX_SHADES] + \
         [Line2D([], [], color="0.3", lw=1.1), Line2D([], [], color="0.3", lw=0.9, ls="--")]
    ls = [f"held-out box, {q}th pct of $\\mathcal{{S}}(k\\simeq10)$" for q in PCT] + ["adapted (per-box MAP)", "zero-shot (prior mean)"]
    fig.subplots_adjust(left=0.1, right=0.99, top=0.965, bottom=0.115, wspace=0.06, hspace=0.06)
    fig.legend(hs, ls, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.005), fontsize=PS.FS - 1,
               columnspacing=1.5, handlelength=1.8)
    PS.save(fig, OUT / "fit_overlay_v4")


if __name__ == "__main__":
    main(sys.argv[1:])

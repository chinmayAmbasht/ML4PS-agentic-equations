"""Appendix figure (owner 2026-11-14): the finite-volume covariance C_fv of S per suite (top row) and the Hartlap-inflated
C_fv / h used in the correlated benchmark loss L (bottom row), as matshows on one shared symmetric-log colour scale.
C_fv is the engine's matrix (empirical 16x16 over the 27 CV boxes, shrinkage 0.10 toward the diagonal; PaperUpdate/Lcorr.py).
Usage: python -m PaperUpdate.figs.fig_cfv_appendix
"""
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(M))

import numpy as np

from PaperUpdate import Lcorr as LC
from Scripts.common import config as C
from Scripts.figures import panelspec as PS

OUT = M / "PaperUpdate" / "figs" / "out"


def main():
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    PS.style()
    K = np.asarray(C.K_GRID)
    mats = {s: LC.cfv(s) for s in C.SUITES}
    vmax = max(np.abs(m / LC.H).max() for m in mats.values())
    norm = SymLogNorm(linthresh=1e-6, vmin=-vmax, vmax=vmax, base=10)
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.45), layout="constrained")
    tick = [0, 4, 8, 12, 15]
    for c, s in enumerate(C.SUITES):
        for r, (Mx, lab) in enumerate(((mats[s], r"$C_\mathrm{fv}$"), (mats[s] / LC.H, r"$C_\mathrm{fv}/h$"))):
            a = ax[r, c]
            im = a.matshow(Mx, cmap="RdBu_r", norm=norm)
            a.set_xticks(tick, [f"{K[i]:.2g}" for i in tick], fontsize=PS.FS - 3)
            a.set_yticks(tick, [f"{K[i]:.2g}" for i in tick], fontsize=PS.FS - 3)
            a.xaxis.set_ticks_position("bottom")
            if r == 0:
                a.set_title(PS.SUITE_TEX[s], fontsize=PS.FS)
            if c == 0:
                a.set_ylabel(lab + "\n" + r"$k\ [h\,\mathrm{Mpc}^{-1}]$", fontsize=PS.FS - 1)
            if r == 1:
                a.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$", fontsize=PS.FS - 1)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"covariance of $\mathcal{S}$", fontsize=PS.FS - 1)
    fig.suptitle(rf"$h=(27-16-2)/26={LC.H:.3f}$", fontsize=PS.FS - 1, y=1.04)
    PS.save(fig, OUT / "cfv_matrices_v4")
    for s, m in mats.items():
        d = np.sqrt(np.diag(m)); R = m / np.outer(d, d)
        print(s, "diag range %.1e-%.1e" % (np.diag(m).min(), np.diag(m).max()), "min corr %.2f" % R.min())


if __name__ == "__main__":
    main()

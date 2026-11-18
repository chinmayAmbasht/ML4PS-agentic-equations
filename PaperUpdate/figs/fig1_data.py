"""Paper Fig. 1 at z = 0, k <= 9.92 (Plans/03). Data only, no model.
 (a) pk_suppression_cv: S(k) of the 27 CV boxes per suite (median line, 16-84% band; native bins <= 9.92, markers at the 16
     analysis bins).
 (b) fb_contours_k1p1: S at k ~ 1.1 against f_b/(Omega_b/Omega_m) for the LH boxes (D_valid), 1 sigma covariance ellipse per
     suite, points faint.
Usage: python -m PaperUpdate.figs.fig1_data
"""
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(M))

import h5py
import numpy as np

from Scripts.common import config as C
from Scripts.figures import panelspec as PS

OUT = M / "PaperUpdate" / "figs" / "out"
DATA = M / "Artifacts" / "data"
K16 = np.asarray(C.K_GRID)


def fig_a():
    import matplotlib.pyplot as plt
    PS.style()
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    with h5py.File(DATA / "pk_suppression_z000_allk.h5") as h:
        k = h["k_hmpc"][:]
        m = (k <= C.KMAX + 1e-9)
        i16 = [int(np.argmin(abs(k - kk))) for kk in K16]
        for s in C.SUITES:
            d = h[f"{s}/CV/delta"][:]
            med, lo, hi = (np.percentile(d, q, axis=0) for q in (50, 16, 84))
            c = PS.SUITE_COLOR[s]
            ax.fill_between(k[m], lo[m], hi[m], color=c, alpha=0.22, lw=0)
            ax.plot(k[m], med[m], color=c, lw=1.2, label=PS.SUITE_TEX[s])
            ax.plot(k[i16], med[i16], "o", ms=2.4, color=c)
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_xlim(k[m][0] * 0.95, C.KMAX * 1.05)
    ax.set_xlabel(PS.K_LABEL); ax.set_ylabel(PS.S_LABEL)
    ax.legend(frameon=False, loc="lower left")
    PS.save(fig, OUT / "pk_suppression_cv")


def ellipse(ax, x, y, c):
    from matplotlib.patches import Ellipse
    cov = np.cov(x, y); w, v = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    e = Ellipse((x.mean(), y.mean()), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang, fc=c, ec=c, alpha=0.35, lw=1.2)
    ax.add_patch(e)
    ax.add_patch(Ellipse((x.mean(), y.mean()), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang, fc="none", ec=c, lw=1.2))


def fig_b(k_t=1.1):
    import matplotlib.pyplot as plt
    PS.style()
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    with h5py.File(DATA / "pk_suppression_z000_allk.h5") as h, h5py.File(DATA / "observables_z000.h5") as o:
        k = h["k_hmpc"][:]; i = int(np.argmin(abs(k - k_t)))
        for s in C.SUITES:
            ok = o[f"{s}/LH/D_valid"][:]
            q = (o[f"{s}/LH/f_b"][:] / (o[f"{s}/LH/Omega_b"][:] / o[f"{s}/LH/Omega_m"][:]))[ok]
            y = h[f"{s}/LH/delta"][:, i][ok]
            c = PS.SUITE_COLOR[s]
            ax.scatter(q, y, s=1.5, color=c, alpha=0.12, lw=0, rasterized=True)
            ellipse(ax, q, y, c)
            ax.plot([], [], color=c, lw=4, alpha=0.6, label=PS.SUITE_TEX[s])
    ax.set_xlabel(r"$f_b/(\Omega_b/\Omega_m)$"); ax.set_ylabel(rf"$\mathcal{{S}}(k={k[i]:.2f}\,h\,\mathrm{{Mpc}}^{{-1}})$")
    ax.set_ylim(-0.42, 0.06)
    ax.legend(frameon=False, loc="lower right")
    PS.save(fig, OUT / "fb_contours_k1p1")
    return float(k[i])


if __name__ == "__main__":
    fig_a(); print("k used for (b):", fig_b())

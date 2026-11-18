"""Paper Fig. 1 at z = 0, k <= 9.92, in the style of the original Neurips/Figures panels (owner 2026-11-14: "make it in the old
style again"). Colours sampled from the original PDFs.
 (a) pk_suppression_cv: S(k) of the 27 CV boxes per suite, median line + 16-84% band, all native bins with k <= 9.92.
 (b) fb_contours_k1p1: 1 sigma (68%) KDE credible regions of S(k=1.11) against f_b/(Omega_b/Omega_m) of halos with
     5e12 < M < 1e14 Msun/h (the f_b of D = Omega_b/Omega_m - f_b; owner 2026-11-14), LH boxes with 0.2 < Omega_m < 0.4, as in the original panel.
Usage: python -m PaperUpdate.figs.fig1_oldstyle [kmax]
"""
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(M))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from Scripts.common import config as C

OUT = M / "PaperUpdate" / "figs" / "out"
DATA = M / "Artifacts" / "data"
COL = {"IllustrisTNG": "#2a78d6", "SIMBA": "#eb6834", "Astrid": "#1baf7a", "Swift-EAGLE": "#4a3aa7"}


def style():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 15, "axes.labelsize": 17,
                         "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.linewidth": 1.1,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "xtick.minor.visible": True, "ytick.minor.visible": True,
                         "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 3, "ytick.minor.size": 3,
                         "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": True, "legend.framealpha": 1.0,
                         "text.latex.preamble": r"\usepackage{amsmath}"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_a(kmax):
    style()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    with h5py.File(DATA / "pk_suppression_z000_allk.h5") as h:
        k = h["k_hmpc"][:]; m = k <= kmax + 1e-9
        for s in C.SUITES:
            d = h[f"{s}/CV/delta"][:]
            lo, med, hi = (np.percentile(d, q, axis=0) for q in (16, 50, 84))
            ax.fill_between(k[m], lo[m], hi[m], color=COL[s], alpha=0.2, lw=0)
            ax.plot(k[m], med[m], color=COL[s], lw=2.2, label=s)
    ax.axhline(0, color="0.5", lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_xlim(k[m][0], 10.05)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$"); ax.set_ylabel(r"$\mathcal{S}(k) = P_\mathrm{hydro}/P_\mathrm{dmo} - 1$")
    ax.legend(loc="lower left")
    save(fig, "pk_suppression_cv")


def hpd_level(z, frac=0.68):
    zs = np.sort(z.ravel())[::-1]; c = np.cumsum(zs); c /= c[-1]
    return zs[np.searchsorted(c, frac)]


def fig_b(k_t=1.1):
    style()
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    import os
    os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns_fig1"))
    from Scripts.data import suppression_io as IO
    i = int(np.argmin(abs(np.asarray(C.K_GRID) - k_t))); k = np.asarray(C.K_GRID)
    X, Y = np.meshgrid(np.linspace(0.0, 1.1, 300), np.linspace(-0.4, 0.1, 300))
    for s in C.SUITES:
        d = IO.load(s, "LH")       # the analysis loader: the boxes, D and f_b every fit uses (D = Omega_b/Omega_m - f_b)
        om = np.asarray(d.obs["Omega_m"], float); w = np.asarray(d.obs["Omega_b_over_Omega_m"], float)
        D = np.asarray(d.obs["D"], float).reshape(len(om), -1)[:, 0]
        x = (w - D) / w; y = d.y[:, i]
        ok = np.isfinite(x) & np.isfinite(y) & d.mask[:, i] & (om > 0.2) & (om < 0.4)
        kde = gaussian_kde(np.vstack([x[ok], y[ok]]))
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        lev = hpd_level(Z)
        ax.contourf(X, Y, Z, levels=[lev, Z.max() * 1.01], colors=[COL[s]], alpha=0.22)
        ax.contour(X, Y, Z, levels=[lev], colors=[COL[s]], linewidths=2.2)
        print(s, "boxes", int(ok.sum()), "of", len(om))
    ax.axhline(0, color="0.4", lw=0.8, ls=":")
    ax.set_xlim(0.12, 1.0); ax.set_ylim(-0.31, 0.075)
    ax.set_xlabel(r"$f_b/(\Omega_b/\Omega_m)$"); ax.set_title(rf"$k={k[i]:.1f}\ h/\mathrm{{Mpc}}$", fontsize=17)
    ax.text(0.03, 0.975, r"$5\times10^{12} < M\ (M_\odot/h) < 10^{14}$" "\n" r"$0.2 < \Omega_m < 0.4$", transform=ax.transAxes,
            color="0.45", fontsize=13, va="top")
    hs = [Patch(fc=COL[s], ec=COL[s], alpha=0.25, lw=1.5) for s in C.SUITES] + [Line2D([], [], color="0.25", lw=1.8)]
    ax.legend(hs, list(C.SUITES) + [r"$1\sigma$ credible region"], loc="lower right")
    save(fig, "fb_contours_k1p1")


if __name__ == "__main__":
    fig_a(float(sys.argv[1]) if len(sys.argv) > 1 else C.KMAX); fig_b()

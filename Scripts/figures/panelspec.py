"""Shared figure style (port of v1 Scripts/figures/panelspec.py, style only).

Every figure: usetex, \\mathcal{S} for P_hydro/P_dmo - 1, PDF + native 200-dpi PNG, a README beside it.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 9
S_LABEL = r"$\mathcal{S}(k) = P_\mathrm{hydro}/P_\mathrm{dmo} - 1$"
K_LABEL = r"$k\ [h\,\mathrm{Mpc}^{-1}]$"

# Okabe-Ito, CVD-validated in v1 (Plans/05 there)
SUITE_COLOR = {"IllustrisTNG": "#0072B2", "SIMBA": "#D55E00", "Astrid": "#009E73", "Swift-EAGLE": "#CC79A7"}
SUITE_TEX = {"IllustrisTNG": r"IllustrisTNG", "SIMBA": r"SIMBA", "Astrid": r"Astrid", "Swift-EAGLE": r"Swift-EAGLE"}
MODEL_PALETTE = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#000000"]
BOX_SHADES = ["#56B4E9", "#0072B2", "#08306B"]          # three representative boxes, light to dark


def style():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": FS, "axes.labelsize": FS,
                         "legend.fontsize": FS - 1, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
                         "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"})


def save(fig, path):
    """path without suffix -> .pdf and a native 200-dpi .png."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

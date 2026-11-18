"""Paper Fig. 3 (v4): benchmark loss L (Hartlap sigma_k) from PaperUpdate/Benchmark/bench.json. Plans/03.
Left: mean over the four suites for within-suite / pooled / transfer zero-shot / transfer adapted (per-box MAP).
Right: transfer per held-out suite. Owner 2026-11-14: among the adapted models only the highest-evidence form (rank 1) also\nshows its zero-shot bar (hatched); baselines without adaptation (RF, published van Daalen) show their single transfer bar.
RF and published van Daalen receive no target data; van Daalen (refit) is run through the same Stage-2 + per-box MAP pipeline\nas the analytic forms (owner 2026-11-13), so it has zero-shot and adapted bars.
Usage: python -m PaperUpdate.figs.fig3_benchmark T47 T36 T43 T39 T51
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
# Wong (2011) colour-blind-safe palette (owner 2026-11-18). Wong has 8 colours, so the 9th series (published van Daalen)
# is encoded by fill instead of hue: a white bar with a black outline ("outline").
BASE = [("RF", "Random forest", "#000000", ""), ("vD20 refit", "van Daalen+20 (fit)", "#F0E442", ""),
        ("vD20 published", "van Daalen+20 (publ.)", "outline", "")]
FORM_COLORS = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00"]
PYSR = [("PySR (unconstrained)", "PySR", "#CC79A7", "")]     # owner 2026-11-14: template variant not shown



def solid(c):
    """Bar style for a solid (adapted / single-transfer) bar: Wong colour with a thin dark edge (keeps yellow visible on
    white), or a white bar with a black outline for the fill-encoded 9th series."""
    if c == "outline":
        return dict(facecolor="white", edgecolor="black", lw=0.8)
    return dict(color=c, edgecolor="0.15", lw=0.3)

def main(keys):
    import matplotlib.pyplot as plt
    PS.style()
    B = json.loads((M / "PaperUpdate" / "Benchmark" / "bench_corr.json").read_text())
    R = {r["key"]: r for r in json.loads((M / "PaperUpdate" / "ranking.json").read_text())}
    models = [(k, n, c, h) for k, n, c, h in BASE] + [m for m in PYSR if m[0] in B] + [(k, f"Rank {R[k]['rank']}", FORM_COLORS[i], "") for i, k in enumerate(keys)]   # owner 2026-11-18: rank numbers, no trial ids
    groups = [("within", "Within-suite"), ("pooled", "Pooled"), ("zs", "Transfer\nno target data"), ("map", "Transfer\nMAP")]

    best = min(R.values(), key=lambda r: r["rank"])["key"]      # highest log Z_cross: the only adapted loop form shown zero-shot
    zshow = {best} | {m[0] for m in PYSR}                       # PySR: zero-shot = the LOSO search itself (3 suites), plus adapted

    def val(k, g, suite=None):
        b = B[k]
        if g == "zs" and "transfer_map" in b and k not in zshow:
            return np.nan
        key = {"within": "within", "pooled": "pooled",
               "zs": "transfer_zero_shot" if "transfer_zero_shot" in b else "transfer_population",
               "map": "transfer_map" if "transfer_map" in b else None}[g]
        if key is None or key not in b:
            return np.nan
        return b[key]["mean"] if suite is None else b[key]["per_suite"][suite]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw={"width_ratios": [1, 1.25]})
    nm = len(models); w = 0.8 / nm
    for i, (k, n, c, h) in enumerate(models):
        x = np.arange(len(groups)) + (i - (nm - 1) / 2) * w
        v = [val(k, g) for g, _ in groups]
        zsi = [gi for gi, (g, _) in enumerate(groups) if g == "zs"][0]
        if k in zshow and k in B and "transfer_map" in B[k]:       # its zero-shot bar in the hollow hatched style, the rest solid
            a1.bar(x[zsi], v[zsi], w * 0.92, facecolor="white", edgecolor=c, hatch="////", lw=0.8)
            v[zsi] = np.nan
        a1.bar(x, v, w * 0.92, label=n, **solid(c))
    a1.set_xticks(range(len(groups)), [t for _, t in groups])
    a1.set_yscale("log"); a1.set_ylabel(r"$L$ (correlated, $C_\mathrm{fv}/h$)")
    a1.grid(axis="y", which="major", color="0.88", lw=0.5); a1.set_axisbelow(True)

    S = list(C.SUITES)
    adapt = {k for k, *_ in models if "transfer_map" in B[k]}
    both = zshow & adapt                                    # zero-shot + adapted side by side only for the best-evidence form
    nbar = len(models) + len(both); w2 = 0.84 / nbar
    for si, s in enumerate(S):
        j = 0
        for k, n, c, h in models:
            if k in adapt:
                for g, hatch, alpha in ((("zs", "////", 0.45),) if k in both else ()) + (("map", "", 1.0),):
                    x = si + (j - (nbar - 1) / 2) * w2
                    if g == "zs":
                        a2.bar(x, val(k, g, s), w2 * 0.92, facecolor="white", edgecolor=c, hatch="////", lw=0.8)
                    else:
                        a2.bar(x, val(k, g, s), w2 * 0.92, **solid(c))
                    j += 1
            else:
                x = si + (j - (nbar - 1) / 2) * w2
                a2.bar(x, val(k, "zs", s), w2 * 0.92, **solid(c)); j += 1
    a2.set_xticks(range(len(S)), [PS.SUITE_TEX[s] for s in S]); a2.set_xlabel("held-out suite")
    a2.set_yscale("log"); a2.grid(axis="y", which="major", color="0.88", lw=0.5); a2.set_axisbelow(True)
    from matplotlib.patches import Patch
    hs, ls = a1.get_legend_handles_labels()
    hs += [Patch(fc="white", hatch="////", ec="0.35", lw=0.8), Patch(fc="0.45", ec="0.15", lw=0.3)]
    ls += ["zero-shot", "adapted (per-box MAP)"]   # owner 2026-11-18: compact legend (details in caption / Table 1)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fig.legend(hs, ls, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.02), fontsize=PS.FS - 1.5,
               columnspacing=0.9, handlelength=1.4, handletextpad=0.4, labelspacing=0.25)
    PS.save(fig, OUT / "pooled_comparison_v4")


if __name__ == "__main__":
    main(sys.argv[1:])

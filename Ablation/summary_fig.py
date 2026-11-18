"""One-figure summary of the ablation (Plans/05_ablation.md), replacing the md tables for reading.
Left: best log Z_cross/box per search with the T47 +-sigma_fold band and gap in sigma_fold. Right: L_MAP (fvh) of each search's best.
Reads Ablation/ablation.json. Output: Ablation/figs/ablation_summary.{pdf,png}.
Usage: python -m Ablation.summary_fig
"""
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M))
import matplotlib.pyplot as plt
from Scripts.figures import panelspec as PS

A = json.loads((M / "Ablation" / "ablation.json").read_text())
ref, sf = A["ref_T47_logZ_box"], A["sigma_fold_box"]
LAB = {"Agent": ("Agent loop (T47)", "#0072B2"), "A1-C": ("A1-C: PySR C, log Z pick", "#CC79A7"),
       "A3": ("A3: seed-grammar enumeration", "#009E73"), "Best seed": ("Seed form (paper rank 1)", "0.45"),
       "PySR C": ("PySR C, MSE pick (Fig. 3)", "#CC79A7"), "A2": ("A2: PySR two-kernel template", "#882255"),
       "A1-U": ("A1-U: PySR U, log Z pick", "#E69F00"), "PySR U": ("PySR U, MSE pick (Fig. 3)", "#E69F00")}
rows = []
for r in A["rows"]:
    k = next(k for k in LAB if r["arm"].startswith(k))
    rows.append((k, r))
rows.sort(key=lambda kr: kr[1]["logZ_box"])
PS.style()
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})
for i, (k, r) in enumerate(rows):
    lab, c = LAB[k]
    mk = "o" if not k.startswith("PySR") else "D"
    hollow = k.startswith("PySR")
    a.plot(r["logZ_box"], i, mk, ms=7, color=c, mfc="white" if hollow else c, mew=1.5, zorder=3)
    gap = "" if k == "Agent" else f"${(r['logZ_box'] - ref) / sf:+.1f}\\sigma$"
    a.text(r["logZ_box"] - 0.2, i, f"{r['logZ_box']:.2f}  {gap}" if gap else f"{r['logZ_box']:.2f}",
           ha="right", va="center", fontsize=PS.FS - 2, color="0.25")
    n = r.get("n_eval")
    if r.get("L_map_fvh") is not None:
        b.plot(r["L_map_fvh"], i, mk, ms=7, color=c, mfc="white" if hollow else c, mew=1.5, zorder=3)
        b.text(r["L_map_fvh"] + 2.2, i, f"{r['L_map_fvh']:.1f}", va="center", fontsize=PS.FS - 2, color="0.25")
a.axvspan(ref - sf, ref + sf, color="#0072B2", alpha=0.12, lw=0)
a.axvline(ref, color="#0072B2", lw=0.6, ls=":")
a.set_yticks(range(len(rows)))
a.set_yticklabels([f"{LAB[k][0]}  [{r['n_eval']}]" for k, r in rows], fontsize=PS.FS - 1)
a.set_xlim(28.3, 36.0)
a.set_xlabel("best $\\log Z_\\mathrm{cross}$ [nats/box]  (higher is better)")
b.set_xlim(10, 50); b.set_xlabel("$L_\\mathrm{MAP}$ (lower is better)")
for ax in (a, b):
    ax.grid(axis="x", color="0.9", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
a.set_title("same harness, constraints and features for every search; [ ] = full evaluations;\n"
            "$\\sigma$ = gap to T47 in units of $\\sigma_\\mathrm{fold}$ (band); hollow diamonds = existing MSE picks",
            fontsize=PS.FS - 2, loc="left", color="0.3")
fig.tight_layout()
out = M / "Ablation" / "figs"; out.mkdir(exist_ok=True)
PS.save(fig, out / "ablation_summary")
print(out / "ablation_summary.png")

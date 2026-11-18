"""ML4PS z = 0 loop figure. Plotting only.
(a) log Z_cross per box (/3726) per trial, coloured by lane, with the running best and the paper's base forms (Base/) as reference lines;
(b) log Z_cross vs mean chi^2_nu; (c) log Z_cross vs L_map (x1.30 widened prior); (d) per-suite chi^2_nu for the best base
form, the loop's best log Z_cross form and the loop's best chi^2 form.

Inputs: Loop/{trial_state.yaml, lanes.json}, Base/trial_state.yaml
Output: Figures/fig_ml4ps_loop.{pdf,png}
Usage:  python -m Loop.plot_loop
"""
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M))

import numpy as np
import yaml

from Loop import form_labels as FL
from Scripts.figures import panelspec as PS

LANE_COLOR = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
BASE_NAME = {"T05": "paper rank 1", "T02": "paper rank 2", "T01": "paper rank 3", "T03": "paper rank 12"}
N_BOX = 3726           # held-out boxes (TNG 943, SIMBA 918, Astrid 927, Swift 938): log Z_cross is quoted per box (owner 2026-11-11)


def load(d, lanes=None):
    out = []
    for t in yaml.safe_load((M / d / "trial_state.yaml").read_text())["trials"]:
        if t.get("logZ_cross") is None:
            continue
        out.append({"trial": t["trial"], "n": int(t["trial"][1:]), "Z": t["logZ_cross"], "sf": t["sigma_fold"],
                    "mean": t["chi2_mean"], "Lmap": t["L_map"], "per": t["chi2_per_suite"],
                    "lane": lanes[t["trial"]] if lanes else None})
    return sorted(out, key=lambda p: p["n"])


def main():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    lanes = json.loads((M / "Loop" / "lanes.json").read_text())
    pts = load("Loop", lanes); base = load("Base")
    for b in base:
        b["name"] = BASE_NAME[b["trial"]]
    R = []
    for p in pts:
        if not R or p["Z"] > R[-1]["Z"]:
            R.append(p)
    best = R[-1]; bbase = max(base, key=lambda b: b["Z"]); bchi = min(pts, key=lambda p: p["mean"])
    PS.style()
    cap = [FL.label(best["trial"]), FL.label(bchi["trial"]), FL.label(bbase["name"]), FL.NOTATION]
    H = 8.0 + 0.36 * len(cap) + 0.1
    fig, ax = plt.subplots(2, 2, figsize=(11.5, H))
    for i, t in enumerate(reversed(cap)):
        fig.text(0.5, (0.06 + 0.36 * i) / H, t, ha="center", va="bottom", fontsize=PS.FS - (3 if i == 0 else 1))
    a1, a2, a3, a4 = ax.ravel()
    s = N_BOX

    done = []
    for b in sorted(base, key=lambda b: -b["Z"]):
        a1.axhline(b["Z"] / s, color="0.55", lw=0.8, ls=":")
        near = [d for d in done if abs(d["Z"] - b["Z"]) < 0.08 * N_BOX]
        if near:
            near[0]["txt"].set_text(f"paper ranks {near[0]['name'].split()[-1]}, {b['name'].split()[-1]} (tied) [ranks {FL.RANK[FL.key_of(near[0]['name'])]}, {FL.RANK[FL.key_of(b['name'])]}]")
            continue
        b["txt"] = a1.text(50.8, b["Z"] / s, f"{b['name']} [rank {FL.RANK[FL.key_of(b['name'])]}]", ha="right", va="bottom", fontsize=PS.FS - 3, color="0.35")
        done.append(b)
    a1.axhspan((best["Z"] - best["sf"]) / s, best["Z"] / s, color="#F3E7C4", zorder=0, lw=0)
    for p in pts:
        a1.scatter(p["n"], p["Z"] / s, s=22, color=LANE_COLOR[p["lane"]], edgecolors="white", lw=0.4, zorder=3)
    xs = [p["n"] for p in R] + [pts[-1]["n"]]; ys = [p["Z"] / s for p in R] + [R[-1]["Z"] / s]
    a1.step(xs, ys, where="post", color="#B8860B", lw=1.4, zorder=2)
    a1.annotate(f"{FL.rank_tag(best['trial'])}: {best['Z']/s:.3f}", (best["n"], best["Z"] / s), xytext=(-8, 6), textcoords="offset points",
                fontsize=PS.FS - 2, color="#7A5A00", ha="right")
    ylo = 31.0; a1.set_ylim(ylo, best["Z"] / s + 0.4); a1.set_xlim(0, 51)
    a1.set_xlabel("trial number"); a1.set_ylabel(r"$\log Z_\mathrm{cross}/N_\mathrm{box}$ [nats/box]  $\uparrow$ better")
    a1.set_title(r"(a) $\log Z_\mathrm{cross}$ per trial; band: best $-\,\sigma_\mathrm{fold}$", fontsize=PS.FS)
    low = [p for p in pts if p["Z"] / s < ylo]
    for p in low:
        a1.annotate(f"rank {FL.RANK[p['trial']]} ({p['Z']/s:.2f}) below", (p["n"], ylo), xytext=(0, 8), textcoords="offset points",
                    fontsize=PS.FS - 3, ha="center", color=LANE_COLOR[p["lane"]], arrowprops=dict(arrowstyle="->", lw=0.6,
                    color=LANE_COLOR[p["lane"]]))

    for a, key, lab in [(a2, "mean", r"mean $\chi^2_\nu$ over held-out suites ($C_\mathrm{fv}$)  $\leftarrow$ better"),
                        (a3, "Lmap", r"$L_\mathrm{map}$ (prior $\times1.30$)  $\leftarrow$ better")]:
        for p in pts:
            a.scatter(p[key], p["Z"] / s, s=22, color=LANE_COLOR[p["lane"]], edgecolors="white", lw=0.4, zorder=3)
        for b in base:
            a.scatter(b[key], b["Z"] / s, s=40, marker="s", color="#BBBBBB", edgecolors="0.3", lw=0.6, zorder=4)
            a.annotate(FL.rank_tag(b["name"]), (b[key], b["Z"] / s), xytext=(5, 6 if b["name"] == "paper rank 3" else -3), textcoords="offset points",
                       fontsize=PS.FS - 3, color="0.35")
        for p, dy in [(best, -12), (bchi, 6)]:
            a.annotate(FL.rank_tag(p["trial"]), (p[key], p["Z"] / s), xytext=(5, dy), textcoords="offset points", fontsize=PS.FS - 2,
                       color="#7A5A00")
        a.set_xlabel(lab); a.set_ylabel(r"$\log Z_\mathrm{cross}/N_\mathrm{box}$ [nats/box]")
        a.set_ylim(min(min(q["Z"] for q in pts + base) / s - 0.2, ylo), best["Z"] / s + 0.4)
    a2.set_title(r"(b) evidence vs held-out $\chi^2_\nu$", fontsize=PS.FS)
    a3.set_title(r"(c) evidence vs adapted loss $L_\mathrm{map}$", fontsize=PS.FS)

    refs = [(f"best base form: {FL.rank_tag(bbase['name'])} ({bbase['Z']/s:.2f})", bbase["per"], "#BBBBBB"),
            (f"loop best evidence: {FL.rank_tag(best['trial'])} ({best['Z']/s:.2f})", best["per"], LANE_COLOR[best["lane"]]),
            (f"loop best $\\chi^2_\\nu$: {FL.rank_tag(bchi['trial'])} ({bchi['Z']/s:.2f})", bchi["per"], LANE_COLOR[bchi["lane"]])]
    suites = list(PS.SUITE_COLOR); w = 0.26
    for i, (lab, per, col) in enumerate(refs):
        a4.bar(np.arange(4) + (i - 1) * w, [per[x] for x in suites], width=w - 0.03, color=col, label=lab, zorder=3,
               hatch="//" if i == 2 else None, edgecolor="white" if i < 2 else "0.2", lw=0.4)
    a4.set_xticks(range(4), ["TNG", "SIMBA", "Astrid", "Swift"])
    a4.set_ylabel(r"$\chi^2_\nu$ of held-out suite"); a4.set_yscale("log"); a4.minorticks_off()
    a4.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"]); a4.set_ylim(0.7, 80)
    a4.legend(frameon=False, fontsize=PS.FS - 2, loc="upper left")
    a4.set_title("(d) per held-out suite", fontsize=PS.FS)

    h = [Line2D([], [], marker="o", ls="", color=LANE_COLOR[l], ms=5, label=f"lane {l}") for l in LANE_COLOR]
    h += [Line2D([], [], color="#B8860B", lw=1.4, label="running best"),
          Line2D([], [], marker="s", ls="", color="#BBBBBB", mec="0.3", ms=6, label="paper base forms (this harness)")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=PS.FS - 1, bbox_to_anchor=(0.5, (0.36 * len(cap) + 0.1) / H))
    fig.suptitle(rf"ML4PS $z=0$ main loop, $k\le9.92$, 16 bins (50 trials, $p\le6$): best form [{FL.rank_tag(best['trial'])}] "
                 rf"$\log Z_\mathrm{{cross}}/N_\mathrm{{box}}$ {best['Z']/s:.3f} ($+{(best['Z']-bbase['Z'])/s:.3f}$ vs best base form)",
                 fontsize=PS.FS + 1)
    fig.tight_layout(rect=(0, (0.36 * len(cap) + 0.45) / H, 1, 1 - 0.25 / H))
    PS.save(fig, M / "Figures" / "fig_ml4ps_loop")
    print("running best:", " -> ".join(f"{p['trial']} ({p['Z']/s:.3f})" for p in R))


if __name__ == "__main__":
    main()

"""Ablation summary (Plans/05_ablation.md): table + best-so-far figure. Reads Loop/, Base/, PySR/, Ablation/<arm>/harness only.
Outputs: Ablation/ablation.json, Ablation/RESULT_table.md, PaperUpdate/figs/out/ablation_v4.{pdf,png}, Ablation/ablation_table.tex.
Usage: python -m Ablation.analyze        (all)
       python -m Ablation.analyze fig    (figure only, from ablation.json)
"""
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M))

import numpy as np
import yaml

NBOX = 3726
ARMS = {"A1": "A1: PySR hall of fame, selected by $\\log Z$", "A2": "A2: PySR two-kernel template, selected by $\\log Z$",
        "A3": "A3: seed-grammar enumeration"}


def trials(p):
    p = Path(p)
    return (yaml.safe_load(p.read_text()) or {}).get("trials", []) if p.exists() else []


def scored(t):
    return t.get("logZ_cross") is not None and t.get("status") in ("scored", "presentable")


def arm_trials():
    """Arm name -> trial list in evaluation order. A1 is split into its U and C halves for the table."""
    out = {"agent": trials(M / "Loop" / "trial_state.yaml")}
    for a in ARMS:
        out[a] = trials(M / "Ablation" / a / "harness" / "trial_state.yaml")
    out["A1-U"] = [t for t in out["A1"] if "[ablation A1 U_" in t.get("hypothesis", "")]
    out["A1-C"] = [t for t in out["A1"] if "[ablation A1 C_" in t.get("hypothesis", "")]
    return out


def arm_best():
    best = {}
    for a, T in arm_trials().items():
        if a in ("agent", "A1"):
            continue
        S = [t for t in T if scored(t)]
        if S:
            best[a] = max(S, key=lambda t: t["logZ_cross"])
    return best


def curve(T):
    v = [t["logZ_cross"] / NBOX if scored(t) else -np.inf for t in T]
    return np.maximum.accumulate(v) if v else np.array([])


def main():
    A = arm_trials()
    rk = {r["key"]: r for r in json.loads((M / "PaperUpdate" / "ranking.json").read_text())}
    t47 = rk["T47"]; ref, sf = t47["logZ_box"], t47["sigma_fold_box"]
    pysr = json.loads((M / "PySR" / "summary.json").read_text())
    base = [t for t in trials(M / "Base" / "trial_state.yaml") if t.get("logZ_cross")]
    p1 = max(base, key=lambda t: t["logZ_cross"])
    wm = json.loads((M / "Ablation" / "winners_map.json").read_text()) if (M / "Ablation" / "winners_map.json").exists() else {}
    bench = json.loads((M / "PaperUpdate" / "Benchmark" / "bench_corr.json").read_text())

    def row(name, constraints, space, objective, T, lmap=None):
        S = [t for t in T if scored(t)]
        b = max(S, key=lambda t: t["logZ_cross"]) if S else None
        nfail = sum(t.get("status") == "gate0_fail" for t in T); nrej = sum(t.get("status") == "rejected" for t in T)
        lz = b["logZ_cross"] / NBOX if b else None
        return {"arm": name, "constraints": constraints, "space": space, "objective": objective, "n_eval": len(T),
                "n_scored": len(S), "n_gate0_fail": nfail, "n_rejected": nrej, "best_trial": b and b["trial"],
                "best_expr": b and b["expr"], "logZ_box": lz, "d_logZ_box": lz - ref if b else None,
                "d_sigma": (lz - ref) / sf if b else None, "chi2_nu": b and b.get("chi2_mean"),
                "marginalized": b and b.get("marginalized"), "p": b and b.get("n_coeff"), "L_map_fvh": lmap}

    R = [row("Agent loop (T01-T60)", "Gate 0, p<=6, 1-2 nl", "open (any sympy expr)", "log Z_cross feedback", A["agent"],
             bench["T47"]["transfer_map"]["mean"]),
         {**row("PySR U, MSE 'best' (Plan 04)", "none", "PySR U operators", "weighted MSE", []),
          "n_eval": 1, "n_scored": 1, "logZ_box": pysr["U"]["logZ_box"], "d_logZ_box": pysr["U"]["logZ_box"] - ref,
          "d_sigma": (pysr["U"]["logZ_box"] - ref) / sf, "chi2_nu": pysr["U"]["harness"]["chi2_mean"],
          "L_map_fvh": bench["PySR (unconstrained)"]["transfer_map"]["mean"], "note": "base mode (<=4 nl)"},
         {**row("PySR C, MSE 'best' (Plan 04)", "G0_1-3 by template", "W(k e^-f) g template", "weighted MSE", []),
          "n_eval": 1, "n_scored": 1, "logZ_box": pysr["C"]["logZ_box"], "d_logZ_box": pysr["C"]["logZ_box"] - ref,
          "d_sigma": (pysr["C"]["logZ_box"] - ref) / sf, "chi2_nu": pysr["C"]["harness"]["chi2_mean"],
          "L_map_fvh": wm.get("PySR C", {}).get("mean"), "note": "base mode (<=4 nl)"},
         row("A1-U: PySR U hall of fame, log Z selection", "none (Gate 0 at scoring)", "PySR U operators", "MSE search, log Z selection",
             A["A1-U"], wm.get("A1-U", {}).get("mean")),
         row("A1-C: PySR C hall of fame, log Z selection", "G0_1-3 by template", "W(k e^-f) g template", "MSE search, log Z selection",
             A["A1-C"], wm.get("A1-C", {}).get("mean")),
         row("A2: PySR two-kernel template, log Z selection", "G0_1-3 by template", "sum of two W2 kernels", "MSE search, log Z selection",
             A["A2"], wm.get("A2", {}).get("mean")),
         row("A3: seed-grammar enumeration (top 60 of prescreen)", "Gate 0, p<=6, 1-2 nl", "seed grammar (292 forms)",
             "log Z_cross (R={} prescreen, then full)", A["A3"], wm.get("A3", {}).get("mean")),
         {**row("Best seed form (paper rank 1, P1)", "Gate 0", "-", "-", []), "n_eval": 1, "n_scored": 1,
          "logZ_box": p1["logZ_cross"] / NBOX, "d_logZ_box": p1["logZ_cross"] / NBOX - ref, "d_sigma": (p1["logZ_cross"] / NBOX - ref) / sf,
          "chi2_nu": p1.get("chi2_mean"), "note": "3 nl (base mode)"}]
    (M / "Ablation" / "ablation.json").write_text(json.dumps({"ref_T47_logZ_box": ref, "sigma_fold_box": sf, "rows": R,
                                                               "curves": {a: curve(A[a]).tolist() for a in ("agent", "A1-U", "A1-C", "A2", "A3")}},
                                                              indent=1, default=float))
    f = lambda x, d=2: "–" if x is None else f"{x:.{d}f}"
    md = ["| arm | constraints | search space | objective | full evals | scored | best log Z/box | Δ vs T47 | Δ/σ_fold | χ²_ν | L_MAP |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['arm']} | {r['constraints']} | {r['space']} | {r['objective']} | {r['n_eval']} | {r['n_scored']} | {f(r['logZ_box'])} | "
           f"{f(r['d_logZ_box'])} | {f(r['d_sigma'], 1)} | {f(r['chi2_nu'])} | {f(r['L_map_fvh'], 1)} |" for r in R]
    (M / "Ablation" / "RESULT_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    tex(R)
    figure(A, ref, sf, p1["logZ_cross"] / NBOX, pysr)


def tex(R):
    f = lambda x, d=2: "--" if x is None else f"{x:.{d}f}"
    short = ["Agent loop", "Unconstrained symbolic regression, MSE selection", "Constrained symbolic regression, MSE selection",
             "Unconstrained symbolic regression, $\\log Z$ selection", "Constrained symbolic regression, $\\log Z$ selection",
             "A2 (not shown)", "Seed-grammar enumeration", "Best seed form"]
    L = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Search & Equations & $\log Z_\mathrm{cross}$/box & $\Delta$ vs rank 1 & $\chi^2_\nu$ & $L_\mathrm{MAP}$ \\", r"\midrule"]
    for s, r in zip(short, R):
        if r["arm"].startswith("A2"):  # A2 dropped from the paper table and figure (owner 2026-11-19); kept in ablation.json
            continue
        L.append(f"{s} & {r['n_eval']} & {f(r['logZ_box'])} & {f(r['d_logZ_box'])} & {f(r['chi2_nu'], 1)} & {f(r['L_map_fvh'], 1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    p = M / "Ablation" / "ablation_table.tex"
    p.write_text("\n".join(L) + "\n")
    (M / "Ablation" / "ablation_table.tex").write_text("\n".join(L) + "\n")


def figure(A=None, ref=None, sf=None, p1=None, pysr=None):
    """ML4PS figure: best log Z_cross so far vs full evaluations. A2 left out (owner 2026-11-19); it stays in the table.
    Reads the curves and reference lines from Ablation/ablation.json, so `python -m Ablation.analyze fig` redraws without
    touching the table."""
    import matplotlib.pyplot as plt
    from Scripts.figures import panelspec as PS
    J = json.loads((M / "Ablation" / "ablation.json").read_text())
    ref, sf = J["ref_T47_logZ_box"], J["sigma_fold_box"]
    rows = {r["arm"].split(":")[0].split(" (")[0]: r for r in J["rows"]}
    p1, mse_c = rows["Best seed form"]["logZ_box"], rows["PySR C, MSE 'best'"]["logZ_box"]
    PS.style()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5})
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    blue = "#0072B2"
    ax.axhspan(ref - sf, ref + sf, color=blue, alpha=0.13, lw=0, zorder=0)
    ax.text(1.8, ref + sf - 0.06, r"Best agent form $\pm\,\sigma_\mathrm{fold}$", color=blue, fontsize=8, va="top")
    ax.axhline(p1, color="0.45", lw=0.9, ls="-.", zorder=1)
    ax.text(60.5, p1 + 0.07, "Best seed form", color="0.35", fontsize=8, ha="right", va="bottom")
    ax.axhline(mse_c, color="#CC79A7", lw=0.9, ls=":", zorder=1)
    ax.text(60.5, mse_c - 0.08, "Constrained symbolic regression, MSE selection", color="#B0578F", fontsize=8, ha="right", va="top")
    sty = [("agent", "Agent loop", blue, "-", 2.0, (59.5, ref + 0.1, "right", "bottom")),
           ("A3", "Seed-grammar enumeration", "#009E73", "-", 1.5, (59.5, 34.23 - 0.1, "right", "top")),
           ("A1-C", "Constrained symbolic regression,\n" r"$\log Z$ selection", "#B0578F", "--", 1.5, (19.0, 34.33 + 0.12, "center", "bottom")),
           ("A1-U", r"Unconstrained symbolic regression, $\log Z$ selection", "#E69F00", "--", 1.5, (9.0, 30.19 + 0.12, "left", "bottom"))]
    for a, lab, c, ls, lw, (tx, ty, ha, va) in sty:
        y = np.array(J["curves"][a], float)
        x = np.arange(1, len(y) + 1); ok = np.isfinite(y)
        ax.step(x[ok], y[ok], where="post", color=c, ls=ls, lw=lw, zorder=3)
        ax.plot(x[ok][-1], y[ok][-1], "o", ms=3.5, color=c, zorder=4)
        ax.text(tx, ty, lab, color=c, fontsize=8.5, ha=ha, va=va, zorder=5, multialignment="center", linespacing=1.0)
    ax.set_xlabel(r"full $\log Z_\mathrm{cross}$ evaluations")
    ax.set_ylabel(r"best $\log Z_\mathrm{cross}$ [nats/box]")
    ax.set_xlim(0, 61.5); ax.set_ylim(28.6, ref + sf + 0.15)
    ax.grid(color="0.92", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(top=False, right=False)
    fig.tight_layout()
    PS.save(fig, M / "PaperUpdate" / "figs" / "out" / "ablation_v4")


if __name__ == "__main__":
    figure() if sys.argv[1:] == ["fig"] else main()

"""Master ranking of every analytic form SANDBOX-ML4PS scored at k <= 10 on the registered 16-bin grid (41-node ladder):
Base (4 paper forms; van Daalen = Gate-0 fail, unranked) + main loop T01-T60 (T51-T60 = Chi2Push X01-X10). Ranked by
log Z_cross (per box = total / 3726). Cut14 runs are listed in a separate, non-comparable table. Plans/03_rank_and_figures.md.

Outputs: PaperUpdate/RANKING.md, PaperUpdate/ranking.csv, PaperUpdate/ranking.json
Usage: python -m PaperUpdate.rank_all
"""
import csv
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M))

import yaml

from PaperUpdate.rescore_all import BASE_KEY, targets

NBOX = 3726
NK = 16
SUITES = ("IllustrisTNG", "SIMBA", "Astrid", "Swift-EAGLE")
OUTD = M / "PaperUpdate"
CAVEAT = {"T43": "one non-promoted ladder step not converged (Evidence/RESULT.md); score is a lower bound"}


def tex_eq(key, expr):
    from Loop import form_labels as FL
    k = {"P1": "rank1", "P2": "rank2", "P3": "rank3", "P12": "rank12"}.get(key, key)
    if k in FL.EQ:
        return FL.EQ[k]
    import sympy as sp
    syms = {n: sp.Symbol(n) for n in ("Omega_m", "Omega_b_over_Omega_m", "f_b", "sigma_8", "D", "k")}
    syms["Omega_b_over_Omega_m"] = sp.Symbol(r"(\Omega_b/\Omega_m)")
    syms["Omega_m"] = sp.Symbol(r"\Omega_m"); syms["sigma_8"] = sp.Symbol(r"\sigma_8")
    return sp.latex(sp.sympify(expr, locals=syms))


def rows():
    T = targets()
    Lall = {}
    p = OUTD / "L_all.jsonl"
    if p.exists():
        for l in p.read_text().splitlines():
            r = json.loads(l); Lall[r["key"]] = r
    Lc = json.loads((OUTD / "L_corr.json").read_text()) if (OUTD / "L_corr.json").exists() else {}
    out = []
    for key, (src, sub, t, used) in T.items():
        pp = int(t["n_coeff"]) if "n_coeff" in t else len(t["lin"]) + len(t["nl"])
        r = {"key": key, "source": src, "trial": t["trial"], "submission": str(Path(sub).relative_to(M)), "p": pp,
             "expr": t["expr"], "logZ_total": t["logZ_cross"], "logZ_box": t["logZ_cross"] / NBOX,
             "sigma_fold_box": t["sigma_fold"] / NBOX, "marginalized": t["marginalized"],
             "chi2_nu_mean": t["chi2_mean"], "chi2_13": t["chi2_mean"] * (NK - pp) / 13,
             **{f"chi2_{j}": t["chi2_per_suite"][j] for j in SUITES},
             "L_zero_stored": t["L_zero_shot"], "L_map_stored": t["L_map"], "L_stored_sigma": used,
             "caveat": CAVEAT.get(key, "")}
        if key in Lall:
            for s in ("paper", "inflated", "hartlap"):
                r[f"L_zero_{s}"] = Lall[key][f"{s}_sigma"]["L_zero"]; r[f"L_map_{s}"] = Lall[key][f"{s}_sigma"]["L_map"]
            r["L_check"] = Lall[key]["check_matches_stored"]
        if key in Lc:
            r["L_zero_corr"] = Lc[key]["L_zero"]; r["L_map_corr"] = Lc[key]["L_map"]
            for j in SUITES:
                r[f"L_zero_corr_{j}"] = Lc[key]["L_zero_per_suite"][j]; r[f"L_map_corr_{j}"] = Lc[key]["L_map_per_suite"][j]
        out.append(r)
    out.sort(key=lambda r: -r["logZ_total"])
    best = out[0]
    for i, r in enumerate(out, 1):
        r["rank"] = i
        r["dlogZ_box"] = r["logZ_box"] - best["logZ_box"]
        r["d_in_sigma_fold"] = (r["logZ_total"] - best["logZ_total"]) / best["sigma_fold_box"] / NBOX
        r["near_tie"] = abs(r["d_in_sigma_fold"]) < 1.0
    return out


def cut14():
    f = M / "Chi2Push" / "cut14" / "trial_state.yaml"     # side study, not shipped in the public repo
    if not f.exists():
        return []
    Y = yaml.safe_load(f.read_text())["trials"]
    names = {f"T{i:02d}": f"X{i:02d}" for i in range(1, 11)} | {"T11": "R_T07", "T12": "R_T39", "T13": "R_T47"}
    o = [{"key": names.get(t["trial"], t["trial"]), "p": len(t["lin"]) + len(t["nl"]), "logZ_box": t["logZ_cross"] / NBOX,
          "marginalized": t["marginalized"], "chi2_nu_mean": t["chi2_mean"], "L_zero": t["L_zero_shot"], "L_map": t["L_map"]}
         for t in Y if t.get("status") == "scored"]
    return sorted(o, key=lambda r: -r["logZ_box"])


def check(R):
    """Cross-check against Evidence/results.jsonl (full grid, x1) and Loop/L_inflated.jsonl."""
    by = {r["trial"] if r["source"].startswith("Loop") else r["key"]: r for r in R}
    ev_map = {"rank1": "P1", "rank2": "P2", "rank3": "P3", "rank12": "P12"}
    n = 0
    for l in (M / "Evidence" / "results.jsonl").read_text().splitlines():
        e = json.loads(l)
        if e.get("n_nodes") in (41, 21) and e.get("sigma_scale") == 1.0:
            k = ev_map.get(e["label"], e["label"])
            if k in by:
                assert abs(by[k]["logZ_total"] - e["logZ_cross"]) < 0.5, (k, by[k]["logZ_total"], e["logZ_cross"]); n += 1
    lab = {"paper rank 1": "P1", "paper rank 2": "P2", "paper rank 3": "P3", "paper rank 12": "P12"}
    for l in (M / "Loop" / "L_inflated.jsonl").read_text().splitlines():
        e = json.loads(l); k = lab.get(e["label"], e["label"])
        if k in by and "L_map_inflated" in by[k]:
            assert abs(by[k]["L_map_inflated"] - e["inflated_sigma"]["L_map"]) < 1e-2 * max(1, e["inflated_sigma"]["L_map"]), k
            n += 1
    return n


def md(R, C14, nchk):
    has = all("L_map_hartlap" in r for r in R)
    nc = sum("L_map_corr" in r for r in R)
    f = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"
    L = ["# Master ranking of all k ≤ 10 forms in SANDBOX-ML4PS (16-bin grid, 41-node ladder)", "",
         f"{len(R)} forms: Base (4 paper forms) and the main loop T01–T60 (T01–T50 by lane agents; T51–T60 orchestrator-proposed, "
         "resubmitted from Chi2Push X01–X10 on 2026-11-14 with identical scores). van Daalen 2020 (Base T04) "
         "fails Gate 0 and is unranked. They are ranked by log Z_cross per box (total ÷ 3726). Δ is measured from rank 1, in "
         "nats/box and in units of rank 1's σ_fold; † marks a near-tie with the best form (|Δ| < σ_fold). "
         "χ²₁₃ = χ²_ν·(16−p)/13. **L is the correlated loss** ΔSᵀ (C_fv,j/h)⁻¹ ΔS, using each held-out suite's own shrunk CV covariance and "
         "the Hartlap factor h = 0.346 (`Lcorr.py`), for the zero-shot (L₀) and per-box MAP (L_MAP) predictions. "
         f"It is available for {nc}/{len(R)} forms so far. The diagonal-σ_k values (paper, inflated, diagonal Hartlap) are in `ranking.csv`.",
         f"Cross-checks passed: {nchk} values against `Evidence/results.jsonl` and `Loop/L_inflated.jsonl`. "
         + ("Every refit reproduced its stored L." if has and all(r.get("L_check") for r in R) else
            ("**L refit check failed for: " + ", ".join(r["key"] for r in R if r.get("L_check") is False) + "**" if has else
             "Hartlap L is pending (rescore_all has not finished).")), "",
         "| rank | id | source | p | log Z / box | Δ / box | Δ/σ_fold | σ_fold | marginalized | χ²_ν | χ²₁₃ | χ²_ν TNG/SIMBA/Astrid/Swift | L₀ (corr) | L_MAP (corr) | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in R:
        L.append(f"| {r['rank']} | {r['key']}{' †' if r['near_tie'] and r['rank'] > 1 else ''} | {r['source']} | {r['p']} | "
                 f"{r['logZ_box']:.3f} | {r['dlogZ_box']:+.3f} | {r['d_in_sigma_fold']:+.2f} | {r['sigma_fold_box']:.2f} | "
                 f"{', '.join(r['marginalized'])} | {r['chi2_nu_mean']:.2f} | {r['chi2_13']:.2f} | "
                 + " / ".join(f"{r['chi2_'+j]:.1f}" for j in SUITES) + " | "
                 f"{f(r.get('L_zero_corr'), 1)} | {f(r.get('L_map_corr'), 1)} | {r['caveat']} |")
    L += ["", "## Equations", "", "| id | S(k) (engine expression) |", "|---|---|"]
    L += [f"| {r['key']} | `{r['expr']}` |" for r in R]
    if C14: L += ["", "## Side table: Chi2Push cut14 (14 bins, k > 0.7). Not comparable with the table above, because the data vector differs.", "",
          "| rank | id | p | log Z / box | marginalized | χ²_ν | L_zero (infl.) | L_map (infl.) |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {i} | {r['key']} | {r['p']} | {r['logZ_box']:.3f} | {', '.join(r['marginalized'])} | {r['chi2_nu_mean']:.2f} | "
          f"{r['L_zero']:.2f} | {r['L_map']:.2f} |" for i, r in enumerate(C14, 1)]
    return "\n".join(L) + "\n"


def main():
    R = rows()
    nchk = check(R)
    for r in R:
        r["tex"] = tex_eq(r["key"], r["expr"])
    (OUTD / "ranking.json").write_text(json.dumps(R, indent=1))
    keys = [k for k in R[0] if k not in ("tex",)]
    allk = sorted({k for r in R for k in r if k != "tex"}, key=lambda k: (k not in keys, keys.index(k) if k in keys else 0))
    with open(OUTD / "ranking.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=allk, extrasaction="ignore"); w.writeheader()
        for r in R:
            w.writerow({k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()})
    (OUTD / "RANKING.md").write_text(md(R, cut14(), nchk))
    for r in R[:12]:
        print(r["rank"], r["key"], f"{r['logZ_box']:.3f}", f"{r['d_in_sigma_fold']:+.2f}", r["marginalized"], f"{r['chi2_13']:.2f}")


if __name__ == "__main__":
    main()

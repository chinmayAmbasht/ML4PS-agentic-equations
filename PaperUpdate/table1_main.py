"""Main-text Table 1 (owner 2026-11-18): ranks 1, 2, 3, 10, 49 + PySR (unconstrained, as in Fig. 3) + van Daalen+20 (fit).
Columns follow the reviewer comments: origin of each form (RevB 6), sigma_fold next to the gaps (RevB 7), zero-shot and adapted
L reported separately (RevC W2) with the same numbers as Fig. 3 (bench_corr.json), van Daalen's failed checks (RevB 3/4).
Writes outputs/tables/table1_rows.tex (main-text Table 1). Usage: python -m PaperUpdate.table1_main
"""
import json
from pathlib import Path

from PaperUpdate.make_tex import DST, EQ, M, NBOX, msym

RANKS = [1, 2, 3, 10, 49]


def num(x):
    return f"{x:.1f}" if x < 1000 else f"{x:.0f}"


def main():
    R = json.loads((M / "PaperUpdate" / "ranking.json").read_text())
    B = json.loads((M / "PaperUpdate" / "Benchmark" / "bench_corr.json").read_text())
    P = json.loads((M / "PySR" / "summary.json").read_text())["U"]
    best = min(R, key=lambda r: r["rank"])
    rows = []
    for r in sorted((r for r in R if r["rank"] in RANKS), key=lambda r: r["rank"]):
        b = B[r["key"]]
        rows.append(f"{r['rank']} & $\\mathcal{{S}} = {EQ[r['key']]}$ & "   # agent rows carry no source line (paper, 2026-11-21)
                    f"{len(r['marginalized'])} & {msym(r['marginalized'])} & ${r['dlogZ_box']:+.2f}$ & {r['sigma_fold_box']:.2f} & "
                    f"{num(b['transfer_zero_shot']['mean'])} & {num(b['transfer_map']['mean'])} \\\\")
    # PySR (unconstrained): pooled search equation through the same harness; rank = where it would place among the 64
    lz = P["logZ_box"]
    prank = 1 + sum(r["logZ_box"] > lz for r in R)
    h = P["harness"]
    b = B["PySR (unconstrained)"]
    rows.append(f"({prank}) & $\\mathcal{{S}} = -A\\,D\\,k^2/(q+k/k_0)^2$\\newline {{\\scriptsize PySR, pooled search, $p=2$}} & "
                f"{len(h['marginalized'])} & $A$ & ${lz - best['logZ_box']:+.2f}$ & {h['sigma_fold'] / NBOX:.2f} & "
                f"{num(b['transfer_zero_shot']['mean'])}$^\\dagger$ & {num(b['transfer_map']['mean'])} \\\\")
    b = B["vD20 refit"]
    rows.append(r"--- & $\mathcal{S} = -A\,e^{d\bar f_b}\,(2k)^a\,\dfrac{1+(2C\bar f_b)^{b-a}}{1+(C\bar f_b/k)^{b-a}}$"
                r"\newline {\scriptsize van Daalen+20, refit, $p=5$} & \multicolumn{4}{c}{fails $k^2$ and boundedness checks} & "
                f"{num(b['transfer_zero_shot']['mean'])} & {num(b['transfer_map']['mean'])}$^\\ddagger$ \\\\")
    out = "\n\\addlinespace[2pt]\n".join(rows) + "\n"
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "table1_rows.tex").write_text(out)
    print(out)


if __name__ == "__main__":
    main()

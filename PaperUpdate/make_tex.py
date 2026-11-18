"""Shared helpers for the paper tables (equation LaTeX, coefficient symbols) and the appendix pool table (all 64 forms).
Usage: python -m PaperUpdate.make_tex   -> outputs/tables/pool_table.tex
"""
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M))

from Loop import form_labels as FL
from PaperUpdate.expr_tex import tex

DST = M / "outputs" / "tables"
NBOX = 3726
EQ = dict(FL.EQ) | {
    "T51": r"-D\,W(u_0)\left[A+G\,\Omega_m+B\,\frac{u_0}{1+u_0}\right]",
    "T53": r"-D\,W(u_0)\left[A+G\,q+B\,\frac{u_0}{1+u_0}\right]",
    "T56": r"-AD\,[q\,W(k\sqrt{D}/0.68)+(1-q)\,\frac{u_0^4}{1+u_0^4}]+(E f_b-CD)\,W(u_1)",
    "P1": FL.EQ["rank1"], "P2": FL.EQ["rank2"], "P3": FL.EQ["rank3"], "P12": FL.EQ["rank12"]}
NOTATION = (r"$W(u)=u^2/(1+u^2)$, $q=f_b/(\Omega_b/\Omega_m)$, $\omega_b=\Omega_b/\Omega_m$, $u_0=k\sqrt{D}/k_0$, "
            r"$u_1=k\,(D/0.05)^{0.75}/k_1$")
SYM = {"k0": "k_0", "k1": "k_1", "k2": "k_2", "k00": "k_{00}", "kc": "k_c", "kb": "k_b", "alpha": r"\alpha", "beta": r"\beta",
       "A1": "A_1", "A2": "A_2", "dl": r"\delta", "b1": "b_1", "b2": "b_2"}


def msym(names):
    return ", ".join(f"${SYM.get(n, n)}$" for n in names) if names else "---"


def fnum(x, d=2):
    return "---" if x is None else f"{x:.{d}f}"


def table1(R, keys):
    rows = []
    for r in R:
        if r["key"] not in keys:
            continue
        rows.append(f"{r['rank']} & $\\displaystyle \\mathcal{{S}} = {EQ.get(r['key'], tex(r['expr']))}$\\newline "
                    f"{{\\scriptsize {r['key']}, $p={r['p']}$}} & {len(r['marginalized'])} & {msym(r['marginalized'])} & "
                    f"${r['dlogZ_box']:+.3f}$ & {r['sigma_fold_box']:.2f} \\\\\n\\addlinespace[1pt]")
    return "\n".join(rows) + "\n"


def pool(R):
    L = [r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\begin{longtable}{r l r p{0.46\textwidth} r l r r r r}",
         r"\caption{\NoteV{caption to be written. All " + str(len(R)) + r" forms scored on the 16-bin $k\le9.92$ grid, ranked by "
         r"$\log Z_\mathrm{cross}$ per box; $\Delta$ in nats/box from rank 1; $\chi^2_{13}=\chi^2_\nu(16-p)/13$; $L$ correlated, "
         r"$\Delta\mathcal{S}^\top(C_{\mathrm{fv},j}/h)^{-1}\Delta\mathcal{S}$, zero-shot / adapted. " + NOTATION + r".}\label{tab:pool_full}}\\",
         r"\toprule Rank & id & $p$ & Functional form $\mathcal{S}$ & $M$ & Marginalized & $\Delta\log Z$ & $\chi^2_{13}$ & $L_0$ & $L_\mathrm{MAP}$ \\ \midrule\endfirsthead",
         r"\toprule Rank & id & $p$ & Functional form $\mathcal{S}$ & $M$ & Marginalized & $\Delta\log Z$ & $\chi^2_{13}$ & $L_0$ & $L_\mathrm{MAP}$ \\ \midrule\endhead",
         r"\bottomrule\endlastfoot"]
    for r in R:
        L.append(f"{r['rank']} & {r['key']} & {r['p']} & ${EQ.get(r['key'], tex(r['expr']))}$ & {len(r['marginalized'])} & "
                 f"{msym(r['marginalized'])} & ${r['dlogZ_box']:+.2f}$ & {r['chi2_13']:.2f} & {fnum(r.get('L_zero_corr'), 1)} & "
                 f"{fnum(r.get('L_map_corr'), 1)} \\\\")
    L += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(L) + "\n"


def main(keys):
    """Write the appendix pool table (all 64 forms) to outputs/tables/pool_table.tex."""
    R = json.loads((M / "PaperUpdate" / "ranking.json").read_text())
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "pool_table.tex").write_text(pool(R))
    print("wrote", DST / "pool_table.tex")


if __name__ == "__main__":
    main(sys.argv[1:])

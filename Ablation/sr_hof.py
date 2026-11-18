"""A1 / A2 (Plans/05_ablation.md): every hall-of-fame equation of a PySR search -> engine submission -> manifest, so that the
selection is by log Z_cross (Loop.evaluate) instead of PySR's MSE/'best' choice.
  A1: PySR/runs/U_pooled and PySR/runs/C_pooled (the Plan 04 searches; nothing re-run)
  A2: Ablation/A2/run (the two-kernel template, Ablation/run_search_c2.py)
Conversion: PySR.to_form.to_submission with the loop rule (max 6 coefficients, max 2 nonlinear). If an equation has free
constants but none is nonlinear, its most influential linear constant is declared nonlinear (range x/÷10, flagged
'nl_promoted') because the harness needs 1-2 nonlinear coefficients. Equations without constants are recorded, not scored.
Usage: python -m Ablation.sr_hof A1|A2   -> Ablation/<arm>/manifest.jsonl, Ablation/<arm>/subs/*.json
"""
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M))

import numpy as np
import pandas as pd
import sympy as sp

from PySR import to_form as TF

SRC = {"A1": [("U", M / "PySR" / "runs" / "U_pooled"), ("C", M / "PySR" / "runs" / "C_pooled")],
       "A2": [("C2", M / "Ablation" / "A2" / "run")]}
_orig_sym = TF.sym_expr


def sym_expr(variant, eq, consts=None):
    if variant != "C2":
        return _orig_sym(variant, eq, consts)
    parts = dict(p.strip().split(" = ", 1) for p in eq.split(";"))
    e = {n: TF._parse(parts[n], TF.F_ARGS, consts) for n in ("a1", "f1", "a2", "f2")}

    def W(f):
        u2 = (TF.k / sp.exp(f)) ** 2
        return u2 / (1 + u2)
    return e["a1"] * W(e["f1"]) + e["a2"] * W(e["f2"])


TF.sym_expr = sym_expr
_simplify = sp.simplify


def _fast_zero(e):
    """Numeric stand-in for sp.simplify(e) == 0 in to_form's linearity check (symbolic simplify hangs on nested exps):
    e evaluated at 40 random points (features in their data range, constants at random O(1) values); returns 0 if |e| < 1e-9."""
    e = sp.sympify(e)
    if e == 0:
        return sp.Integer(0)
    syms = sorted(e.free_symbols, key=str)
    rng = np.random.default_rng(0)
    R = {"k": (0.3, 10), "D": (0.005, 0.12), "QQ": (0.2, 1.0), "Omega_m": (0.1, 0.5), "sigma_8": (0.6, 1.0),
         "Omega_b_over_Omega_m": (0.1, 0.25)}
    f = sp.lambdify(syms, e, "numpy")
    for _ in range(40):
        v = [rng.uniform(*R.get(str(s), (-2.0, 2.0))) for s in syms]
        with np.errstate(all="ignore"):
            x = complex(f(*v))
        if np.isfinite(x) and abs(x) > 1e-9:
            return e
    return sp.Integer(0)


sp.simplify = _fast_zero          # only this process; to_form.py itself is unchanged


def sample_data():
    from PySR.run_search import suite_rows
    from Scripts.common import config as C
    Xs = np.concatenate([suite_rows(s)[0] for s in C.SUITES])
    X = Xs[np.random.default_rng(0).choice(len(Xs), 6000, replace=False)]
    return [X[:, i] for i in range(6)]


def convert(variant, eq, data):
    consts = {}
    TF.sym_expr(variant, eq, consts)
    if not consts:
        return None, "no numeric constants (nothing to fit)"
    sub = TF.to_submission(variant, eq, data=data, max_coeff=6, max_nl=2)
    flag = None
    if not sub["nl"]:
        c = sub["lin"].pop(0)                                        # most influential (to_submission orders by influence)
        v = sub["pysr_values"][c]
        sub["nl"][c] = {"lo": v / 10, "hi": v * 10, "spacing": "log"} if v > 0 else {"lo": v * 10, "hi": v / 10, "spacing": "lin"}
        sub["roles"][c] = "other"
        flag = "nl_promoted"
    return sub, flag


def main(arm):
    data = sample_data()
    d = M / "Ablation" / arm
    (d / "subs").mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, run in SRC[arm]:
        hof = pd.read_csv(run / "hof.csv").sort_values("complexity")
        best = json.loads((run / "result.json").read_text())["best_equation"]
        for _, r in hof.iterrows():
            cid = f"{variant}_c{int(r['complexity']):02d}"
            rec = {"cid": cid, "variant": variant, "complexity": int(r["complexity"]), "mse_loss": float(r["loss"]),
                   "equation": r["equation"], "pysr_best": r["equation"] == best}
            try:
                sub, flag = convert(variant, r["equation"], data)
            except Exception as ex:      # noqa: BLE001
                sub, flag = None, f"conversion error: {type(ex).__name__}: {ex}"[:300]
            if sub is None:
                rec["skip"] = flag
            else:
                sub["hypothesis"] = f"[ablation {arm} {cid}] PySR {variant} hall of fame, complexity {rec['complexity']}: {r['equation']}"
                p = d / "subs" / f"{cid}.json"
                p.write_text(json.dumps(sub, indent=1))
                rec.update({"sub": str(p.relative_to(M)), "flag": flag, "n_lin": len(sub["lin"]), "n_nl": len(sub["nl"]),
                            "frozen": sub.get("frozen", {})})
            rows.append(rec)
            print(cid, rec.get("skip") or f"lin {rec['n_lin']} nl {rec['n_nl']} {flag or ''}", flush=True)
    with open(d / "manifest.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])

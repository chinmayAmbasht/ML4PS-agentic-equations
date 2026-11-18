"""Turn a PySR equation into an engine submission (Plans/04_pysr.md): numeric constants -> coefficients.
U strings use variable names k, Dx, q, Om, s8, wb. C strings are "f = ...; g = ..." with positional #i:
f(#1..#5) = (D, q, Om, s8, wb), g(#1..#6) = (u = k/(k+1), D, q, Om, s8, wb); S = W(k/exp(f)) * g, W(u) = u^2/(1+u^2).
A constant is linear if S is affine in it and jointly linear with the other linear ones; otherwise nonlinear (range x/÷10).
At most 6 coefficients (4 nonlinear, base mode); extra constants are frozen at their PySR values (least influential first).
Usage: python -m PySR.to_form <run dir>  -> <run dir>/submission.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import sympy as sp

k, D, q, Om, s8, wb = sp.symbols("k D QQ Omega_m sigma_8 Omega_b_over_Omega_m")
ENG = {"Dx": D, "q": q, "Om": Om, "s8": s8, "wb": wb, "k": k}
F_ARGS = [D, q, Om, s8, wb]
G_ARGS = [k / (k + 1), D, q, Om, s8, wb]


_NUM = re.compile(r"(?<![\w#.])(\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)")


def _parse(s, names, consts=None):
    s = s.replace("^", "**")
    if consts is not None:      # each PySR literal -> its own symbol BEFORE sympy rewrites (keeps 1 constant = 1 coefficient)
        def sub(m):
            c = f"c{len(consts)}"; consts[c] = float(m.group(1)); return c
        s = _NUM.sub(sub, s)
    s = re.sub(r"#(\d+)", lambda m: f"__a{m.group(1)}", s)
    loc = {f"__a{i + 1}": v for i, v in enumerate(names)} if isinstance(names, list) else dict(names)
    loc.update({"exp": sp.exp, "log": sp.log, "square": lambda x: x ** 2})
    if consts is not None:
        loc.update({c: sp.Symbol(c) for c in consts})
    return sp.sympify(s, locals=loc)


def sym_expr(variant, eq, consts=None):
    if variant == "U":
        return _parse(eq, ENG, consts)
    parts = dict(p.strip().split(" = ", 1) for p in eq.split(";"))
    f = _parse(parts["f"], F_ARGS, consts); g = _parse(parts["g"], G_ARGS, consts)
    u2 = (k / sp.exp(f)) ** 2
    return u2 / (1 + u2) * g


def to_submission(variant, eq, data=None, max_coeff=6, max_nl=4):
    consts = {}
    Sc = sym_expr(variant, eq, consts)
    assert not Sc.atoms(sp.Float), f"unparsed literal in {eq}: {Sc.atoms(sp.Float)}"
    cs = [sp.Symbol(c) for c in consts]
    val = {sp.Symbol(c): v for c, v in consts.items()}
    # influence of each constant on the data (|dS/dc * c| mean) to choose which to free
    infl = {}
    if data is not None:
        fn = sp.lambdify([k, D, q, Om, s8, wb] + list(cs), [sp.diff(Sc, c) for c in cs], "numpy")
        g = fn(*data, *[val[c] for c in cs])
        infl = {c: float(np.nanmean(np.abs(np.broadcast_to(gi, data[0].shape) * val[c]))) for c, gi in zip(cs, g)}
    order = sorted(cs, key=lambda c: -infl.get(c, abs(val[c])))
    lin, nl = [], []
    for c in order:
        if len(lin) + len(nl) >= max_coeff:
            break
        affine = sp.simplify(sp.diff(Sc, c, 2)) == 0 and all(sp.simplify(sp.diff(Sc, c, l)) == 0 for l in lin)
        if affine:
            lin.append(c)
        elif len(nl) < max_nl:
            nl.append(c)
    free = set(lin) | set(nl)
    Sf = Sc.subs({c: val[c] for c in cs if c not in free})
    names = {c: (f"A{i}" if c in lin else f"b{i}") for i, c in enumerate(order) if c in free}
    Sf = Sf.subs({c: sp.Symbol(n) for c, n in names.items()})
    expr = str(Sf).replace("QQ", "(f_b/Omega_b_over_Omega_m)")
    feats = sorted({str(s) for s in Sf.free_symbols} & {"D", "QQ", "Omega_m", "sigma_8", "Omega_b_over_Omega_m"})
    feats = ["f_b" if f == "QQ" else f for f in feats]
    if "f_b" in feats and "Omega_b_over_Omega_m" not in feats:
        feats.append("Omega_b_over_Omega_m")

    def rng(v):
        return {"lo": v / 10, "hi": v * 10, "spacing": "log"} if v > 0 else {"lo": v * 10, "hi": v / 10, "spacing": "lin"}
    missing = {n for n in names.values() if n not in {str(x) for x in Sf.free_symbols}}
    assert not missing, f"coefficients lost in simplification: {missing}"
    sub = {"expr": expr, "lin": [names[c] for c in lin], "nl": {names[c]: rng(val[c]) for c in nl},
           "roles": {**{names[c]: "amplitude" for c in lin}, **{names[c]: "other" for c in nl}}, "features": feats,
           "hypothesis": f"PySR {variant} pooled-search equation: {eq}",
           "pysr_values": {names[c]: val[c] for c in free}, "frozen": {str(c): val[c] for c in cs if c not in free}}
    return sub


if __name__ == "__main__":
    import pandas as pd
    run = Path(sys.argv[1])
    res = json.loads((run / "result.json").read_text())
    sub = to_submission(res["variant"], res["best_equation"])
    (run / "submission.json").write_text(json.dumps(sub, indent=1))
    print(json.dumps(sub, indent=1))

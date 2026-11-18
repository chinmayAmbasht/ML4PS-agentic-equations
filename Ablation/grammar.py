"""A3 fixed grammar (Plans/05_ablation.md): seed-only primitives from Base/sub (R01, R02, R03, R12), no loop-discovered motifs.

S = -D * sum_{m=1..M} A_m * w_m * W2(k / (k_{s(m)} * mod_m)),   M in {1, 2},  W2(u) = u^2/(1+u^2)
  w_m   in {1, Omega_b/Omega_m, 1 - Omega_b/Omega_m (R01), f_b (raw feature)}
  mod_m in {1, D^beta (R01), 1/(1 + alpha*D) (R02), (Omega_m/0.3)^(1/3) (R03)}
  scale symbols k_m: one per kernel or shared (R01 shares beta; R12 = two scales); beta/alpha shared when both kernels use them
  amplitudes: one shared A (R01) or one per kernel (R12)
Loop rule: p <= 6, 1-2 nonlinear coefficients (k_m, beta, alpha), 41 nodes. Duplicates removed by a numeric fingerprint.
Usage: python -m Ablation.grammar   -> Ablation/A3/grammar.jsonl
"""
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp

M = Path(__file__).resolve().parents[1]
OUT = M / "Ablation" / "A3" / "grammar.jsonl"

W = {"1": "1", "wb": "Omega_b_over_Omega_m", "1-wb": "(1 - Omega_b_over_Omega_m)", "fb": "f_b"}
MODS = ("none", "Dbeta", "alphaD", "Om")
RANGE = {"k": {"lo": 0.05, "hi": 50.0, "spacing": "log"}, "beta": {"lo": -1.5, "hi": 1.5, "spacing": "lin"},
         "alpha": {"lo": -5.0, "hi": 30.0, "spacing": "lin"}}
ROLE = {"k": "scale", "beta": "exponent", "alpha": "other"}


def kernel(w, mod, ks):
    sc = {"none": ks, "Dbeta": f"{ks}*D**beta", "alphaD": f"({ks}/(1 + alpha*D))", "Om": f"{ks}*(Omega_m/0.3)**(1/3)"}[mod]
    u = f"(k/({sc}))"
    ww = "" if w == "1" else f"{W[w]}*"
    return f"{ww}({u}**2/(1 + {u}**2))"


def build(kernels, shared_amp):
    """kernels: list of (w, mod, scale_symbol)."""
    if shared_amp:
        expr = "-A*D*(" + " + ".join(kernel(*kk) for kk in kernels) + ")"
        lin = ["A"]
    else:
        lin = [f"A{i + 1}" for i in range(len(kernels))]
        expr = "-D*(" + " + ".join(f"{a}*{kernel(*kk)}" for a, kk in zip(lin, kernels)) + ")"
    nl_names = sorted({kk[2] for kk in kernels}) + [p for p in ("beta", "alpha") if any(kk[1] == {"beta": "Dbeta", "alpha": "alphaD"}[p] for kk in kernels)]
    nl = {n: dict(RANGE["k" if n.startswith("k") else n]) for n in nl_names}
    roles = {**{a: "amplitude" for a in lin}, **{n: ROLE["k" if n.startswith("k") else n] for n in nl}}
    feats = {"D"} | ({"Omega_b_over_Omega_m"} if any(kk[0] in ("wb", "1-wb") for kk in kernels) else set()) \
        | ({"f_b"} if any(kk[0] == "fb" for kk in kernels) else set()) | ({"Omega_m"} if any(kk[1] == "Om" for kk in kernels) else set())
    return {"expr": expr, "lin": lin, "nl": nl, "roles": roles, "features": sorted(feats)}


_RNG = np.random.default_rng(0)
_PTS = {"k": _RNG.uniform(0.1, 10, 7), "D": _RNG.uniform(0.01, 0.1, 7), "f_b": _RNG.uniform(0.02, 0.15, 7),
        "Omega_m": _RNG.uniform(0.1, 0.5, 7), "Omega_b_over_Omega_m": _RNG.uniform(0.1, 0.2, 7)}


def fingerprint(sub):
    """Expression family fingerprint invariant to coefficient renaming: values at random points for 3 random coefficient draws,
    after summing over permutations is unnecessary because kernels are generated in canonical (sorted) order."""
    names = list(_PTS) + list(sub["lin"]) + list(sub["nl"])
    e = sp.sympify(sub["expr"], locals={n: sp.Symbol(n) for n in names})
    syms = sorted(e.free_symbols, key=str)
    co = [s for s in syms if str(s) not in _PTS]
    f = sp.lambdify(syms, e, "numpy")
    rng = np.random.default_rng(1)
    vals = []
    for _ in range(2):
        c = {str(s): rng.uniform(0.3, 2.0) for s in co}
        args = [_PTS[str(s)] if str(s) in _PTS else c[str(s)] for s in syms]
        vals.append(np.asarray(f(*args), float))
    return tuple(np.round(np.concatenate(vals), 9))


def enumerate_forms():
    forms = []
    kopts = [(w, m) for w in W for m in MODS]
    # M = 1
    for w, m in kopts:
        forms.append(("M1", [(w, m, "k1")], True))
    # M = 2, unordered pairs of kernels
    for (a, b) in itertools.combinations_with_replacement(range(len(kopts)), 2):
        (w1, m1), (w2, m2) = kopts[a], kopts[b]
        for shared_scale in (False, True):
            ks = ("k1", "k1") if shared_scale else ("k1", "k2")
            if shared_scale and (w1, m1) == (w2, m2):
                continue                                    # identical kernels: reduces to M = 1
            for shared_amp in (True, False):
                forms.append(("M2", [(w1, m1, ks[0]), (w2, m2, ks[1])], shared_amp))
    out, seen = [], {}
    for tag, kk, sa in forms:
        sub = build(kk, sa)
        if not 1 <= len(sub["nl"]) <= 2 or len(sub["lin"]) + len(sub["nl"]) > 6:
            continue
        if not sa and len(kk) == 2 and kk[0] == kk[1]:
            continue                                        # A1, A2 on the same kernel: not identifiable
        fp = fingerprint(sub)
        if fp in seen:
            continue
        seen[fp] = True
        gid = f"G{len(out) + 1:03d}"
        sub["hypothesis"] = f"[ablation A3 grammar {gid}] {tag} kernels={kk} shared_amp={sa}"
        out.append({"gid": gid, "kernels": kk, "shared_amp": sa, "sub": sub})
    return out


if __name__ == "__main__":
    F = enumerate_forms()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        for r in F:
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(F), "forms;", Counter((len(r["kernels"]), len(r["sub"]["nl"])) for r in F))

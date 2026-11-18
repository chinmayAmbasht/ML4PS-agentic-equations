"""A2 (Plans/05_ablation.md): PySR with a template matched to the agents' structural space, a sum of two W2 kernels:
S = a1(x) * W(k / exp(f1(x))) + a2(x) * W(k / exp(f2(x))),  W(u) = u^2/(1+u^2),  x = (Dx, q, Om, s8, wb), ops {+,-,*,exp}.
=> S -> 0 as k^2 and S bounded as k -> oo by construction (G0_1-3). Everything else as PySR/run_search.py (Plan 04):
rows, features, weighted-MSE loss (weight h / diag C_fv,j), maxsize 30, 40 populations, batch 2000, procs 7, seed 0.
Pooled search only (all 4 suites), as the Plan 04 pooled searches. Output: Ablation/A2/run/{hof.csv, result.json}.
Usage: python -m Ablation.run_search_c2 [timeout_s] [seed] [outdir]
"""
import json
import os
import sys
import time
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_abl", f"ns_pysr"))
sys.path.insert(0, str(M))

import numpy as np

from PySR.run_search import FEATS, suite_rows
from Scripts.common import config as C

X5 = "Dx, q, Om, s8, wb"
COMBINE = (f"a1({X5}) * ((k / exp(f1({X5})))^2 / (1 + (k / exp(f1({X5})))^2)) + "
           f"a2({X5}) * ((k / exp(f2({X5})))^2 / (1 + (k / exp(f2({X5})))^2))")


def main(timeout=2700, seed=0, out=None):
    from pysr import PySRRegressor, TemplateExpressionSpec
    out = Path(out) if out else M / "Ablation" / "A2" / "run"
    out.mkdir(parents=True, exist_ok=True)
    parts = [suite_rows(s) for s in C.SUITES]
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts]); w = np.concatenate([p[2] for p in parts])
    spec = TemplateExpressionSpec(expressions=["a1", "f1", "a2", "f2"], variable_names=FEATS, combine=COMBINE)
    m = PySRRegressor(binary_operators=["+", "-", "*"], unary_operators=["exp"], nested_constraints={"exp": {"exp": 1}},
                      expression_spec=spec, maxsize=30, populations=40, niterations=100000, timeout_in_seconds=timeout,
                      batching=True, batch_size=2000, procs=7, model_selection="best", random_state=seed,
                      output_directory=str(out), run_id="search", progress=False, verbosity=0)
    t0 = time.time()
    m.fit(X, y, weights=w, variable_names=FEATS)
    eq = m.equations_
    eq.drop(columns=[c for c in eq.columns if c in ("lambda_format", "sympy_format")], errors="ignore").to_csv(out / "hof.csv", index=False)
    best = m.get_best()
    res = {"variant": "C2", "fold": "pooled", "train": list(C.SUITES), "seed": seed, "timeout_s": timeout,
           "seconds": round(time.time() - t0), "n_rows": int(len(y)), "combine": COMBINE,
           "best_equation": str(best["equation"]), "best_complexity": int(best["complexity"]), "best_loss": float(best["loss"])}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 2700, int(a[1]) if len(a) > 1 else 0, a[2] if len(a) > 2 else None)

"""One PySR search (Plans/04_pysr.md). Variant U = unconstrained, C = template-constrained
S = W(k / exp(f(x))) * g(k/(k+1), x), W(u) = u^2/(1+u^2), operators {+,-,*,exp} -> S ~ k^2 at small k, bounded at large k.
Fold = a suite name (train on the other 3, score that suite zero-shot) or "pooled" (train on all 4).
Rows: (box, k) of the analysis boxes; features k, Dx=D, q=f_b/(Omega_b/Omega_m), Om, s8, wb=Omega_b/Omega_m.
Search loss: weighted MSE, weight = 1/diag(C_fv,j/h)[k] by the row's suite. Reported L: correlated (PaperUpdate/Lcorr.py).
Output: PySR/runs/<variant>_<fold>/{hof.csv, result.json, model.pkl}.
Usage: python -m PySR.run_search U IllustrisTNG [timeout_s] [seed]
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns_pysr"))
sys.path.insert(0, str(M))

import numpy as np

from PaperUpdate import Lcorr as LC
from Scripts.common import config as C

FEATS = ["k", "Dx", "q", "Om", "s8", "wb"]
K = np.asarray(C.K_GRID)


def suite_rows(s):
    from Scripts.data import suppression_io as IO
    d = IO.load(s, "LH")
    nb = len(d.y)
    D = np.asarray(d.obs["D"], float).reshape(nb, -1)[:, 0]
    wb = np.asarray(d.obs["Omega_b_over_Omega_m"], float)
    om = np.asarray(d.obs["Omega_m"], float); s8 = np.asarray(d.obs["sigma_8"], float)
    q = (wb - D) / wb
    ok = d.mask.all(axis=1) & np.isfinite(D)
    box = np.column_stack([D, q, om, s8, wb])[ok]
    y = d.y[ok]
    X = np.column_stack([np.tile(K, len(box)), np.repeat(box, len(K), axis=0)])
    w = np.tile(LC.H / np.diag(LC.cfv(s)), len(box))
    return X, y.ravel(), w, y, box


def model_for(variant, timeout, seed, outdir):
    from pysr import PySRRegressor, TemplateExpressionSpec
    common = dict(maxsize=30, populations=40, niterations=100000, timeout_in_seconds=timeout, batching=True, batch_size=2000,
                  procs=7, model_selection="best", random_state=seed, output_directory=str(outdir), run_id="search",
                  progress=False, verbosity=0)
    if variant == "U":
        return PySRRegressor(binary_operators=["+", "-", "*", "/"], unary_operators=["exp", "log", "square"],
                             nested_constraints={"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0}}, **common)
    spec = TemplateExpressionSpec(
        expressions=["f", "g"], variable_names=FEATS,
        combine="((k / exp(f(Dx, q, Om, s8, wb)))^2 / (1 + (k / exp(f(Dx, q, Om, s8, wb)))^2)) * g(k / (k + 1), Dx, q, Om, s8, wb)")
    return PySRRegressor(binary_operators=["+", "-", "*"], unary_operators=["exp"],
                         nested_constraints={"exp": {"exp": 1}}, expression_spec=spec, **common)


def main(variant, fold, timeout=2700, seed=0):
    out = M / "PySR" / "runs" / f"{variant}_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    train = [s for s in C.SUITES if s != fold]
    parts = [suite_rows(s) for s in train]
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts]); w = np.concatenate([p[2] for p in parts])
    t0 = time.time()
    m = model_for(variant, timeout, seed, out)
    m.fit(X, y, weights=w, variable_names=FEATS)
    eq = m.equations_
    eq.drop(columns=[c for c in eq.columns if c in ("lambda_format", "sympy_format")], errors="ignore").to_csv(out / "hof.csv", index=False)
    best = m.get_best()
    res = {"variant": variant, "fold": fold, "train": train, "seed": seed, "timeout_s": timeout, "seconds": round(time.time() - t0),
           "n_rows": int(len(y)), "best_equation": str(best["equation"]), "best_complexity": int(best["complexity"]),
           "best_loss": float(best["loss"])}
    if fold in C.SUITES:
        Xh, _, _, Yh, _ = suite_rows(fold)
        pred = m.predict(Xh).reshape(Yh.shape)
        res["L_zero_corr"] = LC.L(pred, Yh, fold)
        res["n_heldout_boxes"] = int(len(Yh))
    try:
        with open(out / "model.pkl", "wb") as fh:
            pickle.dump(m, fh)
    except Exception as e:      # noqa: BLE001 - Julia objects may not pickle; the hall of fame CSV is the record
        res["pickle_error"] = repr(e)[:200]
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 2700, int(a[3]) if len(a) > 3 else 0)

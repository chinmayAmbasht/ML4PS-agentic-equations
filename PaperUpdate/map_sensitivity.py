"""Sensitivity of the adapted (per-box MAP) L to the MAP settings. Owner 2026-11-14. Reporting only; stored scores untouched.

Settings (same Stage-2 fold prior N(mu_j, w Sigma_j), same Nelder-Mead from mu_j as Scripts.analysis.plot_trial._boxes):
  current : w = 1.30, likelihood C_fv + diag C_disc     (the stored PaperUpdate/preds/<key>.npz; not refit here)
  w1      : w = 1.00, likelihood C_fv + diag C_disc
  fvh     : w = 1.00, likelihood C_fv,j / h             (the same metric as the benchmark L; Lcorr.cfv)
Each setting's MAP predictions are scored with the correlated L (Lcorr.L). Also checks that 'current' recomputed from the
stored npz equals bench_corr.json's transfer_map.
Output: PaperUpdate/map_sensitivity.json (merged per key); preds in PaperUpdate/preds_sens/<key>__<setting>.npz.
Usage: python -m PaperUpdate.map_sensitivity <group>   with group in {loop, pysr, vd}
"""
import json
import os
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
GROUP = sys.argv[1] if len(sys.argv) > 1 else "loop"
NS = {"loop": "ns_PU", "pysr": "ns_pysr_pipe", "vd": "ns_VD"}[GROUP]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"{NS}"))      # reuse the fold caches of the original fits
os.environ.setdefault("PK_FROZEN", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"frozen.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.path.insert(0, str(M))

import numpy as np
from scipy.optimize import minimize

OUT = M / "PaperUpdate" / "map_sensitivity.json"
PD = M / "PaperUpdate" / "preds_sens"
SETTINGS = {"w1": (1.00, "fv+disc"), "fvh": (1.00, "fv/h")}
BENCH = {"PYSR_U": "PySR (unconstrained)", "VD": "vD20 refit"}


def _boxes(args):
    form, f, b, suite, widen, lik = args
    from PaperUpdate import Lcorr as LC
    from Scripts.common import config as C
    from Scripts.engine import evidence as E
    from Scripts.engine import stage1 as S1
    mu = np.asarray(f["mu"], float)
    Si = np.linalg.inv(widen * np.asarray(f["Sigma"], float))
    cov = b.cfv + np.diag(f["cdisc_diag"]) if lik == "fv+disc" else LC.cfv(suite) / LC.H
    mp = np.full_like(b.y, np.nan)
    for i in range(len(b.y)):
        o = {k: v[[i]] for k, v in b.obs.items()}; mi = b.mask[[i]]
        wt = S1.BlockWhitener(cov, C.N_K, mi)
        yw = wt.w(b.y[[i]])

        def neg(c):
            p = form.predict(o, C.REDSHIFTS, E.to_native(form, f["order"], c))
            if not np.all(np.isfinite(p[mi])):
                return 1e30
            r = yw - wt.w(p); d = c - mu
            return 0.5 * float(np.sum(r * r)) + 0.5 * float(d @ Si @ d)
        c = minimize(neg, mu.copy(), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000}).x
        mp[i] = form.predict(o, C.REDSHIFTS, E.to_native(form, f["order"], c))[0]
    return np.where(b.mask, mp, np.nan), np.where(b.mask, b.y, np.nan)


def forms():
    from Scripts.common import config as C
    if GROUP == "loop":
        from PaperUpdate.rescore_all import targets
        T = targets()
        for k in ["T47", "T36", "T43", "T39", "T51"]:
            sub = json.loads(Path(T[k][1]).read_text())
            yield k, sub, (C.NL_GRID_NODES if len(sub["nl"]) <= 2 else 21)
    elif GROUP == "pysr":
        sub = json.loads((M / "PySR" / "summary.json").read_text())["U"]["submission"]     # the scored U submission
        yield "PYSR_U", sub, (C.NL_GRID_NODES if len(sub["nl"]) <= 2 else 21)
    else:
        yield "VD", json.loads((M / "Base" / "sub" / "R32_vanDaalen.json").read_text()), 11


def main():
    from Loop.plot_fits import form_of
    from PaperUpdate import Lcorr as LC
    from Scripts.common import config as C
    from Scripts.engine import chi2 as X2
    from Scripts.engine import scoring as SC
    from Scripts.engine.par import pmap
    from Scripts.run.score_forms import build_blocks
    PD.mkdir(exist_ok=True)
    B = json.loads((M / "PaperUpdate" / "Benchmark" / "bench_corr.json").read_text())
    for key, sub, n in forms():
        form = form_of(key, sub, n)
        blocks, _, _ = build_blocks(form, C.REDSHIFTS)
        fp = X2.fold_params(SC.cross_suite_folds(form, blocks, C.REDSHIFTS))
        z = np.load(M / "PaperUpdate" / "preds" / f"{key}.npz")
        rec = {"current": {s: LC.L(z[f"{s}__map"], z[f"{s}__y"], s) for s in C.SUITES}}
        ref = B[BENCH.get(key, key)]["transfer_map"]["per_suite"]
        rec["current_matches_bench"] = all(abs(rec["current"][s] - ref[s]) < 1e-6 * max(1, ref[s]) for s in C.SUITES)
        for name, (w, lik) in SETTINGS.items():
            out = dict(zip(C.SUITES, pmap(_boxes, [(form, fp[s], blocks[s], s, w, lik) for s in C.SUITES])))
            np.savez_compressed(PD / f"{key}__{name}.npz", **{f"{s}__map": out[s][0] for s in out})
            rec[name] = {s: LC.L(out[s][0], out[s][1], s) for s in C.SUITES}
        for k in ("current", *SETTINGS):
            rec[k] = {"per_suite": rec[k], "mean": float(np.mean(list(rec[k].values())))}
        allr = json.loads(OUT.read_text()) if OUT.exists() else {}
        allr[key] = rec
        OUT.write_text(json.dumps(allr, indent=1))
        print(key, rec["current_matches_bench"], {k: round(rec[k]["mean"], 2) for k in ("current", *SETTINGS)}, flush=True)


if __name__ == "__main__":
    main()

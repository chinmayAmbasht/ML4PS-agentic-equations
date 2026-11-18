"""ML4PS G0_U (the owner-approved end-slope consistency rule of 2026-11-10, recalibrated on the current grid).

Main loop (Plans/01_ml4ps_k10.md): 16 bins with k <= 9.92. At z = 0 the data mostly still dives at the last bins (16-30 % of TNG,
Astrid and Swift boxes have a non-negative last step, SIMBA 51 %), so an upturn is not required. The fitted end step
S(9.92) - S(8.67) must lie inside the pooled-data [1 %, 99 %] range (Artifacts/upturn_calib_z0.json, made by
Scripts/analysis/upturn_calib_z0.py), for >= F_PASS of boxes per coefficient set: every fold's mu and every training suite's
Stage-1 c_hat, over all scored boxes of all suites (as Scripts/engine/gate0.upturn).
"""
import json

import numpy as np

from ..common import config as C
from . import evidence as E

CALIB = json.loads((C.ARTIFACTS / "upturn_calib_z0.json").read_text())
# the band uses only the last two bins, so an opt-in low-k cut (C.KMIN, Chi2Push/) keeps it valid
assert CALIB["k_grid"][-len(C.K_GRID):] == list(C.K_GRID) and (C.KMIN > 0 or len(CALIB["k_grid"]) == C.N_K) and tuple(C.REDSHIFTS) == (0.0,)
BAND = (CALIB["end_step_quantiles"][CALIB["q"].index(0.01)], CALIB["end_step_quantiles"][CALIB["q"].index(0.99)])
F_PASS = 0.90


def upturn_z0(form, folds, blocks, redshifts):
    K = np.asarray(C.K_GRID, float)
    lo, hi = BAND
    sets = []
    for j, f in folds.items():
        sets.append((f"mu[{j}]", E.to_native(form, f["order"], np.asarray(f["mu"], float))))
        sets += [(f"c_hat[{j}/{s}]", E.to_native(form, f["order"], np.asarray(r["c_hat"], float))) for s, r in f["stage1"].items()]
    feats = []
    for b in blocks.values():
        v = b.mask[:, 0]
        feats.append([np.asarray(b.obs[x], float).reshape(len(b.y), -1)[:, 0][v] for x in form.features])
    F = [np.concatenate([fs[i] for fs in feats])[:, None] for i in range(len(form.features))]
    fr, worst = {}, []
    for name, c in sets:
        with np.errstate(all="ignore"):
            S = np.asarray(form._f_full(K[None, :], 0.0, *F, *[c[x] for x in form.nl], *[c[x] for x in form.lin]), float)
        S = np.broadcast_to(S, (len(F[0]) if F else 1, len(K)))
        step = S[:, -1] - S[:, -2]
        fr[name] = float(np.mean(np.isfinite(step) & (step >= lo) & (step <= hi)))
        if fr[name] < F_PASS:
            worst.append({"set": name, "frac_pass": fr[name], "median_end_step": float(np.nanmedian(step))})
    return {"G0_U": not worst, "min_frac_pass": min(fr.values()), "band": [lo, hi], "worst": worst[:6],
            "rule": "end step S(k_16) - S(k_15) in pooled-data [1%, 99%] at z = 0, >= 0.90 of boxes per coefficient set"}

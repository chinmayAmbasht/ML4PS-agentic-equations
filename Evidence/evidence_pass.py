"""Evidence pass (owner 2026-11-11, after the ML4PS loop): full-resolution log Z_cross with the promotion ladder for the top loop
forms and the paper's base forms, with convergence diagnostics and two sensitivities.

Per form:
  - grid check: the ladder on the coarse loop grid (11 nodes per nonlinear coefficient) and on the registered full grid
    (C.NL_GRID_NODES = 41; 21 for forms with 3+ nonlinear coefficients, disclosed in the output), recording per-fold
    log Z_pred, the selected marginalized set, sigma_fold, and every step's per-fold quadrature convergence and tail mass;
  - prior-width sensitivity (reviewer B item 2): the full-grid ladder with the fitted Sigma_suite of every fold scaled by
    0.5 and 2.0 (the Stage-2 result is rescaled after the registered fit; stage2.py is not modified).
Output: Evidence/results.jsonl (one line per (form, setting)), Evidence/pass.log.

Usage:  python -m Evidence.evidence_pass <label>=<submission.json> ...
"""
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

EV = Path(__file__).resolve().parent
SBX = EV.parent
os.environ.setdefault("PK_NS_DIR", str(EV / "ns"))
os.environ.setdefault("PK_FROZEN", str(EV / "candidates_frozen.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.path.insert(0, str(SBX))

import numpy as np

from Scripts.common import config as C
from Scripts.engine import evidence as E
from Scripts.engine import gate0 as G0
from Scripts.engine import scoring as SC
from Scripts.engine.forms import Form
from Scripts.run.score_forms import build_blocks

COARSE = 11                     # the coarse grid of the first (k <= 20) loop, kept as a grid check
_FIT = E.S2.fit
SCALE = {"v": 1.0}


def _scaled_fit(*a, **kw):
    r = _FIT(*a, **kw)
    return r if SCALE["v"] == 1.0 else dataclasses.replace(r, Sigma=np.asarray(r.Sigma) * SCALE["v"])


E.S2.fit = _scaled_fit          # in-process only; the registered fit runs unchanged, its Sigma is rescaled afterwards


def form_of(label, sub, n_nodes):
    nl = {k: {"lo": float(v["lo"]), "hi": float(v["hi"]), "spacing": v["spacing"], "n_nodes": n_nodes} for k, v in sub["nl"].items()}
    return Form(label, sub["expr"], list(sub["lin"]), nl, dict(sub["roles"]), list(sub["features"]))


def ladder_record(label, sub, n_nodes, scale):
    SCALE["v"] = scale
    form = form_of(label, sub, n_nodes)
    t0 = time.time()
    blocks, _, _ = build_blocks(form, C.REDSHIFTS)
    folds = SC.cross_suite_folds(form, blocks, C.REDSHIFTS)
    lad = SC.ladder(form, folds, blocks, C.REDSHIFTS)
    sel = lad["steps"][[s["R"] for s in lad["steps"]].index(lad["selected_R"])]
    return {"label": label, "n_nodes": n_nodes, "sigma_scale": scale, "logZ_cross": lad["logZ_cross"],
            "marginalized": lad["selected_R"], "sigma_fold": sel["sigma_fold"], "logZ_pred_per_fold": sel["per_fold"],
            "ladder": [{"R": s["R"], "logZ_cross": s["logZ_cross"], "promoted": s["promoted"], "gain": s.get("gain"),
                        "threshold": s.get("threshold_sigma_fold_of_simpler"), "converged": s.get("conv"),
                        "tail_mass": s.get("tail")} for s in lad["steps"]],
            "all_converged": all(all(v for v in (s.get("conv") or {}).values()) for s in lad["steps"]),
            "mu_per_fold": {j: dict(zip(f.coeff_order, np.round(f.s2.mu, 4).tolist())) for j, f in folds.items()},
            "Sigma_diag_per_fold": {j: dict(zip(f.coeff_order, np.round(np.diag(f.s2.Sigma), 5).tolist())) for j, f in folds.items()},
            "seconds": round(time.time() - t0)}


def main(items):
    for it in items:
        label, path = it.split("=", 1)
        sub = json.loads(Path(path).read_text())
        g0 = G0.check(form_of(label, sub, COARSE), redshifts=C.REDSHIFTS)
        if not (g0["G0_1"] and g0["G0_2"] and g0["G0_3"]):
            print(f"{label}: Gate 0 fail, skipped", flush=True); continue
        full = C.NL_GRID_NODES if len(sub["nl"]) <= 2 else 21
        for n, sc in ((COARSE, 1.0), (full, 1.0), (full, 0.5), (full, 2.0)):
            r = ladder_record(label, sub, n, sc)
            with open(EV / "results.jsonl", "a") as f:
                f.write(json.dumps(r, default=float) + "\n")
            print(f"{label} nodes {n} Sigma x{sc}: logZ_cross {r['logZ_cross']:.1f} marg {r['marginalized']} "
                  f"sigma_fold {r['sigma_fold']:.1f} converged {r['all_converged']} ({r['seconds']}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

"""ML4PS main-loop submit step (Plans/01_ml4ps_k10.md; first run Plans/00_ml4ps.md): z = 0 only, 16 k-bins with k <= 9.92 (the
paper's 13 bins <= 10 plus 3 fills), p <= 6 with 1-2 nonlinear coefficients. Copied from SANDBOX-v2/Loop/evaluate.py and extended with the paper's selection metric. One trial at a time (lock).

A submission is a JSON file:
  {"expr": "...", "lin": ["A", ...], "nl": {"k0": {"lo": 0.1, "hi": 10, "spacing": "log"}, ...},
   "roles": {"A": "amplitude", "k0": "scale", ...}, "features": ["D", ...],
   "hypothesis": "...", "predicted_logZ_cross": 107500, "predicted_chi2_mean": 6.0}
The harness assigns T<nn>, freezes the form (Loop/candidates_frozen.jsonl) before any number exists, then runs
Gate 0 G0_1-3 -> LOSO folds on the full grid (C.LOOP_NL_NODES = 41 per nonlinear coefficient) -> G0_U (z = 0 end-slope band) ->
  log Z_cross with the forward promotion ladder (the paper's selection metric; which coefficients to marginalize) ->
  per-suite chi^2_nu (C_fv, per-box MAP) and the benchmark loss L (zero-shot and per-box MAP under the 1.30-widened prior).
Results: Loop/trial_state.yaml (appended), Loop/residuals/T<nn>.md. Every submission counts against BUDGET.

Usage:  python -m Loop.evaluate <submission.json>      (PK_ML4PS_LOOP_DIR redirects state/freeze/residuals for dry runs)
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

LOOP = Path(os.environ.get("PK_ML4PS_LOOP_DIR", Path(__file__).resolve().parent))
SBX = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", str(LOOP / "ns"))
os.environ.setdefault("PK_FROZEN", str(LOOP / "candidates_frozen.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.path.insert(0, str(SBX))

import numpy as np
import yaml

from Scripts.common import config as C
from Scripts.engine import chi2 as X2
from Scripts.engine import gate0 as G0
from Scripts.engine import g0u_z0
from Scripts.engine import registry as REG
from Scripts.engine import scoring as SC
from Scripts.engine.forms import ALLOWED_FEATURES, Form, ROLES
from Scripts.run.score_forms import build_blocks

STATE, PRES, RESID, LOCK = LOOP / "trial_state.yaml", LOOP / "presentable.jsonl", LOOP / "residuals", LOOP / ".trial.lock"
BUDGET = int(os.environ.get("PK_ML4PS_BUDGET", "50"))   # Plans/01_ml4ps_k10.md (owner 2026-11-11); 60 for T51-T60 = Chi2Push X01-X10 (owner 2026-11-14)
LOCK_WAIT = 3600


def _trials():
    return (yaml.safe_load(STATE.read_text()) or {}).get("trials", []) if STATE.exists() else []


def _append(rec):
    t = _trials() + [rec]
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump({"budget": BUDGET, "n_used": len(t), "trials": t}, sort_keys=False, width=120))
    os.replace(tmp, STATE)


def _check(sub):
    """Form limits (Plans/00 sec. 5). Returns an error string or None."""
    lin, nl = list(sub.get("lin", [])), dict(sub.get("nl", {}))
    if len(lin) + len(nl) > C.MAX_COEFF:
        return f"{len(lin) + len(nl)} coefficients > {C.MAX_COEFF}"
    if not 1 <= len(nl) <= (6 if os.environ.get("PK_ML4PS_BASE") else 2):   # base mode: the paper's forms (Base/, not the loop)
        return f"{len(nl)} nonlinear coefficients (need 1-2)"
    bad = [f for f in sub.get("features", []) if f not in ALLOWED_FEATURES]
    if bad:
        return f"features not allowed: {bad} (allowed: {list(ALLOWED_FEATURES)})"
    if any(x in sub.get("expr", "") for x in C.FORBIDDEN_FEATURES):
        return "forbidden feature in the expression"
    if set(sub.get("roles", {})) != set(lin) | set(nl) or not set(sub["roles"].values()) <= set(ROLES):
        return f"roles must cover every coefficient with one of {ROLES}"
    return None


def _residual_md(tid, expr, x2):
    z = list(C.REDSHIFTS); K = C.K_GRID
    L = [f"# Residuals {tid}", "", f"`{expr}`", "",
         "Per held-out suite (leave-one-suite-out, per-box MAP prediction): mean signed and mean squared residual in units of the "
         "C_fv diagonal sigma, per (z, k). Positive = data above the prediction.", ""]
    for j, r in x2["residuals"].items():
        L += [f"## {j} (held out; chi2_nu {x2['per_suite'][j]:.2f}, {r['n_box']} boxes)", "",
              "mean signed | " + " | ".join(f"k={k:.2f}" for k in K), "|".join(["---"] * (len(K) + 1))]
        L += [f"z={zz:.2f} | " + " | ".join(f"{v:+.2f}" for v in row) for zz, row in zip(z, r["resid_mean_signed"])]
        L += ["", "mean squared | " + " | ".join(f"k={k:.2f}" for k in K), "|".join(["---"] * (len(K) + 1))]
        L += [f"z={zz:.2f} | " + " | ".join(f"{v:.1f}" for v in row) for zz, row in zip(z, r["resid_mean_sq"])]
        L += [""]
    RESID.mkdir(exist_ok=True)
    (RESID / f"{tid}.md").write_text("\n".join(L))


def _holder_alive():
    try:
        pid = int(LOCK.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return time.time() - LOCK.stat().st_mtime < LOCK_WAIT    # no pid (old-style lock): trust it for an hour
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _acquire():
    """Wait for the lock (another agent's trial is being evaluated); take over a lock whose process has died."""
    t0, said = time.time(), False
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode()); os.close(fd); return
        except FileExistsError:
            try:
                if not _holder_alive():
                    os.unlink(LOCK); continue
            except FileNotFoundError:
                continue
        if time.time() - t0 > LOCK_WAIT:
            sys.exit("HARNESS ERROR: waited an hour for Loop/.trial.lock (trial not counted)")
        if not said:
            print("another agent's trial is being evaluated; waiting for the lock ...", flush=True); said = True
        time.sleep(5)


def run(path):
    _acquire()
    try:
        n = len(_trials())
        if n >= BUDGET:
            sys.exit(f"budget exhausted ({n}/{BUDGET} trials)")
        tid = f"T{n + 1:02d}"
        sub = json.loads(Path(path).read_text())
        t0 = time.time()
        rec = {"trial": tid, "expr": sub.get("expr"), "lin": sub.get("lin"), "nl": sub.get("nl"), "features": sub.get("features"),
               "hypothesis": sub.get("hypothesis", ""), "predicted_logZ_cross": sub.get("predicted_logZ_cross"), "predicted_chi2_mean": sub.get("predicted_chi2_mean"),
               "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
        err = _check(sub)
        form = None
        if not err:
            try:
                nn = C.LOOP_NL_NODES if len(sub["nl"]) <= 2 else 21      # 3+ nl (base mode only, paper rank 1): 21, as Evidence/
                nl = {k: {"lo": float(v["lo"]), "hi": float(v["hi"]), "spacing": v["spacing"], "n_nodes": nn}
                      for k, v in sub["nl"].items()}
                form = Form(tid, sub["expr"], list(sub["lin"]), nl, dict(sub["roles"]), list(sub["features"]))
            except Exception as ex:     # noqa: BLE001 -- does not compile: recorded, counts as a trial
                err = f"does not compile: {type(ex).__name__}: {ex}"
        if err:
            _append({**rec, "status": "rejected", "reason": err, "seconds": round(time.time() - t0)})
            print(f"{tid}: REJECTED {err}"); return
        REG.freeze(form, proposer="ml4ps-loop-agent", batch="ml4ps-loop")
        g0 = G0.check(form, redshifts=C.REDSHIFTS)
        if not (g0["G0_1"] and g0["G0_2"] and g0["G0_3"]):
            bad = {k: g0[k] for k in ("G0_1", "G0_2", "G0_3")}
            _append({**rec, "status": "gate0_fail", "gate0": bad, "seconds": round(time.time() - t0)})
            print(f"{tid}: GATE 0 FAIL {bad}"); return
        blocks, _, _ = build_blocks(form, C.REDSHIFTS)
        folds = SC.cross_suite_folds(form, blocks, C.REDSHIFTS)
        fp = X2.fold_params(folds)
        up = g0u_z0.upturn_z0(form, fp, blocks, C.REDSHIFTS)
        lad = SC.ladder(form, folds, blocks, C.REDSHIFTS)
        x2 = X2.per_suite(form, fp, blocks, C.REDSHIFTS)
        lz = SC.L_zero_shot(form, folds, blocks, C.REDSHIFTS)
        lm = SC.map_diagnostic_widened_prior(form, folds, blocks, C.REDSHIFTS)
        _residual_md(tid, form.expr, x2)
        ok = bool(up["G0_U"] and x2["presentable"])
        rec.update({"status": "presentable" if ok else "scored",
                    "gate0": {"G0_1": True, "G0_2": True, "G0_3": True, "G0_U": up["G0_U"],
                              "G0_U_min_frac_pass": round(up["min_frac_pass"], 3)},
                    "logZ_cross": round(lad["logZ_cross"], 1), "marginalized": lad["selected_R"],
                    "sigma_fold": round(lad["steps"][[st["R"] for st in lad["steps"]].index(lad["selected_R"])]["sigma_fold"], 1),
                    "ladder": [{"R": st["R"], "logZ_cross": round(st["logZ_cross"], 1), "promoted": st["promoted"]} for st in lad["steps"]],
                    "logZ_pred_per_fold": {j: round(v, 1) for j, v in
                                           lad["steps"][[st["R"] for st in lad["steps"]].index(lad["selected_R"])]["per_fold"].items()},
                    "L_zero_shot": round(lz["L_zero_shot"], 3), "L_zero_shot_per_fold": {j: round(v, 3) for j, v in lz["per_fold"].items()},
                    "L_map": round(lm["L_map_diagnostic_widened_prior"], 3), "L_map_per_fold": {j: round(v, 3) for j, v in lm["per_fold"].items()},
                    "chi2_per_suite": {j: round(v, 3) for j, v in x2["per_suite"].items()},
                    "chi2_mean": round(x2["mean"], 3), "chi2_std": round(x2["std"], 3),
                    "chi2_Cfv_Cdisc_per_suite": {j: round(v, 3) for j, v in x2["per_suite_Cfv_Cdisc"].items()},
                    "fold_mu": {j: dict(zip(f["order"], np.round(f["mu"], 4).tolist())) for j, f in fp.items()},
                    "residuals": f"Loop/residuals/{tid}.md", "n_coeff": form.n_coeff, "nl_nodes": nn, "seconds": round(time.time() - t0)})
        _append(rec)
        if ok:
            row = next(r for r in REG.frozen_rows() if r["form_id"] == tid)
            with open(PRES, "a") as f:
                f.write(json.dumps({"form_id": tid, "frozen": row, "chi2_mean": x2["mean"], "chi2_std": x2["std"]}) + "\n")
        print(f"{tid}: {rec['status'].upper()} logZ_cross {lad['logZ_cross']:.1f} marg {lad['selected_R']} "
              f"chi2 mean {x2['mean']:.2f} std {x2['std']:.2f} L_zero {lz['L_zero_shot']:.2f} L_map {lm['L_map_diagnostic_widened_prior']:.2f} "
              f"per suite {rec['chi2_per_suite']} G0_U {up['G0_U']} ({rec['seconds']}s)")
    except SystemExit:
        raise
    except Exception:            # noqa: BLE001 -- harness error: recorded, the trial is NOT counted
        print("HARNESS ERROR (trial not counted):\n" + traceback.format_exc()); sys.exit(2)
    finally:
        os.unlink(LOCK)


if __name__ == "__main__":
    run(sys.argv[1])

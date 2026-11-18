"""Full registered scoring of frozen forms (v2): Gate 0 (G0_1-3 + G0_U), cross-suite folds on the 41-node grid, ladder ->
log Z_cross, per-suite chi^2, and the zero-shot L. This is the log-evidence step the cheap loop hands to TACC.
Ported from v1 Scripts/run/score_forms.py (matched-subset, reference and k-cut modes removed).

Usage:  python -m Scripts.run.score_forms --forms ID [ID ...] [--nproc 1]
"""
import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from ..common import config as C
from ..common import provenance as P
from ..data import suppression_io as IO
from ..engine import chi2 as X2
from ..engine import cov as CV
from ..engine import evidence as E
from ..engine import gate0 as G0
from ..engine import registry as REG
from ..engine import scoring as SC
from ..engine.forms import Form, obs_with_fbm

SCORES = C.NS / "scores"


def form_from_frozen(row, n_nodes=None) -> Form:
    r = dict(row["record"])
    nl = {k: {"n_nodes": n_nodes or C.NL_GRID_NODES, **({k2: v2 for k2, v2 in v.items() if k2 != "n_nodes"} if n_nodes else v)}
          for k, v in r["nl"].items()}
    return Form(r["form_id"], r["expr"], r["lin"], nl, r["roles"], r["features"], r.get("notes", ""))


def build_blocks(form, redshifts):
    blocks, nscored, nexcl = {}, {}, {}
    for s in C.SUITES:
        d = IO.load(s, "LH", redshifts)
        obs = obs_with_fbm(d.obs)
        ok = form.feature_ok(obs)
        cv = IO.load(s, "CV", redshifts, require_valid=False)
        cf = CV.cfv(cv.y, len(redshifts))
        fb = form.feature_block_ok(obs)            # a block is scored only where every feature the form uses is finite
        mask = d.mask & (np.repeat(fb, C.N_K, axis=1) if fb.ndim == 2 else fb[:, None])
        ok &= mask.any(axis=1)
        blocks[s] = E.SuiteBlock(s, {f: np.asarray(obs[f])[ok] for f in form.features}, d.y[ok], mask[ok], cf["C"])
        nscored[s] = int(ok.sum()); nexcl[s] = int((~ok).sum())
    return blocks, nscored, nexcl


def _upsert(row):
    """On the cluster (PK_NO_REGISTRY=1) nodes write only per-form row JSON; the registry is rebuilt locally (tacc/merge)."""
    if os.environ.get("PK_NO_REGISTRY"):
        d = SCORES / "registry_rows"; d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{row['form_id']}.json.tmp"
        tmp.write_text(json.dumps(REG._clean(row), indent=1, default=str)); os.replace(tmp, d / f"{row['form_id']}.json")
        return
    REG.upsert(row)


def _log(fid, msg):
    C.LOGS.mkdir(parents=True, exist_ok=True)
    with open(C.LOGS / f"score_{fid}.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def score_one(row):
    redshifts = C.REDSHIFTS
    fid = row["form_id"]
    t0 = time.time()
    base = {"form_id": fid, "symbolic_expr": row["expr"], "seed": C.SEED}
    try:
        form = form_from_frozen(row, n_nodes=C.NL_GRID_NODES)   # loop freezes carry the coarse grid; full scoring is 41
        assert all(v.n_nodes == C.NL_GRID_NODES for v in form.nl.values())
    except Exception as ex:      # noqa: BLE001 -- unscorable as submitted; never repaired
        _upsert({**base, "notes": f"UNSCORABLE as submitted: {type(ex).__name__}: {ex}",
                 "_provenance": P.provenance("unscorable", redshifts=list(redshifts))})
        _log(fid, f"unscorable: {ex}")
        return fid, "unscorable"
    try:
        base.update(n_coeff=form.n_coeff, coeff_roles=form.roles)
        g0 = G0.check(form, redshifts=redshifts)
        _log(fid, f"gate0 G0_1-3 {g0['gate0_passed']}")
        blocks, nscored, nexcl = build_blocks(form, redshifts)
        folds = SC.cross_suite_folds(form, blocks, redshifts)
        _log(fid, f"folds done {time.time()-t0:.0f}s")
        lad = SC.ladder(form, folds, blocks, redshifts)
        _log(fid, f"ladder done R={lad['selected_R']} {time.time()-t0:.0f}s")
        R = lad["selected_R"]
        sel = lad["steps"][[s["R"] for s in lad["steps"]].index(R)] if R else lad["steps"][0]
        lz = SC.L_zero_shot(form, folds, blocks, redshifts)
        fold_par = X2.fold_params(folds)
        g0 = G0.apply_upturn(g0, G0.upturn(form, fold_par, blocks, redshifts))
        _log(fid, f"gate0 with G0_U {g0['gate0_passed']} (G0_U {g0['G0_U']})")
        x2 = X2.per_suite(form, fold_par, blocks, redshifts)
        _log(fid, f"chi2 mean {x2['mean']:.3f} std {x2['std']:.3f} {time.time()-t0:.0f}s")
        prov = P.provenance("cross_suite_logZ", redshifts=list(redshifts), quad_nodes=E.QUAD_NODES,
                            n_alternations=E.N_ALTERNATIONS, n_boxes_scored=nscored, n_boxes_feature_excluded=nexcl)
        mu_all = np.mean([np.array(f.s2.mu) for f in folds.values()], axis=0)
        Sig_all = np.mean([np.array(f.s2.Sigma) for f in folds.values()], axis=0)
        ev = np.linalg.eigvalsh(Sig_all)
        out = {
            **base,
            "G0_1": g0["G0_1"], "G0_2": g0["G0_2"], "G0_3": g0["G0_3"], "G0_U": g0["G0_U"], "gate0_passed": g0["gate0_passed"],
            "G0_by_redshift": g0["G0_by_redshift"], "G0_early_time_flag": g0["G0_early_time_flag"],
            "marg_coords": R, "n_marginalized": len(R), "ladder_mode": "forward one-at-a-time, sigma_fold ddof=1",
            "S_cross": lad["logZ_cross"], "sigma_fold": sel["sigma_fold"], "S_cross_by_marg_set": lad["all_tried"],
            "rank_promotion_arithmetic": lad["steps"],
            "L_zero_shot": lz["L_zero_shot"],
            "chi2_nu_per_suite": x2["per_suite"], "chi2_nu_mean": x2["mean"], "chi2_nu_std": x2["std"],
            "chi2_nu_Cfv_Cdisc_per_suite": x2["per_suite_Cfv_Cdisc"], "presentable": x2["presentable"],
            "n_boxes_scored": nscored,
            "n_boxz_scored": {s: int(b.mask.reshape(len(b.y), len(redshifts), C.N_K)[:, :, 0].sum()) for s, b in blocks.items()},
            "tail_mass": sel.get("tail", {}),
            "mu": dict(zip(folds[C.SUITES[0]].coeff_order, mu_all.tolist())),
            "Sigma_suite_eigenvalues": ev.tolist(),
            "variance_fraction": (ev / ev.sum()).tolist() if ev.sum() > 0 else None,
            "C_finite_volume": "per-suite, 27 CV boxes, shrink 0.10, Hartlap reported not applied",
            "C_discrepancy": {j: float(np.sum(v["cdisc_diag"])) for j, v in fold_par.items()},
            "cfv_assembly": C.CFV_ASSEMBLY, "notes": f"elapsed {time.time()-t0:.0f}s; v2 zfb6, 16 k-bins <= {C.KMAX}",
            "_provenance": prov,
        }
        _upsert(out)
        SCORES.mkdir(parents=True, exist_ok=True)
        tmpf = SCORES / f".{fid}.json.tmp"
        tmpf.write_text(json.dumps({"row": REG._clean(out), "ladder": REG._clean(lad), "L_zero_shot": lz, "folds": fold_par,
                                    "gate0": REG._clean(g0), "chi2": REG._clean(x2), "_provenance": prov}, indent=1, default=str))
        os.replace(tmpf, SCORES / f"{fid}.json")
        _log(fid, f"DONE logZ_cross={lad['logZ_cross']:.2f} {time.time()-t0:.0f}s")
        return fid, "ok"
    except Exception as ex:      # noqa: BLE001 -- record the failure, never delete
        _log(fid, f"ERROR {traceback.format_exc()}")
        _upsert({**base, "notes": f"SCORING ERROR (recorded, not a rejection): {type(ex).__name__}: {ex}",
                 "_provenance": P.provenance("scoring_error", redshifts=list(redshifts))})
        return fid, "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", nargs="+", required=True)
    ap.add_argument("--nproc", type=int, default=1)
    a = ap.parse_args()
    rows = [r for r in REG.frozen_rows() if r["form_id"] in a.forms]
    assert {r["form_id"] for r in rows} == set(a.forms), f"not frozen: {set(a.forms) - {r['form_id'] for r in rows}}"
    print(f"scoring {len(rows)} forms at z={list(C.REDSHIFTS)}", flush=True)
    if a.nproc == 1:                     # in-process, so PK_NPAR can parallelize inside the form
        for r in rows:
            print("%s: %s" % score_one(r), flush=True)
        return
    with Pool(a.nproc, maxtasksperchild=1) as pool:
        for fid, st in pool.imap_unordered(score_one, rows):
            print(f"{fid}: {st}", flush=True)


if __name__ == "__main__":
    main()

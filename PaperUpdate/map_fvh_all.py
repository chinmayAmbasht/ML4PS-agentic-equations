"""Per-box MAP in the Fig. 3 setting ('fvh': Stage-2 fold prior unwidened, likelihood C_fv,j/h) for every ranked form, so the
App. B pool table's L_MAP column matches Table 1 and Fig. 3 (owner 2026-11-19). Same MAP as PaperUpdate/map_sensitivity._boxes.

Per form: rebuild the Stage-2 fold prior on the form's registered grid (41 nodes per nonlinear coefficient, 21 for 3+; the
rule of map_sensitivity / Loop.plot_fits), check it by reproducing the stored zero-shot predictions in PaperUpdate/preds/<key>.npz
(max |diff| recorded as zs_maxdiff), then fit the MAP for every held-out box.
Outputs: PaperUpdate/preds_sens/<key>__fvh.npz (skipped if present: T47/T36/T43/T39/T51 from map_sensitivity) and
PaperUpdate/L_map_fvh.json {key: {per_suite, mean, zs_maxdiff}}. Resumable: keys already in the json are skipped.
Usage: python -m PaperUpdate.map_fvh_all [keys ...]
"""
import json
import os
import sys
import time
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", "ns_fvh"))
os.environ.setdefault("PK_FROZEN", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", "frozen.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.path.insert(0, str(M))

import numpy as np

_argv, sys.argv = sys.argv, sys.argv[:1]          # map_sensitivity reads its group from argv at import
from PaperUpdate.map_sensitivity import _boxes
sys.argv = _argv

OUT = M / "PaperUpdate" / "L_map_fvh.json"
PD = M / "PaperUpdate" / "preds_sens"


def main(keys):
    from Loop.plot_fits import form_of
    from PaperUpdate import Lcorr as LC
    from PaperUpdate.rescore_all import targets
    from Scripts.common import config as C
    from Scripts.engine import chi2 as X2
    from Scripts.engine import evidence as E
    from Scripts.engine import scoring as SC
    from Scripts.engine.par import pmap
    from Scripts.run.score_forms import build_blocks
    T = targets()
    R = json.loads((M / "PaperUpdate" / "ranking.json").read_text())
    order = [r["key"] for r in R if r["key"] in T]
    keys = keys or order
    allr = json.loads(OUT.read_text()) if OUT.exists() else {}
    for key in keys:
        if key in allr:
            continue
        t0 = time.time()
        sub = json.loads(Path(T[key][1]).read_text())
        n = C.NL_GRID_NODES if len(sub["nl"]) <= 2 else 21
        z = np.load(M / "PaperUpdate" / "preds" / f"{key}.npz")
        pf = PD / f"{key}__fvh.npz"
        if pf.exists():
            zm = np.load(pf); mp = {s: zm[f"{s}__map"] for s in C.SUITES}; zsd = None
        else:
            form = form_of(key, sub, n)
            blocks, _, _ = build_blocks(form, C.REDSHIFTS)
            fp = X2.fold_params(SC.cross_suite_folds(form, blocks, C.REDSHIFTS))
            zsd = 0.0
            for s in C.SUITES:
                b, f = blocks[s], fp[s]
                zs = form.predict(b.obs, C.REDSHIFTS, E.to_native(form, f["order"], np.asarray(f["mu"], float)))
                zs = np.where(b.mask, zs, np.nan); ref = z[f"{s}__zs"]
                ok = np.isfinite(zs) & np.isfinite(ref)
                zsd = max(zsd, float(np.max(np.abs(zs[ok] - ref[ok]))) if ok.any() else np.inf)
            out = dict(zip(C.SUITES, pmap(_boxes, [(form, fp[s], blocks[s], s, 1.00, "fv/h") for s in C.SUITES])))
            mp = {s: out[s][0] for s in C.SUITES}
            np.savez_compressed(pf, **{f"{s}__map": mp[s] for s in C.SUITES})
        ps = {s: LC.L(mp[s], z[f"{s}__y"], s) for s in C.SUITES}
        allr = json.loads(OUT.read_text()) if OUT.exists() else {}
        allr[key] = {"per_suite": ps, "mean": float(np.mean(list(ps.values()))), "zs_maxdiff": zsd, "nodes": n}
        OUT.write_text(json.dumps(allr, indent=1))
        print(f"{key} L_map_fvh={allr[key]['mean']:.2f} zs_maxdiff={zsd} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

"""Per-box MAP L for each arm's best form (Plans/05_ablation.md), with the Fig. 3 setting: prior widening 1.00, likelihood
C_fv,j/h ('fvh' of PaperUpdate/map_sensitivity.py), scored with the correlated L (PaperUpdate/Lcorr.L). Same Stage-2 fold prior
and Nelder-Mead as map_sensitivity._boxes. Output: Ablation/winners_map.json (merged per key). Reads only.
Usage: python -m Ablation.winners_map
"""
import json
import os
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_abl", f"ns_winners"))
os.environ.setdefault("PK_FROZEN", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_abl", f"frozen_winners.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.argv = sys.argv[:1] + ["loop"]          # map_sensitivity reads a group from argv at import; unused here
sys.path.insert(0, str(M))

import numpy as np

OUT = M / "Ablation" / "winners_map.json"


def main():
    from Ablation.analyze import arm_best
    from Loop.plot_fits import form_of
    from PaperUpdate import Lcorr as LC
    from PaperUpdate.map_sensitivity import _boxes
    from Scripts.common import config as C
    from Scripts.engine import chi2 as X2
    from Scripts.engine import scoring as SC
    from Scripts.engine.par import pmap
    from Scripts.run.score_forms import build_blocks
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = dict(arm_best())
    pc = json.loads((M / "PySR" / "summary.json").read_text())["C"]["submission"]     # PySR C MSE pick (Plan 04), for the table
    todo["PySR C"] = {k: pc[k] for k in ("expr", "lin", "nl", "features")}
    for key, t in todo.items():
        sub = {k: t[k] for k in ("expr", "lin", "nl", "features")}
        sub["roles"] = {c: "other" for c in list(t["lin"]) + list(t["nl"])}
        if res.get(key, {}).get("expr") == t["expr"]:
            continue
        form = form_of(f"W_{key}", sub, C.NL_GRID_NODES if len(sub["nl"]) <= 2 else 21)
        blocks, _, _ = build_blocks(form, C.REDSHIFTS)
        fp = X2.fold_params(SC.cross_suite_folds(form, blocks, C.REDSHIFTS))
        out = dict(zip(C.SUITES, pmap(_boxes, [(form, fp[s], blocks[s], s, 1.00, "fv/h") for s in C.SUITES])))
        per = {s: LC.L(out[s][0], out[s][1], s) for s in C.SUITES}
        res[key] = {"expr": t["expr"], "per_suite": per, "mean": float(np.mean(list(per.values())))}
        OUT.write_text(json.dumps(res, indent=1))
        print(key, round(res[key]["mean"], 2), flush=True)


if __name__ == "__main__":
    main()

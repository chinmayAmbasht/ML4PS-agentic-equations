"""van Daalen+20 (registered Base form R32, 4 nonlinear + amplitude A) through the same held-out pipeline as the analytic
forms: Stage-2 fold prior from the 3 training suites, zero-shot at the fold mean, per-box MAP (prior x1.30, C_fv + C_disc),
Loop.plot_fits.fit. The form fails Gate 0 (physical checks), which does not block this diagnostic fit. Owner 2026-11-13.
Output: PaperUpdate/preds/VD.npz + PaperUpdate/vd_pipeline.json (fold prior means / widths).
Usage: python -m PaperUpdate.vd_pipeline [n_nodes]
"""
import json
import os
import sys
import time
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns_VD"))
os.environ.setdefault("PK_FROZEN", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"frozen.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.path.insert(0, str(M))

import numpy as np


def main(n):
    from Loop.plot_fits import form_of
    from Scripts.analysis.plot_trial import _boxes
    from Scripts.common import config as C
    from Scripts.engine import chi2 as X2
    from Scripts.engine import scoring as SC
    from Scripts.engine.par import pmap
    from Scripts.run.score_forms import build_blocks
    t0 = time.time()
    sub = json.loads((M / "Base" / "sub" / "R32_vanDaalen.json").read_text())
    form = form_of("VD", sub, n)
    blocks, _, _ = build_blocks(form, C.REDSHIFTS)
    fp = X2.fold_params(SC.cross_suite_folds(form, blocks, C.REDSHIFTS))
    print("folds done", round(time.time() - t0), "s", flush=True)
    res = dict(zip(C.SUITES, pmap(_boxes, [(form, fp[j], blocks[j]) for j in C.SUITES])))
    np.savez_compressed(M / "PaperUpdate" / "preds" / "VD.npz",
                        **{f"{j}__{q}": res[j][q] for j in res for q in ("y", "zs", "map", "chi2_box")})
    info = {j: {"order": list(fp[j]["order"]), "mu": np.asarray(fp[j]["mu"]).tolist(),
                "sd": np.sqrt(np.diag(np.asarray(fp[j]["Sigma"]))).tolist()} for j in C.SUITES}
    (M / "PaperUpdate" / "vd_pipeline.json").write_text(json.dumps({"n_nodes": n, "folds": info}, indent=1))
    print(json.dumps(info, indent=1)); print("total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)

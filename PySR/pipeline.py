"""PySR benchmark assembly (Plans/04_pysr.md), after the 10 searches:
 1. pooled-search equation -> engine submission (PySR/to_form.py; constants -> coefficients, influence from the data);
 2. full harness in base mode into PySR/harness (Gate 0, Stage 1/2, log Z_cross + ladder, chi2, L);
 3. held-out per-box fit (Loop.plot_fits.fit) -> PaperUpdate/preds/PYSR_<v>.npz; correlated L zero-shot (Stage-2 prior mean) and MAP;
 4. within / pooled / population-transfer refits of that form (PaperUpdate.Benchmark.bench);
 5. LOSO-search zero-shot (the PySR equation and constants found on 3 suites, applied to the 4th) from the fold runs;
 -> PaperUpdate/Benchmark/bench_corr.json entries "PySR (unconstrained)" / "PySR (constrained)", PySR/summary.json.
Usage: python -m PySR.pipeline U C
"""
import json
import os
import subprocess
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns_pysr_pipe"))
os.environ.setdefault("PK_FROZEN", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"frozen.jsonl"))
os.environ.setdefault("PK_NPAR", "6")
sys.path.insert(0, str(M))

import numpy as np
import yaml

from PaperUpdate import Lcorr as LC
from PySR.run_search import suite_rows
from PySR.to_form import to_submission
from Scripts.common import config as C

NAME = {"U": "PySR (unconstrained)", "C": "PySR (constrained)"}
RUNS = M / "PySR" / "runs"
HARN = M / "PySR" / "harness"


def build_sub(v):
    res = json.loads((RUNS / f"{v}_pooled" / "result.json").read_text())
    Xs = np.concatenate([suite_rows(s)[0] for s in C.SUITES])
    idx = np.random.default_rng(0).choice(len(Xs), 6000, replace=False)
    X = Xs[idx]
    data = [X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]]      # k, D, q, Om, s8, wb
    sub = to_submission(v, res["best_equation"], data=data)
    p = RUNS / f"{v}_pooled" / "submission.json"
    p.write_text(json.dumps(sub, indent=1))
    return p, sub


def harness(v, p):
    HARN.mkdir(exist_ok=True)
    env = dict(os.environ, PK_ML4PS_BASE="1", PK_ML4PS_LOOP_DIR=str(HARN), PK_ML4PS_BUDGET="50")
    out = subprocess.run([sys.executable, "-m", "Loop.evaluate", str(p)], cwd=M, env=env, capture_output=True, text=True)
    tail = (out.stdout + out.stderr).strip().splitlines()[-3:]
    st = yaml.safe_load((HARN / "trial_state.yaml").read_text())["trials"][-1]
    return st, tail


def pipeline_fit(v, sub):
    from Loop.plot_fits import fit
    n = C.NL_GRID_NODES if len(sub["nl"]) <= 2 else 21
    _, res = fit(f"PYSR_{v}", sub, n)
    key = f"PYSR_{v}"
    np.savez_compressed(M / "PaperUpdate" / "preds" / f"{key}.npz",
                        **{f"{j}__{q}": res[j][q] for j in res for q in ("y", "zs", "map", "chi2_box")})
    zs = {s: LC.L(res[s]["zs"], res[s]["y"], s) for s in C.SUITES}
    mp = {s: LC.L(res[s]["map"], res[s]["y"], s) for s in C.SUITES}
    return key, zs, mp


def main(variants):
    from PaperUpdate.Benchmark import bench as BN
    D = BN.data()
    bf = BN.OUT
    B = json.loads(bf.read_text())
    summ = json.loads((M / "PySR" / "summary.json").read_text()) if (M / "PySR" / "summary.json").exists() else {}
    for v in variants:
        p, sub = build_sub(v)
        st, tail = harness(v, p)
        print(v, "harness:", st.get("status"), st.get("logZ_cross"), st.get("marginalized"), tail[-1:], flush=True)
        key, zs_pipe, mp = pipeline_fit(v, sub)
        f = BN.form_fit_factory(sub)
        r = {"within": BN.summarize(BN.run_within(f, D)), "pooled": BN.summarize(BN.run_pooled(f, D)),
             "transfer_population": BN.summarize(BN.run_transfer(f, D))}
        loso = {s: json.loads((RUNS / f"{v}_{s}" / "result.json").read_text())["L_zero_corr"] for s in C.SUITES}
        r["transfer_zero_shot"] = BN.summarize(loso)                 # the PySR search itself, trained on 3 suites
        r["transfer_pipeline_zero_shot"] = BN.summarize(zs_pipe)     # Stage-2 prior mean of the pooled form
        r["transfer_map"] = BN.summarize(mp)
        r["pipeline_key"] = key
        B[NAME[v]] = r
        summ[v] = {"submission": sub, "harness": {k: st.get(k) for k in ("status", "logZ_cross", "marginalized", "sigma_fold",
                                                                         "chi2_mean", "gate0", "L_zero_shot", "L_map")},
                   "logZ_box": (st["logZ_cross"] / 3726) if st.get("logZ_cross") else None,
                   "bench": {k: v2["mean"] for k, v2 in r.items() if isinstance(v2, dict)},
                   "loso": {s: json.loads((RUNS / f"{v}_{s}" / "result.json").read_text()) for s in C.SUITES},
                   "pooled": json.loads((RUNS / f"{v}_pooled" / "result.json").read_text())}
        bf.write_text(json.dumps(B, indent=1))
        (M / "PySR" / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
        print(v, {k: round(x["mean"], 1) for k, x in r.items() if isinstance(x, dict)}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["U", "C"])

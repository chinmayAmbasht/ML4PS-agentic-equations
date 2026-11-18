"""A3 prescreen (Plans/05_ablation.md): cheap proxy for every grammar form, mirroring the agents' off-budget dry runs.
Proxy = log Z_cross with R = {} (no ladder) on an 11-node nonlinear grid, using the harness's own LOSO folds (Gate 0 G0_1-3 first).
This is only an ordering; the 11-node grid is known to misrank near-ties, so the top 60 are re-scored by Loop.evaluate (41 nodes).
Usage: python -m Ablation.prescreen [n_workers]   -> Ablation/A3/prescreen.jsonl (resumable)
"""
import json
import os
import sys
import time
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_abl", f"ns_prescreen"))
os.environ.setdefault("PK_FROZEN", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_abl", f"frozen_prescreen.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.path.insert(0, str(M))

NN = int(os.environ.get("ABL_NN", "11"))
GR = M / "Ablation" / "A3" / "grammar.jsonl"
OUT = M / "Ablation" / "A3" / "prescreen.jsonl"


def proxy(row):
    from Scripts.common import config as C
    from Scripts.engine import gate0 as G0
    from Scripts.engine import scoring as SC
    from Scripts.engine.forms import Form
    from Scripts.run.score_forms import build_blocks
    sub = row["sub"]; t0 = time.time()
    nl = {k: {**v, "n_nodes": NN} for k, v in sub["nl"].items()}
    f = Form(f"{row['gid']}_n{NN}", sub["expr"], list(sub["lin"]), nl, dict(sub["roles"]), list(sub["features"]))
    g0 = G0.check(f, redshifts=C.REDSHIFTS)
    rec = {"gid": row["gid"], "gate0": {k: bool(g0[k]) for k in ("G0_1", "G0_2", "G0_3")}}
    if all(rec["gate0"].values()):
        bl, _, _ = build_blocks(f, C.REDSHIFTS)
        folds = SC.cross_suite_folds(f, bl, C.REDSHIFTS)
        r = SC.logZ_cross(f, folds, bl, [], C.REDSHIFTS)
        rec.update({"proxy_logZ0": r["logZ_cross"], "sigma_fold": r["sigma_fold"]})
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    rows = [json.loads(l) for l in GR.read_text().splitlines()]
    done = {json.loads(l)["gid"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    only = sys.argv[1:]
    for row in rows:
        if row["gid"] in done or (only and row["gid"] not in only):
            continue
        try:
            rec = proxy(row)
        except Exception as ex:        # noqa: BLE001 -- recorded, the form is dropped from the ordering
            rec = {"gid": row["gid"], "error": f"{type(ex).__name__}: {ex}"[:300]}
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec, flush=True)


if __name__ == "__main__":
    main()

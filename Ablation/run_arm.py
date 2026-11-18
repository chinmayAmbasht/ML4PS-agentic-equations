"""Full scoring of an ablation arm through the unchanged loop harness (Plans/05_ablation.md).
Each manifest entry -> `python -m Loop.evaluate <sub>` with PK_ML4PS_LOOP_DIR=Ablation/<arm>/harness (own trial ids, lock, state),
own PK_NS_DIR / PK_FROZEN under $ML4PS_CACHE/ml4ps_abl, loop rule (no base mode: 1-2 nonlinear coefficients, 41 nodes).
Resumable: an entry whose hypothesis tag is already in the arm's trial_state.yaml is skipped. Loop/ is never written.
  A1, A2: Ablation/<arm>/manifest.jsonl in complexity order (from Ablation/sr_hof.py)
  A3    : the top N (default 60) of Ablation/A3/prescreen.jsonl by proxy, best first (submissions from grammar.jsonl)
Usage: python -m Ablation.run_arm A1|A2|A3 [N]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

M = Path(__file__).resolve().parents[1]
# Loop.evaluate unchanged; this only lets yaml.safe_dump write numpy floats (log Z of forms with no linear coefficient is
# np.float64, which crashed the dump after scoring and left the trial id frozen).
SHIM = ("import sys, runpy, yaml, numpy as np; "
        "yaml.SafeDumper.add_representer(np.float64, lambda d, x: d.represent_float(float(x))); "
        "sys.argv = ['Loop.evaluate', sys.argv[1]]; runpy.run_module('Loop.evaluate', run_name='__main__')")


def entries(arm, n):
    d = M / "Ablation" / arm
    if arm != "A3":
        for r in (json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines()):
            if "sub" in r:
                yield r["cid"], M / r["sub"]
        return
    G = {r["gid"]: r for r in (json.loads(l) for l in (d / "grammar.jsonl").read_text().splitlines())}
    P = [json.loads(l) for l in (d / "prescreen.jsonl").read_text().splitlines()]
    P = sorted((p for p in P if "proxy_logZ0" in p), key=lambda p: -p["proxy_logZ0"])[:n]
    (d / "subs").mkdir(exist_ok=True)
    for rank, p in enumerate(P, 1):
        sub = dict(G[p["gid"]]["sub"])
        sub["hypothesis"] = f"[ablation A3 {p['gid']}] prescreen rank {rank}: " + sub["hypothesis"]
        f = d / "subs" / f"{p['gid']}.json"
        f.write_text(json.dumps(sub, indent=1))
        yield p["gid"], f


def main(arm, n=60):
    harn = M / "Ablation" / arm / "harness"
    harn.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PK_ML4PS_LOOP_DIR=str(harn), PK_ML4PS_BUDGET="1000",
               PK_NS_DIR=os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_abl", f"ns_{arm}"), PK_FROZEN=os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_abl", f"frozen_{arm}.jsonl"))
    env.setdefault("PK_NPAR", "4")
    env.pop("PK_ML4PS_BASE", None)
    st = harn / "trial_state.yaml"
    for eid, sub in entries(arm, n):
        done = (yaml.safe_load(st.read_text()) or {}).get("trials", []) if st.exists() else []
        tag = f"[ablation {arm} {eid}]"
        if any(tag in (t.get("hypothesis") or "") for t in done):
            continue
        out = subprocess.run([sys.executable, "-c", SHIM, str(sub)], cwd=M, env=env, capture_output=True, text=True)
        tail = (out.stdout + out.stderr).strip().splitlines()[-1:] or [""]
        print(arm, eid, tail[0][:300], flush=True)
        now = (yaml.safe_load(st.read_text()) or {}).get("trials", []) if st.exists() else []
        if len(now) == len(done):                     # harness error: trial not recorded -> release its frozen id
            fz = Path(env["PK_FROZEN"]); tid = f"T{len(done) + 1:02d}"
            if fz.exists():
                keep = [l for l in fz.read_text().splitlines(True) if json.loads(l)["record"]["form_id"] != tid]
                fz.write_text("".join(keep))
            print(arm, eid, f"NOT RECORDED; released frozen id {tid}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 60)

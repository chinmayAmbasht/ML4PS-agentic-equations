"""Refit all 64 k <= 10 forms of SANDBOX-ML4PS (Base 4 + Loop 50 + Chi2Push/full16 10) on the registered 16-bin grid and save
their held-out per-box predictions (LOSO; zero-shot at the fold mean, per-box MAP under prior x1.30, the Loop.plot_fits fit),
then evaluate L_zero / L_map with three sigma_k: paper, inflated (2026-11-11) and Hartlap (2026-11-13). Reporting only.

Check: L with the sigma_k the form was originally scored with must equal the stored L (paper for Base/Loop, inflated for
Chi2Push).
Outputs: PaperUpdate/preds/<key>.npz, PaperUpdate/L_all.jsonl (appended, existing keys skipped).
Usage: python -m PaperUpdate.rescore_all [keys... | all] [--shard i/n]
"""
import glob
import json
import os
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns_PU"))
os.environ.setdefault("PK_FROZEN", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"frozen.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.path.insert(0, str(M))

import numpy as np
import yaml

OUT = M / "PaperUpdate" / "L_all.jsonl"
PRED = M / "PaperUpdate" / "preds"
BASE = {"T05": "Base/sub/R01_two_scale.json", "T02": "Base/sub/R02_k0_of_D.json", "T01": "Base/sub/R03_k0_Om.json",
        "T03": "Base/sub/R12_two_amp.json"}
BASE_KEY = {"T05": "P1", "T02": "P2", "T01": "P3", "T03": "P12"}


def _match(trials, files):
    subs = {}
    ds = [(f, json.loads(Path(f).read_text())) for f in files]
    for t in trials:
        for f, d in ds:
            if d["expr"] == t["expr"] and d.get("hypothesis", "") == t.get("hypothesis", ""):
                subs[t["trial"]] = f
                break
        else:
            for f, d in ds:
                if d["expr"] == t["expr"]:
                    subs[t["trial"]] = f
                    break
    return subs


def targets():
    """key -> (source, submission path, stored trial record, sigma_k the stored L used)."""
    out = {}
    L = yaml.safe_load((M / "Loop" / "trial_state.yaml").read_text())["trials"]
    s = _match(L, sorted(glob.glob(str(M / "Loop" / "submissions" / "*.json"))))
    for t in L:
        if t.get("status") == "scored":
            orch = t.get("hypothesis", "").startswith("[orchestrator-proposed")
            out[t["trial"]] = ("Loop (orchestrator)" if orch else "Loop", s[t["trial"]], t, "inflated" if orch else "paper")
    B = {t["trial"]: t for t in yaml.safe_load((M / "Base" / "trial_state.yaml").read_text())["trials"]}
    for tid, p in BASE.items():
        out[BASE_KEY[tid]] = ("Base", str(M / p), B[tid], "paper")
    if any(t["trial"] == "T51" for t in L):          # Chi2Push X01-X10 resubmitted as main-loop T51-T60 (owner 2026-11-14)
        return out
    X = yaml.safe_load((M / "Chi2Push" / "full16" / "trial_state.yaml").read_text())["trials"]
    s = _match(X, sorted(glob.glob(str(M / "Chi2Push" / "sub" / "X*.json"))))
    for t in X:
        out["X" + t["trial"][1:]] = ("Chi2Push/full16", s[t["trial"]], t, "inflated")
    return out


def L_of(res, sk):
    lz, lm = {}, {}
    for j, d in res.items():
        m = np.isfinite(d["y"]).all(axis=1)
        lz[j] = float(np.mean(np.sum(((d["zs"][m] - d["y"][m]) / sk) ** 2, axis=1)))
        lm[j] = float(np.mean(np.sum(((d["map"][m] - d["y"][m]) / sk) ** 2, axis=1)))
    return {"L_zero": float(np.mean(list(lz.values()))), "L_map": float(np.mean(list(lm.values()))),
            "L_zero_per_suite": lz, "L_map_per_suite": lm}


def SIG():
    from Scripts.common import config as C
    return {"paper": np.asarray(C.SIGMA_K_PAPER),
            "inflated": np.asarray(json.loads((M / "Artifacts" / "sigma_k_inflated.json").read_text())["sigma_k_inflated"]),
            "hartlap": np.asarray(json.loads((M / "PaperUpdate" / "sigma_k_hartlap.json").read_text())["sigma_k_hartlap"])}


def main(argv):
    from Loop.plot_fits import fit
    from Scripts.common import config as C
    shard = None
    if "--shard" in argv:
        i = argv.index("--shard"); a, b = argv[i + 1].split("/"); shard = (int(a), int(b)); argv = argv[:i] + argv[i + 2:]
    T = targets()
    want = list(T) if argv in ([], ["all"]) else argv
    if shard:
        want = [k for n, k in enumerate(want) if n % shard[1] == shard[0]]
    PRED.mkdir(exist_ok=True)
    sig = SIG()
    for key in want:
        done = {json.loads(l)["key"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
        if key in done:
            continue
        src, path, st, used = T[key]
        sub = json.loads(Path(path).read_text())
        n = C.NL_GRID_NODES if len(sub["nl"]) <= 2 else 21
        _, res = fit(key, sub, n)
        np.savez_compressed(PRED / f"{key}.npz", **{f"{j}__{q}": res[j][q] for j in res for q in ("y", "zs", "map", "chi2_box")})
        Ls = {name: L_of(res, s) for name, s in sig.items()}
        ref = Ls[used]
        ok = (abs(ref["L_map"] - st["L_map"]) < 1e-2 * max(1.0, st["L_map"])
              and abs(ref["L_zero"] - st["L_zero_shot"]) < 1e-2 * max(1.0, st["L_zero_shot"]))
        rec = {"key": key, "source": src, "trial": st["trial"], "submission": str(Path(path).relative_to(M)), "nl_nodes": n,
               "stored_sigma": used, "check_matches_stored": bool(ok),
               "stored": {"L_zero": st["L_zero_shot"], "L_map": st["L_map"]}, **{f"{k}_sigma": v for k, v in Ls.items()}}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{key}: Lz {ref['L_zero']:.2f}/{st['L_zero_shot']:.2f} Lm {ref['L_map']:.2f}/{st['L_map']:.2f} ok={ok} | "
              f"hartlap Lz {Ls['hartlap']['L_zero']:.2f} Lm {Ls['hartlap']['L_map']:.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

"""Re-score the benchmark loss L (L_zero, L_map) of the main-loop trials and the base forms with the over-estimated sigma_k
(Artifacts/sigma_k_inflated.json; owner 2026-11-11). Reporting only: nothing is written to trial_state.yaml, and log Z_cross,
the ladder and chi^2 do not depend on sigma_k.

The per-box predictions are recomputed once per form (fold prior from the other 3 suites; zero-shot at the fold mean, per-box MAP
under prior x1.30 and C_fv + C_disc, via Scripts.analysis.plot_trial._boxes, the same fit as L_map). Both L's are then evaluated
with the paper's sigma_k (a check: must equal the stored L) and with the inflated sigma_k.

Output: Loop/L_inflated.jsonl, one line per form (appended; forms already present are skipped).
Usage:  python -m Loop.rescore_L [T47 T36 ... | all]
"""
import glob
import json
import os
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns_L"))
os.environ.setdefault("PK_FROZEN", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"frozen.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.path.insert(0, str(M))

import numpy as np
import yaml

from Loop.plot_fits import fit
from Scripts.common import config as C

OUT = M / "Loop" / "L_inflated.jsonl"
BASE = {"T05": ("paper rank 1", "Base/sub/R01_two_scale.json"), "T02": ("paper rank 2", "Base/sub/R02_k0_of_D.json"),
        "T01": ("paper rank 3", "Base/sub/R03_k0_Om.json"), "T03": ("paper rank 12", "Base/sub/R12_two_amp.json")}


def targets():
    S = {t["trial"]: t for t in yaml.safe_load((M / "Loop" / "trial_state.yaml").read_text())["trials"]}
    B = {t["trial"]: t for t in yaml.safe_load((M / "Base" / "trial_state.yaml").read_text())["trials"]}
    subs = {}
    for f in sorted(glob.glob(str(M / "Loop" / "submissions" / "*.json"))):
        d = json.loads(Path(f).read_text())
        for k, t in S.items():
            if k not in subs and d["expr"] == t["expr"] and d.get("hypothesis", "") == t["hypothesis"]:
                subs[k] = f
    out = {k: (k, subs[k], S[k]) for k in S}
    out.update({name: (name, str(M / p), B[k]) for k, (name, p) in BASE.items()})
    return out


def L_of(res, sk):
    lz, lm = {}, {}
    for j, d in res.items():
        m = np.isfinite(d["y"]).all(axis=1)
        lz[j] = float(np.mean(np.sum(((d["zs"][m] - d["y"][m]) / sk) ** 2, axis=1)))
        lm[j] = float(np.mean(np.sum(((d["map"][m] - d["y"][m]) / sk) ** 2, axis=1)))
    return {"L_zero": float(np.mean(list(lz.values()))), "L_map": float(np.mean(list(lm.values()))),
            "L_zero_per_suite": lz, "L_map_per_suite": lm}


def main(argv):
    T = targets()
    want = list(T) if argv in ([], ["all"]) else argv
    done = {json.loads(l)["label"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    skp, ski = np.asarray(C.SIGMA_K_PAPER), np.asarray(json.loads((M / "Artifacts" / "sigma_k_inflated.json").read_text())["sigma_k_inflated"])
    for key in want:
        label, path, st = T[key]
        if label in done:
            continue
        sub = json.loads(Path(path).read_text())
        n = C.NL_GRID_NODES if len(sub["nl"]) <= 2 else 21
        _, res = fit(label, sub, n)
        p, i = L_of(res, skp), L_of(res, ski)
        ok = abs(p["L_map"] - st["L_map"]) < 1e-2 * max(1.0, st["L_map"]) and abs(p["L_zero"] - st["L_zero_shot"]) < 1e-2 * max(1.0, st["L_zero_shot"])
        rec = {"label": label, "submission": str(Path(path).relative_to(M)), "nl_nodes": n, "check_paper_matches_stored": bool(ok),
               "stored": {"L_zero": st["L_zero_shot"], "L_map": st["L_map"]}, "paper_sigma": p, "inflated_sigma": i}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{label}: L_zero {p['L_zero']:.2f} -> {i['L_zero']:.2f}   L_map {p['L_map']:.2f} -> {i['L_map']:.2f}   "
              f"stored match {ok}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

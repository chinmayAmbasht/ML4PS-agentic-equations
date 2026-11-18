"""Correlated L (PaperUpdate/Lcorr.py) for every form whose held-out predictions are saved in PaperUpdate/preds/<key>.npz.
Zero-shot and per-box MAP; per held-out suite and mean. No refit. Output: PaperUpdate/L_corr.json (rewritten).
Usage: python -m PaperUpdate.Lcorr_all
"""
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M))

import numpy as np

from PaperUpdate import Lcorr as LC
from Scripts.common import config as C

OUT = M / "PaperUpdate" / "L_corr.json"


def main():
    res = {}
    for p in sorted((M / "PaperUpdate" / "preds").glob("*.npz")):
        z = np.load(p)
        lz = {s: LC.L(z[f"{s}__zs"], z[f"{s}__y"], s) for s in C.SUITES}
        lm = {s: LC.L(z[f"{s}__map"], z[f"{s}__y"], s) for s in C.SUITES}
        res[p.stem] = {"L_zero": float(np.mean(list(lz.values()))), "L_map": float(np.mean(list(lm.values()))),
                       "L_zero_per_suite": lz, "L_map_per_suite": lm}
    OUT.write_text(json.dumps({"_meta": {"loss": "dS^T (C_fv,j/h)^-1 dS, own-suite C_fv (shrink 0.10), h=%.4f" % LC.H}, **res}, indent=1))
    for k, v in res.items():
        print(k, round(v["L_zero"], 2), round(v["L_map"], 2))


if __name__ == "__main__":
    main()

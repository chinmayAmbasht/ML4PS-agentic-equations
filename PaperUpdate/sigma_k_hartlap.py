"""Hartlap-inflated sigma_k for the benchmark loss L (owner 2026-11-13, Plans/03_rank_and_figures.md). Reporting only.

    sigma_k^H(k) = max_j sqrt( diag C_fv,j(k,k) / h ),   h = (N_cv - N_k - 2) / (N_cv - 1) = (27 - 16 - 2) / 26

j runs over the 4 suites; C_fv,j is suite j's own CV covariance of S (as used by the engine, blockdiag, shrink 0.10 -- the
diagonal is unaffected by the shrinkage). No C_disc, no paper floor. C_fv does not depend on the form.

Output: PaperUpdate/sigma_k_hartlap.json.   Usage: python -m PaperUpdate.sigma_k_hartlap
"""
import json
import os
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns"))
os.environ.setdefault("PK_FROZEN", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"frozen.jsonl"))
os.environ.setdefault("PK_NPAR", "4")
sys.path.insert(0, str(M))

import numpy as np

from Loop.plot_fits import form_of
from Scripts.common import config as C
from Scripts.run.score_forms import build_blocks

N_CV = 27
REF = "Base/sub/R03_k0_Om.json"
OUT = M / "PaperUpdate" / "sigma_k_hartlap.json"


def main():
    nk = len(C.K_GRID)
    h = (N_CV - nk - 2) / (N_CV - 1)
    blocks, _, _ = build_blocks(form_of("ref", json.loads((M / REF).read_text()), 11), C.REDSHIFTS)
    sd = {j: np.sqrt(np.diag(np.asarray(blocks[j].cfv)) / h) for j in C.SUITES}
    A = np.array([sd[j] for j in C.SUITES])
    sk = A.max(axis=0)
    paper = np.asarray(C.SIGMA_K_PAPER)
    infl = np.asarray(json.loads((M / "Artifacts" / "sigma_k_inflated.json").read_text())["sigma_k_inflated"])
    rec = {"k": list(C.K_GRID), "hartlap_h": h, "n_cv": N_CV, "n_k": nk, "sigma_k_hartlap": sk.round(6).tolist(),
           "per_suite": {j: sd[j].round(6).tolist() for j in C.SUITES},
           "envelope_suite": [C.SUITES[i] for i in A.argmax(axis=0)],
           "ratio_to_paper": (sk / paper).round(3).tolist(), "ratio_to_inflated": (sk / infl).round(3).tolist(),
           "rule": "max_j sqrt(diag C_fv,j / h); h = (27-16-2)/26; no C_disc, no paper floor"}
    OUT.write_text(json.dumps(rec, indent=1))
    for i, kk in enumerate(C.K_GRID):
        print(f"k={kk:6.3f}  sigma_H={sk[i]:.5f}  paper={paper[i]:.5f}  x{sk[i]/paper[i]:.2f}  infl x{sk[i]/infl[i]:.2f}  {rec['envelope_suite'][i]}")


if __name__ == "__main__":
    main()

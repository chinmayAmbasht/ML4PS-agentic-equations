"""Over-estimated sigma_k for the benchmark loss L (owner 2026-11-11): the paper's sigma_k is smaller than sqrt(diag C_fv) and
sqrt(diag C_disc) at k <~ 6 (Figures/diagnostics/fig_cov_ratio_diag), so L over-weights the large scales. Reporting only:
sigma_k never enters log Z_cross, the ladder, chi^2 or any per-box fit.

    sigma_k^infl(k) = max( sigma_k^paper(k),  max_j sqrt( C_fv,j(k,k) + C_disc,j(k,k) ) )

j runs over the 4 held-out suites (C_fv,j is suite j's own CV covariance, C_disc,j the fold's discrepancy). C_disc is form
dependent; it is taken from paper rank 1 (Base/sub/R01_two_scale.json), a pre-loop form whose C_disc is the largest of the
forms checked (rank 1, T36, T39), so the envelope does not depend on the loop's results and bounds both errors for those forms.

Output: Artifacts/sigma_k_inflated.json (read by Scripts.common.config).
Usage:  python -m Scripts.analysis.sigma_k_inflated
"""
import json
import os
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[2]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns"))
sys.path.insert(0, str(M))

import numpy as np

from Scripts.common import config as C

REF = "Base/sub/R01_two_scale.json"
OUT = M / "Artifacts" / "sigma_k_inflated.json"


def main():
    from Loop.plot_cov_ratio import covs
    cv = covs("rank1", json.loads((M / REF).read_text()))
    tot = np.array([np.sqrt(np.diag(cf) + np.diag(cd)) for cf, cd in cv.values()])
    paper = np.asarray(C.SIGMA_K_PAPER)
    infl = np.maximum(paper, tot.max(axis=0))
    rec = {"k": list(C.K_GRID), "sigma_k_paper": paper.tolist(), "sigma_k_inflated": infl.round(6).tolist(),
           "ratio_to_paper": (infl / paper).round(3).tolist(),
           "envelope_suite": [list(cv)[i] for i in tot.argmax(axis=0)],
           "rule": "max(sigma_k_paper, max_j sqrt(diag C_fv,j + diag C_disc,j)); C_disc from paper rank 1 folds",
           "reference_form": REF}
    OUT.write_text(json.dumps(rec, indent=1))
    for k, p, s, r, j in zip(C.K_GRID, paper, infl, rec["ratio_to_paper"], rec["envelope_suite"]):
        print(f"k {k:6.3f}  paper {p:.5f}  inflated {s:.5f}  x{r:.2f}  ({j if s > p else 'paper'})")


if __name__ == "__main__":
    main()

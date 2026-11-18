"""ML4PS G0_U calibration at z = 0 on the current K_GRID (Plans/01_ml4ps_k10.md): pooled LH boxes of all 4 suites, end step
S(k_last) - S(k_prev), quantiles -> Artifacts/upturn_calib_z0.json (read by Scripts/engine/g0u_z0.py; band = [1 %, 99 %]).

Usage:  python -m Scripts.analysis.upturn_calib_z0
"""
import json

import numpy as np

from ..common import config as C
from ..data import suppression_io as IO

Q = [0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995]


def main():
    steps, per = [], {}
    for s in C.SUITES:
        sd = IO.load(s, "LH", redshifts=(0.0,))
        y = np.where(sd.mask, sd.y, np.nan)
        st = y[:, -1] - y[:, -2]; st = st[np.isfinite(st)]
        steps.append(st); per[s] = {"n_box": int(st.size), "frac_nonneg": round(float(np.mean(st >= 0)), 3),
                                    "median": round(float(np.median(st)), 5)}
    st = np.concatenate(steps)
    r = {"k_grid": list(C.K_GRID), "redshift": 0.0, "q": Q, "end_step_quantiles": np.quantile(st, Q).tolist(), "n_box": int(st.size),
         "per_suite": per, "note": f"pooled LH boxes of all 4 suites; end step S({C.K_GRID[-1]:.2f})-S({C.K_GRID[-2]:.2f})"}
    (C.ARTIFACTS / "upturn_calib_z0.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "k_grid"}, indent=1))


if __name__ == "__main__":
    main()

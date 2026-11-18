"""G0_U calibration from the simulation data only (no forms). Per suite and z (LH, valid blocks), on the registered k-grid:
the median S(k) curve, where the trough sits, the rise S(k_max) - min_{k >= k*} S and the end step S(k_max) - S(k_prev)
across boxes. Used to propose k*, z_split, Delta_hi and f_pass for owner approval (Plans/00 sec. 6).

Usage:  python -m Scripts.analysis.upturn_calib [--kstar 3.89]          (PK_KMAX=20 shows the uncut grid for context)
"""
import argparse
import json

import numpy as np

from ..common import config as C
from ..data import suppression_io as IO


def measure(kstar):
    K = np.asarray(C.K_GRID)
    sel = K >= kstar
    out = {}
    for s in C.SUITES:
        d = IO.load(s, "LH")
        Y = d.y.reshape(len(d.y), C.N_Z, C.N_K); M = d.mask.reshape(len(d.y), C.N_Z, C.N_K)[:, :, 0]
        for iz, z in enumerate(C.REDSHIFTS):
            S = Y[M[:, iz], iz, :]
            med = np.median(S, axis=0)
            Ss = S[:, sel]
            rise = Ss[:, -1] - Ss.min(axis=1)
            step = Ss[:, -1] - Ss[:, -2]
            q = lambda v: [float(np.quantile(v, p)) for p in (0.16, 0.5, 0.84)]
            out[f"{s}|{z:.2f}"] = {
                "n_box": int(len(S)), "median_curve": med.tolist(),
                "k_trough_median_curve": float(K[int(np.argmin(med))]),
                "frac_trough_inside": float(np.mean(np.argmin(S, axis=1) < len(K) - 1)),
                "rise_q16_50_84": q(rise), "end_step_q16_50_84": q(step),
                "frac_end_step_ge0": float(np.mean(step >= 0))}
    return {"k_grid": K.tolist(), "k_star": kstar, "redshifts": list(C.REDSHIFTS), "per_suite_z": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kstar", type=float, default=3.0)
    a = ap.parse_args()
    r = measure(a.kstar)
    print(f"k grid (N_K={C.N_K}): {' '.join(f'{k:.2f}' for k in r['k_grid'])};  k* = {a.kstar}")
    print(f"{'suite':12s} {'z':>5s} {'n':>4s} {'k_trough':>8s} {'rise q16/50/84':>22s} {'end step q16/50/84':>26s} {'f(step>=0)':>10s}")
    for key, v in r["per_suite_z"].items():
        s, z = key.split("|")
        print(f"{s:12s} {z:>5s} {v['n_box']:4d} {v['k_trough_median_curve']:8.2f} "
              f"{' '.join(f'{x:+.3f}' for x in v['rise_q16_50_84']):>22s} {' '.join(f'{x:+.4f}' for x in v['end_step_q16_50_84']):>26s} "
              f"{v['frac_end_step_ge0']:10.2f}")
    out = C.NS / "calibration"; out.mkdir(parents=True, exist_ok=True)
    (out / f"upturn_calib_kmax{C.KCUT}_kstar{a.kstar}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()

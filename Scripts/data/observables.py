"""Per-box observables from the z=0 halo profiles (spec 4.4, 4.5; bootstrap 6).

f_b      = mean over halos with FB_MASS_MIN < M200c < FB_MASS_MAX of halo/f_baryon_200c  (mean_ratio)
D        = OmegaBaryon/Omega_m - f_b
Never imputed, never floored: no in-mask halo -> f_b = NaN with a reason in f_b_source; D <= 0 or
non-finite -> D_valid = False and the box is excluded downstream.
Subgrid parameters are written ONLY to the group 'diagnostic_labels_NOT_FEATURES'.

Usage:  python -m Scripts.data.observables
"""
import hashlib
import json
from multiprocessing import Pool

import h5py
import numpy as np

from ..common import config as C
from ..common import provenance as P
from . import loader as L


def _one_box(suite, set_name, box):
    path = C.DATA_ROOT / C.PROFILE_DIR / suite / set_name / box / C.PROFILE_FILE
    rec = {"box_id": box}
    if not path.exists():
        rec.update(f_b=np.nan, f_b_source="profile file missing", n_halo_in_mask=0, Omega_b=np.nan,
                   Omega_b_source="missing", Omega_m=np.nan, sigma_8=np.nan, f_b_M=None, labels={})
        return rec
    with h5py.File(P.log_read(path), "r") as f:
        M = f["halo/M200c_Msun_h"][()]
        fb_h = f["halo/f_baryon_200c"][()]
        m = (M > C.FB_MASS_MIN) & (M < C.FB_MASS_MAX) & np.isfinite(fb_h)
        n = int(m.sum())
        if n >= C.FB_MIN_HALOS:
            rec["f_b"], rec["f_b_source"] = float(np.mean(fb_h[m])), f"mean_ratio over {n} halos"
        else:
            rec["f_b"] = np.nan
            rec["f_b_source"] = f"no halo with {C.FB_MASS_MIN:.0e}<M200c<{C.FB_MASS_MAX:.0e} (catalog has {len(M)})"
        rec["n_halo_in_mask"] = n
        if "OmegaBaryon" in f.attrs:
            rec["Omega_b"], rec["Omega_b_source"] = float(f.attrs["OmegaBaryon"]), "attr OmegaBaryon"
        else:
            rec["Omega_b"], rec["Omega_b_source"] = C.OMEGA_B_FALLBACK, "REGISTERED FALLBACK 0.049"
        sp = f["simulation_parameters"]
        rec["Omega_m"] = float(sp["Omega_m"][()])
        rec["sigma_8"] = float(sp["sigma_8"][()])
        st = f.get("stacks/f_baryon_200c")
        if st is not None:
            rec["f_b_M"] = {"mean": st["mean"][()], "count": st["count"][()],
                            "edges_log10": f["stacks/mass_bin_edges_log10_M200c_Msun_h"][()]}
        else:
            rec["f_b_M"] = None
        rec["labels"] = {k: float(sp[k][()]) for k in ("A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "seed") if k in sp}
    return rec


def _one_set(args):
    suite, set_name = args
    boxes = L.list_boxes(suite, set_name)
    return suite, set_name, [_one_box(suite, set_name, b) for b in boxes], P.reads()


def build(nproc=6):
    tasks = [(s, st) for s in C.SUITES for st in C.SCORED_SETS]
    with Pool(min(nproc, len(tasks))) as pool:
        res = pool.map(_one_set, tasks)
    bad = [p for r in res for p in r[3] if not (p.startswith(str(C.SANDBOX)) or p.startswith(str(C.DATA_ROOT)))]
    assert not bad, f"out-of-scope reads: {bad[:3]}"
    out = C.DATA_OUT / "observables_z000.h5"
    C.DATA_OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    with h5py.File(out, "w") as f:
        f.attrs["_provenance"] = P.dumps(P.provenance("observables", redshifts=[0.0],
                                                      fb_min_halos=C.FB_MIN_HALOS,
                                                      definition="D = OmegaBaryon/Omega_m - f_b,200c(z=0)"))
        for suite, set_name, recs, _ in res:
            g = f.create_group(f"{suite}/{set_name}")
            fb = np.array([r["f_b"] for r in recs])
            ob = np.array([r["Omega_b"] for r in recs])
            om = np.array([r["Omega_m"] for r in recs])
            D = ob / om - fb                      # NaN propagates; never floored
            valid = np.isfinite(D) & (D > 0)
            g.create_dataset("box_ids", data=np.array([r["box_id"] for r in recs], dtype="S16"))
            g.create_dataset("f_b", data=fb)
            g.create_dataset("D", data=D)
            g.create_dataset("D_valid", data=valid)
            g.create_dataset("Omega_b", data=ob)
            g.create_dataset("Omega_m", data=om)
            g.create_dataset("sigma_8", data=np.array([r["sigma_8"] for r in recs]))
            g.create_dataset("n_halo_in_mask", data=np.array([r["n_halo_in_mask"] for r in recs]))
            g.create_dataset("f_b_source", data=np.array([r["f_b_source"] for r in recs], dtype="S96"))
            g.create_dataset("Omega_b_source", data=np.array([r["Omega_b_source"] for r in recs], dtype="S32"))
            has = [r["f_b_M"] is not None for r in recs]
            if any(has):
                ref = next(r["f_b_M"] for r in recs if r["f_b_M"] is not None)
                nb = len(ref["mean"])
                fbm = np.full((len(recs), nb), np.nan); cnt = np.zeros((len(recs), nb), int)
                for i, r in enumerate(recs):
                    if r["f_b_M"] is not None:
                        assert np.array_equal(r["f_b_M"]["edges_log10"], ref["edges_log10"])
                        fbm[i], cnt[i] = r["f_b_M"]["mean"], r["f_b_M"]["count"]
                g.create_dataset("f_b_M_mean", data=fbm)
                g.create_dataset("f_b_M_count", data=cnt)
                g.create_dataset("f_b_M_edges_log10", data=ref["edges_log10"])
            d = g.create_group("diagnostic_labels_NOT_FEATURES")
            for k in C.FORBIDDEN_FEATURES:
                d.create_dataset(k, data=np.array([r["labels"].get(k, np.nan) for r in recs]))
            excl = ~valid
            missing = np.array([r["f_b_source"] == "profile file missing" for r in recs])
            nohalo = np.array([r["f_b_source"].startswith("no halo") for r in recs])
            dnonpos = excl & ~missing & ~nohalo           # halos measured but D <= 0 (f_b >= cosmic)
            summary[f"{suite}/{set_name}"] = {
                "n_box": len(recs), "n_excluded": int(excl.sum()),
                "n_excluded_profile_file_absent": int(missing.sum()),
                "n_excluded_no_halo_in_mask": int(nohalo.sum()),
                "n_excluded_D_nonpositive": int(dnonpos.sum()),
                "profile_absent_box_ids": [r["box_id"] for r, m in zip(recs, missing) if m],
                "n_profile_present_Omega_b_fallback": int(sum(r["Omega_b_source"] not in ("attr OmegaBaryon", "missing")
                                                              for r in recs)),
                "Omega_m_median_no_halo_excluded": float(np.median(om[nohalo])) if nohalo.any() else None,
                "Omega_m_median_retained": float(np.median(om[valid])) if valid.any() else None,
                "Omega_m_of_profile_absent": "unknown in scope (no parameter source in DATA_ROOT)",
            }
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (C.LOGS / "observables_summary.json").write_text(json.dumps({"path": str(out), "sha256": sha,
                                                                  "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))
    return out


if __name__ == "__main__":
    build()

"""Per-(box, z) observables from the owner-supplied measured f_b(z) files (v1 Plans/11, anchor `fb_measured_z`).

Source: DATA_ROOT/f_b_z/<Suite>_LH_baryon_power_z0_2.hdf5 (one row per LH realization per native snapshot; untrusted
data, read-only). For each registered anchor z and box, the f_b row is the snapshot whose `redshift_nominal` equals
the hydro Pk label this box's spectrum uses at z (loader.box_label), so target and anchor come from the same snapshot.

f_b(z)     = fb_main_over_cosmic * cosmic_baryon_fraction   (mean over halos with 5e12 < M200c < 1e14 of f_baryon_200c)
D(z)       = Omega_b/Omega_m - f_b(z)
f_b_M_i(z) = fb_over_cosmic[:, i] * cosmic_baryon_fraction, valid per bin (fb_valid)
Never imputed, never floored: an invalid f_b, D(z) <= 0 or a missing row leaves the (box, z) block invalid.
Subgrid parameters are written ONLY to 'diagnostic_labels_NOT_FEATURES'. LH only (CV enters only C_fv).

Usage:  python -m Scripts.data.observables_zfb        (PK_ANALYSIS not required; the grid is the zfb12 grid)
"""
import hashlib
import json

import h5py
import numpy as np

from ..common import config as C
from ..common import provenance as P
from . import loader as L

ZGRID = (0.00, 0.10, 0.21, 0.33, 0.47, 0.61, 0.77, 0.95, 1.15, 1.36, 1.60, 2.00)
NOMINAL_TOL = 1e-9


def src(suite):
    return C.DATA_ROOT / C.FB_Z_DIR / f"{suite}_LH_baryon_power_z0_2.hdf5"


def _suite(suite):
    with h5py.File(P.log_read(src(suite)), "r") as f:
        sim = f["simulation"][()].astype(str)
        zn = f["redshift_nominal"][()]
        fbm = f["fb_main_over_cosmic"][()]
        fbm_ok = f["fb_main_valid"][()].astype(bool)
        fbb = f["fb_over_cosmic"][()]
        fbb_ok = f["fb_valid"][()].astype(bool)
        nh = f["n_halos"][()]
        nmain = f["n_halos_main"][()]
        cbf = f["cosmic_baryon_fraction"][()]
        names = [n.decode() for n in f["parameters/names"][()]]
        par = f["parameters/values"][()]
        edges = f["mass_bin_edges_log10_M200c_Msun_h"][()]
        prof = f["profile_available"][()].astype(bool)
    col = {n: i for i, n in enumerate(names)}
    boxes = L.list_boxes(suite, "LH")
    nb, nz = len(boxes), len(ZGRID)
    out = {k: np.full((nb, nz), np.nan) for k in ("f_b", "D", "cosmic_baryon_fraction", "z_nominal")}
    out["f_b_M_mean"] = np.full((nb, nz, 6), np.nan)
    out["f_b_M_count"] = np.zeros((nb, nz, 6), int)
    out["f_b_M_valid"] = np.zeros((nb, nz, 6), bool)
    out["n_halo_in_mask"] = np.full((nb, nz), -1, int)
    out["block_valid"] = np.zeros((nb, nz), bool)
    reason = np.full((nb, nz), "", dtype=object)
    om, ob, s8 = np.full(nb, np.nan), np.full(nb, np.nan), np.full(nb, np.nan)
    labels = {k: np.full(nb, np.nan) for k in C.FORBIDDEN_FEATURES}
    rows = {}
    for i, s in enumerate(sim):
        rows.setdefault(s, []).append(i)
    for b_i, b in enumerate(boxes):
        rr = rows.get(b)
        if not rr:
            reason[b_i, :] = "no rows in f_b_z file"
            continue
        rr = np.array(rr)
        pv = par[rr]
        for nm, arr in (("Omega_m", om), ("Omega_b", ob), ("sigma_8", s8)):
            v = pv[:, col[nm]]
            assert np.allclose(v, v[0], rtol=0, atol=0) or np.nanmax(np.abs(v - v[0])) < 1e-12, f"{suite}/{b}: {nm} varies"
            arr[b_i] = v[0]
        for k in labels:
            labels[k][b_i] = pv[0, col[k]]
        for j, z in enumerate(ZGRID):
            lab = float(L.box_label(suite, "hydro", "LH", b, z))
            hit = rr[np.abs(zn[rr] - lab) < NOMINAL_TOL]
            assert len(hit) <= 1, f"{suite}/{b} z={z}: {len(hit)} f_b rows at nominal {lab}"
            if len(hit) == 0:
                reason[b_i, j] = f"no f_b row at nominal z={lab:.2f}"
                continue
            r = hit[0]
            out["z_nominal"][b_i, j] = zn[r]
            out["cosmic_baryon_fraction"][b_i, j] = cbf[r]
            out["n_halo_in_mask"][b_i, j] = nmain[r]
            out["f_b_M_count"][b_i, j] = nh[r]
            out["f_b_M_valid"][b_i, j] = fbb_ok[r]
            out["f_b_M_mean"][b_i, j] = np.where(fbb_ok[r], fbb[r] * cbf[r], np.nan)
            if not prof[r]:
                reason[b_i, j] = "profile unavailable"
                continue
            if not fbm_ok[r]:
                reason[b_i, j] = f"no halo in main mass range (n={nmain[r]})"
                continue
            fb = fbm[r] * cbf[r]
            D = ob[b_i] / om[b_i] - fb
            out["f_b"][b_i, j], out["D"][b_i, j] = fb, D
            if not (np.isfinite(D) and D > 0):
                reason[b_i, j] = "D(z) <= 0"
                continue
            out["block_valid"][b_i, j] = True
            reason[b_i, j] = "ok"
    return dict(boxes=boxes, Omega_m=om, Omega_b=ob, sigma_8=s8, labels=labels, edges=edges, reason=reason, **out)


def build():
    C.OBS_ZFB.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with h5py.File(C.OBS_ZFB, "w") as f:
        f.attrs["_provenance"] = P.dumps(P.provenance("observables_zfb", redshifts=list(ZGRID),
                                                      source="DATA_ROOT/f_b_z (owner upload 2026-10-22)",
                                                      definition="D(z) = Omega_b/Omega_m - fb_main_over_cosmic*cosmic_baryon_fraction",
                                                      pairing="f_b row redshift_nominal == hydro Pk label used for that box at z"))
        f.create_dataset("redshifts", data=np.array(ZGRID))
        for s in C.SUITES:
            r = _suite(s)
            g = f.create_group(f"{s}/LH")
            g.create_dataset("box_ids", data=np.array(r["boxes"], dtype="S16"))
            for k in ("f_b", "D", "block_valid", "Omega_m", "Omega_b", "sigma_8", "f_b_M_mean", "f_b_M_count",
                      "f_b_M_valid", "n_halo_in_mask", "cosmic_baryon_fraction", "z_nominal"):
                g.create_dataset(k, data=r[k])
            g.create_dataset("f_b_M_edges_log10", data=r["edges"])
            g.create_dataset("block_reason", data=r["reason"].astype("S64"))
            d = g.create_group("diagnostic_labels_NOT_FEATURES")
            for k, v in r["labels"].items():
                d.create_dataset(k, data=v)
            bv = r["block_valid"]
            summary[s] = {"n_box": len(r["boxes"]), "n_valid_by_z": dict(zip([f"{z:.2f}" for z in ZGRID], bv.sum(0).tolist())),
                          "n_box_all_z_valid": int(bv.all(1).sum()), "n_box_any_z_valid": int(bv.any(1).sum()),
                          "reasons": {k: int(v) for k, v in zip(*np.unique(r["reason"].astype(str), return_counts=True))}}
    bad = P.out_of_scope_reads()
    assert not bad, f"out-of-scope reads: {bad[:3]}"
    sha = hashlib.sha256(C.OBS_ZFB.read_bytes()).hexdigest()
    (C.OBS_ZFB.parent / "observables_zfb_summary.json").write_text(json.dumps({"path": str(C.OBS_ZFB), "sha256": sha,
                                                                               "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    build()

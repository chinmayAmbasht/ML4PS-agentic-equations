"""Build S(k, z) = P_hydro/P_dm - 1 from raw CAMELS spectra (spec 8.2 'rebuild convention').

Writes Artifacts/data/pk_suppression_z<ZZZ>_allk.h5, one per registered redshift, in the same layout
as the frozen z=0 artifact: <Suite>/<Set>/{box_ids, delta}, root k_hmpc, plus _provenance and the
anchor -> per-directory label map.

Usage:  python -m Scripts.data.build_suppression [--z 0.00 0.47 ...]
"""
import argparse
import hashlib
import json
from multiprocessing import Pool

import h5py
import numpy as np

from ..common import config as C
from ..common import provenance as P
from . import loader as L


def reference_k() -> np.ndarray:
    k, _ = L.read_pk(L.pk_path("IllustrisTNG", "hydro", "LH", "LH_0", L.match_label(0.0, L.available_labels("IllustrisTNG", "hydro"))))
    return k


def _build_one(args):
    suite, set_name, redshifts = args
    k_ref = reference_k()
    boxes = L.list_boxes(suite, set_name)
    delta = {z: np.empty((len(boxes), C.N_K_RAW)) for z in redshifts}
    used = {z: {"hydro": set(), "dm": set()} for z in redshifts}
    max_off, k_native0 = 0.0, None
    for i, b in enumerate(boxes):
        for z in redshifts:
            lh = L.box_label(suite, "hydro", set_name, b, z)
            ld = L.box_label(suite, "dm", set_name, b, z)
            used[z]["hydro"].add(lh); used[z]["dm"].add(ld)
            kh, ph = L.read_pk(L.pk_path(suite, "hydro", set_name, b, lh))
            kd, pd = L.read_pk(L.pk_path(suite, "dm", set_name, b, ld))
            if k_native0 is None:
                k_native0 = kh
            # k grid identical across redshift and boxes within a suite (asserted bitwise), and hydro==DM
            assert np.array_equal(kh, k_native0), f"{suite}/{set_name}/{b} z={z}: k differs within suite"
            assert np.array_equal(kd, kh) or np.max(np.abs(kd / kh - 1)) < C.K_GRID_NORMALIZE_TOL
            _, off = L.common_k(kh, k_ref)
            max_off = max(max_off, off)
            delta[z][i] = ph / pd - 1.0
    return {"suite": suite, "set": set_name, "boxes": boxes, "delta": delta,
            "labels_used": {f"{z:.2f}": {k: sorted(v) for k, v in u.items()} for z, u in used.items()},
            "k_rel_offset_vs_ref": max_off, "reads": P.reads()}


def build(redshifts, nproc=6):
    C.DATA_OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(s, st, tuple(redshifts)) for s in C.SUITES for st in C.SCORED_SETS]
    with Pool(min(nproc, len(tasks))) as pool:
        results = pool.map(_build_one, tasks)
    k_ref = reference_k()
    lmap = L.label_map(redshifts)
    n_reads = sum(len(r["reads"]) for r in results)
    bad = [p for r in results for p in r["reads"]
           if not (p.startswith(str(C.SANDBOX)) or p.startswith(str(C.DATA_ROOT)))]
    assert not bad, f"out-of-scope reads: {bad[:3]}"
    written = {}
    for z in redshifts:
        out = C.DATA_OUT / f"pk_suppression_z{int(round(z * 100)):03d}_allk.h5"
        prov = P.provenance("data_build", redshifts=[z],
                            source="DATA_ROOT raw Pk_m_z=<label>.txt, hydro and DM",
                            definition="delta = P_hydro/P_dm - 1, total matter",
                            label_map=lmap[f"{z:.2f}"],
                            k_reference="IllustrisTNG/LH/LH_0 native grid",
                            k_rel_offset_vs_ref={f"{r['suite']}/{r['set']}": r["k_rel_offset_vs_ref"] for r in results},
                            labels_used={f"{r['suite']}/{r['set']}": r["labels_used"][f"{z:.2f}"] for r in results},
                            n_files_read_total=n_reads)
        with h5py.File(out, "w") as f:
            f.create_dataset("k_hmpc", data=k_ref)
            f.attrs["redshift_anchor"] = z
            f.attrs["definition"] = "delta = P_hydro/P_dm - 1, matter"
            f.attrs["k_nyquist"] = C.K_NYQUIST
            f.attrs["_provenance"] = P.dumps(prov)
            for r in results:
                g = f.create_group(f"{r['suite']}/{r['set']}")
                g.create_dataset("box_ids", data=np.array(r["boxes"], dtype="S16"))
                g.create_dataset("delta", data=r["delta"][z])
        written[f"{z:.2f}"] = {"path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        print(f"wrote {out.name}")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, nargs="+", default=list(C.REDSHIFTS))
    ap.add_argument("--nproc", type=int, default=6)
    a = ap.parse_args()
    w = build(a.z, a.nproc)
    C.LOGS.mkdir(parents=True, exist_ok=True)
    (C.LOGS / f"build_suppression_{'_'.join(w)}.json").write_text(json.dumps(w, indent=1))

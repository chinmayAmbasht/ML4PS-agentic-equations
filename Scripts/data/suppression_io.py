"""The engine's ONLY reader of targets and observables (spec 8.5, 8.6).

Returns, per (suite, set), the stacked target vector y_b = [S(k_1..k_16, z_1), ..., S(k_1..k_16, z_nz)]
(k fastest within each redshift block), joined to observables by box_id -- never by array position.
Selection is by registered redshift (file name from the anchor), never by file order.
"""
from dataclasses import dataclass, field

import h5py
import numpy as np

from ..common import config as C
from ..common.provenance import log_read
from .loader import kgrid_columns


def zfile(z: float):
    return C.DATA_OUT / f"pk_suppression_z{int(round(z * 100)):03d}_allk.h5"


@dataclass
class SuiteData:
    suite: str
    set_name: str
    redshifts: tuple
    box_ids: np.ndarray            # (n_box,) str
    y: np.ndarray                  # (n_box, n_k * n_z)
    mask: np.ndarray               # (n_box, n_k * n_z) bool -- True = entry available
    obs: dict = field(default_factory=dict)   # permitted features only, aligned to box_ids
    valid: np.ndarray = None       # D_valid (box-level exclusion, spec 4.4)


def load(suite: str, set_name: str, redshifts=None, require_valid=True) -> SuiteData:
    redshifts = C.REDSHIFTS if redshifts is None else redshifts
    blocks, ids0 = [], None
    for z in redshifts:
        with h5py.File(log_read(zfile(z)), "r") as f:
            assert abs(float(f.attrs["redshift_anchor"]) - z) < 1e-12
            cols = kgrid_columns(f["k_hmpc"][()])
            ids = f[f"{suite}/{set_name}/box_ids"][()].astype(str)
            d = f[f"{suite}/{set_name}/delta"][()][:, cols]
        if ids0 is None:
            ids0 = ids
        assert np.array_equal(ids, ids0), "box order differs across redshift files"
        blocks.append(d)
    y = np.concatenate(blocks, axis=1)               # k fastest within each z block
    assert y.shape[1] == C.N_K * len(redshifts)
    if C.ANCHOR_DESIGN == "fb_measured_z" and set_name == "LH":
        return _load_zfb(suite, set_name, tuple(redshifts), ids0, y, require_valid)
    with h5py.File(log_read(C.DATA_OUT / "observables_z000.h5"), "r") as f:
        g = f[f"{suite}/{set_name}"]
        oid = g["box_ids"][()].astype(str)
        pos = {b: i for i, b in enumerate(oid)}
        idx = np.array([pos[b] for b in ids0])          # join by box_id
        obs = {k: g[k][()][idx] for k in ("f_b", "D", "Omega_b", "Omega_m", "sigma_8")}
        obs["Omega_b_over_Omega_m"] = obs["Omega_b"] / obs["Omega_m"]
        if "f_b_M_mean" in g:
            obs["f_b_M"] = g["f_b_M_mean"][()][idx]
            obs["f_b_M_count"] = g["f_b_M_count"][()][idx]
        valid = g["D_valid"][()][idx]
    for k in obs:
        assert not any(b in k for b in C.FORBIDDEN_FEATURES[:4]), f"forbidden feature {k}"
    # under fb_frozen_z0 nothing is ragged: entry mask is all-True for every retained box (8.6)
    mask = np.isfinite(y)
    sd = SuiteData(suite, set_name, tuple(redshifts), ids0, y, mask, obs, valid)
    if require_valid:
        sd = subset(sd, valid)
    return sd


def subset(sd: SuiteData, sel) -> SuiteData:
    return SuiteData(sd.suite, sd.set_name, sd.redshifts, sd.box_ids[sel], sd.y[sel], sd.mask[sel],
                     {k: v[sel] for k, v in sd.obs.items()}, sd.valid[sel])


def _load_zfb(suite, set_name, redshifts, ids0, y, require_valid):
    """v1 Plans/11 anchor `fb_measured_z`: per-(box, z) f_b, D, f_b_M from observables_zfb.h5; box-level cosmology.
    Entry mask = block validity (f_b valid, D(z) > 0) repeated over k, AND finite target. Never imputed."""
    with h5py.File(log_read(C.OBS_ZFB), "r") as f:
        zg = [float(v) for v in f["redshifts"][()]]
        zi = [zg.index(z) for z in redshifts]                   # exact registered labels, asserted by index()
        g = f[f"{suite}/{set_name}"]
        oid = g["box_ids"][()].astype(str)
        pos = {b: i for i, b in enumerate(oid)}
        idx = np.array([pos[b] for b in ids0])                  # join by box_id
        obs = {k: g[k][()][idx][:, zi] for k in ("f_b", "D")}
        for k in ("Omega_b", "Omega_m", "sigma_8"):
            obs[k] = g[k][()][idx]
        obs["Omega_b_over_Omega_m"] = obs["Omega_b"] / obs["Omega_m"]
        obs["f_b_M"] = g["f_b_M_mean"][()][idx][:, zi, :]
        obs["f_b_M_count"] = g["f_b_M_count"][()][idx][:, zi, :]
        bv = g["block_valid"][()][idx][:, zi]
    for k in obs:
        assert not any(b in k for b in C.FORBIDDEN_FEATURES[:4]), f"forbidden feature {k}"
    assert np.all(np.isfinite(obs["D"][bv]) & (obs["D"][bv] > 0))
    mask = np.repeat(bv, C.N_K, axis=1) & np.isfinite(y)
    sd = SuiteData(suite, set_name, tuple(redshifts), ids0, y, mask, obs, mask.reshape(len(y), len(redshifts), C.N_K).any(axis=2).any(axis=1))
    if require_valid:
        sd = subset(sd, sd.valid)
    return sd

"""Raw CAMELS spectrum loader (spec 4.1, 8.2).

Encodes the four measured hazards of spec 8.2: non-uniform suite layout, redshift labels that disagree
across directories (tolerance-matched, asserted), the z=127 IC output, and Swift-EAGLE's 4.8e-10
k-grid offset (asserted below K_GRID_NORMALIZE_TOL, then the common grid is adopted).
"""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..common import config as C
from ..common.provenance import log_read

_PK_RE = re.compile(r"^Pk_m_z=(\d+\.\d+)\.txt$")


def suite_dir(suite: str, kind: str) -> Path:
    hydro, dm = C.SUITE_DIRS[suite]
    return C.DATA_ROOT / (hydro if kind == "hydro" else dm)


def list_boxes(suite: str, set_name: str) -> list[str]:
    """Box ids present in BOTH hydro and DM trees, ordered by integer index."""
    def ids(kind):
        d = suite_dir(suite, kind) / set_name
        return {p.name for p in d.iterdir() if p.is_dir() and p.name.startswith(set_name + "_")}
    h, d = ids("hydro"), ids("dm")
    assert h == d, f"{suite}/{set_name}: hydro/DM box sets differ ({len(h ^ d)} mismatched)"
    return sorted(h, key=lambda s: int(s.split("_")[1]))


@lru_cache(maxsize=None)
def available_labels(suite: str, kind: str, set_name: str = "LH", box: str | None = None) -> tuple[str, ...]:
    d = suite_dir(suite, kind) / set_name / (box or f"{set_name}_0")
    labs = []
    for p in d.iterdir():
        m = _PK_RE.match(p.name)
        if m and float(m.group(1)) not in C.Z_EXCLUDE:
            labs.append(m.group(1))
    return tuple(sorted(labs, key=float))


def match_label(anchor: float, labels: tuple[str, ...]) -> str:
    """Nearest label to the anchor (spec 8.2), asserted within Z_LABEL_MATCH_TOL and strictly nearer than
    the runner-up (no tie). Dense ladders (Astrid_DM, Swift) carry e.g. 0.02 next to 0.00, so a
    'one label in the window' rule is wrong; nearest-unambiguous is the registered rule. Never string-match."""
    d = sorted((abs(float(l) - anchor), l) for l in labels)
    assert d and d[0][0] <= C.Z_LABEL_MATCH_TOL + 1e-12, f"anchor {anchor}: nearest {d[:1]} outside tol"
    assert len(d) == 1 or d[1][0] > d[0][0] + 1e-9, f"anchor {anchor}: tie between {d[0][1]} and {d[1][1]}"
    return d[0][1]


def label_map(redshifts=C.REDSHIFTS, set_name: str = "LH") -> dict:
    """anchor -> {suite: {hydro: label, dm: label}} plus median of matched labels."""
    out = {}
    for z in redshifts:
        row, vals = {}, []
        for s in C.SUITES:
            lh = match_label(z, available_labels(s, "hydro", set_name))
            ld = match_label(z, available_labels(s, "dm", set_name))
            row[s] = {"hydro": lh, "dm": ld}
            vals += [float(lh), float(ld)]
        out[f"{z:.2f}"] = {"labels": row, "median_label": float(np.median(vals)),
                           "worst_offset": float(max(abs(v - z) for v in vals))}
    return out


def read_pk(path: Path) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(log_read(path), dtype=np.float64)
    assert a.shape == (C.N_K_RAW, 2), f"{path}: shape {a.shape}"
    return a[:, 0], a[:, 1]


def pk_path(suite: str, kind: str, set_name: str, box: str, label: str) -> Path:
    return suite_dir(suite, kind) / set_name / box / f"Pk_m_z={label}.txt"


def box_label(suite: str, kind: str, set_name: str, box: str, anchor: float) -> str:
    """Per-box label resolution (asserted), so a box with an unusual ladder cannot silently mis-pair."""
    return match_label(anchor, available_labels(suite, kind, set_name, box))


def common_k(k_native: np.ndarray, k_ref: np.ndarray) -> tuple[np.ndarray, float]:
    """Assert float-level agreement with the reference grid and adopt it. Returns (k_ref, max rel offset)."""
    rel = float(np.max(np.abs(k_native / k_ref - 1.0)))
    assert rel < C.K_GRID_NORMALIZE_TOL, f"k-grid offset {rel:.3e} exceeds {C.K_GRID_NORMALIZE_TOL}"
    return k_ref, rel


def kgrid_columns(k: np.ndarray) -> np.ndarray:
    """Indices of the registered (k-cut) K_GRID points by EXACT float64 match (never rounding)."""
    idx = [int(np.flatnonzero(k == kk)[0]) if np.any(k == kk) else -1 for kk in C.K_GRID]
    assert all(i >= 0 for i in idx), f"exact k match {sum(i >= 0 for i in idx)}/{len(idx)}"
    assert len(idx) == C.N_K, "k columns do not match the (cut) registered grid"
    return np.array(idx)

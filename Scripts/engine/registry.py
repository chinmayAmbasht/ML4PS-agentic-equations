"""Freeze file (spec 1.4) and the single-writer registry (spec 11).

candidates_frozen.jsonl : append-only; one row per form_id, written BEFORE any score; editing a frozen
                          expression raises -- a changed form needs a new form_id.
registry.csv            : one row per form_id, upsert by form_id, non-finite -> null, _provenance column.
"""
import csv
import fcntl
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ..common import config as C

import os as _os
FROZEN = Path(_os.environ.get("PK_FROZEN", C.ARTIFACTS / "candidates_frozen.jsonl"))   # opt-in other freeze file
REGISTRY = C.NS / "registry.csv"

COLUMNS = ["form_id", "symbolic_expr", "n_coeff", "coeff_roles",
           "G0_1", "G0_2", "G0_3", "G0_U", "gate0_passed", "G0_by_redshift", "G0_early_time_flag",
           "marg_coords", "n_marginalized", "ladder_mode",
           "S_cross", "S_same_suite", "S_pooled", "sigma_fold", "S_cross_by_marg_set",
           "rank_promotion_arithmetic",
           "S_cross_redshift", "S_cross_both",
           "L_zero_shot", "L_map_diagnostic_widened_prior",
           "chi2_nu_per_suite", "chi2_nu_mean", "chi2_nu_std", "chi2_nu_Cfv_Cdisc_per_suite", "presentable",
           "per_box_logZ", "n_boxes_scored", "n_boxz_scored", "tail_mass",
           "mu", "Sigma_suite_eigenvalues", "variance_fraction",
           "coeff_z_drift", "coeff_z_drift_over_cross_suite_sd",
           "C_finite_volume", "C_discrepancy", "cfv_assembly", "separability_test_spread",
           "seed", "notes", "_provenance"]


def _now():
    return datetime.now(timezone.utc).isoformat()


def frozen_rows(path=FROZEN):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def freeze(form, proposer: str, batch: str, path=FROZEN, raw: dict = None, compile_error: str = None) -> dict:
    """Freeze a compiled Form, or (raw=submitted dict, compile_error=...) a form that does not compile."""
    import hashlib
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fid = form.form_id if form is not None else raw["form_id"]
    expr = form.expr if form is not None else raw["expr"]
    sha = hashlib.sha256(expr.encode()).hexdigest()
    record = form.to_record() if form is not None else raw
    for r in frozen_rows(path):
        if r["form_id"] == fid:
            if r["sha256"] != sha or r["record"] != record:
                raise ValueError(f"{fid} already frozen with a different expression; use a new form_id")
            return r
    row = {"form_id": fid, "expr": expr, "sha256": sha, "features": list(record.get("features", [])),
           "n_coeff": form.n_coeff if form is not None else None, "proposer": proposer, "batch": batch,
           "frozen_at": _now(), "record": record, "compiles": form is not None, "compile_error": compile_error}
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, sort_keys=True) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    return row


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "tolist"):
        return _clean(v.tolist())
    return v


def _cell(v):
    v = _clean(v)
    if v is None:
        return "null"
    if isinstance(v, (dict, list, bool)):
        return json.dumps(v, sort_keys=True)
    return str(v)


def upsert(row: dict, path=REGISTRY):
    """Single writer: upsert by form_id. Requires the form to be frozen first (spec 1.4)."""
    assert "_provenance" in row and row.get("form_id"), "registry rows need form_id and _provenance"
    frozen = {r["form_id"]: r for r in frozen_rows()}
    assert row["form_id"] in frozen, f"{row['form_id']} not frozen before scoring"
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    unknown = set(row) - set(COLUMNS)
    assert not unknown, f"unknown registry columns {unknown}"
    with open(str(path) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows = []
        if path.exists():
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        rows = [r for r in rows if r["form_id"] != row["form_id"]]
        rows.append({c: _cell(row.get(c)) for c in COLUMNS})
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader(); w.writerows(rows)
        tmp.replace(path)
        fcntl.flock(lk, fcntl.LOCK_UN)

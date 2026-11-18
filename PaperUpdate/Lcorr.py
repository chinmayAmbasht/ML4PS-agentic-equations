"""Correlated benchmark loss (owner 2026-11-13): for a box b of suite j,
    L_b = dS_b^T (C_fv,j / h)^{-1} dS_b,   h = (27 - 16 - 2)/26 (Hartlap),
with C_fv,j suite j's own engine CV covariance (empirical 16x16 over the 27 CV boxes, shrinkage 0.10 toward the diagonal).
L(set) = mean over boxes within a suite, then mean over suites. No C_disc. Replaces the diagonal sigma_k weighting.
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

M = Path(__file__).resolve().parents[1]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns_Lc"))
sys.path.insert(0, str(M))

import numpy as np

from Scripts.common import config as C

N_CV = 27
H = (N_CV - C.N_K - 2) / (N_CV - 1)


@lru_cache(maxsize=None)
def cfv(suite):
    from Scripts.data import suppression_io as IO
    from Scripts.engine import cov as CV
    return np.asarray(CV.cfv(IO.load(suite, "CV", require_valid=False).y, 1)["C"], float)


@lru_cache(maxsize=None)
def whitener(suite):
    """Matrix Wm with Wm @ d = whitened residual, so |Wm d|^2 = d^T (C/h)^{-1} d."""
    Lc = np.linalg.cholesky(cfv(suite) / H)
    return np.linalg.inv(Lc)


def whiten(d, suite):
    return d @ whitener(suite).T


def L_boxes(pred, y, suite):
    r = whiten(pred - y, suite)
    return np.sum(r * r, axis=1)


def L(pred, y, suite):
    m = np.isfinite(y).all(axis=1) & np.isfinite(pred).all(axis=1)
    return float(np.mean(L_boxes(pred[m], y[m], suite)))

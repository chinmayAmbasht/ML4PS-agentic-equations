"""Covariances (spec 7.2, 8.5; Plans/02 section 3).

C_fv  : per suite, from all 27 CV boxes. S = sigma_{kz} sigma_{k'z'} R_k[k,k'] R_z[z,z'] (R_k = mean over z of per-z
        CV k-k correlations, R_z = mean over k of per-k CV z-z correlations; at n_z = 1 S is exactly the empirical
        covariance), then C_fv = (1-CV_SHRINKAGE) S + CV_SHRINKAGE diag(S). Hartlap reported, not applied.
C_disc: diagonal over the stacked index, moment-matched from TRAINING-suite Stage-1 residuals only.
"""
import numpy as np

from ..common import config as C


def _corr(a):
    c = np.cov(a, rowvar=False)
    s = np.sqrt(np.diag(c))
    return c / np.outer(s, s), c


def cfv_pieces(y_cv: np.ndarray, n_z: int):
    """y_cv: (27, n_k*n_z) CV targets, k fastest. Returns sigma (n_y,), UNSHRUNK R_k, R_z, and per-bin pieces."""
    nb, ny = y_cv.shape
    assert ny == C.N_K * n_z
    Y = y_cv.reshape(nb, n_z, C.N_K)
    sigma = np.std(y_cv, axis=0, ddof=1)
    Rk_per_z = np.stack([_corr(Y[:, i, :])[0] for i in range(n_z)])
    Rk = Rk_per_z.mean(axis=0)
    if n_z > 1:
        Rz_per_k = np.stack([_corr(Y[:, :, j])[0] for j in range(C.N_K)])
        Rz = Rz_per_k.mean(axis=0)
    else:
        Rz_per_k = np.ones((C.N_K, 1, 1))
        Rz = np.ones((1, 1))
    return sigma, Rk, Rz, Rk_per_z, Rz_per_k


def assemble(sigma, Rk, Rz):
    """Separable assembly sigma_{kz} sigma_{k'z'} R_k R_z, index (z, k) with k fastest."""
    return np.kron(Rz, Rk) * np.outer(sigma, sigma)


def shrink(S):
    """Shrinkage toward the diagonal: (1-l) S + l diag(S) == shrinking the correlation toward identity."""
    return (1.0 - C.CV_SHRINKAGE) * S + C.CV_SHRINKAGE * np.diag(np.diag(S))


def lw_lambda(y, T):
    """Ledoit-Wolf shrinkage intensity of the sample covariance toward a given target T (target variance ignored)."""
    n = y.shape[0]
    X = y - y.mean(0)
    S = X.T @ X / (n - 1)
    var = np.zeros_like(S)
    for i in range(n):
        xi = X[i][:, None] * X[i][None, :]
        var += (xi - S * (n - 1) / n) ** 2
    var *= n / ((n - 1) ** 3)
    return float(np.clip(var.sum() / ((S - T) ** 2).sum(), 0.0, 1.0)), S


def cfv(y_cv: np.ndarray, n_z: int) -> dict:
    """C_fv. n_z = 1: exactly the z=0 arm. n_z > 1 (registered S2', Plans/04 D4.0, user 2026-10-09):
    S = (1-lam) S_emp + lam S_sep with lam Ledoit-Wolf toward the separable assembly, then diagonal shrinkage."""
    if C.CFV_MODE == "blockdiag":
        return cfv_blockdiag(y_cv, n_z)
    sigma, Rk, Rz, Rk_per_z, Rz_per_k = cfv_pieces(y_cv, n_z)
    lam = None
    if n_z == 1:
        # with one redshift the separable assembly IS the empirical covariance; use it exactly (reduction test 8.11)
        S = np.cov(y_cv, rowvar=False)
    else:
        Ssep = assemble(sigma, Rk, Rz)
        lam, Semp = lw_lambda(y_cv, Ssep)
        S = (1.0 - lam) * Semp + lam * Ssep
    Cm = shrink(S)
    ev = np.linalg.eigvalsh(Cm)
    return {"C": Cm, "sigma": sigma, "R_k": Rk, "R_z": Rz, "cond": float(ev[-1] / ev[0]), "lw_lambda": lam,
            "min_eig": float(ev[0]), "hartlap_factor": C.HARTLAP_FACTOR, "hartlap_applied": C.HARTLAP_APPLIED,
            "n_cv": int(y_cv.shape[0]), "_Rk_per_z": Rk_per_z, "_Rz_per_k": Rz_per_k}


def cfv_blockdiag(y_cv: np.ndarray, n_z: int) -> dict:
    """v1 Plans/11 (owner 2026-10-22): block-diagonal in z. Block z = empirical 16x16 CV covariance at that z with the
    registered diagonal shrinkage, i.e. the z=0 arm at every z; NO cross-z terms (CV z-z correlation ignored, disclosed)."""
    nb, ny = y_cv.shape
    assert ny == C.N_K * n_z
    Cm = np.zeros((ny, ny))
    for i in range(n_z):
        sl = slice(i * C.N_K, (i + 1) * C.N_K)
        Cm[sl, sl] = shrink(np.cov(y_cv[:, sl], rowvar=False))
    ev = np.linalg.eigvalsh(Cm)
    return {"C": Cm, "sigma": np.std(y_cv, axis=0, ddof=1), "cond": float(ev[-1] / ev[0]), "lw_lambda": None,
            "min_eig": float(ev[0]), "hartlap_factor": C.HARTLAP_FACTOR, "hartlap_applied": C.HARTLAP_APPLIED,
            "n_cv": int(nb), "assembly": "blockdiag_z"}


def cfv_z0_reference(y_cv: np.ndarray) -> np.ndarray:
    """Independent z=0-only implementation of the published C_fv (16x16), for gate item 12."""
    S = np.cov(y_cv, rowvar=False)
    return (1.0 - C.CV_SHRINKAGE) * S + C.CV_SHRINKAGE * np.diag(np.diag(S))


def cfv_kronecker(y_cv: np.ndarray, n_z: int) -> np.ndarray:
    """Sensitivity fallback: R_z (x) C_k, a single k-shape rescaled per redshift."""
    nb = y_cv.shape[0]
    Y = y_cv.reshape(nb, n_z, C.N_K)
    sigma, Rk, Rz, _, _ = cfv_pieces(y_cv, n_z)
    Rk = (1.0 - C.CV_SHRINKAGE) * Rk + C.CV_SHRINKAGE * np.eye(C.N_K)
    amp = np.sqrt(np.mean(sigma.reshape(n_z, C.N_K) ** 2, axis=1))         # one scale per z
    shape = np.sqrt(np.mean((sigma.reshape(n_z, C.N_K) / amp[:, None]) ** 2, axis=0))  # one k-shape
    Ck = Rk * np.outer(shape, shape)
    return np.kron(Rz * np.outer(amp, amp), Ck)


def separability(cf: dict) -> dict:
    """Spread of per-bin empirical correlations against the registered shared ones (spec 8.5)."""
    Rk_unshrunk, Rz = cf["R_k"], cf["R_z"]
    off_k = ~np.eye(C.N_K, dtype=bool)
    dk = [float(np.max(np.abs(r[off_k] - Rk_unshrunk[off_k]))) for r in cf["_Rk_per_z"]]
    nz = Rz.shape[0]
    if nz > 1:
        off_z = ~np.eye(nz, dtype=bool)
        dz = [float(np.max(np.abs(r[off_z] - Rz[off_z]))) for r in cf["_Rz_per_k"]]
        dz_low = dz[:3]
    else:
        dz, dz_low = [], []
    return {"kk_maxabs_dev_per_z": dk, "zz_maxabs_dev_per_k": dz, "zz_maxabs_dev_lowest3k": dz_low,
            "kk_dev_max": max(dk) if dk else None, "zz_dev_max": max(dz) if dz else None}


def c_disc(residuals: dict, cfv_diag: dict, training: list, held_out=None) -> np.ndarray:
    """Diagonal moment-matched discrepancy (spec 7.2 eq.).

    residuals: suite -> (n_s, n_y) Stage-1 residuals at the MAP node, for TRAINING suites only.
    The held-out suite must not be present in `residuals` or `training` (machine-checked).
    """
    if held_out is not None:
        assert held_out not in training, "held-out suite listed as a C_disc training suite"
        assert held_out not in residuals, "held-out residuals passed to C_disc"
    assert set(residuals) == set(training)
    # Plans/11 ragged blocks: masked entries arrive as NaN and are left out of the per-entry mean (never filled)
    mean = lambda r: np.mean(r ** 2, axis=0) if np.isfinite(r).all() else np.nanmean(r ** 2, axis=0)
    terms = [mean(residuals[s]) - cfv_diag[s] for s in training]
    assert np.isfinite(terms).all(), "an entry has no valid training box in some suite"
    return np.diag(np.maximum(0.0, np.mean(terms, axis=0)))


def separability_null(cf: dict, n_sim: int = 200, seed: int = 42) -> dict:
    """Noise baseline for the separability statistics: draw n_cv samples from the ASSEMBLED (separable)
    covariance itself, recompute the same max-deviation statistics. Observed >> null => separability violated."""
    rng = np.random.default_rng(seed)
    n_cv, nz = cf["n_cv"], cf["R_z"].shape[0]
    Ls = np.linalg.cholesky(cf["C"])
    obs = separability(cf)
    kk, zz, zz3 = [], [], []
    for _ in range(n_sim):
        y = (Ls @ rng.standard_normal((Ls.shape[0], n_cv))).T
        s = separability(cfv(y, nz))
        kk.append(s["kk_dev_max"]); zz.append(s["zz_dev_max"] or 0.0); zz3.append(max(s["zz_maxabs_dev_lowest3k"] or [0.0]))
    pct = lambda a, v: float(np.mean(np.array(a) >= v))
    return {"kk_obs": obs["kk_dev_max"], "kk_null_median": float(np.median(kk)), "kk_null_p95": float(np.percentile(kk, 95)),
            "kk_p_value": pct(kk, obs["kk_dev_max"]),
            "zz_obs": obs["zz_dev_max"], "zz_null_median": float(np.median(zz)), "zz_null_p95": float(np.percentile(zz, 95)),
            "zz_p_value": pct(zz, obs["zz_dev_max"]) if obs["zz_dev_max"] is not None else None,
            "zz_low3k_obs": max(obs["zz_maxabs_dev_lowest3k"]) if obs["zz_maxabs_dev_lowest3k"] else None,
            "zz_low3k_null_p95": float(np.percentile(zz3, 95)),
            "zz_low3k_p_value": pct(zz3, max(obs["zz_maxabs_dev_lowest3k"])) if obs["zz_maxabs_dev_lowest3k"] else None,
            "n_sim": n_sim}

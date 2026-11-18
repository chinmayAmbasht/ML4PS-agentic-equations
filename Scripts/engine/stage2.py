"""Stage 2: population mean and between-suite covariance (spec 6.2; Plans/02 section 5, decision D1).

    c_hat_s ~ N(mu, Sigma_suite + W_s),
    Sigma_suite ~ IW(nu = p + 2, Psi = diag(var_s c_hat_s, ddof=1) + mean_s W_s)   [D1 as amended 2026-10-09]

MAP over Sigma_suite (Cholesky-parameterised, L-BFGS, deterministic start); mu is the closed-form GLS
solution given Sigma. Off-diagonals retained.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class Stage2Result:
    mu: np.ndarray
    Sigma: np.ndarray
    nu: float
    Psi: np.ndarray
    converged: bool
    objective: float
    n_suites: int


def quantization_var(form, n_lin: int) -> np.ndarray:
    """Grid quantisation variance (du^2/12) per nonlinear coordinate; zero for linear coefficients.
    A discrete grid cannot resolve a coefficient below its node spacing (v1 Plans/02 amendment A1)."""
    q = np.zeros(n_lin + len(form.nl))
    for i, c in enumerate(form.nl):
        u = form.nl[c].to_coord(form.nl[c].nodes())
        q[n_lin + i] = (np.mean(np.diff(u)) ** 2) / 12.0
    return q


def _gls_mu(chats, Ws, Sigma):
    A = 0; b = 0
    for c, W in zip(chats, Ws):
        Pi = np.linalg.inv(Sigma + W)
        A = A + Pi; b = b + Pi @ c
    return np.linalg.solve(A, b)


def _unpack(theta, p):
    L = np.zeros((p, p))
    L[np.tril_indices(p)] = theta
    L[np.diag_indices(p)] = np.exp(L[np.diag_indices(p)])
    return L @ L.T


def _objective(theta, chats, Ws, nu, Psi, p):
    S = _unpack(theta, p)
    try:
        mu = _gls_mu(chats, Ws, S)
        val = 0.0
        for c, W in zip(chats, Ws):
            Lt = np.linalg.cholesky(S + W)
            r = np.linalg.solve(Lt, c - mu)
            val += -np.sum(np.log(np.diag(Lt))) - 0.5 * r @ r
        Ls = np.linalg.cholesky(S)
        Li = np.linalg.inv(Ls)
        val += -(nu + p + 1) * np.sum(np.log(np.diag(Ls))) - 0.5 * np.trace(Psi @ Li.T @ Li)
    except np.linalg.LinAlgError:
        return 1e300
    return -val if np.isfinite(val) else 1e300


def fit(chats, Ws, Psi=None, nu=None) -> Stage2Result:
    chats = [np.asarray(c, float) for c in chats]
    Ws = [np.asarray(W, float) for W in Ws]
    p = len(chats[0])
    if Psi is None:   # D1 (amended): training-suite scatter + fit noise; never the held-out suite
        scat = np.var(np.stack(chats), axis=0, ddof=1) if len(chats) > 1 else np.zeros(p)
        Psi = np.diag(scat) + np.mean(Ws, axis=0)
    nu = p + 2 if nu is None else nu
    assert np.all(np.linalg.eigvalsh(Psi) > 0), "IW scale Psi must be positive definite"
    S0 = np.diag(np.var(np.stack(chats), axis=0, ddof=1) if len(chats) > 1 else np.diag(Psi)) + Psi
    L0 = np.linalg.cholesky(S0)
    th0 = L0[np.tril_indices(p)].copy()
    di = [i for i, (r, c) in enumerate(zip(*np.tril_indices(p))) if r == c]
    th0[di] = np.log(th0[di])
    res = minimize(_objective, th0, args=(chats, Ws, nu, Psi, p), method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-9})
    S = _unpack(res.x, p)
    return Stage2Result(_gls_mu(chats, Ws, S), S, nu, Psi, bool(res.success), float(res.fun), len(chats))

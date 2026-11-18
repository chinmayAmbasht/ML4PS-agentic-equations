"""Held-out predictive evidence log Z_cross (spec 7.1-7.3; Plans/02 section 6, decision D2).

Per fold j:  Stage 1 + C_disc (2 alternations) on training suites only -> Stage 2 (mu, Sigma)
             Z_pred,j(R) = int N(c_R | mu_R, Sigma_RR) prod_b L_jb(c_R, c_-R(c_R); C_fv,j + C_disc) dc_R
The box product sits INSIDE the integral. Nothing is fitted, profiled or argmax'd on the held-out suite
except the D2 quadrature centring (`_adaptive_center`), a numerical device whose location is never stored.
"""
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import stage1 as S1
from . import stage2 as S2
from .cov import c_disc
from .par import pmap
from ..common import config as C
from .stage1 import BlockWhitener, Whitener, LOG2PI

N_ALTERNATIONS = 2
QUAD_NODES = {1: 17, 2: 17, 3: 13, 4: 9, 5: 7}     # registered per |R|; convergence checked at +4 (v1 Plans/02 A2)
QUAD_CONV_TOL = 0.01                               # nats


@dataclass
class SuiteBlock:
    """Scored boxes of one suite (LH, D_valid, features finite) plus its C_fv."""
    name: str
    obs: dict
    y: np.ndarray
    mask: np.ndarray
    cfv: np.ndarray


@dataclass
class FoldFit:
    held_out: object
    training: list
    s1: dict                     # suite -> Stage1Result (final round)
    cdisc: np.ndarray
    s2: S2.Stage2Result
    coeff_order: list
    cdisc_history: list = field(default_factory=list)


def fit_training(form, blocks: dict, training: list, redshifts, held_out=None) -> FoldFit:
    ny = next(iter(blocks.values())).y.shape[1]
    Cd = np.zeros((ny, ny))
    hist = []
    for it in range(N_ALTERNATIONS + 1):
        s1 = dict(zip(training, pmap(lambda s: S1.run(form, blocks[s].obs, blocks[s].y, blocks[s].cfv + Cd, redshifts,
                                                      blocks[s].mask), list(training))))   # Plans/12: suites independent
        if it == N_ALTERNATIONS:
            break
        res = {s: S1.residuals_at(form, blocks[s].obs, blocks[s].y, redshifts, s1[s].map_coeffs) for s in training}
        if C.CFV_MODE == "blockdiag":        # Plans/11: masked blocks never enter C_disc
            res = {s: np.where(blocks[s].mask, r, np.nan) for s, r in res.items()}
        Cd = c_disc(res, {s: np.diag(blocks[s].cfv) for s in training}, training, held_out)
        hist.append(np.diag(Cd).copy())
    order = s1[training[0]].coeff_order
    q = S2.quantization_var(form, len(form.lin))
    chats = [s1[s].c_hat for s in training]
    Ws = [s1[s].W + np.diag(q) for s in training]
    s2 = S2.fit(chats, Ws)
    return FoldFit(held_out, list(training), s1, Cd, s2, order, hist)


def to_native(form, order, u):
    """Coefficient vector in Stage-2 coordinates -> dict in native units."""
    d = {}
    for name, v in zip(order, u):
        d[name] = form.nl[name].from_coord(v) if name in form.nl else v
    return d


def make_loglik(form, block: SuiteBlock, Cpred, redshifts):
    nb, ny = block.y.shape
    if C.CFV_MODE == "blockdiag":
        # Plans/11: block-diagonal Cpred, ragged blocks: each box contributes only its valid z-blocks
        wt = BlockWhitener(Cpred, C.N_K, block.mask)
        const = -0.5 * wt.N * LOG2PI - 0.5 * wt.logdet_boxes
        valid = block.mask
    else:
        wt = Whitener(Cpred)
        const = -0.5 * nb * ny * LOG2PI - 0.5 * nb * wt.logdet
        valid = None
    yw = wt.w(block.y)

    def ll(coeffs: dict) -> float:
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                pred = form.predict(block.obs, redshifts, coeffs)
        except (OverflowError, ZeroDivisionError):
            return -np.inf                     # undefined prediction (Python-float overflow): excluded, never repaired
        if not np.all(np.isfinite(pred if valid is None else pred[valid])):
            return -np.inf                     # undefined prediction: excluded, never repaired
        r = yw - wt.w(pred)
        return const - 0.5 * float(np.einsum("by,by->", r, r))
    return ll


def conditional(mu, Sigma, R_idx):
    """Return (mu_R, Sigma_RR, function c_R -> full vector with c_-R at its conditional mean)."""
    p = len(mu)
    R = np.array(R_idx, int)
    O = np.array([i for i in range(p) if i not in set(R_idx)], int)
    muR, SRR = mu[R], Sigma[np.ix_(R, R)]
    if len(O) and len(R):
        B = Sigma[np.ix_(O, R)] @ np.linalg.inv(SRR)
    else:
        B = np.zeros((len(O), len(R)))

    def full(cR):
        c = np.empty(p)
        c[R] = cR
        if len(O):
            c[O] = mu[O] + (B @ (np.asarray(cR) - muR) if len(R) else 0.0)
        return c
    return muR, SRR, full


def conditional_precision_route(mu, Sigma, R_idx, cR):
    """Independent route (precision matrix) for gate item 8."""
    p = len(mu)
    R = np.array(R_idx, int); O = np.array([i for i in range(p) if i not in set(R_idx)], int)
    Lam = np.linalg.inv(Sigma)
    c = np.empty(p); c[R] = cR
    c[O] = mu[O] - np.linalg.solve(Lam[np.ix_(O, O)], Lam[np.ix_(O, R)] @ (np.asarray(cR) - mu[R]))
    return c


def _log_gauss(x, m, S):
    d = x - m
    sgn, ld = np.linalg.slogdet(S)
    return -0.5 * (len(m) * LOG2PI + ld + d @ np.linalg.solve(S, d))


def _adaptive_center(logf, x0, scale):
    """D2: Laplace centring for the quadrature only. Returns (centre, covariance) of a Gaussian proposal."""
    def obj(x):
        v = -logf(x)
        return v if np.isfinite(v) else 1e300   # centring device only: an undefined region is a wall, not data
    r = minimize(obj, x0, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
    m = r.x
    d = len(m); h = 1e-4 * np.maximum(scale, 1e-12)
    H = np.zeros((d, d))
    f0 = obj(m)
    for i in range(d):
        for j in range(i, d):
            ei = np.zeros(d); ej = np.zeros(d); ei[i] = h[i]; ej[j] = h[j]
            H[i, j] = H[j, i] = (obj(m + ei + ej) - obj(m + ei - ej) - obj(m - ei + ej) + obj(m - ei - ej)) / (4 * h[i] * h[j])
    w, V = np.linalg.eigh(H)
    w = np.where(w > 0, w, 1.0 / np.maximum(scale.max(), 1e-12) ** 2)   # guard non-PD Hessian (recorded)
    return m, V @ np.diag(1.0 / w) @ V.T, bool(np.all(np.linalg.eigvalsh(H) > 0))


def gh_integrate(logf, m, Cov, n):
    """log int exp(logf(x)) dx with Gauss-Hermite nodes on N(m, Cov). Returns (logI, weight at outer nodes)."""
    d = len(m)
    x1, w1 = hermgauss(n)
    A = np.linalg.cholesky(Cov)
    grids = np.meshgrid(*([x1] * d), indexing="ij")
    X = np.stack([g.ravel() for g in grids], axis=1)
    lw = np.sum(np.log(np.stack(np.meshgrid(*([w1] * d), indexing="ij"), 0).reshape(d, -1).T), axis=1)
    vals = np.array([logf(m + np.sqrt(2.0) * A @ x) for x in X])
    terms = lw + vals + np.sum(X ** 2, axis=1)
    logI = logsumexp(terms) + 0.5 * d * np.log(2.0) + np.sum(np.log(np.diag(A)))
    outer = np.any(np.abs(X) >= np.abs(x1).max() - 1e-12, axis=1)
    tail = float(np.exp(logsumexp(terms[outer]) - logsumexp(terms))) if outer.any() else 0.0
    return float(logI), tail


def logZ_pred(form, fold: FoldFit, block: SuiteBlock, R_names, redshifts, n_nodes=None, check_conv=True):
    order = fold.coeff_order
    R_idx = [order.index(c) for c in R_names]
    mu, Sig = fold.s2.mu, fold.s2.Sigma
    ll = make_loglik(form, block, block.cfv + fold.cdisc, redshifts)
    muR, SRR, full = conditional(mu, Sig, R_idx)
    if not R_idx:
        return {"logZ": ll(to_native(form, order, mu)), "n_nodes": 0, "tail_mass": 0.0, "converged": True,
                "conv_delta": 0.0, "hessian_pd": True}

    def logf(cR):
        return _log_gauss(cR, muR, SRR) + ll(to_native(form, order, full(cR)))

    m, Cov, pd = _adaptive_center(logf, muR.copy(), np.sqrt(np.diag(SRR)))
    n = n_nodes or QUAD_NODES[len(R_idx)]
    lz, tail = gh_integrate(logf, m, Cov, n)
    out = {"logZ": lz, "n_nodes": n, "tail_mass": tail, "hessian_pd": pd}
    if check_conv:
        lz2, _ = gh_integrate(logf, m, Cov, n + 4)
        out["conv_delta"] = abs(lz2 - lz)
        out["converged"] = out["conv_delta"] < QUAD_CONV_TOL
    return out

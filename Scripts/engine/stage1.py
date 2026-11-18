"""Stage 1: one coefficient vector per suite, fit JOINTLY across that suite's boxes (spec 6.1).

At each node eta_g of a product grid with a discrete uniform prior, the linear coefficients are integrated
analytically under N(mu_p, Sigma_p). Returns the suite evidence, node weights, and the FULL posterior mean and
covariance of c = (c_lin, u) where u is the Stage-2 coordinate of eta (log10 for log-spaced coefficients).
Per-box Stage 1 is deliberately not implemented (spec 6.1).
"""
import itertools
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import logsumexp

from ..common import config as C

LOG2PI = np.log(2.0 * np.pi)


@dataclass
class Stage1Result:
    logZ: float
    logZ_nodes: np.ndarray
    P: np.ndarray
    c_hat: np.ndarray            # (n_lin + n_nl,)
    W: np.ndarray                # full covariance, incl. lin-nl cross terms
    map_node: int
    map_coeffs: dict             # native units, for C_disc residuals
    n_boxes: int
    n_nodes: int
    coeff_order: list


def node_list(form):
    names = list(form.nl)
    axes = [form.nl[c].nodes() for c in names]
    return names, [dict(zip(names, v)) for v in itertools.product(*axes)] if names else [{}]


class Whitener:
    """Cholesky whitening of a fixed C_fit; also the masked (ragged) path, grouped by mask pattern."""

    def __init__(self, Cfit):
        self.C = Cfit
        self.L = np.linalg.cholesky(Cfit)
        self.logdet = 2.0 * np.sum(np.log(np.diag(self.L)))

    def w(self, A):
        """A: (n_b, n_y[, m]) -> whitened, same shape."""
        if A.ndim == 2:
            return solve_triangular(self.L, A.T, lower=True).T
        nb, ny, m = A.shape
        B = solve_triangular(self.L, A.transpose(1, 0, 2).reshape(ny, nb * m), lower=True)
        return B.reshape(ny, nb, m).transpose(1, 0, 2)


class BlockWhitener:
    """v1 Plans/11: block-diagonal C (one n_k x n_k block per z) with a per-entry availability mask that is constant
    within each z-block. Whitening is per block, then masked entries are set to exactly 0, so a masked block contributes
    nothing to any sum -- identical to deleting it. logdet_boxes: sum over each box's valid blocks."""

    def __init__(self, Cfit, n_k, mask):
        ny = Cfit.shape[0]
        self.n_k, self.n_z = n_k, ny // n_k
        assert self.n_z * n_k == ny
        Bm = np.kron(np.eye(self.n_z, dtype=bool), np.ones((n_k, n_k), bool))
        assert not np.any(Cfit[~Bm]), "BlockWhitener needs an exactly block-diagonal C"
        self.mask = np.asarray(mask, bool)
        mb = self.mask.reshape(len(self.mask), self.n_z, n_k)
        assert np.all(mb == mb[:, :, :1]), "mask must be constant within each z-block"
        self.bmask = mb[:, :, 0]                                     # (n_b, n_z)
        self.Ls = [np.linalg.cholesky(Cfit[i * n_k:(i + 1) * n_k, i * n_k:(i + 1) * n_k]) for i in range(self.n_z)]
        self.logdet_z = np.array([2.0 * np.sum(np.log(np.diag(L))) for L in self.Ls])
        self.logdet_boxes = float(np.sum(self.bmask * self.logdet_z[None, :]))
        self.N = int(self.mask.sum())

    def w(self, A):
        """A: (n_b, n_y[, m]) -> per-block whitened, masked entries exactly 0. Masked entries are zeroed BEFORE the
        (within-block) solve, and the mask is all-or-nothing per block, so L^-1 0 = 0 exactly and nothing leaks."""
        mk = self.mask if A.ndim == 2 else self.mask[:, :, None]
        A = np.where(mk, A, 0.0)
        out = np.empty(A.shape)
        nk = self.n_k
        for i, L in enumerate(self.Ls):
            sl = slice(i * nk, (i + 1) * nk)
            if A.ndim == 2:
                out[:, sl] = solve_triangular(L, A[:, sl].T, lower=True).T
            else:
                nb, _, m = A.shape
                B = solve_triangular(L, A[:, sl, :].transpose(1, 0, 2).reshape(nk, nb * m), lower=True)
                out[:, sl, :] = B.reshape(nk, nb, m).transpose(1, 0, 2)
        return out


def _node_terms(form, obs, redshifts, eta, yw, wt):
    h0, X = form.design(obs, redshifts, eta)
    rw = yw - wt.w(h0)
    Xw = wt.w(X) if X.shape[-1] else X
    XtX = np.einsum("bym,byn->mn", Xw, Xw)
    Xtr = np.einsum("bym,by->m", Xw, rw)
    rr = float(np.einsum("by,by->", rw, rw))
    return XtX, Xtr, rr


def _groups(mask):
    """Group boxes by identical availability pattern (ragged path, spec 8.6)."""
    pats, inv = np.unique(mask, axis=0, return_inverse=True)
    return [(pats[i], np.flatnonzero(inv.ravel() == i)) for i in range(len(pats))]


def run(form, obs, y, Cfit, redshifts, mask=None, force_masked=False) -> Stage1Result:
    """obs: dict of feature arrays (n_b,), y: (n_b, n_y), Cfit: (n_y, n_y)."""
    nb, ny = y.shape
    mu_p, Sig_p = form.lin_prior()
    nlin = len(form.lin)
    if C.CFV_MODE == "blockdiag" and not force_masked:
        # Plans/11: one group; per-block whitening with masked blocks zeroed (== deleting them)
        m = np.ones((nb, ny), bool) if mask is None else np.asarray(mask, bool)
        assert m.any(axis=1).all(), "a box with no valid block reached Stage 1"
        wt = BlockWhitener(Cfit, C.N_K, m)
        prep = [(np.ones(ny, bool), np.arange(nb), wt, obs, wt.w(np.where(m, y, 0.0)))]
        return _run_nodes(form, prep, False, wt.N, wt.logdet_boxes, redshifts, nb, mu_p, Sig_p, nlin)
    if mask is None or (mask.all() and not force_masked):
        groups = [(np.ones(ny, bool), np.arange(nb))]
        masked = False
    else:
        groups = _groups(mask)
        masked = True
    prep = []
    for pat, idx in groups:
        sub = np.ix_(pat, pat)
        wt = Whitener(Cfit[sub])
        o = {kk: v[idx] for kk, v in obs.items()}
        yw = wt.w(y[np.ix_(idx, pat)])
        prep.append((pat, idx, wt, o, yw))
    N_tot = sum(len(idx) * pat.sum() for pat, idx in groups)
    logdetC_tot = sum(len(idx) * wt.logdet for pat, idx, wt, o, yw in prep)
    return _run_nodes(form, prep, masked, N_tot, logdetC_tot, redshifts, nb, mu_p, Sig_p, nlin)


def _run_nodes(form, prep, masked, N_tot, logdetC_tot, redshifts, nb, mu_p, Sig_p, nlin):
    names, nodes = node_list(form)
    logZ = np.empty(len(nodes))
    M = np.zeros((len(nodes), nlin)); V = np.zeros((len(nodes), nlin, nlin))
    if nlin:
        Sp_inv = np.linalg.inv(Sig_p)
        logdet_Sp = np.linalg.slogdet(Sig_p)[1]
        pr = float(mu_p @ Sp_inv @ mu_p)
    for g, eta in enumerate(nodes):
        XtX = np.zeros((nlin, nlin)); Xtr = np.zeros(nlin); rr = 0.0
        for pat, idx, wt, o, yw in prep:
            if not masked:
                a, b, c = _node_terms(form, o, redshifts, eta, yw, wt)
            else:
                a, b, c = _node_terms_masked(form, o, redshifts, eta, yw, wt, pat)
            XtX += a; Xtr += b; rr += c
        base = -0.5 * N_tot * LOG2PI - 0.5 * logdetC_tot
        if nlin:
            Lam = Sp_inv + XtX
            q = Sp_inv @ mu_p + Xtr
            cf = cho_factor(Lam, lower=True)
            m = cho_solve(cf, q)
            logdet_L = 2.0 * np.sum(np.log(np.diag(cf[0])))
            logZ[g] = base - 0.5 * logdet_Sp - 0.5 * logdet_L - 0.5 * ((rr + pr) - q @ m)
            M[g] = m
            V[g] = cho_solve(cf, np.eye(nlin))
        else:
            logZ[g] = base - 0.5 * rr
    lse = logsumexp(logZ)
    P = np.exp(logZ - lse)
    U = np.array([[form.nl[c].to_coord(nd[c]) for c in names] for nd in nodes]) if names else np.zeros((len(nodes), 0))
    Cg = np.concatenate([M, U], axis=1)
    c_hat = P @ Cg
    D = Cg - c_hat
    W = np.einsum("g,gi,gj->ij", P, D, D)
    W[:nlin, :nlin] += np.einsum("g,gij->ij", P, V)
    gm = int(np.argmax(logZ))
    mapc = {c: float(M[gm, i]) for i, c in enumerate(form.lin)}
    mapc.update({c: float(nodes[gm][c]) for c in names})
    return Stage1Result(float(lse - np.log(len(nodes))), logZ, P, c_hat, W, gm, mapc, nb, len(nodes),
                        list(form.lin) + names)


def _node_terms_masked(form, o, redshifts, eta, yw, wt, pat):
    h0, X = form.design(o, redshifts, eta)
    h0 = h0[:, pat]; X = X[:, pat, :]
    rw = yw - wt.w(h0)
    Xw = wt.w(X) if X.shape[-1] else X
    return (np.einsum("bym,byn->mn", Xw, Xw), np.einsum("bym,by->m", Xw, rw),
            float(np.einsum("by,by->", rw, rw)))


def residuals_at(form, obs, y, redshifts, coeffs):
    return y - form.predict(obs, redshifts, coeffs)

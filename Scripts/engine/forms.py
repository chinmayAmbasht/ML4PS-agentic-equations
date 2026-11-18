"""Candidate functional forms: parsing, linearity check, basis evaluation (v1 Plans/02 section 1).

A form is linear in `lin` and arbitrary in `nl`:   S = h0(k, z, x; eta) + sum_j c_j g_j(k, z, x; eta).
"""
import hashlib
import json
from dataclasses import dataclass, field, asdict

import numpy as np
import sympy as sp

from ..common import config as C

# feature symbols a form may use; f_b_M_<i> are the per-mass-bin stacked f_b (stacks/f_baryon_200c)
BASE_FEATURES = ("D", "f_b", "Omega_m", "sigma_8", "Omega_b_over_Omega_m")
N_FB_M_BINS = 6
FB_M_FEATURES = tuple(f"f_b_M_{i}" for i in range(N_FB_M_BINS))
ALLOWED_FEATURES = BASE_FEATURES + FB_M_FEATURES
ROLES = ("amplitude", "scale", "exponent", "redshift-evolution", "other")


@dataclass
class NLSpec:
    lo: float
    hi: float
    spacing: str          # "log" | "lin"
    n_nodes: int = 41          # registered 2026-10-09 (was 21); Plans/02 amendment A5

    def nodes(self) -> np.ndarray:
        assert self.spacing in ("log", "lin") and self.hi > self.lo and 2 <= self.n_nodes <= 41
        if self.spacing == "log":
            assert self.lo > 0
            return np.logspace(np.log10(self.lo), np.log10(self.hi), self.n_nodes)
        return np.linspace(self.lo, self.hi, self.n_nodes)

    def to_coord(self, v):       # Stage-2 / evidence coordinate
        return np.log10(v) if self.spacing == "log" else v

    def from_coord(self, u):
        return 10.0 ** u if self.spacing == "log" else u


@dataclass
class Form:
    form_id: str
    expr: str
    lin: list
    nl: dict                     # name -> NLSpec (or dict)
    roles: dict                  # coeff name -> role
    features: list = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        self.nl = {k: (v if isinstance(v, NLSpec) else NLSpec(**v)) for k, v in self.nl.items()}
        self._compile()

    # ---- identity ----------------------------------------------------------------------------
    def canonical(self) -> str:
        return json.dumps({"expr": self.expr, "lin": list(self.lin),
                           "nl": {k: asdict(v) for k, v in sorted(self.nl.items())},
                           "roles": dict(sorted(self.roles.items())), "features": sorted(self.features)},
                          sort_keys=True)

    def sha256(self) -> str:
        return hashlib.sha256(self.expr.encode()).hexdigest()   # spec 1.4: hash of the expression string

    @property
    def coeff_names(self) -> list:
        return list(self.lin) + list(self.nl)

    @property
    def n_coeff(self) -> int:
        return len(self.coeff_names)

    # ---- compile -----------------------------------------------------------------------------
    def _compile(self):
        k, z = sp.symbols("k z", positive=True)
        feats = {f: sp.Symbol(f, positive=True) for f in ALLOWED_FEATURES}
        csyms = {c: sp.Symbol(c, real=True) for c in self.coeff_names}
        for c in self.nl:
            csyms[c] = sp.Symbol(c, positive=(self.nl[c].spacing == "log"), real=True)
        bad = set(self.features) - set(ALLOWED_FEATURES)
        assert not bad, f"{self.form_id}: features not permitted: {bad}"
        for f in self.features:
            assert not any(x in f for x in C.FORBIDDEN_FEATURES[:4]), f"forbidden feature {f}"
        assert set(self.roles) == set(self.coeff_names), f"{self.form_id}: roles must cover every coefficient"
        assert all(r in ROLES for r in self.roles.values())
        assert not set(self.lin) & set(self.nl)
        local = {"k": k, "z": z, **{f: feats[f] for f in self.features}, **csyms}
        e = sp.sympify(self.expr, locals=local)
        used = {s.name for s in e.free_symbols}
        allowed = {"k", "z"} | set(self.features) | set(self.coeff_names)
        assert used <= allowed, f"{self.form_id}: unknown symbols {used - allowed}"
        missing = set(self.coeff_names) - used
        assert not missing, f"{self.form_id}: declared coefficients not in expression: {missing}"
        lin_s = [csyms[c] for c in self.lin]
        for a in lin_s:
            for b in lin_s:
                assert sp.simplify(sp.diff(e, a, b)) == 0, f"{self.form_id}: not linear in {a},{b}"
        basis = [sp.diff(e, a) for a in lin_s]
        h0 = sp.simplify(e.subs({a: 0 for a in lin_s}))
        args = [k, z] + [feats[f] for f in self.features] + [csyms[c] for c in self.nl]
        self._sym = {"k": k, "z": z, "feats": feats, "coeffs": csyms, "expr": e}
        self._f_basis = [sp.lambdify(args, g, "numpy") for g in basis]
        self._f_h0 = sp.lambdify(args, h0, "numpy")
        self._f_full = sp.lambdify(args + lin_s, e, "numpy")

    # ---- evaluation --------------------------------------------------------------------------
    @staticmethod
    def grid_kz(redshifts):
        kk = np.tile(np.array(C.K_GRID), len(redshifts))          # k fastest within each z block
        zz = np.repeat(np.array(redshifts, float), C.N_K)
        return kk, zz

    def _args(self, obs, redshifts, eta):
        kk, zz = self.grid_kz(redshifts)
        fa = [_feat_cols(obs[f], C.N_K) for f in self.features]
        return [kk[None, :], zz[None, :]] + fa + [float(eta[c]) for c in self.nl]

    def design(self, obs, redshifts, eta):
        """Return (h0, X): h0 (n_b, n_y), X (n_b, n_y, n_lin), at nonlinear values eta (dict, native units)."""
        a = self._args(obs, redshifts, eta)
        nb = len(next(iter(obs.values())))
        ny = C.N_K * len(redshifts)
        h0 = np.broadcast_to(np.asarray(self._f_h0(*a), float), (nb, ny)).copy()
        X = np.stack([np.broadcast_to(np.asarray(g(*a), float), (nb, ny)) for g in self._f_basis], axis=-1) \
            if self.lin else np.zeros((nb, ny, 0))
        return h0, X

    def predict(self, obs, redshifts, coeffs):
        """coeffs: dict of every coefficient in native units."""
        h0, X = self.design(obs, redshifts, {c: coeffs[c] for c in self.nl})
        cl = np.array([coeffs[c] for c in self.lin])
        return h0 + (X @ cl if self.lin else 0.0)

    def feature_ok(self, obs) -> np.ndarray:
        """Boxes whose used features are all finite (undefined predictions are excluded, never repaired).
        Plans/11: with per-z features (n_b, n_z) a box is kept if at least one z-block has every feature finite."""
        bo = self.feature_block_ok(obs)
        return bo if bo.ndim == 1 else bo.any(axis=1)

    def feature_block_ok(self, obs) -> np.ndarray:
        """(n_b,) for box-level features only, else (n_b, n_z): blocks whose used features are all finite."""
        nb = len(next(iter(obs.values())))
        ok = np.ones(nb, bool)
        for f in self.features:
            a = np.isfinite(np.asarray(obs[f], float))
            if a.ndim > ok.ndim:
                ok = ok[:, None]
            ok = ok & (a[:, None] if a.ndim < ok.ndim else a)
        return ok

    # ---- priors ------------------------------------------------------------------------------
    def lin_prior(self):
        mu = np.array([C.MU_A if self.roles[c] == "amplitude" else 0.0 for c in self.lin])
        return mu, np.diag(np.full(len(self.lin), C.SIGMA_A ** 2))

    def to_record(self) -> dict:
        return {"form_id": self.form_id, "expr": self.expr, "lin": list(self.lin),
                "nl": {k: asdict(v) for k, v in self.nl.items()}, "roles": self.roles,
                "features": list(self.features), "notes": self.notes}


def obs_with_fbm(obs: dict) -> dict:
    """Expose per-mass-bin f_b columns as f_b_M_<i> features ((n_b, 6) or, Plans/11, (n_b, n_z, 6))."""
    out = dict(obs)
    if "f_b_M" in obs:
        for i in range(obs["f_b_M"].shape[-1]):
            out[f"f_b_M_{i}"] = obs["f_b_M"][..., i]
    return out


def _feat_cols(v, n_k):
    """Feature -> broadcastable against (n_b, n_y). Box-level (n_b,) -> (n_b, 1), unchanged. Per-z (n_b, n_z) (v1 Plans/11)
    -> repeated over k to (n_b, n_z*n_k), k fastest; non-finite (invalid-block) entries get a finite placeholder (the
    feature's median over valid entries) so masked blocks stay finite -- they are masked out of every likelihood."""
    a = np.asarray(v, float)
    if a.ndim == 1:
        return a[:, None]
    assert a.ndim == 2
    fin = np.isfinite(a)
    if not fin.all():
        a = np.where(fin, a, np.median(a[fin]) if fin.any() else 1.0)
    return np.repeat(a, n_k, axis=1)

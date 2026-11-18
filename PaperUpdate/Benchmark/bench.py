"""Fig. 3 benchmark (Plans/03): correlated L (PaperUpdate/Lcorr.py: own-suite C_fv / Hartlap h) for baselines and the best
analytic forms, three settings. (Superseded diagonal-sigma_k results: bench.json.)

Boxes: the z = 0 LH boxes of every suite that the engine scores (Scripts.data.suppression_io.load, D valid), 16 bins.
L(set) = mean over suites j of the mean over boxes b of j of dS_b^T (C_fv,j/h)^-1 dS_b.

Settings
  within   : 5-fold CV inside each suite (train and test on the same simulator)
  pooled   : 5-fold CV over the 4 suites pooled (test boxes from simulators seen in training)
  transfer : leave-one-suite-out (train on 3 suites, predict the 4th)

Models
  RF            : scikit-learn RandomForestRegressor (300 trees, min_samples_leaf=5, seed 0), multi-output over the 16 bins;
                  features f_b/(Omega_b/Omega_m), D, Omega_m, sigma_8, Omega_b/Omega_m and the 6 mass-binned f_b(M)/(Omega_b/Omega_m)
                  (Pandey+23-style; NaN mass bins -> -1). No target data at test time.
  vD20 published: van Daalen+20 Eq. 5 with the Table-3 f_bar,200c constants (a=2.111, b=0.0038, c=1.371, d=-5.816, e=-0.4005);
                  our f_b,200c uses a 5e12-1e14 Msun mask rather than their 6e13-2e14 Msun. Calibrated for k <= 1 only.
  vD20 refit    : same functional form, (a, b, c, d, e) refit by correlated (C_fv/h-whitened) least squares on the training boxes.
  analytic forms: population fit of all coefficients by correlated (C_fv/h-whitened) least squares on the training boxes (variable
                  projection: linear coefficients solved exactly, nonlinear ones by a start grid + Nelder-Mead) for
                  within / pooled / transfer-population. Their transfer zero-shot and per-box MAP use the registered pipeline
                  (fold prior from Stage 2, PaperUpdate/preds/<key>.npz).

Output: PaperUpdate/Benchmark/bench.json.  Usage: python -m PaperUpdate.Benchmark.bench [form keys ...]
"""
import itertools
import json
import os
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[2]
os.environ.setdefault("PK_NS_DIR", os.path.join(os.environ.get("ML4PS_CACHE", "cache"), "ml4ps_fits", f"ns_B"))
os.environ.setdefault("PK_NPAR", "1")
sys.path.insert(0, str(M))

import numpy as np
from scipy.optimize import least_squares, minimize

from Scripts.common import config as C
from Scripts.data import suppression_io as IO

OUT = M / "PaperUpdate" / "Benchmark" / "bench_corr.json"
from PaperUpdate import Lcorr as LC
K = np.asarray(C.K_GRID)
SEED = 0
VD200 = dict(a=2.111, b=0.0038, c=1.371, d=-5.816, e=-0.4005)


def data():
    out = {}
    for s in C.SUITES:
        d = IO.load(s, "LH")
        o = {k: np.asarray(v, float).reshape(len(d.y), -1)[:, 0] if np.ndim(v) <= 2 else np.asarray(v, float)[:, 0, :]
             for k, v in d.obs.items() if k in ("f_b", "D", "Omega_m", "sigma_8", "Omega_b_over_Omega_m", "f_b_M")}
        ok = d.mask.all(axis=1) & np.isfinite(o["D"]) & np.isfinite(o["f_b"])
        o = {k: v[ok] for k, v in o.items()}
        o["_suite"] = np.full(int(ok.sum()), s, dtype=object)
        out[s] = (o, d.y[ok])
    return out


def feats(o):
    w = o["Omega_b_over_Omega_m"]
    fm = np.where(np.isfinite(o["f_b_M"]), o["f_b_M"] / w[:, None], -1.0)
    return np.column_stack([o["f_b"] / w, o["D"], o["Omega_m"], o["sigma_8"], w, fm])


def L(pred, y, suite):
    return LC.L(pred, y, suite)


def wres(o, d):
    """Per-box correlated whitening of residuals d (n_b, 16) by each box's own suite C_fv / h."""
    out = np.empty_like(d)
    for s in np.unique(o["_suite"]):
        m = o["_suite"] == s
        out[m] = LC.whiten(d[m], s)
    return out


def cat(parts):
    o = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    return o, np.concatenate([p[1] for p in parts])


def sub(o, i):
    return {k: v[i] for k, v in o.items()}


# ---------------- models: fit(o, y) -> predictor(o) ----------------
def rf_fit(o, y):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=6).fit(feats(o), y)
    return lambda q: m.predict(feats(q))


def vd_pred(o, p):
    ft = o["f_b"] / o["Omega_b_over_Omega_m"]
    X = (p["c"] * ft[:, None]) ** (p["b"] - p["a"])
    with np.errstate(all="ignore"):
        r = (2 ** p["a"] + 2 ** p["b"] * X) / (K[None, :] ** -p["a"] + X * K[None, :] ** -p["b"])
        return -r * np.exp(p["d"] * ft[:, None] + p["e"])


def vd_pub_fit(o, y):
    return lambda q: vd_pred(q, VD200)


def vd_refit(o, y):
    names = list(VD200)
    def res(v):
        r = wres(o, vd_pred(o, dict(zip(names, v))) - y).ravel()
        return np.where(np.isfinite(r), r, 1e3)
    v = least_squares(res, [VD200[n] for n in names], method="trf", max_nfev=4000).x
    p = dict(zip(names, v))
    return lambda q: vd_pred(q, p)


def form_fit_factory(sub_json):
    from Loop.plot_fits import form_of
    form = form_of("bench", sub_json, 11)
    nl = list(form.nl)

    def obs(o):
        return {f: o[f] for f in form.features}

    def lin_solve(o, y, eta):
        h0, X = form.design(obs(o), C.REDSHIFTS, eta)
        if not (np.isfinite(h0).all() and np.isfinite(X).all()):
            return None, 1e30
        r = wres(o, y - h0)
        if not form.lin:
            return np.zeros(0), float(np.sum(r ** 2))
        Xw = np.stack([wres(o, X[..., i]) for i in range(X.shape[-1])], axis=-1)
        A = Xw.reshape(-1, Xw.shape[-1]); b = r.ravel()
        good = np.isfinite(A).all(axis=1) & np.isfinite(b)
        if good.mean() < 0.99:
            return None, 1e30
        c, *_ = np.linalg.lstsq(A[good], b[good], rcond=None)
        return c, float(np.sum((b[good] - A[good] @ c) ** 2))

    def to_eta(u):
        return {n: form.nl[n].from_coord(x) for n, x in zip(nl, u)}

    def fit(o, y):
        grids = [np.linspace(form.nl[n].to_coord(form.nl[n].lo), form.nl[n].to_coord(form.nl[n].hi), 6)[1:-1] for n in nl]
        best = (np.inf, None)
        for u in itertools.product(*grids):
            _, s = lin_solve(o, y, to_eta(u))
            if s < best[0]:
                best = (s, np.array(u))
        if nl:
            u = minimize(lambda u: lin_solve(o, y, to_eta(u))[1], best[1], method="Nelder-Mead",
                         options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 3000}).x
        else:
            u = np.zeros(0)
        eta = to_eta(u); c, _ = lin_solve(o, y, eta)
        coeffs = dict(eta) | dict(zip(form.lin, c))
        return lambda q: form.predict(obs(q), C.REDSHIFTS, coeffs)
    return fit


# ---------------- settings ----------------
def kfold(n, k=5):
    idx = np.random.default_rng(SEED).permutation(n)
    return np.array_split(idx, k)


def run_within(fitter, D):
    out = {}
    for s, (o, y) in D.items():
        pred = np.full_like(y, np.nan)
        for te in kfold(len(y)):
            tr = np.setdiff1d(np.arange(len(y)), te)
            pred[te] = fitter(sub(o, tr), y[tr])(sub(o, te))
        out[s] = L(pred, y, s)
    return out


def run_pooled(fitter, D):
    o, y = cat(list(D.values()))
    lab = np.concatenate([[s] * len(D[s][1]) for s in D])
    pred = np.full_like(y, np.nan)
    for te in kfold(len(y)):
        tr = np.setdiff1d(np.arange(len(y)), te)
        pred[te] = fitter(sub(o, tr), y[tr])(sub(o, te))
    return {s: L(pred[lab == s], y[lab == s], s) for s in D}


def run_transfer(fitter, D):
    out = {}
    for s in D:
        o, y = cat([D[t] for t in D if t != s])
        out[s] = L(fitter(o, y)(D[s][0]), D[s][1], s)
    return out


def pipeline_transfer(key):
    p = M / "PaperUpdate" / "preds" / f"{key}.npz"
    if not p.exists():
        return None
    z = np.load(p); r = {"zero_shot": {}, "map": {}}
    for s in C.SUITES:
        y = z[f"{s}__y"]; m = np.isfinite(y).all(axis=1)
        r["zero_shot"][s] = L(z[f"{s}__zs"][m], y[m], s); r["map"][s] = L(z[f"{s}__map"][m], y[m], s)
    return r


def summarize(d):
    return {"per_suite": d, "mean": float(np.mean(list(d.values())))}


def main(keys):
    D = data()
    print({s: len(v[1]) for s, v in D.items()}, flush=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["_meta"] = {"loss": "correlated: dS^T (C_fv,j/h)^-1 dS, own-suite C_fv, h=%.4f" % LC.H, "n_boxes": {s: len(v[1]) for s, v in D.items()}, "seed": SEED}
    models = {"RF": rf_fit, "vD20 published": vd_pub_fit, "vD20 refit": vd_refit}
    R = json.loads((M / "PaperUpdate" / "ranking.json").read_text())
    subs = {r["key"]: r["submission"] for r in R}
    for k in keys:
        models[k] = form_fit_factory(json.loads((M / subs[k]).read_text()))
    for name, f in models.items():
        if name in res and name not in keys:
            continue
        r = {"within": summarize(run_within(f, D)), "pooled": summarize(run_pooled(f, D)),
             "transfer_population": summarize(run_transfer(f, D))}
        if name in keys:
            pt = pipeline_transfer(name)
            if pt:
                r["transfer_zero_shot"] = summarize(pt["zero_shot"]); r["transfer_map"] = summarize(pt["map"])
        res[name] = r
        OUT.write_text(json.dumps(res, indent=1))
        print(name, {k: round(v["mean"], 2) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

"""Per-held-out-suite reduced chi^2 (v2 presentable bar). Ported from v1's zfb_post.chi2 and Refinement screen._chi2.

For each leave-one-suite-out fold j (trained on the other 3 suites), each box of the held-out suite j gets its MAP
coefficients under the box likelihood (C_fv + diag C_disc) times the fold prior N(mu_j, 1.30 Sigma_j), Nelder-Mead from
mu_j (= v1 scoring.map_diagnostic_widened_prior, masked path). Residuals are then whitened by C_fv only (the registered
bar) and by C_fv + C_disc (reported). chi2_nu[j] = sum r^2 / sum (n_valid - n_coeff) over suite j's boxes.

Presentable (Plans/00 sec. 5): mean_j chi2_nu (C_fv) <= C.CHI2_MEAN_MAX and std_j (ddof=0, 4 suites) <= C.CHI2_STD_MAX.
"""
import numpy as np
from scipy.optimize import minimize

from ..common import config as C
from . import evidence as E
from . import stage1 as S1
from .par import pmap


def _suite(args):
    form, f, b, redshifts = args
    W = C.PRIOR_WIDENING_FOR_MAP_DIAGNOSTIC
    mu, Si = np.asarray(f["mu"], float), np.linalg.inv(W * np.asarray(f["Sigma"], float))
    cd = np.diag(f["cdisc_diag"])
    rfv = rall = 0.0; nu = 0
    sq_k = np.zeros((len(redshifts), C.N_K)); sg_k = np.zeros((len(redshifts), C.N_K)); n_k = np.zeros((len(redshifts), C.N_K))
    for i in range(len(b.y)):
        o = {k: v[[i]] for k, v in b.obs.items()}; mi = b.mask[[i]]
        wt = S1.BlockWhitener(b.cfv + cd, C.N_K, mi); wf = S1.BlockWhitener(b.cfv, C.N_K, mi)
        yw = wt.w(b.y[[i]])

        def neg(c):
            pred = form.predict(o, redshifts, E.to_native(form, f["order"], c))
            if not np.all(np.isfinite(pred[mi])):
                return 1e30
            r = yw - wt.w(pred); d = c - mu
            return 0.5 * float(np.sum(r * r)) + 0.5 * float(d @ Si @ d)
        c = minimize(neg, mu.copy(), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000}).x
        pred = form.predict(o, redshifts, E.to_native(form, f["order"], c))
        r_all = yw - wt.w(pred); r_fv = wf.w(b.y[[i]]) - wf.w(pred)
        rall += float(np.sum(r_all ** 2)); rfv += float(np.sum(r_fv ** 2))
        nu += int(mi.sum()) - form.n_coeff
        # residual table (diagonal-whitened by C_fv, per (z, k)): the loop agent's feedback, never a score
        d = np.where(mi[0], (b.y[i] - pred[0]) / np.sqrt(np.diag(b.cfv)), np.nan).reshape(len(redshifts), C.N_K)
        ok = np.isfinite(d)
        sg_k[ok] += d[ok]; sq_k[ok] += d[ok] ** 2; n_k[ok] += 1
    with np.errstate(invalid="ignore"):
        return {"chi2_nu_Cfv": rfv / nu, "chi2_nu_Cfv_Cdisc": rall / nu, "nu": nu, "n_box": len(b.y),
                "resid_mean_signed": (sg_k / n_k).tolist(), "resid_mean_sq": (sq_k / n_k).tolist()}


def per_suite(form, fold_par, blocks, redshifts):
    """fold_par: {suite j: {"mu", "Sigma", "order", "cdisc_diag"}} (the stored-fold form). Suites run in parallel (PK_NPAR)."""
    js = list(fold_par)
    res = dict(zip(js, pmap(_suite, [(form, fold_par[j], blocks[j], redshifts) for j in js])))
    v = np.array([res[j]["chi2_nu_Cfv"] for j in js])
    out = {"per_suite": {j: res[j]["chi2_nu_Cfv"] for j in js},
           "per_suite_Cfv_Cdisc": {j: res[j]["chi2_nu_Cfv_Cdisc"] for j in js},
           "mean": float(v.mean()), "std": float(v.std()),
           "pooled": float(sum(res[j]["chi2_nu_Cfv"] * res[j]["nu"] for j in js) / sum(res[j]["nu"] for j in js)),
           "residuals": {j: {k: res[j][k] for k in ("resid_mean_signed", "resid_mean_sq", "n_box")} for j in js},
           "prediction": "per-box MAP, fold prior x1.30; whitened by C_fv (bar) and C_fv + C_disc (reported)"}
    out["presentable"] = bool(out["mean"] <= C.CHI2_MEAN_MAX and out["std"] <= C.CHI2_STD_MAX)
    return out


def fold_params(folds):
    """FoldFit objects (scoring.cross_suite_folds) -> the stored-fold dict used here and by gate0.upturn."""
    return {j: {"mu": f.s2.mu.tolist(), "Sigma": f.s2.Sigma.tolist(), "order": f.coeff_order,
                "cdisc_diag": np.diag(f.cdisc).tolist(), "s2_converged": f.s2.converged,
                "stage1": {s: {"logZ": r.logZ, "c_hat": r.c_hat.tolist(), "W": r.W.tolist(),
                               "max_P": float(r.P.max()), "map_node": r.map_node} for s, r in f.s1.items()}}
            for j, f in folds.items()}

"""Regimes, promotion ladder, and diagnostic L (spec 7.3-7.5; Plans/02 section 6).

cross-suite : the ONLY selecting regime (log Z_cross = sum_j log Z_pred,j)
same-suite  : per suite, SEED split of LH boxes; fit on one half, integrate over N(c_hat, W) on the other
pooled      : fit on all suites, evaluate each suite in-sample (optimistic bound)
"""
import numpy as np

from ..common import config as C
from . import evidence as E
from . import stage1 as S1
from .par import pmap


def sigma_fold(vals):
    return float(np.std(vals, ddof=1))            # D3


def cross_suite_folds(form, blocks, redshifts, suites=None):
    suites = list(suites or blocks)
    fits = pmap(lambda j: E.fit_training(form, blocks, [s for s in suites if s != j], redshifts, held_out=j), suites)
    return dict(zip(suites, fits))


def logZ_cross(form, folds, blocks, R, redshifts):
    return logZ_cross_many(form, folds, blocks, [R], redshifts)[0]


def logZ_cross_many(form, folds, blocks, Rs, redshifts):
    """log Z_cross for several marginalization sets; the (R, fold) units are independent (v1 Plans/12 pmap)."""
    js = list(folds)
    units = [(i, j) for i in range(len(Rs)) for j in js]
    res = pmap(lambda u: E.logZ_pred(form, folds[u[1]], blocks[u[1]], Rs[u[0]], redshifts), units)
    out = []
    for i, R in enumerate(Rs):
        per = {j: res[i * len(js) + n] for n, j in enumerate(js)}
        vals = [per[j]["logZ"] for j in folds]
        out.append({"logZ_cross": float(np.sum(vals)), "per_fold": per, "sigma_fold": sigma_fold(vals), "R": list(R)})
    return out


def ladder(form, folds, blocks, redshifts):
    """Forward one-at-a-time promotion from R = {} (spec 7.4). Never argmax over subsets."""
    names = form.coeff_names
    cap = min(C.LADDER_MAX_COEFF, len(names))
    cur = logZ_cross(form, folds, blocks, [], redshifts)
    steps = [{"R": [], "logZ_cross": cur["logZ_cross"], "sigma_fold": cur["sigma_fold"],
              "per_fold": {j: v["logZ"] for j, v in cur["per_fold"].items()}, "promoted": True,
              "conv": {j: v.get("converged", True) for j, v in cur["per_fold"].items()}}]
    tried = {}
    R = []
    while len(R) < cap:
        cands = {}
        todo = [c for c in names if c not in R]
        for c, r in zip(todo, logZ_cross_many(form, folds, blocks, [R + [c] for c in todo], redshifts)):
            cands[c] = r
            tried["+".join(R + [c])] = r["logZ_cross"]
        best = max(cands, key=lambda c: (cands[c]["logZ_cross"], -names.index(c)))
        gain = cands[best]["logZ_cross"] - cur["logZ_cross"]
        ok = gain > cur["sigma_fold"]
        steps.append({"R": R + [best], "logZ_cross": cands[best]["logZ_cross"],
                      "sigma_fold": cands[best]["sigma_fold"], "gain": gain,
                      "threshold_sigma_fold_of_simpler": cur["sigma_fold"], "promoted": bool(ok),
                      "per_fold": {j: v["logZ"] for j, v in cands[best]["per_fold"].items()},
                      "conv": {j: v.get("converged", True) for j, v in cands[best]["per_fold"].items()},
                      "tail": {j: v.get("tail_mass", 0.0) for j, v in cands[best]["per_fold"].items()},
                      "candidates": {c: cands[c]["logZ_cross"] for c in cands}})
        if not ok:
            break
        R = R + [best]
        cur = cands[best]
    return {"selected_R": R, "steps": steps, "logZ_cross": steps[[s["R"] for s in steps].index(R)]["logZ_cross"]
            if R else steps[0]["logZ_cross"], "all_tried": tried}


def same_suite(form, blocks, redshifts, R, seed=C.SEED):
    """Noise-floor regime: fit Stage 1 on a random half, evaluate the other half integrating over N(c_hat, W)."""
    from .stage2 import Stage2Result, quantization_var
    rng = np.random.default_rng(seed)
    perms = {s: rng.permutation(len(b.y)) for s, b in blocks.items()}      # drawn in suite order, as before

    def one(s):
        b, perm = blocks[s], perms[s]
        tr, ev = perm[: len(perm) // 2], perm[len(perm) // 2:]
        sub = lambda idx: E.SuiteBlock(s, {k: v[idx] for k, v in b.obs.items()}, b.y[idx], b.mask[idx], b.cfv)
        btr, bev = sub(tr), sub(ev)
        f = E.fit_training(form, {s: btr}, [s], redshifts)
        # predictive for the held-out half is the suite's own Stage-1 posterior (not a population)
        q = quantization_var(form, len(form.lin))
        f.s2 = Stage2Result(f.s1[s].c_hat, f.s1[s].W + np.diag(q), 0, np.zeros(1), True, 0.0, 1)
        return E.logZ_pred(form, f, bev, list(R), redshifts, check_conv=False)["logZ"]
    ss = list(blocks)
    out = dict(zip(ss, pmap(one, ss)))
    return {"per_suite": out, "S_same_suite": float(sum(out.values()))}


def pooled(form, blocks, redshifts, R):
    f = E.fit_training(form, blocks, list(blocks), redshifts)
    ss = list(blocks)
    per = dict(zip(ss, pmap(lambda s: E.logZ_pred(form, f, blocks[s], R, redshifts, check_conv=False)["logZ"], ss)))
    return {"per_suite": per, "S_pooled": float(sum(per.values()))}


def L_metric(form, block, redshifts, coeffs):
    """SIGMA_K-weighted squared error summed over (k[, z]) bins, averaged over boxes (spec 7.5). Reporting only."""
    sk = np.tile(np.array(C.SIGMA_K), len(redshifts))
    pred = form.predict(block.obs, redshifts, coeffs)
    if not block.mask.all():
        # Plans/11 ragged blocks: per-z mean over the boxes valid at that z; total = sum over z (== the complete-mask
        # definition when nothing is missing)
        nz = len(redshifts)
        e2 = np.where(block.mask, ((pred - block.y) / sk) ** 2, 0.0).reshape(len(block.y), nz, C.N_K).sum(axis=2)
        bm = block.mask.reshape(len(block.y), nz, C.N_K)[:, :, 0]
        per_z = [float(e2[bm[:, i], i].mean()) if bm[:, i].any() else float("nan") for i in range(nz)]
        return float(np.nansum(per_z)), per_z
    per_box = np.sum(((pred - block.y) / sk) ** 2, axis=1)
    per_z = np.mean(np.sum((((pred - block.y) / sk) ** 2).reshape(len(block.y), len(redshifts), C.N_K), axis=2), axis=0)
    return float(np.mean(per_box)), per_z.tolist()


def L_zero_shot(form, folds, blocks, redshifts):
    """Zero-shot: held-out suite predicted at the fold's population mean, no held-out data used."""
    per, perz = {}, {}
    for j, f in folds.items():
        per[j], perz[j] = L_metric(form, blocks[j], redshifts, E.to_native(form, f.coeff_order, f.s2.mu))
    return {"L_zero_shot": float(np.mean(list(per.values()))), "per_fold": per, "per_fold_per_z": perz}


def map_diagnostic_widened_prior(form, folds, blocks, redshifts, widen=None):
    """spec 7.5 -- DIAGNOSTIC ONLY, computed strictly after selection; never feeds log Z_cross, the ladder, or R.
    Per held-out box: MAP of the full coefficient vector under N(mu, widen*Sigma) x that box's likelihood,
    then L from that point prediction. 'widen' multiplies the prior covariance (registered 1.30)."""
    from scipy.optimize import minimize
    widen = C.PRIOR_WIDENING_FOR_MAP_DIAGNOSTIC if widen is None else widen
    per, perz = {}, {}
    sk = np.tile(np.array(C.SIGMA_K), len(redshifts))
    for j, f in folds.items():
        b = blocks[j]
        mu, S = f.s2.mu, widen * f.s2.Sigma
        Si = np.linalg.inv(S)
        bd = C.CFV_MODE == "blockdiag"          # Plans/11: each box's likelihood and L over its valid blocks only
        wt = S1.Whitener(b.cfv + f.cdisc) if not bd else None
        Ls, E2 = [], []
        for i in range(len(b.y)):
            o = {k: v[[i]] for k, v in b.obs.items()}
            mi = b.mask[[i]]
            if bd:
                wt = S1.BlockWhitener(b.cfv + f.cdisc, C.N_K, mi)
            yw = wt.w(b.y[[i]])

            def neg(c):
                pred = form.predict(o, redshifts, E.to_native(form, f.coeff_order, c))
                if not np.all(np.isfinite(pred[mi] if bd else pred)):
                    return 1e30
                r = yw - wt.w(pred)
                d = c - mu
                return 0.5 * float(np.sum(r * r)) + 0.5 * float(d @ Si @ d)
            c = minimize(neg, mu.copy(), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000}).x
            pred = form.predict(o, redshifts, E.to_native(form, f.coeff_order, c))
            if bd:
                E2.append(np.where(mi[0], ((pred[0] - b.y[i]) / sk) ** 2, 0.0))
            else:
                Ls.append(float(np.sum(((pred - b.y[[i]]) / sk) ** 2)))
        if bd:
            nz = len(redshifts)
            e2 = np.array(E2).reshape(len(b.y), nz, C.N_K).sum(axis=2)
            bm = b.mask.reshape(len(b.y), nz, C.N_K)[:, :, 0]
            perz[j] = [float(e2[bm[:, z], z].mean()) if bm[:, z].any() else float("nan") for z in range(nz)]
            per[j] = float(np.nansum(perz[j]))
        else:
            per[j] = float(np.mean(Ls))
    return {"L_map_diagnostic_widened_prior": float(np.mean(list(per.values()))), "per_fold": per,
            "per_fold_per_z": perz,
            "prior_widening": widen, "label": "DIAGNOSTIC (spec 7.5): per-box MAP under widened prior; NOT a prediction"}


# ---- secondary fold axes (spec 8.7; Plans/04 "registered before any redshift score") -- VALIDATION ONLY ----

def _zcols(redshifts, keep):
    nk = C.N_K
    return np.concatenate([np.arange(i * nk, (i + 1) * nk) for i, z in enumerate(redshifts) if z in keep]).astype(int)


def _sub(b, cols, name=None):
    """Column sub-vector of a block. All four suites have a complete mask at every registered z (checked), so the
    masked sub-vector path reduces exactly to this column subset; asserted, not assumed."""
    if C.CFV_MODE == "blockdiag":
        # Plans/11 ragged blocks: per-z features follow their z-columns; boxes with no valid block in `cols` drop out
        zi = np.unique(np.asarray(cols) // C.N_K)
        keep = b.mask[:, cols].any(axis=1)
        obs = {k: (v[:, zi] if np.ndim(v) == 2 and np.shape(v)[1] == b.y.shape[1] // C.N_K else v)[keep]
               for k, v in b.obs.items()}
        return E.SuiteBlock(name or b.name, obs, b.y[keep][:, cols], b.mask[keep][:, cols], b.cfv[np.ix_(cols, cols)])
    assert b.mask[:, cols].all(), "incomplete mask: use the ragged Stage-1 path"
    return E.SuiteBlock(name or b.name, b.obs, b.y[:, cols], b.mask[:, cols], b.cfv[np.ix_(cols, cols)])


def _extend_cdisc(cd_train, redshifts, held):
    """C_disc on the full stack: training-z entries as fitted; held-out-z entries = per-k mean over training z."""
    nk = C.N_K
    tr = [z for z in redshifts if z not in held]
    d = np.diag(cd_train).reshape(len(tr), nk)
    full = np.zeros((len(redshifts), nk))
    for i, z in enumerate(redshifts):
        full[i] = d[tr.index(z)] if z in tr else d.mean(axis=0)
    return np.diag(full.ravel())


def cross_redshift(form, blocks, redshifts, R):
    """6 folds; fold j holds out z_j for all suites. Stage 1 + C_disc on the other redshifts (all suites),
    Stage 2 across suites (recorded), each suite's held-out block evaluated under its OWN Stage-1 posterior
    N(c_hat_s, W_s + q) over R, complement at its conditional mean (registered design)."""
    from .stage2 import Stage2Result, quantization_var
    q = quantization_var(form, len(form.lin))
    per, detail = {}, {}

    def fold(zj):
        tr_z = tuple(z for z in redshifts if z != zj)
        ctr, cho = _zcols(redshifts, tr_z), _zcols(redshifts, (zj,))
        sub = {s: _sub(b, ctr) for s, b in blocks.items()}
        fit = E.fit_training(form, sub, list(blocks), tr_z)
        cd_full = _extend_cdisc(fit.cdisc, redshifts, (zj,))
        lz = {}
        for s, b in blocks.items():
            r = fit.s1[s]
            f_s = E.FoldFit(None, [s], {s: r}, cd_full[np.ix_(cho, cho)],
                            Stage2Result(r.c_hat, r.W + np.diag(q), 0, np.zeros(1), True, 0.0, 1), r.coeff_order)
            lz[s] = E.logZ_pred(form, f_s, _sub(b, cho), list(R), (zj,))
        return (float(sum(v["logZ"] for v in lz.values())),
                {"per_suite": {s: v["logZ"] for s, v in lz.items()},
                 "converged": {s: v.get("converged", True) for s, v in lz.items()},
                 "stage2_mu": fit.s2.mu.tolist(), "cdisc_heldout_diag_sum": float(np.trace(cd_full[np.ix_(cho, cho)]))})
    for zj, (v, d) in zip(redshifts, pmap(fold, list(redshifts))):     # Plans/12: folds independent, order kept
        per[zj], detail[zj] = v, d
    vals = list(per.values())
    return {"S_cross_redshift": float(np.sum(vals)), "sigma_fold_redshift": sigma_fold(vals),
            "per_fold": per, "detail": detail, "R": list(R),
            "label": "VALIDATION ONLY (spec 8.7): never summed with or ranked beside S_cross"}


def cross_both(form, blocks, redshifts, R):
    """24 folds (s, z_j): Stage 1 + C_disc on the other 3 suites at the other redshifts; (s, z_j) predicted
    under the population N(mu, Sigma). Reported only, never the headline."""
    per = {f"{s}@{zj}": cross_both_fold(form, blocks, redshifts, R, s, zj) for zj in redshifts for s in blocks}
    return cross_both_collect(per, R)


def cross_both_fold(form, blocks, redshifts, R, s, zj):
    """One cross-both fold (s, z_j); split out so folds can run in parallel on the same code path."""
    tr_z = tuple(z for z in redshifts if z != zj)
    ctr, cho = _zcols(redshifts, tr_z), _zcols(redshifts, (zj,))
    sub = {t: _sub(b, ctr) for t, b in blocks.items()}
    fit = E.fit_training(form, sub, [t for t in blocks if t != s], tr_z, held_out=s)
    fit.cdisc = _extend_cdisc(fit.cdisc, redshifts, (zj,))[np.ix_(cho, cho)]
    return E.logZ_pred(form, fit, _sub(blocks[s], cho), list(R), (zj,))["logZ"]


def cross_both_collect(per, R):
    vals = list(per.values())
    return {"S_cross_both": float(np.sum(vals)), "sigma_fold_both": sigma_fold(vals), "per_fold": per, "R": list(R),
            "label": "VALIDATION ONLY (spec 8.7): never summed with or ranked beside S_cross"}

"""Gate 0: three physical checks before any fitting, at EVERY registered redshift (spec 5, 8.8).

G0_1  S -> 0 as k -> 0
G0_2  leading low-k term ~ k^2 (no k^0, no k^1); a leading k^n with n > 2 passes and is noted
G0_3  S bounded as k -> infinity
G0_U  (v2) small-scale upturn from stellar feedback at the FITTED coefficients on the data k-grid (upturn(); needs the
      cross-suite folds, so it is merged after fitting; scores never depend on Gate 0). v1's ratio window G0_4 is dropped.
early-time flag (recorded, NOT gating): |S(k, z=z_max)| << |S(k, z=0)|
Symbolic (SymPy) with coefficients and features positive, plus a numeric probe over the registered ranges.
"""
import signal

import numpy as np
import sympy as sp

from ..common import config as C


class _Timeout(Exception):
    pass


def _with_timeout(fn, sec=20):
    def h(*_):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, h)
    signal.alarm(sec)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _symbolic(form, zval):
    k = form._sym["k"]; zs = form._sym["z"]
    e = form._sym["expr"].subs(zs, sp.nsimplify(zval))
    out = {}
    try:
        lim0 = _with_timeout(lambda: sp.limit(e, k, 0, "+"))
        out["G0_1"] = bool(sp.simplify(lim0) == 0)
    except Exception as ex:          # noqa: BLE001 -- recorded, numeric fallback decides
        out["G0_1"] = None; out["G0_1_err"] = type(ex).__name__
    try:
        ser = _with_timeout(lambda: sp.series(e, k, 0, 3).removeO())
        c0 = sp.simplify(ser.coeff(k, 0)); c1 = sp.simplify(ser.coeff(k, 1)); c2 = sp.simplify(ser.coeff(k, 2))
        out["G0_2"] = bool(c0 == 0 and c1 == 0)
        out["G0_2_leading"] = "k^0" if c0 != 0 else ("k^1" if c1 != 0 else ("k^2" if c2 != 0 else "k^n,n>2"))
        # fractional powers: series may contain k**(1/2) etc. -> any power < 2 fails
        powers = [t.as_coeff_exponent(k)[1] for t in sp.Add.make_args(sp.expand(ser)) if t.has(k)]
        if any(p.is_number and p < 2 for p in powers):
            out["G0_2"] = False
    except Exception as ex:          # noqa: BLE001
        out["G0_2"] = None; out["G0_2_err"] = type(ex).__name__
    try:
        liminf = _with_timeout(lambda: sp.limit(e, k, sp.oo))
        out["G0_3"] = bool(liminf.is_finite) if liminf.is_finite is not None else None
    except Exception as ex:          # noqa: BLE001
        out["G0_3"] = None; out["G0_3_err"] = type(ex).__name__
    return out


def _probe_points(form, rng, n=24):
    pts = []
    for _ in range(n):
        c = {}
        for name in form.lin:
            c[name] = abs(rng.normal(C.MU_A, 1.0)) if form.roles[name] == "amplitude" else rng.normal(0, 1.0)
        for name, s in form.nl.items():
            nodes = s.nodes(); c[name] = float(rng.choice(nodes))
        f = {"D": rng.uniform(0.005, 0.12), "f_b": rng.uniform(0.03, 0.15), "Omega_m": rng.uniform(0.1, 0.5),
             "sigma_8": rng.uniform(0.6, 1.0), "Omega_b_over_Omega_m": rng.uniform(0.1, 0.5)}
        for i in range(6):
            f[f"f_b_M_{i}"] = rng.uniform(0.03, 0.16)
        pts.append((c, f))
    return pts


def _mp_eval(form):
    """50-digit evaluation of the full expression (v1 Plans/03 A4): a double-precision probe at k ~ 1e-6 suffers
    catastrophic cancellation in forms like 1 - 1/sqrt(1 + x^2) and reports a spurious slope."""
    import mpmath
    if not hasattr(form, "_f_mp"):
        k = form._sym["k"]; z = form._sym["z"]
        args = [k, z] + [form._sym["feats"][f] for f in form.features] + \
               [form._sym["coeffs"][c] for c in form.nl] + [form._sym["coeffs"][c] for c in form.lin]
        form._f_mp = sp.lambdify(args, form._sym["expr"], "mpmath")
    mpmath.mp.dps = 50
    return form._f_mp


def _numeric(form, zval, rng):
    """Numeric probe (50-digit): small-k scaling slope, k->0 value, large-k boundedness."""
    import mpmath
    # Plans/08 A14: registered grids allow transition scales down to ~1e-8 h/Mpc, so the low-k slope is probed at 1e-12
    ks = [mpmath.mpf("1e-12"), mpmath.mpf("2e-12"), mpmath.mpf("1e-11"), mpmath.mpf("2e-11")]
    # Plans/06 A9: registered grids allow transition scales up to ~1e8 h/Mpc (e.g. k1 D^beta (1+z)^b_z), so the large-k
    # boundedness probe extends to 1e12; the test compares S(1e12) with S(1e8).
    kb = [mpmath.mpf(x) for x in ("1e2", "1e4", "1e6", "1e8", "1e10", "1e12")]
    fm = _mp_eval(form)
    ok1 = ok2 = ok3 = True
    for c, f in _probe_points(form, rng):
        args = [mpmath.mpf(f[x]) for x in form.features]
        lin = [mpmath.mpf(c[x]) for x in form.lin]
        nl = [mpmath.mpf(c[x]) for x in form.nl]

        def ev(kk):
            out = []
            for v in kk:
                try:
                    r = fm(v, mpmath.mpf(zval), *args, *nl, *lin)
                    out.append(float(mpmath.re(r)) if abs(mpmath.im(r)) == 0 else np.nan)
                except (ZeroDivisionError, ValueError, OverflowError):
                    out.append(np.nan)
            return np.array(out)
        s = ev(ks); sb = ev(kb)
        if not np.all(np.isfinite(s)):
            ok1 = False
            continue
        nz = np.abs(s) > 0
        if nz[0] and nz[1]:
            slope = np.log(abs(s[1]) / abs(s[0])) / np.log(2.0)
            if slope < 0.5:          # S does not vanish as k -> 0 (constant term)
                ok1 = False
            if slope < 1.9:          # leading power below k^2
                ok2 = False
        if not np.all(np.isfinite(sb)) or abs(sb[-1]) > 1e3 or abs(sb[-1]) > 10 * max(abs(sb[-3]), 1e-3):
            ok3 = False
    return {"G0_1": ok1, "G0_2": ok2, "G0_3": ok3}


def check(form, redshifts=C.REDSHIFTS, seed=C.SEED):
    rng = np.random.default_rng(seed)
    by_z = {}
    for z in redshifts:
        sym = _symbolic(form, z)
        num = _numeric(form, z, rng)
        r = {}
        for g in ("G0_1", "G0_2", "G0_3"):
            r[g] = sym[g] if sym.get(g) is not None else num[g]
            r[f"{g}_numeric"] = num[g]
            r[f"{g}_symbolic"] = sym.get(g)
            if sym.get(g) is not None and sym[g] != num[g]:
                r[f"{g}_disagree"] = True
                r[g] = bool(sym[g] and num[g])   # disagreement resolved conservatively; recorded
        r["G0_2_leading"] = sym.get("G0_2_leading")
        by_z[f"{z:.2f}"] = r
    flags = {g: all(by_z[z][g] for z in by_z) for g in ("G0_1", "G0_2", "G0_3")}
    # early-time flag: response at highest z vs z=0, at probe points
    early = None
    if len(redshifts) > 1:
        ratios = []
        for c, f in _probe_points(form, np.random.default_rng(seed + 1)):
            args = [f[x] for x in form.features]
            ev = lambda zz: np.asarray(form._f_full(np.array(C.K_GRID), zz, *args, *[c[x] for x in form.nl],
                                                    *[c[x] for x in form.lin]), float)
            with np.errstate(all="ignore"):
                a0, a1 = np.max(np.abs(ev(min(redshifts)))), np.max(np.abs(ev(max(redshifts))))
            ratios.append(a1 / a0 if a0 > 0 else np.nan)
        early = {"median_ratio_zmax_over_z0": float(np.nanmedian(ratios)),
                 "flag_small": bool(np.nanmedian(ratios) < 0.1)}
    return {**flags, "gate0_passed": all(flags.values()), "G0_by_redshift": by_z, "G0_early_time_flag": early}


def _upturn_params():
    p = (C.G0_U_KSTAR, C.G0_U_ZSPLIT, C.G0_U_DHI, C.G0_U_FPASS)
    assert None not in p, "G0_U parameters are not registered yet (Scripts/analysis/upturn_calib.py, owner approval)"
    return p


def upturn_box(S, K, z):
    """Per-box G0_U verdict on S (n_box, n_k) over the data k-grid K. On k >= k*: at z >= z_split the rise
    S(k_max) - min S must be >= dhi (an upturn); at z < z_split the end step S(k_max) - S(k_prev) must be >= 0 (no further
    dive; flat or a mild upturn passes). Returns (pass (n_box,), rise (n_box,), end_step (n_box,))."""
    kstar, zsplit, dhi, _ = _upturn_params()
    sel = np.asarray(K) >= kstar
    assert sel.sum() >= 2, "G0_U needs at least two k-bins at k >= k*"
    Ss = S[:, sel]
    rise = Ss[:, -1] - np.min(Ss, axis=1)
    step = Ss[:, -1] - Ss[:, -2]
    ok = rise >= dhi if z >= zsplit else step >= 0.0
    return ok & np.isfinite(rise), rise, step


def upturn(form, folds, blocks, redshifts):
    """G0_U (v2, owner 2026-11-08): stellar feedback enhances S at small scales, so S turns back up at high k.
    Evaluated like v1's ratio check: at every fold's prior mean mu and every training suite's Stage-1 c_hat (Stage-2
    coordinates -> native), over the feature values of all scored boxes of all suites at each z (valid blocks only), on the
    data k-grid. Passes when, for every (z, coefficient set), at least G0_U_FPASS of the boxes pass upturn_box.
    folds: the stored form {suite: {"mu", "order", "stage1": {s: {"c_hat"}}}}."""
    from . import evidence as E
    kstar, zsplit, dhi, fpass = _upturn_params()
    K = np.asarray(C.K_GRID, float)
    sets = []
    for j, f in folds.items():
        sets.append((f"mu[{j}]", E.to_native(form, f["order"], np.asarray(f["mu"], float))))
        sets += [(f"c_hat[{j}/{s}]", E.to_native(form, f["order"], np.asarray(r["c_hat"], float))) for s, r in f["stage1"].items()]
    by_z, worst = {}, []
    for iz, z in enumerate(redshifts):
        feats = []
        for b in blocks.values():
            v = b.mask[:, iz * C.N_K]
            feats.append([(np.asarray(b.obs[x], float)[:, iz] if np.ndim(b.obs[x]) == 2 else np.asarray(b.obs[x], float))[v]
                          for x in form.features])
        F = [np.concatenate([fs[i] for fs in feats])[:, None] for i in range(len(form.features))]
        fr = []
        for name, c in sets:
            with np.errstate(all="ignore"):
                S = np.asarray(form._f_full(K[None, :], z, *F, *[c[x] for x in form.nl], *[c[x] for x in form.lin]), float)
            S = np.broadcast_to(S, (len(F[0]) if F else 1, len(K)))
            ok, rise, step = upturn_box(S, K, z)
            f_ok = float(ok.mean())
            fr.append(f_ok)
            if f_ok < fpass:
                worst.append({"z": float(z), "set": name, "frac_pass": f_ok, "median_rise": float(np.nanmedian(rise)),
                              "median_end_step": float(np.nanmedian(step))})
        by_z[f"{z:.2f}"] = {"min_frac_pass": min(fr), "rule": "rise >= dhi" if z >= zsplit else "end step >= 0"}
    worst = sorted(worst, key=lambda w: w["frac_pass"])[:10]
    return {"G0_U": not worst and all(v["min_frac_pass"] >= fpass for v in by_z.values()), "by_z": by_z, "worst": worst,
            "n_coeff_sets": len(sets), "params": {"k_star": kstar, "z_split": zsplit, "delta_hi": dhi, "f_pass": fpass},
            "rule": "v2 G0_U: upturn at high k from stellar feedback; fold mu and Stage-1 c_hat, all boxes, data k-grid"}


def apply_upturn(g0, up):
    """Merge G0_U into a Gate-0 record (gate0_passed = G0_1 & G0_2 & G0_3 & G0_U)."""
    g0 = dict(g0); g0["G0_U"] = up["G0_U"]; g0["upturn"] = up
    g0["gate0_passed"] = bool(g0["G0_1"] and g0["G0_2"] and g0["G0_3"] and up["G0_U"])
    return g0

"""Registered constants for SANDBOX-ML4PS (copied from SANDBOX-v2 2026-11-10; ML4PS changes marked ML4PS). Single import point.

ML4PS: z = 0 only, the published paper's 16 k-bins (0.356-19.98 h/Mpc) and its sigma_k reporting table (Plans/00_ml4ps.md).

Ported from v1 (data and statistics choices unchanged unless marked v2). Source of record: Plans/00_bootstrap.md (2026-11-08).
v2 is fixed to the zfb6 analysis with the k <= 12 h/Mpc cut. Never edit a value here after scoring has started without a
dated amendment in Plans/ written before the rescore.
"""
from pathlib import Path

# --- paths -------------------------------------------------------------------------------------
SANDBOX = Path(__file__).resolve().parents[2]
import os as _os0, sys as _sys0
DATA_ROOT = Path(_os0.environ.get("CAMELS_ROOT", SANDBOX / "camels_raw")).resolve()   # raw CAMELS files (only needed to rebuild Artifacts/data)
ARTIFACTS = SANDBOX / "Artifacts"
DATA_OUT = ARTIFACTS / "data"
LOGS = ARTIFACTS / "logs"
PYTHON = _sys0.executable

# --- suites and sets (spec 4.1, 10) ------------------------------------------------------------
SUITES = ["IllustrisTNG", "SIMBA", "Astrid", "Swift-EAGLE"]
SCORED_SETS = ["LH", "CV"]          # 1P is diagnostic only and never scored; CROCODILE not part of project
SEED = 42

# hydro / gravity-only directories relative to DATA_ROOT (spec 8.2 -- layout is NOT uniform)
SUITE_DIRS = {
    "IllustrisTNG": ("IllustrisTNG", "IllustrisTNG_DM"),
    "Astrid": ("Astrid", "Astrid_DM"),
    "SIMBA": ("SIMBA/L25n256", "SIMBA_DM/L25n256"),
    "Swift-EAGLE": ("Swift-EAGLE/L25n256", "Swift-EAGLE_DM/L25n256"),
}
PROFILE_DIR = "Profiles"            # Profiles/<Suite>/<Set>/<box>/profiles_z000_snap090.hdf5
PROFILE_FILE = "profiles_z000_snap090.hdf5"

# --- k grid (ML4PS main loop, owner 2026-11-11; Plans/01_ml4ps_k10.md): 16 native bins with k <= 10 h/Mpc -- the published
# paper's 13 bins <= 10 plus 3 fills at the log-midpoints of the widest gaps (1.619, 3.388, 8.668). Exact raw values (all present
# in the all-k z = 0 file by exact float match; never interpolated). The k <= 20 run (paper's 16 bins) is in Archive_k20/.
KMAX = 10.0
N_K = 16
K_GRID = [0.3559856489878782, 0.6031648866026231, 0.8579160745827348, 1.1134643105957107, 1.3740399143213644,
          1.6191547045702983, 1.8706866734302718, 2.378144976376236, 2.8786531626794947, 3.3881595783338163,
          3.8883159107320853, 4.894678666149941, 6.151460297009074, 7.65922714936463, 8.667638839289207,
          9.923426157758477]
assert len(K_GRID) == N_K == 16 and max(K_GRID) <= KMAX
K_NYQUIST = 32.169908772759484
K_GRID_NORMALIZE_TOL = 1e-8         # Swift-EAGLE native grid offset is ~4.8e-10 relative (spec 8.2)
N_K_RAW = 443                       # rows in every Pk_m_z=<Z>.txt

# REPORTING ONLY, never the likelihood: the paper's sigma_k^survey weights of the benchmark loss L at the paper's bins; at the
# 3 fill bins (1.619, 3.388, 8.668) log-log interpolated between the neighbouring paper values.
SIGMA_K_PAPER = [0.004924, 0.010072, 0.016245, 0.021262, 0.023966, 0.026314, 0.028568, 0.032752, 0.036514, 0.040064, 0.043331,
                 0.049398, 0.061515, 0.076592, 0.086676, 0.099234]
import os as _os
import json as _json
# Owner 2026-11-11: over-estimate sigma_k so it bounds sqrt(diag C_fv) and sqrt(diag C_disc) (Scripts/analysis/sigma_k_inflated.py
# -> Artifacts/sigma_k_inflated.json). PK_SIGMA_K=paper restores the paper's values. Trials scored before 2026-11-11 18:30 IST
# (the 50-trial main loop, Base/) store L with the paper's sigma_k.
_SKI = Path(__file__).resolve().parents[2] / "Artifacts" / "sigma_k_inflated.json"
SIGMA_K_SOURCE = "paper" if _os.environ.get("PK_SIGMA_K") == "paper" or not _SKI.exists() else "inflated"
SIGMA_K = SIGMA_K_PAPER if SIGMA_K_SOURCE == "paper" else _json.loads(_SKI.read_text())["sigma_k_inflated"]
KCUT = KMAX
# Opt-in low-k cut (Chi2Push/, owner 2026-11-11 "do both"): PK_KMIN=0.7 keeps the 14 bins with k > 0.7 (drops 0.356, 0.603).
# Unset -> the registered 16-bin grid, nothing changes.
KMIN = float(_os.environ.get("PK_KMIN", "0"))
if KMIN > 0:
    _keep = [i for i, kk in enumerate(K_GRID) if kk > KMIN]
    K_GRID = [K_GRID[i] for i in _keep]
    SIGMA_K_PAPER = [SIGMA_K_PAPER[i] for i in _keep]
    SIGMA_K = [SIGMA_K[i] for i in _keep]
    N_K = len(K_GRID)

# --- redshifts (spec 8.3, 10) ------------------------------------------------------------------
Z_LABEL_MATCH_TOL = 0.02
Z_EXCLUDE = (127.00,)
ANALYSIS = "z0_k10"                 # ML4PS main loop: z = 0 only, k <= 9.92
REDSHIFTS = (0.00,)                 # ML4PS
ZFB_GRID = (0.00, 0.10, 0.21, 0.33, 0.47, 0.61, 0.77, 0.95, 1.15, 1.36, 1.60, 2.00)   # z labels in observables_zfb.h5
N_Z = len(REDSHIFTS)
ANCHOR_DESIGN = "fb_measured_z"     # D(z) = Omega_b/Omega_m - f_b(z), measured per (box, snapshot); ragged blocks
CFV_MODE = "blockdiag"              # per-z empirical N_K x N_K CV covariance, diag shrink 0.10, NO cross-z terms
NS = ARTIFACTS / ANALYSIS           # namespace for registry / scores / logs
LOGS = NS / "logs"
if _os.environ.get("PK_NS_DIR"):    # opt-in separate namespace (e.g. Loop/); unset -> nothing changes
    NS = Path(_os.environ["PK_NS_DIR"])
    LOGS = NS / "logs"
FB_Z_DIR = "f_b_z"                  # DATA_ROOT/f_b_z/<Suite>_LH_baryon_power_z0_2.hdf5
OBS_ZFB = DATA_OUT / "observables_zfb.h5"

# --- observables (spec 4.4, bootstrap 6) -------------------------------------------------------
FB_CONVENTION = "mean_ratio"        # mean over in-mask halos of per-halo f_baryon_200c
FB_MASS_MIN = 5.0e12                # M200c [Msun/h], exclusive
FB_MASS_MAX = 1.0e14                # M200c [Msun/h], exclusive
FB_MIN_HALOS = 1
OMEGA_B_FALLBACK = 0.049            # registered CAMELS value; used only if OmegaBaryon attr absent
FORBIDDEN_FEATURES = ("A_SN1", "A_SN2", "A_AGN1", "A_AGN2", "seed")
PERMITTED_FEATURES = ("f_b", "D", "f_b_M", "Omega_b_over_Omega_m", "Omega_m", "sigma_8", "z")

# --- fitting / evidence (spec 10, bootstrap 6) -------------------------------------------------
MU_A = 3.278
SIGMA_A = 3.5
CV_SHRINKAGE = 0.10
N_CV_BOXES = 27
HARTLAP_FACTOR = (N_CV_BOXES - 1) / (N_CV_BOXES - N_K - 2)   # 26/9; REPORTED, NOT APPLIED
HARTLAP_APPLIED = False
LADDER_MAX_COEFF = 5
SIGMA_SUITE_REG = "inverse_wishart_map(nu=p+2, Psi=diag(training-suite var of c_hat, ddof=1) + mean training W_s)"
NL_GRID_RULE = "log-uniform for positive scales, linear for exponents, 41 nodes/dim"
NL_GRID_NODES = 41
# Disclosure carried into every artifact that quotes a prior width (Plans/02 D1 amendment, user decision 2026-10-09):
SIGMA_SUITE_3SUITE_RECOVERY = {"median_fitted_over_true": [0.20, 0.24], "mean_fitted_over_true": [0.28, 0.31],
                               "best_achievable_mean": [0.91, 1.02], "synthetic": True, "n_reps": 200}
QUAD_NODES_START = 17
PRIOR_WIDENING_FOR_MAP_DIAGNOSTIC = 1.30   # diagnostic only, never selection
# v2 Gate-0 item G0_U (owner 2026-11-08): small-scale upturn from stellar feedback. On k >= G0_U_KSTAR, at z >= G0_U_ZSPLIT
# S(k_max) - min S >= G0_U_DHI; at z < G0_U_ZSPLIT the last step S(k_max) - S(k_prev) >= 0. Passes when >= G0_U_FPASS of
# boxes per (z, coefficient set) do.
# Set from data (Scripts/analysis/upturn_calib.py on the k <= 15 grid) and owner-approved 2026-11-08 (Plans/00 sec. 4).
G0_U_KSTAR = 2.38
G0_U_ZSPLIT = 1.15
G0_U_DHI = 0.005
G0_U_FPASS = 0.90
# v2 presentable bar (cheap loop): over the 4 LOSO held-out suites, mean chi2_nu (C_fv only) <= 2 and std <= 1
CHI2_MEAN_MAX = 2.0
CHI2_STD_MAX = 1.0
LOOP_NL_NODES = 41                  # ML4PS main loop scores on the full grid (owner 2026-11-11; 11 misranked forms)
MAX_COEFF = 6
ALLOW_CONDITIONING = False
CFV_ASSEMBLY = (f"{ANALYSIS}: block-diagonal in z; block z = empirical {N_K}x{N_K} CV covariance at z (27 CV boxes), diag "
                "shrink 0.10; NO cross-z terms (CV z-z correlation ignored, disclosed); per suite")

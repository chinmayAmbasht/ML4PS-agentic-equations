"""_provenance stamping and read-path logging (spec 11, 12.2 'data provenance')."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import config as C

_READS: set[str] = set()


def log_read(path) -> Path:
    """Record an input path; every reader in the sandbox goes through this."""
    p = Path(path).resolve()
    _READS.add(str(p))
    return p


def reads() -> list[str]:
    return sorted(_READS)


def out_of_scope_reads() -> list[str]:
    roots = (str(C.SANDBOX.resolve()), str(C.DATA_ROOT.resolve()))
    return [p for p in _READS if not any(p == r or p.startswith(r + "/") for r in roots)]


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=C.SANDBOX, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    commit = run("rev-parse", "HEAD") or None
    if commit is None and (C.SANDBOX / "PACKAGE_COMMIT").exists():
        # Plans/12 cluster package: no .git shipped; the commit the package was built from is recorded instead
        return {"commit": (C.SANDBOX / "PACKAGE_COMMIT").read_text().strip(), "dirty": False, "source": "PACKAGE_COMMIT"}
    return {"commit": commit, "dirty": bool(run("status", "--porcelain", "--", "Scripts"))}


def execution_site() -> dict:
    """v1 Plans/12: where a number was computed (local vs TACC node), and with how many workers."""
    import os, platform, socket
    import numpy, scipy
    return {"host": socket.gethostname(), "tacc_system": os.environ.get("TACC_SYSTEM"),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "pk_npar": os.environ.get("PK_NPAR", "1"),
            "platform": platform.platform(), "python": platform.python_version(),
            "numpy": numpy.__version__, "scipy": scipy.__version__}


def provenance(protocol: str, redshifts=None, **extra) -> dict:
    z = list(C.REDSHIFTS if redshifts is None else redshifts)
    d = {
        "n_k": C.N_K,
        "n_z": len(z),
        "redshifts": z,
        "anchor_design": C.ANCHOR_DESIGN,
        "suites": list(C.SUITES),
        "protocol": protocol,
        "code_state": git_state(),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "hartlap_applied": C.HARTLAP_APPLIED,
        "fb_mask": [C.FB_MASS_MIN, C.FB_MASS_MAX],
        "fb_convention": C.FB_CONVENTION,
        "sigma_suite_prior": C.SIGMA_SUITE_REG,
        "sigma_suite_3suite_recovery_synthetic": C.SIGMA_SUITE_3SUITE_RECOVERY,
        "nl_grid_nodes": C.NL_GRID_NODES,
        "execution_site": execution_site(),
    }
    d.update(extra)
    return d


def dumps(d: dict) -> str:
    return json.dumps(d, indent=1, sort_keys=True, default=str)

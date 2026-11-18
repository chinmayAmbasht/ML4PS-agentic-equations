"""Score one candidate equation with the same harness that scored all 64 forms in the paper.

The harness applies the physical checks, then Stage 1, Stage 2, and the held-out log Z_cross with the promotion ladder,
followed by chi^2_nu and the benchmark loss. Results go to a scratch directory under cache/; the recorded agent loop in
Loop/ is never modified.

A submission is a JSON file, for example Loop/submissions/A17_T36_split_knee_amps.json (paper rank 1):
  {"expr": "-A*D*k**2/(1 + (k/k0)**2)", "lin": ["A"], "nl": {"k0": {"lo": 0.01, "hi": 10, "spacing": "log"}},
   "roles": {"A": "amplitude", "k0": "scale"}, "features": ["D"], "hypothesis": "..."}

Usage:  python score_form.py <submission.json> [--base]
  --base  allow 3-4 nonlinear coefficients (used for the four seed forms in Base/; the agent loop allowed 1-2).
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    sub = Path(argv[0]).resolve()
    cache = Path(os.environ.get("ML4PS_CACHE", ROOT / "cache")).resolve()
    out = cache / f"score_{sub.stem}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ML4PS_CACHE=str(cache), PK_ML4PS_LOOP_DIR=str(out), PK_ML4PS_BUDGET="1000000")
    if "--base" in argv[1:]:
        env["PK_ML4PS_BASE"] = "1"
    rc = subprocess.call([sys.executable, "-m", "Loop.evaluate", str(sub)], cwd=ROOT, env=env)
    print(f"\nResults: {out / 'trial_state.yaml'} (scores) and {out / 'residuals'} (per-suite residuals)")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Figure labels for the ML4PS main-loop forms: the equation (usetex) and the rank by log Z_cross among all scored forms
(50 loop trials + 4 paper base forms), in place of trial ids (owner 2026-11-11).

Shorthand (printed by NOTATION in every figure that uses it):
  W(u) = u^2/(1+u^2),  q = f_b/(Omega_b/Omega_m) (retained baryon fraction),  u0 = k sqrt(D)/k0,  u1 = k (D/0.05)^0.75/k1.

label(key) -> "S = <eq> [rank n]";  rank_tag(key) -> "rank n" (+ ", paper rank m" for a base form);  key: "T47", "rank1", ...
"""
import re
from pathlib import Path

import yaml

M = Path(__file__).resolve().parents[1]
NOTATION = (r"$W(u)=u^2/(1+u^2)$,\ \ $q=f_b/(\Omega_b/\Omega_m)$,\ \ $u_0=k\sqrt{D}/k_0$,\ \ "
            r"$u_1=k\,(D/0.05)^{0.75}/k_1$;\ \ rank: by $\log Z_\mathrm{cross}$ among all 54 scored forms (50 loop trials, 4 paper base forms)")
_K = r"(E f_b-CD)\,W(u_1)"
_KNEE = r"-AD\,[q\,W(k\sqrt{D}/%s)+(1-q)\,W(u_0)]"
EQ = {
    "T47": r"-D\,[A\,q\,W(k\sqrt{D}/0.68)+B\,(1-q)\,W(u_0)]+" + _K,
    "T36": _KNEE % "0.68" + "+" + _K,
    "T43": _KNEE % "0.8" + "+" + _K,
    "T46": _KNEE % "0.68" + "+" + _K + r"-GD\,\frac{(k/0.4)^2}{[1+(k/0.4)^2]^2}",
    "T40": _KNEE % "1.0" + "+" + _K,
    "T44": _KNEE % "0.68" + r"+(E f_b-CD)\,W(u_1)\frac{1+2u_1}{1+u_1}",
    "T45": _KNEE % "0.68" + r"+\left(E f_b-\frac{CD}{\Omega_b/\Omega_m}\right)W(u_1)",
    "T49": r"-D\,[A\,q\,W(u_1)+B\,(1-q)\,W(u_0)]+" + _K,
    "T42": r"-AD\,[\sqrt{q}\,W(k\sqrt{D}/0.68)+(1-\sqrt{q})\,W(u_0)]+" + _K,
    "T39": _KNEE % "0.68" + r"+(E f_b-CD)\,\frac{u_1^3}{(1+u_1^2)^{3/2}}",
    "T41": _KNEE % "0.68" + r"+(E f_b-CD)\,\frac{u_1^2}{(1+u_1)^2}",
    "T20": r"-AD\,W(u_0)+(E f_b-CD)\,W(k\,(D/0.05)^{0.5}/k_1)",
    "T07": r"-D\,W(u_0)\left[A+B\,\frac{u_0}{1+u_0}\right]",
    "rank1": r"-AD\left[\frac{\Omega_b}{\Omega_m}W\!\left(\frac{k}{k_1D^\beta}\right)+\left(1-\frac{\Omega_b}{\Omega_m}\right)W\!\left(\frac{k}{k_2D^\beta}\right)\right]",
    "rank2": r"-AD\,W(k\,(1+\alpha D)/k_{00})",
    "rank3": r"-AD\,W\!\left(\frac{k}{k_{00}(\Omega_m/0.3)^{1/3}}\right)",
    "rank12": r"-D\,[A_1W(k/k_1)+A_2W(k/(k_1e^{d_l}))]",
}
BASE_IDS = {"T05": "rank1", "T02": "rank2", "T01": "rank3", "T03": "rank12"}   # Base/ trial id -> paper-rank key


def _ranks():
    z = [(t["logZ_cross"], t["trial"]) for t in yaml.safe_load((M / "Loop" / "trial_state.yaml").read_text())["trials"]]
    z += [(t["logZ_cross"], BASE_IDS[t["trial"]]) for t in yaml.safe_load((M / "Base" / "trial_state.yaml").read_text())["trials"]
          if t.get("logZ_cross") is not None]
    return {k: i + 1 for i, (_, k) in enumerate(sorted(z, reverse=True))}


RANK = _ranks()


def key_of(label):
    """'T47' / 'rank1' / 'paper rank 1' -> canonical key."""
    m = re.fullmatch(r"(?:paper )?rank\s*(\d+)", label)
    return f"rank{m.group(1)}" if m else label


def rank_tag(label):
    k = key_of(label)
    return f"rank {RANK[k]}" + (f", paper rank {k[4:]}" if k.startswith("rank") else "")


def label(label, rank=True):
    k = key_of(label)
    return rf"$\mathcal{{S}}={EQ[k]}$" + (f" [{rank_tag(k)}]" if rank else "")

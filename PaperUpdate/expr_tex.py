"""Compact LaTeX for engine expression strings: W(x) = x^2/(1+x^2) is folded, q = f_b/(Omega_b/Omega_m), names to TeX."""
import re


def _match(s, i):
    """index of the ')' matching s[i] == '('."""
    d = 0
    for j in range(i, len(s)):
        d += s[j] == "("; d -= s[j] == ")"
        if d == 0:
            return j
    raise ValueError(s)


def fold_W(s):
    changed = True
    while changed:
        changed = False
        for i in [m.start() for m in re.finditer(r"\(", s)]:
            j = _match(s, i)
            x = s[i + 1:j]
            pat = f"({x})**2/(1+({x})**2)"
            if s.startswith(pat, i) and not s.startswith("**", i + len(pat)):
                rep = f"W[{x}]"
                if i > 0 and s[i - 1] == "(" and s[i + len(pat):i + len(pat) + 1] == ")":
                    s = s[:i - 1] + rep + s[i + len(pat) + 1:]
                else:
                    s = s[:i] + rep + s[i + len(pat):]
                changed = True
                break
    return s


NAMES = [("Omega_b_over_Omega_m", r"\omega_b"), ("Omega_m", r"\Omega_m"), ("sigma_8", r"\sigma_8"), ("f_b", "f_b"),
         ("k00", "k_{00}"), ("exp", r"\exp"), ("alpha", r"\alpha"), ("beta", r"\beta"), ("dl", r"\delta"), ("sqrt", r"\sqrt")]


def tex(expr):
    s = expr.replace(" ", "")
    s = s.replace("(f_b/Omega_b_over_Omega_m)", "q")
    s = fold_W(s)
    s = re.sub(r"\*\*\(([^()]*)\)", r"^{\1}", s)
    s = re.sub(r"\*\*([0-9.]+|[A-Za-z_]\w*)", r"^{\1}", s)
    s = re.sub(r"\bk([0-9a-z])\b", r"k_{\1}", s)
    for a, b in NAMES:
        s = re.sub(rf"\b{a}\b", lambda m: b, s)
    s = s.replace("*", r"\,").replace("[", r"(").replace("]", r")").replace("W(", r"W(")
    s = s.replace("^{0.5}", "^{1/2}")
    return s

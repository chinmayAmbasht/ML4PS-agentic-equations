"""Owner 2026-11-14: Fig. 3's adapted bars use the per-box MAP with prior widening 1.00 and likelihood C_fv,j/h only (the 'fvh'
setting of PaperUpdate/map_sensitivity.py, i.e. the same metric as L). Writes those L into bench_corr.json 'transfer_map' (and
PySR/summary.json bench) and keeps the previous values (widening 1.30, C_fv + diag C_disc) under 'transfer_map_w130_disc'.
Idempotent. Usage: python -m PaperUpdate.apply_map_setting
"""
import json
from pathlib import Path

M = Path(__file__).resolve().parents[1]
NAME = {"PYSR_U": "PySR (unconstrained)", "VD": "vD20 refit"}
SN = json.loads((M / "PaperUpdate" / "map_sensitivity.json").read_text())
bf = M / "PaperUpdate" / "Benchmark" / "bench_corr.json"
B = json.loads(bf.read_text())
for k, r in SN.items():
    b = B[NAME.get(k, k)]
    if b.get("map_setting") != "fvh":
        b["transfer_map_w130_disc"] = b["transfer_map"]
    b["transfer_map"] = {"per_suite": r["fvh"]["per_suite"], "mean": r["fvh"]["mean"]}
    b["map_setting"] = "fvh"
bf.write_text(json.dumps(B, indent=1))
sf = M / "PySR" / "summary.json"
S = json.loads(sf.read_text())
S["U"]["bench"].setdefault("transfer_map_w130_disc", S["U"]["bench"]["transfer_map"])
S["U"]["bench"]["transfer_map"] = SN["PYSR_U"]["fvh"]["mean"]
sf.write_text(json.dumps(S, indent=1, default=str))
print({NAME.get(k, k): round(B[NAME.get(k, k)]["transfer_map"]["mean"], 1) for k in SN})

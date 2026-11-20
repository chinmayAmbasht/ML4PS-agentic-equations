"""Rescore Fig. 3 benchmark models on k <= 1 bins only (van Daalen calibrated range). Fits unchanged (16 bins);
evaluation restricted: L = dS_m^T (C_fv,j[m,m]/h_m)^-1 dS_m, h_m = (27-n_m-2)/26. Writes vd_k1.json next to this script.
MAP/zero-shot read PaperUpdate/preds/{T47,VD}.npz, which hold the OLD MAP setting (prior x1.30, C_fv + C_disc); zero-shot is unaffected.
Usage: python -m PaperUpdate.Benchmark.vd_k1"""
import json, sys
from pathlib import Path
import numpy as np
M = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(M))
from PaperUpdate import Lcorr as LC
from PaperUpdate.Benchmark import bench as B
m = B.K <= 1.0; nm = int(m.sum()); hm = (27 - nm - 2) / 26
def Lk(pred, y, suite):
    ok = np.isfinite(y).all(1) & np.isfinite(pred).all(1)
    d = (pred - y)[ok][:, m]; Ci = np.linalg.inv(LC.cfv(suite)[np.ix_(m, m)] / hm)
    return float(np.mean(np.einsum("bi,ij,bj->b", d, Ci, d)))
B.L = Lk
D = B.data()
R = json.loads((M / "PaperUpdate" / "ranking.json").read_text()); subs = {r["key"]: r["submission"] for r in R}
models = {"RF": B.rf_fit, "vD20 published": B.vd_pub_fit, "vD20 refit": B.vd_refit,
          "T47": B.form_fit_factory(json.loads((M / subs["T47"]).read_text()))}
out = {"_meta": {"k_bins": B.K[m].tolist(), "h": hm}}
for n, f in models.items():
    r = {"within": B.summarize(B.run_within(f, D)), "pooled": B.summarize(B.run_pooled(f, D)),
         "transfer_population": B.summarize(B.run_transfer(f, D))}
    out[n] = r; print(n, {k: round(v["mean"], 2) for k, v in r.items()}, flush=True)
for key, name in (("T47", "T47"), ("VD", "vD20 refit")):
    z = np.load(M / "PaperUpdate" / "preds" / f"{key}.npz")
    for tag, f in (("zero_shot", "zs"), ("map", "map")):
        d = {s: Lk(z[f"{s}__{f}"], z[f"{s}__y"], s) for s in B.C.SUITES}
        out[name][f"transfer_{tag}"] = B.summarize(d)
    print(name, "zs/map", round(out[name]["transfer_zero_shot"]["mean"], 2), round(out[name]["transfer_map"]["mean"], 2), flush=True)
Path(__file__).with_name("vd_k1.json").write_text(json.dumps(out, indent=1))

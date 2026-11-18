"""Attach held-out pipeline results (zero-shot at the Stage-2 fold mean, per-box MAP) from PaperUpdate/preds/<key>.npz to an
existing model entry of bench_corr.json, without rerunning the population fits. Used for van Daalen (owner 2026-11-13: let
van Daalen be fit to the data with MAP, like the analytic forms): python -m PaperUpdate.Benchmark.attach_pipeline "vD20 refit"=VD
"""
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(M))

from PaperUpdate.Benchmark.bench import OUT, pipeline_transfer, summarize


def main(args):
    res = json.loads(OUT.read_text())
    for a in args:
        name, key = a.split("=")
        pt = pipeline_transfer(key)
        assert pt, key
        res[name]["transfer_zero_shot"] = summarize(pt["zero_shot"]); res[name]["transfer_map"] = summarize(pt["map"])
        res[name]["pipeline_key"] = key
        print(name, {k: round(v["mean"], 2) for k, v in res[name].items() if isinstance(v, dict)})
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])

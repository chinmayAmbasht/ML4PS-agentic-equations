"""Rewrite ONLY outputs/tables/pool_table.tex (App. B, tab:pool_full) with the L_MAP column in the Fig. 3 MAP setting
(PaperUpdate/L_map_fvh.json from map_fvh_all.py). Every other column comes from ranking.json exactly as make_tex.pool writes it.
Refuses to write if any ranked form lacks an fvh value, or if the new file differs from the current one outside the L_MAP column.
Usage: python -m PaperUpdate.pool_table_fvh
"""
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(M))

from PaperUpdate import make_tex as MT


def strip_last(line):
    """Row without its last numeric cell, for the 'only L_MAP changed' check."""
    return line.rsplit("&", 1)[0] if line.rstrip().endswith("\\\\") and "&" in line else line


def main():
    R = json.loads((M / "PaperUpdate" / "ranking.json").read_text())
    F = json.loads((M / "PaperUpdate" / "L_map_fvh.json").read_text())
    miss = [r["key"] for r in R if r["key"] not in F]
    if miss:
        sys.exit(f"missing fvh L_MAP for {len(miss)} forms: {miss}")
    for r in R:
        r["L_map_corr"] = F[r["key"]]["mean"]
    new = MT.pool(R)
    p = MT.DST / "pool_table.tex"
    old = p.read_text()
    a, b = old.splitlines(), new.splitlines()
    bad = [(x, y) for x, y in zip(a, b) if strip_last(x) != strip_last(y)]
    if len(a) != len(b) or bad:
        sys.exit(f"pool_table.tex differs outside the L_MAP column ({len(bad)} rows); not written. First: {bad[:1]}")
    p.write_text(new)
    (M / "PaperUpdate" / "pool_table_fvh.tex").write_text(new)
    print("wrote", p, "| rows changed:", sum(x != y for x, y in zip(a, b)))


if __name__ == "__main__":
    main()

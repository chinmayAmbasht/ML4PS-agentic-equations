#!/bin/bash
# 10 PySR searches, sequential, 2700 s each (Plans/04_pysr.md). Pooled first: they feed the pipeline pass.
cd "$(dirname "$0")/.." || exit 1
mkdir -p PySR/logs
for spec in "U pooled" "C pooled" "U IllustrisTNG" "C IllustrisTNG" "U SIMBA" "C SIMBA" "U Astrid" "C Astrid" "U Swift-EAGLE" "C Swift-EAGLE"; do
  read -r v f <<< "$spec"
  if [ -f "PySR/runs/${v}_${f}/result.json" ]; then echo "$(date +%T) skip $v $f (done)"; continue; fi
  echo "$(date +%T) start $v $f"
  python -m PySR.run_search "$v" "$f" 2700 0 > "PySR/logs/${v}_${f}.log" 2>&1
  rc=$?
  echo "$(date +%T) done $v $f exit=$rc $(grep -o '"L_zero_corr": [0-9.]*' "PySR/logs/${v}_${f}.log")"
done

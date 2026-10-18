#!/usr/bin/env bash
# Reruns every course project (CPU; CUDA projects also on the GPU when available) and rebuilds the reports.
# Usage: bash run_all.sh [01 02 ...]   (default: all projects)
cd "$(dirname "$0")"
sel=("$@"); [ ${#sel[@]} -eq 0 ] && sel=(01 02 03 04 05 06 07)
for n in "${sel[@]}"; do
  d=$(ls -d ${n}_*/ | head -1); d=${d%/}
  echo "================ $d"; start=$(date +%s)
  if bash "$d/run.sh" > "$d/run_all_log.txt" 2>&1; then status=OK; else status=FAILED; fi
  if command -v pdflatex >/dev/null; then (cd "$d/report" && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null); fi
  echo "$d: $status in $(( $(date +%s) - start )) s (log: $d/run_all_log.txt)"
done

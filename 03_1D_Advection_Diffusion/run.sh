#!/usr/bin/env bash
# Project 3: 1-D advection-diffusion of an instantaneous dye release.
# Runs the corrected C code for u = 0, 1, 1.5, 2 m/s (only the U literal is substituted), then plots.
set -e
cd "$(dirname "$0")"; mkdir -p build results
for u in 0.0 1.0 1.5 2.0; do
  sed "s/#define U 2.0 /#define U $u /" code/advection_diffusion.c > build/ad_u$u.c
  gcc -O2 -o build/ad_u$u build/ad_u$u.c -lm
  (cd build && ./ad_u$u && mv results.txt ../results/results_u$u.txt)
done
python3 code/postprocess.py

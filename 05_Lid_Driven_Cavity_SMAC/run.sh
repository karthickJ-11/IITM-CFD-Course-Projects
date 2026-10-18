#!/usr/bin/env bash
# Project 5: SMAC lid-driven cavity, Re = 100. Runs the 99x99-cell case (as in the code) and a
# 49x49-cell case for a grid check, then post-processes.  ~4 min on one CPU core.
set -e
cd "$(dirname "$0")"; mkdir -p build/g100 build/g50 results/grid_100 results/grid_50
gcc -O3 -march=native -o build/g100/smac code/smac_cavity.c -lm
sed 's/#define NX 100 /#define NX 50 /; s/#define NY 100/#define NY 50/' code/smac_cavity.c > build/g50/smac50.c
gcc -O3 -march=native -o build/g50/smac build/g50/smac50.c -lm
(cd build/g100 && ./smac | tee console_log.txt && mv *.txt ../../results/grid_100/)
(cd build/g50  && ./smac | tee console_log.txt && mv *.txt ../../results/grid_50/)
python3 code/postprocess.py

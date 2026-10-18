#!/usr/bin/env bash
# Iteration-by-iteration animation of the six corrected solvers of Project 6 into video/ (needs ffmpeg).
# The codes are run unchanged; each writes the grid after every iteration to temperature_data.txt.
set -e
cd "$(dirname "$0")"; mkdir -p build/video video
for s in point_gauss_seidel line_gauss_seidel psor lsor adi adi_relaxation; do
  gcc -O2 -o build/video/$s code/$s.c -lm
  (mkdir -p build/video/run_$s && cd build/video/run_$s && ../$s > log.txt)
done
python3 code/make_video.py

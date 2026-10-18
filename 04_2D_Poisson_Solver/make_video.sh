#!/usr/bin/env bash
# Iteration animation of Project 4 into video/ (needs ffmpeg): Jacobi, Gauss-Seidel and SUR on the
# 21 x 21 grid (delta = 0.05), both boundary-condition cases. The same codes are rebuilt in build/video with
# only delta, CASE and the output interval (every 100 -> every 5 iterations) substituted.
set -e
cd "$(dirname "$0")"; mkdir -p build/video video
for m in jacobi gauss_seidel sur; do for c in 1 2; do
  tag=${m}_case${c}
  sed "s/#define delta_x 0.01/#define delta_x 0.05/; s/#define delta_y 0.01/#define delta_y 0.05/; s/#define CASE 1 /#define CASE $c /; s/#define WRITE_EVERY 100 /#define WRITE_EVERY 5 /" code/$m.c > build/video/$tag.c
  gcc -O2 -w -o build/video/$tag build/video/$tag.c -lm
  (mkdir -p build/video/run_$tag && cd build/video/run_$tag && ../$tag > log.txt)
done; done
python3 code/make_video.py

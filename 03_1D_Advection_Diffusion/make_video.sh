#!/usr/bin/env bash
# Time-step animation of Project 3 into video/ (needs ffmpeg). The same code is rebuilt in build/ with
# only the output test (every 1 s -> every 5 steps = 0.05 s) and the end time (100 s -> 12 s) substituted, for u = 0, 1, 1.5, 2 m/s.
set -e
cd "$(dirname "$0")"; mkdir -p build/video video
for u in 0.0 1.0 1.5 2.0; do
  sed "s/#define U 2.0 /#define U $u /; s/#define T_MAX 100.0/#define T_MAX 12.0/; s/if(fmod(current_time, 1.0) <= DT\/2.0)/if((n + 1) % 5 == 0)/" \
      code/advection_diffusion.c > build/video/ad_u$u.c
  gcc -O2 -o build/video/ad_u$u build/video/ad_u$u.c -lm
  (cd build/video && ./ad_u$u > /dev/null && mv results.txt results_u$u.txt)
done
python3 code/make_video.py

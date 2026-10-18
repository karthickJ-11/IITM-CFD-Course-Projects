#!/usr/bin/env bash
# Builds the time-step animations of Project 7 into video/ (needs ffmpeg).
# Uses results/temperature_dt0.1.txt from run.sh, plus one extra run of the same code with dt = 2.5 s
# (gamma = 0.625 > 1/2) continued to t = 60 s to show the growth of the unstable mode.
set -e
cd "$(dirname "$0")"; mkdir -p build video
[ -f results/temperature_dt0.1.txt ] || bash run.sh
sed "s/double dt = 0.001;/double dt = 2.5;/; s/double t = 10.0;/double t = 60.0;/" code/assignment_1.c > build/v_unstable.c
gcc -O2 -o build/v_unstable build/v_unstable.c -lm
( cd build && ./v_unstable > /dev/null && mv temperature_output.txt ../build/video_unstable.txt )
python3 code/make_video.py

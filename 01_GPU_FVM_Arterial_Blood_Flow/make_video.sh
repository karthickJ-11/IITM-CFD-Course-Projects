#!/usr/bin/env bash
# Time-step animation of Project 1 into video/ (needs ffmpeg): start-up of Bird-Carreau flow in the LAD model,
# t = 0 ... 0.3 s, one VTK snapshot every 150 time steps (~5 ms). Unmodified code; only command-line arguments.
# GPU if nvcc is present, else CPU emulation (~3 min).
set -e
cd "$(dirname "$0")"; mkdir -p build/video video
if command -v nvcc >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then nvcc -O3 -arch=native -o build/video/bf code/Blood_flow.cu
else tools/emulate.sh code/Blood_flow.cu build/video/bf -w; fi
(cd build/video && rm -f blood_flow_t*.vtk && ./bf 0.3 150 1.85 1000 1 > log.txt && tail -1 log.txt)
python3 code/make_video.py

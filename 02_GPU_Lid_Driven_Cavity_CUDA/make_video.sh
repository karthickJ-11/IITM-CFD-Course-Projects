#!/usr/bin/env bash
# Time-step animation of Project 2 into video/ (needs ffmpeg). 129 x 129 grid, the CUDA code with only the grid,
# Dt and OUTPUT_INTERVAL (20000 -> 100 steps) substituted; run on the GPU if nvcc is present, else CPU emulation.
set -e
cd "$(dirname "$0")"; mkdir -p build/video video
sed 's/#define DX 0.0025/#define DX 0.0078125/; s/#define DY 0.0025/#define DY 0.0078125/; s/#define Dt 1.5e-4 /#define Dt 1.0e-3 /; s/#define OUTPUT_INTERVAL 20000/#define OUTPUT_INTERVAL 100/' \
    code/Lid_driven_Cavity.cu > build/video/lid129_video.cu
if command -v nvcc >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then nvcc -O3 -arch=native -o build/video/lid build/video/lid129_video.cu
else tools/emulate.sh build/video/lid129_video.cu build/video/lid -w; fi
(cd build/video && rm -rf output && ./lid > log.txt && tail -1 log.txt)
python3 code/make_video.py

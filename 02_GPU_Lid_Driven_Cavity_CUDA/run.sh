#!/usr/bin/env bash
# Project 2: CUDA vorticity-stream function lid-driven cavity (Re = 100).
#  * If nvcc and an NVIDIA GPU are available: builds code/Lid_driven_Cavity.cu for the 129, 257 and 401 grids
#    and runs them on the GPU (results/run_<N>_gpu, timings in results/timing.txt).
#  * Always (unless "gpu"): runs the SAME kernels serially on the CPU through tools/cuda_cpu_shim.h
#    (results/run_<N>) -- the verification path used for the report and the single-core baseline for the
#    GPU speed-up.
#  * "study": additionally repeats the three grids with a 100x tighter steady-state tolerance
#    (results/grid_study_tight/run_<N>) for the grid-convergence study (about 25 min on one CPU core).
# Usage: bash run.sh [gpu|study]
set -e
cd "$(dirname "$0")"; mkdir -p build results
declare -A H=( [129]=0.0078125 [257]=0.00390625 [401]=0.0025 )   # grid spacing
declare -A T=( [129]=1.0e-3    [257]=3.5e-4     [401]=1.5e-4 )   # time step (diffusion number < 1/4)
mk () {  # $1 = grid, $2 = output file, $3 = extra sed expression
  sed "s/#define DX 0.0025/#define DX ${H[$1]}/; s/#define DY 0.0025/#define DY ${H[$1]}/; s/#define Dt 1.5e-4 /#define Dt ${T[$1]} /; $3" \
      code/Lid_driven_Cavity.cu > "$2"; }
run_case () {  # $1 = executable, $2 = run directory
  mkdir -p "$2" && (cd "$2" && "$1" > console_log.txt && grep -E "Converged|wall time" console_log.txt)
  mv "$2/output/flow_999999.dat" "$2/" && rm -rf "$2/output"; }
: > results/timing.txt
if command -v nvcc >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee results/gpu_info.txt
  for g in 129 257 401; do
    mk $g build/lid${g}.cu ""; nvcc -O3 -arch=native -o build/lid${g}_gpu build/lid${g}.cu
    run_case "$PWD/build/lid${g}_gpu" results/run_${g}_gpu
    echo "GPU $g: $(grep 'wall time' results/run_${g}_gpu/console_log.txt)" >> results/timing.txt
  done
else
  echo "nvcc / NVIDIA GPU not found: skipping the GPU runs" | tee -a results/timing.txt
fi
if [ "$1" != "gpu" ]; then
  for g in 129 257 401; do
    mk $g build/lid${g}.cu ""; tools/emulate.sh build/lid${g}.cu build/lid${g}_cpu -w
    run_case "$PWD/build/lid${g}_cpu" results/run_${g}
    echo "CPU (serial emulation) $g: $(grep 'wall time' results/run_${g}/console_log.txt)" >> results/timing.txt
  done
fi
if [ "$1" == "study" ]; then
  for g in 129 257 401; do
    mk $g build/lid${g}_tight.cu 's/#define Error (1.0e-2 \* Dt)/#define Error (1.0e-4 * Dt)/'
    tools/emulate.sh build/lid${g}_tight.cu build/lid${g}_tight -w
    run_case "$PWD/build/lid${g}_tight" results/grid_study_tight/run_${g}
  done
fi
cat results/timing.txt
python3 code/postprocess.py

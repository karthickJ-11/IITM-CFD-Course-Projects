#!/usr/bin/env bash
# Project 1: GPU finite-volume solver for Bird-Carreau blood flow in a 2-D LAD artery model.
# Builds code/Blood_flow.cu and produces every result used in report/report.pdf:
#   results/newtonian_20mm      lambda = 0 (Newtonian, mu = mu0), L = 20 mm, t = 0.3 s   (Poiseuille check)
#   results/carreau_full        Bird-Carreau, L = 140 mm, t = 2.0 s                     (main result)
#   results/ablation_*          first 20 ms of the full case with / without the block correction
# GPU path (nvcc + NVIDIA GPU present): the same cases are run on the GPU (results/gpu_*), timings in results/timing.txt.
# CPU path (default; "gpu" skips it): the unmodified kernels run serially through tools/cuda_cpu_shim.h.
# Usage: bash run.sh [gpu|cpu]        Total CPU-emulation time: about 15 min on one core.
set -e
cd "$(dirname "$0")"; mkdir -p build results
newt () { sed 's/^#define L  0.14f/#define L  0.02f/; s/^#define lambda 3.13f/#define lambda 0.0f /; s/^#define del_P 2.8863e3f/#define del_P 412.33f/' code/Blood_flow.cu > build/bf_newt.cu; }
cp code/Blood_flow.cu build/bf_full.cu; newt
case_run () {  # $1 exe, $2 dir, rest = arguments
  local exe=$1 dir=$2; shift 2; mkdir -p "$dir"; rm -f "$dir"/*.vtk
  (cd "$dir" && "$exe" "$@" > console_log.txt); tail -2 "$dir/console_log.txt"
  ls "$dir"/blood_flow_t*.vtk 2>/dev/null | sort | head -n -1 | xargs -r rm -f   # keep only the last VTK snapshot
}
: > results/timing.txt
if [ "$1" != "cpu" ] && command -v nvcc >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee results/gpu_info.txt
  nvcc -O3 -arch=native -o build/bf_full_gpu build/bf_full.cu
  nvcc -O3 -arch=native -o build/bf_newt_gpu build/bf_newt.cu
  case_run "$PWD/build/bf_newt_gpu" results/gpu_newtonian_20mm 0.3 3000 1.85 1000 1
  case_run "$PWD/build/bf_full_gpu" results/gpu_carreau_full 2.0 15000 1.85 1000 1
  for d in gpu_newtonian_20mm gpu_carreau_full; do echo "GPU $d: $(grep 'Total simulation time' results/$d/console_log.txt)" >> results/timing.txt; done
else
  echo "nvcc / NVIDIA GPU not found (or 'cpu' given): GPU runs skipped" | tee -a results/timing.txt
fi
if [ "$1" != "gpu" ]; then
  tools/emulate.sh build/bf_full.cu build/bf_full_cpu -w
  tools/emulate.sh build/bf_newt.cu build/bf_newt_cpu -w
  case_run "$PWD/build/bf_newt_cpu" results/newtonian_20mm 0.3 3000 1.85 1000 1
  case_run "$PWD/build/bf_full_cpu" results/carreau_full 2.0 15000 1.85 1000 1
  case_run "$PWD/build/bf_full_cpu" results/ablation_block_corr 0.02 100000 1.85 1000 1
  case_run "$PWD/build/bf_full_cpu" results/ablation_point_sor 0.02 100000 1.85 1000 0
  for d in newtonian_20mm carreau_full; do echo "CPU (serial emulation) $d: $(grep 'Total simulation time' results/$d/console_log.txt)" >> results/timing.txt; done
fi
cat results/timing.txt
python3 code/postprocess.py

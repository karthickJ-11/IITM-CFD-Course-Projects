#!/usr/bin/env bash
# emulate.sh <file.cu> <out_binary> [extra g++ flags]: build a .cu file for CPU execution with cuda_cpu_shim.h
set -e
here="$(cd "$(dirname "$0")" && pwd)"; src="$1"; out="$2"; shift 2
tmp="${out}_emu.cpp"
sed -e 's|#include <cuda.h>|#include "cuda_cpu_shim.h"|' -e 's|#include <cuda_runtime.h>||' \
    -e 's/\([A-Za-z_][A-Za-z0-9_]*\)<<<\([^,]*\), *\([^>]*\)>>>(\(.*\));/LAUNCH(\1, \2, \3, \4);/' "$src" > "$tmp"
g++ -O3 -march=native -I"$here" "$@" -o "$out" "$tmp" -lm

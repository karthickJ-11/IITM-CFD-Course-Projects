#!/usr/bin/env bash
# Project 6: builds and runs the six corrected solvers, the six original solvers (for comparison),
# an omega sweep (PSOR/LSOR/ADI-relaxed) and a grid study (LSOR), then makes all plots.
set -e
cd "$(dirname "$0")"
[ -f results/raw_runs.zip ] && (cd results && unzip -qo raw_runs.zip && rm raw_runs.zip) || true
mkdir -p build results/corrected results/original results/omega_sweep results/grid_study
run () { # $1 src  $2 outdir  $3 tag
  gcc -O2 -o build/$3 $1 -lm
  ( mkdir -p build/run_$3 && cd build/run_$3 && ../$3 > ../../$2/$3_log.txt
    if [ -f final_field.dat ]; then mv final_field.dat ../../$2/$3_field.dat
    else # original codes: take the "Final Temperature Grid" block printed (1 decimal) in temperature_data.txt
      python3 -c "import sys;L=open('temperature_data.txt').read().split('Final Temperature Grid:')[1].strip().splitlines();open(sys.argv[1],'w').write('\n'.join(l for l in L if l[:1] in ' 0123456789')+'\n')" ../../$2/$3_field.dat
    fi
    rm -f temperature_data.txt )   # per-iteration grid dump (large) not kept
}
for s in point_gauss_seidel line_gauss_seidel psor lsor adi adi_relaxation; do run code/$s.c results/corrected $s; done
for s in point-gauss_sidel Line_gauss_sidel PSOR Lsor ADI_method ADI_method_with_relaxation; do run original/code_JulNov2024/$s.c results/original $s; done
for w in 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.65 1.7 1.75 1.8 1.85 1.9 1.95; do
  for s in psor lsor adi_relaxation; do
    sed -E "s/double omega = 1\.5;/double omega = $w;/" code/$s.c > build/${s}_w$w.c
    run build/${s}_w$w.c results/omega_sweep ${s}_w$w
  done
done
for n in 11 21 41; do m=$((2*n-1))
  sed -E "s/#define M 41/#define M $m/; s/#define N 21/#define N $n/; s/#define ERROR_THRESHOLD 0.01/#define ERROR_THRESHOLD 1e-9/" code/lsor.c > build/lsor_n$n.c
  run build/lsor_n$n.c results/grid_study lsor_n$n
done
python3 code/postprocess.py
(cd results && zip -qr raw_runs.zip corrected original omega_sweep grid_study && rm -rf corrected original omega_sweep grid_study)

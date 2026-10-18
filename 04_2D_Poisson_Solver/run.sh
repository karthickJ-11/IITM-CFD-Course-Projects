#!/usr/bin/env bash
# Project 4: 2-D Poisson equation. Runs Jacobi, Gauss-Seidel and SUR for both boundary-condition
# cases and delta = 0.1, 0.05, 0.01 (only the delta/CASE constants are substituted), keeps the
# final field of every run, then post-processes.
set -e
cd "$(dirname "$0")"; mkdir -p build results/fields
: > results/iteration_counts.txt
for m in jacobi gauss_seidel sur; do for c in 1 2; do for d in 0.1 0.05 0.01; do
  tag=${m}_case${c}_d${d}
  sed "s/#define delta_x 0.01/#define delta_x $d/; s/#define delta_y 0.01/#define delta_y $d/; s/#define CASE 1 /#define CASE $c /" code/$m.c > build/$tag.c
  gcc -O2 -w -o build/$tag build/$tag.c -lm
  ( cd build && msg=$(./$tag | tail -1) && echo "$m case$c d=$d : $msg" >> ../results/iteration_counts.txt
    python3 -c "import sys;B=[b for b in open('temperature_field.txt').read().split('\n\n\n') if b.strip()];open(sys.argv[1],'w').write(B[-1].strip()+'\n')" ../results/fields/$tag.dat
    rm -f temperature_field.txt )
done; done; done
cat results/iteration_counts.txt
# same codes, tight tolerance (ERROR = 1e-6, Max_iteration = 200000): fully converged reference runs
mkdir -p results/fields_tight; : > results/iteration_counts_tight.txt
for m in jacobi gauss_seidel sur; do for c in 1 2; do for d in 0.1 0.05 0.01; do
  tag=${m}_case${c}_d${d}_tight
  sed "s/#define delta_x 0.01/#define delta_x $d/; s/#define delta_y 0.01/#define delta_y $d/; s/#define CASE 1 /#define CASE $c /; s/#define ERROR 0.0001/#define ERROR 1e-6/; s/#define Max_iteration 10000/#define Max_iteration 200000/; s/#define WRITE_EVERY 100 /#define WRITE_EVERY 100000 /" code/$m.c > build/$tag.c
  gcc -O2 -w -o build/$tag build/$tag.c -lm
  ( cd build && msg=$(./$tag | tail -1) && echo "$m case$c d=$d : $msg" >> ../results/iteration_counts_tight.txt
    python3 -c "import sys;B=[b for b in open('temperature_field.txt').read().split('\n\n\n') if b.strip()];open(sys.argv[1],'w').write(B[-1].strip()+'\n')" ../results/fields_tight/${m}_case${c}_d${d}.dat
    rm -f temperature_field.txt )
done; done; done
cat results/iteration_counts_tight.txt
python3 code/postprocess.py

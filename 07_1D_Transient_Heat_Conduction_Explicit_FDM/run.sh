#!/usr/bin/env bash
# Builds and runs the explicit FTCS solver for the three time steps used in the report,
# then makes all plots. Usage:  bash run.sh      (needs gcc, python3 with numpy+matplotlib)
set -e
cd "$(dirname "$0")"
mkdir -p results build
for DT in 0.1 0.01 0.001 2.5; do
  # same source; only the dt literal is substituted for each run (dt=2.5 -> gamma>0.5 stability check)
  sed "s/double dt = 0.001;/double dt = ${DT};/" code/assignment_1.c > build/a1_dt${DT}.c
  gcc -O2 -o build/a1_dt${DT} build/a1_dt${DT}.c -lm
  ( cd build && ./a1_dt${DT} > /dev/null && mv temperature_output.txt ../results/temperature_dt${DT}.txt )
  echo "dt=${DT} done"
done
python3 code/plot_results.py
# ---- accuracy studies (same source; only CX, dt and the end time are substituted) ----
mkdir -p results/study
run_case () {  # $1 = CX, $2 = dt, $3 = t_end, $4 = tag
  sed "s/#define CX 51/#define CX $1/; s/double dt = 0.001;/double dt = $2;/; s/double t = 10.0;/double t = $3;/" code/assignment_1.c > build/s_$4.c
  gcc -O2 -o build/s_$4 build/s_$4.c -lm
  ( cd build && ./s_$4 > /dev/null && tail -n $(( $1 + 2 )) temperature_output.txt > ../results/study/final_$4.txt && rm temperature_output.txt )
}
# (a) grid refinement at fixed gamma = 3/8, t = 48 s  (dx = 4, 2, 1, 0.5 cm; all dt exact in binary)
for c in "26 6.0" "51 1.5" "101 0.375" "201 0.09375"; do set -- $c; run_case $1 $2 48.0 grid_CX$1; done
# (b) gamma sweep at dx = 2 cm, t = 42 s (gamma = dt/4)
for d in 0.25 0.5 0.6666666666666666 1.0 1.5 1.75 2.0; do run_case 51 $d 42.0 gamma_dt$d; done
python3 code/study.py

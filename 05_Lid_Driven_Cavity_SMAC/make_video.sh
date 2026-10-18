#!/usr/bin/env bash
# Time-step animation of Project 5 into video/ (needs ffmpeg). The solver is unchanged; a copy in build/video
# gets one added call that writes the stream function, vorticity and velocity every 200 steps (frame files).
set -e
cd "$(dirname "$0")"; mkdir -p build/video/frames video
python3 - <<'PY'
s = open("code/smac_cavity.c").read()
snap = '''
/* [VIDEO ONLY] snapshot writer, inserted by make_video.sh into build/video/smac_video.c */
static void snapshot(int step) {
    char name[64]; sprintf(name, "frames/frame_%06d.dat", step); FILE *f = fopen(name, "w");
    compute_vorticity();
    for (int j = 0; j <= NC; j++) for (int i = 0; i <= NC; i++)
        fprintf(f, "%f %f %e %e %e %e\\n", i * DX, j * DY, S[i][j], Vor[i][j], 0.5 * (u[i][j] + u[i][j + 1]), 0.5 * (v[i][j] + v[i + 1][j]));
    fclose(f);
}
int main'''
s = s.replace("\nint main", snap, 1)
s = s.replace("change = update_velocity();", "change = update_velocity();\n        if (t == 1 || t % 200 == 0) snapshot(t);   /* [VIDEO ONLY] */", 1)
open("build/video/smac_video.c", "w").write(s)
PY
gcc -O3 -march=native -o build/video/smac_video build/video/smac_video.c -lm
(cd build/video && rm -f frames/* && ./smac_video > log.txt && tail -1 log.txt)
python3 code/make_video.py

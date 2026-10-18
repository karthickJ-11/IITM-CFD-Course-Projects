"""Project 2 post-processing: vorticity-stream function lid-driven cavity (Re = 100).
Reads the Tecplot point files written by Lid_driven_Cavity.cu (X Y U V Vorticity Stream) from
results/run_<N>/flow_999999.dat and writes the figures, results/summary.txt and results/grid_study.txt.
Extrema are located to sub-grid accuracy (quadratic fit through the extreme node and its neighbours);
the grid study follows Celik et al. (2008), J. Fluids Eng. 130, 078001 (observed order, Richardson
extrapolation and grid convergence index for non-uniform refinement ratios)."""
import numpy as np, os, glob, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results")
import zipfile  # raw run outputs are shipped zipped (results/raw_runs.zip); extract what is missing
if os.path.exists(os.path.join(R, "raw_runs.zip")):
    with zipfile.ZipFile(os.path.join(R, "raw_runs.zip")) as _z:
        for _m in _z.namelist():
            if not os.path.exists(os.path.join(R, _m)): _z.extract(_m, R)
# Ghia, Ghia & Shin (1982), Re = 100: u(0.5, y) and v(x, 0.5)
gy = np.array([1.0, .9766, .9688, .9609, .9531, .8516, .7344, .6172, .5, .4531, .2813, .1719, .1016, .0703, .0625, .0547, 0.0])
gu = np.array([1.0, .84123, .78871, .73722, .68717, .23151, .00332, -.13641, -.20581, -.21090, -.15662, -.10150, -.06434, -.04775, -.04192, -.03717, 0.0])
gx = np.array([1.0, .9688, .9609, .9531, .9453, .9063, .8594, .8047, .5, .2344, .2266, .1563, .0938, .0781, .0703, .0625, 0.0])
gv = np.array([0.0, -.05906, -.07391, -.08864, -.10313, -.16914, -.22445, -.24533, .05454, .17527, .17507, .16077, .12317, .10890, .10091, .09233, 0.0])

def load(f):
    a = np.loadtxt(f, skiprows=3); x = np.unique(a[:, 0]); y = np.unique(a[:, 1]); n = len(x)
    F = {k: a[:, c].reshape(len(y), n) for k, c in (("u", 2), ("v", 3), ("w", 4), ("psi", 5))}  # rows: y, cols: x
    return x, y, F

def ext1d(s, f, kind):
    """sub-grid extremum of samples f(s) by a parabola through the extreme node and its neighbours"""
    k = int(np.argmin(f) if kind == "min" else np.argmax(f)); k = min(max(k, 1), len(f) - 2)
    h = s[k + 1] - s[k]; fm, f0, fp = f[k - 1], f[k], f[k + 1]
    den = fm - 2 * f0 + fp; d = 0.5 * (fm - fp) / den if den != 0 else 0.0
    return f0 - 0.25 * (fm - fp) * d, s[k] + d * h

def ext2d(x, y, S):
    """sub-grid minimum of S: 2-D quadratic least-squares fit on the 3x3 stencil of the minimum node"""
    j, i = np.unravel_index(np.argmin(S), S.shape); h = x[1] - x[0]
    X, Y = np.meshgrid([-1, 0, 1], [-1, 0, 1]); Z = S[j - 1:j + 2, i - 1:i + 2]
    A = np.c_[np.ones(9), X.ravel(), Y.ravel(), X.ravel()**2, X.ravel() * Y.ravel(), Y.ravel()**2]
    c = np.linalg.lstsq(A, Z.ravel(), rcond=None)[0]
    M = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]]); dxy = np.linalg.solve(M, -c[1:3])
    val = c[0] + c[1] * dxy[0] + c[2] * dxy[1] + c[3] * dxy[0]**2 + c[4] * dxy[0] * dxy[1] + c[5] * dxy[1]**2
    return val, x[i] + dxy[0] * h, y[j] + dxy[1] * h

def centre(xs, F, axis):
    i = np.searchsorted(xs, 0.5)
    if abs(xs[i] - 0.5) < 1e-12: return F[:, i] if axis == 1 else F[i, :]
    w = (0.5 - xs[i - 1]) / (xs[i] - xs[i - 1])
    return (1 - w) * F[:, i - 1] + w * F[:, i] if axis == 1 else (1 - w) * F[i - 1, :] + w * F[i, :]

runs = {}
for d in sorted(glob.glob(os.path.join(R, "run_*")), key=lambda s: (("gpu" in s), int(os.path.basename(s).split("_")[1]))):
    f = os.path.join(d, "flow_999999.dat")
    if not os.path.exists(f): f = os.path.join(d, "output", "flow_999999.dat")
    if os.path.exists(f): runs[os.path.basename(d)[4:]] = (load(f), d)

rows = []
for name, ((x, y, F), d) in runs.items():
    h = x[1] - x[0]; S = F["psi"]; Wv = F["w"]
    lap = (S[1:-1, 2:] + S[1:-1, :-2] + S[2:, 1:-1] + S[:-2, 1:-1] - 4 * S[1:-1, 1:-1]) / h**2
    res = np.abs(lap + Wv[1:-1, 1:-1]); res_in = res[2:-2, 2:-2].max()   # interior (2 nodes away from walls)
    pmin, px, py = ext2d(x, y, S)
    uc = centre(x, F["u"], 1); vc = centre(y, F["v"], 0)
    umin, yu = ext1d(y, uc, "min"); vmax, xv1 = ext1d(x, vc, "max"); vmin, xv2 = ext1d(x, vc, "min")
    eu = np.interp(gy, y, uc) - gu; ev = np.interp(gx, x, vc) - gv
    # secondary (corner) eddies: extremum of psi of opposite sign in the bottom corners
    br = S[(y < 0.25)][:, x > 0.75]; bl = S[(y < 0.25)][:, x < 0.25]
    rows.append(dict(run=name, N=len(x), psi=pmin, xc=px, yc=py, eu=np.abs(eu).max(), ev=np.abs(ev).max(),
                     res=res.max(), res_in=res_in, umin=umin, yu=yu, vmax=vmax, xv1=xv1, vmin=vmin, xv2=xv2,
                     br=br.max(), bl=bl.max()))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    lev = np.sort(np.concatenate([[-0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -1e-3, -1e-4, -1e-5], [1e-6, 1e-5]]))
    ax[0].contour(x, y, S, levels=lev, colors="k", linewidths=0.6); ax[0].plot(px, py, "r+", ms=10)
    ax[0].set_title(rf"Stream function, $\psi_{{min}}$ = {pmin:.5f}")
    c = ax[1].contourf(x, y, np.clip(Wv, -5, 5), levels=np.linspace(-5, 5, 21), cmap="RdBu_r", extend="both")
    fig.colorbar(c, ax=ax[1], shrink=0.8); ax[1].set_title(r"Vorticity $\omega$ (clipped to $\pm 5$)")
    sp = np.hypot(F["u"], F["v"]); c = ax[2].contourf(x, y, sp, levels=21, cmap="magma"); fig.colorbar(c, ax=ax[2], shrink=0.8)
    s = max(1, len(x) // 25)
    ax[2].quiver(x[::s], y[::s], F["u"][::s, ::s], F["v"][::s, ::s], color="w", scale=12, width=0.003); ax[2].set_title("Velocity magnitude and vectors")
    for a in ax: a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
    fig.suptitle(f"Vorticity-stream function solver (CUDA kernels), Re = 100, grid {len(x)}x{len(y)}", y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(R, f"fig_fields_{name}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# centreline comparison
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for name, ((x, y, F), d) in runs.items():
    ax[0].plot(centre(x, F["u"], 1), y, label=f"{len(x)}x{len(y)}"); ax[1].plot(x, centre(y, F["v"], 0), label=f"{len(x)}x{len(y)}")
ax[0].plot(gu, gy, "ko", mfc="none", label="Ghia et al. (1982)"); ax[1].plot(gx, gv, "ko", mfc="none", label="Ghia et al. (1982)")
ax[0].set_xlabel("u(0.5, y)"); ax[0].set_ylabel("y"); ax[1].set_xlabel("x"); ax[1].set_ylabel("v(x, 0.5)")
for a in ax: a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_centreline_vs_ghia.png"), dpi=200); plt.close(fig)

# convergence histories
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for name, ((x, y, F), d) in runs.items():
    hf = os.path.join(d, "convergence_history.txt")
    if os.path.exists(hf):
        hh = np.atleast_2d(np.loadtxt(hf)); dt = hh[0, 1] / hh[0, 0]
        ax.semilogy(hh[:, 1], hh[:, 2] / dt, label=f"{len(x)}x{len(y)}, $\\Delta t$ = {dt:g}")
ax.axhline(1e-2, color="k", ls="--", lw=0.8)
ax.set_xlabel("t"); ax.set_ylabel(r"max$|\omega^{n+1}-\omega^n|/\Delta t$"); ax.legend(fontsize=8); ax.set_title("Convergence to steady state")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_convergence_history.png"), dpi=200); plt.close(fig)

with open(os.path.join(R, "summary.txt"), "w") as fh:
    fh.write("run | N | psi_min | x_c | y_c | u_min(0.5,y) @y | v_max(x,0.5) @x | v_min(x,0.5) @x | max|u-uGhia| | max|v-vGhia| | "
             "max|lap(psi)+w| all / interior | psi max bottom-right / bottom-left eddy\n")
    for r in rows:
        fh.write(f"{r['run']} | {r['N']} | {r['psi']:.6f} | {r['xc']:.4f} | {r['yc']:.4f} | {r['umin']:.5f} @{r['yu']:.4f} | "
                 f"{r['vmax']:.5f} @{r['xv1']:.4f} | {r['vmin']:.5f} @{r['xv2']:.4f} | {r['eu']:.4f} | {r['ev']:.4f} | "
                 f"{r['res']:.3g} / {r['res_in']:.3g} | {r['br']:.3e} / {r['bl']:.3e}\n")
    fh.write("Ghia (1982): psi_min = -0.103423 at (0.6172, 0.7344); u_min = -0.21090 (y = 0.4531), "
             "v_max = 0.17527 (x = 0.2344), v_min = -0.24533 (x = 0.8047); eddies BR 1.25e-5, BL 1.75e-6\n")

# grid convergence study: tightly converged runs (run.sh study) if present, else the CPU-emulation runs
tight = sorted(glob.glob(os.path.join(R, "grid_study_tight", "run_*", "flow_999999.dat")))
if len(tight) >= 3:
    cpu, src = [], "results/grid_study_tight (steady-state tolerance |dw/dt| < 1e-4)"
    for f in tight:
        x, y, F = load(f); uc = centre(x, F["u"], 1); vc = centre(y, F["v"], 0)
        cpu.append(dict(N=len(x), psi=ext2d(x, y, F["psi"])[0], umin=ext1d(y, uc, "min")[0],
                        vmax=ext1d(x, vc, "max")[0], vmin=ext1d(x, vc, "min")[0]))
    cpu.sort(key=lambda r: -r["N"])
else:
    cpu, src = sorted([r for r in rows if r["run"].isdigit()], key=lambda r: -r["N"]), "results/run_<N> (|dw/dt| < 1e-2)"
if len(cpu) >= 3:
    f1, f2, f3 = cpu[0], cpu[1], cpu[2]; h1, h2, h3 = [1.0 / (r["N"] - 1) for r in (f1, f2, f3)]
    r21, r32 = h2 / h1, h3 / h2
    out = [f"Grid convergence (Celik et al. 2008): grids {f1['N']}, {f2['N']}, {f3['N']}  (r21 = {r21:.4f}, r32 = {r32:.4f}); data: {src}",
           "quantity | phi1 (finest) | phi2 | phi3 | observed p | extrapolated phi | GCI_fine [%]"]
    plotdata = []
    for key, label in (("psi", "psi_min"), ("umin", "u_min"), ("vmax", "v_max"), ("vmin", "v_min")):
        p1, p2, p3 = f1[key], f2[key], f3[key]; e21, e32 = p2 - p1, p3 - p2
        s = np.sign(e32 / e21) if e21 != 0 else 1.0; p = 2.0
        for _ in range(200):
            q = np.log((r21**p - s) / (r32**p - s)); p = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        ext = (r21**p * p1 - p2) / (r21**p - 1); gci = 1.25 * abs((p1 - p2) / p1) / (r21**p - 1) * 100
        out.append(f"{label} | {p1:.6f} | {p2:.6f} | {p3:.6f} | {p:.2f} | {ext:.6f} | {gci:.3f}")
        plotdata.append((label, [h1, h2, h3], [abs(p1 - ext), abs(p2 - ext), abs(p3 - ext)], p))
    open(os.path.join(R, "grid_study.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for (label, hs, es, p), m in zip(plotdata, "osd^"):
        ax.loglog(hs, es, m + "-", label=f"{label}  (p = {p:.2f})")
    hh = np.array([h1, h3]); ax.loglog(hh, 0.5 * max(e for _, _, es, _ in plotdata for e in es) * (hh / h3)**2, "k--", lw=0.8, label=r"slope 2")
    ax.set_xlabel("grid spacing h"); ax.set_ylabel(r"$|\phi_h - \phi_{ext}|$"); ax.legend(fontsize=8)
    ax.set_title("Grid convergence of benchmark quantities"); fig.tight_layout(); fig.savefig(os.path.join(R, "fig_grid_convergence.png"), dpi=200); plt.close(fig)
print(open(os.path.join(R, "summary.txt")).read())

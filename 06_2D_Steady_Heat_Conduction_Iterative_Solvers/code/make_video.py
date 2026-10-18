"""Animation for Project 6: temperature field of the 41 x 21 plate after each iteration of the six solvers."""
import numpy as np, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams.update({"font.family": "serif", "font.size": 10})
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(H, "..", "build", "video"); V = os.path.join(H, "..", "video")
M, N = 41, 21
def read(f):
    grids, cur = [], []
    for line in open(f):
        s = line.strip()
        if not s: continue
        if s[0].isdigit() or s[0] == "-":
            try: cur.append([float(v) for v in s.split()])
            except ValueError: continue
            if len(cur) == M: grids.append(np.array(cur)); cur = []
        else: cur = []
    return grids
names = [("point_gauss_seidel", "Point Gauss-Seidel"), ("line_gauss_seidel", "Line Gauss-Seidel"), ("psor", r"PSOR ($\omega$=1.5)"),
         ("lsor", r"LSOR ($\omega$=1.5)"), ("adi", "ADI"), ("adi_relaxation", r"ADI with relaxation ($\omega$=1.5)")]
G = {k: read(os.path.join(B, f"run_{k}", "temperature_data.txt"))[:-1] for k, _ in names}  # last block = repeated final grid
for k in G: print(k, len(G[k]) - 1, "iterations")
final = {k: G[k][-1] for k in G}; nmax = max(len(g) for g in G.values())
x = np.linspace(0, 1, N); y = np.linspace(2, 0, M); lv = np.linspace(30, 100, 15)
fig = plt.figure(figsize=(12.8, 7.2)); gs = fig.add_gridspec(2, 7, height_ratios=[1.45, 1], width_ratios=[1]*6 + [0.12], hspace=0.45)
axs = [fig.add_subplot(gs[0, i]) for i in range(6)]; cax = fig.add_subplot(gs[0, 6]); ah = fig.add_subplot(gs[1, :6]); ttl = fig.suptitle("")
err = {k: np.array([np.abs(g - final[k]).max() for g in G[k]]) for k in G}; cols = plt.cm.tab10(np.arange(6))
lines = {}
for c, (key, lab) in zip(cols, names):
    ah.semilogy(np.arange(len(err[key])), np.maximum(err[key], 1e-2), color=c, lw=1, alpha=0.35)
    lines[key], = ah.semilogy([], [], color=c, lw=2, label=lab)
ah.set_xlim(0, nmax); ah.set_ylim(1e-2, 100); ah.set_xlabel("iteration"); ah.set_ylabel(r"max$|T-T_{final}|$ [$^\circ$C]"); ah.grid(alpha=0.3); ah.legend(fontsize=8, ncol=3)
sm = plt.cm.ScalarMappable(cmap="inferno", norm=matplotlib.colors.Normalize(30, 100)); fig.colorbar(sm, cax=cax, label=r"T [$^\circ$C]")
def upd(k):
    for a, (key, lab) in zip(axs, names):
        g = G[key]; i = min(k, len(g) - 1); F = g[i]; a.clear()
        a.contourf(x, y, F, levels=lv, cmap="inferno", extend="both"); a.contour(x, y, F, levels=lv, colors="w", linewidths=0.3)
        done = "converged" if i == len(g) - 1 else f"iter {i}"
        a.set_title(f"{lab}\n{done} ({len(g) - 1} its)", fontsize=8.5)
        a.set_aspect("equal"); a.set_xticks([0, 1]); a.set_yticks([0, 1, 2]) if a is axs[0] else a.set_yticks([])
        lines[key].set_data(np.arange(i + 1), np.maximum(err[key][:i + 1], 1e-2))
    ttl.set_text(f"2-D steady conduction, plate 1 m x 2 m (T = 100 at the bottom, 30 elsewhere): iteration {k}")
frames = list(range(0, 40)) + list(range(40, nmax + 6, 3))
anim = FuncAnimation(fig, upd, frames=frames)
anim.save(os.path.join(V, "iterative_solvers_convergence.mp4"), writer=FFMpegWriter(fps=12, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
print("wrote video/iterative_solvers_convergence.mp4")

"""Animation for Project 4: field every 5 iterations for Jacobi / Gauss-Seidel / SUR, both cases (delta = 0.05)."""
import numpy as np, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams.update({"font.family": "serif", "font.size": 10})
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(H, "..", "build", "video"); V = os.path.join(H, "..", "video")
def read(f):
    blocks = [b for b in open(f).read().split("\n\n\n") if b.strip()]
    return [np.array([[float(v) for v in l.split()] for l in b.strip().splitlines()]) for b in blocks]
meth = [("jacobi", "Jacobi"), ("gauss_seidel", "Gauss-Seidel"), ("sur", "SUR")]
G = {(m, c): read(os.path.join(B, f"run_{m}_case{c}", "temperature_field.txt")) for m, _ in meth for c in (1, 2)}
# last block is the final state (written after the loop); block k holds iteration 5(k+1)
for k, v in G.items(): print(k, len(v), "frames")
fin = {k: v[-1] for k, v in G.items()}; n = max(len(v) for v in G.values()); x = np.linspace(0, 1, 21)
lv = {c: np.linspace(min(fin[(m, c)].min() for m, _ in meth), max(fin[(m, c)].max() for m, _ in meth), 22) for c in (1, 2)}
fig, axs = plt.subplots(2, 3, figsize=(12.8, 7.2)); fig.subplots_adjust(hspace=0.38, top=0.88); ttl = fig.suptitle("")
def upd(k):
    for r, c in enumerate((1, 2)):
        for col, (m, lab) in enumerate(meth):
            a = axs[r, col]; g = G[(m, c)]; i = min(k, len(g) - 1); F = g[i]; a.clear()
            a.contourf(x, x, F.T, levels=lv[c], cmap="inferno" if c == 1 else "viridis", extend="both")
            a.contour(x, x, F.T, levels=lv[c], colors="w", linewidths=0.3); a.set_aspect("equal")
            it = "converged" if i == len(g) - 1 else f"iteration {5 * (i + 1)}"
            a.set_title(f"Case {c}, {lab}: {it}\nmax|T - T_final| = {np.abs(F - fin[(m, c)]).max():.3f}", fontsize=9)
            a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
    ttl.set_text(f"2-D Poisson equation with point sources, 21 x 21 grid ($\\Delta$ = 0.05): iteration {5 * (k + 1)}")
anim = FuncAnimation(fig, upd, frames=list(range(n + 8)))
anim.save(os.path.join(V, "poisson_iterations_jacobi_gs_sur.mp4"), writer=FFMpegWriter(fps=10, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
print("wrote video/poisson_iterations_jacobi_gs_sur.mp4")

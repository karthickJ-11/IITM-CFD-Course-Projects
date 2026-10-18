"""Animation for Project 3 (1-D advection-diffusion, FTBS upwind + central diffusion). Output: video/*.mp4"""
import numpy as np, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(H, "..", "build", "video"); V = os.path.join(H, "..", "video")
Dm, dx, dt, Cd = 0.5, 0.2, 0.01, 10.0; M = Cd * dx
def read(f):
    out, t, rows = [], None, []
    for line in open(f):
        if line.startswith("# Time ="):
            if t is not None and rows: out.append((t, np.array(rows)))
            t = float(line.split("=")[1].split()[0]); rows = []
        elif line.strip() and not line.startswith("#") and not line.startswith("Position"):
            rows.append([float(v) for v in line.split()])
    if rows: out.append((t, np.array(rows)))
    return out
us = [0.0, 1.0, 1.5, 2.0]; D = {u: read(os.path.join(B, f"results_u{u}.txt")) for u in us}
n = min(len(D[u]) for u in us)
fig, axs = plt.subplots(2, 2, figsize=(12.8, 7.2), sharex=True); axs = axs.ravel(); L = {}
for a, u in zip(axs, us):
    cfl = u * dt / dx; Dn = u * dx / 2 * (1 - cfl)
    l1, = a.plot([], [], "-", color="tab:blue", lw=2, label="FTBS (C code)"); l2, = a.plot([], [], "k:", lw=1.3, label=r"exact, $D$")
    l3, = a.plot([], [], "r--", lw=1.1, label=r"exact, $D+D_{num}$"); L[u] = (l1, l2, l3, Dn)
    a.set_xlim(-10, 10); a.set_ylim(0, 0.8); a.set_title(rf"u = {u:g} m/s,  CFL = {cfl:.2f},  $D_{{num}}$ = {Dn:.3f} m$^2$/s", fontsize=11)
    a.set_ylabel(r"C [kg/m$^3$]")
axs[2].set_xlabel("x [m]"); axs[3].set_xlabel("x [m]"); axs[0].legend(loc="upper left", fontsize=9)
ttl = fig.suptitle(""); fig.tight_layout(rect=(0, 0, 1, 0.95))
def ex(x, t, u, Dd): return M / np.sqrt(4 * np.pi * Dd * t) * np.exp(-(x - u * t) ** 2 / (4 * Dd * t)) if t > 0 else 0 * x
def upd(k):
    for u in us:
        t, b = D[u][k]; x = b[:, 0]; xf = np.linspace(-10, 10, 801); l1, l2, l3, Dn = L[u]
        l1.set_data(x, b[:, 1]); l2.set_data(xf, ex(xf, t, u, Dm)); l3.set_data(xf, ex(xf, t, u, Dm + Dn))
        l1.axes.set_ylim(0, max(0.5, 1.15 * float(np.max(ex(xf, t, u, Dm)))))
    ttl.set_text(f"Advection-diffusion of a dye release (dx = 0.2 m, dt = 0.01 s):  t = {D[us[0]][k][0]:.2f} s")
anim = FuncAnimation(fig, upd, frames=range(1, n))
anim.save(os.path.join(V, "advection_diffusion_u0_1_1.5_2.mp4"), writer=FFMpegWriter(fps=20, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
print("wrote video/advection_diffusion_u0_1_1.5_2.mp4  frames", n - 1)

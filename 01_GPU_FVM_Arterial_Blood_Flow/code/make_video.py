"""Animation for Project 1: start-up of Bird-Carreau blood flow in the 2-D LAD model (VTK snapshots)."""
import numpy as np, os, glob, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams.update({"font.family": "serif", "font.size": 10})
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(H, "..", "build", "video"); V = os.path.join(H, "..", "video")
def read_vtk(f):
    L = open(f).read().split("\n"); i = 0; d = {}
    while i < len(L):
        s = L[i].split()
        if s and s[0] == "DIMENSIONS": nx, ny = int(s[1]) - 1, int(s[2]) - 1
        if s and s[0] == "SCALARS":
            d[s[1]] = np.array(L[i + 2:i + 2 + nx * ny], float).reshape(ny, nx); i += 2 + nx * ny; continue
        if s and s[0] == "VECTORS":
            v = np.array(" ".join(L[i + 1:i + 1 + nx * ny]).split(), float).reshape(ny, nx, 3); d["u"], d["v"] = v[..., 0], v[..., 1]; i += 1 + nx * ny; continue
        i += 1
    return d, nx, ny
fs = sorted(glob.glob(os.path.join(B, "blood_flow_t*.vtk")), key=lambda f: float(os.path.basename(f)[12:-4]))
d0, nx, ny = read_vtk(fs[0]); h = 1e-4; x = (np.arange(nx) + 0.5) * h * 1e3; y = (np.arange(ny) + 0.5) * h * 1e3; m = x <= 30
fig = plt.figure(figsize=(12.8, 7.2)); gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], hspace=0.55, wspace=0.18)
a_u, a_mu, a_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0]); a_p = fig.add_subplot(gs[:, 1]); ttl = fig.suptitle("")
from matplotlib import colors
cb1 = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 0.6), cmap="viridis"), ax=a_u, pad=0.01, label="|u| [m/s]")
cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=colors.LogNorm(3.5, 56), cmap="cividis"), ax=a_mu, pad=0.01, label=r"$\mu$ [mPa s]")
def upd(k):
    d, _, _ = read_vtk(fs[k]); t = float(os.path.basename(fs[k])[12:-4])
    for a in (a_u, a_mu, a_c, a_p): a.clear()
    a_u.contourf(x[m], y, d["velocity_magnitude"][:, m], levels=np.linspace(0, 0.6, 25), cmap="viridis", extend="max")
    a_u.set_title("velocity magnitude, entrance region (y stretched)", fontsize=9, loc="left"); a_u.set_ylabel("y [mm]")
    a_mu.contourf(x[m], y, np.clip(d["viscosity"][:, m] * 1e3, 3.5, 56), levels=np.logspace(np.log10(3.5), np.log10(56), 25), norm=colors.LogNorm(3.5, 56), cmap="cividis")
    a_mu.set_title("Bird-Carreau viscosity", fontsize=9, loc="left"); a_mu.set_ylabel("y [mm]")
    a_c.plot(x, d["u"][ny // 2], "tab:blue"); a_c.set_xlim(0, 140); a_c.set_ylim(0.50, 0.60); a_c.set_xlabel("x [mm]"); a_c.set_ylabel("centreline u [m/s]")
    a_c.axhline(0.5636, color="k", ls="--", lw=0.8); a_c.grid(alpha=0.3); a_c.set_title("centreline velocity along the artery (dashed: developed Carreau value)", fontsize=9, loc="left")
    for xs, c in ((1, "tab:orange"), (10, "tab:green"), (100, "tab:red")):
        i = int(xs / 0.1); a_p.plot(d["u"][:, i], y, "-", color=c, label=f"x = {xs} mm")
    a_p.plot(1.5 * 0.39 * (1 - (2 * y / 3.5 - 1) ** 2), y, "k:", label="inlet profile"); a_p.set_xlim(0, 0.65); a_p.set_xlabel("u [m/s]"); a_p.set_ylabel("y [mm]")
    a_p.legend(fontsize=8, loc="lower right"); a_p.grid(alpha=0.3); a_p.set_title("velocity profiles", fontsize=10)
    ttl.set_text(f"GPU finite-volume blood flow in the LAD (Bird-Carreau, 1400 x 35 cells): start-up from rest,  t = {t * 1e3:.1f} ms")
anim = FuncAnimation(fig, upd, frames=len(fs))
anim.save(os.path.join(V, "blood_flow_LAD_startup.mp4"), writer=FFMpegWriter(fps=8, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
print("wrote video/blood_flow_LAD_startup.mp4,", len(fs), "frames")

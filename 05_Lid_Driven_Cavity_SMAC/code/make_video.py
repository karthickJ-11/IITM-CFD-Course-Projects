"""Animation for Project 5 (SMAC lid-driven cavity, Re = 100, 99 x 99 cells): start-up to steady state."""
import numpy as np, os, glob, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams.update({"font.family": "serif", "font.size": 10})
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(H, "..", "build", "video", "frames"); V = os.path.join(H, "..", "video")
fs = sorted(glob.glob(os.path.join(B, "frame_*.dat"))); n = int(round(np.sqrt(len(np.loadtxt(fs[0])))))
hist = np.loadtxt(os.path.join(H, "..", "build", "video", "convergence_history.txt"))
fig, ax = plt.subplots(1, 3, figsize=(12.8, 7.2), gridspec_kw={"width_ratios": [1, 1, 0.9]}); fig.subplots_adjust(wspace=0.45, left=0.05, right=0.97); ttl = fig.suptitle("")
def upd(k):
    a = np.loadtxt(fs[k]); x = a[:n, 0]; y = a[::n, 1]; S, W, U, Vv = (a[:, c].reshape(n, n) for c in (2, 3, 4, 5))
    step = int(os.path.basename(fs[k])[6:12]); t = step * 1e-3
    for q in ax[:2]: q.clear()
    lev = np.sort(np.concatenate([-np.array([0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 1e-3, 1e-4]), [1e-6, 1e-5]]))
    ax[0].contourf(x, y, np.hypot(U, Vv), levels=np.linspace(0, 1, 21), cmap="magma")
    ax[0].contour(x, y, S, levels=lev, colors="w", linewidths=0.6, linestyles="solid"); ax[0].set_title("speed and streamlines", fontsize=10)
    ax[1].contourf(x, y, np.clip(W, -6, 6), levels=np.linspace(-6, 6, 25), cmap="RdBu_r", extend="both"); ax[1].set_title("vorticity (clipped to $\\pm$6)", fontsize=10)
    for q in ax[:2]: q.set_aspect("equal"); q.set_xlabel("x"); q.set_ylabel("y")
    ax[2].clear(); h = hist[hist[:, 0] <= max(step, 100)]
    ax[2].semilogy(hist[:, 1], hist[:, 2], color="0.85"); ax[2].semilogy(h[:, 1], h[:, 2], "tab:blue")
    ax[2].axhline(1e-7, color="k", ls="--", lw=0.8); ax[2].set_xlabel("t"); ax[2].set_ylabel(r"max$|u^{n+1}-u^n|$"); ax[2].set_title("convergence", fontsize=10); ax[2].grid(alpha=0.3); ax[2].set_box_aspect(1)
    ttl.set_text(f"SMAC lid-driven cavity, Re = 100, 99 x 99 cells:  step {step},  t = {t:.2f},  $\\psi_{{min}}$ = {S.min():.5f}")
anim = FuncAnimation(fig, upd, frames=len(fs))
anim.save(os.path.join(V, "smac_cavity_startup_to_steady.mp4"), writer=FFMpegWriter(fps=10, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
print("wrote video/smac_cavity_startup_to_steady.mp4,", len(fs), "frames")

"""Animations for Project 7 (1-D transient conduction, explicit FTCS). Output: video/*.mp4"""
import numpy as np, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.grid": True, "grid.alpha": 0.3})
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results"); V = os.path.join(H, "..", "video"); B = os.path.join(H, "..", "build")
L, alpha = 100.0, 1.0
def read(path):
    prof, t, rows = [], None, []
    for line in open(path):
        if line.startswith("Time:"):
            if t is not None: prof.append((t, np.array(rows)))
            t = float(line.split()[1]); rows = []
        elif "\t" in line and not line.startswith("DX"): rows.append(float(line.split("\t")[1]))
    if rows: prof.append((t, np.array(rows)))
    return prof
def exact(x, t, N=600):
    if t == 0: e = np.zeros_like(x); e[0] = 1; return e
    n = np.arange(1, N + 1)[:, None]
    return 1 - x / L - (2 / np.pi) * np.sum(np.sin(n * np.pi * x / L) * np.exp(-(n * np.pi) ** 2 * alpha * t / L ** 2) / n, axis=0)
def save(anim, name, fps):
    anim.save(os.path.join(V, name), writer=FFMpegWriter(fps=fps, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"]), dpi=100)
    print("wrote video/" + name)
# 1. stable run, dt = 0.1 s (gamma = 0.025): every time step
P = read(os.path.join(R, "temperature_dt0.1.txt")); x = np.linspace(0, L, len(P[0][1])); xf = np.linspace(0, L, 801)
fig, ax = plt.subplots(1, 2, figsize=(12.8, 7.2), gridspec_kw={"width_ratios": [2, 1]})
ln_e, = ax[0].plot([], [], "k-", lw=1.5, label="exact (Fourier series)"); ln_n, = ax[0].plot([], [], "o", color="tab:red", ms=5, label="FTCS (C code)")
ax[0].set_xlim(0, 30); ax[0].set_ylim(-0.05, 1.05); ax[0].set_xlabel("x [cm]"); ax[0].set_ylabel("T (non-dimensional)"); ax[0].legend(loc="upper right")
ln_err, = ax[1].plot([], [], "-", color="tab:blue"); ax[1].set_xlim(0, 30); ax[1].set_ylim(-0.03, 0.03); ax[1].set_xlabel("x [cm]"); ax[1].set_ylabel(r"$T_{FTCS}-T_{exact}$")
ttl = fig.suptitle(""); fig.tight_layout(rect=(0, 0, 1, 0.95))
def upd(k):
    t, T = P[k]; ln_e.set_data(xf, exact(xf, t)); ln_n.set_data(x, T); ln_err.set_data(x, T - exact(x, t))
    ttl.set_text(f"1-D transient conduction, explicit FTCS:  $\\Delta t$ = 0.1 s, $\\gamma$ = 0.025,  step {k},  t = {t:.1f} s")
    return ln_e, ln_n, ln_err
save(FuncAnimation(fig, upd, frames=len(P), blit=False), "conduction_FTCS_dt0.1_every_step.mp4", fps=12); plt.close(fig)
# 2. unstable run, dt = 2.5 s (gamma = 0.625)
f = os.path.join(B, "video_unstable.txt")
if os.path.exists(f):
    P = read(f); x = np.linspace(0, L, len(P[0][1]))
    fig, ax = plt.subplots(figsize=(12.8, 7.2)); ln_e, = ax.plot([], [], "k-", lw=1.5, label="exact"); ln_n, = ax.plot([], [], "o-", color="tab:red", ms=5, label=r"FTCS, $\gamma$ = 0.625 > 1/2")
    ax.set_xlim(0, 40); ax.set_xlabel("x [cm]"); ax.set_ylabel("T"); ax.legend(loc="upper right"); ttl = ax.set_title("")
    def upd2(k):
        t, T = P[k]; ln_e.set_data(xf, exact(xf, t)); ln_n.set_data(x, T); m = max(1.2, 1.1 * np.abs(T).max()); ax.set_ylim(-m, m)
        ttl.set_text(f"Instability for $\\gamma$ > 1/2: odd-even mode amplified by |G| = |1-4$\\gamma$| = 1.5 per step  (step {k}, t = {t:.1f} s, max|T| = {np.abs(T).max():.3g})")
        return ln_e, ln_n
    frames = [k for k in range(len(P)) for _ in range(6)]   # hold each step for 6 frames
    save(FuncAnimation(fig, lambda i: upd2(frames[i]), frames=len(frames)), "conduction_FTCS_dt2.5_instability.mp4", fps=6); plt.close(fig)

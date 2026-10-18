"""Project 1 post-processing: GPU finite-volume solver for non-Newtonian (Bird-Carreau) blood flow in a
2-D model of the proximal LAD artery (L = 0.14 m, W = 3.5 mm, mean inflow U = 0.39 m/s).

Inputs (written by code/Blood_flow.cu, see run.sh):
  results/newtonian_20mm/   lambda = 0 (Newtonian, mu = mu0), L = 20 mm   -> Poiseuille verification
  results/carreau_full/     Bird-Carreau, full LAD domain                 -> main results
  results/ablation_point_sor/, results/ablation_block_corr/  short start-up runs with / without the block correction
Reference solutions:
  Newtonian: u = 1.5 U (1 - eta^2), G = -dp/dx = 12 mu U / W^2, tau_w = 6 mu U / W; plus the exact solution of the
  discrete equations (ghost-cell walls): u_j = (G/2mu) y_j (W - y_j) + G h^2 / (8 mu).
  Carreau: fully developed channel flow  mu(gamma) gamma = G s  (s = distance from the centreline), solved for
  gamma(s), integrated for u(s), with G found from the flow-rate constraint (brentq)."""
import numpy as np, os, csv, matplotlib
from scipy.optimize import brentq
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results")
import zipfile  # raw run outputs are shipped zipped (results/raw_runs.zip); extract what is missing
if os.path.exists(os.path.join(R, "raw_runs.zip")):
    with zipfile.ZipFile(os.path.join(R, "raw_runs.zip")) as _z:
        for _m in _z.namelist():
            if not os.path.exists(os.path.join(R, _m)): _z.extract(_m, R)
RHO, W, U, DXY = 1060.0, 0.0035, 0.39, 1e-4
MU0, MUI, LAM, NIDX, A = 0.056, 0.0035, 3.13, 0.6, 2.0

def carreau(g, lam=LAM):
    return MUI + (MU0 - MUI) * (1.0 + (lam * g)**A)**((NIDX - 1.0) / A)

def gamma_of_tau(tau, lam=LAM):
    """invert tau = mu(gamma) gamma (monotone for n > 0) by vectorised bisection in log space"""
    tau = np.atleast_1d(np.asarray(tau, float)); lo = np.full_like(tau, 1e-12); hi = np.full_like(tau, 1e7)
    for _ in range(200):
        mid = np.sqrt(lo * hi); f = carreau(mid, lam) * mid - tau
        lo = np.where(f < 0, mid, lo); hi = np.where(f >= 0, mid, hi)
    g = np.sqrt(lo * hi); g[tau <= 0] = 0.0; return g

def developed(Ubar, lam=LAM, m=4001):
    """fully developed channel flow of a generalised Newtonian fluid with flow rate Ubar*W"""
    b = W / 2; s = np.linspace(0, b, m)
    def prof(G):
        g = gamma_of_tau(G * s, lam)
        u = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(s))]); u = u[-1] - u   # u(s) = int_s^b gamma
        return g, u
    G = brentq(lambda G: np.trapezoid(prof(G)[1], s) / b - Ubar, 1.0, 1e7, xtol=1e-10, rtol=1e-12)
    g, u = prof(G); return dict(G=G, s=s, u=u, gamma=g, mu=carreau(g, lam), tau_w=G * b)

def load_fields(d):
    a = np.loadtxt(os.path.join(d, "final_fields.dat")); x = np.unique(a[:, 0]); y = np.unique(a[:, 1])
    nx, ny = len(x), len(y); F = {k: a[:, c].reshape(ny, nx) for k, c in (("u", 2), ("v", 3), ("p", 4), ("mu", 5), ("g", 6))}
    return x, y, F

def load_hist(d):
    with open(os.path.join(d, "blood_flow_results.csv")) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

def wss(F, lam=LAM):
    """tau_w from the ghost-cell wall gradient 2 u_0 / h (as in Blood_flow.cu); bottom and top wall"""
    gb = 2 * F["u"][0] / DXY; gt = 2 * F["u"][-1] / DXY
    return carreau(gb, lam) * gb, carreau(gt, lam) * gt

summary = []
def note(s):
    print(s); summary.append(s)

# ---------------------------------------------------------------- 1. Newtonian verification
dN = os.path.join(R, "newtonian_20mm")
if os.path.exists(os.path.join(dN, "final_fields.dat")):
    x, y, F = load_fields(dN); nx, ny = len(x), len(y); h = DXY
    i = np.argmin(abs(x - 0.015)); u = F["u"][:, i]
    Qd = u.sum() * h / (U * W)                                   # discrete flow rate / (U W)
    sel = (x > 0.005) & (x < 0.019); G = -np.polyfit(x[sel], F["p"][ny // 2, sel], 1)[0]
    Gex = 12 * MU0 * U / W**2; Gdisc = 12 * MU0 * Qd * U / W**2 / (1 + 2 / ny**2)
    uex = 1.5 * U * (1 - (2 * y / W - 1)**2); g = G / MU0; udisc = g / 2 * y * (W - y) + g * h * h / 8
    tb, tt = wss(F, lam=0.0); tw = 0.5 * (tb[i] + tt[i]); twex = 6 * MU0 * U / W
    note("== Newtonian verification (mu = mu0 = 0.056 Pa s, L = 20 mm, section x = 15 mm)")
    note(f"   G = -dp/dx: computed {G:.2f} Pa/m, analytical {Gex:.2f} (rel {G/Gex-1:+.3e}); exact discrete {Gdisc:.2f} (rel {G/Gdisc-1:+.2e})")
    note(f"   max|u - u_analytical| = {abs(u-uex).max():.3e} m/s ({abs(u-uex).max()/(1.5*U)*100:.3f} % of u_max); "
         f"max|u - u_discrete-exact| = {abs(u-udisc).max():.2e} m/s")
    note(f"   tau_w = {tw:.3f} Pa (analytical 6 mu U / W = {twex:.3f} Pa, rel {tw/twex-1:+.2e});  max|v| = {abs(F['v']).max():.2e} m/s")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(uex, y * 1e3, "k-", lw=1, label="analytical 1.5U(1-$\\eta^2$)"); ax[0].plot(u, y * 1e3, "o", ms=3.5, mfc="none", label="CUDA solver (x = 15 mm)")
    ax[0].set_xlabel("u [m/s]"); ax[0].set_ylabel("y [mm]"); ax[0].legend(fontsize=8); ax[0].set_title("Velocity profile")
    ax[1].plot((u - uex) * 1e3, y * 1e3, "s-", ms=3, label="u - u$_{analytical}$"); ax[1].plot((u - udisc) * 1e6, y * 1e3, "^-", ms=3, label="(u - u$_{discrete}$) x 10$^3$")
    ax[1].set_xlabel("error [mm/s]"); ax[1].set_ylabel("y [mm]"); ax[1].legend(fontsize=8); ax[1].set_title("Profile error")
    ax[2].plot(x * 1e3, F["p"][ny // 2], "-", label="computed (centreline)"); ax[2].plot(x * 1e3, Gex * (x[-1] - x), "k--", lw=1, label="analytical slope 12$\\mu$U/W$^2$")
    ax[2].set_xlabel("x [mm]"); ax[2].set_ylabel("p [Pa]"); ax[2].legend(fontsize=8); ax[2].set_title("Pressure along the axis")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig7_newtonian_verification.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 2. Carreau, full LAD domain
dC = os.path.join(R, "carreau_full")
if os.path.exists(os.path.join(dC, "final_fields.dat")):
    x, y, F = load_fields(dC); nx, ny = len(x), len(y); jc = ny // 2; xm = x * 1e3; ym = y * 1e3
    Qd = F["u"][:, nx // 2].sum() * DXY / W                      # discrete mean velocity
    dev = developed(Qd); newt0 = 12 * MU0 * Qd / W**2; newti = 12 * MUI * Qd / W**2
    sdev = np.abs(y - W / 2); udev = np.interp(sdev, dev["s"], dev["u"]); mudev = np.interp(sdev, dev["s"], dev["mu"]); gdev = np.interp(sdev, dev["s"], dev["gamma"])
    uc = F["u"][jc]; ucd = dev["u"][0]
    ucf = uc[np.argmin(abs(x - 0.12))]                            # computed developed centreline value
    k = np.where(np.abs(uc[x < 0.12] - ucf) > 0.01 * abs(uc[0] - ucf))[0]; Le = x[k[-1] + 1] if len(k) else np.nan
    sel = x > 0.08; G = -np.polyfit(x[sel], F["p"][jc, sel], 1)[0]
    tb, tt = wss(F); tw = 0.5 * (tb + tt); i120 = np.argmin(abs(x - 0.12)); u120 = F["u"][:, i120]
    note("== Bird-Carreau, LAD domain 140 x 3.5 mm (1400 x 35 cells)")
    note(f"   mean inflow {Qd:.5f} m/s; fully developed (ODE): G = {dev['G']:.1f} Pa/m, u_centre = {ucd:.4f} m/s, "
         f"tau_w = {dev['tau_w']:.3f} Pa, wall shear rate {dev['gamma'][-1]:.1f} 1/s, wall viscosity {dev['mu'][-1]*1e3:.3f} mPa s")
    note(f"   computed (x > 80 mm): G = {G:.1f} Pa/m (rel {G/dev['G']-1:+.2e}); at x = 120 mm: u_centre = {uc[i120]:.4f} m/s, "
         f"max|u - u_ODE| = {abs(u120-udev).max():.2e} m/s ({abs(u120-udev).max()/ucd*100:.2f} %), tau_w = {tw[i120]:.3f} Pa (rel {tw[i120]/dev['tau_w']-1:+.2e})")
    note(f"   Newtonian references at the same flow rate: G(mu0) = {newt0:.0f} Pa/m, G(mu_inf) = {newti:.0f} Pa/m; "
         f"tau_w(mu0) = {6*MU0*Qd/W:.2f} Pa, tau_w(mu_inf) = {6*MUI*Qd/W:.3f} Pa")
    note(f"   pressure drop inlet cell -> outlet cell: {F['p'][jc,0]-F['p'][jc,-1]:.1f} Pa (Poiseuille with mu0 would need {newt0*(x[-1]-x[0]):.0f} Pa; initial guess del_P = 2886.3 Pa)")
    note(f"   centreline development length (1 % of the inlet excess): L_e = {Le*1e3:.1f} mm  = {Le/W:.2f} W")
    # field plots (y exaggerated)
    fig, ax = plt.subplots(4, 1, figsize=(12, 9.5), sharex=True)
    for a, (key, lab, cmap) in zip(ax, (("u", "u [m/s]", "viridis"), ("g", r"$\dot\gamma$ [1/s]", "magma"),
                                         ("mu", r"$\mu$ [mPa s]", "cividis"), ("p", "p [Pa]", "coolwarm"))):
        Z = F[key] * (1e3 if key == "mu" else 1)
        if key == "g": Z = np.log10(np.maximum(Z, 1e-1)); lab = r"log$_{10}\dot\gamma$ [1/s]"
        c = a.contourf(xm, ym, Z, levels=30, cmap=cmap); fig.colorbar(c, ax=a, pad=0.01); a.set_ylabel("y [mm]"); a.set_title(lab, fontsize=9, loc="left")
    ax[-1].set_xlabel("x [mm]  (y axis stretched 20x)"); fig.suptitle("Bird-Carreau blood flow in the LAD model: steady fields", y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig1_fields.png"), dpi=180); plt.close(fig)
    # entrance region close-up with vectors
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True); m = x < 0.02
    c = ax[0].contourf(xm[m], ym, F["u"][:, m], levels=30, cmap="viridis"); fig.colorbar(c, ax=ax[0], pad=0.01, label="u [m/s]")
    ax[0].quiver(xm[m][::12], ym[::3], F["u"][::3, m][:, ::12], F["v"][::3, m][:, ::12] * 20, scale=12, width=0.0015, color="w")
    ax[0].set_ylabel("y [mm]"); ax[0].set_title("Entrance region: u and (u, 20v) vectors", fontsize=9, loc="left")
    c = ax[1].contourf(xm[m], ym, F["v"][:, m] * 1e3, levels=30, cmap="RdBu_r"); fig.colorbar(c, ax=ax[1], pad=0.01, label="v [mm/s]")
    ax[1].set_ylabel("y [mm]"); ax[1].set_xlabel("x [mm]"); ax[1].set_title("Transverse velocity: the core flattens, fluid moves towards the walls", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig2_entrance_region.png"), dpi=180); plt.close(fig)
    # profiles
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for xs in (0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.12):
        i = np.argmin(abs(x - xs)); ax[0].plot(F["u"][:, i], ym, "-", lw=1.2, label=f"x = {x[i]*1e3:.1f} mm")
    ax[0].plot(1.5 * Qd * (1 - (2 * y / W - 1)**2), ym, "k:", lw=1.2, label="inlet / Newtonian Poiseuille")
    ax[0].plot(udev, ym, "ko", ms=3, mfc="none", label="Carreau, fully developed (ODE)")
    ax[0].set_xlabel("u [m/s]"); ax[0].set_ylabel("y [mm]"); ax[0].legend(fontsize=7.5); ax[0].set_title("Axial velocity profiles")
    ax[1].plot((u120 - udev) * 1e3, ym, "s-", ms=3); ax[1].set_xlabel(r"u$_{num}$ - u$_{ODE}$ [mm/s]  at x = 120 mm"); ax[1].set_ylabel("y [mm]")
    ax[1].set_title("Deviation from the fully developed Carreau solution")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig3_profiles.png"), dpi=200); plt.close(fig)
    # development, pressure, WSS
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    ax[0].plot(xm, uc, label="computed"); ax[0].axhline(ucd, color="k", ls="--", lw=0.8, label="fully developed (ODE)")
    ax[0].axhline(1.5 * Qd, color="gray", ls=":", lw=0.8, label="inlet 1.5U"); ax[0].axvline(Le * 1e3, color="r", lw=0.8, label=f"L$_e$ = {Le*1e3:.1f} mm")
    ax[0].set_xscale("log"); ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("centreline u [m/s]"); ax[0].legend(fontsize=7.5); ax[0].set_title("Flow development")
    ax[1].plot(xm, F["p"][jc], label="computed (centreline)"); ax[1].plot(xm, dev["G"] * (x[-1] - x), "k--", lw=0.8, label="developed Carreau gradient")
    ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel("p [Pa]"); ax[1].legend(fontsize=8); ax[1].set_title("Axial pressure")
    ax[2].plot(xm, tw, label="computed (mean of both walls)"); ax[2].axhline(dev["tau_w"], color="k", ls="--", lw=0.8, label="Carreau developed")
    ax[2].axhline(6 * MUI * Qd / W, color="tab:green", ls=":", label=r"Newtonian $\mu_\infty$")
    ax[2].set_xscale("log"); ax[2].set_xlabel("x [mm]"); ax[2].set_ylabel(r"$\tau_w$ [Pa]"); ax[2].legend(fontsize=8)
    ax[2].set_title(rf"Wall shear stress (Newtonian $\mu_0$: {6*MU0*Qd/W:.1f} Pa)")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig4_development_pressure_wss.png"), dpi=200); plt.close(fig)
    # viscosity and shear rate across the section
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    ax[0].plot(F["g"][:, i120], ym, "o", ms=3, mfc="none", label="computed (x = 120 mm)"); ax[0].plot(gdev, ym, "k-", lw=1, label="ODE")
    ax[0].set_xlabel(r"$\dot\gamma$ [1/s]"); ax[0].set_ylabel("y [mm]"); ax[0].legend(fontsize=8); ax[0].set_title("Shear rate")
    ax[1].plot(F["mu"][:, i120] * 1e3, ym, "o", ms=3, mfc="none", label="computed"); ax[1].plot(mudev * 1e3, ym, "k-", lw=1, label="ODE")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\mu$ [mPa s]"); ax[1].set_ylabel("y [mm]"); ax[1].legend(fontsize=8); ax[1].set_title("Apparent viscosity")
    gg = np.logspace(-2, 4, 300); ax[2].loglog(gg, carreau(gg) * 1e3, "k-", lw=1, label="Bird-Carreau law")
    ax[2].loglog(np.maximum(F["g"].ravel()[::7], 1e-2), F["mu"].ravel()[::7] * 1e3, ".", ms=2, alpha=0.4, label="all cells")
    ax[2].axhline(MU0 * 1e3, color="gray", ls=":", lw=0.8); ax[2].axhline(MUI * 1e3, color="gray", ls=":", lw=0.8)
    ax[2].set_xlabel(r"$\dot\gamma$ [1/s]"); ax[2].set_ylabel(r"$\mu$ [mPa s]"); ax[2].legend(fontsize=8); ax[2].set_title("Rheology sampled by the flow")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig5_shear_viscosity.png"), dpi=200); plt.close(fig)
    # convergence history
    Hh = load_hist(dC)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].semilogy(Hh["Time"], Hh["RMS_dudt"]); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel(r"RMS $\partial u/\partial t$ [m/s$^2$]"); ax[0].set_title("Approach to steady state")
    ax[1].plot(Hh["Time"], Hh["PoissonSweeps"], ".", ms=2); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("SOR sweeps per step"); ax[1].set_title("Pressure-correction work")
    ax2 = ax[1].twinx(); ax2.plot(Hh["Time"], Hh["dt"] * 1e6, "r-", lw=0.8); ax2.set_ylabel(r"$\Delta t$ [$\mu$s]", color="r"); ax2.grid(False)
    ax[2].semilogy(Hh["Time"], Hh["MaxDivRel"], label=r"max$|\nabla\cdot u|\,\Delta x/U$"); ax[2].semilogy(Hh["Time"], np.abs(Hh["Qout_over_Qin"] - 1) + 1e-9, label=r"$|Q_{out}/Q_{in}-1|$")
    ax[2].set_xlabel("t [s]"); ax[2].legend(fontsize=8); ax[2].set_title("Mass conservation")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig6_convergence.png"), dpi=200); plt.close(fig)
    note(f"   run: {int(Hh['TimeStep'][-1])} time steps to t = {Hh['Time'][-1]:.3f} s; dt {Hh['dt'].min():.3e}-{Hh['dt'].max():.3e} s; "
         f"SOR sweeps/step median {np.median(Hh['PoissonSweeps']):.0f}, max {Hh['PoissonSweeps'].max():.0f}; final RMS du/dt {Hh['RMS_dudt'][-1]:.2e} m/s^2; "
         f"max|div|dx/U <= {Hh['MaxDivRel'].max():.1e}")
    np.savetxt(os.path.join(R, "carreau_developed_ODE.csv"), np.c_[dev["s"], dev["u"], dev["gamma"], dev["mu"]], delimiter=",",
               header=f"s_from_centre_m,u_m_s,gamma_1_s,mu_Pa_s  (G = {dev['G']:.4f} Pa/m, Ubar = {Qd:.6f} m/s)", comments="")

# ---------------------------------------------------------------- 3. block-correction ablation
dA, dB = os.path.join(R, "ablation_point_sor"), os.path.join(R, "ablation_block_corr")
if all(os.path.exists(os.path.join(d, "final_fields.dat")) for d in (dA, dB)):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for d, lab in ((dA, "point SOR only (as original)"), (dB, "SOR + block correction")):
        x, y, F = load_fields(d); Hh = load_hist(d)
        # flow rate through every internal u-face column, from the cell-centred field: Q_i ~ sum_j u_c
        Q = F["u"].sum(0) * DXY / (1.5 * U * W * np.mean(1 - (2 * y / W - 1)**2))
        ax[0].plot(x * 1e3, Q, label=lab); ax[1].semilogy(Hh["Time"] * 1e3, Hh["MaxDivRel"], label=lab)
        note(f"== ablation {os.path.basename(d)}: t = {Hh['Time'][-1]*1e3:.1f} ms, Q(x)/Q_in in [{Q.min():.4f}, {Q.max():.4f}], "
             f"final max|div|dx/U = {Hh['MaxDivRel'][-1]:.2e}, sweeps/step {Hh['PoissonSweeps'][-1]:.0f}")
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel(r"$Q(x)/Q_{in}$"); ax[0].legend(fontsize=8); ax[0].set_title("Cross-sectional flow rate after the start-up")
    ax[1].set_xlabel("t [ms]"); ax[1].set_ylabel(r"max$|\nabla\cdot u|\,\Delta x/U$"); ax[1].legend(fontsize=8); ax[1].set_title("Residual divergence after projection")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig8_block_correction.png"), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 4. GPU vs CPU-emulation (when run.sh found a GPU)
for cpu_d, gpu_d in (("newtonian_20mm", "gpu_newtonian_20mm"), ("carreau_full", "gpu_carreau_full")):
    a, b = os.path.join(R, cpu_d), os.path.join(R, gpu_d)
    if all(os.path.exists(os.path.join(d, "final_fields.dat")) for d in (a, b)):
        _, _, Fa = load_fields(a); _, _, Fb = load_fields(b)
        note(f"== GPU vs CPU emulation ({cpu_d}): max|du| = {abs(Fa['u']-Fb['u']).max():.2e} m/s, max|dp| = {abs(Fa['p']-Fb['p']).max():.2e} Pa")
tf = os.path.join(R, "timing.txt")
if os.path.exists(tf): note("== timing\n" + open(tf).read().strip())

open(os.path.join(R, "summary.txt"), "w").write("\n".join(summary) + "\n")

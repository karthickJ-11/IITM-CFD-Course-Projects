"""Accuracy studies for Project 7 (run by run.sh): spatial convergence at fixed gamma and the effect of gamma.
Modified equation of FTCS:  T_t = alpha T_xx + alpha dx^2/12 (1 - 6 gamma) T_xxxx + O(dx^4)
-> second order in dx; the decay-rate error of each Fourier mode is proportional to (1 - 6 gamma)."""
import numpy as np, glob, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.grid":True,"grid.alpha":0.3})
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results"); S=os.path.join(R,"study"); L=100.0
def last(f):
    lines=open(f).read().strip().splitlines(); t=float(lines[0].split()[1]); T=np.array([float(l.split("\t")[1]) for l in lines[1:]])
    return t,T
def exact(x,t,N=2000):
    n=np.arange(1,N+1)[:,None]; return 1-x/L-(2/np.pi)*np.sum(np.sin(n*np.pi*x/L)*np.exp(-(n*np.pi)**2*t/L**2)/n,axis=0)
out=["(a) grid refinement, gamma = 3/8, t = 48 s","CX | dx [cm] | max|T-T_exact| | observed order"]
dxs,errs=[],[]
for cx in (26,51,101,201):
    t,T=last(os.path.join(S,f"final_grid_CX{cx}.txt")); x=np.linspace(0,L,cx); e=np.abs(T-exact(x,t)).max(); dxs.append(L/(cx-1)); errs.append(e)
for k,(d,e) in enumerate(zip(dxs,errs)):
    p="" if k==0 else f"{np.log(errs[k-1]/e)/np.log(dxs[k-1]/d):.2f}"; out.append(f"{len(np.arange(0,L+1e-9,d))} | {d:g} | {e:.3e} | {p}")
out+=["","(b) gamma sweep, dx = 2 cm, t = 42 s","dt [s] | gamma | max|T-T_exact| | leading-term factor |1-6 gamma|"]
gs,ge=[],[]
for f in sorted(glob.glob(os.path.join(S,"final_gamma_dt*.txt")),key=lambda s:float(s.split("dt")[-1][:-4])):
    dt=float(f.split("dt")[-1][:-4]); t,T=last(f); x=np.linspace(0,L,51); e=np.abs(T-exact(x,t)).max(); g=dt/4
    gs.append(g); ge.append(e); out.append(f"{dt:.4g} | {g:.4f} | {e:.3e} | {abs(1-6*g):.3f}")
open(os.path.join(R,"accuracy_study.txt"),"w").write("\n".join(out)+"\n"); print("\n".join(out))
fig,ax=plt.subplots(1,3,figsize=(14,3.9))
d=np.array(dxs); ax[0].loglog(d,errs,"o-",label=r"FTCS, $\gamma=3/8$"); ax[0].loglog(d,errs[-1]*(d/d[-1])**2,"k--",lw=0.8,label="slope 2")
ax[0].set_xlabel(r"$\Delta x$ [cm]"); ax[0].set_ylabel(r"max$|T-T_{exact}|$ at t = 48 s"); ax[0].legend(fontsize=8); ax[0].set_title("Grid refinement")
gg=np.linspace(0.01,0.5,200); ax[1].semilogy(gs,ge,"s-",label="FTCS, dx = 2 cm")
ax[1].axvline(1/6,color="r",lw=0.8,ls=":",label=r"$\gamma=1/6$"); ax[1].set_xlabel(r"$\gamma=\alpha\Delta t/\Delta x^2$"); ax[1].set_ylabel(r"max$|T-T_{exact}|$ at t = 42 s")
ax[1].legend(fontsize=8); ax[1].set_title(r"Error vs $\gamma$ at $\Delta x$ = 2 cm")
th=np.linspace(0,np.pi,300)
for g in (0.1,1/6,0.25,0.5,0.6):
    ax[2].plot(th,1-4*g*np.sin(th/2)**2,label=rf"$\gamma$={g:.3g}")
ax[2].plot(th,np.exp(-0.25*th**2),"k--",lw=0.8,label=r"exact $e^{-\gamma\theta^2}$, $\gamma$=0.25")
ax[2].axhline(-1,color="gray",lw=0.8); ax[2].set_xlabel(r"$\theta=k\Delta x$"); ax[2].set_ylabel(r"$G=1-4\gamma\sin^2(\theta/2)$")
ax[2].legend(fontsize=7); ax[2].set_title(r"Amplification factor (|G|>1 for $\gamma$>1/2)")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig6_accuracy_study.png"),dpi=200)

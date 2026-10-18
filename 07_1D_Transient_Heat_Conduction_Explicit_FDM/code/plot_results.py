"""Post-processing for Project 7 (1-D transient conduction, explicit FTCS).
Reads results/temperature_dt*.txt written by assignment_1.c and produces the report figures.
Analytical reference: finite slab, T(0)=1, T(L)=0, T(x,0)=0
  T = 1 - x/L - (2/pi) * sum_n (1/n) sin(n pi x/L) exp(-n^2 pi^2 alpha t / L^2)
"""
import numpy as np, matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.grid":True,"grid.alpha":0.3})
here=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(here,"..","results")
L,CX,alpha=100.0,51,1.0; x=np.linspace(0,L,CX)
def read(dt):
    prof={}; t=None; rows=[]
    for line in open(os.path.join(R,f"temperature_dt{dt}.txt")):
        if line.startswith("Time:"):
            if t is not None: prof[round(t,6)]=np.array(rows)
            t=float(line.split()[1]); rows=[]
        elif "\t" in line and not line.startswith("DX"):
            rows.append(float(line.split("\t")[1]))
    if t is not None and rows: prof[round(t,6)]=np.array(rows)
    return prof
def exact(t,N=400):
    if t==0: e=np.zeros_like(x); e[0]=1; return e
    n=np.arange(1,N+1)[:,None]
    return 1-x/L-(2/np.pi)*np.sum(np.sin(n*np.pi*x/L)*np.exp(-(n*np.pi)**2*alpha*t/L**2)/n,axis=0)
times=[0.0,0.1,0.5,1.0,2.0,10.0]; cols=plt.cm.viridis(np.linspace(0,0.9,len(times)))
data={dt:read(dt) for dt in ["0.1","0.01","0.001"]}
# Fig 1: profiles for each dt with analytical overlay
fig,ax=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
for a,(dt,pr) in zip(ax,data.items()):
    for t,c in zip(times,cols):
        k=min(pr,key=lambda s:abs(s-t)); a.plot(x,pr[k],"o",ms=3,color=c,label=f"FDM t={t:g} s")
        a.plot(x,exact(t),"-",lw=1,color=c)
    g=alpha*float(dt)/(L/(CX-1))**2
    a.set_title(rf"$\Delta t$={dt} s,  $\gamma$={g:.2e}"); a.set_xlabel("x (cm)"); a.set_xlim(0,30)
ax[0].set_ylabel("T (non-dimensional)"); ax[0].legend(fontsize=7,ncol=2,title="markers: FDM, lines: exact",title_fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R,"fig1_profiles_vs_exact.png"),dpi=200)
# Fig 2: error vs dt at t=10
fig,ax=plt.subplots(figsize=(5,3.6)); errs=[]
for dt,pr in data.items():
    e=np.max(np.abs(pr[10.0]-exact(10.0))); errs.append(e)
ax.loglog([0.1,0.01,0.001],errs,"s-",label=r"$\|T_{FDM}-T_{exact}\|_\infty$ at t=10 s")
ax.set_xlabel(r"$\Delta t$ (s)"); ax.set_ylabel("max abs error"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(R,"fig2_error_vs_dt.png"),dpi=200)
# Fig 3: space-time map (dt=0.001)
pr=data["0.001"]; ts=sorted(pr); ts=ts[::max(1,len(ts)//400)]
Z=np.array([pr[t] for t in ts]); fig,ax=plt.subplots(figsize=(6,3.6))
m=ax.pcolormesh(x,ts,Z,shading="auto",cmap="inferno"); fig.colorbar(m,label="T")
ax.set_xlim(0,30); ax.set_xlabel("x (cm)"); ax.set_ylabel("t (s)"); ax.set_title(r"Temperature evolution, $\Delta t$=0.001 s")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig3_space_time.png"),dpi=200)
# Fig 4: old (in-place MATLAB sweep) vs corrected explicit scheme, t=0.1 and 10 s, dt=0.1
def sweep(dt,tend,inplace):
    g=alpha*dt/(L/(CX-1))**2; T=np.zeros(CX); out={}
    for j in range(0,int(round(tend/dt))+1):
        T[0]=1; T[-1]=0; Told=T.copy()
        for i in range(1,CX-1):
            src=T if inplace else Told
            T[i]=g*src[i-1]+(1-2*g)*Told[i]+g*src[i+1]
        out[round((j+1)*dt if inplace else (j+1)*dt,6)]=T.copy()
    return out
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
for a,t in zip(ax,[0.1,10.0]):
    old=sweep(0.1,t,True); new=data["0.1"]
    # original MATLAB stored the state after (j+1) updates under the label j*dt
    a.plot(x[1:-1],old[round(t+0.1,6)][1:-1],"o",mfc="none",label="original MATLAB (in-place, label j*dt)")
    a.plot(x,new[t],"s",ms=3,label="corrected C/MATLAB (explicit FTCS)")
    a.plot(x,exact(t),"k-",lw=1,label="analytical")
    a.set_xlim(0,30 if t>1 else 10); a.set_title(rf"$\Delta t$=0.1 s, t={t:g} s"); a.set_xlabel("x (cm)")
ax[0].set_ylabel("T"); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R,"fig4_old_vs_corrected.png"),dpi=200)
# Fig 5: stability check dt=2.5 (gamma=0.625>0.5)
f=os.path.join(R,"temperature_dt2.5.txt")
if os.path.exists(f):
    pr=read("2.5"); fig,ax=plt.subplots(figsize=(6,3.6))
    for t in [2.5,5.0,7.5,10.0]:
        k=min(pr,key=lambda s:abs(s-t)); ax.plot(x,pr[k],"o-",ms=3,label=f"t={k:g} s")
    ax.set_xlim(0,30); ax.set_xlabel("x (cm)"); ax.set_ylabel("T"); ax.set_title(r"$\Delta t$=2.5 s, $\gamma$=0.625 > 0.5 (unstable)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(R,"fig5_stability_gamma_gt_half.png"),dpi=200)
with open(os.path.join(R,"summary.txt"),"w") as fh:
    for dt,e in zip(["0.1","0.01","0.001"],errs): fh.write(f"dt={dt}  gamma={alpha*float(dt)/4:.2e}  max|err| at t=10 s = {e:.3e}\n")
    fh.write(f"T(x=2 cm, t=10 s): FDM dt=0.001 -> {data['0.001'][10.0][1]:.4f}, exact -> {exact(10.0)[1]:.4f}\n")
print(open(os.path.join(R,"summary.txt")).read())

"""Project 3 post-processing: 1-D advection-diffusion of an instantaneous point release.
Scheme in advection_diffusion.c: forward Euler in time, first-order upwind (FTBS) advection,
central diffusion. Exact solution (infinite domain, release of mass M = Cd*dx at x0 = 0):
    C(x,t) = M / sqrt(4 pi D t) * exp(-(x - u t)^2 / (4 D t))
Modified-equation analysis of the scheme gives an extra (numerical) diffusion
    D_num = (u dx / 2) * (1 - CFL),   CFL = u dt / dx,
so the computed plume spreads like the exact solution with D_eff = D + D_num."""
import numpy as np, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.grid":True,"grid.alpha":0.3})
H=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(H,"..","results")
Dm,dx,dt,Cd=0.5,0.2,0.01,10.0; M=Cd*dx
def read(f):
    blocks={}; t=None; rows=[]
    for line in open(f):
        if line.startswith("# Time ="):
            if t is not None: blocks[t]=np.array(rows)
            t=round(float(line.split("=")[1].split()[0]),2); rows=[]
        elif line.strip() and not line.startswith("#") and not line.startswith("Position"):
            rows.append([float(v) for v in line.split()])
    if t is not None and rows: blocks[t]=np.array(rows)
    return blocks
def exact(x,t,u,D):
    return M/np.sqrt(4*np.pi*D*t)*np.exp(-(x-u*t)**2/(4*D*t))
us=[0.0,1.0,1.5,2.0]; data={u:read(os.path.join(R,f"results_u{u}.txt")) for u in us}
# Fig 1: profiles t=1..10 s for u=0,1,2 with exact (D) and exact (D_eff)
fig,ax=plt.subplots(1,3,figsize=(14,4.2),sharey=True)
for a,u in zip(ax,[0.0,1.0,2.0]):
    cfl=u*dt/dx; Dn=u*dx/2*(1-cfl); cols=plt.cm.viridis(np.linspace(0,0.9,10))
    for k,t in enumerate(range(1,11)):
        b=data[u][float(t)]; x=b[:,0]
        a.plot(x,b[:,1],"-",color=cols[k],lw=1.2,label=f"t={t} s" if u==0 else None)
        if t in (1,4,8): a.plot(x,exact(x,t,u,Dm),":",color="k",lw=0.9); a.plot(x,exact(x,t,u,Dm+Dn),"--",color="r",lw=0.8)
    a.set_title(rf"u={u:g} m/s, CFL={cfl:.2f}, $D_{{num}}$={Dn:.3f} m$^2$/s"); a.set_xlabel("x (m)")
ax[0].set_ylabel(r"C (kg/m$^3$)"); ax[0].legend(fontsize=7,ncol=2)
ax[2].plot([],[],"k:",label=r"exact, $D$"); ax[2].plot([],[],"r--",label=r"exact, $D+D_{num}$"); ax[2].legend(fontsize=8,loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig1_profiles_vs_exact.png"),dpi=200)
# moments
mom={}
for u in us:
    ts=sorted(k for k in data[u] if k>0); m0=[];m1=[];m2=[]
    for t in ts:
        x,c=data[u][t][:,0],data[u][t][:,1]; mass=np.sum(c)*dx; xc=np.sum(x*c)*dx/mass if mass>0 else np.nan
        var=np.sum((x-xc)**2*c)*dx/mass if mass>0 else np.nan; m0.append(mass); m1.append(xc); m2.append(var)
    mom[u]=(np.array(ts),np.array(m0),np.array(m1),np.array(m2))
fig,ax=plt.subplots(1,3,figsize=(14,3.8))
rows=[]
for u in us:
    t,m0,m1,m2=mom[u]; cfl=u*dt/dx; Dn=u*dx/2*(1-cfl)
    ax[0].plot(t,m0/M,label=f"u={u:g}"); ax[1].plot(t,m1,label=f"u={u:g}")
    De=Dm+Dn; ok=(t>=1)&(u*t+3*np.sqrt(2*De*t)<=9.5)&(3*np.sqrt(2*De*t)<=9.5)   # plume (+-3 sigma) inside the domain
    p=np.polyfit(t[ok],m2[ok],1); Dfit=p[0]/2
    ax[2].plot(t[ok],m2[ok],label=rf"u={u:g}: $\sigma^2$ slope/2 = {Dfit:.3f}")
    rows.append((u,cfl,Dm*dt/dx**2,Dn,Dm+Dn,Dfit,np.polyfit(t[ok],m1[ok],1)[0],int(ok.sum())))
ax[0].set_xlabel("t (s)"); ax[0].set_ylabel(r"mass / $M_0$"); ax[0].set_title("Mass in domain"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("t (s)"); ax[1].set_ylabel(r"centroid $\bar x$ (m)"); ax[1].set_title("Plume centroid"); ax[1].set_xlim(0,12); ax[1].set_ylim(-1,11); ax[1].legend(fontsize=8)
ax[2].set_xlabel("t (s)"); ax[2].set_ylabel(r"variance $\sigma^2$ (m$^2$)"); ax[2].set_title(r"Spreading: $\sigma^2=2D_{eff}t$"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R,"fig2_moments.png"),dpi=200)
# Fig 3: space-time maps
fig,ax=plt.subplots(1,3,figsize=(14,3.8))
for a,u in zip(ax,[0.0,1.0,2.0]):
    ts=sorted(k for k in data[u] if k<=20); Z=np.array([data[u][t][:,1] for t in ts]); x=data[u][ts[0]][:,0]
    m=a.pcolormesh(x,ts,np.log10(np.maximum(Z,1e-4)),shading="auto",cmap="magma",vmin=-3,vmax=1); a.set_title(f"u={u:g} m/s"); a.set_xlabel("x (m)")
    if u>0: a.plot(u*np.array(ts),ts,"c--",lw=0.8,label=r"$x=ut$"); a.legend(fontsize=8,loc="lower right")
ax[0].set_ylabel("t (s)"); fig.colorbar(m,ax=ax,label=r"log$_{10}$ C",shrink=0.9)
fig.savefig(os.path.join(R,"fig3_space_time.png"),dpi=200,bbox_inches="tight")
# Fig 4: original vs corrected (u=1.5, same code path) + original label offset
orig=os.path.join(H,"..","original","results_original_u1.5.txt")
if os.path.exists(orig):
    o=read(orig); n=data[1.5]; fig,ax=plt.subplots(figsize=(6.4,3.8)); dmax=0
    for t in (1.0,3.0,6.0,9.0):
        ax.plot(o[t][:,0],o[t][:,1],"o",ms=2.5,mfc="none",label=f"original output t={t:g}"); ax.plot(n[t][:,0],n[t][:,1],"-",lw=1,label=f"corrected t={t:g}")
        dmax=max(dmax,np.max(np.abs(o[t][:,1]-n[t][:,1])))
    ax.set_xlabel("x (m)"); ax.set_ylabel("C"); ax.set_title(f"u=1.5 m/s: original vs corrected (max diff {dmax:.1e})"); ax.legend(fontsize=6,ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(R,"fig4_original_vs_corrected.png"),dpi=200)
with open(os.path.join(R,"summary.txt"),"w") as fh:
    fh.write("u | CFL | diffusion number | D_num theory | D+D_num | D_eff from variance | centroid speed | n points\n")
    for r in rows: fh.write(" | ".join(f"{v:.4f}" for v in r)+"\n")
    for u in us:
        t,m0,m1,m2=mom[u]; fh.write(f"u={u}: mass/M0 at t=1,5,10,20 s = {[round(float(m0[np.argmin(abs(t-k))]/M),4) for k in (1,5,10,20)]}\n")
    for u in (0.0,1.0,2.0):
        b=data[u][4.0]; x=b[:,0]; cfl=u*dt/dx; Dn=u*dx/2*(1-cfl)
        fh.write(f"u={u}: t=4 s peak FDM={b[:,1].max():.4f}, exact(D)={exact(x,4,u,Dm).max():.4f}, exact(D_eff)={exact(x,4,u,Dm+Dn).max():.4f}; L2 err vs exact(D)={np.sqrt(np.mean((b[:,1]-exact(x,4,u,Dm))**2)):.2e}, vs exact(D_eff)={np.sqrt(np.mean((b[:,1]-exact(x,4,u,Dm+Dn))**2)):.2e}\n")
print(open(os.path.join(R,"summary.txt")).read())

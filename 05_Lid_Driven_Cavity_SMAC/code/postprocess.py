"""Project 5 post-processing: SMAC lid-driven cavity, Re = 100.
Reads results/grid_100 (99x99 cells) and results/grid_50 (49x49 cells) written by smac_cavity.c.
Benchmark: Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387-411, Tables I and II (Re = 100)."""
import numpy as np, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.grid":True,"grid.alpha":0.3})
H=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(H,"..","results")
# Ghia et al. (1982) Re=100
gy=np.array([1.0,.9766,.9688,.9609,.9531,.8516,.7344,.6172,.5,.4531,.2813,.1719,.1016,.0703,.0625,.0547,0.0])
gu=np.array([1.0,.84123,.78871,.73722,.68717,.23151,.00332,-.13641,-.20581,-.21090,-.15662,-.10150,-.06434,-.04775,-.04192,-.03717,0.0])
gx=np.array([1.0,.9688,.9609,.9531,.9453,.9063,.8594,.8047,.5,.2344,.2266,.1563,.0938,.0781,.0703,.0625,0.0])
gv=np.array([0.0,-.05906,-.07391,-.08864,-.10313,-.16914,-.22445,-.24533,.05454,.17527,.17507,.16077,.12317,.10890,.10091,.09233,0.0])
G_PSI,G_XC,G_YC=-0.103423,0.6172,0.7344
def load(d,name):
    a=np.loadtxt(os.path.join(R,d,name)); x=np.unique(a[:,0]); y=np.unique(a[:,1])
    return x,y,a[:,2].reshape(len(y),len(x))   # rows = y, cols = x
def centre(xs,F,axis):  # value on x=0.5 (axis=1) or y=0.5 (axis=0) by linear interpolation
    i=np.searchsorted(xs,0.5); w=(0.5-xs[i-1])/(xs[i]-xs[i-1])
    return (1-w)*F[:,i-1]+w*F[:,i] if axis==1 else (1-w)*F[i-1,:]+w*F[i,:]
res={}
for d in ["grid_100","grid_50"]:
    if not os.path.exists(os.path.join(R,d,"u_values.txt")): continue
    x,y,U=load(d,"u_values.txt"); _,_,V=load(d,"v_values.txt"); _,_,S=load(d,"streamlines.txt")
    _,_,W=load(d,"vorticity.txt"); _,_,Pp=load(d,"pressure.txt")
    k=np.unravel_index(np.argmin(S),S.shape)
    res[d]=dict(x=x,y=y,U=U,V=V,S=S,W=W,P=Pp,uc=centre(x,U,1),vc=centre(y,V,0),psimin=S[k],xc=x[k[1]],yc=y[k[0]])
f=res["grid_100"]; x,y=f["x"],f["y"]
# Fig 1: streamlines, vorticity, pressure
fig,ax=plt.subplots(1,3,figsize=(13,4.3))
lev=np.concatenate([np.array([-0.1,-0.09,-0.07,-0.05,-0.03,-0.01,-1e-3,-1e-4,-1e-5]),np.array([1e-6,1e-5,5e-5,1e-4,2.5e-4,5e-4])])
ax[0].contour(x,y,f["S"],levels=np.sort(lev),colors="k",linewidths=0.7); ax[0].plot(f["xc"],f["yc"],"r+",ms=10)
ax[0].set_title(rf"Stream function, $\psi_{{min}}$={f['psimin']:.4f}")
c=ax[1].contourf(x,y,np.clip(f["W"],-5,5),levels=np.linspace(-5,5,21),cmap="RdBu_r",extend="both"); fig.colorbar(c,ax=ax[1],shrink=0.8)
ax[1].contour(x,y,f["W"],levels=[-3,-2,-1,-0.5,0,0.5,1,2,3],colors="k",linewidths=0.4); ax[1].set_title(r"Vorticity $\omega=\partial v/\partial x-\partial u/\partial y$")
Pc=f["P"]-f["P"][len(y)//2,len(x)//2]; lo,hi=np.percentile(Pc,[2,98])   # lid corners are singular: clip
c=ax[2].contourf(x,y,np.clip(Pc,lo,hi),levels=np.linspace(lo,hi,21),cmap="viridis",extend="both"); fig.colorbar(c,ax=ax[2],shrink=0.8)
ax[2].contour(x,y,np.clip(Pc,lo,hi),levels=np.linspace(lo,hi,21),colors="k",linewidths=0.3); ax[2].set_title(r"Pressure $p-p_{centre}$ (2-98 percentile)")
for a in ax: a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig1_streamlines_vorticity_pressure.png"),dpi=200)
# Fig 2: centreline validation
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
for d,st in [("grid_50","--"),("grid_100","-")]:
    if d in res:
        n=len(res[d]["x"])-1
        ax[0].plot(res[d]["uc"],res[d]["y"],st,label=f"SMAC {n}x{n}"); ax[1].plot(res[d]["x"],res[d]["vc"],st,label=f"SMAC {n}x{n}")
ax[0].plot(gu,gy,"ko",mfc="none",label="Ghia et al. (1982)"); ax[1].plot(gx,gv,"ko",mfc="none",label="Ghia et al. (1982)")
ax[0].set_xlabel("u(0.5, y)"); ax[0].set_ylabel("y"); ax[0].set_title("u on vertical centreline"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("x"); ax[1].set_ylabel("v(x, 0.5)"); ax[1].set_title("v on horizontal centreline"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(R,"fig2_centreline_vs_ghia.png"),dpi=200)
# Fig 3: convergence history
h=np.loadtxt(os.path.join(R,"grid_100","convergence_history.txt"))
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ax[0].semilogy(h[:,1],h[:,2]); ax[0].axhline(1e-7,color="k",ls="--",lw=0.8); ax[0].set_xlabel("t"); ax[0].set_ylabel(r"max$|u^{n+1}-u^n|$"); ax[0].set_title("Approach to steady state")
ax[1].plot(h[:,1],h[:,3]); ax[1].set_xlabel("t"); ax[1].set_ylabel("SOR iterations / step"); ax[1].set_title("Pressure-correction cost")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig3_convergence_history.png"),dpi=200)
# Fig 4: velocity vectors + speed
fig,ax=plt.subplots(figsize=(5.2,4.6)); sp=np.hypot(f["U"],f["V"]); s=4
c=ax.contourf(x,y,sp,levels=21,cmap="magma"); fig.colorbar(c,ax=ax,label="|u|")
ax.quiver(x[::s],y[::s],f["U"][::s,::s],f["V"][::s,::s],color="w",scale=12,width=0.003); ax.set_aspect("equal"); ax.set_title("Velocity field"); ax.set_xlabel("x"); ax.set_ylabel("y")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig4_velocity_vectors.png"),dpi=200)
# error table
def err(d):
    r=res[d]; eu=np.interp(gy,r["y"],r["uc"])-gu; ev=np.interp(gx,r["x"],r["vc"])-gv
    return np.max(np.abs(eu)),np.sqrt(np.mean(eu**2)),np.max(np.abs(ev)),np.sqrt(np.mean(ev**2))
with open(os.path.join(R,"summary.txt"),"w") as fh:
    fh.write("grid | psi_min | vortex centre | max|u-u_Ghia| | rms u | max|v-v_Ghia| | rms v | u_min | v_max | v_min\n")
    for d in ["grid_50","grid_100"]:
        if d in res:
            r=res[d]; e=err(d); n=len(r["x"])-1
            fh.write(f"{n}x{n} | {r['psimin']:.5f} | ({r['xc']:.4f},{r['yc']:.4f}) | {e[0]:.4f} | {e[1]:.4f} | {e[2]:.4f} | {e[3]:.4f} | {r['uc'].min():.5f} | {r['vc'].max():.5f} | {r['vc'].min():.5f}\n")
    fh.write(f"Ghia: psi_min={G_PSI}, centre=({G_XC},{G_YC}), u_min=-0.21090, v_max=0.17527, v_min=-0.24533\n")
    fh.write(f"steady state at t={h[-1,1]:.2f} (last logged step {int(h[-1,0])})\n")

# secondary (corner) eddies: Ghia et al. Re=100: BR psi=1.25374e-5 at (0.9453,0.0625); BL psi=1.74877e-6 at (0.0313,0.0391)
with open(os.path.join(R,"summary.txt"),"a") as fh:
    for d in ["grid_50","grid_100"]:
        if d not in res: continue
        r=res[d]; X,Y=np.meshgrid(r["x"],r["y"]); n=len(r["x"])-1
        for nm,m in [("BR",(X>0.7)&(Y<0.3)),("BL",(X<0.3)&(Y<0.3))]:
            Sm=np.where(m,r["S"],-np.inf); k=np.unravel_index(np.argmax(Sm),Sm.shape)
            fh.write(f"{n}x{n} {nm} eddy: psi_max={r['S'][k]:.3e} at ({r['x'][k[1]]:.4f},{r['y'][k[0]]:.4f})\n")
    fh.write("Ghia BR: 1.25374e-5 at (0.9453,0.0625); BL: 1.74877e-6 at (0.0313,0.0391)\n")
print(open(os.path.join(R,"summary.txt")).read())

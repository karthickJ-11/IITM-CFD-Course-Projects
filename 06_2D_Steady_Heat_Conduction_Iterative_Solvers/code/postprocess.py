"""Project 6 post-processing: contours, convergence histories, accuracy against a direct sparse
solve of the same discrete system and against the analytical Fourier series, heat balance,
omega sweep and grid study."""
import numpy as np, os, re, glob, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scipy.sparse as sp, scipy.sparse.linalg as spla
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.grid":True,"grid.alpha":0.3})
H=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(H,"..","results")
K=50.0
def readlog(f):
    it,err=[],[]; Q=None
    for l in open(f):
        m=re.match(r"Iteration (\d+) Max error: ([0-9.eE+-]+)",l)
        if m: it.append(int(m.group(1))); err.append(float(m.group(2)))
        m=re.search(r"Total Heat Transfer along Bottom Boundary: ([-0-9.]+)",l)
        if m: Q=float(m.group(1))
    return np.array(it),np.array(err),Q
def direct(M,N,Tb=100.0,To=30.0):
    T=np.zeros((M,N)); T[0,:]=To; T[:,0]=To; T[:,-1]=To; T[-1,:]=Tb; T[-1,0]=To; T[-1,-1]=To
    n=(M-2)*(N-2); idx=lambda i,j:(i-1)*(N-2)+(j-1)
    A=sp.lil_matrix((n,n)); b=np.zeros(n)
    for i in range(1,M-1):
        for j in range(1,N-1):
            k=idx(i,j); A[k,k]=4
            for ii,jj in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                if 1<=ii<=M-2 and 1<=jj<=N-2: A[k,idx(ii,jj)]=-1
                else: b[k]+=T[ii,jj]
    x=spla.spsolve(A.tocsr(),b); T[1:-1,1:-1]=x.reshape(M-2,N-2); return T
def analytic(M,N,W=1.0,Hh=2.0,nt=399):
    x=np.linspace(0,W,N); y=np.linspace(Hh,0,M)  # row 0 = top (y=H), row M-1 = bottom (y=0)
    X,Y=np.meshgrid(x,y); S=np.zeros_like(X)
    for n in range(1,nt+1,2):
        S+=4/(n*np.pi)*np.sin(n*np.pi*X/W)*np.exp(-n*np.pi*Y/W)*(1-np.exp(-2*n*np.pi*(Hh-Y)/W))/(1-np.exp(-2*n*np.pi*Hh/W))
    return 30+70*S
def Qbottom(T,M,N): dx=1/(N-1); dy=2/(M-1); return K*np.sum((T[-1,1:-1]-T[-2,1:-1])/dy*dx)
def Qout(T,M,N):
    dx=1/(N-1); dy=2/(M-1)
    top=K*np.sum((T[1,1:-1]-T[0,1:-1])/dy*dx); left=K*np.sum((T[1:-1,1]-T[1:-1,0])/dx*dy); right=K*np.sum((T[1:-1,-2]-T[1:-1,-1])/dx*dy)
    return top,left,right
M,N=41,21; Tref=direct(M,N); Tan=analytic(M,N)
names={"point_gauss_seidel":"Point Gauss-Seidel","line_gauss_seidel":"Line Gauss-Seidel","psor":r"PSOR ($\omega$=1.5)",
       "lsor":r"LSOR ($\omega$=1.5)","adi":"ADI","adi_relaxation":r"ADI relaxed ($\omega$=1.5)"}
orig={"point_gauss_seidel":"point-gauss_sidel","line_gauss_seidel":"Line_gauss_sidel","psor":"PSOR","lsor":"Lsor","adi":"ADI_method","adi_relaxation":"ADI_method_with_relaxation"}
rows=[]
fig,ax=plt.subplots(figsize=(6.4,4.2))
for k,lab in names.items():
    it,err,Q=readlog(os.path.join(R,"corrected",f"{k}_log.txt")); T=np.loadtxt(os.path.join(R,"corrected",f"{k}_field.dat"))
    ito,erro,Qo=readlog(os.path.join(R,"original",f"{orig[k]}_log.txt"))
    To=np.loadtxt(os.path.join(R,"original",f"{orig[k]}_field.dat")) if os.path.exists(os.path.join(R,"original",f"{orig[k]}_field.dat")) else None
    e=np.max(np.abs(T-Tref)); rows.append((lab,it[-1],Q,e,ito[-1],Qo)); ax.semilogy(it,err,label=lab)
ax.axhline(0.01,color="k",ls="--",lw=0.8); ax.set_xlabel("iteration"); ax.set_ylabel(r"max $|T^{k+1}-T^{k}|$ (°C)")
ax.set_title("Convergence history (21 x 41)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(R,"fig2_convergence_history.png"),dpi=200)
# contours
T=np.loadtxt(os.path.join(R,"corrected","lsor_field.dat")); x=np.linspace(0,1,N); y=np.linspace(2,0,M)
fig,ax=plt.subplots(1,3,figsize=(11,5.2))
for a,(F,t) in zip(ax,[(T,"LSOR (corrected code)"),(Tan,"Analytical (Fourier series)"),(np.abs(T-Tan),"|LSOR - analytical|")]):
    lv=np.linspace(30,100,15) if "|" not in t else 12
    c=a.contourf(x,y,F,levels=lv,cmap="inferno" if "|" not in t else "viridis"); a.set_aspect("equal"); a.set_title(t,fontsize=10)
    if "|" not in t: a.contour(x,y,F,levels=lv,colors="w",linewidths=0.4)
    fig.colorbar(c,ax=a,shrink=0.85,label="T (°C)"); a.set_xlabel("x"); a.set_ylabel("y")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig1_temperature_contours.png"),dpi=200)
# old vs corrected ADI-relaxation field and centreline
Toadi=np.loadtxt(os.path.join(R,"original","ADI_method_with_relaxation_field.dat")) if os.path.exists(os.path.join(R,"original","ADI_method_with_relaxation_field.dat")) else None
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
jc=N//2; ax[0].plot(y,Tref[:,jc],"k-",label="direct solve (reference)"); ax[0].plot(y,T[:,jc],"o",ms=3,label="LSOR corrected")
if Toadi is not None: ax[0].plot(y,Toadi[:,jc],"x",ms=4,label="original 'ADI with relaxation'")
ax[0].set_xlabel("y (bottom = 0)"); ax[0].set_ylabel("T at x = 0.5 (°C)"); ax[0].legend(fontsize=8); ax[0].set_title("Vertical centreline")
labs=[r[0].split(" (")[0] for r in rows]; xo=np.arange(len(rows))
ax[1].bar(xo-0.2,[r[4] for r in rows],0.4,label="original code"); ax[1].bar(xo+0.2,[r[1] for r in rows],0.4,label="corrected code")
ax[1].set_xticks(xo); ax[1].set_xticklabels(labs,rotation=30,ha="right",fontsize=8); ax[1].set_ylabel("iterations to 0.01 °C"); ax[1].legend(fontsize=8); ax[1].set_title("Iterations: original vs corrected")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig3_original_vs_corrected.png"),dpi=200)
# omega sweep
fig,ax=plt.subplots(figsize=(6,3.8)); sweep={}
for s,lab in [("psor","PSOR"),("lsor","LSOR"),("adi_relaxation","ADI relaxed")]:
    ws,its=[],[]
    for f in sorted(glob.glob(os.path.join(R,"omega_sweep",f"{s}_w*_log.txt"))):
        w=float(re.search(r"_w([0-9.]+)_log",f).group(1)); it,_,_=readlog(f)
        if len(it): ws.append(w); its.append(it[-1])
    o=np.argsort(ws); ws=np.array(ws)[o]; its=np.array(its)[o]; sweep[lab]=(ws,its); ax.plot(ws,its,"o-",ms=3,label=lab)
rhoJ=0.5*(np.cos(np.pi/(N-1))+np.cos(np.pi/(M-1))); wopt=2/(1+np.sqrt(1-rhoJ**2))
ax.axvline(wopt,color="k",ls="--",lw=0.8,label=rf"theory $\omega_{{opt}}$={wopt:.3f} (PSOR)")
ax.set_xlabel(r"$\omega$"); ax.set_ylabel("iterations"); ax.set_yscale("log"); ax.legend(fontsize=8); ax.set_title("Effect of relaxation factor")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig4_omega_sweep.png"),dpi=200)
# grid study: heat transfer through bottom vs grid
gs=[]
for n in (11,21,41):
    m=2*n-1; f=os.path.join(R,"grid_study",f"lsor_n{n}_field.dat")
    if os.path.exists(f):
        Tg=np.loadtxt(f); ea=np.max(np.abs(Tg[1:-1,1:-1]-analytic(m,n)[1:-1,1:-1])); i2=(m-1)//2
        gs.append((n,m,Qbottom(Tg,m,n),Tg[i2,n//2],analytic(m,n)[i2,n//2],ea))
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
h=[1/(g[0]-1) for g in gs]
ax[0].semilogx(h,[g[2] for g in gs],"s-"); ax[0].set_xlabel(r"$\Delta x=\Delta y$"); ax[0].set_ylabel(r"$Q_{bottom}$ (W per unit depth)"); ax[0].set_title("Bottom heat input vs grid"); ax[0].invert_xaxis()
ax[1].loglog(h,[abs(g[3]-g[4]) for g in gs],"o-",label=r"$|T_h-T_{exact}|$ at plate centre")
ax[1].loglog(h,[abs(gs[0][3]-gs[0][4])*(hh/h[0])**2 for hh in h],"k--",lw=0.8,label=r"slope 2"); ax[1].set_xlabel(r"$\Delta x$"); ax[1].legend(fontsize=8); ax[1].set_title("Spatial convergence"); ax[1].invert_xaxis()
fig.tight_layout(); fig.savefig(os.path.join(R,"fig5_grid_study.png"),dpi=200)
top,left,right=Qout(Tref,M,N); Qb=Qbottom(Tref,M,N)
with open(os.path.join(R,"summary.txt"),"w") as fh:
    fh.write("scheme | iters(corrected) | Q_bottom(corrected,W) | max|T-T_direct| | iters(original) | Q(original formula,W)\n")
    for r in rows: fh.write(f"{r[0]} | {r[1]} | {r[2]} | {r[3]:.3e} | {r[4]} | {r[5]}\n")
    fh.write(f"Direct solve: Q_bottom={Qb:.2f} W ; out: top={top:.2f}, left={left:.2f}, right={right:.2f}, sum={top+left+right:.2f}\n")
    fh.write(f"max|direct - analytical| interior = {np.max(np.abs(Tref-Tan)[1:-1,1:-1]):.3e} C\n")
    fh.write(f"rho_Jacobi={rhoJ:.5f}  omega_opt(PSOR)={wopt:.4f}\n")
    for lab,(ws,its) in sweep.items(): fh.write(f"{lab}: best omega={ws[np.argmin(its)]} iters={its.min()}\n")
    for g in gs: fh.write(f"grid {g[0]}x{g[1]}: Q_bottom={g[2]:.2f} W, T_centre={g[3]:.4f}, exact={g[4]:.4f}, max err={g[5]:.3e}\n")
print(open(os.path.join(R,"summary.txt")).read())

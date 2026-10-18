"""Project 4 post-processing: 2-D Poisson equation on the unit square.
For each run the final field written by the C code is compared with
  (a) the direct sparse solution of the SAME discrete system (5-point stencil, nodal sources
      Q = +1 at (0.45,0.5) and -1 at (0.55,0.5), second-order Neumann closure in Case 2), and
  (b) the analytical solution of the source-free (Laplace) problem.
Case 1: T = 10 on x = 0, 1; T = 0 on y = 0, 1.
Case 2: T = 0 on x = 0, y = 0, y = 1; dT/dx = -1 on x = 1."""
import numpy as np, os, re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scipy.sparse as sp, scipy.sparse.linalg as spla
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.grid":True,"grid.alpha":0.3})
H=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(H,"..","results")
import zipfile  # raw run outputs are shipped zipped (results/raw_runs.zip); extract what is missing
if os.path.exists(os.path.join(R, "raw_runs.zip")):
    with zipfile.ZipFile(os.path.join(R, "raw_runs.zip")) as _z:
        for _m in _z.namelist():
            if not os.path.exists(os.path.join(R, _m)): _z.extract(_m, R)
methods=[("jacobi","Jacobi"),("gauss_seidel","Gauss-Seidel"),("sur",r"SUR ($\alpha$=0.7)")]
deltas=[0.1,0.05,0.01]
def src_nodes(d): return [(int(0.45/d+0.5),int(0.5/d+0.5),1.0),(int(0.55/d+0.5),int(0.5/d+0.5),-1.0)]
def direct(case,d,qscale=1.0,with_src=True):
    n=int(round(1/d))+1; idx=-np.ones((n,n),int); k=0
    for i in range(n):
        for j in range(n):
            interior=(0<i<n-1 and 0<j<n-1); neu=(case==2 and i==n-1 and 0<j<n-1)
            if interior or neu: idx[i,j]=k; k+=1
    T=np.zeros((n,n))
    if case==1: T[0,:]=10; T[-1,:]=10; T[:,0]=0; T[:,-1]=0; T[0,0]=T[0,-1]=T[-1,0]=T[-1,-1]=10
    Q=np.zeros((n,n))
    if with_src:
        for i,j,q in src_nodes(d): Q[i,j]+=q*qscale
    A=sp.lil_matrix((k,k)); b=np.zeros(k)
    for i in range(n):
        for j in range(n):
            r=idx[i,j]
            if r<0: continue
            if case==2 and i==n-1:  # 3T_N - 4T_{N-1} + T_{N-2} = 2 d g
                A[r,r]=3; A[r,idx[i-1,j]]=-4; A[r,idx[i-2,j]]=1; b[r]=2*d*(-1.0); continue
            A[r,r]=-4; b[r]=d*d*Q[i,j]
            for ii,jj in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                if idx[ii,jj]>=0: A[r,idx[ii,jj]]+=1
                else: b[r]-=T[ii,jj]
    x=spla.spsolve(A.tocsr(),b)
    for i in range(n):
        for j in range(n):
            if idx[i,j]>=0: T[i,j]=x[idx[i,j]]
    return T
def analytic(case,d,nt=401):
    n=int(round(1/d))+1; x=np.linspace(0,1,n); y=np.linspace(0,1,n); X,Y=np.meshgrid(x,y,indexing="ij"); S=np.zeros_like(X)
    for m in range(1,nt+1,2):
        if case==1: S+=40/(m*np.pi)*np.sin(m*np.pi*Y)*np.cosh(m*np.pi*(X-0.5))/np.cosh(m*np.pi/2)
        else:       # sinh(m pi x)/cosh(m pi) written in overflow-free exponential form
            S+=-4/(m*np.pi)**2*np.sin(m*np.pi*Y)*(np.exp(m*np.pi*(X-1))-np.exp(-m*np.pi*(X+1)))/(1+np.exp(-2*m*np.pi))
    if case==1: S[0,:]=10; S[-1,:]=10
    return S
its={}
for l in open(os.path.join(R,"iteration_counts.txt")):
    m=re.match(r"(\w+) case(\d) d=([0-9.]+) : Converged after (\d+)",l)
    if m: its[(m.group(1),int(m.group(2)),float(m.group(3)))]=int(m.group(4))
its_t={}
for l in open(os.path.join(R,"iteration_counts_tight.txt")):
    mm=re.match(r"(\w+) case(\d) d=([0-9.]+) : (Converged after|Failed to converge after) (\d+)",l)
    if mm: its_t[(mm.group(1),int(mm.group(2)),float(mm.group(3)))]=int(mm.group(5))
orig_table={("jacobi",1):[185,559,7491],("gauss_seidel",1):[91,308,4441],("sur",1):[165,553,7753],
            ("jacobi",2):[166,558,7474],("gauss_seidel",2):[149,496,7081],("sur",2):[264,883,12345]}
rows=[]; ref={}
for c in (1,2):
    for d in deltas:
        Td=direct(c,d); ref[(c,d)]=Td; Ta=analytic(c,d)
        for m,lab in methods:
            F=np.loadtxt(os.path.join(R,"fields",f"{m}_case{c}_d{d}.dat"))
            n=Td.shape[0]; lo,hi=int(round(0.2/d)),int(round(0.8/d))+1; ic=n//2
            Ft=np.loadtxt(os.path.join(R,"fields_tight",f"{m}_case{c}_d{d}.dat"))
            rows.append((c,d,m,its[(m,c,d)],np.max(np.abs(F-Td)),np.max(np.abs(Td-Ta)[lo:hi,lo:hi]),abs(Td[ic,ic]-Ta[ic,ic]),its_t[(m,c,d)],np.max(np.abs(Ft-Td))))
# Fig 1: contours (GS, delta=0.01) both cases + source effect
fig,ax=plt.subplots(1,3,figsize=(14,4.4))
for a,c in zip(ax[:2],(1,2)):
    F=np.loadtxt(os.path.join(R,"fields_tight",f"gauss_seidel_case{c}_d0.01.dat")); x=np.linspace(0,1,F.shape[0])
    cf=a.contourf(x,x,F.T,levels=21,cmap="inferno" if c==1 else "viridis"); a.contour(x,x,F.T,levels=21,colors="w",linewidths=0.3)
    fig.colorbar(cf,ax=a,shrink=0.85,label="T"); a.set_title(f"Case {c}: Gauss-Seidel, $\\Delta$=0.01 (tol 1e-6)"); a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
    for i,j,q in src_nodes(0.01): a.plot(i*0.01,j*0.01,"c+" if q>0 else "cx",ms=8)
dT=direct(1,0.01)-direct(1,0.01,with_src=False); x=np.linspace(0,1,dT.shape[0])
cf=ax[2].contourf(x,x,dT.T,levels=np.linspace(-np.abs(dT).max(),np.abs(dT).max(),21),cmap="RdBu_r"); fig.colorbar(cf,ax=ax[2],shrink=0.85,format="%.0e")
ax[2].set_xlim(0.3,0.7); ax[2].set_ylim(0.3,0.7); ax[2].set_aspect("equal"); ax[2].set_title(r"Source contribution $T_Q-T_{Q=0}$ (nodal Q)"); ax[2].set_xlabel("x")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig1_contours_and_source_effect.png"),dpi=200)
# Fig 2: iterations vs grid, corrected vs original table
fig,ax=plt.subplots(1,2,figsize=(11,4))
N=np.array([1/d for d in deltas])
for c,a in zip((1,2),ax):
    for (m,lab),col in zip(methods,["C0","C1","C2"]):
        a.loglog(N,[its[(m,c,d)] for d in deltas],"o-",color=col,label=f"{lab} (corrected)")
        a.loglog(N,orig_table[(m,c)],"s--",color=col,mfc="none",alpha=0.7,label=f"{lab} (original table)")
    a.loglog(N,its[("jacobi",1,0.1)]*(N/N[0])**2,"k:",lw=0.8,label=r"$\propto N^2$")
    a.set_xlabel(r"$N=1/\Delta$"); a.set_ylabel("iterations to 1e-4"); a.set_title(f"Case {c}"); a.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(R,"fig2_iterations_vs_grid.png"),dpi=200)
# Fig 3: error at termination
fig,ax=plt.subplots(1,2,figsize=(11,4))
for c,a in zip((1,2),ax):
    for m,lab in methods:
        e=[r[4] for r in rows if r[0]==c and r[2]==m]; a.loglog(N,e,"o-",label=f"{lab}, tol 1e-4")
        e2=[max(r[8],1e-9) for r in rows if r[0]==c and r[2]==m]; a.loglog(N,e2,"s--",mfc="none",label=f"{lab}, tol 1e-6")
    a.set_xlabel(r"$N=1/\Delta$"); a.set_ylabel(r"max$|T_{iter}-T_{direct}|$"); a.set_title(f"Case {c}: error left by the stopping test"); a.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(R,"fig3_termination_error.png"),dpi=200)
# Fig 4: centreline profiles vs analytical
fig,ax=plt.subplots(1,2,figsize=(11,4))
for c,a in zip((1,2),ax):
    for d in deltas:
        F=np.loadtxt(os.path.join(R,"fields",f"gauss_seidel_case{c}_d{d}.dat")); jm=F.shape[1]//2; x=np.linspace(0,1,F.shape[0])
        a.plot(x,F[:,jm],"o-" if d>0.02 else "-",ms=3,label=f"GS, $\\Delta$={d}, tol 1e-4")
        a.plot(x,ref[(c,d)][:,jm],"k:",lw=0.8)
        if d==0.01:
            Ft=np.loadtxt(os.path.join(R,"fields_tight",f"gauss_seidel_case{c}_d{d}.dat")); a.plot(x,Ft[:,jm],"m-",lw=1.2,label=f"GS, $\\Delta$={d}, tol 1e-6")
    Ta=analytic(c,0.005); xa=np.linspace(0,1,Ta.shape[0]); a.plot(xa,Ta[:,Ta.shape[1]//2],"r--",lw=1,label="analytical (Laplace)")
    a.set_xlabel("x"); a.set_ylabel("T(x, 0.5)"); a.set_title(f"Case {c}: horizontal centreline (dotted: direct solve)"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R,"fig4_centreline.png"),dpi=200)
# extension: unit point dipole (Dirac scaling Q = q / dx^2) -- illustrates what a 'point source of strength 1' would give
fig,ax=plt.subplots(figsize=(5,4.2)); dTd=direct(1,0.01,qscale=1/0.01**2)-direct(1,0.01,with_src=False)
cf=ax.contourf(x,x,dTd.T,levels=np.linspace(-0.25,0.25,21),cmap="RdBu_r",extend="both"); fig.colorbar(cf,ax=ax,shrink=0.85)
ax.set_aspect("equal"); ax.set_title(r"Extension: unit point dipole ($Q=\pm1/\Delta^2$)"); ax.set_xlabel("x"); ax.set_ylabel("y")
fig.tight_layout(); fig.savefig(os.path.join(R,"fig5_extension_point_dipole.png"),dpi=200)
with open(os.path.join(R,"summary.txt"),"w") as fh:
    fh.write("case | delta | method | iters(1e-4) | max|T-T_direct| (1e-4) | max|T_direct-T_exact| on [0.2,0.8]^2 | |T_direct-T_exact| at centre | iters(1e-6) | max|T-T_direct| (1e-6)\n")
    for r in rows: fh.write(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.3e} | {r[5]:.3e} | {r[6]:.3e} | {r[7]} | {r[8]:.3e}\n")
    fh.write(f"nodal-source contribution (case 1, d=0.01): max|T_Q - T_0| = {np.abs(dT).max():.3e}\n")
    fh.write(f"Dirac-scaled dipole (extension): max|dT| = {np.abs(dTd).max():.3e}\n")
    for d in deltas: fh.write(f"theory d={d}: rho_J=cos(pi d)={np.cos(np.pi*d):.6f}, rho_GS={np.cos(np.pi*d)**2:.6f}\n")
print(open(os.path.join(R,"summary.txt")).read())

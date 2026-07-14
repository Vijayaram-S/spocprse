"""
Corrected single-qubit experiments for:
'Structure-Preserving Fourth-Order Lie-Group Integration and Noise-Metric-Aware
 Optimal Control for Random Schrodinger Equations'

Fixes vs. original notebook:
  * Correct CFM4 coefficients: Omega1 = -i h (a1 H(t1) + a2 H(t2)),
    Omega2 = -i h (a2 H(t1) + a1 H(t2)), a1,2 = 1/4 +- sqrt(3)/6  (Gauss nodes
    c1,2 = 1/2 -+ sqrt(3)/6).  Original code advanced 2h per step (order 2).
  * Convergence measured against an INDEPENDENT reference integrator.
  * Deterministic gate cost for the surrogate formulations (as in the paper);
    ensemble-averaged gate cost kept as an explicit baseline (robust GRAPE).
  * All RNG seeds fixed.
Pure numpy (no scipy).
"""
import numpy as np, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'serif','font.size':12,'axes.labelsize':14,
 'legend.fontsize':11,'xtick.labelsize':11,'ytick.labelsize':11,'figure.dpi':110,
 'savefig.dpi':300,'savefig.bbox':'tight','mathtext.fontset':'cm',
 'axes.grid':True,'grid.alpha':0.5,'grid.linestyle':'--'})

OUT='/tmp/rse/out'; os.makedirs(OUT,exist_ok=True)
def savefig(name):
    plt.savefig(f'{OUT}/{name}.pdf'); plt.savefig(f'{OUT}/{name}.png'); plt.close()
    print('saved',name,flush=True)

# ---------------- setup ----------------
T=5.0; N=200; h=T/N
omega0=1.0
tgrid=np.linspace(0,T,N+1)              # N steps
c1,c2=0.5-np.sqrt(3)/6.0,0.5+np.sqrt(3)/6.0
a1,a2=0.25+np.sqrt(3)/6.0,0.25-np.sqrt(3)/6.0
tn1=tgrid[:-1]+c1*h; tn2=tgrid[:-1]+c2*h        # quadrature node times (N each)
theta=np.pi/2
V=np.array([[np.cos(theta/2),-1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2),np.cos(theta/2)]])
I2=np.eye(2,dtype=complex)

# ---------------- Matern nu=3/2 sampling at arbitrary times ----------------
def matern_cov(ts,sigma,ell):
    r=np.abs(ts[:,None]-ts[None,:]); f=np.sqrt(3.0)*r/ell
    return sigma**2*(1.0+f)*np.exp(-f)+1e-10*np.eye(len(ts))
def sample_matern(rng,ts,sigma,ell,S):
    L=np.linalg.cholesky(matern_cov(ts,sigma,ell))
    return (L@rng.standard_normal((len(ts),S))).T    # (S,len(ts))

# times where noise is needed: quadrature nodes (for CFM4) ; grid+mid (RK4)
ts_nodes=np.sort(np.concatenate([tn1,tn2]))
idx1=np.searchsorted(ts_nodes,tn1); idx2=np.searchsorted(ts_nodes,tn2)

# ---------------- batched CFM4 propagator ----------------
def prop_cfm4(ux,uy,nz,ndx,ndy,return_traj=False):
    """CFM4, vectorized. Controls ux,uy: (B,N). Noise dict entries are tuples
    (node1,node2) each (S,N), or None:
      nz  : additive sigma_z noise eta(t)
      ndx : relative amplitude noise on the x control line, u_x -> u_x(1+delta_x)
      ndy : relative amplitude noise on the y control line."""
    ux=np.atleast_2d(ux); uy=np.atleast_2d(uy)
    B=ux.shape[0]
    def pair(p):
        if p is None: return None
        return (np.atleast_2d(p[0]),np.atleast_2d(p[1]))
    nz,ndx,ndy=pair(nz),pair(ndx),pair(ndy)
    S=1
    for p in (nz,ndx,ndy):
        if p is not None: S=max(S,p[0].shape[0])
    z1=0.5*omega0+(nz[0] if nz is not None else 0.0)
    z2=0.5*omega0+(nz[1] if nz is not None else 0.0)
    z1=np.broadcast_to(np.atleast_2d(z1),(S,N))[None,:,:]
    z2=np.broadcast_to(np.atleast_2d(z2),(S,N))[None,:,:]
    ux_=ux[:,None,:]; uy_=uy[:,None,:]                    # (B,1,N)
    fx1=1.0+(ndx[0][None,:,:] if ndx is not None else 0.0)  # (1,S,N) or scalar
    fx2=1.0+(ndx[1][None,:,:] if ndx is not None else 0.0)
    fy1=1.0+(ndy[0][None,:,:] if ndy is not None else 0.0)
    fy2=1.0+(ndy[1][None,:,:] if ndy is not None else 0.0)
    o1=ux_*fx1-1j*uy_*fy1                                  # (B,S,N)
    o2=ux_*fx2-1j*uy_*fy2
    Es=[]
    for (wa,wb) in ((a1,a2),(a2,a1)):
        w=-1j*h*(wa*z1+wb*z2)
        o=-1j*h*(wa*o1+wb*o2)
        v=np.sqrt(np.abs(w)**2+np.abs(o)**2)
        cv=np.cos(v)
        vs=np.where(v>1e-14,v,1.0)
        sv=np.where(v>1e-14,np.sin(vs)/vs,1.0)
        Es.append((cv+sv*w, sv*o, -sv*np.conj(o), cv-sv*w))
    U00=np.ones((B,S),complex); U01=np.zeros((B,S),complex)
    U10=np.zeros((B,S),complex); U11=np.ones((B,S),complex)
    traj=[]
    for n in range(N):
        for (E00,E01,E10,E11) in Es:
            a_=np.broadcast_to(E00[...,n],(B,S)); b_=np.broadcast_to(E01[...,n],(B,S))
            c_=np.broadcast_to(E10[...,n],(B,S)); d_=np.broadcast_to(E11[...,n],(B,S))
            nU00=a_*U00+b_*U10; nU01=a_*U01+b_*U11
            nU10=c_*U00+d_*U10; nU11=c_*U01+d_*U11
            U00,U01,U10,U11=nU00,nU01,nU10,nU11
        if return_traj: traj.append((U00.copy(),U01.copy(),U10.copy(),U11.copy()))
    if return_traj: return traj
    return U00,U01,U10,U11

ZERO=None
def noise_pairs(raw,scale,clip=3.0):
    """raw: (S,K) standard field at ts_nodes; returns node pair (S,N),(S,N) scaled+clipped"""
    x=np.clip(raw,-clip,clip)*scale
    return (x[:,idx1],x[:,idx2])

def infid(U00,U01,U10,U11):
    tr=np.conj(V[0,0])*U00+np.conj(V[1,0])*U01+np.conj(V[0,1])*U10+np.conj(V[1,1])*U11
    return 1.0-np.abs(tr)/2.0

def dF(U00,U01,U10,U11,W00,W01,W10,W11):
    """Riemannian distance ||log(U0^dag U)||_F for SU(2): sqrt(2)*phi."""
    tr=np.conj(W00)*U00+np.conj(W01)*U01+np.conj(W10)*U10+np.conj(W11)*U11
    phi=np.arccos(np.clip(np.real(tr)/2.0,-1.0,1.0))
    return np.sqrt(2.0)*phi

def Jnoise(ux,uy,lx,ly): return h*np.sum(np.sqrt(lx*ux**2+ly*uy**2),axis=-1)

# ---------------- optimizer: projected Adam, batched central FD ----------------
def optimize(lam,lx,ly,train_e1,train_e2,iters=450,lr0=0.05,seed_init=True,tag=''):
    """train_e1/e2 None => deterministic gate cost; else ensemble-mean infidelity."""
    ux=0.5*np.sin(2*np.pi*tgrid[:-1]/T); uy=0.5*np.cos(2*np.pi*tgrid[:-1]/T)
    u=np.concatenate([ux,uy])
    if train_e1 is None: e1=np.zeros((1,N)); e2=np.zeros((1,N))
    else: e1,e2=train_e1,train_e2
    def J_batch(ub):                                    # ub (B,2N)
        Ux=ub[:,:N]; Uy=ub[:,N:]
        c=infid(*prop_cfm4(Ux,Uy,e1,e2)).mean(axis=1)   # (B,)
        return c+lam*Jnoise(Ux,Uy,lx,ly)
    eps=3e-7; m=np.zeros(2*N); vAd=np.zeros(2*N); b1,b2=0.9,0.999
    for it in range(1,iters+1):
        P=np.vstack([u[None,:]+eps*np.eye(2*N),u[None,:]])
        Jv=J_batch(P)
        g=(Jv[:2*N]-Jv[-1])/eps
        lr=lr0*(0.3 if it>iters*0.6 else 1.0)*(0.1 if it>iters*0.85 else 1.0)
        m=b1*m+(1-b1)*g; vAd=b2*vAd+(1-b2)*g*g
        mh=m/(1-b1**it); vh=vAd/(1-b2**it)
        u=np.clip(u-lr*mh/(np.sqrt(vh)+1e-12),-2.0,2.0)
        if it%200==0: print(f'  [{tag}] iter {it}  J={Jv[-1]:.6f}',flush=True)
    ux,uy=u[:N],u[N:]
    ci=float(infid(*prop_cfm4(ux,uy,None,None,None))[0,0])
    print(f'  [{tag}] clean infidelity {ci:.3e}',flush=True)
    return ux,uy,ci

results={}
import os

import sys
OUTD=OUT
RES=f'{OUTD}/results.json'
def load_res():
    try: return json.load(open(RES))
    except: return {}
def save_res(r): json.dump(r,open(RES,'w'),indent=1)

SCEN=os.environ.get('SCEN','M')   # 'M' multiplicative (theory-consistent), 'Z' dephasing
SIG_X=0.10; SIG_Y=0.20; SIG_Z=0.05; ELL=0.4
rng_train=np.random.default_rng(12345)
Lstd=np.linalg.cholesky(matern_cov(ts_nodes,1.0,ELL))    # unit-variance field
def draw_std(rng,S): return (Lstd@rng.standard_normal((len(ts_nodes),S))).T
def make_noise(rng,S,sx=1.0,sell=None):
    """returns (nz,ndx,ndy) for prop_cfm4; sx scales all amplitudes."""
    global Lstd
    if sell is not None:
        Ls=np.linalg.cholesky(matern_cov(ts_nodes,1.0,sell))
        dr=lambda: (Ls@rng.standard_normal((len(ts_nodes),S))).T
    else:
        dr=lambda: draw_std(rng,S)
    if SCEN=='M':
        return (None, noise_pairs(dr(),SIG_X*sx), noise_pairs(dr(),SIG_Y*sx))
    else:
        return (noise_pairs(dr(),SIG_Z*sx), None, None)
tr_noise=make_noise(rng_train,30)

def opt_ckpt(name,lam,lx,ly,ens,iters=450,lr0=0.05,chunk=10**9):
    """checkpointed projected-Adam optimization"""
    f=f'{OUTD}/ckpt_{SCEN}_{name}.npz'
    nz,ndx,ndy=tr_noise if ens else (None,None,None)
    def J_batch(ub):
        Ux=ub[:,:N]; Uy=ub[:,N:]
        c=infid(*prop_cfm4(Ux,Uy,nz,ndx,ndy)).mean(axis=1)
        return c+lam*Jnoise(Ux,Uy,lx,ly)
    if os.path.exists(f):
        d=np.load(f); u,m,vA,it0=d['u'],d['m'],d['v'],int(d['it'])
        if it0>=iters:
            ux,uy=u[:N],u[N:]
            ci=float(infid(*prop_cfm4(ux,uy,None,None,None))[0,0])
            return ux,uy,ci,True
    else:
        ux0=0.5*np.sin(2*np.pi*tgrid[:-1]/T); uy0=0.5*np.cos(2*np.pi*tgrid[:-1]/T)
        u=np.concatenate([ux0,uy0]); m=np.zeros(2*N); vA=np.zeros(2*N); it0=0
    eps=3e-7; b1,b2=0.9,0.999
    it_end=min(iters,it0+chunk)
    for it in range(it0+1,it_end+1):
        P=np.vstack([u[None,:]+eps*np.eye(2*N),u[None,:]])
        Jv=J_batch(P)
        g=(Jv[:2*N]-Jv[-1])/eps
        lr=lr0*(0.3 if it>iters*0.6 else 1.0)*(0.1 if it>iters*0.85 else 1.0)
        m=b1*m+(1-b1)*g; vA=b2*vA+(1-b2)*g*g
        mh=m/(1-b1**it); vh=vA/(1-b2**it)
        u=np.clip(u-lr*mh/(np.sqrt(vh)+1e-12),-2.0,2.0)
    np.savez(f,u=u,m=m,v=vA,it=it_end)
    ux,uy=u[:N],u[N:]
    ci=float(infid(*prop_cfm4(ux,uy,None,None,None))[0,0])
    done=it_end>=iters
    print(f'[{name}] it={it_end}/{iters} clean={ci:.3e} done={done}',flush=True)
    return ux,uy,ci,done

VAR=dict(A=('A',0.0,1,1,False),B=('B',0.005,1,1,False),
         C=('C',0.005,1,4,False),D=('D',0.0,1,1,True))
def get(name,**kw):
    nm,lam,lx,ly,ens=VAR[name]; return opt_ckpt(nm,lam,lx,ly,ens,**kw)

def val_noise():
    return make_noise(np.random.default_rng(777),1000)
def evaluate(ux,uy,nzz): return infid(*prop_cfm4(ux,uy,*nzz))[0]
def stats(x): return dict(mean=float(x.mean()),std=float(x.std()),wc=float(x.max()))

stage=sys.argv[1] if len(sys.argv)>1 else 'help'
r=load_res()

if stage=='opts':
    for nm in ['A','B','C']: get(nm)
    print('OPTS_DONE',flush=True)

elif stage=='optD':
    _,_,_,done=get('D',iters=300,chunk=55)
    print('D_DONE' if done else 'D_PARTIAL',flush=True)

elif stage=='eval':
    vn=val_noise()
    out={}; infs={}
    for nm in ['A','B','C','D']:
        it=300 if nm=='D' else 450
        ux,uy,ci,done=get(nm,iters=it)
        assert done, nm+' not done'
        iv=evaluate(ux,uy,vn); infs[nm]=iv
        out[nm]=dict(**stats(iv),clean=ci,Jnoise_aniso=float(Jnoise(ux,uy,1,4)))
    r['baseline']=out; save_res(r)
    print(json.dumps(out,indent=1),flush=True)
    inf_A,inf_B,inf_C,inf_D=infs['A'],infs['B'],infs['C'],infs['D']
    plt.figure(figsize=(8,5))
    bins=np.linspace(0,max(inf_A.max(),inf_C.max())*1.02,60)
    plt.hist(inf_A,bins=bins,alpha=.6,label='noise-blind ($\\lambda=0$)',color='tab:red')
    plt.hist(inf_C,bins=bins,alpha=.6,label='geometry-aware ($\\lambda=0.005$, $l_y=4$)',color='tab:blue')
    plt.xlabel('gate infidelity'); plt.ylabel('count (1000 samples)'); plt.legend()
    savefig('figure4_histogram')
    plt.figure(figsize=(8,5))
    data=[inf_A,inf_D,inf_B,inf_C]
    labels=['noise-blind\n$\\lambda=0$','ensemble\nrobust GRAPE','isotropic\n$l_y=1$','anisotropic\n$l_y=4$']
    bp=plt.boxplot(data,labels=labels,showfliers=False,patch_artist=True,whis=(5,95))
    for p,c in zip(bp['boxes'],['tab:red','tab:purple','tab:green','tab:blue']):
        p.set_facecolor(c); p.set_alpha(.55)
    plt.yscale('log'); plt.ylabel('gate infidelity (1000 validation samples)')
    savefig('figure11_baseline_comparison')
    with open(f'{OUTD}/table4_baseline.csv','w') as fcsv:
        fcsv.write('variant,mean,std,worst_case,clean,Jnoise\n')
        for nm,lab in [('A','noise-blind'),('D','ensemble-GRAPE'),('B','isotropic'),('C','anisotropic')]:
            o=out[nm]; fcsv.write(f"{lab},{o['mean']:.6f},{o['std']:.6f},{o['wc']:.6f},{o['clean']:.2e},{o['Jnoise_aniso']:.4f}\n")
    for nm,fn,ttl in [('A','figure8a_controls_blind','noise-blind ($\\lambda=0$)'),
                      ('C','figure8b_controls_aware','geometry-aware ($\\lambda=0.005,\\ l_y=4$)')]:
        ux,uy,_,_=get(nm,iters=450)
        plt.figure(figsize=(8,4.6))
        plt.step(tgrid[:-1],ux,where='post',label='$u_x(t)$',lw=1.8)
        plt.step(tgrid[:-1],uy,where='post',label='$u_y(t)$',lw=1.8)
        plt.xlabel('time $t$'); plt.ylabel('control amplitude'); plt.title(ttl); plt.legend()
        savefig(fn)
    print('EVAL_DONE',flush=True)

elif stage=='unitarity':
    sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]],complex);sz=np.array([[1,0],[0,-1]],complex)
    ux_C,uy_C,_,_=get('C',iters=450)
    ts_all=np.sort(np.unique(np.concatenate([tgrid,tgrid[:-1]+h/2,ts_nodes])))
    rr=np.random.default_rng(2024)
    pz=sample_matern(rr,ts_all,SIG_Z,ELL,1)[0]
    px=np.clip(sample_matern(rr,ts_all,1.0,ELL,1)[0],-3,3)*SIG_X
    py=np.clip(sample_matern(rr,ts_all,1.0,ELL,1)[0],-3,3)*SIG_Y
    def Hfull(t):
        n=min(int(t/h),N-1)
        if SCEN=='M':
            return 0.5*omega0*sz+ux_C[n]*(1+np.interp(t,ts_all,px))*sx+uy_C[n]*(1+np.interp(t,ts_all,py))*sy
        return 0.5*omega0*sz+ux_C[n]*sx+uy_C[n]*sy+np.interp(t,ts_all,pz)*sz
    def e2x2(O):
        v=np.sqrt(abs(O[0,0])**2+abs(O[0,1])**2)
        return np.cos(v)*I2+(np.sin(v)/v)*O if v>1e-14 else I2
    dc=[];drk=[]
    U=I2.copy()
    for n in range(N):
        H1=Hfull(tgrid[n]+c1*h);H2=Hfull(tgrid[n]+c2*h)
        U=e2x2(-1j*h*(a2*H1+a1*H2))@e2x2(-1j*h*(a1*H1+a2*H2))@U
        dc.append(np.linalg.norm(U.conj().T@U-I2,'fro'))
    U=I2.copy()
    for n in range(N):
        t0=tgrid[n]
        F=lambda X,t: -1j*Hfull(t)@X
        k1=F(U,t0);k2=F(U+0.5*h*k1,t0+h/2);k3=F(U+0.5*h*k2,t0+h/2);k4=F(U+h*k3,t0+h)
        U=U+(h/6)*(k1+2*k2+2*k3+k4)
        drk.append(np.linalg.norm(U.conj().T@U-I2,'fro'))
    r['unitarity']={'cfm4_max':float(max(dc)),'rk4_final':float(drk[-1])}; save_res(r)
    plt.figure(figsize=(8,5))
    plt.semilogy(tgrid[1:],np.maximum(dc,1e-17),label='CFM4 (structure-preserving)',lw=2)
    plt.semilogy(tgrid[1:],np.maximum(drk,1e-17),label='RK4',lw=2)
    plt.xlabel('time $t$'); plt.ylabel(r'$\|U^\dagger U-I\|_F$'); plt.legend()
    savefig('figure1_structure_preservation')
    with open(f'{OUTD}/table1_structure_preservation.csv','w') as fcsv:
        fcsv.write('cfm4_max_defect,rk4_final_defect\n')
        fcsv.write(f"{max(dc):.3e},{drk[-1]:.3e}\n")
    print('UNITARITY_DONE',json.dumps(r['unitarity']),flush=True)

elif stage=='pareto':
    vn=val_noise()
    lams=[0.0,1e-4,1e-3,5e-3,1e-2]; par=[]; ctrls={}
    for l in lams:
        if l==0.0: nm='A'
        elif l==0.005: nm='C'
        else: nm=f'P{l:g}'; VAR[nm]=(nm,l,1,4,False)
        ux,uy,ci,done=get(nm,iters=450)
        ctrls[l]=(ux,uy)
        par.append((l,float(Jnoise(ux,uy,1,4)),ci,float(evaluate(ux,uy,vn).mean())))
    r['pareto']=par; save_res(r)
    plt.figure(figsize=(8,5))
    plt.plot([p[1] for p in par],[p[2] for p in par],'o-',lw=2,ms=8)
    for p in par: plt.annotate(f'$\\lambda={p[0]:g}$',(p[1],p[2]),textcoords='offset points',xytext=(6,6),fontsize=10)
    plt.xlabel(r'geometric noise cost $J_{\mathrm{noise}}$'); plt.ylabel('deterministic gate infidelity')
    plt.yscale('symlog',linthresh=1e-6)
    savefig('figure3_pareto_frontier')
    with open(f'{OUTD}/table2_pareto.csv','w') as fcsv:
        fcsv.write('lambda,Jnoise,clean_infidelity,mean_noisy_infidelity\n')
        for p in par: fcsv.write(f'{p[0]:g},{p[1]:.4f},{p[2]:.3e},{p[3]:.5f}\n')
    # theorem verification scatter
    plt.figure(figsize=(8,5))
    cols=['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
    maxr=0.0
    for (l,col) in zip(lams,cols):
        ux,uy=ctrls[l]
        U0=prop_cfm4(ux,uy,None,None,None)
        sub=tuple(None if p is None else (p[0][:300],p[1][:300]) for p in vn)
        Un=prop_cfm4(ux,uy,*sub)
        d=dF(Un[0][0],Un[1][0],Un[2][0],Un[3][0],U0[0][0,0],U0[1][0,0],U0[2][0,0],U0[3][0,0])
        jn=float(Jnoise(ux,uy,1,4))
        if jn>0: maxr=max(maxr,float(np.max(d))/jn)
        plt.scatter([jn]*300,d,s=14,alpha=.3,color=col,label=f'$\\lambda={l:g}$')
    xl=np.linspace(0,max(Jnoise(u[0],u[1],1,4) for u in ctrls.values())*1.1,50)
    plt.plot(xl,xl,'k--',lw=2,label='$y=x$')
    plt.xlabel(r'$J_{\mathrm{noise}}$'); plt.ylabel(r'$d_F(U(T,\omega),U_0(T))$'); plt.legend()
    r['theorem']={'max_ratio_dF_over_Jnoise':maxr}; save_res(r)
    savefig('figure5_theorem_verification')
    print('PARETO_DONE maxratio',maxr,flush=True)

elif stage=='sweeps':
    vn=val_noise()
    ux_A,uy_A,_,_=get('A',iters=450); ux_C,uy_C,_,_=get('C',iters=450)
    rng_sw=np.random.default_rng(31415)
    sig_list=[0.2,0.6,1.0,1.6,2.0]; ell_list=[0.1,0.2,0.5,1.0]
    sw_s={'blind':[],'aware':[]}
    for sg in sig_list:
        nn=make_noise(rng_sw,300,sx=sg)
        sw_s['blind'].append(float(evaluate(ux_A,uy_A,nn).mean()))
        sw_s['aware'].append(float(evaluate(ux_C,uy_C,nn).mean()))
    sw_l={'blind':[],'aware':[]}
    for lc in ell_list:
        nn=make_noise(rng_sw,300,sell=lc)
        sw_l['blind'].append(float(evaluate(ux_A,uy_A,nn).mean()))
        sw_l['aware'].append(float(evaluate(ux_C,uy_C,nn).mean()))
    r['sweep_sigma']={'sigma':sig_list,**sw_s}; r['sweep_ell']={'ell':ell_list,**sw_l}; save_res(r)
    plt.figure(figsize=(8,5))
    plt.semilogy(sig_list,sw_s['blind'],'o-',color='tab:red',lw=2,label='noise-blind')
    plt.semilogy(sig_list,sw_s['aware'],'s-',color='tab:blue',lw=2,label='geometry-aware')
    plt.xlabel(r'noise amplitude $\sigma$'); plt.ylabel('mean gate infidelity'); plt.legend()
    savefig('figure6_noise_amplitude')
    plt.figure(figsize=(8,5))
    plt.semilogy(ell_list,sw_l['blind'],'o-',color='tab:red',lw=2,label='noise-blind')
    plt.semilogy(ell_list,sw_l['aware'],'s-',color='tab:blue',lw=2,label='geometry-aware')
    plt.xlabel(r'correlation length $\ell$'); plt.ylabel('mean gate infidelity'); plt.legend()
    savefig('figure7_correlation_length')
    print('SWEEPS_DONE',flush=True)

elif stage=='aniso':
    vn=val_noise()
    ly_list=[1,2,4,8,16]; ai=[]; an=[]
    for lyv in ly_list:
        if lyv==1: nm='B'
        elif lyv==4: nm='C'
        else: nm=f'L{lyv}'; VAR[nm]=(nm,0.005,1,lyv,False)
        ux,uy,ci,done=get(nm,iters=450)
        ai.append(float(evaluate(ux,uy,vn).mean()))
        an.append(float(Jnoise(ux,uy,1,lyv)))
    r['anisotropy']={'ly':ly_list,'mean_infid':ai,'Jnoise':an}; save_res(r)
    plt.figure(figsize=(7,4.6))
    plt.plot(ly_list,ai,'o-',lw=2,ms=8); plt.xscale('log',base=2)
    plt.xlabel(r'anisotropy weight $l_y$'); plt.ylabel('mean gate infidelity')
    savefig('figure10a_anisotropy_infidelity')
    plt.figure(figsize=(7,4.6))
    plt.plot(ly_list,an,'s-',color='tab:green',lw=2,ms=8); plt.xscale('log',base=2)
    plt.xlabel(r'anisotropy weight $l_y$'); plt.ylabel(r'geometric noise cost $J_{\mathrm{noise}}$')
    savefig('figure10b_anisotropy_noise')
    print('ANISO_DONE',flush=True)
else:
    print('stages: opts optD eval unitarity pareto sweeps aniso')

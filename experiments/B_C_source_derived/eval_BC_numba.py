import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
W=np.load(ROOT/'trained_weights_prelim.npz')
mean_arr=W['mean_arr'].astype(np.float64); std_arr=W['std_arr'].astype(np.float64)
d2_CA=W['d2_CA'].astype(np.float64); d2_CB=W['d2_CB'].astype(np.float64); d2_bias=float(W['d2_bias']); d2_gain=W['d2_gain'].astype(np.float64); d2_obias=W['d2_obias'].astype(np.float64); d2_dtc=float(W['d2_dtc']); d2_steps=int(W['d2_steps'])
ae_W1=W['ae_W1'].astype(np.float64); ae_b1=W['ae_b1'].astype(np.float64); ae_W2=W['ae_W2'].astype(np.float64); ae_b2=W['ae_b2'].astype(np.float64)
ae_Wd1=W['ae_Wd1'].astype(np.float64); ae_bd1=W['ae_bd1'].astype(np.float64); ae_Wd2=W['ae_Wd2'].astype(np.float64); ae_bd2=W['ae_bd2'].astype(np.float64)
aec_CA=W['aec_CA'].astype(np.float64); aec_CB=W['aec_CB'].astype(np.float64); aec_bias=float(W['aec_bias']); aec_gain=W['aec_gain'].astype(np.float64); aec_obias=W['aec_obias'].astype(np.float64); aec_dtc=float(W['aec_dtc']); aec_steps=int(W['aec_steps'])
mlp_W1=W['mlp_W1'].astype(np.float64); mlp_b1=W['mlp_b1'].astype(np.float64); mlp_W2=W['mlp_W2'].astype(np.float64); mlp_b2=W['mlp_b2'].astype(np.float64)
V3=np.load(ROOT/'v3_weights.npz')
v3_CA=V3['CA'].astype(np.float64); v3_CB=V3['CB'].astype(np.float64); v3_bias=float(V3['bias']); v3_gain=V3['gain'].astype(np.float64); v3_obias=V3['obias'].astype(np.float64); v3_steps=int(V3['steps']); v3_dtc=float(V3['dtc'])

M_NOM=0.826; G=9.81; L=0.166; KETA=7.64e-6; KYAW=0.014; TAU_M=0.005; WMAX=1000.0
I_NOM=np.array([0.0047,0.005,0.0074],dtype=np.float64)
KD=1.19e-4; KZ=2.32e-4
rpos=np.array([[ L/np.sqrt(2), L/np.sqrt(2),0],[ L/np.sqrt(2),-L/np.sqrt(2),0],[-L/np.sqrt(2),-L/np.sqrt(2),0],[-L/np.sqrt(2), L/np.sqrt(2),0]],dtype=np.float64)
rdir=np.array([1.,-1.,1.,-1.],dtype=np.float64)
Aalloc=np.zeros((4,4),dtype=np.float64); Aalloc[0,:]=1
for i in range(4):
    x,y,_=rpos[i]
    Aalloc[1,i]=y; Aalloc[2,i]=-x; Aalloc[3,i]=KYAW*rdir[i]
Ainv=np.linalg.inv(Aalloc)
KX=np.array([5.,5.,10.]); KV=np.array([4.,4.,8.]); KR=np.array([0.3,0.3,0.3]); KW=np.array([0.03,0.03,0.03])
# causal polynomial derivative coefficients, 9-point window ending at current sample
DT=0.01; NWIN=9; tt=np.arange(-(NWIN-1),1)*DT; XX=np.c_[np.ones(NWIN),tt,tt*tt]; PP=np.linalg.pinv(XX); C1=PP[1].astype(np.float64); C2=(2*PP[2]).astype(np.float64)

@njit(cache=True)
def traj(t):
    freq=0.08; om=2*np.pi*freq
    p=np.empty(3); v=np.empty(3); a=np.empty(3)
    p[0]=-2+2*np.cos(om*t); p[1]=2*np.sin(om*t); p[2]=1.5+0.25*np.sin(0.5*om*t)
    v[0]=-2*om*np.sin(om*t); v[1]=2*om*np.cos(om*t); v[2]=0.25*0.5*om*np.cos(0.5*om*t)
    a[0]=-2*om*om*np.cos(om*t); a[1]=-2*om*om*np.sin(om*t); a[2]=-0.25*(0.5*om)**2*np.sin(0.5*om*t)
    return p,v,a

@njit(cache=True)
def norm3(a): return np.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
@njit(cache=True)
def cross(a,b):
    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])

@njit(cache=True)
def q_to_R(q): # x,y,z,w
    x,y,z,w=q
    R=np.empty((3,3))
    R[0,0]=1-2*(y*y+z*z); R[0,1]=2*(x*y-z*w); R[0,2]=2*(x*z+y*w)
    R[1,0]=2*(x*y+z*w); R[1,1]=1-2*(x*x+z*z); R[1,2]=2*(y*z-x*w)
    R[2,0]=2*(x*z-y*w); R[2,1]=2*(y*z+x*w); R[2,2]=1-2*(x*x+y*y)
    return R

@njit(cache=True)
def q_step(q,w,dt):
    x,y,z,s=q; wx,wy,wz=w
    dq=np.empty(4)
    dq[0]=0.5*( s*wx + y*wz - z*wy )
    dq[1]=0.5*( s*wy + z*wx - x*wz )
    dq[2]=0.5*( s*wz + x*wy - y*wx )
    dq[3]=0.5*(-x*wx - y*wy - z*wz )
    q2=q+dt*dq; n=np.sqrt(np.sum(q2*q2)); return q2/n

@njit(cache=True)
def desired_R(F):
    b3=F/(norm3(F)+1e-12)
    c1=np.array([1.,0.,0.]); b2=cross(b3,c1); nb=norm3(b2)
    if nb<1e-8:
        c1=np.array([0.,1.,0.]); b2=cross(b3,c1); nb=norm3(b2)
    b2=b2/nb; b1=cross(b2,b3)
    R=np.empty((3,3))
    for i in range(3): R[i,0]=b1[i]; R[i,1]=b2[i]; R[i,2]=b3[i]
    return R

@njit(cache=True)
def cenn_forward(inp,CA,CB,bias,gain,obias,steps,dtc):
    x=np.zeros(9)
    C=inp.shape[0]
    inv=inp.reshape((C,9))
    for _ in range(steps):
        y=np.minimum(1.0,np.maximum(-1.0,x))
        drive=CA@y
        for c in range(C): drive += CB[c]@inv[c]
        x=x+dtc*(-x+drive+bias)
    y=np.minimum(1.0,np.maximum(-1.0,x)).reshape((3,3))
    out=np.empty(3)
    for j in range(3): out[j]=3.0*np.tanh(y[2,j]*gain[j]+obias[j])
    return out

@njit(cache=True)
def normalize_planes(planes):
    out=np.empty((3,3,3))
    for c in range(3):
        for i in range(3):
            for j in range(3):
                z=(planes[c,i,j]-mean_arr[c,i,j])/std_arr[c,i,j]
                if z>6: z=6
                if z<-6: z=-6
                out[c,i,j]=z
    return out

@njit(cache=True)
def ae_reconstruct(xn):
    rec=np.empty((3,3,3))
    for i in range(3):
        for j in range(3):
            q=np.array([xn[0,i,j],xn[1,i,j],xn[2,i,j]])
            h=np.tanh(ae_W1@q+ae_b1)
            z=np.tanh(ae_W2@h+ae_b2)
            hd=np.tanh(ae_Wd1@z+ae_bd1)
            qr=ae_Wd2@hd+ae_bd2
            rec[0,i,j]=qr[0]; rec[1,i,j]=qr[1]; rec[2,i,j]=qr[2]
    return rec

@njit(cache=True)
def residual(planes,mode,t):
    if mode==0: return np.zeros(3)
    xn=normalize_planes(planes)
    rho=0.35
    if mode==1: # MLP
        flat=xn.reshape(27); h=np.tanh(mlp_W1@flat+mlp_b1); raw=mlp_W2@h+mlp_b2
        return rho*3*np.tanh(raw/3)
    if mode==2: return rho*cenn_forward(xn,d2_CA,d2_CB,d2_bias,d2_gain,d2_obias,d2_steps,d2_dtc)
    if mode==3 or mode==4:
        if mode==4 and t>=3.0 and t<=4.0: return np.zeros(3)
        rec=ae_reconstruct(xn)
        return rho*cenn_forward(rec,aec_CA,aec_CB,aec_bias,aec_gain,aec_obias,aec_steps,aec_dtc)
    if mode==5:
        rep=np.empty((3,3,3))
        for c in range(3): rep[c]=xn[0]
        return rho*cenn_forward(rep,v3_CA,v3_CB,v3_bias,v3_gain,v3_obias,v3_steps,v3_dtc)
    return np.zeros(3)

@njit(cache=True)
def run_episode(seed,scenario,ctrl,resmode,T=6.0,dt=0.01):
    np.random.seed(1000+seed)
    steps=int(T/dt)
    mass=M_NOM; I=I_NOM.copy(); eff=np.ones(4)
    if scenario==3 or scenario==7: # model/compound
        rr=0.2 if scenario==3 else 0.12; s=np.random.uniform(-rr,rr); mass=M_NOM*(1+s); I=I_NOM*(1+np.random.uniform(-rr,rr))
    if scenario==6 or scenario==7:
        rr=0.30 if scenario==6 else 0.10
        for i in range(4): eff[i]=1+np.random.uniform(-rr,rr)
    payload=0.; ptoggle=np.random.uniform(2.0,4.0)
    if scenario==5 or scenario==7:
        lo,hi=(0.35,0.50) if scenario==5 else (0.12,0.22); payload=M_NOM*np.random.uniform(lo,hi)
    ftoggle=np.random.uniform(2.0,3.5); fend=ftoggle+np.random.uniform(1.0,2.0); fvec=np.zeros(3)
    if scenario==2 or scenario==7:
        d=np.random.normal(0,1,3); d[2]*=.3; d=d/(norm3(d)+1e-12); fvec=(0.8 if scenario==2 else 0.25)*d
    wmean=np.zeros(3)
    if scenario==1 or scenario==7:
        d=np.random.normal(0,1,3); d[2]*=.3; d=d/(norm3(d)+1e-12); wmean=(3.0 if scenario==1 else 1.0)*d
    wind=wmean.copy()
    p,v,_=traj(0.0); p=p+np.random.normal(0,.02,3); v=v+np.random.normal(0,.02,3)
    q=np.array([0.,0.,0.,1.]); w=np.zeros(3); hover=np.sqrt((M_NOM*G/4)/KETA); omega=np.ones(4)*hover
    prev_vm=v.copy(); a_lp=np.zeros(3); prev_applied=np.zeros(3); hist=np.zeros((NWIN,9)); hcount=0; d1lp=np.zeros(9); d2lp=np.zeros(9)
    cmdq=np.zeros((5,3)); qcount=0
    sse=0.; maxerr=0.; sehead=0.; seff=0.; sres=0.
    for k in range(steps):
        t=k*dt; mact=mass+(payload if (payload>0 and t>=ptoggle) else 0)
        Fext=fvec if ((scenario==2 or scenario==7) and t>=ftoggle and t<fend) else np.zeros(3)
        if norm3(wmean)>0:
            theta=1.5; sig=.45 if scenario==1 else .25
            wind += theta*(wmean-wind)*dt + sig*np.sqrt(dt)*np.random.normal(0,1,3)
        else: wind*=.95
        pm=p+np.random.normal(0,.002,3); vm=v+np.random.normal(0,.005,3)
        vdot=(vm-prev_vm)/dt if k>0 else np.zeros(3); innov=vdot-prev_applied; a_lp=.85*a_lp+.15*innov; prev_vm=vm.copy()
        pr,vr,ar=traj(t); e=pm-pr; ev=vm-vr; mvec=np.empty(9)
        for j in range(3): mvec[j]=e[j]; mvec[3+j]=ev[j]; mvec[6+j]=a_lp[j]
        # shift history
        if hcount<NWIN:
            hist[hcount]=mvec; hcount+=1
        else:
            for hh in range(NWIN-1): hist[hh]=hist[hh+1]
            hist[NWIN-1]=mvec
        d1=np.zeros(9); d2=np.zeros(9)
        if hcount==NWIN:
            for j in range(9):
                s1=0.; s2=0.
                for hh in range(NWIN): s1+=C1[hh]*hist[hh,j]; s2+=C2[hh]*hist[hh,j]
                d1[j]=s1; d2[j]=s2
            d1lp=.8*d1lp+.2*d1; d2lp=.8*d2lp+.2*d2
        planes=np.empty((3,3,3))
        for a in range(3):
            for b in range(3):
                j=a*3+b; planes[0,a,b]=mvec[j]; planes[1,a,b]=.1*d1lp[j]; planes[2,a,b]=.01*d2lp[j]
        res=residual(planes,resmode,t)
        if ctrl==0: ades=ar-(KX/M_NOM)*e-(KV/M_NOM)*ev+res
        elif ctrl==1: ades=ar-np.array([4.,4.,7.])*e-np.array([3.,3.,5.])*ev+res
        else: ades=ar-np.array([6.,6.,9.])*e-np.array([4.5,4.5,6.5])*ev+res
        for j in range(3):
            if ades[j]>5: ades[j]=5
            if ades[j]<-5: ades[j]=-5
        # command delay scenario (40 ms)
        cmdq[qcount%5]=ades; qcount+=1; delay=4 if scenario==4 else 0
        idx=(qcount-1-delay)%5 if qcount>delay else 0; applied=cmdq[idx].copy(); prev_applied=applied.copy()
        R=q_to_R(q); Fdes=M_NOM*(applied+np.array([0.,0.,G])); Rd=desired_R(Fdes)
        S=Rd.T@R-R.T@Rd; eR=.5*np.array([S[2,1],S[0,2],S[1,0]])
        tcmd=-KR*eR-KW*w; thrust=Fdes@(R@np.array([0.,0.,1.])); thrust=max(0.,thrust)
        wrench=np.array([thrust,tcmd[0],tcmd[1],tcmd[2]]); fcmd=Ainv@wrench
        for i in range(4):
            if fcmd[i]<0: fcmd[i]=0
            if fcmd[i]>KETA*WMAX*WMAX: fcmd[i]=KETA*WMAX*WMAX
        ocmd=np.sqrt(fcmd/KETA); alpha=1-np.exp(-dt/TAU_M); omega=omega+alpha*(ocmd-omega)
        fi=eff*KETA*omega*omega; Tact=np.sum(fi); torque=Aalloc[1:,:]@fi; air=R.T@(v-wind); drag=-np.array([KD,KD,KZ])*np.sum(omega)*air
        Fw=R@(np.array([0.,0.,Tact])+drag)+np.array([0.,0.,-mact*G])+Fext; acc=Fw/mact; v=v+acc*dt; p=p+v*dt
        Iw=I*w; cr=cross(w,Iw); wdot=np.array([(torque[0]-cr[0])/I[0],(torque[1]-cr[1])/I[1],(torque[2]-cr[2])/I[2]]); w=w+wdot*dt; q=q_step(q,w,dt)
        err=norm3(p-pr); sse+=err*err; maxerr=max(maxerr,err); R=q_to_R(q); yaw=np.arctan2(R[1,0],R[0,0]); sehead+=abs(yaw); seff+=np.mean((omega-hover)**2); sres+=np.mean(res*res)
        if not np.isfinite(err) or norm3(p)>30: return 30.,30.,0.,180.,1e8,np.sqrt(sres/(k+1))
    return np.sqrt(sse/steps),maxerr,1.0 if maxerr<5 else 0.0,sehead/steps*180/np.pi,seff/steps,np.sqrt(sres/steps)

SCENARIOS=['nominal','wind3','force_step','model20','latency40','payload50','rotoreff30','compound']
CONTROLLERS=[('Geo',0,0),('LQR-outer',1,0),('LMPC-H40',2,0),('Geo+MLP-D2',0,1),('Geo+CeNN-V3',0,5),('Geo+CeNN-D2',0,2),('Geo+AE-CeNN',0,3),('Geo+AE-CeNN dropout',0,4)]

def main(nseeds=10):
    # warmup compile
    print('JIT warmup...'); print(run_episode(0,0,0,0))
    rows=[]; t0=time.time()
    for si,sc in enumerate(SCENARIOS):
        for name,ctrl,resm in CONTROLLERS:
            for seed in range(nseeds):
                vals=run_episode(seed,si,ctrl,resm)
                rows.append((sc,name,seed,*vals))
    df=pd.DataFrame(rows,columns=['Scenario','Controller','SeedIndex','pos_rmse_m','pos_max_m','success','heading_mae_deg','control_effort','residual_rms_ms2'])
    summ=df.groupby(['Scenario','Controller'],sort=False).agg(
        rmse_mean=('pos_rmse_m','mean'),rmse_sd=('pos_rmse_m','std'),max_mean=('pos_max_m','mean'),success_pct=('success',lambda x:100*x.mean()),heading_deg=('heading_mae_deg','mean'),effort=('control_effort','mean'),residual_rms=('residual_rms_ms2','mean')).reset_index()
    df.to_csv(ROOT/'situation_C_trials.csv',index=False); summ.to_csv(ROOT/'situation_C_summary.csv',index=False)
    print(summ[['Scenario','Controller','rmse_mean','success_pct']].to_string(index=False))
    print('runtime',time.time()-t0)

if __name__=='__main__':
    import sys
    main(int(sys.argv[1]) if len(sys.argv)>1 else 10)

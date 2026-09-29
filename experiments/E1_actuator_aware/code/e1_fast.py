import numpy as np, pandas as pd, math, json, time, hashlib
from pathlib import Path
from scipy.stats import wilcoxon
from numba import njit

OUT=Path('/mnt/data/E1_actuator_aware_results_v2'); OUT.mkdir(exist_ok=True)
DT=0.01; T_FINAL=20.; NS=int(T_FINAL/DT); H=int(0.30/DT); UMAX=5.5; MASS=.5
DRAG=np.array([.005,.005,.01]); TAU_BASE=np.array([.075,.075,.045]); PMAX=np.array([1.4,1.4,1.25]); VMAX=np.array([3.,3.,2.5])
KP=np.array([2.8,2.8,3.5]); KD=np.array([2.3,2.3,2.7]); RB=.30
SCEN_NAMES=['nominal','dryden3','lag2x','rotor_eff','force','compound']
SCENS={
'nominal':(np.zeros(3),np.zeros(3),1.0,np.ones(3),0.0,0),
'dryden3':(np.array([2.,.5,0.]),np.array([4.,3.5,2.]),1.0,np.ones(3),0.0,0),
'lag2x':(np.zeros(3),np.zeros(3),2.0,np.ones(3),0.0,0),
'rotor_eff':(np.zeros(3),np.zeros(3),1.25,np.array([.84,.90,.82]),0.0,0),
'force':(np.zeros(3),np.zeros(3),1.0,np.ones(3),0.0,1),
'compound':(np.array([2.5,.5,0.]),np.array([4.5,4.,2.5]),2.0,np.array([.84,.88,.80]),.06,1),
}
CAL=np.arange(10000,10020); PILOT=np.arange(11000,11020); TEST=np.arange(12000,12100); FAULT=np.arange(13000,13100)

# RotorPy-pinned Dryden implementation
class Gust:
    def __init__(self,L,sigma,rng,dt0=.05):
        self.dt0=dt0; V=1.; b=2*np.sqrt(3)*L/V; c=2*L/V
        self.alpha=sigma*np.sqrt(2*L/np.pi/V); self.beta=self.alpha*b; self.delta=2*c; self.gamma=c*c
        self.u1=self.u2=self.y1=self.y2=0.; self.rng=rng
    def run(self,dt):
        C1=1+2*self.delta/dt+4*self.gamma/dt**2; C2=2-8*self.gamma/dt**2; C3=1-2*self.delta/dt+4*self.gamma/dt**2
        C4=self.alpha+2*self.beta/dt; C5=2*self.alpha; C6=self.alpha-2*self.beta/dt
        u=self.rng.uniform(-1,1); y=(C4*u+C5*self.u1+C6*self.u2-C2*self.y1-C3*self.y2)/C1
        self.u2,self.u1=self.u1,u; self.y2,self.y1=self.y1,y; return y
    def get(self,dt):
        if dt<=self.dt0:return self.run(dt)
        t=0;y=0
        while t<dt-1e-12:
            inc=min(self.dt0,dt-t); y=self.run(inc); t+=inc
        return y

def gen_inputs(seed,sn):
    mean,sigma,ts,gain,delay,force=SCENS[sn]; rng=np.random.default_rng(int(seed))
    Lzft=3.281*2.; Lxft=Lzft/((.177+.000823*Lzft)**1.2); L=[Lxft/3.281,Lxft/3.281,Lzft/3.281]
    sub=rng.integers(0,2**32-1,3,dtype=np.uint32); gust=[Gust(L[i],sigma[i],np.random.default_rng(int(sub[i]))) for i in range(3)]
    wind=np.empty((NS,3)); noise=rng.normal(0,.035,(NS,3)); ext=np.zeros((NS,3))
    for k in range(NS):
        wind[k]=mean+np.array([gust[i].get(DT) for i in range(3)])
        t=k*DT
        if force:
            if 7<=t<8: ext[k]=[1.0,-.7,.5]
            elif 13<=t<13.4: ext[k]=[-1.4,.9,-.6]
    return wind,noise,ext,TAU_BASE*ts,gain,int(round(delay/DT))

t=np.arange(NS)*DT; W=np.array([.42,.34,.29]); A=np.array([1.6,1.2,.45]); PH=np.array([0,.6,.2])
REF_X=A*np.sin(t[:,None]*W+PH); REF_X[:,2]+=1.6
REF_V=A*W*np.cos(t[:,None]*W+PH); REF_A=-A*W*W*np.sin(t[:,None]*W+PH)

@njit
def intersect(lo,hi,alpha,beta,L,U):
    if abs(beta)<1e-12:
        if alpha<L or alpha>U:return 1.,-1.
        return lo,hi
    a=(L-alpha)/beta; b=(U-alpha)/beta
    if a>b:a,b=b,a
    if a>lo:lo=a
    if b<hi:hi=b
    return lo,hi

@njit
def filt_lm(ep,ev,ar,up,dmax):
    out=np.empty(3); active=False; infeas=False
    for ax in range(3):
        lo=-UMAX;hi=UMAX
        for n in range(1,H+1):
            tt=n*DT; dbv=tt*dmax[ax]; av=ev[ax]-tt*ar[ax]
            lo,hi=intersect(lo,hi,av,tt,-VMAX[ax]+dbv,VMAX[ax]-dbv)
            if lo>hi:break
            dbp=.5*tt*tt*dmax[ax]; ap=ep[ax]+tt*ev[ax]-.5*tt*tt*ar[ax]
            lo,hi=intersect(lo,hi,ap,.5*tt*tt,-PMAX[ax]+dbp,PMAX[ax]-dbp)
            if lo>hi:break
        if lo>hi:
            infeas=True; active=True; val=ar[ax]-KP[ax]*ep[ax]-KD[ax]*ev[ax]
            out[ax]=min(UMAX,max(-UMAX,val))
        else:
            val=up[ax]
            if val<lo: val=lo; active=True
            elif val>hi: val=hi; active=True
            out[ax]=val
    return out,active,infeas

@njit
def filt_aa(ep,ev,aa,ar,up,tauh,dmax):
    out=np.empty(3); active=False; infeas=False
    for ax in range(3):
        lo=-UMAX;hi=UMAX
        # affine robust state: base + c*u, radius
        bp=ep[ax];bv=ev[ax];ba=aa[ax]; cp=0.;cv=0.;ca=0.; rp=0.;rv=0.;ra=0.
        for n in range(H):
            nbp=bp+DT*bv; nbv=bv+DT*(ba-ar[ax]); nba=ba*(1-DT/tauh[ax])
            ncp=cp+DT*cv; ncv=cv+DT*ca; nca=ca*(1-DT/tauh[ax])+DT/tauh[ax]
            nrp=rp+DT*rv; nrv=rv+DT*ra+DT*dmax[ax]; nra=ra*(1-DT/tauh[ax])
            bp,bv,ba=nbp,nbv,nba;cp,cv,ca=ncp,ncv,nca;rp,rv,ra=nrp,nrv,nra
            lo,hi=intersect(lo,hi,bp,cp,-PMAX[ax]+rp,PMAX[ax]-rp)
            if lo>hi:break
            lo,hi=intersect(lo,hi,bv,cv,-VMAX[ax]+rv,VMAX[ax]-rv)
            if lo>hi:break
        if lo>hi:
            infeas=True; active=True; val=ar[ax]-KP[ax]*ep[ax]-KD[ax]*ev[ax]
            out[ax]=min(UMAX,max(-UMAX,val))
        else:
            val=up[ax]
            if val<lo:val=lo;active=True
            elif val>hi:val=hi;active=True
            out[ax]=val
    return out,active,infeas

@njit
def run_one(seed,wind,noise,ext,tau,gain,delay_steps,filter_id,dml,dea,tauh,fault_id,collect_obs=False):
    np.random.seed(seed)
    ep=np.random.normal(0,.04,3);ev=np.random.normal(0,.03,3);aa=np.zeros(3)
    queue=np.zeros((delay_steps+1,3)); qidx=0
    se=0.;mx=0.;gt1=0;viol=0;act=0;inf=0;du=0.;prev=np.zeros(3)
    # sums for calibration regressions/quantile cannot be done here; optional arrays not returned
    for k in range(NS):
        ar=REF_A[k]
        res=RB*np.tanh(-1.2*ep-.35*ev)
        up=ar-KP*ep-KD*ev+res
        for j in range(3):
            if up[j]>UMAX:up[j]=UMAX
            if up[j]<-UMAX:up[j]=-UMAX
        tt=k*DT
        if fault_id==1 and 10<=tt<10.30:
            vv=np.array([4.5,-4.,3.8]); up=up+vv
            for j in range(3): up[j]=min(UMAX,max(-UMAX,up[j]))
        elif fault_id==2 and 10<=tt<10.80:
            up=np.array([UMAX,-UMAX,UMAX])
        if filter_id==0: uc=up;ac=False;ii=False
        elif filter_id==1: uc,ac,ii=filt_lm(ep,ev,ar,up,dml)
        else: uc,ac,ii=filt_aa(ep,ev,aa,ar,up,tauh,dea)
        # queue
        delayed=queue[qidx].copy(); queue[qidx]=uc; qidx=(qidx+1)%(delay_steps+1)
        ureal=gain*delayed
        adot=(ureal-aa)/tau
        aan=aa+DT*adot
        actual_v=REF_V[k]+ev; rel=actual_v-wind[k]; nr=np.sqrt(np.sum(rel*rel))
        drag=-(nr*DRAG*rel)/MASS
        dext=drag+ext[k]+noise[k]
        evdot=aa-ar+dext
        evn=ev+DT*evdot; epn=ep+DT*evn
        er=np.sqrt(np.sum(ep*ep));se+=er*er
        if er>mx:mx=er
        if er>1:gt1+=1
        bad=False
        for j in range(3):
            if abs(ep[j])>PMAX[j] or abs(ev[j])>VMAX[j]:bad=True
        if bad:viol+=1
        if ac:act+=1
        if ii:inf+=1
        dd=uc-prev;du+=np.sum(dd*dd);prev=uc
        ep,ev,aa=epn,evn,aan
    return math.sqrt(se/NS),mx,gt1/NS,viol/NS,act/NS,inf/NS,math.sqrt(du/NS)

# separate calibration simulator returning arrays, python/numpy: only 120 runs -> okay
def calibration_data():
    U=[];AA=[];AD=[];AR=[];DE=[];ED=[]
    for sn in ['nominal','dryden3','lag2x','force']:
      for seed in CAL:
        wind,noise,ext,tau,gain,delay=gen_inputs(seed,sn)
        rng=np.random.default_rng(int(seed)); ep=rng.normal(0,.04,3);ev=rng.normal(0,.03,3);aa=np.zeros(3);queue=[np.zeros(3) for _ in range(delay+1)]
        for k in range(NS):
            ar=REF_A[k]; res=RB*np.tanh(-1.2*ep-.35*ev); u=np.clip(ar-KP*ep-KD*ev+res,-UMAX,UMAX)
            delayed=queue.pop(0);queue.append(u.copy());ureal=gain*delayed;adot=(ureal-aa)/tau
            rel=(REF_V[k]+ev)-wind[k];dext=-(np.linalg.norm(rel)*DRAG*rel)/MASS+ext[k]+noise[k];evdot=aa-ar+dext
            U.append(u.copy());AA.append(aa.copy());AD.append(adot.copy());AR.append(ar.copy());DE.append(dext.copy());ED.append(evdot.copy())
            aa=aa+DT*adot;ev=ev+DT*evdot;ep=ep+DT*ev
    U=np.array(U);AA=np.array(AA);AD=np.array(AD);AR=np.array(AR);DE=np.array(DE);ED=np.array(ED)
    tauh=[]
    for ax in range(3):
        z=U[:,ax]-AA[:,ax]; beta=(z@AD[:,ax])/(z@z);tauh.append(1/beta)
    tauh=np.array(tauh); dl=ED-(U-AR)
    return tauh,1.10*np.quantile(abs(dl),.995,axis=0),1.10*np.quantile(abs(DE),.995,axis=0),len(U)

def run_set(seeds, fault_id=0, scenarios=SCEN_NAMES):
    rows=[]
    for sn in scenarios:
      for seed in seeds:
        wind,noise,ext,tau,gain,delay=gen_inputs(seed,sn)
        for fid,name in enumerate(['P','LM','AA']):
            vals=run_one(int(seed),wind,noise,ext,tau,gain,delay,fid,DML,DEA,TAUH,fault_id)
            rows.append((seed,sn,name,['none','impulse','stuck'][fault_id],*vals))
    return rows

def summarize(df):
    return df.groupby(['fault','scenario','filter']).agg(n=('seed','count'),rmse=('rmse','mean'),rmse_sd=('rmse','std'),max_error=('max_error','mean'),gt1=('gt1','mean'),violation=('violation','mean'),active=('active','mean'),infeasible=('infeasible','mean'),du=('du','mean')).reset_index()

def print_pilot(s):
    print(s[['scenario','filter','rmse','violation','active','infeasible','gt1']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))

if __name__=='__main__':
  st=time.time(); TAUH,DML,DEA,NOBS=calibration_data(); print('cal',TAUH,DML,DEA,NOBS,'time',time.time()-st)
  # warm compile
  w,n,e,tau,g,d=gen_inputs(11000,'nominal');print('warm',run_one(11000,w,n,e,tau,g,d,2,DML,DEA,TAUH,0))
  pilot=pd.DataFrame(run_set(PILOT),columns=['seed','scenario','filter','fault','rmse','max_error','gt1','violation','active','infeasible','du'])
  ps=summarize(pilot); print('\nPILOT');print_pilot(ps)
  pilot.to_csv(OUT/'pilot_metrics.csv',index=False); ps.to_csv(OUT/'pilot_summary.csv',index=False)
  json.dump({'tau_hat':TAUH.tolist(),'dmax_lumped':DML.tolist(),'dmax_ext':DEA.tolist(),'nobs':NOBS},open(OUT/'calibration.json','w'),indent=2)
  print('pilot runtime',time.time()-st)

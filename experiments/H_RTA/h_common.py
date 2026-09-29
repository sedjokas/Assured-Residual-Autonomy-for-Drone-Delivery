import os, json, math
import numpy as np, pandas as pd
from scipy.linalg import solve_discrete_are
DT=0.01;FREQ=100;EPISODE_S=12.0;STEPS=int(EPISODE_S/DT);T=np.arange(STEPS)*DT;TAU=0.1;AUTH=0.35;RES_MAX=3.0;UMAX=6.0;WARMUP=101
w=np.array([0.72,0.93,0.51])
PREF=np.stack([1.05*np.sin(w[0]*T),0.85*np.sin(w[1]*T+0.35),1.25+0.38*np.sin(w[2]*T)],1)
VREF=np.stack([1.05*w[0]*np.cos(w[0]*T),0.85*w[1]*np.cos(w[1]*T+0.35),0.38*w[2]*np.cos(w[2]*T)],1)
AREF=np.stack([-1.05*w[0]**2*np.sin(w[0]*T),-0.85*w[1]**2*np.sin(w[1]*T+0.35),-0.38*w[2]**2*np.sin(w[2]*T)],1)
KP=np.array([4.2,4.2,5.0]);KD=np.array([2.9,2.9,3.2])
def geo(p,v,k):return AREF[k]-KP*(p-PREF[k])-KD*(v-VREF[k])

def scenario(name,seeds,intensity=None):
    n=len(seeds);mass=np.ones(n);E=np.repeat(np.eye(3)[None],n,0);drag=np.full((n,3),.035);delay=np.zeros(n,int)
    ext=np.zeros((STEPS,n,3));wind=np.zeros((STEPS,n,3));pn=np.zeros((STEPS,n,3));vn=np.zeros((STEPS,n,3));force_windows=[]
    if name=='force_step':force_windows=[(300,500),(800,1000)]
    if name=='compound':force_windows=[(350,500),(850,1000)]
    for j,s in enumerate(seeds):
        rr=np.random.default_rng(910000+int(s)*131+sum(map(ord,name))*17+(0 if intensity is None else int(100*float(intensity))))
        sp=.0025;sv=.006
        if name.startswith('model'):
            lev=.20 if intensity is None else float(intensity);mass[j]=rr.uniform(1-lev,1+lev);drag[j]*=rr.uniform(1-lev,1+lev);E[j]=np.diag(rr.uniform(1-.9*lev,1+.9*lev,3))
        if name=='payload50':mass[j]=rr.uniform(1.35,1.50);E[j,0,2]=rr.uniform(-.04,.04);E[j,1,2]=rr.uniform(-.04,.04)
        if name=='rotoreff30':
            eff=rr.uniform(.70,1.0,3);eff[rr.integers(0,3)]*=rr.uniform(.68,.82);E[j]=np.diag(eff);C=rr.normal(0,.045,(3,3));np.fill_diagonal(C,0);E[j]+=C;sv=.009
        if name=='latency40':delay[j]=4
        if name=='compound':mass[j]=rr.uniform(1.15,1.30);drag[j]*=rr.uniform(.85,1.25);E[j]=np.diag(rr.uniform(.86,1.05,3));delay[j]=2;sp=.004;sv=.010
        if name in ('wind3','compound','wind_int'):
            mag=(3.0 if name=='wind3' else 1.8) if intensity is None else float(intensity);d1=rr.normal(size=3);d1/=np.linalg.norm(d1)+1e-12;d2=rr.normal(size=3);d2/=np.linalg.norm(d2)+1e-12;x=mag*d1
            for k in range(STEPS):
                mu=mag*(d1 if k<600 else d2);x+=.025*(mu-x)+rr.normal(0,.11 if mag>=2.5 else .08,3);wind[k,j]=x
        if name in ('force_step','compound'):
            base_mag=.80/.826 if name=='force_step' else .45/.826
            for a,b in force_windows:
                d=rr.normal(size=3);d/=np.linalg.norm(d)+1e-12;ext[a:b,j]=base_mag*d
        nr=np.random.default_rng(420000+int(s)*97+sum(map(ord,name))*13);pn[:,j]=nr.normal(0,sp,(STEPS,3));vn[:,j]=nr.normal(0,sv,(STEPS,3))
    return dict(mass=mass,E=E,drag=drag,delay=delay,ext=ext,wind=.20*wind+.022*wind*np.abs(wind),pn=pn,vn=vn,force_windows=force_windows)

xt=np.arange(-8,1)*DT;XX=np.stack([np.ones(9),xt,xt**2],1);PI=np.linalg.pinv(XX);CD1=PI[1];CD2=2*PI[2]
class FeatureState:
    def __init__(self,n):self.n=n;self.m=[];self.d1=np.zeros((n,9));self.d2=np.zeros((n,9));self.ahat=np.zeros((n,3));self.pv=None;self.pu=np.zeros((n,3))
    def feature(self,p,v,k,pn,vn):
        pm=p+pn;vm=v+vn;raw=np.zeros_like(v) if self.pv is None else (vm-self.pv)/DT-self.pu;self.ahat=.92*self.ahat+.08*np.clip(raw,-8,8)
        m=np.concatenate([pm-PREF[k],vm-VREF[k],self.ahat],1);self.m.append(m.copy())
        if len(self.m)>9:self.m.pop(0)
        ar=np.stack(([self.m[0]]*(9-len(self.m)))+self.m,0);d1=np.tensordot(CD1,ar,(0,0));d2=np.tensordot(CD2,ar,(0,0));self.d1=.72*self.d1+.28*d1;self.d2=.82*self.d2+.18*d2;self.pv=vm.copy()
        return np.stack([m,TAU*self.d1,TAU*TAU*self.d2],-1)
    def cmd(self,u):self.pu=u.copy()

P=np.load('/mnt/data/situation_H_final/H_frozen_packs.npz')
tm=P['tm'];ts=P['ts'];zmu=P['zmu'];zsig=P['zsig'];det_med=P['det_med'];det_mad=P['det_mad']
EC=[(P['EC_w1'],P['EC_b1']),(P['EC_w2'],P['EC_b2'])];CC=[P['CC_A'],P['CC_B'],float(P['CC_b'][0]),P['CC_g'],P['CC_ob']]
PG={'h1':(P['PG_h1_w'],P['PG_h1_b']),'h2':(P['PG_h2_w'],P['PG_h2_b']),'gate':(P['PG_gate_w'],P['PG_gate_b']),'gmax':float(P['PG_gmax'][0]),'thr':float(P['PG_thr'][0]),'temp':float(P['PG_temp'][0])}
TRACK_ENV=float(P['TRACK_ENV'][0]);HIST_ABS_ENV=float(P['HIST_ABS_ENV'][0]);DELTA_AUTH_MAX=float(P['DELTA_AUTH_MAX'][0]);ACT_MARGIN=float(P['ACT_MARGIN'][0]);PERSIST_N=3

def enc(par,x):
    (w1,b1),(w2,b2)=par;return np.tanh(np.tanh(x@w1.T+b1)@w2.T+b2)
def cf(par,z):
    A9,B9,b,g,ob=par;n=len(z);inp=np.transpose(z,(0,2,1));ff=np.zeros((n,9))
    for c in range(3):ff+=inp[:,c]@B9[c].T
    x=np.zeros_like(ff)
    for _ in range(4):x+=.32*(-x+np.tanh(x)@A9.T+ff+b)
    return np.tanh(np.tanh(x.reshape(n,3,3))[:,2,:]*g+ob)*RES_MAX
def cur_lat(tn):return (enc(EC,tn.reshape(-1,3)).reshape(len(tn),9,3)-zmu)/zsig
def cur_pred(tn):return AUTH*cf(CC,cur_lat(tn))
def detector_raw(tn,lag):
    obs=np.sqrt(np.mean(tn[:,6:9,0]**2,axis=1));dobs=np.sqrt(np.mean(tn[:,6:9,1]**2,axis=1));innov=np.sqrt(np.mean((tn-lag)**2,axis=(1,2)));track=np.sqrt(np.mean(tn[:,0:6,0]**2,axis=1));return np.stack([obs,dobs,innov,track],axis=1)
def detector_score(tn,lag):
    r=detector_raw(tn,lag);z=np.maximum(0,(r-det_med)/det_mad);return np.sqrt(np.mean(z*z,axis=1))
def alpha_from_score(score):
    x=np.clip((score-PG['thr'])/max(PG['temp'],1e-6),-30,30);return 1/(1+np.exp(-x))
def event_pred(tn,lag):
    z=cur_lat(tn);(w1,b1)=PG['h1'];(w2,b2)=PG['h2'];(wg,bg)=PG['gate'];zh=np.tanh(np.tanh(lag.reshape(-1,3)@w1.T+b1)@w2.T+b2).reshape(len(tn),9,3)
    score=detector_score(tn,lag);alpha=alpha_from_score(score);glearn=PG['gmax']/(1+np.exp(-(np.concatenate([z,zh],-1)@wg.T+bg)));g=alpha[:,None,None]*glearn
    return AUTH*cf(CC,z+g*zh),g,alpha,score

def action_scale(basecur,delta):
    mx=np.max(np.abs(delta),axis=1);s=np.minimum(1.0,DELTA_AUTH_MAX/(mx+1e-12));lim=UMAX-ACT_MARGIN
    for j in range(3):
        d=delta[:,j];b=basecur[:,j];pos=d>1e-12;neg=d<-1e-12
        s[pos]=np.minimum(s[pos],np.maximum(0,(lim-b[pos])/(d[pos]+1e-12)));s[neg]=np.minimum(s[neg],np.maximum(0,(-lim-b[neg])/(d[neg]-1e-12)))
    return np.clip(s,0,1)
def make_fault_schedule(name,n,seeds):
    deadline=np.ones((STEPS,n),dtype=bool);corrupt=np.zeros((STEPS,n),dtype=bool);spikes=np.zeros((STEPS,n),dtype=bool)
    if ('deadline_bursts' in name) or ('combined_faults' in name):deadline[400:425]=False;deadline[800:815]=False
    if ('history_corruption' in name) or ('combined_faults' in name):corrupt[450:500]=True;corrupt[850:880]=True
    if ('sensor_spikes' in name) or ('combined_faults' in name):
        for j,s in enumerate(seeds):
            rr=np.random.default_rng(880000+int(s)*31)
            for kk in rr.choice(np.arange(300,1050),3,replace=False):spikes[kk,j]=True
    return deadline,corrupt,spikes
def base_scenario(kind,seeds,intensity=None):
    if kind=='wind_onset':
        sc=scenario('wind_int',seeds,intensity=float(intensity));sc['wind'][:300]=0;sc['wind'][900:]=0;return sc
    if kind=='force_intensity':
        sc=scenario('force_step',seeds);sc['ext']*=float(intensity)/0.8;return sc
    if kind.startswith('wind3_'):return scenario('wind3',seeds)
    if kind.startswith('force_'):return scenario('force_step',seeds)
    if kind.startswith('compound_'):return scenario('compound',seeds)
    if kind in ('nominal','deadline_bursts','history_corruption','sensor_spikes','combined_faults'):return scenario('nominal',seeds)
    if kind=='compound_faults':return scenario('compound',seeds)
    return scenario(kind,seeds)
def simulate_H(kind,seeds,controller='H',intensity=None,collect_trace=False):
    sc=base_scenario(kind,seeds,intensity);n=len(seeds);p=np.repeat(PREF[0][None],n,0);v=np.repeat(VREF[0][None],n,0)
    for j,s in enumerate(seeds):
        rr=np.random.default_rng(990000+int(s)*23+sum(map(ord,kind))*7);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
    deadline_ok,corrupt_mask,spike_mask=make_fault_schedule(kind,n,seeds);fs=FeatureState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];hist=[]
    err=np.zeros((STEPS,n));sat=np.zeros((STEPS,n));gate=np.zeros((STEPS,n));alphaT=np.zeros((STEPS,n));scaleT=np.zeros((STEPS,n));reason=np.zeros((STEPS,n),dtype=np.int16);consecutive=np.zeros(n,dtype=np.int16);prev_delta=np.zeros((n,3))
    for k in range(STEPS):
        pn=sc['pn'][k].copy();vn=sc['vn'][k].copy()
        if spike_mask[k].any():
            for j in np.where(spike_mask[k])[0]:
                rr=np.random.default_rng(991000+int(seeds[j])*13+k);pn[j]+=rr.normal(0,.8,3);vn[j]+=rr.normal(0,1.5,3)
        te=fs.feature(p,v,k,pn,vn);tn=(te-tm)/ts;hist.append(tn.copy());
        if len(hist)>11:hist.pop(0)
        lag=hist[0] if len(hist)<11 else hist[-11].copy()
        if corrupt_mask[k].any():
            for j in np.where(corrupt_mask[k])[0]:
                rr=np.random.default_rng(992000+int(seeds[j])*17+k);lag[j]=20*lag[j]+rr.normal(0,10,lag[j].shape)
        base=geo(p,v,k);rc=cur_pred(tn);re,graw,a,_=event_pred(tn,lag);gm=graw.mean((1,2));delta=re-rc;alphaT[k]=a;gate[k]=gm
        if controller=='Current':r=rc;scale=np.zeros(n)
        elif controller=='G':
            if not deadline_ok[k].all():r=rc+prev_delta.copy()
            else:r=re;prev_delta=delta.copy()
            scale=np.ones(n)
        else:
            detected=a>.5;consecutive=np.where(detected,consecutive+1,0);persist=consecutive>=PERSIST_N;finite=np.isfinite(lag).all((1,2))&np.isfinite(delta).all(1)&np.isfinite(tn).all((1,2));histok=np.max(np.abs(lag),axis=(1,2))<=HIST_ABS_ENV;track=np.sqrt(np.mean(tn[:,0:6,0]**2,axis=1));stateok=track<=TRACK_ENV;dead=deadline_ok[k];proj=action_scale(base+rc,delta);admiss=persist&finite&histok&stateok&dead;scale=np.where(admiss,proj,0.0);reason[k]=np.where(~persist,1,0);reason[k]=np.where(persist&~finite,2,reason[k]);reason[k]=np.where(persist&finite&~histok,3,reason[k]);reason[k]=np.where(persist&finite&histok&~stateok,4,reason[k]);reason[k]=np.where(persist&finite&histok&stateok&~dead,5,reason[k]);reason[k]=np.where(admiss&(proj<.999),6,reason[k]);r=rc+scale[:,None]*delta
        raw=base+r;u=np.clip(raw,-UMAX,UMAX);fs.cmd(u);sat[k]=np.any(np.abs(raw)>=UMAX-1e-12,1);q.append(u.copy());q.pop(0);ud=np.empty_like(u)
        for dd in np.unique(sc['delay']):m=sc['delay']==dd;ud[m]=q[-1-int(dd)][m]
        act+=(DT/.055)*(ud-act);acc=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);v+=DT*acc;p+=DT*v;err[k]=np.linalg.norm(p-PREF[k],axis=1);scaleT[k]=scale
    rows=[];label={'Current':'CurrentOnly-AE-CeNN-494','G':'G-DetectorGate-noRTA','H':'H-RTA-SupervisedGate'}[controller]
    for j,s in enumerate(seeds):
        e=err[WARMUP:,j];rows.append([kind,label,int(s),np.sqrt(np.mean(e*e)),np.quantile(e,.95),np.max(e),100*np.mean(sat[WARMUP:,j]),100*np.mean(e>1),np.mean(gate[WARMUP:,j]),100*np.mean(scaleT[WARMUP:,j]>0),np.mean(scaleT[WARMUP:,j])])
    tr={'error':err,'gate':gate,'alpha':alphaT,'scale':scaleT,'reason':reason,'deadline_ok':deadline_ok,'corrupt_mask':corrupt_mask,'spike_mask':spike_mask} if collect_trace else None
    return rows,tr

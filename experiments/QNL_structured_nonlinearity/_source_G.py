import os, json, time, hashlib, zipfile, platform, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numba import njit
from scipy.linalg import solve_discrete_are
from scipy.stats import wilcoxon, binomtest
import matplotlib.pyplot as plt

OUT='/mnt/data/situation_F_dual_gated_final'
os.makedirs(OUT,exist_ok=True)
SEED=20260911
np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
DT=0.01; FREQ=100; EPISODE_S=12.0; STEPS=int(EPISODE_S/DT)
T=np.arange(STEPS)*DT; TAU=0.1; AUTH=0.35; RES_MAX=3.0; UMAX=6.0
WARMUP=101
SCENARIOS=['nominal','wind3','force_step','model20','latency40','payload50','rotoreff30','compound']
TRAIN_SCENARIOS=['nominal','wind3','force_step','model20']
TEST_SEEDS=np.arange(300,500)
ABL_SEEDS=np.arange(300,400)
TRAIN_SEEDS=np.arange(100,116)

protocol={
'name':'D.2-Confirmatory','status':'confirmatory benchmark-derived simulation, not flight validation and not native full AdaptiveQuadBench/acados execution',
'control_frequency_hz':FREQ,'episode_length_s':EPISODE_S,'main_test_seeds':'300-499 (200 paired per scenario)','secondary_ablation_seeds':'300-399 (100 paired per scenario)','training_seeds':'100-115 across nominal/wind3/force_step/model20 only','history_window_samples':101,'history_window_s':1.0,
'history_features_per_scalar':['current','lag10','lag100','min','Q1','median','Q3','max'],'quantiles':'exact order statistics over all 101 causal samples; Q1/Q2/Q3 are indices 25/50/75','primary_endpoint':'position RMSE after the 1.01 s history warm-up','secondary_endpoints':['p95 tracking error','max tracking error','post-disturbance IAE','recovery time','control effort','action saturation fraction','residual RMS','tracking excursion >1 m','inference latency'],'authority_multiplier':AUTH,'effective_residual_bound_m_s2':AUTH*RES_MAX,
'main_controllers':['Geo','LQR-outer','LMPC-H40-surrogate','Geo+CeNN-D2','C: Geo+AE-CeNN','CapacityMatched-AE-CeNN','D2-History-AE-CeNN','History-MLP'],'secondary_controls':['D2-ShuffledHistory','D2-LagOnly','D2-StatsOnly','D2-History-AE-CeNN dropout'],
'pre_registered_success':{'architecture_confirmation':'D2 has positive paired RMSE improvement CI vs C in at least 4 of 7 stressed scenarios without worse tracking-excursion rate or action saturation','history_attribution':'D2 outperforms both capacity-matched no-history and shuffled-history controls in a majority of stressed regimes; otherwise attribute gain only to the composite architecture, not history itself'}}
json.dump(protocol,open(f'{OUT}/protocol_preregistered.json','w'),indent=2)

w=np.array([0.72,0.93,0.51])
PREF=np.stack([1.05*np.sin(w[0]*T),0.85*np.sin(w[1]*T+0.35),1.25+0.38*np.sin(w[2]*T)],1)
VREF=np.stack([1.05*w[0]*np.cos(w[0]*T),0.85*w[1]*np.cos(w[1]*T+0.35),0.38*w[2]*np.cos(w[2]*T)],1)
AREF=np.stack([-1.05*w[0]**2*np.sin(w[0]*T),-0.85*w[1]**2*np.sin(w[1]*T+0.35),-0.38*w[2]**2*np.sin(w[2]*T)],1)

A1=np.array([[1,DT],[0,1.0]]); B1=np.array([[0.5*DT*DT],[DT]])
A=np.kron(np.eye(3),A1); B=np.zeros((6,3))
for j in range(3): B[2*j:2*j+2,j]=B1[:,0]
Q=np.diag([5,.5,5,.5,6,.6]); R=.18*np.eye(3)
P=solve_discrete_are(A,B,Q,R); KL=np.linalg.solve(R+B.T@P@B,B.T@P@A)
P=Q.copy()
for _ in range(40): KH=np.linalg.solve(R+B.T@P@B,B.T@P@A); P=Q+A.T@P@(A-B@KH)
KP=np.array([4.2,4.2,5.0]); KD=np.array([2.9,2.9,3.2])
def geo(p,v,k): return AREF[k]-KP*(p-PREF[k])-KD*(v-VREF[k])
def gain(p,v,k,K):
    ep=p-PREF[k]; ev=v-VREF[k]; x=np.stack([ep[:,0],ev[:,0],ep[:,1],ev[:,1],ep[:,2],ev[:,2]],1); return AREF[k]-x@K.T

def scenario(name,seeds,intensity=None):
    n=len(seeds); mass=np.ones(n); E=np.repeat(np.eye(3)[None],n,0); drag=np.full((n,3),.035); delay=np.zeros(n,int)
    ext=np.zeros((STEPS,n,3)); wind=np.zeros((STEPS,n,3)); pn=np.zeros((STEPS,n,3)); vn=np.zeros((STEPS,n,3))
    force_windows=[]
    if name=='force_step': force_windows=[(300,500),(800,1000)]
    if name=='compound': force_windows=[(350,500),(850,1000)]
    for j,s in enumerate(seeds):
        rr=np.random.default_rng(910000+int(s)*131+sum(map(ord,name))*17+(0 if intensity is None else int(100*float(intensity))))
        sp=.0025; sv=.006
        if name.startswith('model'):
            lev=.20 if intensity is None else float(intensity); mass[j]=rr.uniform(1-lev,1+lev); drag[j]*=rr.uniform(1-lev,1+lev); E[j]=np.diag(rr.uniform(1-.9*lev,1+.9*lev,3))
        if name=='payload50': mass[j]=rr.uniform(1.35,1.50); E[j,0,2]=rr.uniform(-.04,.04); E[j,1,2]=rr.uniform(-.04,.04)
        if name=='rotoreff30':
            eff=rr.uniform(.70,1.0,3); eff[rr.integers(0,3)]*=rr.uniform(.68,.82); E[j]=np.diag(eff); C=rr.normal(0,.045,(3,3)); np.fill_diagonal(C,0); E[j]+=C; sv=.009
        if name=='latency40': delay[j]=4
        if name=='compound': mass[j]=rr.uniform(1.15,1.30); drag[j]*=rr.uniform(.85,1.25); E[j]=np.diag(rr.uniform(.86,1.05,3)); delay[j]=2; sp=.004; sv=.010
        if name in ('wind3','compound','wind_int'):
            mag=(3.0 if name=='wind3' else 1.8) if intensity is None else float(intensity)
            d1=rr.normal(size=3); d1/=np.linalg.norm(d1)+1e-12; d2=rr.normal(size=3); d2/=np.linalg.norm(d2)+1e-12; x=mag*d1
            for k in range(STEPS):
                mu=mag*(d1 if k<600 else d2); x += .025*(mu-x)+rr.normal(0,.11 if mag>=2.5 else .08,3); wind[k,j]=x
        if name in ('force_step','compound'):
            base_mag=.80/.826 if name=='force_step' else .45/.826
            for a,b in force_windows:
                d=rr.normal(size=3); d/=np.linalg.norm(d)+1e-12; ext[a:b,j]=base_mag*d
        nr=np.random.default_rng(420000+int(s)*97+sum(map(ord,name))*13); pn[:,j]=nr.normal(0,sp,(STEPS,3)); vn[:,j]=nr.normal(0,sv,(STEPS,3))
    return dict(mass=mass,E=E,drag=drag,delay=delay,ext=ext,wind=.20*wind+.022*wind*np.abs(wind),pn=pn,vn=vn,force_windows=force_windows)

xt=np.arange(-8,1)*DT; XX=np.stack([np.ones(9),xt,xt**2],1); PI=np.linalg.pinv(XX); CD1=PI[1]; CD2=2*PI[2]
class FeatureState:
    def __init__(self,n): self.n=n; self.m=[]; self.d1=np.zeros((n,9)); self.d2=np.zeros((n,9)); self.ahat=np.zeros((n,3)); self.pv=None; self.pu=np.zeros((n,3))
    def feature(self,p,v,k,pn,vn):
        pm=p+pn; vm=v+vn; raw=np.zeros_like(v) if self.pv is None else (vm-self.pv)/DT-self.pu; self.ahat=.92*self.ahat+.08*np.clip(raw,-8,8)
        m=np.concatenate([pm-PREF[k],vm-VREF[k],self.ahat],1); self.m.append(m.copy())
        if len(self.m)>9:self.m.pop(0)
        ar=np.stack(([self.m[0]]*(9-len(self.m)))+self.m,0); d1=np.tensordot(CD1,ar,(0,0)); d2=np.tensordot(CD2,ar,(0,0)); self.d1=.72*self.d1+.28*d1; self.d2=.82*self.d2+.18*d2; self.pv=vm.copy()
        return np.stack([m,TAU*self.d1,TAU*TAU*self.d2],-1)
    def cmd(self,u): self.pu=u.copy()

@njit(cache=False)
def _hist_update(ring,sortedv,pos,x):
    M=x.size; out=np.empty((M,8),np.float64); lag10pos=(pos-10)%101; lag100pos=(pos-100)%101
    for m in range(M):
        old=ring[pos,m]; new=x[m]; j=0
        while j<101 and sortedv[m,j]!=old:j+=1
        if j==101:
            j=0; bd=abs(sortedv[m,0]-old)
            for q in range(1,101):
                dd=abs(sortedv[m,q]-old)
                if dd<bd:bd=dd;j=q
        for q in range(j,100):sortedv[m,q]=sortedv[m,q+1]
        ins=0
        while ins<100 and sortedv[m,ins]<=new:ins+=1
        for q in range(100,ins,-1):sortedv[m,q]=sortedv[m,q-1]
        sortedv[m,ins]=new;ring[pos,m]=new
        out[m,0]=new;out[m,1]=ring[lag10pos,m];out[m,2]=ring[lag100pos,m];out[m,3]=sortedv[m,0];out[m,4]=sortedv[m,25];out[m,5]=sortedv[m,50];out[m,6]=sortedv[m,75];out[m,7]=sortedv[m,100]
    return out
class ExactHistory:
    def __init__(self,n):self.n=n;self.ring=None;self.sortedv=None;self.pos=0
    def update(self,tn):
        x=tn.reshape(-1).astype(np.float64)
        if self.ring is None:self.ring=np.repeat(x[None,:],101,axis=0).copy();self.sortedv=np.repeat(x[:,None],101,axis=1).copy();self.pos=0
        meta=_hist_update(self.ring,self.sortedv,self.pos,x);self.pos=(self.pos+1)%101
        return meta.reshape(self.n,9,3,8).reshape(self.n,9,24)

class AE(nn.Module):
    def __init__(self,di,h,z):super().__init__();self.e1=nn.Linear(di,h);self.e2=nn.Linear(h,z);self.d1=nn.Linear(z,h);self.d2=nn.Linear(h,di)
    def encode(self,x):return torch.tanh(self.e2(torch.tanh(self.e1(x))))
    def forward(self,x):return self.d2(torch.tanh(self.d1(self.encode(x))))
class CeNN(nn.Module):
    def __init__(self):super().__init__();self.A=nn.Parameter(.05*torch.randn(3,3));self.B=nn.Parameter(.08*torch.randn(3,3,3));self.b=nn.Parameter(torch.zeros(1));self.g=nn.Parameter(torch.ones(3));self.ob=nn.Parameter(torch.zeros(3))
    def conv(self,x,k):
        y=torch.zeros_like(x);xp=torch.nn.functional.pad(x,(1,1,1,1))
        for i in range(3):
            for j in range(3):y+=k[i,j]*xp[:,i:i+3,j:j+3]
        return y
    def forward(self,inp):
        x=torch.zeros((len(inp),3,3),device=inp.device);ff=torch.zeros_like(x)
        for c in range(3):ff+=self.conv(inp[:,c],self.B[c])
        for _ in range(4):x=x+.32*(-x+self.conv(torch.tanh(x),self.A)+ff+self.b)
        return torch.tanh(torch.tanh(x)[:,2,:]*self.g+self.ob)
class HistMLP(nn.Module):
    def __init__(self):super().__init__();self.a=nn.Linear(216,4);self.b=nn.Linear(4,12);self.c=nn.Linear(12,3)
    def forward(self,x):return torch.tanh(self.c(torch.tanh(self.b(torch.tanh(self.a(x))))))

def train_ae(X,di,h,z,seed,epochs=5):
    torch.manual_seed(seed);m=AE(di,h,z);opt=torch.optim.Adam(m.parameters(),lr=.003);X=torch.tensor(X,dtype=torch.float32);N=len(X);g=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        ix=torch.randperm(N,generator=g)
        for st in range(0,N,4096):
            q=ix[st:st+4096];clean=X[q];noise=clean+.05*torch.randn(clean.shape,generator=g);loss=((m(noise)-clean)**2).mean();opt.zero_grad();loss.backward();opt.step()
    return m.eval()
def train_pred(m,X,Y,seed,epochs=6):
    torch.manual_seed(seed);opt=torch.optim.Adam(m.parameters(),lr=.003);X=torch.tensor(X,dtype=torch.float32);Y=torch.tensor(Y/RES_MAX,dtype=torch.float32);N=len(X);g=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        ix=torch.randperm(N,generator=g)
        for st in range(0,N,4096):
            q=ix[st:st+4096];loss=((m(X[q])-Y[q])**2).mean();opt.zero_grad();loss.backward();opt.step()
    return m.eval()
def apply_t(m,x):
    with torch.no_grad():return m(torch.tensor(x,dtype=torch.float32)).numpy()
def enc_t(m,x):
    with torch.no_grad():return m.encode(torch.tensor(x,dtype=torch.float32)).numpy()
def latt(x):return np.transpose(x,(0,2,1)).reshape(-1,3,3,3)

def collect(name,seeds):
    sc=scenario(name,seeds);n=len(seeds);p=np.repeat(PREF[0][None],n,0);v=np.repeat(VREF[0][None],n,0);fs=FeatureState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];Ts=[];Ys=[]
    for k in range(STEPS):
        temp=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k]);base=geo(p,v,k);u=np.clip(base,-UMAX,UMAX);fs.cmd(u);q.append(u.copy());q.pop(0);ud=np.array([q[-1-int(d)][j] for j,d in enumerate(sc['delay'])]);act+=(DT/.055)*(ud-act)
        other=sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);ctrl=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None];a=ctrl+other;rhs=base-other
        ideal=np.array([np.linalg.solve(sc['E'][j]/sc['mass'][j],rhs[j])-base[j] for j in range(n)]);ideal=np.clip(ideal,-RES_MAX,RES_MAX);Ts.append(temp);Ys.append(ideal);v+=DT*a;p+=DT*v
    return np.stack(Ts),np.stack(Ys)




# SITUATION G — disturbance-detected sparse gating of a protected current latent
import copy, math
OUT='/mnt/data/situation_G_event_gate';os.makedirs(OUT,exist_ok=True)
VAL_SEEDS_G=np.arange(2400,2440)
TEST_SEEDS_G=np.arange(3200,3300)
EVENT_SEEDS=np.arange(3200,3250)
DETECT_Q_CANDIDATES=[0.99,0.995]
GMAX_CANDIDATES=[0.35,0.55]
TEMP_FRAC=0.35
GATE_REG=0.0008

protocol_G={
'name':'Situation G — disturbance-detected sparse latent-history gate',
'status':'targeted confirmatory benchmark-derived simulation',
'control_frequency_hz':FREQ,'episode_length_s':EPISODE_S,
'training_seeds':'100-115 across nominal/wind3/force_step/model20',
'validation_seeds':'2400-2439, hyperparameter selection only',
'test_seeds':'3200-3299, 100 paired seeds per scenario, untouched by A-F',
'event_analysis_seeds':'3200-3249',
'architecture':'frozen exact current-only AE-CeNN + separate 100-ms history encoder + nominal-calibrated causal anomaly detector + bounded multiplicative gate',
'fusion':'z_fused = z_current + alpha(detector) * g_learned(z_current,z_history) * z_history',
'detector_features':['RMS normalized disturbance-observer level','RMS normalized disturbance-observer derivative','RMS current-vs-100ms-lag innovation','RMS normalized position/velocity tracking state'],
'detector_calibration':'robust median/MAD on nominal training samples only; threshold selected from nominal-score quantiles before final test',
'quantile_candidates':DETECT_Q_CANDIDATES,'gmax_candidates':GMAX_CANDIDATES,
'primary_endpoint':'position RMSE after 1.01 s warm-up',
'gate_behavior_endpoint':'nominal mean gate < 0.01 and stressed mean gate >= 3x nominal',
'pre_registered_success':{
 'control_benefit':'positive paired RMSE CI vs current-only in at least 4/7 stressed regimes; nominal degradation <0.2%',
 'detector_specificity':'positive paired RMSE CI vs detector-shuffled control in at least 4/7 stressed regimes',
 'gate_separation':'nominal mean gate <0.01 and average stressed gate >3x nominal',
 'force_step_response':'median causal detector latency <=0.20 s for the two external-force onsets'
},
'claim_boundary':'benchmark-derived simulation only; no native full AdaptiveQuadBench/acados, HIL, flight-validation, or generic CeNN-superiority claim'
}
json.dump(protocol_G,open(f'{OUT}/protocol_G_preregistered.json','w'),indent=2)

t0=time.time();np.random.seed(SEED+7000);torch.manual_seed(SEED+7000);torch.set_num_threads(1)
print('collect training data',flush=True)
# Collect the same frozen training corpus as E/F, plus nominal-only data for detector calibration.
TT=[];YY=[]
for scn in TRAIN_SCENARIOS:
    t,y=collect(scn,TRAIN_SEEDS);TT.append(t);YY.append(y)
TT=np.concatenate(TT,1);YY=np.concatenate(YY,1);tm=TT.reshape(-1,3).mean(0);ts=TT.reshape(-1,3).std(0)+1e-6;TTn=(TT-tm)/ts
Tnom,_=collect('nominal',TRAIN_SEEDS);Tnomn=(Tnom-tm)/ts
Fc=[];Fl=[];Y=[]
for k in range(WARMUP,STEPS,2):
    Fc.append(TTn[k]);Fl.append(TTn[k-10]);Y.append(YY[k])
Fc=np.concatenate(Fc);Fl=np.concatenate(Fl);Y=np.concatenate(Y)
rz=np.random.default_rng(SEED+71);ix=rz.choice(len(Fc),min(24000,len(Fc)),replace=False);Fc=Fc[ix];Fl=Fl[ix];Y=Y[ix]

# Exact strong current-only path, deterministic Situation-E/F recipe.
def trainE(F,di,h,a,b):
    cells=F.reshape(-1,di);rr=np.random.default_rng(a);cells=cells[rr.choice(len(cells),min(50000,len(cells)),replace=False)]
    ae=train_ae(cells,di,h,3,a,4);z=enc_t(ae,F.reshape(-1,di)).reshape(len(F),9,3);zm=z.reshape(-1,3).mean(0);zs=z.reshape(-1,3).std(0)+1e-6
    ce=train_pred(CeNN(),latt((z-zm)/zs),Y,b,5);return ae,zm,zs,ce
print('train frozen current path',flush=True)
ae_cur,zmu,zsig,cenn_cur=trainE(Fc,3,64,11,12)
zcur=enc_t(ae_cur,Fc.reshape(-1,3)).reshape(len(Fc),9,3);zcur_n=(zcur-zmu)/zsig

# Pretrain a small history encoder as in Situation F.
hr=np.random.default_rng(105);hc=Fl.reshape(-1,3);hc=hc[hr.choice(len(hc),min(50000,len(hc)),replace=False)]
ae_hist=train_ae(hc,3,8,3,105,4)

# Detector construction: all computations are causal and use current/100ms-lag information already available.
def detector_raw(tn,lag):
    obs=np.sqrt(np.mean(tn[:,6:9,0]**2,axis=1))
    dobs=np.sqrt(np.mean(tn[:,6:9,1]**2,axis=1))
    innov=np.sqrt(np.mean((tn-lag)**2,axis=(1,2)))
    track=np.sqrt(np.mean(tn[:,0:6,0]**2,axis=1))
    return np.stack([obs,dobs,innov,track],axis=1)

# Nominal robust calibration from all valid causal samples.
Dnom=[]
for k in range(WARMUP,STEPS,2):Dnom.append(detector_raw(Tnomn[k],Tnomn[k-10]))
Dnom=np.concatenate(Dnom,0)
det_med=np.median(Dnom,axis=0);det_mad=np.median(np.abs(Dnom-det_med),axis=0)*1.4826+1e-6

def detector_score(tn,lag):
    r=detector_raw(tn,lag);z=np.maximum(0,(r-det_med)/det_mad);return np.sqrt(np.mean(z*z,axis=1))
Snom=np.concatenate([detector_score(Tnomn[k],Tnomn[k-10]) for k in range(WARMUP,STEPS,2)])

class EventGatedSide(nn.Module):
    def __init__(self,cenn,pre,gmax):
        super().__init__();self.h1=nn.Linear(3,8);self.h2=nn.Linear(8,3)
        with torch.no_grad():
            self.h1.weight.copy_(pre.e1.weight);self.h1.bias.copy_(pre.e1.bias);self.h2.weight.copy_(pre.e2.weight);self.h2.bias.copy_(pre.e2.bias)
        self.gate=nn.Linear(6,3);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,0.0)
        self.gmax=float(gmax);self.cenn=copy.deepcopy(cenn).eval()
        for p in self.cenn.parameters():p.requires_grad_(False)
    def forward(self,zc,aux,alpha,ret=False):
        zh=torch.tanh(self.h2(torch.tanh(self.h1(aux))))
        glearn=self.gmax*torch.sigmoid(self.gate(torch.cat([zc,zh],-1)))
        g=alpha[:,None,None]*glearn
        zf=zc+g*zh;o=self.cenn(zf.permute(0,2,1).reshape(-1,3,3,3))
        return (o,g) if ret else o

def alpha_from_score(score,threshold,temp):
    x=np.clip((score-threshold)/max(temp,1e-6),-30,30);return 1/(1+np.exp(-x))

def train_event_side(gmax,q,seed,epochs=8):
    thr=float(np.quantile(Snom,q));q95=float(np.quantile(Snom,0.95));temp=max(0.10,(thr-q95)*TEMP_FRAC)
    scr=detector_score(Fc,Fl)
    alp=alpha_from_score(scr,thr,temp).astype(np.float32)
    torch.manual_seed(seed);m=EventGatedSide(cenn_cur,ae_hist,gmax)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.0025)
    Z=torch.tensor(zcur_n,dtype=torch.float32);A=torch.tensor(Fl,dtype=torch.float32);AL=torch.tensor(alp,dtype=torch.float32);YT=torch.tensor(Y/RES_MAX,dtype=torch.float32);N=len(Z);gg=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order=torch.randperm(N,generator=gg)
        for st in range(0,N,4096):
            ix=order[st:st+4096];o,g=m(Z[ix],A[ix],AL[ix],True)
            loss=((o-YT[ix])**2).mean()+GATE_REG*(g*g).mean();opt.zero_grad();loss.backward();opt.step()
    return m.eval(),thr,temp

# Fast NumPy inference helpers.
def lin(l):return l.weight.detach().numpy(),l.bias.detach().numpy()
def epack(ae):return [lin(ae.e1),lin(ae.e2)]
def enc(par,x):
    (w1,b1),(w2,b2)=par;return np.tanh(np.tanh(x@w1.T+b1)@w2.T+b2)
def km(k):
    M=np.zeros((9,9))
    for oi in range(3):
      for oj in range(3):
       o=3*oi+oj
       for a in range(3):
        for b in range(3):
         ii=oi+a-1;jj=oj+b-1
         if 0<=ii<3 and 0<=jj<3:M[o,3*ii+jj]+=k[a,b]
    return M
def cp(c):return [km(c.A.detach().numpy()),np.stack([km(c.B.detach().numpy()[j]) for j in range(3)]),float(c.b.detach()),c.g.detach().numpy(),c.ob.detach().numpy()]
def cf(par,z):
    A9,B9,b,g,ob=par;n=len(z);inp=np.transpose(z,(0,2,1));ff=np.zeros((n,9))
    for c in range(3):ff+=inp[:,c]@B9[c].T
    x=np.zeros_like(ff)
    for _ in range(4):x+=.32*(-x+np.tanh(x)@A9.T+ff+b)
    return np.tanh(np.tanh(x.reshape(n,3,3))[:,2,:]*g+ob)*RES_MAX
EC=epack(ae_cur);CC=cp(cenn_cur)
def cur_lat(tn):return (enc(EC,tn.reshape(-1,3)).reshape(len(tn),9,3)-zmu)/zsig
def cur_pred(tn):return AUTH*cf(CC,cur_lat(tn))
def spack(m,thr,temp):return {'h1':lin(m.h1),'h2':lin(m.h2),'gate':lin(m.gate),'gmax':m.gmax,'thr':thr,'temp':temp}
def event_pred(p,tn,lag,score_override=None):
    z=cur_lat(tn);(w1,b1)=p['h1'];(w2,b2)=p['h2'];(wg,bg)=p['gate'];zh=np.tanh(np.tanh(lag.reshape(-1,3)@w1.T+b1)@w2.T+b2).reshape(len(tn),9,3)
    score=detector_score(tn,lag) if score_override is None else score_override
    alpha=alpha_from_score(score,p['thr'],p['temp'])
    glearn=p['gmax']/(1+np.exp(-(np.concatenate([z,zh],-1)@wg.T+bg)))
    g=alpha[:,None,None]*glearn
    return AUTH*cf(CC,z+g*zh),g,alpha,score

# A matching always-on side branch control with the same trained weights but alpha == 1.
def always_pred(p,tn,lag):
    z=cur_lat(tn);(w1,b1)=p['h1'];(w2,b2)=p['h2'];(wg,bg)=p['gate'];zh=np.tanh(np.tanh(lag.reshape(-1,3)@w1.T+b1)@w2.T+b2).reshape(len(tn),9,3)
    glearn=p['gmax']/(1+np.exp(-(np.concatenate([z,zh],-1)@wg.T+bg)));g=glearn
    return AUTH*cf(CC,z+g*zh),g

def simulate(name,seeds,kind,pack=None,shuffle_detector=False,collect_trace=False):
    sc=scenario(name,seeds);n=len(seeds);p=np.repeat(PREF[0][None],n,0);v=np.repeat(VREF[0][None],n,0)
    for j,s in enumerate(seeds):rr=np.random.default_rng(810000+int(s)*19+sum(map(ord,name))*5);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
    fs=FeatureState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];hist=[];err=np.zeros((STEPS,n));eff=np.zeros((STEPS,n));sat=np.zeros((STEPS,n));res=np.zeros((STEPS,n));gt=np.zeros((STEPS,n));al=np.zeros((STEPS,n));scores=np.zeros((STEPS,n));perm=np.random.default_rng(277000+sum(map(ord,name))).permutation(n)
    for k in range(STEPS):
        te=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k]);tn=(te-tm)/ts;hist.append(tn.copy());
        if len(hist)>11:hist.pop(0)
        lag=hist[0] if len(hist)<11 else hist[-11]
        if kind=='LQR':base=gain(p,v,k,KL);r=np.zeros_like(base);gm=np.zeros(n);aa=np.zeros(n);ss=np.zeros(n)
        else:
            base=geo(p,v,k)
            if kind=='Current':r=cur_pred(tn);gm=np.zeros(n);aa=np.zeros(n);ss=np.zeros(n)
            elif kind in ('Event','DetectorShuffle'):
                score=detector_score(tn,lag);sover=score[perm] if kind=='DetectorShuffle' else score
                r,g,aa,ss=event_pred(pack,tn,lag,sover);gm=g.mean((1,2))
            elif kind=='AlwaysOpen':r,g=always_pred(pack,tn,lag);gm=g.mean((1,2));aa=np.ones(n);ss=detector_score(tn,lag)
        raw=base+r;u=np.clip(raw,-UMAX,UMAX);fs.cmd(u);sat[k]=np.any(np.abs(raw)>=UMAX-1e-12,1);q.append(u.copy());q.pop(0);ud=np.empty_like(u)
        for dd in np.unique(sc['delay']):m=sc['delay']==dd;ud[m]=q[-1-int(dd)][m]
        act+=(DT/.055)*(ud-act);a=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);v+=DT*a;p+=DT*v;err[k]=np.linalg.norm(p-PREF[k],axis=1);eff[k]=np.sum(u*u,1);res[k]=np.linalg.norm(r,axis=1);gt[k]=gm;al[k]=aa;scores[k]=ss
    labels={'LQR':'LQR-outer','Current':'CurrentOnly-AE-CeNN-494','Event':'EventDetected-GatedHistory','DetectorShuffle':'DetectorShuffled-GatedHistory','AlwaysOpen':'AlwaysOpen-HistorySide'};lab=labels[kind];rows=[]
    for j,s in enumerate(seeds):
        e=err[WARMUP:,j];rows.append([name,lab,int(s),np.sqrt(np.mean(e*e)),np.quantile(e,.95),np.max(e),np.mean(eff[WARMUP:,j]),100*np.mean(sat[WARMUP:,j]),np.sqrt(np.mean(res[WARMUP:,j]**2)),np.mean(gt[WARMUP:,j]),np.quantile(gt[WARMUP:,j],.95),np.mean(al[WARMUP:,j]),np.quantile(al[WARMUP:,j],.95)])
    trace=None
    if collect_trace:trace={'gate':gt,'alpha':al,'score':scores,'error':err}
    return rows,trace

# Hyperparameter selection using validation seeds only.
print('validate detector threshold and gate bound',flush=True)
val=[];models={};val_scen=['nominal','wind3','force_step','model20','compound']
for qi,qdet in enumerate(DETECT_Q_CANDIDATES):
  for gi,gmax in enumerate(GMAX_CANDIDATES):
    m,thr,temp=train_event_side(gmax,qdet,700+qi*10+gi,5);pk=spack(m,thr,temp);models[(qdet,gmax)]=(m,pk)
    rr=[]
    for scn in val_scen:
        r,_=simulate(scn,VAL_SEEDS_G,'Current');rr+=r;r,_=simulate(scn,VAL_SEEDS_G,'Event',pk);rr+=r
    d=pd.DataFrame(rr,columns=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','control_effort','saturation_pct','residual_rms_ms2','gate_mean','gate_p95','alpha_mean','alpha_p95'])
    pv=d.pivot_table(index=['Scenario','SeedIndex'],columns='Controller',values='rmse_m');imp={};gmeans={}
    for scn in val_scen:
        qv=pv.loc[scn];imp[scn]=float((100*(qv['CurrentOnly-AE-CeNN-494']-qv['EventDetected-GatedHistory'])/qv['CurrentOnly-AE-CeNN-494']).mean());gmeans[scn]=float(d[(d.Scenario==scn)&(d.Controller=='EventDetected-GatedHistory')].gate_mean.mean())
    stress=np.mean([imp[x] for x in val_scen if x!='nominal']);nom=imp['nominal'];nomg=gmeans['nominal'];stressg=np.mean([gmeans[x] for x in val_scen if x!='nominal'])
    score=stress-4*max(0,-nom-.2)-30*max(0,nomg-.01)+0.15*min(5,stressg/(nomg+1e-6))
    val.append([qdet,gmax,thr,temp,stress,nom,nomg,stressg,score,json.dumps(imp),json.dumps(gmeans)])
val=pd.DataFrame(val,columns=['detector_q','gmax','threshold','temperature','stress_improvement_pct','nominal_improvement_pct','nominal_gate_mean','stress_gate_mean','selection_score','scenario_improvement_json','gate_mean_json']);val.to_csv(f'{OUT}/hyperparameter_validation.csv',index=False)
bestrow=val.sort_values(['selection_score','detector_q','gmax'],ascending=[False,False,True]).iloc[0];bestq=float(bestrow.detector_q);bestg=float(bestrow.gmax);print('selected',bestq,bestg,flush=True)
# Retrain selected architecture from scratch with a new seed before final test.
mg,thr,temp=train_event_side(bestg,bestq,901,7);PG=spack(mg,thr,temp)
protocol_G.update({'selected_detector_q':bestq,'selected_gmax':bestg,'selected_threshold':thr,'selected_temperature':temp});json.dump(protocol_G,open(f'{OUT}/protocol_G.json','w'),indent=2)

# Final main test.
print('main final test',flush=True);cols=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','control_effort','saturation_pct','residual_rms_ms2','gate_mean','gate_p95','alpha_mean','alpha_p95'];rows=[]
for scn in SCENARIOS:
    print(' ',scn,flush=True)
    for kind in ['LQR','Current','AlwaysOpen','Event','DetectorShuffle']:
        r,_=simulate(scn,TEST_SEEDS_G,kind,PG);rows+=r
main=pd.DataFrame(rows,columns=cols);main.to_csv(f'{OUT}/main_trials_100seeds.csv',index=False)
summary=main.groupby(['Scenario','Controller'],sort=False).agg(n=('SeedIndex','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),p95_mean=('p95_error_m','mean'),max_mean=('max_error_m','mean'),effort_mean=('control_effort','mean'),sat_mean=('saturation_pct','mean'),residual_rms=('residual_rms_ms2','mean'),gate_mean=('gate_mean','mean'),gate_p95_mean=('gate_p95','mean'),alpha_mean=('alpha_mean','mean'),alpha_p95_mean=('alpha_p95','mean')).reset_index();summary.to_csv(f'{OUT}/main_summary_100seeds.csv',index=False)

def pair(sc,a,b,seed):
    A=main[(main.Scenario==sc)&(main.Controller==a)].sort_values('SeedIndex');B=main[(main.Scenario==sc)&(main.Controller==b)].sort_values('SeedIndex');d=A.rmse_m.to_numpy()-B.rmse_m.to_numpy();rg=np.random.default_rng(seed);ii=rg.integers(0,len(d),(8000,len(d)));bs=d[ii].mean(1)
    try:p=wilcoxon(d).pvalue
    except:p=1
    return [sc,a,b,A.rmse_m.mean(),B.rmse_m.mean(),d.mean(),100*d.mean()/A.rmse_m.mean(),np.quantile(bs,.025),np.quantile(bs,.975),np.median(d),d.mean()/(d.std(ddof=1)+1e-12),p]
comparators=['CurrentOnly-AE-CeNN-494','AlwaysOpen-HistorySide','DetectorShuffled-GatedHistory','LQR-outer'];st=[]
for ci,c in enumerate(comparators):
    for si,s in enumerate(SCENARIOS):st.append(pair(s,c,'EventDetected-GatedHistory',9000+ci*100+si))
st=pd.DataFrame(st,columns=['Scenario','Comparator','Proposal','Comparator_RMSE_m','Proposal_RMSE_m','Improvement_m','Relative_improvement_pct','CI95_low_m','CI95_high_m','Median_improvement_m','Paired_effect_dz','Wilcoxon_p']);pv=st.Wilcoxon_p.to_numpy();order=np.argsort(pv);adj=np.empty(len(pv));run=0
for rank,i in enumerate(order):run=max(run,(len(pv)-rank)*pv[i]);adj[i]=min(1,run)
st['Holm_adjusted_p']=adj;st.to_csv(f'{OUT}/paired_stats_100seeds.csv',index=False)

# Force-step event-response trace on a 50-seed subset.
_,trace=simulate('force_step',EVENT_SEEDS,'Event',PG,collect_trace=True)
alpha=trace['alpha'];gate=trace['gate'];score=trace['score'];events=[300,800];evrows=[]
for onset in events:
    for j,s in enumerate(EVENT_SEEDS):
        # Detection is defined as alpha > 0.5, i.e. score above the nominal-calibrated threshold.
        ids=np.where(alpha[onset:onset+80,j]>0.5)[0];lat=np.nan if len(ids)==0 else ids[0]*DT
        pre=np.mean(gate[max(WARMUP,onset-100):onset,j]);post=np.mean(gate[onset:onset+100,j]);evrows.append([int(s),onset*DT,lat,pre,post,np.max(alpha[onset:onset+80,j]),np.max(score[onset:onset+80,j])])
ev=pd.DataFrame(evrows,columns=['SeedIndex','Onset_s','Detection_latency_s','Gate_pre_mean','Gate_post_1s_mean','Alpha_peak_0p8s','Score_peak_0p8s']);ev.to_csv(f'{OUT}/force_step_event_detection_50seeds.csv',index=False)

stressed=SCENARIOS[1:]
def vv(c):return st[st.Comparator==c].set_index('Scenario')
vc=vv('CurrentOnly-AE-CeNN-494');va=vv('AlwaysOpen-HistorySide');vs=vv('DetectorShuffled-GatedHistory')
gsum=summary[summary.Controller=='EventDetected-GatedHistory'].set_index('Scenario')
nomg=float(gsum.loc['nominal','gate_mean']);stressg=float(gsum.loc[stressed,'gate_mean'].mean())
latmed=float(np.nanmedian(ev.Detection_latency_s));detectrate=float(np.mean(np.isfinite(ev.Detection_latency_s)))
crit={'selected_detector_q':bestq,'selected_gmax':bestg,'nominal_gate_mean':nomg,'mean_stressed_gate':stressg,'stress_to_nominal_gate_ratio':stressg/(nomg+1e-12),'positive_CI_vs_current_stressed':[s for s in stressed if vc.loc[s,'CI95_low_m']>0],'positive_CI_vs_alwaysopen_stressed':[s for s in stressed if va.loc[s,'CI95_low_m']>0],'positive_CI_vs_detector_shuffled_stressed':[s for s in stressed if vs.loc[s,'CI95_low_m']>0],'nominal_improvement_vs_current_pct':float(vc.loc['nominal','Relative_improvement_pct']),'median_force_detection_latency_s':latmed,'force_detection_rate':detectrate,'control_benefit_pass':bool(sum(vc.loc[stressed,'CI95_low_m']>0)>=4 and vc.loc['nominal','Relative_improvement_pct']>-0.2),'detector_specificity_pass':bool(sum(vs.loc[stressed,'CI95_low_m']>0)>=4),'gate_separation_pass':bool(nomg<.01 and stressg>3*nomg),'force_response_pass':bool(latmed<=.20 and detectrate>=.90)}
json.dump(crit,open(f'{OUT}/criteria_results.json','w'),indent=2)

# Publication-style figures.
sel=['LQR-outer','CurrentOnly-AE-CeNN-494','AlwaysOpen-HistorySide','EventDetected-GatedHistory','DetectorShuffled-GatedHistory'];piv=summary[summary.Controller.isin(sel)].pivot(index='Scenario',columns='Controller',values='rmse_mean').reindex(SCENARIOS);ax=piv.plot(kind='bar',figsize=(13.5,6));ax.set_yscale('log');ax.set_ylabel('Mean position RMSE [m]');ax.set_xlabel('Scenario');ax.set_title('Situation G: disturbance-detected sparse history gating');ax.legend(title='',ncol=2,fontsize=8);plt.tight_layout();plt.savefig(f'{OUT}/figG1_rmse.png',dpi=220,bbox_inches='tight');plt.close()
e=[]
for s in SCENARIOS:
    for c in ['CurrentOnly-AE-CeNN-494','AlwaysOpen-HistorySide','DetectorShuffled-GatedHistory']:
        r=st[(st.Scenario==s)&(st.Comparator==c)].iloc[0];e.append([s,c,r.Relative_improvement_pct])
e=pd.DataFrame(e,columns=['Scenario','Comparator','Relative_improvement_pct']);e.to_csv(f'{OUT}/effect_summary.csv',index=False);ax=e.pivot(index='Scenario',columns='Comparator',values='Relative_improvement_pct').reindex(SCENARIOS).plot(kind='bar',figsize=(12.5,5.3));ax.axhline(0,linewidth=.8);ax.set_ylabel('Event-gated history RMSE improvement [%]');ax.set_xlabel('Scenario');ax.set_title('Control benefit and detector-specificity controls');plt.tight_layout();plt.savefig(f'{OUT}/figG2_effect.png',dpi=220,bbox_inches='tight');plt.close()
# Gate/alpha separation.
gp=gsum.reindex(SCENARIOS);fig=plt.figure(figsize=(10.5,4.8));ax=fig.add_subplot(111);ax.bar(SCENARIOS,gp.gate_mean.values);ax.set_ylabel('Mean effective gate');ax.set_xlabel('Scenario');ax.set_title('Disturbance-conditioned use of the history branch');ax.tick_params(axis='x',rotation=35);plt.tight_layout();plt.savefig(f'{OUT}/figG3_gate_separation.png',dpi=220,bbox_inches='tight');plt.close()
# Force event profile.
_,tr1=simulate('force_step',EVENT_SEEDS,'Event',PG,collect_trace=True);mean_alpha=tr1['alpha'].mean(1);mean_gate=tr1['gate'].mean(1);fig=plt.figure(figsize=(11,4.8));ax=fig.add_subplot(111);ax.plot(T,mean_alpha,label='detector opening alpha');ax.plot(T,mean_gate,label='effective gate');ax.axvline(3.0,linestyle='--');ax.axvline(5.0,linestyle='--');ax.axvline(8.0,linestyle='--');ax.axvline(10.0,linestyle='--');ax.set_xlabel('Time [s]');ax.set_ylabel('Opening / gate');ax.set_title('Force-step: causal detector and effective history gate');ax.legend();plt.tight_layout();plt.savefig(f'{OUT}/figG4_force_event_gate.png',dpi=220,bbox_inches='tight');plt.close()

# Report.
report=['# SITUATION G — Disturbance-detected sparse history gating','', 'Status: **frozen targeted confirmatory snapshot**. Situations A–F remain preserved.','', '## Architecture','The exact strong current-only AE-CeNN path is frozen. The 100-ms history branch remains separate. A nominal-calibrated causal anomaly detector multiplies the learned history gate, so the side branch can be nearly closed during nominal behavior and open only when the current observation departs from nominal dynamics.','', '## Detector','The detector uses four causal aggregate features: disturbance-observer magnitude, derivative of the disturbance observer, current-vs-100-ms-lag innovation, and tracking-state magnitude. Each is robustly standardized using only nominal training data. The scalar anomaly score is the RMS of positive robust z-scores.','', '## Protocol','- Hyperparameters selected only on validation seeds 2400–2439.','- Final test: 100 new paired seeds 3200–3299 per scenario, 12 s at 100 Hz.','- Main controls: current-only, always-open history side, detector-shuffled gate, and LQR.','- Force-step event analysis uses 50 final-test seeds and measures detector latency causally.','', '## Main mean RMSE [m]','',piv.round(5).to_markdown(),'','## Pre-registered criteria','```json',json.dumps(crit,indent=2),'```','', '## Claim boundary',protocol_G['claim_boundary']]
open(f'{OUT}/SITUATION_G_REPORT.md','w').write('\n'.join(report));json.dump({'elapsed_s':time.time()-t0,'criteria':crit,'selected_detector_q':bestq,'selected_gmax':bestg},open(f'{OUT}/manifest_G.json','w'),indent=2)

# Preserve named state and archive.
open('/mnt/data/SITUATION_G.md','w').write('\n'.join(report))
index='/mnt/data/SITUATIONS_A_B_C_D_E_F_INDEX.md';idx=open(index).read() if os.path.exists(index) else ''
if 'SITUATION G' not in idx:idx+='\n- **SITUATION G** — nominal-calibrated disturbance detector controlling a bounded separate 100-ms history latent branch; current-only AE-CeNN remains frozen.\n'
open('/mnt/data/SITUATIONS_A_B_C_D_E_F_G_INDEX.md','w').write(idx)
zip_path=f'{OUT}/Situation_G_FINAL_bundle.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:
    for fn in os.listdir(OUT):
        pth=os.path.join(OUT,fn)
        if os.path.isfile(pth) and fn!=os.path.basename(zip_path):zf.write(pth,arcname=fn)
    if os.path.exists(__file__):zf.write(__file__,arcname='run_situation_G.py')
    for pth in ['/mnt/data/SITUATION_A.md','/mnt/data/SITUATION_B.md','/mnt/data/SITUATION_C.md','/mnt/data/situation_D_final/SITUATION_D.md','/mnt/data/SITUATION_D2_CONFIRMATORY.md','/mnt/data/SITUATION_E.md','/mnt/data/SITUATION_F.md','/mnt/data/SITUATIONS_A_B_C_D_E_F_G_INDEX.md']:
        if os.path.exists(pth):zf.write(pth,arcname='preserved/'+os.path.basename(pth))
with zipfile.ZipFile(zip_path) as zf:assert zf.testzip() is None
sha=hashlib.sha256(open(zip_path,'rb').read()).hexdigest();open(f'{OUT}/bundle_sha256.txt','w').write(sha+'  '+os.path.basename(zip_path)+'\n')
print('G COMPLETE',time.time()-t0);print('selected q/gmax',bestq,bestg);print(piv.round(5).to_string());print(json.dumps(crit,indent=2));print(sha)

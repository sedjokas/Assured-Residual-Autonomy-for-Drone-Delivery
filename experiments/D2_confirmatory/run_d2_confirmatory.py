import os, json, time, hashlib, zipfile, platform, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numba import njit
from scipy.linalg import solve_discrete_are
from scipy.stats import wilcoxon, binomtest
import matplotlib.pyplot as plt

OUT='/mnt/data/d2_confirmatory'
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

@njit(cache=True)
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

print('Collecting training data...',flush=True)
TT=[];YY=[]
for scn in TRAIN_SCENARIOS:
    t,y=collect(scn,TRAIN_SEEDS);TT.append(t);YY.append(y)
TT=np.concatenate(TT,1);YY=np.concatenate(YY,1);tm=TT.reshape(-1,3).mean(0);ts=TT.reshape(-1,3).std(0)+1e-6
Hs=[];Ys=[]
for ei in range(TT.shape[1]):
    hist=ExactHistory(1)
    for k in range(STEPS):
        tn=(TT[k,ei:ei+1]-tm)/ts;h=hist.update(tn)
        if k%2==0 and k>=WARMUP:Hs.append(h[0]);Ys.append(YY[k,ei])
H=np.asarray(Hs);HY=np.asarray(Ys);hm=H.reshape(-1,24).mean(0);hs=H.reshape(-1,24).std(0)+1e-6;Hn=(H-hm)/hs;current=H[..., [0,8,16]]
rng=np.random.default_rng(SEED);idx=rng.choice(len(Hn),min(32000,len(Hn)),replace=False);Hn_s=Hn[idx];HY_s=HY[idx];current_s=current[idx]
hcells=Hn_s.reshape(-1,24);hcells=hcells[rng.choice(len(hcells),min(70000,len(hcells)),replace=False)];ccells=current_s.reshape(-1,3);ccells=ccells[rng.choice(len(ccells),min(70000,len(ccells)),replace=False)]

print('Training models...',flush=True)
aeC=train_ae(ccells,3,6,2,11,5);xc=apply_t(aeC,current_s.reshape(-1,3)).reshape(len(current_s),9,3);cennC=train_pred(CeNN(),latt(xc),HY_s,12,6)
cenn0=train_pred(CeNN(),latt(current_s),HY_s,13,6)
aeCap=train_ae(ccells,3,65,3,21,5);xcap=apply_t(aeCap,current_s.reshape(-1,3)).reshape(len(current_s),9,3);cennCap=train_pred(CeNN(),latt(xcap),HY_s,22,6)
aeD2=train_ae(hcells,24,16,3,31,6);z=enc_t(aeD2,Hn_s.reshape(-1,24)).reshape(len(Hn_s),9,3);zmean=z.reshape(-1,3).mean(0);zstd=z.reshape(-1,3).std(0)+1e-6;zn=(z-zmean)/zstd;cennD2=train_pred(CeNN(),latt(zn),HY_s,32,7)
hmlp=train_pred(HistMLP(),Hn_s.reshape(len(Hn_s),-1),HY_s,33,7)

def mask_lag(X):
    Y=np.zeros_like(X)
    for c in range(3):b=8*c;Y[...,b:b+3]=X[...,b:b+3]
    return Y
def mask_stats(X):
    Y=np.zeros_like(X)
    for c in range(3):b=8*c;Y[...,b]=X[...,b];Y[...,b+3:b+8]=X[...,b+3:b+8]
    return Y
def shuffle_past_train(X,seed=123):
    Y=X.copy();rr=np.random.default_rng(seed);perm=rr.permutation(len(X))
    for c in range(3):b=8*c;Y[...,b+1:b+8]=X[perm,...,b+1:b+8]
    return Y
def train_d2_variant(X,seed):
    cells=X.reshape(-1,24);cells=cells[rng.choice(len(cells),min(70000,len(cells)),replace=False)];ae=train_ae(cells,24,16,3,seed,6);zz=enc_t(ae,X.reshape(-1,24)).reshape(len(X),9,3);zm=zz.reshape(-1,3).mean(0);zs=zz.reshape(-1,3).std(0)+1e-6;ce=train_pred(CeNN(),latt((zz-zm)/zs),HY_s,seed+1,7);return ae,zm,zs,ce
aeSh,zShm,zShs,ceSh=train_d2_variant(shuffle_past_train(Hn_s,77),41);aeLag,zLagm,zLags,ceLag=train_d2_variant(mask_lag(Hn_s),51);aeStat,zStatm,zStats,ceStat=train_d2_variant(mask_stats(Hn_s),61)

def linpar(l):return l.weight.detach().numpy(),l.bias.detach().numpy()
def ex_ae(m):return [linpar(m.e1),linpar(m.e2),linpar(m.d1),linpar(m.d2)]
def ae_np(par,x,encode=False):
    (w1,b1),(w2,b2),(w3,b3),(w4,b4)=par;h=np.tanh(x@w1.T+b1);z=np.tanh(h@w2.T+b2)
    if encode:return z
    return np.tanh(z@w3.T+b3)@w4.T+b4
def ex_c(m):return [m.A.detach().numpy(),m.B.detach().numpy(),float(m.b.detach()),m.g.detach().numpy(),m.ob.detach().numpy()]
def conv_np(x,k):
    xp=np.pad(x,((0,0),(1,1),(1,1)));y=np.zeros_like(x)
    for i in range(3):
        for j in range(3):y+=k[i,j]*xp[:,i:i+3,j:j+3]
    return y
def cenn_np(par,inp):
    AA,BB,b,g,ob=par;x=np.zeros((len(inp),3,3));ff=np.zeros_like(x)
    for c in range(3):ff+=conv_np(inp[:,c],BB[c])
    for _ in range(4):x+=.32*(-x+conv_np(np.tanh(x),AA)+ff+b)
    return np.tanh(np.tanh(x)[:,2,:]*g+ob)*RES_MAX
AE_C=ex_ae(aeC);CE_C=ex_c(cennC);CE_0=ex_c(cenn0);AE_CAP=ex_ae(aeCap);CE_CAP=ex_c(cennCap);AE_D2=ex_ae(aeD2);CE_D2=ex_c(cennD2);AE_SH=ex_ae(aeSh);CE_SH=ex_c(ceSh);AE_LAG=ex_ae(aeLag);CE_LAG=ex_c(ceLag);AE_STAT=ex_ae(aeStat);CE_STAT=ex_c(ceStat)
HMW1,HMB1=linpar(hmlp.a);HMW2,HMB2=linpar(hmlp.b);HMW3,HMB3=linpar(hmlp.c)
def hist_mlp_np(x):return np.tanh(np.tanh(np.tanh(x@HMW1.T+HMB1)@HMW2.T+HMB2)@HMW3.T+HMB3)*RES_MAX
def shuffle_past_test(h):
    y=h.copy();r=np.roll(h,1,axis=0)
    for c in range(3):b=8*c;y[...,b+1:b+8]=r[...,b+1:b+8]
    return y
def d2_path(aepar,cepar,h,zmu,zsig):
    zz=ae_np(aepar,h.reshape(-1,24),True).reshape(len(h),9,3);return cenn_np(cepar,latt((zz-zmu)/zsig))
def predict(kind,temp,hist=None):
    tn=(temp-tm)/ts;n=len(tn)
    if kind=='CeNN0':o=cenn_np(CE_0,latt(tn))
    elif kind=='C':o=cenn_np(CE_C,latt(ae_np(AE_C,tn.reshape(-1,3)).reshape(n,9,3)))
    elif kind=='Capacity':o=cenn_np(CE_CAP,latt(ae_np(AE_CAP,tn.reshape(-1,3)).reshape(n,9,3)))
    else:
        h=(hist.update(tn)-hm)/hs
        if kind=='D2':o=d2_path(AE_D2,CE_D2,h,zmean,zstd)
        elif kind=='HistMLP':o=hist_mlp_np(h.reshape(n,-1))
        elif kind=='Shuffled':o=d2_path(AE_SH,CE_SH,shuffle_past_test(h),zShm,zShs)
        elif kind=='LagOnly':o=d2_path(AE_LAG,CE_LAG,mask_lag(h),zLagm,zLags)
        elif kind=='StatsOnly':o=d2_path(AE_STAT,CE_STAT,mask_stats(h),zStatm,zStats)
        else:raise ValueError(kind)
    return AUTH*np.clip(o,-RES_MAX,RES_MAX)

def recovery_metric(err,windows):
    if not windows:return np.full(err.shape[1],np.nan),np.full(err.shape[1],np.nan)
    n=err.shape[1];rec=[];iae=[]
    for a,b in windows:
        pre=np.median(err[max(0,a-100):a],axis=0)+.05;vals=np.full(n,(STEPS-b)*DT)
        for j in range(n):
            for k in range(b,max(b,STEPS-25)):
                if np.all(err[k:min(k+25,STEPS),j]<=pre[j]):vals[j]=(k-b)*DT;break
        rec.append(vals);iae.append(np.sum(err[a:min(a+200,STEPS)],axis=0)*DT)
    return np.mean(np.stack(rec),axis=0),np.mean(np.stack(iae),axis=0)

LABELS={'Geo':'Geo','LQR':'LQR-outer','LMPC':'LMPC-H40-surrogate','CeNN0':'Geo+CeNN-D2','C':'C: Geo+AE-CeNN','Capacity':'CapacityMatched-AE-CeNN','D2':'D2-History-AE-CeNN','HistMLP':'History-MLP','Shuffled':'D2-ShuffledHistory','LagOnly':'D2-LagOnly','StatsOnly':'D2-StatsOnly'}
def run(name,seeds,ctrl,drop=False,intensity=None):
    sc=scenario(name,seeds,intensity=intensity);n=len(seeds);p=np.repeat(PREF[0][None],n,0);v=np.repeat(VREF[0][None],n,0)
    for j,s in enumerate(seeds):rr=np.random.default_rng(700000+int(s)*19+sum(map(ord,name))*5);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
    fs=FeatureState(n);hist=ExactHistory(n) if ctrl in ('D2','HistMLP','Shuffled','LagOnly','StatsOnly') else None;act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)]
    errs=np.zeros((STEPS,n));efforts=np.zeros((STEPS,n));residuals=np.zeros((STEPS,n));sats=np.zeros((STEPS,n))
    for k in range(STEPS):
        te=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k])
        if ctrl=='Geo':base=geo(p,v,k);r=np.zeros_like(base)
        elif ctrl=='LQR':base=gain(p,v,k,KL);r=np.zeros_like(base)
        elif ctrl=='LMPC':base=gain(p,v,k,KH)-.18*fs.ahat;r=np.zeros_like(base)
        else:
            base=geo(p,v,k);r=predict(ctrl,te,hist)
            if drop and 800<=k<900:r[:]=0.0
        raw=base+r;u=np.clip(raw,-UMAX,UMAX);fs.cmd(u);sats[k]=np.any(np.abs(raw)>=UMAX-1e-12,axis=1);q.append(u.copy());q.pop(0);ud=np.array([q[-1-int(d)][j] for j,d in enumerate(sc['delay'])]);act+=(DT/.055)*(ud-act)
        a=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);v+=DT*a;p+=DT*v;e=np.linalg.norm(p-PREF[k],axis=1);errs[k]=e;efforts[k]=np.sum(u*u,axis=1);residuals[k]=np.linalg.norm(r,axis=1)
    rec,iae=recovery_metric(errs,sc['force_windows']);rows=[];label=LABELS[ctrl]+(' dropout' if drop else '')
    for j,s in enumerate(seeds):
        e=errs[WARMUP:,j];rows.append([name,label,int(s),float(intensity) if intensity is not None else np.nan,np.sqrt(np.mean(e*e)),np.quantile(e,.95),np.max(e),float(np.max(e)>1.0),np.mean(efforts[WARMUP:,j]),100*np.mean(sats[WARMUP:,j]),np.sqrt(np.mean(residuals[WARMUP:,j]**2)),rec[j],iae[j]])
    return rows

cols=['Scenario','Controller','SeedIndex','Intensity','rmse_m','p95_error_m','max_error_m','excursion_gt_1m','control_effort','saturation_pct','residual_rms_ms2','recovery_s','post_onset_IAE_m_s']
print('Running main 200-seed campaign...',flush=True);rows=[]
for scn in SCENARIOS:
    print('main',scn,flush=True)
    for c in ['Geo','LQR','LMPC','CeNN0','C','Capacity','D2','HistMLP']:rows+=run(scn,TEST_SEEDS,c)
main=pd.DataFrame(rows,columns=cols);main.to_csv(f'{OUT}/main_trials_200seeds.csv',index=False)
summary=main.groupby(['Scenario','Controller'],sort=False).agg(n=('SeedIndex','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),rmse_median=('rmse_m','median'),p95_error_mean=('p95_error_m','mean'),max_error_mean=('max_error_m','mean'),excursion_pct=('excursion_gt_1m',lambda x:100*x.mean()),effort_mean=('control_effort','mean'),saturation_pct_mean=('saturation_pct','mean'),residual_rms=('residual_rms_ms2','mean'),recovery_mean_s=('recovery_s','mean'),post_onset_IAE_mean=('post_onset_IAE_m_s','mean')).reset_index();summary.to_csv(f'{OUT}/main_summary_200seeds.csv',index=False)

def paired_stats(df,sc,a,b,B=10000):
    xa=df[(df.Scenario==sc)&(df.Controller==a)].sort_values('SeedIndex');xb=df[(df.Scenario==sc)&(df.Controller==b)].sort_values('SeedIndex');d=xa.rmse_m.to_numpy()-xb.rmse_m.to_numpy();rg=np.random.default_rng(12345+sum(map(ord,sc))+sum(map(ord,a)));ii=rg.integers(0,len(d),(B,len(d)));bs=d[ii].mean(1)
    try:p=wilcoxon(d,zero_method='wilcox').pvalue
    except ValueError:p=1.0
    dz=d.mean()/(d.std(ddof=1)+1e-12);ea=xa.excursion_gt_1m.to_numpy().astype(bool);eb=xb.excursion_gt_1m.to_numpy().astype(bool);n10=np.sum(ea&~eb);n01=np.sum(~ea&eb);disc=n10+n01;pm=1.0 if disc==0 else binomtest(int(n10),int(disc),.5).pvalue
    return [sc,a,b,xa.rmse_m.mean(),xb.rmse_m.mean(),d.mean(),100*d.mean()/xa.rmse_m.mean(),np.quantile(bs,.025),np.quantile(bs,.975),np.median(d),dz,p,int(n10),int(n01),pm]
comparators=['Geo','LQR-outer','LMPC-H40-surrogate','Geo+CeNN-D2','C: Geo+AE-CeNN','CapacityMatched-AE-CeNN','History-MLP'];st=[]
for scn in SCENARIOS:
    for co in comparators:st.append(paired_stats(main,scn,co,'D2-History-AE-CeNN'))
st=pd.DataFrame(st,columns=['Scenario','Comparator','Proposal','Comparator_RMSE_m','D2_RMSE_m','Absolute_improvement_m','Relative_improvement_pct','CI95_low_m','CI95_high_m','Median_improvement_m','Paired_effect_dz','Wilcoxon_p','Comparator_only_excursions','D2_only_excursions','McNemar_exact_p']);pv=st.Wilcoxon_p.to_numpy();order=np.argsort(pv);adj=np.empty(len(pv));running=0.0
for rank,i in enumerate(order):running=max(running,(len(pv)-rank)*pv[i]);adj[i]=min(1,running)
st['Holm_adjusted_p']=adj;st.to_csv(f'{OUT}/main_paired_statistics_200seeds.csv',index=False)

print('Running 100-seed ablations...',flush=True);sec=[]
for scn in SCENARIOS:
    print('ablation',scn,flush=True)
    for c in ['D2','Shuffled','LagOnly','StatsOnly']:sec+=run(scn,ABL_SEEDS,c)
    sec+=run(scn,ABL_SEEDS,'D2',drop=True)
secondary=pd.DataFrame(sec,columns=cols);secondary.to_csv(f'{OUT}/secondary_ablation_trials_100seeds.csv',index=False);secsummary=secondary.groupby(['Scenario','Controller'],sort=False).agg(n=('SeedIndex','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),excursion_pct=('excursion_gt_1m',lambda x:100*x.mean()),saturation_pct_mean=('saturation_pct','mean'),recovery_mean_s=('recovery_s','mean')).reset_index();secsummary.to_csv(f'{OUT}/secondary_ablation_summary_100seeds.csv',index=False)
secstats=[]
for scn in SCENARIOS:
    for co in ['D2-ShuffledHistory','D2-LagOnly','D2-StatsOnly','D2-History-AE-CeNN dropout']:secstats.append(paired_stats(secondary,scn,co,'D2-History-AE-CeNN'))
secstats=pd.DataFrame(secstats,columns=st.columns[:-1]);pv=secstats.Wilcoxon_p.to_numpy();order=np.argsort(pv);adj=np.empty(len(pv));running=0.0
for rank,i in enumerate(order):running=max(running,(len(pv)-rank)*pv[i]);adj[i]=min(1,running)
secstats['Holm_adjusted_p']=adj;secstats.to_csv(f'{OUT}/secondary_ablation_paired_statistics_100seeds.csv',index=False)

print('Running intensity envelope...',flush=True);envrows=[];envseeds=np.arange(600,700)
for lev in [1.5,3.0,4.5]:
    for c in ['LQR','C','Capacity','D2','HistMLP']:envrows+=run('wind_int',envseeds,c,intensity=lev)
for lev in [.10,.20,.30]:
    for c in ['LQR','C','Capacity','D2','HistMLP']:envrows+=run('model_int',envseeds,c,intensity=lev)
envdf=pd.DataFrame(envrows,columns=cols);envdf.to_csv(f'{OUT}/intensity_envelope_trials_100seeds.csv',index=False);envsum=envdf.groupby(['Scenario','Intensity','Controller'],sort=False).agg(rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),p95_error_mean=('p95_error_m','mean'),excursion_pct=('excursion_gt_1m',lambda x:100*x.mean())).reset_index();envsum.to_csv(f'{OUT}/intensity_envelope_summary.csv',index=False)

def nparam(m):return sum(p.numel() for p in m.parameters())
params={'Geo+CeNN-D2':nparam(cenn0),'C: Geo+AE-CeNN':nparam(aeC)+nparam(cennC),'CapacityMatched-AE-CeNN':nparam(aeCap)+nparam(cennCap),'D2-History-AE-CeNN':nparam(aeD2)+nparam(cennD2),'History-MLP':nparam(hmlp),'D2-ShuffledHistory':nparam(aeSh)+nparam(ceSh),'D2-LagOnly':nparam(aeLag)+nparam(ceLag),'D2-StatsOnly':nparam(aeStat)+nparam(ceStat)};pd.DataFrame({'Controller':list(params.keys()),'Parameters':list(params.values())}).to_csv(f'{OUT}/parameter_counts.csv',index=False)
def latency_samples(fn,N=2500):
    for _ in range(100):fn()
    vals=np.empty(N)
    for i in range(N):t0=time.perf_counter_ns();fn();vals[i]=(time.perf_counter_ns()-t0)/1000.0
    return dict(mean_us=float(vals.mean()),median_us=float(np.median(vals)),p95_us=float(np.quantile(vals,.95)),p99_us=float(np.quantile(vals,.99)))
zero3=np.zeros((1,9,3));zero24=np.zeros((1,9,24));hb=ExactHistory(1);hb.update(zero3)
lat={'CeNN-D2 neural':latency_samples(lambda:cenn_np(CE_0,latt(zero3))),'C neural':latency_samples(lambda:cenn_np(CE_C,latt(ae_np(AE_C,zero3.reshape(-1,3)).reshape(1,9,3)))),'Capacity neural':latency_samples(lambda:cenn_np(CE_CAP,latt(ae_np(AE_CAP,zero3.reshape(-1,3)).reshape(1,9,3)))),'D2 neural only':latency_samples(lambda:d2_path(AE_D2,CE_D2,zero24,zmean,zstd)),'History-MLP neural only':latency_samples(lambda:hist_mlp_np(zero24.reshape(1,-1)))}
def d2full():hh=(hb.update(zero3)-hm)/hs;return d2_path(AE_D2,CE_D2,hh,zmean,zstd)
lat['D2 exact-history + neural']=latency_samples(d2full);json.dump(lat,open(f'{OUT}/latency_microbenchmark.json','w'),indent=2)

stressed=[s for s in SCENARIOS if s!='nominal'];vsC=st[st.Comparator=='C: Geo+AE-CeNN'].set_index('Scenario');vsCap=st[st.Comparator=='CapacityMatched-AE-CeNN'].set_index('Scenario');vsSh=secstats[secstats.Comparator=='D2-ShuffledHistory'].set_index('Scenario');positive_C=[s for s in stressed if vsC.loc[s,'CI95_low_m']>0];positive_cap=[s for s in stressed if vsCap.loc[s,'CI95_low_m']>0];positive_sh=[s for s in stressed if vsSh.loc[s,'CI95_low_m']>0];ms=summary.set_index(['Scenario','Controller']);d2_exc=np.mean([ms.loc[(s,'D2-History-AE-CeNN'),'excursion_pct'] for s in stressed]);c_exc=np.mean([ms.loc[(s,'C: Geo+AE-CeNN'),'excursion_pct'] for s in stressed]);d2_sat=np.mean([ms.loc[(s,'D2-History-AE-CeNN'),'saturation_pct_mean'] for s in stressed]);c_sat=np.mean([ms.loc[(s,'C: Geo+AE-CeNN'),'saturation_pct_mean'] for s in stressed])
criteria={'D2_vs_C_positive_CI_stressed_count':len(positive_C),'scenarios':positive_C,'architecture_confirmation_pass':bool(len(positive_C)>=4 and d2_exc<=c_exc+1e-12 and d2_sat<=c_sat+1e-12),'D2_vs_capacity_positive_CI_stressed_count':len(positive_cap),'scenarios_capacity':positive_cap,'D2_vs_shuffled_positive_CI_stressed_count':len(positive_sh),'scenarios_shuffled':positive_sh,'history_attribution_pass':bool(len(positive_cap)>=4 and len(positive_sh)>=4),'mean_stressed_excursion_pct_D2':float(d2_exc),'mean_stressed_excursion_pct_C':float(c_exc),'mean_stressed_saturation_pct_D2':float(d2_sat),'mean_stressed_saturation_pct_C':float(c_sat)};json.dump(criteria,open(f'{OUT}/preregistered_criteria_results.json','w'),indent=2)

order=SCENARIOS;sel=['Geo','LQR-outer','LMPC-H40-surrogate','C: Geo+AE-CeNN','CapacityMatched-AE-CeNN','D2-History-AE-CeNN','History-MLP'];piv=summary[summary.Controller.isin(sel)].pivot(index='Scenario',columns='Controller',values='rmse_mean').reindex(order);ax=piv.plot(kind='bar',figsize=(13.5,6));ax.set_yscale('log');ax.set_ylabel('Mean position RMSE after 1.01 s warm-up [m]');ax.set_xlabel('Scenario');ax.set_title('D.2-Confirmatory: 200 paired held-out seeds per scenario');ax.legend(title='',ncol=2,fontsize=8);plt.tight_layout();plt.savefig(f'{OUT}/fig1_main_rmse_200seeds.png',dpi=220,bbox_inches='tight');plt.close()
effect=[]
for s in order:
    for co in ['C: Geo+AE-CeNN','CapacityMatched-AE-CeNN','History-MLP']:
        r=st[(st.Scenario==s)&(st.Comparator==co)].iloc[0];effect.append([s,co,r.Relative_improvement_pct,r.CI95_low_m,r.CI95_high_m])
effect=pd.DataFrame(effect,columns=['Scenario','Comparator','Relative_improvement_pct','CI_low_m','CI_high_m']);effect.to_csv(f'{OUT}/effect_summary.csv',index=False);ax=effect.pivot(index='Scenario',columns='Comparator',values='Relative_improvement_pct').reindex(order).plot(kind='bar',figsize=(12.5,5.4));ax.axhline(0,linewidth=.8);ax.set_ylabel('D2 RMSE improvement [%]');ax.set_xlabel('Scenario');ax.set_title('Incremental value of exact causal history conditioning');plt.tight_layout();plt.savefig(f'{OUT}/fig2_d2_incremental_effect.png',dpi=220,bbox_inches='tight');plt.close()
sp=secsummary.pivot(index='Scenario',columns='Controller',values='rmse_mean').reindex(order);ax=sp[['D2-History-AE-CeNN','D2-ShuffledHistory','D2-LagOnly','D2-StatsOnly']].plot(kind='bar',figsize=(12.5,5.4));ax.set_yscale('log');ax.set_ylabel('Mean position RMSE [m]');ax.set_xlabel('Scenario');ax.set_title('History attribution ablations (100 paired seeds)');plt.tight_layout();plt.savefig(f'{OUT}/fig3_history_ablation.png',dpi=220,bbox_inches='tight');plt.close()
fig=plt.figure(figsize=(8.2,5.2));ax=fig.add_subplot(111);windsum=envsum[envsum.Scenario=='wind_int']
for co in ['LQR-outer','C: Geo+AE-CeNN','CapacityMatched-AE-CeNN','D2-History-AE-CeNN','History-MLP']:
    q=windsum[windsum.Controller==co].sort_values('Intensity');ax.plot(q.Intensity,q.rmse_mean,marker='o',label=co)
ax.set_xlabel('Wind intensity [m/s]');ax.set_ylabel('Mean position RMSE [m]');ax.set_title('Wind disturbance envelope (100 paired seeds per level)');ax.legend(fontsize=8);plt.tight_layout();plt.savefig(f'{OUT}/fig4_wind_intensity_envelope.png',dpi=220,bbox_inches='tight');plt.close()
fig=plt.figure(figsize=(8.2,5.2));ax=fig.add_subplot(111);modsum=envsum[envsum.Scenario=='model_int']
for co in ['LQR-outer','C: Geo+AE-CeNN','CapacityMatched-AE-CeNN','D2-History-AE-CeNN','History-MLP']:
    q=modsum[modsum.Controller==co].sort_values('Intensity');ax.plot(100*q.Intensity,q.rmse_mean,marker='o',label=co)
ax.set_xlabel('Model mismatch range [±%]');ax.set_ylabel('Mean position RMSE [m]');ax.set_title('Model-mismatch envelope (100 paired seeds per level)');ax.legend(fontsize=8);plt.tight_layout();plt.savefig(f'{OUT}/fig5_model_intensity_envelope.png',dpi=220,bbox_inches='tight');plt.close()

report=['# D.2-CONFIRMATORY — exact 101-sample history, 200 paired seeds','','Status: frozen confirmatory benchmark-derived simulation snapshot. Situations A–D remain preserved separately.','','## Protocol','- 200 paired held-out seeds per main scenario (300–499); 12 s episodes at 100 Hz.','- Exact 101-sample causal history. Metadata per scalar: current, lag-10, lag-100, min, Q1, median, Q3, max.','- Primary endpoint: position RMSE after the 1.01 s history warm-up.','- Fixed effective residual authority ±1.05 m/s².','- D2 has 966 parameters; capacity-matched no-history control 959; history MLP 967.','- Independent AdaptiveQuadBench/RotorPy-derived outer-loop reproduction, not native full AdaptiveQuadBench/acados execution.','','## Main RMSE [m]','',piv.round(5).to_markdown(),'','## Pre-registered criteria','```json',json.dumps(criteria,indent=2),'```','','## Interpretation guardrails','- A gain over C but not over the capacity-matched control is a capacity effect, not evidence for history.','- A gain over both capacity-matched and shuffled-history controls supports temporally coherent history as a contributor.','- D2 is not claimed to replace LQR/MPC; regime dependence is retained explicitly.','- Tracking excursion >1 m is a surrogate diagnostic, not an externally certified safety constraint.']
open(f'{OUT}/D2_CONFIRMATORY_REPORT.md','w').write('\n'.join(report))
manifest={'script':os.path.basename(__file__),'seed':SEED,'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'protocol':protocol,'parameter_counts':params,'criteria':criteria,'latency_microbenchmark':lat};json.dump(manifest,open(f'{OUT}/manifest.json','w'),indent=2)
zip_path=f'{OUT}/D2_confirmatory_reproducibility_bundle.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:
    for fn in os.listdir(OUT):
        p=os.path.join(OUT,fn)
        if os.path.isfile(p) and fn!=os.path.basename(zip_path):zf.write(p,arcname=fn)
    for p in ['/mnt/data/SITUATION_A.md','/mnt/data/SITUATION_B.md','/mnt/data/SITUATION_C.md','/mnt/data/situation_D_final/SITUATION_D.md','/mnt/data/SITUATIONS_A_B_C_D_INDEX.md']:
        if os.path.exists(p):zf.write(p,arcname='preserved/'+os.path.basename(p))
with zipfile.ZipFile(zip_path) as zf:assert zf.testzip() is None
sha=hashlib.sha256(open(zip_path,'rb').read()).hexdigest();open(f'{OUT}/bundle_sha256.txt','w').write(sha+'  '+os.path.basename(zip_path)+'\n')
print('\n=== COMPLETE ===');print(piv.round(5).to_string());print('\nCriteria',json.dumps(criteria,indent=2));print('SHA256',sha)

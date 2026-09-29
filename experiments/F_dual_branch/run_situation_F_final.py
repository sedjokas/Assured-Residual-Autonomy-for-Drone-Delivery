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



# SITUATION F FINAL — exact Situation-E current baseline + isolated gated history branch
import copy, math
OUT='/mnt/data/situation_F_dual_gated_final';os.makedirs(OUT,exist_ok=True)
VAL_SEEDS=np.arange(1200,1240)
TEST_SEEDS_F=np.arange(2200,2300)
SECONDARY_SEEDS=np.arange(2200,2230)
GATE_CANDIDATES=[0.15,0.35,0.60]
GATE_INIT_FRAC=0.15
GATE_REG=0.0005

protocol={'name':'Situation F final — dual-branch gated latent residual','status':'targeted confirmatory benchmark-derived simulation','control_frequency_hz':100,'episode_length_s':12,'training_scope':'same as Situation E: seeds 100-115 in nominal/wind3/force_step/model20','validation_seeds':'1200-1239, used only for gate-bound selection','test_seeds':'2200-2299, 100 paired seeds per scenario, not used in prior experiments','architecture':'frozen exact Situation-E current-only latent path + separate 100-ms history encoder + bounded dynamic latent gate','fusion':'z_fused=z_current+g(z_current,z_history)*z_history','gate_candidates':GATE_CANDIDATES,'gate_init_fraction':GATE_INIT_FRAC,'gate_regularization':GATE_REG,'capacity_control':'same side architecture receives current temporal triplet instead of lagged triplet','coherence_control':'same trained history model with lag permuted across paired seeds at test','direct_concat_control':'exact Situation-E-style 6->45->3 concatenation control retrained deterministically','primary_endpoint':'position RMSE after 1.01 s warm-up','success_criteria':{'incremental_history':'positive paired CI vs exact current-only in >=4 of 7 stressed regimes, nominal degradation <2%','history_specificity':'positive paired CI vs matched current-side capacity control in >=4 of 7 stressed regimes','temporal_coherence':'positive paired CI vs shuffled history in >=4 of 7 stressed regimes'}}
json.dump(protocol,open(f'{OUT}/protocol_F.json','w'),indent=2)

t0=time.time();np.random.seed(SEED+4500);torch.manual_seed(SEED+4500);torch.set_num_threads(1)
print('collect training data',flush=True)
TT=[];YY=[]
for scn in TRAIN_SCENARIOS:
 t,y=collect(scn,TRAIN_SEEDS);TT.append(t);YY.append(y)
TT=np.concatenate(TT,1);YY=np.concatenate(YY,1);tm=TT.reshape(-1,3).mean(0);ts=TT.reshape(-1,3).std(0)+1e-6;TTn=(TT-tm)/ts
Fc=[];Fs=[];Y=[]
for k in range(WARMUP,STEPS,2):Fc.append(TTn[k]);Fs.append(np.concatenate([TTn[k],TTn[k-10]],axis=-1));Y.append(YY[k])
Fc=np.concatenate(Fc);Fs=np.concatenate(Fs);Y=np.concatenate(Y);rz=np.random.default_rng(SEED+1);ix=rz.choice(len(Fc),min(24000,len(Fc)),replace=False);Fc=Fc[ix];Fs=Fs[ix];Y=Y[ix];Fl=Fs[...,3:]

# EXACT E deterministic trainer.
def trainE(F,di,h,a,b):
 cells=F.reshape(-1,di);rr=np.random.default_rng(a);cells=cells[rr.choice(len(cells),min(50000,len(cells)),replace=False)];ae=train_ae(cells,di,h,3,a,4);z=enc_t(ae,F.reshape(-1,di)).reshape(len(F),9,3);zm=z.reshape(-1,3).mean(0);zs=z.reshape(-1,3).std(0)+1e-6;ce=train_pred(CeNN(),latt((z-zm)/zs),Y,b,5);return ae,zm,zs,ce
print('train exact Situation-E current and concat paths',flush=True)
Mc=trainE(Fc,3,64,11,12);Ms=trainE(Fs,6,45,15,16)
ae_cur,zmu,zsig,cenn_cur=Mc;ae_concat,zcmu,zcsig,cenn_concat=Ms
# Side-history AE pretraining is intentionally tiny.
hr=np.random.default_rng(105);hc=Fl.reshape(-1,3);hc=hc[hr.choice(len(hc),min(50000,len(hc)),replace=False)];ae_hist=train_ae(hc,3,8,3,105,4)
zcur=enc_t(ae_cur,Fc.reshape(-1,3)).reshape(len(Fc),9,3);zcur_n=(zcur-zmu)/zsig

class GatedSide(nn.Module):
 def __init__(self,cenn,pre,gmax):
  super().__init__();self.h1=nn.Linear(3,8);self.h2=nn.Linear(8,3)
  with torch.no_grad():self.h1.weight.copy_(pre.e1.weight);self.h1.bias.copy_(pre.e1.bias);self.h2.weight.copy_(pre.e2.weight);self.h2.bias.copy_(pre.e2.bias)
  self.gate=nn.Linear(6,3);nn.init.zeros_(self.gate.weight);f=GATE_INIT_FRAC;nn.init.constant_(self.gate.bias,math.log(f/(1-f)));self.gmax=float(gmax);self.cenn=copy.deepcopy(cenn).eval()
  for p in self.cenn.parameters():p.requires_grad_(False)
 def forward(self,zc,aux,ret=False):
  zh=torch.tanh(self.h2(torch.tanh(self.h1(aux))));g=self.gmax*torch.sigmoid(self.gate(torch.cat([zc,zh],-1)));zf=zc+g*zh;o=self.cenn(zf.permute(0,2,1).reshape(-1,3,3,3));return (o,g) if ret else o

def train_side(aux,gmax,seed,epochs=7):
 torch.manual_seed(seed);m=GatedSide(cenn_cur,ae_hist,gmax);opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.0025);Z=torch.tensor(zcur_n,dtype=torch.float32);A=torch.tensor(aux,dtype=torch.float32);YT=torch.tensor(Y/RES_MAX,dtype=torch.float32);N=len(Z);gg=torch.Generator().manual_seed(seed)
 for _ in range(epochs):
  order=torch.randperm(N,generator=gg)
  for st in range(0,N,4096):
   q=order[st:st+4096];o,g=m(Z[q],A[q],True);loss=((o-YT[q])**2).mean()+GATE_REG*(g*g).mean();opt.zero_grad();loss.backward();opt.step()
 return m.eval()

# Fast NumPy inference.
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
EC=epack(ae_cur);CC=cp(cenn_cur);ES=epack(ae_concat);CS=cp(cenn_concat)
def cur_lat(tn):return (enc(EC,tn.reshape(-1,3)).reshape(len(tn),9,3)-zmu)/zsig
def cur_pred(tn):return AUTH*cf(CC,cur_lat(tn))
def concat_pred(tn,lag):
 z=enc(ES,np.concatenate([tn,lag],-1).reshape(-1,6)).reshape(len(tn),9,3);return AUTH*cf(CS,(z-zcmu)/zcsig)
def spack(m):return {'h1':lin(m.h1),'h2':lin(m.h2),'gate':lin(m.gate),'gmax':m.gmax}
def side_pred(p,tn,aux):
 z=cur_lat(tn);(w1,b1)=p['h1'];(w2,b2)=p['h2'];(wg,bg)=p['gate'];zh=np.tanh(np.tanh(aux.reshape(-1,3)@w1.T+b1)@w2.T+b2).reshape(len(tn),9,3);g=p['gmax']/(1+np.exp(-(np.concatenate([z,zh],-1)@wg.T+bg)));return AUTH*cf(CC,z+g*zh),g

def simulate(name,seeds,kind,pack=None,drop=False):
 sc=scenario(name,seeds);n=len(seeds);p=np.repeat(PREF[0][None],n,0);v=np.repeat(VREF[0][None],n,0)
 for j,s in enumerate(seeds):rr=np.random.default_rng(700000+int(s)*19+sum(map(ord,name))*5);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
 fs=FeatureState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];hist=[];err=np.zeros((STEPS,n));eff=np.zeros((STEPS,n));sat=np.zeros((STEPS,n));res=np.zeros((STEPS,n));gt=np.zeros((STEPS,n));perm=np.random.default_rng(177000+sum(map(ord,name))).permutation(n)
 for k in range(STEPS):
  te=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k]);tn=(te-tm)/ts;hist.append(tn.copy());
  if len(hist)>11:hist.pop(0)
  lag=hist[0] if len(hist)<11 else hist[-11]
  if kind=='LQR':base=gain(p,v,k,KL);r=np.zeros_like(base);gm=np.zeros(n)
  elif kind=='LMPC':base=gain(p,v,k,KH)-.18*fs.ahat;r=np.zeros_like(base);gm=np.zeros(n)
  else:
   base=geo(p,v,k)
   if kind=='Current':r=cur_pred(tn);gm=np.zeros(n)
   elif kind=='Concat':r=concat_pred(tn,lag);gm=np.zeros(n)
   elif kind=='Hist':r,g=side_pred(pack,tn,lag);gm=g.mean((1,2))
   elif kind=='CurSide':r,g=side_pred(pack,tn,tn);gm=g.mean((1,2))
   elif kind=='Shuffle':r,g=side_pred(pack,tn,lag[perm]);gm=g.mean((1,2))
   if drop and 800<=k<900:r=cur_pred(tn);gm[:]=0
  raw=base+r;u=np.clip(raw,-UMAX,UMAX);fs.cmd(u);sat[k]=np.any(np.abs(raw)>=UMAX-1e-12,1);q.append(u.copy());q.pop(0);ud=np.empty_like(u)
  for dd in np.unique(sc['delay']):m=sc['delay']==dd;ud[m]=q[-1-int(dd)][m]
  act+=(DT/.055)*(ud-act);a=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);v+=DT*a;p+=DT*v;err[k]=np.linalg.norm(p-PREF[k],axis=1);eff[k]=np.sum(u*u,1);res[k]=np.linalg.norm(r,axis=1);gt[k]=gm
 labels={'LQR':'LQR-outer','LMPC':'LMPC-H40-surrogate','Current':'CurrentOnly-AE-CeNN-494','Concat':'DirectConcatLag10-AE-CeNN-496','Hist':'DualBranch-GatedHistory','CurSide':'DualBranch-MatchedCurrentSide','Shuffle':'DualBranch-ShuffledHistory'};lab=labels[kind]+(' side-outage' if drop else '');rows=[]
 for j,s in enumerate(seeds):
  e=err[WARMUP:,j];rows.append([name,lab,int(s),np.sqrt(np.mean(e*e)),np.quantile(e,.95),np.max(e),np.mean(eff[WARMUP:,j]),100*np.mean(sat[WARMUP:,j]),np.sqrt(np.mean(res[WARMUP:,j]**2)),np.mean(gt[WARMUP:,j]),np.quantile(gt[WARMUP:,j],.95)])
 return rows

# Gate-bound validation: select before touching final seeds.
print('validate gate bound',flush=True);val=[];mods={};vs=['nominal','wind3','force_step','model20','compound']
for i,gmax in enumerate(GATE_CANDIDATES):
 m=train_side(Fl,gmax,200+i,7);mods[gmax]=m;p=spack(m);rr=[]
 for scn in vs:rr+=simulate(scn,VAL_SEEDS,'Current');rr+=simulate(scn,VAL_SEEDS,'Hist',p)
 d=pd.DataFrame(rr,columns=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','control_effort','saturation_pct','residual_rms_ms2','gate_mean','gate_p95']);pv=d.pivot_table(index=['Scenario','SeedIndex'],columns='Controller',values='rmse_m');imp={}
 for scn in vs:
  q=pv.loc[scn];imp[scn]=float((100*(q['CurrentOnly-AE-CeNN-494']-q['DualBranch-GatedHistory'])/q['CurrentOnly-AE-CeNN-494']).mean())
 stress=np.mean([imp[x] for x in vs if x!='nominal']);nom=imp['nominal'];score=stress-2*max(0,-nom-2);val.append([gmax,stress,nom,score,json.dumps(imp)])
val=pd.DataFrame(val,columns=['gmax','stress_improvement_pct','nominal_improvement_pct','score','scenario_json']);val.to_csv(f'{OUT}/gate_validation.csv',index=False);best=float(val.sort_values(['score','gmax'],ascending=[False,True]).iloc[0].gmax);print('selected',best,flush=True)
# retrain chosen history and exact same-capacity current-side comparator from scratch
mh=train_side(Fl,best,301,8);mc=train_side(Fc,best,302,8);PH=spack(mh);PC=spack(mc)
protocol['selected_gmax']=best;json.dump(protocol,open(f'{OUT}/protocol_F.json','w'),indent=2)

print('main final test',flush=True);cols=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','control_effort','saturation_pct','residual_rms_ms2','gate_mean','gate_p95'];rows=[]
for scn in SCENARIOS:
 print(' ',scn,flush=True)
 for kind,pk in [('LQR',None),('LMPC',None),('Current',None),('Concat',None),('CurSide',PC),('Hist',PH),('Shuffle',PH)]:rows+=simulate(scn,TEST_SEEDS_F,kind,pk)
main=pd.DataFrame(rows,columns=cols);main.to_csv(f'{OUT}/main_trials_100seeds.csv',index=False);summary=main.groupby(['Scenario','Controller'],sort=False).agg(n=('SeedIndex','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),p95_mean=('p95_error_m','mean'),max_mean=('max_error_m','mean'),effort_mean=('control_effort','mean'),sat_mean=('saturation_pct','mean'),residual_rms=('residual_rms_ms2','mean'),gate_mean=('gate_mean','mean'),gate_p95_mean=('gate_p95','mean')).reset_index();summary.to_csv(f'{OUT}/main_summary_100seeds.csv',index=False)

def pair(sc,a,b,seed):
 A=main[(main.Scenario==sc)&(main.Controller==a)].sort_values('SeedIndex');B=main[(main.Scenario==sc)&(main.Controller==b)].sort_values('SeedIndex');d=A.rmse_m.to_numpy()-B.rmse_m.to_numpy();rg=np.random.default_rng(seed);ii=rg.integers(0,len(d),(8000,len(d)));bs=d[ii].mean(1)
 try:p=wilcoxon(d).pvalue
 except:p=1
 return [sc,a,b,A.rmse_m.mean(),B.rmse_m.mean(),d.mean(),100*d.mean()/A.rmse_m.mean(),np.quantile(bs,.025),np.quantile(bs,.975),np.median(d),d.mean()/(d.std(ddof=1)+1e-12),p]
comparators=['CurrentOnly-AE-CeNN-494','DirectConcatLag10-AE-CeNN-496','DualBranch-MatchedCurrentSide','DualBranch-ShuffledHistory','LQR-outer','LMPC-H40-surrogate'];st=[]
for ci,c in enumerate(comparators):
 for si,s in enumerate(SCENARIOS):st.append(pair(s,c,'DualBranch-GatedHistory',5000+ci*100+si))
st=pd.DataFrame(st,columns=['Scenario','Comparator','Proposal','Comparator_RMSE_m','Proposal_RMSE_m','Improvement_m','Relative_improvement_pct','CI95_low_m','CI95_high_m','Median_improvement_m','Paired_effect_dz','Wilcoxon_p']);pv=st.Wilcoxon_p.to_numpy();order=np.argsort(pv);adj=np.empty(len(pv));run=0
for rank,i in enumerate(order):run=max(run,(len(pv)-rank)*pv[i]);adj[i]=min(1,run)
st['Holm_adjusted_p']=adj;st.to_csv(f'{OUT}/paired_stats_100seeds.csv',index=False)

# side-branch outage on 30 paired seeds
sec=[]
for s in SCENARIOS:sec+=simulate(s,SECONDARY_SEEDS,'Hist',PH);sec+=simulate(s,SECONDARY_SEEDS,'Hist',PH,True)
pd.DataFrame(sec,columns=cols).to_csv(f'{OUT}/side_outage_trials_30seeds.csv',index=False)

stressed=SCENARIOS[1:]
def vv(c):return st[st.Comparator==c].set_index('Scenario')
vc=vv('CurrentOnly-AE-CeNN-494');vm=vv('DualBranch-MatchedCurrentSide');vsh=vv('DualBranch-ShuffledHistory');vd=vv('DirectConcatLag10-AE-CeNN-496')
crit={'selected_gmax':best,'positive_CI_vs_current_stressed':[s for s in stressed if vc.loc[s,'CI95_low_m']>0],'positive_CI_vs_direct_concat_stressed':[s for s in stressed if vd.loc[s,'CI95_low_m']>0],'positive_CI_vs_matched_current_side_stressed':[s for s in stressed if vm.loc[s,'CI95_low_m']>0],'positive_CI_vs_shuffled_stressed':[s for s in stressed if vsh.loc[s,'CI95_low_m']>0],'nominal_improvement_vs_current_pct':float(vc.loc['nominal','Relative_improvement_pct']),'incremental_history_pass':bool(sum(vc.loc[stressed,'CI95_low_m']>0)>=4 and vc.loc['nominal','Relative_improvement_pct']>-2),'history_specificity_pass':bool(sum(vm.loc[stressed,'CI95_low_m']>0)>=4),'temporal_coherence_pass':bool(sum(vsh.loc[stressed,'CI95_low_m']>0)>=4),'mean_gate':{s:float(summary[(summary.Scenario==s)&(summary.Controller=='DualBranch-GatedHistory')].gate_mean.iloc[0]) for s in SCENARIOS},'p95_gate':{s:float(summary[(summary.Scenario==s)&(summary.Controller=='DualBranch-GatedHistory')].gate_p95_mean.iloc[0]) for s in SCENARIOS}}
json.dump(crit,open(f'{OUT}/criteria_results.json','w'),indent=2)
# 494 + 59 history encoder + 21 gate = 574 active parameters
pd.DataFrame({'Controller':['CurrentOnly-AE-CeNN','DirectConcatLag10-AE-CeNN','DualBranch-GatedHistory','DualBranch-MatchedCurrentSide'],'Active_inference_parameters':[494,496,574,574]}).to_csv(f'{OUT}/active_parameter_counts.csv',index=False)

sel=['LQR-outer','LMPC-H40-surrogate','CurrentOnly-AE-CeNN-494','DirectConcatLag10-AE-CeNN-496','DualBranch-MatchedCurrentSide','DualBranch-GatedHistory'];piv=summary[summary.Controller.isin(sel)].pivot(index='Scenario',columns='Controller',values='rmse_mean').reindex(SCENARIOS);ax=piv.plot(kind='bar',figsize=(13.5,6));ax.set_yscale('log');ax.set_ylabel('Mean position RMSE [m]');ax.set_xlabel('Scenario');ax.set_title('Situation F: protected current latent + gated history residual');ax.legend(title='',ncol=2,fontsize=8);plt.tight_layout();plt.savefig(f'{OUT}/figF1_rmse.png',dpi=220,bbox_inches='tight');plt.close()
e=[]
for s in SCENARIOS:
 for c in ['CurrentOnly-AE-CeNN-494','DirectConcatLag10-AE-CeNN-496','DualBranch-MatchedCurrentSide','DualBranch-ShuffledHistory']:
  r=st[(st.Scenario==s)&(st.Comparator==c)].iloc[0];e.append([s,c,r.Relative_improvement_pct])
e=pd.DataFrame(e,columns=['Scenario','Comparator','Relative_improvement_pct']);e.to_csv(f'{OUT}/effect_summary.csv',index=False);ax=e.pivot(index='Scenario',columns='Comparator',values='Relative_improvement_pct').reindex(SCENARIOS).plot(kind='bar',figsize=(13,5.4));ax.axhline(0,linewidth=.8);ax.set_ylabel('Gated-history RMSE improvement [%]');ax.set_xlabel('Scenario');ax.set_title('History benefit, capacity control, and coherence control');plt.tight_layout();plt.savefig(f'{OUT}/figF2_effect.png',dpi=220,bbox_inches='tight');plt.close()
g=summary[summary.Controller=='DualBranch-GatedHistory'].set_index('Scenario').reindex(SCENARIOS);fig=plt.figure(figsize=(10.5,4.8));ax=fig.add_subplot(111);ax.bar(SCENARIOS,g.gate_mean.values);ax.set_ylabel('Mean learned gate');ax.set_xlabel('Scenario');ax.set_title('Bounded use of the history latent side branch');ax.tick_params(axis='x',rotation=35);plt.tight_layout();plt.savefig(f'{OUT}/figF3_gate.png',dpi=220,bbox_inches='tight');plt.close()

report=['# SITUATION F — Protected current latent with gated history residual','', 'Status: **frozen targeted confirmatory snapshot**. A–E remain preserved.','', '## Architecture','The exact strong Situation-E current-only path is preserved and frozen. A separate 3->8->3 encoder sees only the 100-ms lagged temporal triplet. A dynamic bounded gate adds a latent correction: `z_fused = z_current + g * z_history`. Zero gate is exactly the current-only controller.','', '## Protocol','- Current baseline reconstruction follows the deterministic Situation-E training recipe (same sampling and seeds 11/12).','- Gate bound selected on validation seeds 1200–1239; final test uses new seeds 2200–2299.','- 100 paired test seeds per scenario; 12 s at 100 Hz.','- Matched current-side branch (574 active params) isolates capacity from history; shuffled-lag control tests temporal coherence.','- 30-seed 1 s history-side outage diagnostic checks exact fallback to the frozen current path.','', '## Main mean RMSE [m]','',piv.round(5).to_markdown(),'','## Pre-registered criteria','```json',json.dumps(crit,indent=2),'```','', '## Claim boundary','Benchmark-derived simulation evidence only; no native full AdaptiveQuadBench/acados, HIL, flight-validation, or generic CeNN-superiority claim.']
open(f'{OUT}/SITUATION_F_REPORT.md','w').write('\n'.join(report));json.dump({'elapsed_s':time.time()-t0,'criteria':crit,'selected_gmax':best},open(f'{OUT}/manifest_F.json','w'),indent=2)
zip_path=f'{OUT}/Situation_F_FINAL_bundle.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:
 for fn in os.listdir(OUT):
  p=os.path.join(OUT,fn)
  if os.path.isfile(p) and fn!=os.path.basename(zip_path):zf.write(p,arcname=fn)
 if os.path.exists(__file__):zf.write(__file__,arcname='run_situation_F_final.py')
 for pth in ['/mnt/data/SITUATION_A.md','/mnt/data/SITUATION_B.md','/mnt/data/SITUATION_C.md','/mnt/data/situation_D_final/SITUATION_D.md','/mnt/data/SITUATION_D2_CONFIRMATORY.md','/mnt/data/SITUATION_E.md','/mnt/data/SITUATIONS_A_B_C_D_E_INDEX.md']:
  if os.path.exists(pth):zf.write(pth,arcname='preserved/'+os.path.basename(pth))
with zipfile.ZipFile(zip_path) as zf:assert zf.testzip() is None
sha=hashlib.sha256(open(zip_path,'rb').read()).hexdigest();open(f'{OUT}/bundle_sha256.txt','w').write(sha+'  '+os.path.basename(zip_path)+'\n');print('F FINAL COMPLETE',time.time()-t0);print('gmax',best);print(piv.round(5).to_string());print(json.dumps(crit,indent=2));print(sha)

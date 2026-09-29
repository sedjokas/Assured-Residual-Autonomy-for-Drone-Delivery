import os, json, time, platform, sys, hashlib, zipfile, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
OUT=Path('/mnt/data/qcenn_ablation/final')
OUT.mkdir(parents=True,exist_ok=True)
MASTER_SEED=20260912
np.random.seed(MASTER_SEED); torch.manual_seed(MASTER_SEED); torch.set_num_threads(1)
DT=.01; FREQ=100; EPISODE_S=12.; STEPS=int(EPISODE_S/DT); T=np.arange(STEPS)*DT
TAU=.1; AUTH=.35; RES_MAX=3.; UMAX=6.; WARMUP=101
SCENARIOS=['nominal','wind3','force_step','model20','latency40','payload50','rotoreff30','compound']
STRESSED=SCENARIOS[1:]
TRAIN_SCENARIOS=['nominal','wind3','force_step','model20']
TRAIN_SEEDS=np.arange(100,116)
VAL_SEEDS=np.arange(14000,14040)
FINAL_SEEDS=np.arange(16000,16200)
NATIVE_SEEDS=np.arange(18000,18010)
REPLICATE_SEEDS=[12,112,212,312,412]

protocol={
 'name':'Structured quadratic CeNN nonlinearity ablation',
 'status':'pre-specified confirmatory ablation before final manuscript freeze',
 'question':'Does replacing only the CeNN dynamics block by a bounded structured second-order CeNN improve closed-loop residual control?',
 'fixed_elements':['same source-derived 6-DoF-inspired vector benchmark dynamics','same 3->64->3 AE encoder for primary models','same training corpus and ideal residual targets','same four CeNN relaxation steps and dt=0.32','same output tanh and residual authority/bound','same controller, scenarios and paired seeds'],
 'models':{
  'LinearCeNN-494':'baseline CeNN; active inference parameters 494',
  'QCeNN-XX-503':'same AE and CeNN plus 3x3 local bounded XX template (9 parameters)',
  'QCeNN-XXXY-512':'same AE and CeNN plus local bounded XX and XY templates (18 parameters)',
  'LinearCeNN-Capacity-515':'secondary capacity control: linear CeNN with 3->67->3 encoder, 515 active parameters'
 },
 'bounded_quadratic_definition':{
  'xbar':'tanh(X/2)', 'Y':'tanh(X)',
  'XX':'Qxx * (xbar elementwise xbar)',
  'XY':'Qxy * (xbar elementwise Y)',
  'note':'Qxx/Qxy are shared 3x3 local convolution templates initialized at zero; this is a bounded structured approximation to BXX+CXY, chosen to retain local CeNN structure and limit quadratic growth.'
 },
 'replicate_rule':'train 5 initialization replicates per architecture; select the replicate with median validation mean RMSE across nominal,payload50,rotoreff30,compound on seeds 14000-14039; never select the best replicate',
 'final_test':'200 untouched paired seeds 16000-16199 per scenario, all 8 scenarios',
 'native_transfer_spotcheck':'10 untouched RotorPy pinned-source-core seeds 18000-18009 for nominal, wind3, rotoreff30; no retuning; source-derived-trained weights only',
 'primary_endpoint':'position RMSE after 1.01 s warm-up',
 'secondary_endpoints':['p95 and max error','control effort','command saturation fraction','residual RMS','residual-bound duty','CeNN internal-state magnitude','inference latency'],
 'success_criteria':{
  'Q_vs_linear':'QCeNN-XXXY must have 95% paired bootstrap CI >0 improvement in >=4/7 stressed scenarios, average stressed RMSE improvement >=0.25%, nominal degradation <0.2%, and no material increase (>0.05 percentage point) in >1m excursion duty or command saturation.',
  'structure_vs_capacity':'QCeNN-XXXY must outperform the 515-parameter linear capacity control in >=4/7 stressed scenarios by positive paired CI OR have >=0.10% lower average stressed RMSE.',
  'XY_increment':'QCeNN-XXXY must have lower average stressed RMSE than QCeNN-XX; scenario-wise significance is secondary.',
  'stability':'zero non-finite missions; no residual-bound violation by construction; report latent-state tails without post-hoc threshold tuning.'
 },
 'claim_boundary':'source-derived confirmatory simulation plus a small pinned-RotorPy-source-core zero-shot transfer spot check; not full package-native AdaptiveQuadBench/acados, HIL or flight validation.'
}
json.dump(protocol,open(OUT/'protocol_preregistered_before_final_test.json','w'),indent=2)

# Reference and nominal controllers identical to Situations D-G.
w=np.array([.72,.93,.51])
PREF=np.stack([1.05*np.sin(w[0]*T),.85*np.sin(w[1]*T+.35),1.25+.38*np.sin(w[2]*T)],1)
VREF=np.stack([1.05*w[0]*np.cos(w[0]*T),.85*w[1]*np.cos(w[1]*T+.35),.38*w[2]*np.cos(w[2]*T)],1)
AREF=np.stack([-1.05*w[0]**2*np.sin(w[0]*T),-0.85*w[1]**2*np.sin(w[1]*T+.35),-0.38*w[2]**2*np.sin(w[2]*T)],1)
A1=np.array([[1,DT],[0,1.]])
B1=np.array([[.5*DT*DT],[DT]])
A=np.kron(np.eye(3),A1); B=np.zeros((6,3))
for j in range(3): B[2*j:2*j+2,j]=B1[:,0]
Q=np.diag([5,.5,5,.5,6,.6]); R=.18*np.eye(3)
P=solve_discrete_are(A,B,Q,R); KL=np.linalg.solve(R+B.T@P@B,B.T@P@A)
KP=np.array([4.2,4.2,5.]); KD=np.array([2.9,2.9,3.2])
def geo(p,v,k): return AREF[k]-KP*(p-PREF[k])-KD*(v-VREF[k])

def scenario(name,seeds,intensity=None):
 n=len(seeds); mass=np.ones(n); E=np.repeat(np.eye(3)[None],n,0); drag=np.full((n,3),.035); delay=np.zeros(n,int)
 ext=np.zeros((STEPS,n,3)); wind=np.zeros((STEPS,n,3)); pn=np.zeros((STEPS,n,3)); vn=np.zeros((STEPS,n,3)); force_windows=[]
 if name=='force_step': force_windows=[(300,500),(800,1000)]
 if name=='compound': force_windows=[(350,500),(850,1000)]
 for j,s in enumerate(seeds):
  rr=np.random.default_rng(910000+int(s)*131+sum(map(ord,name))*17+(0 if intensity is None else int(100*float(intensity))))
  sp=.0025; sv=.006
  if name.startswith('model'):
   lev=.20 if intensity is None else float(intensity); mass[j]=rr.uniform(1-lev,1+lev); drag[j]*=rr.uniform(1-lev,1+lev); E[j]=np.diag(rr.uniform(1-.9*lev,1+.9*lev,3))
  if name=='payload50': mass[j]=rr.uniform(1.35,1.50); E[j,0,2]=rr.uniform(-.04,.04); E[j,1,2]=rr.uniform(-.04,.04)
  if name=='rotoreff30':
   eff=rr.uniform(.70,1.,3); eff[rr.integers(0,3)]*=rr.uniform(.68,.82); E[j]=np.diag(eff); C=rr.normal(0,.045,(3,3)); np.fill_diagonal(C,0); E[j]+=C; sv=.009
  if name=='latency40': delay[j]=4
  if name=='compound': mass[j]=rr.uniform(1.15,1.30); drag[j]*=rr.uniform(.85,1.25); E[j]=np.diag(rr.uniform(.86,1.05,3)); delay[j]=2; sp=.004; sv=.010
  if name in ('wind3','compound'):
   mag=3. if name=='wind3' else 1.8; d1=rr.normal(size=3); d1/=np.linalg.norm(d1)+1e-12; d2=rr.normal(size=3); d2/=np.linalg.norm(d2)+1e-12; x=mag*d1
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

class AE(nn.Module):
 def __init__(self,di,h,z): super().__init__(); self.e1=nn.Linear(di,h); self.e2=nn.Linear(h,z); self.d1=nn.Linear(z,h); self.d2=nn.Linear(h,di)
 def encode(self,x): return torch.tanh(self.e2(torch.tanh(self.e1(x))))
 def forward(self,x): return self.d2(torch.tanh(self.d1(self.encode(x))))

class BaseCeNN(nn.Module):
 def __init__(self):
  super().__init__(); self.A=nn.Parameter(.05*torch.randn(3,3)); self.B=nn.Parameter(.08*torch.randn(3,3,3)); self.b=nn.Parameter(torch.zeros(1)); self.g=nn.Parameter(torch.ones(3)); self.ob=nn.Parameter(torch.zeros(3))
 def conv(self,x,k):
  y=torch.zeros_like(x); xp=torch.nn.functional.pad(x,(1,1,1,1))
  for i in range(3):
   for j in range(3): y += k[i,j]*xp[:,i:i+3,j:j+3]
  return y
 def dynamics_extra(self,x,y): return 0.0
 def forward(self,inp,return_state=False):
  x=torch.zeros((len(inp),3,3),device=inp.device); ff=torch.zeros_like(x)
  for c in range(3): ff += self.conv(inp[:,c],self.B[c])
  maxabs=torch.zeros(len(inp),device=inp.device)
  for _ in range(4):
   y=torch.tanh(x); xbar=torch.tanh(x/2.0); extra=self.dynamics_extra(xbar,y); x=x+.32*(-x+self.conv(y,self.A)+ff+self.b+extra); maxabs=torch.maximum(maxabs,torch.amax(torch.abs(x),dim=(1,2)))
  out=torch.tanh(torch.tanh(x)[:,2,:]*self.g+self.ob)
  return (out,maxabs) if return_state else out

class LinearCeNN(BaseCeNN): pass
class QCeNNXX(BaseCeNN):
 def __init__(self): super().__init__(); self.Qxx=nn.Parameter(torch.zeros(3,3))
 def dynamics_extra(self,xbar,y): return self.conv(xbar*xbar,self.Qxx)
class QCeNNXXXY(BaseCeNN):
 def __init__(self): super().__init__(); self.Qxx=nn.Parameter(torch.zeros(3,3)); self.Qxy=nn.Parameter(torch.zeros(3,3))
 def dynamics_extra(self,xbar,y): return self.conv(xbar*xbar,self.Qxx)+self.conv(xbar*y,self.Qxy)

def train_ae(X,di,h,z,seed,epochs=4):
 torch.manual_seed(seed); m=AE(di,h,z); opt=torch.optim.Adam(m.parameters(),lr=.003); X=torch.tensor(X,dtype=torch.float32); N=len(X); gen=torch.Generator().manual_seed(seed)
 for _ in range(epochs):
  ix=torch.randperm(N,generator=gen)
  for st in range(0,N,4096):
   q=ix[st:st+4096]; clean=X[q]; noise=clean+.05*torch.randn(clean.shape,generator=gen); loss=((m(noise)-clean)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
 return m.eval()
def train_pred(model,X,Y,seed,epochs=5):
 torch.manual_seed(seed); opt=torch.optim.Adam(model.parameters(),lr=.003); X=torch.tensor(X,dtype=torch.float32); Y=torch.tensor(Y/RES_MAX,dtype=torch.float32); N=len(X); gen=torch.Generator().manual_seed(seed)
 for _ in range(epochs):
  ix=torch.randperm(N,generator=gen)
  for st in range(0,N,4096):
   q=ix[st:st+4096]; pred=model(X[q]); loss=((pred-Y[q])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
 return model.eval()
def enc_t(m,x):
 with torch.no_grad(): return m.encode(torch.tensor(x,dtype=torch.float32)).numpy()
def latt(x): return np.transpose(x,(0,2,1)).reshape(-1,3,3,3)

def collect(name,seeds):
 sc=scenario(name,seeds); n=len(seeds); p=np.repeat(PREF[0][None],n,0); v=np.repeat(VREF[0][None],n,0); fs=FeatureState(n); act=np.zeros((n,3)); q=[np.zeros((n,3)) for _ in range(5)]; Ts=[]; Ys=[]
 for k in range(STEPS):
  temp=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k]); base=geo(p,v,k); u=np.clip(base,-UMAX,UMAX); fs.cmd(u); q.append(u.copy());q.pop(0); ud=np.array([q[-1-int(d)][j] for j,d in enumerate(sc['delay'])]); act+=(DT/.055)*(ud-act)
  other=sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v); ctrl=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]; a=ctrl+other; rhs=base-other
  ideal=np.array([np.linalg.solve(sc['E'][j]/sc['mass'][j],rhs[j])-base[j] for j in range(n)]); ideal=np.clip(ideal,-RES_MAX,RES_MAX); Ts.append(temp); Ys.append(ideal); v+=DT*a; p+=DT*v
 return np.stack(Ts),np.stack(Ys)

# Numpy pack/inference.
def linpack(l): return l.weight.detach().numpy().copy(), l.bias.detach().numpy().copy()
def aepack(ae): return [linpack(ae.e1),linpack(ae.e2)]
def aeenc(pack,x):
 (w1,b1),(w2,b2)=pack; return np.tanh(np.tanh(x@w1.T+b1)@w2.T+b2)
def kernel_matrix(k):
 M=np.zeros((9,9))
 for oi in range(3):
  for oj in range(3):
   o=3*oi+oj
   for a in range(3):
    for b in range(3):
     ii=oi+a-1; jj=oj+b-1
     if 0<=ii<3 and 0<=jj<3: M[o,3*ii+jj]+=k[a,b]
 return M
def cpack(m):
 d={'A':kernel_matrix(m.A.detach().numpy()),'B':np.stack([kernel_matrix(m.B.detach().numpy()[j]) for j in range(3)]),'b':float(m.b.detach()),'g':m.g.detach().numpy().copy(),'ob':m.ob.detach().numpy().copy(),'kind':m.__class__.__name__}
 if hasattr(m,'Qxx'): d['Qxx']=kernel_matrix(m.Qxx.detach().numpy())
 if hasattr(m,'Qxy'): d['Qxy']=kernel_matrix(m.Qxy.detach().numpy())
 return d
def cinfer(pack,z,return_state=False):
 inp=np.transpose(z,(0,2,1)); n=len(z); ff=np.zeros((n,9))
 for c in range(3): ff += inp[:,c]@pack['B'][c].T
 x=np.zeros_like(ff); mx=np.zeros(n)
 for _ in range(4):
  y=np.tanh(x); xb=np.tanh(x/2.0); extra=0.0
  if 'Qxx' in pack: extra=(xb*xb)@pack['Qxx'].T
  if 'Qxy' in pack: extra=extra+(xb*y)@pack['Qxy'].T
  x += .32*(-x+y@pack['A'].T+ff+pack['b']+extra); mx=np.maximum(mx,np.max(np.abs(x),axis=1))
 out=np.tanh(np.tanh(x.reshape(n,3,3))[:,2,:]*pack['g']+pack['ob'])*RES_MAX
 return (out,mx) if return_state else out

def active_params(ae,model):
 # Encoder only is active; decoder is not used at inference.
 return sum(p.numel() for p in [ae.e1.weight,ae.e1.bias,ae.e2.weight,ae.e2.bias])+sum(p.numel() for p in model.parameters())

print('Collect training corpus...',flush=True)
TT=[]; YY=[]
for scn in TRAIN_SCENARIOS:
 t,y=collect(scn,TRAIN_SEEDS); TT.append(t); YY.append(y)
TT=np.concatenate(TT,1); YY=np.concatenate(YY,1); tm=TT.reshape(-1,3).mean(0); ts=TT.reshape(-1,3).std(0)+1e-6; TTn=(TT-tm)/ts
Fc=[]; Y=[]
for k in range(WARMUP,STEPS,2): Fc.append(TTn[k]); Y.append(YY[k])
Fc=np.concatenate(Fc); Y=np.concatenate(Y); rz=np.random.default_rng(MASTER_SEED+1); ix=rz.choice(len(Fc),min(24000,len(Fc)),replace=False); Fc=Fc[ix]; Y=Y[ix]
# Fixed primary AE and capacity-control AE.
cells=Fc.reshape(-1,3); rr=np.random.default_rng(11); cells64=cells[rr.choice(len(cells),min(50000,len(cells)),replace=False)]
ae64=train_ae(cells64,3,64,3,11,4); z64=enc_t(ae64,Fc.reshape(-1,3)).reshape(len(Fc),9,3); zm64=z64.reshape(-1,3).mean(0); zs64=z64.reshape(-1,3).std(0)+1e-6; X64=latt((z64-zm64)/zs64)
rr=np.random.default_rng(11); cells67=cells[rr.choice(len(cells),min(50000,len(cells)),replace=False)]
ae67=train_ae(cells67,3,67,3,11,4); z67=enc_t(ae67,Fc.reshape(-1,3)).reshape(len(Fc),9,3); zm67=z67.reshape(-1,3).mean(0); zs67=z67.reshape(-1,3).std(0)+1e-6; X67=latt((z67-zm67)/zs67)

archs={
 'LinearCeNN-494':(LinearCeNN,ae64,zm64,zs64,X64),
 'QCeNN-XX-503':(QCeNNXX,ae64,zm64,zs64,X64),
 'QCeNN-XXXY-512':(QCeNNXXXY,ae64,zm64,zs64,X64),
 'LinearCeNN-Capacity-515':(LinearCeNN,ae67,zm67,zs67,X67)
}
models={}; train_rows=[]
for name,(cls,ae,zm,zs,X) in archs.items():
 models[name]=[]
 for seed in REPLICATE_SEEDS:
  torch.manual_seed(seed); m=cls(); m=train_pred(m,X,Y,seed,5); models[name].append(m)
  with torch.no_grad(): mse=float(((m(torch.tensor(X[:5000],dtype=torch.float32))-torch.tensor(Y[:5000]/RES_MAX,dtype=torch.float32))**2).mean())
  train_rows.append({'architecture':name,'replicate_seed':seed,'train_subsample_normalized_mse':mse,'active_inference_params':active_params(ae,m)})
pd.DataFrame(train_rows).to_csv(OUT/'training_replicates.csv',index=False)

# Cached encoder/model packs for fast validation/final evaluation.
def make_predictor(name,m):
 cls,ae,zm,zs,_=archs[name]; ep=aepack(ae); cp=cpack(m)
 def pred(tn,diag=False):
  z=aeenc(ep,tn.reshape(-1,3)).reshape(len(tn),9,3); z=(z-zm)/zs; o,mx=cinfer(cp,z,True); r=AUTH*o
  return (r,mx) if diag else r
 return pred,ep,cp,zm,zs

def simulate(name,seeds,predictor):
 sc=scenario(name,seeds); n=len(seeds); p=np.repeat(PREF[0][None],n,0); v=np.repeat(VREF[0][None],n,0); fs=FeatureState(n); act=np.zeros((n,3)); q=[np.zeros((n,3)) for _ in range(5)]
 err=np.zeros((STEPS,n)); effort=np.zeros((STEPS,n)); sat=np.zeros((STEPS,n),bool); resn=np.zeros((STEPS,n)); rbnd=np.zeros((STEPS,n),bool); latent=np.zeros((STEPS,n)); finite=np.ones(n,bool)
 for k in range(STEPS):
  temp=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k]); tn=(temp-tm)/ts; base=geo(p,v,k); res,mx=predictor(tn,True); u=np.clip(base+res,-UMAX,UMAX); fs.cmd(u)
  q.append(u.copy());q.pop(0); ud=np.array([q[-1-int(d)][j] for j,d in enumerate(sc['delay'])]); act+=(DT/.055)*(ud-act)
  other=sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v); ctrl=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]; a=ctrl+other; v+=DT*a; p+=DT*v
  e=np.linalg.norm(p-PREF[min(k+1,STEPS-1)],axis=1); err[k]=e; effort[k]=np.linalg.norm(u,axis=1); sat[k]=np.any(np.abs(base+res)>=UMAX-1e-10,axis=1); resn[k]=np.linalg.norm(res,axis=1); rbnd[k]=np.any(np.abs(res)>AUTH*RES_MAX+1e-9,axis=1); latent[k]=mx; finite &= np.isfinite(p).all(1)&np.isfinite(v).all(1)&np.isfinite(res).all(1)
 sl=slice(WARMUP,None); rows=[]
 for j,s in enumerate(seeds):
  e=err[sl,j]
  rows.append({'scenario':name,'seed':int(s),'rmse_m':float(np.sqrt(np.mean(e*e))),'p95_m':float(np.quantile(e,.95)),'max_m':float(np.max(e)),'excursion_gt1m_pct':float(100*np.mean(e>1.0)),'control_effort_mean':float(np.mean(effort[sl,j])),'command_saturation_pct':float(100*np.mean(sat[sl,j])),'residual_rms_mps2':float(np.sqrt(np.mean(resn[sl,j]**2))),'residual_bound_violation_pct':float(100*np.mean(rbnd[sl,j])),'latent_max':float(np.max(latent[sl,j])),'latent_p999':float(np.quantile(latent[sl,j],.999)),'finite':bool(finite[j])})
 return pd.DataFrame(rows)

# Validation replicate selection (median, not best).
# Pre-final protocol amendment: use a held-out OFFLINE residual-prediction corpus rather than
# closed-loop validation missions. This amendment was made before any final-test seed was run;
# it avoids selecting replicate weights on the same closed-loop endpoint used for confirmation.
amendment={
 'timestamp_stage':'before any 16000-16199 final-test execution',
 'change':'replicate selection changed from median closed-loop validation RMSE to median held-out offline normalized residual MSE',
 'reason':'computational tractability and stronger separation between model-initialization selection and the final closed-loop endpoint',
 'validation_seeds':'14000-14039',
 'validation_scenarios':['nominal','payload50','rotoreff30','compound'],
 'selection_rule':'for each architecture, rank five initialization replicates by mean normalized residual MSE on the held-out validation corpus and freeze the median-ranked replicate; never choose the best replicate',
 'final_test_seen_before_change':False
}
json.dump(amendment,open(OUT/'protocol_amendment_pre_final.json','w'),indent=2)
print('Offline validation replicate selection...',flush=True)
val_scen=['nominal','payload50','rotoreff30','compound']; VF=[]; VY=[]
for scn in val_scen:
 t,y=collect(scn,VAL_SEEDS); tn=(t-tm)/ts
 for k in range(WARMUP,STEPS,3): VF.append(tn[k]); VY.append(y[k])
VF=np.concatenate(VF); VY=np.concatenate(VY)
# Fixed deterministic cap for equal evaluation cost.
rv=np.random.default_rng(MASTER_SEED+909); vidx=rv.choice(len(VF),min(50000,len(VF)),replace=False); VF=VF[vidx]; VY=VY[vidx]
# Encode once per AE width.
def val_input(ae,zm,zs):
 z=enc_t(ae,VF.reshape(-1,3)).reshape(len(VF),9,3); return latt((z-zm)/zs)
VX64=val_input(ae64,zm64,zs64); VX67=val_input(ae67,zm67,zs67)
vr_rows=[]
with torch.no_grad():
 for aname,mlist in models.items():
  VX=VX67 if aname=='LinearCeNN-Capacity-515' else VX64
  tx=torch.tensor(VX,dtype=torch.float32); ty=torch.tensor(VY/RES_MAX,dtype=torch.float32)
  for ridx,m in enumerate(mlist):
   # batched evaluation to limit memory
   ss=0.; nn=0
   for st in range(0,len(VX),4096):
    q=slice(st,min(st+4096,len(VX))); pr=m(tx[q]); loss=((pr-ty[q])**2).sum().item(); ss+=loss; nn+=pr.numel()
   vr_rows.append({'architecture':aname,'replicate_seed':REPLICATE_SEEDS[ridx],'validation_offline_normalized_mse':ss/nn})
vr=pd.DataFrame(vr_rows)
selected={}
for aname in archs:
 q=vr[vr.architecture==aname].sort_values(['validation_offline_normalized_mse','replicate_seed']).reset_index(drop=True); sel=int(q.iloc[len(q)//2].replicate_seed); selected[aname]=sel
vr['selected_median_validation_model']=vr.apply(lambda r:int(r.replicate_seed)==selected[r.architecture],axis=1); vr.to_csv(OUT/'validation_model_selection.csv',index=False)
json.dump(selected,open(OUT/'selected_replicates_frozen.json','w'),indent=2)
print('Selected:',selected,flush=True)
# Freeze selected models before any final-test execution so later stages can run independently.
for aname in archs:
 idx=REPLICATE_SEEDS.index(selected[aname]); m=models[aname][idx]; cls,ae,zm,zs,_=archs[aname]
 torch.save({'model_state_dict':m.state_dict(),'model_class':m.__class__.__name__,'ae_state_dict':ae.state_dict(),'ae_hidden':64 if ae is ae64 else 67,'latent_mean':zm,'latent_std':zs,'feature_mean':tm,'feature_std':ts,'selected_seed':selected[aname]},OUT/(aname.replace('/','_')+'.pt'))
if os.environ.get('QCENN_STAGE','')=='prepare':
 print('Preparation stage complete; selected models frozen before final test.',flush=True)
 sys.exit(0)

# Final untouched 200-seed campaign.
print('Final 200-seed campaign...',flush=True)
final=[]; packs={}
for aname in archs:
 idx=REPLICATE_SEEDS.index(selected[aname]); m=models[aname][idx]; pred,ep,cp,zm,zs=make_predictor(aname,m); packs[aname]=(ep,cp,zm,zs)
 for scn in SCENARIOS:
  d=simulate(scn,FINAL_SEEDS,pred); d['architecture']=aname; final.append(d); print('done',aname,scn,flush=True)
final=pd.concat(final,ignore_index=True); final.to_csv(OUT/'final_trials_200seeds.csv',index=False)
summary=final.groupby(['scenario','architecture'],sort=False).agg(n=('seed','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),p95_mean=('p95_m','mean'),max_mean=('max_m','mean'),excursion_gt1m_mean_pct=('excursion_gt1m_pct','mean'),control_effort_mean=('control_effort_mean','mean'),command_saturation_mean_pct=('command_saturation_pct','mean'),residual_rms_mean=('residual_rms_mps2','mean'),latent_max=('latent_max','max'),latent_p999_mean=('latent_p999','mean'),finite_rate=('finite','mean')).reset_index(); summary.to_csv(OUT/'final_summary_200seeds.csv',index=False)

# Paired bootstrap stats.
def paired_stat(sc,prop,comp,nboot=15000):
 a=final[(final.scenario==sc)&(final.architecture==comp)].sort_values('seed'); b=final[(final.scenario==sc)&(final.architecture==prop)].sort_values('seed'); assert np.array_equal(a.seed.values,b.seed.values)
 d=a.rmse_m.values-b.rmse_m.values; rng=np.random.default_rng(88000+sum(map(ord,sc+prop+comp))); boot=d[rng.integers(0,len(d),(nboot,len(d)))].mean(1)
 try: p=float(wilcoxon(d,zero_method='wilcox').pvalue) if not np.allclose(d,0) else 1.0
 except Exception: p=1.0
 return {'scenario':sc,'proposal':prop,'comparator':comp,'n':len(d),'comparator_rmse':float(a.rmse_m.mean()),'proposal_rmse':float(b.rmse_m.mean()),'improvement_m':float(d.mean()),'improvement_pct':float(100*d.mean()/a.rmse_m.mean()),'ci95_low_m':float(np.quantile(boot,.025)),'ci95_high_m':float(np.quantile(boot,.975)),'wilcoxon_p':p,'proposal_better_fraction':float(np.mean(d>0)),'excursion_delta_pp':float(b.excursion_gt1m_pct.mean()-a.excursion_gt1m_pct.mean()),'saturation_delta_pp':float(b.command_saturation_pct.mean()-a.command_saturation_pct.mean())}
stats=[]
for sc in SCENARIOS:
 for prop,comp in [('QCeNN-XX-503','LinearCeNN-494'),('QCeNN-XXXY-512','LinearCeNN-494'),('QCeNN-XXXY-512','QCeNN-XX-503'),('QCeNN-XXXY-512','LinearCeNN-Capacity-515')]: stats.append(paired_stat(sc,prop,comp))
stats=pd.DataFrame(stats)
# Holm within each comparison family over 7 stressed scenarios.
stats['holm_p_stressed']=np.nan
for pair,grp in stats[stats.scenario!='nominal'].groupby(['proposal','comparator']):
 p=grp.wilcoxon_p.values; idxs=grp.index.values; order=np.argsort(p); adj=np.empty(len(p)); running=0.; m=len(p)
 for rank,ix in enumerate(order): running=max(running,(m-rank)*p[ix]); adj[ix]=min(1.,running)
 stats.loc[idxs,'holm_p_stressed']=adj
stats.to_csv(OUT/'paired_statistics.csv',index=False)

# Criteria evaluation.
def srow(sc,a): return summary[(summary.scenario==sc)&(summary.architecture==a)].iloc[0]
qbase=stats[(stats.proposal=='QCeNN-XXXY-512')&(stats.comparator=='LinearCeNN-494')]
qcap=stats[(stats.proposal=='QCeNN-XXXY-512')&(stats.comparator=='LinearCeNN-Capacity-515')]
qxx=stats[(stats.proposal=='QCeNN-XXXY-512')&(stats.comparator=='QCeNN-XX-503')]
mean_base=np.mean([x for x in qbase[qbase.scenario!='nominal'].improvement_pct])
mean_cap=np.mean([x for x in qcap[qcap.scenario!='nominal'].improvement_pct])
mean_xx=np.mean([x for x in qxx[qxx.scenario!='nominal'].improvement_pct])
nom=float(qbase[qbase.scenario=='nominal'].improvement_pct.iloc[0]) # positive means Q better, negative means degradation
pos_ci=int(np.sum((qbase.scenario!='nominal')&(qbase.ci95_low_m>0)))
pos_ci_cap=int(np.sum((qcap.scenario!='nominal')&(qcap.ci95_low_m>0)))
max_exc=float(qbase[qbase.scenario!='nominal'].excursion_delta_pp.max()); max_sat=float(qbase[qbase.scenario!='nominal'].saturation_delta_pp.max())
criteria={
 'q_vs_linear_positive_CI_scenarios':pos_ci,
 'q_vs_linear_mean_stressed_improvement_pct':float(mean_base),
 'q_vs_linear_nominal_degradation_pct':float(max(0,-nom)),
 'q_vs_linear_max_excursion_increase_pp':max_exc,
 'q_vs_linear_max_saturation_increase_pp':max_sat,
 'q_vs_linear_pass':bool(pos_ci>=4 and mean_base>=.25 and max(0,-nom)<.2 and max_exc<=.05 and max_sat<=.05),
 'structure_vs_capacity_positive_CI_scenarios':pos_ci_cap,
 'structure_vs_capacity_mean_stressed_improvement_pct':float(mean_cap),
 'structure_vs_capacity_pass':bool(pos_ci_cap>=4 or mean_cap>=.10),
 'xy_increment_mean_stressed_improvement_pct':float(mean_xx),
 'xy_increment_pass':bool(mean_xx>0),
 'all_missions_finite':bool(final.finite.all()),
 'residual_bound_violations_observed':int(np.sum(final.residual_bound_violation_pct>0))
}
criteria['primary_overall_pass']=bool(criteria['q_vs_linear_pass'] and criteria['structure_vs_capacity_pass'] and criteria['xy_increment_pass'] and criteria['all_missions_finite'] and criteria['residual_bound_violations_observed']==0)
json.dump(criteria,open(OUT/'completion_criteria.json','w'),indent=2)

# Inference latency microbenchmark, batch=1.
latrows=[]; rng=np.random.default_rng(444); tn=rng.normal(size=(1,9,3))
for aname in archs:
 idx=REPLICATE_SEEDS.index(selected[aname]); pred,*_=make_predictor(aname,models[aname][idx]);
 for _ in range(500): pred(tn)
 tt=[]
 for _ in range(5000):
  t0=time.perf_counter_ns(); pred(tn); tt.append(time.perf_counter_ns()-t0)
 latrows.append({'architecture':aname,'batch':1,'p50_us':np.percentile(tt,50)/1e3,'p95_us':np.percentile(tt,95)/1e3,'p99_us':np.percentile(tt,99)/1e3,'max_us':np.max(tt)/1e3})
pd.DataFrame(latrows).to_csv(OUT/'inference_latency_python.csv',index=False)

# Save selected model weights for exact reproduction.
for aname in archs:
 idx=REPLICATE_SEEDS.index(selected[aname]); m=models[aname][idx]; cls,ae,zm,zs,_=archs[aname]
 torch.save({'model_state_dict':m.state_dict(),'model_class':m.__class__.__name__,'ae_state_dict':ae.state_dict(),'ae_hidden':64 if ae is ae64 else 67,'latent_mean':zm,'latent_std':zs,'feature_mean':tm,'feature_std':ts,'selected_seed':selected[aname]},OUT/(aname.replace('/','_')+'.pt'))

# Native-source zero-shot transfer spot check.
native_rows=[]; native_error=None
try:
 if os.environ.get('QCENN_SKIP_NATIVE','0')=='1': raise RuntimeError('native transfer intentionally deferred to separate stage')
 sys.path.insert(0,'/mnt/data/AP_QI_CeNN_GitHub/experiments/L2_native_transfer')
 import native_validation_l_core as N
 def native_run(scn,seed,aname,steps=800):
  ep,cp,zm,zs=packs[aname]
  params,delay,force_dir,windpar=N.scenario_params(scn,seed); rng=np.random.default_rng(1000000+seed); vr=N.NativeRotorPyCore(params,control_abstraction='cmd_acc',initial_hover=False)
  p0,v0,a0=N.source_flat(0); st=vr.initial_state.copy(); st['x']=p0+rng.normal(0,.012,3); st['v']=v0+rng.normal(0,.018,3)
  fs=FeatureState(1); q=[]; err=[]; latstate=[]
  for k in range(steps):
   p=st['x']; v=st['v']; tn=(fs.feature(p[None],v[None],k,np.zeros((1,3)),np.zeros((1,3)))-tm)/ts
   z=aeenc(ep,tn.reshape(-1,3)).reshape(1,9,3); z=(z-zm)/zs; o,mx=cinfer(cp,z,True); res=AUTH*o; base=geo(p[None],v[None],k); u=(base+res)[0]; fs.cmd(u[None]); total=u+np.array([0.,0.,9.81]); q.append(total.copy()); cmd=q[-1-delay] if len(q)>delay else q[0]
   if scn=='force_step' and 200<=k<400: st['ext_force']=.8*force_dir
   else: st['ext_force']=np.zeros(3)
   if windpar:
    if k==0: wg=N.DrydenWind(*windpar)
    st['wind']=wg.update(DT,rng)
   else: st['wind']=np.zeros(3)
   st=vr.step_rk4(st,{'cmd_acc':cmd},DT,rng); err.append(np.linalg.norm(st['x']-PREF[min(k+1,STEPS-1)])); latstate.append(mx[0])
  e=np.array(err)[WARMUP:]; return {'scenario':scn,'seed':seed,'architecture':aname,'rmse_m':float(np.sqrt(np.mean(e*e))),'p95_m':float(np.quantile(e,.95)),'max_m':float(np.max(e)),'latent_max':float(np.max(latstate[WARMUP:]))}
 for scn in ['nominal','wind3','rotoreff30']:
  for seed in NATIVE_SEEDS:
   for aname in ['LinearCeNN-494','QCeNN-XX-503','QCeNN-XXXY-512']:
    native_rows.append(native_run(scn,int(seed),aname))
  print('native done',scn,flush=True)
 native=pd.DataFrame(native_rows); native.to_csv(OUT/'native_source_transfer_trials_10seeds.csv',index=False)
 nsum=native.groupby(['scenario','architecture']).agg(n=('seed','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),p95_mean=('p95_m','mean'),max_mean=('max_m','mean'),latent_max=('latent_max','max')).reset_index(); nsum.to_csv(OUT/'native_source_transfer_summary_10seeds.csv',index=False)
 nstats=[]
 for scn in ['nominal','wind3','rotoreff30']:
  for prop in ['QCeNN-XX-503','QCeNN-XXXY-512']:
   a=native[(native.scenario==scn)&(native.architecture=='LinearCeNN-494')].sort_values('seed'); b=native[(native.scenario==scn)&(native.architecture==prop)].sort_values('seed'); d=a.rmse_m.values-b.rmse_m.values
   nstats.append({'scenario':scn,'proposal':prop,'comparator':'LinearCeNN-494','n':len(d),'improvement_pct':100*d.mean()/a.rmse_m.mean(),'improvement_m':d.mean(),'proposal_better_fraction':np.mean(d>0)})
 pd.DataFrame(nstats).to_csv(OUT/'native_source_transfer_effects.csv',index=False)
except Exception as e:
 native_error=repr(e); open(OUT/'native_transfer_error.txt','w').write(native_error)

# Figures.
sel_order=['LinearCeNN-494','QCeNN-XX-503','QCeNN-XXXY-512','LinearCeNN-Capacity-515']
piv=summary.pivot(index='scenario',columns='architecture',values='rmse_mean').reindex(SCENARIOS)[sel_order]
ax=piv.plot(kind='bar',figsize=(13,5.8)); ax.set_yscale('log'); ax.set_ylabel('Mean position RMSE [m]'); ax.set_xlabel('Scenario'); ax.set_title('Structured quadratic CeNN nonlinearity ablation — 200 paired seeds'); ax.legend(title='',fontsize=8,ncol=2); plt.tight_layout(); plt.savefig(OUT/'figQ1_rmse.png',dpi=220,bbox_inches='tight'); plt.close()
# Relative effects vs baseline.
eff=qbase.set_index('scenario').reindex(SCENARIOS); ax=eff.improvement_pct.plot(kind='bar',figsize=(10.5,4.8)); ax.axhline(0,linewidth=1); ax.set_ylabel('QCeNN-XXXY improvement vs linear CeNN [%]'); ax.set_xlabel('Scenario'); ax.set_title('Paired RMSE effect of bounded XX+XY CeNN'); plt.tight_layout(); plt.savefig(OUT/'figQ2_relative_effect.png',dpi=220,bbox_inches='tight'); plt.close()
# Latency.
latdf=pd.DataFrame(latrows).set_index('architecture').reindex(sel_order); ax=latdf[['p50_us','p99_us']].plot(kind='bar',figsize=(10.5,4.8)); ax.set_ylabel('Python/NumPy inference latency [µs], batch=1'); ax.set_xlabel('Architecture'); ax.set_title('Inference-cost diagnostic (not embedded/HIL)'); plt.tight_layout(); plt.savefig(OUT/'figQ3_latency.png',dpi=220,bbox_inches='tight'); plt.close()
if native_rows:
 nsum=pd.read_csv(OUT/'native_source_transfer_summary_10seeds.csv'); pp=nsum.pivot(index='scenario',columns='architecture',values='rmse_mean').reindex(['nominal','wind3','rotoreff30'])[['LinearCeNN-494','QCeNN-XX-503','QCeNN-XXXY-512']]; ax=pp.plot(kind='bar',figsize=(9.8,4.8)); ax.set_ylabel('Mean position RMSE [m]'); ax.set_xlabel('Pinned RotorPy-source-core scenario'); ax.set_title('Zero-shot transfer spot check (10 paired seeds)'); ax.legend(title='',fontsize=8); plt.tight_layout(); plt.savefig(OUT/'figQ4_native_transfer.png',dpi=220,bbox_inches='tight'); plt.close()

# Readable report.
latdf=pd.DataFrame(latrows)
def fmt_pct(x): return f'{x:+.3f}%'
report=[]
report += ['# Structured Quadratic CeNN Nonlinearity Ablation — FINAL REPORT','', '## Scope', 'This is a deliberately narrow block-replacement ablation. The AE representation, training corpus, nominal controller, residual authority, output bound, scenarios, mission seeds, CeNN relaxation depth and optimization recipe are held fixed for the three primary models. The only primary architectural change is the addition of bounded local second-order XX and XX+XY templates inside the CeNN state dynamics. A slightly larger linear-encoder control brackets parameter-count effects.','']
report += ['## Models and active inference parameter counts','',pd.DataFrame(train_rows).groupby('architecture').active_inference_params.first().reset_index().to_markdown(index=False),'']
report += ['The second-order terms are evaluated on bounded state transforms: `xbar=tanh(X/2)` and `Y=tanh(X)`. Thus the tested model is a bounded structured realization of the proposed `BXX + CXY` idea rather than an unbounded free quadratic tensor. This was chosen a priori to reduce the risk that quadratic growth overwhelms the `-X` leakage term.','']
report += ['## Replicate selection','', 'Five initialization replicates were trained per architecture. Before any final-test mission was run, a documented protocol amendment changed replicate selection to the **median**, not minimum, held-out offline residual-prediction MSE across nominal/payload/rotor-efficiency/compound. This keeps initialization selection separate from the final closed-loop endpoint.','',vr.to_markdown(index=False),'']
report += ['## Final source-derived campaign — 200 untouched paired seeds/scenario','',summary.round(7).to_markdown(index=False),'']
report += ['## Paired RMSE statistics','',stats.round(7).to_markdown(index=False),'']
report += ['## Pre-specified completion criteria','```json',json.dumps(criteria,indent=2),'```','']
report += ['## Timing diagnostic','',latdf.round(3).to_markdown(index=False),'','These are Python/NumPy software timings on the present CPU and are not HIL or embedded measurements.','']
if native_rows:
 report += ['## Zero-shot pinned RotorPy-source-core transfer spot check','',pd.read_csv(OUT/'native_source_transfer_summary_10seeds.csv').round(7).to_markdown(index=False),'',pd.read_csv(OUT/'native_source_transfer_effects.csv').round(7).to_markdown(index=False),'','This small transfer set was not used for training, replicate selection or retuning. It is descriptive because N=10 per condition.','']
else: report += ['## Native-source transfer spot check','',f'Not completed because of: `{native_error}`','']
report += ['## Interpretation','']
if criteria['primary_overall_pass']:
 report += ['The pre-specified overall criterion **passes**. The bounded structured XX+XY CeNN shows a repeatable closed-loop benefit that cannot be explained solely by its small parameter-count increase, while retaining nominal behavior and bounded residual authority. This supports replacing the linear CeNN block in the final manuscript architecture, subject to preserving the independent RTA/safety layers and to the stated simulation provenance.']
else:
 report += ['The pre-specified overall criterion **does not pass**. Therefore the experiment does **not** support replacing the existing linear CeNN block as the default final architecture. Any isolated scenario gains should be treated as regime-specific effects or exploratory evidence, not as a general performance improvement. The negative/ambiguous result is still informative because it directly tests whether structured second-order state interactions improve the same residual-control task under matched conditions.']
report += ['','### Claim boundary','No generic CeNN superiority is inferred. The capacity control is approximate (515 vs 512 active parameters), not mathematically identical. The source-derived campaign is not the full AdaptiveQuadBench/acados stack; the RotorPy check uses pinned-source-core dynamics only. Independent safety filtering remains necessary even if a more expressive CeNN improves prediction or tracking.']
open(OUT/'QCeNN_ABLATION_FINAL_REPORT.md','w').write('\n'.join(report))

# Bundle and SHA256.
manifest=[]
for p in sorted(OUT.iterdir()):
 if p.is_file() and not p.name.endswith('.zip'):
  h=hashlib.sha256(p.read_bytes()).hexdigest(); manifest.append({'file':p.name,'sha256':h,'bytes':p.stat().st_size})
pd.DataFrame(manifest).to_csv(OUT/'MANIFEST_SHA256.csv',index=False)
bundle=OUT/'QCeNN_structured_nonlinearity_ablation_FINAL.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.iterdir()):
  if p.is_file() and p!=bundle: z.write(p,arcname=p.name)
 z.write(__file__,arcname='run_qcenn_ablation.py')
with zipfile.ZipFile(bundle) as z: assert z.testzip() is None
sha=hashlib.sha256(bundle.read_bytes()).hexdigest(); open(OUT/'bundle_sha256.txt','w').write(sha+'  '+bundle.name+'\n')
print('\nCRITERIA\n',json.dumps(criteria,indent=2)); print('Bundle SHA256',sha); print('REPORT',OUT/'QCeNN_ABLATION_FINAL_REPORT.md')

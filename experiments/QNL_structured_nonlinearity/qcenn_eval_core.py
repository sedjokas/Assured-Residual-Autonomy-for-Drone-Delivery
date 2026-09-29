import os, time, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.linalg import solve_discrete_are
from pathlib import Path
OUT=Path('/mnt/data/qcenn_ablation/final')
DT=.01; FREQ=100; EPISODE_S=12.; STEPS=int(EPISODE_S/DT); T=np.arange(STEPS)*DT
TAU=.1; AUTH=.35; RES_MAX=3.; UMAX=6.; WARMUP=101
w=np.array([.72,.93,.51])
PREF=np.stack([1.05*np.sin(w[0]*T),.85*np.sin(w[1]*T+.35),1.25+.38*np.sin(w[2]*T)],1)
VREF=np.stack([1.05*w[0]*np.cos(w[0]*T),.85*w[1]*np.cos(w[1]*T+.35),.38*w[2]*np.cos(w[2]*T)],1)
AREF=np.stack([-1.05*w[0]**2*np.sin(w[0]*T),-0.85*w[1]**2*np.sin(w[1]*T+.35),-0.38*w[2]**2*np.sin(w[2]*T)],1)
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
 def __init__(self): super().__init__(); self.A=nn.Parameter(.05*torch.randn(3,3)); self.B=nn.Parameter(.08*torch.randn(3,3,3)); self.b=nn.Parameter(torch.zeros(1)); self.g=nn.Parameter(torch.ones(3)); self.ob=nn.Parameter(torch.zeros(3))
 def conv(self,x,k):
  y=torch.zeros_like(x); xp=torch.nn.functional.pad(x,(1,1,1,1))
  for i in range(3):
   for j in range(3): y += k[i,j]*xp[:,i:i+3,j:j+3]
  return y
 def dynamics_extra(self,x,y): return 0.0
 def forward(self,inp):
  x=torch.zeros((len(inp),3,3),device=inp.device); ff=torch.zeros_like(x)
  for c in range(3): ff += self.conv(inp[:,c],self.B[c])
  for _ in range(4):
   y=torch.tanh(x); xb=torch.tanh(x/2.); x=x+.32*(-x+self.conv(y,self.A)+ff+self.b+self.dynamics_extra(xb,y))
  return torch.tanh(torch.tanh(x)[:,2,:]*self.g+self.ob)
class LinearCeNN(BaseCeNN): pass
class QCeNNXX(BaseCeNN):
 def __init__(self): super().__init__(); self.Qxx=nn.Parameter(torch.zeros(3,3))
 def dynamics_extra(self,xb,y): return self.conv(xb*xb,self.Qxx)
class QCeNNXXXY(BaseCeNN):
 def __init__(self): super().__init__(); self.Qxx=nn.Parameter(torch.zeros(3,3)); self.Qxy=nn.Parameter(torch.zeros(3,3))
 def dynamics_extra(self,xb,y): return self.conv(xb*xb,self.Qxx)+self.conv(xb*y,self.Qxy)

def linpack(l): return l.weight.detach().numpy().copy(),l.bias.detach().numpy().copy()
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
def cinfer(pack,z):
 inp=np.transpose(z,(0,2,1)); n=len(z); ff=np.zeros((n,9))
 for c in range(3): ff += inp[:,c]@pack['B'][c].T
 x=np.zeros_like(ff); mx=np.zeros(n)
 for _ in range(4):
  y=np.tanh(x); xb=np.tanh(x/2.); extra=0.
  if 'Qxx' in pack: extra=(xb*xb)@pack['Qxx'].T
  if 'Qxy' in pack: extra=extra+(xb*y)@pack['Qxy'].T
  x += .32*(-x+y@pack['A'].T+ff+pack['b']+extra); mx=np.maximum(mx,np.max(np.abs(x),axis=1))
 out=np.tanh(np.tanh(x.reshape(n,3,3))[:,2,:]*pack['g']+pack['ob'])*RES_MAX
 return out,mx
MODEL_CLASSES={'LinearCeNN':LinearCeNN,'QCeNNXX':QCeNNXX,'QCeNNXXXY':QCeNNXXXY}
def load_predictor(architecture):
 d=torch.load(OUT/(architecture+'.pt'),map_location='cpu',weights_only=False)
 ae=AE(3,int(d['ae_hidden']),3); ae.load_state_dict(d['ae_state_dict']); ae.eval(); cls=MODEL_CLASSES[d['model_class']]; m=cls(); m.load_state_dict(d['model_state_dict']);m.eval()
 ep=aepack(ae); cp=cpack(m); zm=np.array(d['latent_mean']); zs=np.array(d['latent_std']); tm=np.array(d['feature_mean']); ts=np.array(d['feature_std'])
 def pred(tn):
  z=aeenc(ep,tn.reshape(-1,3)).reshape(len(tn),9,3); z=(z-zm)/zs; o,mx=cinfer(cp,z); return AUTH*o,mx
 return pred,tm,ts,ep,cp,zm,zs

def simulate(scn,seeds,architecture):
 predictor,tm,ts,*_=load_predictor(architecture)
 sc=scenario(scn,seeds); n=len(seeds); p=np.repeat(PREF[0][None],n,0); v=np.repeat(VREF[0][None],n,0); fs=FeatureState(n); act=np.zeros((n,3)); q=[np.zeros((n,3)) for _ in range(5)]
 err=np.zeros((STEPS,n)); effort=np.zeros((STEPS,n)); sat=np.zeros((STEPS,n),bool); resn=np.zeros((STEPS,n)); rbnd=np.zeros((STEPS,n),bool); latent=np.zeros((STEPS,n)); finite=np.ones(n,bool)
 for k in range(STEPS):
  temp=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k]); tn=(temp-tm)/ts; base=geo(p,v,k); res,mx=predictor(tn); u=np.clip(base+res,-UMAX,UMAX);fs.cmd(u);q.append(u.copy());q.pop(0);ud=np.array([q[-1-int(d)][j] for j,d in enumerate(sc['delay'])]);act+=(DT/.055)*(ud-act);other=sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);ctrl=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None];a=ctrl+other;v+=DT*a;p+=DT*v
  e=np.linalg.norm(p-PREF[min(k+1,STEPS-1)],axis=1);err[k]=e;effort[k]=np.linalg.norm(u,axis=1);sat[k]=np.any(np.abs(base+res)>=UMAX-1e-10,axis=1);resn[k]=np.linalg.norm(res,axis=1);rbnd[k]=np.any(np.abs(res)>AUTH*RES_MAX+1e-9,axis=1);latent[k]=mx;finite &= np.isfinite(p).all(1)&np.isfinite(v).all(1)&np.isfinite(res).all(1)
 sl=slice(WARMUP,None);rows=[]
 for j,s in enumerate(seeds):
  e=err[sl,j];rows.append({'scenario':scn,'seed':int(s),'architecture':architecture,'rmse_m':float(np.sqrt(np.mean(e*e))),'p95_m':float(np.quantile(e,.95)),'max_m':float(np.max(e)),'excursion_gt1m_pct':float(100*np.mean(e>1.)),'control_effort_mean':float(np.mean(effort[sl,j])),'command_saturation_pct':float(100*np.mean(sat[sl,j])),'residual_rms_mps2':float(np.sqrt(np.mean(resn[sl,j]**2))),'residual_bound_violation_pct':float(100*np.mean(rbnd[sl,j])),'latent_max':float(np.max(latent[sl,j])),'latent_p999':float(np.quantile(latent[sl,j],.999)),'finite':bool(finite[j])})
 return pd.DataFrame(rows)

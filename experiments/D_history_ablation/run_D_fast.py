import os, json, time, zipfile, hashlib, shutil
import numpy as np, pandas as pd
import torch, torch.nn as nn
from scipy.linalg import solve_discrete_are
from scipy.stats import wilcoxon

OUT='/mnt/data/situation_D'; os.makedirs(OUT,exist_ok=True)
SEED=20260911; np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
DT=.01; STEPS=250; T=np.arange(STEPS)*DT; TAU=.1; AUTH=.35; RES_MAX=3.; UMAX=6.
SCENARIOS=['nominal','wind3','force_step','model20','latency40','payload50','rotoreff30','compound']

# reference
w=np.array([.8,1.1,.6]);
PREF=np.stack([np.sin(w[0]*T),.8*np.sin(w[1]*T+.35),1.2+.35*np.sin(w[2]*T)],1)
VREF=np.stack([w[0]*np.cos(w[0]*T),.8*w[1]*np.cos(w[1]*T+.35),.35*w[2]*np.cos(w[2]*T)],1)
AREF=np.stack([-w[0]**2*np.sin(w[0]*T),-.8*w[1]**2*np.sin(w[1]*T+.35),-.35*w[2]**2*np.sin(w[2]*T)],1)

# LQR/H40
A1=np.array([[1,DT],[0,1.]]); B1=np.array([[.5*DT*DT],[DT]])
A=np.kron(np.eye(3),A1); B=np.zeros((6,3))
for j in range(3):B[2*j:2*j+2,j]=B1[:,0]
Q=np.diag([5,.5,5,.5,6,.6]); R=.18*np.eye(3)
P=solve_discrete_are(A,B,Q,R); KL=np.linalg.solve(R+B.T@P@B,B.T@P@A)
P=Q.copy()
for _ in range(40): KH=np.linalg.solve(R+B.T@P@B,B.T@P@A); P=Q+A.T@P@(A-B@KH)
KP=np.array([4.2,4.2,5.0]); KD=np.array([2.9,2.9,3.2])

def geo(p,v,k): return AREF[k]-KP*(p-PREF[k])-KD*(v-VREF[k])
def gain(p,v,k,K):
    ep=p-PREF[k]; ev=v-VREF[k]; x=np.stack([ep[:,0],ev[:,0],ep[:,1],ev[:,1],ep[:,2],ev[:,2]],1); return AREF[k]-x@K.T

# scenario generator
def scenario(name,seeds):
    n=len(seeds); mass=np.ones(n); E=np.repeat(np.eye(3)[None],n,0); drag=np.full((n,3),.035); delay=np.zeros(n,int)
    ext=np.zeros((STEPS,n,3)); wind=np.zeros((STEPS,n,3)); pn=np.zeros((STEPS,n,3)); vn=np.zeros((STEPS,n,3))
    for j,s in enumerate(seeds):
        rr=np.random.default_rng(910000+int(s)*131+sum(map(ord,name))*17); sp=.0025; sv=.006
        if name=='model20': mass[j]=rr.uniform(.8,1.2); drag[j]*=rr.uniform(.8,1.2); E[j]=np.diag(rr.uniform(.82,1.18,3))
        if name=='payload50': mass[j]=rr.uniform(1.35,1.50); E[j,0,2]=rr.uniform(-.04,.04); E[j,1,2]=rr.uniform(-.04,.04)
        if name=='rotoreff30':
            eff=rr.uniform(.70,1.,3); eff[rr.integers(0,3)]*=rr.uniform(.68,.82); E[j]=np.diag(eff); C=rr.normal(0,.045,(3,3));np.fill_diagonal(C,0);E[j]+=C;sv=.009
        if name=='latency40': delay[j]=4
        if name=='compound': mass[j]=rr.uniform(1.15,1.30); drag[j]*=rr.uniform(.85,1.25);E[j]=np.diag(rr.uniform(.86,1.05,3));delay[j]=2;sp=.004;sv=.010
        if name in ('wind3','compound'):
            d=rr.normal(size=3);d/=np.linalg.norm(d)+1e-9; mag=3 if name=='wind3' else 1.8; mu=mag*d; x=mu.copy()
            for k in range(STEPS): x += .025*(mu-x)+rr.normal(0,.11 if name=='wind3' else .08,3); wind[k,j]=x
        if name in ('force_step','compound'):
            d=rr.normal(size=3);d/=np.linalg.norm(d)+1e-9;mag=.80/.826 if name=='force_step' else .45/.826;on=rr.integers(120,170);off=min(STEPS-10,on+rr.integers(90,140));ext[on:off,j]=mag*d
        nr=np.random.default_rng(420000+int(s)*97+sum(map(ord,name))*13); pn[:,j]=nr.normal(0,sp,(STEPS,3));vn[:,j]=nr.normal(0,sv,(STEPS,3))
    return dict(mass=mass,E=E,drag=drag,delay=delay,ext=ext,wind=.20*wind+.022*wind*np.abs(wind),pn=pn,vn=vn)

# derivative coeffs
xt=np.arange(-8,1)*DT; XX=np.stack([np.ones(9),xt,xt**2],1); PI=np.linalg.pinv(XX); CD1=PI[1]; CD2=2*PI[2]
class FState:
    def __init__(self,n):
        self.n=n;self.m=[];self.d1=np.zeros((n,9));self.d2=np.zeros((n,9));self.ahat=np.zeros((n,3));self.pv=None;self.pu=np.zeros((n,3));self.h=None;self.hp=0
    def feature(self,p,v,k,pn,vn):
        pm=p+pn;vm=v+vn;raw=np.zeros_like(v) if self.pv is None else (vm-self.pv)/DT-self.pu;self.ahat=.92*self.ahat+.08*np.clip(raw,-8,8)
        m=np.concatenate([pm-PREF[k],vm-VREF[k],self.ahat],1);self.m.append(m.copy());
        if len(self.m)>9:self.m.pop(0)
        ar=np.stack(([self.m[0]]*(9-len(self.m)))+self.m,0);d1=np.tensordot(CD1,ar,(0,0));d2=np.tensordot(CD2,ar,(0,0));self.d1=.72*self.d1+.28*d1;self.d2=.82*self.d2+.18*d2;self.pv=vm.copy()
        return np.stack([m,TAU*self.d1,TAU*TAU*self.d2],-1)
    def cmd(self,u):self.pu=u.copy()
    def hist(self,tn):
        if self.h is None:self.h=np.repeat(tn[None],101,0);self.hp=0
        else:self.h[self.hp]=tn;self.hp=(self.hp+1)%101
        ci=(self.hp-1)%101; inds=np.array([(ci-d)%101 for d in range(0,101,10)]);s=self.h[inds];ss=np.sort(s,0)
        cur,lag10,lag100=s[0],s[1],s[-1]; vals=[ss[0],ss[2],ss[5],ss[8],ss[-1]]; blocks=[]
        for c in range(3):blocks += [cur[:,:,c],lag10[:,:,c],lag100[:,:,c]]+[z[:,:,c] for z in vals]
        return np.stack(blocks,-1)

# torch models
class AE(nn.Module):
    def __init__(self,di,h,z):super().__init__();self.e1=nn.Linear(di,h);self.e2=nn.Linear(h,z);self.d1=nn.Linear(z,h);self.d2=nn.Linear(h,di)
    def encode(self,x):return torch.tanh(self.e2(torch.tanh(self.e1(x))))
    def forward(self,x):return self.d2(torch.tanh(self.d1(self.encode(x))))
class MLP(nn.Module):
    def __init__(self):super().__init__();self.a=nn.Linear(27,16);self.b=nn.Linear(16,3)
    def forward(self,x):return torch.tanh(self.b(torch.tanh(self.a(x))))
class CeNN(nn.Module):
    def __init__(self):super().__init__();self.A=nn.Parameter(.05*torch.randn(3,3));self.B=nn.Parameter(.08*torch.randn(3,3,3));self.b=nn.Parameter(torch.zeros(1));self.g=nn.Parameter(torch.ones(3));self.ob=nn.Parameter(torch.zeros(3))
    def conv(self,x,k):
        # x N,H,W; k 3,3
        y=torch.zeros_like(x); xp=torch.nn.functional.pad(x,(1,1,1,1))
        for i in range(3):
            for j in range(3):y += k[i,j]*xp[:,i:i+3,j:j+3]
        return y
    def forward(self,inp):
        # inp N,3,3,3
        x=torch.zeros((len(inp),3,3),device=inp.device)
        ff=torch.zeros_like(x)
        for c in range(3):ff+=self.conv(inp[:,c],self.B[c])
        for _ in range(4):x=x+.32*(-x+self.conv(torch.tanh(x),self.A)+ff+self.b)
        return torch.tanh(torch.tanh(x)[:,2,:]*self.g+self.ob)

def tr_ae(X,di,h,z,seed,ep=4):
    torch.manual_seed(seed);m=AE(di,h,z);o=torch.optim.Adam(m.parameters(),lr=.003);X=torch.tensor(X,dtype=torch.float32);N=len(X);g=torch.Generator().manual_seed(seed)
    for _ in range(ep):
        ix=torch.randperm(N,generator=g)
        for st in range(0,N,4096):
            q=ix[st:st+4096];cl=X[q];no=cl+.06*torch.randn(cl.shape,generator=g);lo=((m(no)-cl)**2).mean();o.zero_grad();lo.backward();o.step()
    return m.eval()
def tr_pred(m,X,Y,seed,ep=5):
    torch.manual_seed(seed);o=torch.optim.Adam(m.parameters(),lr=.003);X=torch.tensor(X,dtype=torch.float32);Y=torch.tensor(Y/RES_MAX,dtype=torch.float32);N=len(X);g=torch.Generator().manual_seed(seed)
    for _ in range(ep):
        ix=torch.randperm(N,generator=g)
        for st in range(0,N,4096):
            q=ix[st:st+4096];lo=((m(X[q])-Y[q])**2).mean();o.zero_grad();lo.backward();o.step()
    return m.eval()
def apply_t(m,x):
    with torch.no_grad():return m(torch.tensor(x,dtype=torch.float32)).numpy()
def enc_t(m,x):
    with torch.no_grad():return m.encode(torch.tensor(x,dtype=torch.float32)).numpy()

# Collect training with Geo, returning ordered features/targets per episode.
def collect(name,seeds):
    sc=scenario(name,seeds);n=len(seeds);p=np.repeat(PREF[0][None],n,0);v=np.repeat(VREF[0][None],n,0);fs=FState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];Ts=[];Ys=[]
    for k in range(STEPS):
        temp=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k]);base=geo(p,v,k);u=np.clip(base,-UMAX,UMAX);fs.cmd(u);q.append(u.copy());q.pop(0);ud=np.array([q[-1-int(d)][j] for j,d in enumerate(sc['delay'])]);act+=(DT/.055)*(ud-act);drag=-sc['drag']*v*np.abs(v);other=sc['wind'][k]+sc['ext'][k]+drag;ctrl=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None];a=ctrl+other
        rhs=base-other; ideal=np.array([np.linalg.solve(sc['E'][j]/sc['mass'][j],rhs[j])-base[j] for j in range(n)]);ideal=np.clip(ideal,-RES_MAX,RES_MAX)
        Ts.append(temp);Ys.append(ideal);v+=DT*a;p+=DT*v
    return np.stack(Ts),np.stack(Ys)

# training ordered arrays: 8 seeds x 4 scenarios = 32 episodes
TT=[];YY=[]
for scn in ['nominal','wind3','force_step','model20']:
    t,y=collect(scn,np.arange(100,108)); TT.append(t);YY.append(y)
TT=np.concatenate(TT,1) # steps,episodes,9,3; YY steps,episodes,3
YY=np.concatenate(YY,1)
flat=TT.reshape(-1,9,3); yflat=YY.reshape(-1,3)
rng=np.random.default_rng(SEED);sel=rng.choice(len(flat),min(8000,len(flat)),False);flat=flat[sel];yflat=yflat[sel]
tm=flat.reshape(-1,3).mean(0);ts=flat.reshape(-1,3).std(0)+1e-6;tn=(flat-tm)/ts
# history training from ordered original sequences, normalized
H=[];HY=[]
for ei in range(TT.shape[1]):
    fs=FState(1)
    for k in range(STEPS):H.append(fs.hist(((TT[k,ei:ei+1]-tm)/ts))[0]);HY.append(YY[k,ei])
H=np.array(H);HY=np.array(HY);hm=H.reshape(-1,24).mean(0);hs=H.reshape(-1,24).std(0)+1e-6;hn=(H-hm)/hs;selh=rng.choice(len(hn),min(8000,len(hn)),False);hn=hn[selh];hy=HY[selh]

# train AEs
cells=tn.reshape(-1,3);cells=cells[rng.choice(len(cells),min(25000,len(cells)),False)]
aeC=tr_ae(cells,3,6,2,11,4)
aeD1=[];st=cells.copy()
for j in range(3):m=tr_ae(st,3,6,2,20+j,3);aeD1.append(m);st=apply_t(m,st)
hcells=hn.reshape(-1,24);hcells=hcells[rng.choice(len(hcells),min(25000,len(hcells)),False)]
aeD2=tr_ae(hcells,24,16,3,31,4);aeD3=[];st=hcells.copy()
for j in range(3):m=tr_ae(st,24,16,3,40+j,3);aeD3.append(m);st=apply_t(m,st)
# train predictors
mlp=tr_pred(MLP(),tn.reshape(len(tn),-1),yflat,51,4)
def latt(x):return np.transpose(x,(0,2,1)).reshape(-1,3,3,3)
cenn=tr_pred(CeNN(),latt(tn),yflat,52,5)
xc=apply_t(aeC,tn.reshape(-1,3)).reshape(len(tn),9,3);cennC=tr_pred(CeNN(),latt(xc),yflat,53,5)
xd=tn.reshape(-1,3)
for m in aeD1:xd=apply_t(m,xd)
xd=xd.reshape(len(tn),9,3);cennD1=tr_pred(CeNN(),latt(xd),yflat,54,5)
z2=enc_t(aeD2,hn.reshape(-1,24)).reshape(len(hn),9,3);z2m=z2.reshape(-1,3).mean(0);z2s=z2.reshape(-1,3).std(0)+1e-6;z2n=(z2-z2m)/z2s;cennD2=tr_pred(CeNN(),latt(z2n),hy,55,5)
st=hn.reshape(-1,24)
for m in aeD3[:-1]:st=apply_t(m,st)
z3=enc_t(aeD3[-1],st).reshape(len(hn),9,3);z3m=z3.reshape(-1,3).mean(0);z3s=z3.reshape(-1,3).std(0)+1e-6;z3n=(z3-z3m)/z3s;cennD3=tr_pred(CeNN(),latt(z3n),hy,56,5)

# export to numpy for fast eval
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
    A,B,b,g,ob=par;x=np.zeros((len(inp),3,3));ff=np.zeros_like(x)
    for c in range(3):ff+=conv_np(inp[:,c],B[c])
    for _ in range(4):x+=.32*(-x+conv_np(np.tanh(x),A)+ff+b)
    return np.tanh(np.tanh(x)[:,2,:]*g+ob)*RES_MAX
AEc=ex_ae(aeC);AED1=[ex_ae(m) for m in aeD1];AED2=ex_ae(aeD2);AED3=[ex_ae(m) for m in aeD3]
C0=ex_c(cenn);CC=ex_c(cennC);CD1p=ex_c(cennD1);CD2p=ex_c(cennD2);CD3p=ex_c(cennD3)
MW1,MB1=linpar(mlp.a);MW2,MB2=linpar(mlp.b)
def mlp_np(x):return np.tanh(np.tanh(x@MW1.T+MB1)@MW2.T+MB2)*RES_MAX

def pred(kind,temp,fs,k):
    n=len(temp);tn=(temp-tm)/ts
    if kind=='MLP':o=mlp_np(tn.reshape(n,-1))
    elif kind=='CeNN':o=cenn_np(C0,latt(tn))
    elif kind=='C':x=ae_np(AEc,tn.reshape(-1,3)).reshape(n,9,3);o=cenn_np(CC,latt(x))
    elif kind=='D1':
        x=tn.reshape(-1,3)
        for a in AED1:x=ae_np(a,x)
        o=cenn_np(CD1p,latt(x.reshape(n,9,3)))
    elif kind in ('D2','D3'):
        h=(fs.hist(tn)-hm)/hs;x=h.reshape(-1,24)
        if kind=='D2':z=ae_np(AED2,x,True);z=(z-z2m)/z2s;o=cenn_np(CD2p,latt(z.reshape(n,9,3)))
        else:
            for a in AED3[:-1]:x=ae_np(a,x)
            z=ae_np(AED3[-1],x,True);z=(z-z3m)/z3s;o=cenn_np(CD3p,latt(z.reshape(n,9,3)))
    return AUTH*np.clip(o,-RES_MAX,RES_MAX)

# rollout fast
def run(name,seeds,ctrl,drop=False):
    sc=scenario(name,seeds);n=len(seeds);p=np.repeat(PREF[0][None],n,0);v=np.repeat(VREF[0][None],n,0)
    for j,s in enumerate(seeds):rr=np.random.default_rng(700000+int(s)*19+sum(map(ord,name))*5);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
    fs=FState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];es=[];ef=[];rs=[]
    for k in range(STEPS):
        te=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k])
        if ctrl=='Geo':base=geo(p,v,k);r=np.zeros_like(base)
        elif ctrl=='LQR':base=gain(p,v,k,KL);r=np.zeros_like(base)
        elif ctrl=='LMPC':base=gain(p,v,k,KH)-.18*fs.ahat;r=np.zeros_like(base)
        else:base=geo(p,v,k);r=pred(ctrl,te,fs,k);r[:] = 0 if (drop and 150<=k<250) else r
        u=np.clip(base+r,-UMAX,UMAX);fs.cmd(u);q.append(u.copy());q.pop(0);ud=np.array([q[-1-int(d)][j] for j,d in enumerate(sc['delay'])]);act+=(DT/.055)*(ud-act);a=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);v+=DT*a;p+=DT*v;e=np.linalg.norm(p-PREF[k],axis=1);es.append(e);ef.append(np.sum(u*u,1));rs.append(np.linalg.norm(r,axis=1))
    es=np.array(es);ef=np.array(ef);rs=np.array(rs);rows=[]
    label={'Geo':'Geo','LQR':'LQR-outer','LMPC':'LMPC-H40','MLP':'Geo+MLP-D2','CeNN':'Geo+CeNN-D2','C':'Geo+AE-CeNN','D1':'Geo+D1-3AE-CeNN','D2':'Geo+D2-HistAE-CeNN','D3':'Geo+D3-3HistAE-CeNN'}[ctrl]+(' dropout' if drop else '')
    for j,s in enumerate(seeds):rows.append([name,label,int(s),np.sqrt(np.mean(es[:,j]**2)),np.max(es[:,j]),float(np.max(es[:,j])<2.5),np.mean(ef[:,j]),np.sqrt(np.mean(rs[:,j]**2))])
    return rows

controllers=['Geo','LQR','LMPC','MLP','CeNN','C','D1','D2','D3'];rows=[];seeds=np.arange(200,240)
for scn in SCENARIOS:
    print('eval',scn,flush=True)
    for c in controllers:rows += run(scn,seeds,c)
# dropout diagnostic 15 paired seeds
for scn in SCENARIOS:
    for c in ['D1','D2','D3']:rows += run(scn,np.arange(200,215),c,True)
cols=['Scenario','Controller','SeedIndex','pos_rmse_m','pos_max_m','success','control_effort','residual_rms_ms2'];tr=pd.DataFrame(rows,columns=cols);tr.to_csv(f'{OUT}/situation_D_trials.csv',index=False)
sm=tr.groupby(['Scenario','Controller'],sort=False).agg(n=('SeedIndex','size'),rmse_mean=('pos_rmse_m','mean'),rmse_sd=('pos_rmse_m','std'),max_mean=('pos_max_m','mean'),success_pct=('success',lambda x:100*x.mean()),effort_mean=('control_effort','mean'),residual_rms=('residual_rms_ms2','mean')).reset_index();sm.to_csv(f'{OUT}/situation_D_summary.csv',index=False)

# paired stats main 100 seeds
props=['Geo+D1-3AE-CeNN','Geo+D2-HistAE-CeNN','Geo+D3-3HistAE-CeNN'];comps=['Geo','LMPC-H40','Geo+CeNN-D2','Geo+AE-CeNN'];st=[]
for scn in SCENARIOS:
  for pr in props:
   for co in comps:
    a=tr[(tr.Scenario==scn)&(tr.Controller==co)].sort_values('SeedIndex').pos_rmse_m.to_numpy()[:40];b=tr[(tr.Scenario==scn)&(tr.Controller==pr)].sort_values('SeedIndex').pos_rmse_m.to_numpy()[:40];d=a-b;rg=np.random.default_rng(987);ii=rg.integers(0,40,(3000,40));bs=d[ii].mean(1);p=wilcoxon(d).pvalue
    st.append([scn,pr,co,a.mean(),b.mean(),d.mean(),100*d.mean()/a.mean(),np.quantile(bs,.025),np.quantile(bs,.975),p])
st=pd.DataFrame(st,columns=['Scenario','Proposal','Comparator','Comparator_RMSE_m','Proposal_RMSE_m','Absolute_improvement_m','Relative_improvement_pct','CI95_low_m','CI95_high_m','Wilcoxon_p']);pv=st.Wilcoxon_p.to_numpy();order=np.argsort(pv);adj=np.empty(len(pv));runv=0
for rank,i in enumerate(order):runv=max(runv,(len(pv)-rank)*pv[i]);adj[i]=min(1,runv)
st['Holm_adjusted_p']=adj;st.to_csv(f'{OUT}/situation_D_paired_statistics.csv',index=False)

# parameters and microbench
def np_count(m):return sum(p.numel() for p in m.parameters())
pars={'CeNN-D2':np_count(cenn),'C_AE-CeNN':np_count(aeC)+np_count(cennC),'D1_3AE-CeNN':sum(np_count(x) for x in aeD1)+np_count(cennD1),'D2_HistAE-CeNN':np_count(aeD2)+np_count(cennD2),'D3_3HistAE-CeNN':sum(np_count(x) for x in aeD3)+np_count(cennD3)}
def bench(fn,N=3000):
    for _ in range(50):fn()
    t=time.perf_counter();
    for _ in range(N):fn()
    return (time.perf_counter()-t)*1e6/N
x3=np.zeros((9,3));x24=np.zeros((9,24));in3=np.zeros((1,3,3,3))
lat={};lat['CeNN-D2']=bench(lambda:cenn_np(C0,in3),1200);lat['C_AE-CeNN']=bench(lambda:cenn_np(CC,latt(ae_np(AEc,x3).reshape(1,9,3))),1200)
lat['D1_3AE-CeNN']=bench(lambda:cenn_np(CD1p,latt(ae_np(AED1[2],ae_np(AED1[1],ae_np(AED1[0],x3))).reshape(1,9,3))),900)
lat['D2_HistAE-CeNN']=bench(lambda:cenn_np(CD2p,latt(((ae_np(AED2,x24,True)-z2m)/z2s).reshape(1,9,3))),900)
lat['D3_3HistAE-CeNN']=bench(lambda:cenn_np(CD3p,latt(((ae_np(AED3[-1],ae_np(AED3[1],ae_np(AED3[0],x24)),True)-z3m)/z3s).reshape(1,9,3))),700)
info={'dt_s':DT,'episode_s':STEPS*DT,'holdout_seeds':'200-239 (40 paired)','dropout_seeds':'200-214 (15 paired diagnostic)','history':{'lag10_s':.1,'lag100_s':1.0,'features_per_original_input':8,'cell_input_D2_D3':24,'boxplot':'five-number summary estimated from 11 evenly spaced samples across lags 0..100'},'authority_multiplier':AUTH,'effective_residual_bound_ms2':AUTH*RES_MAX,'parameters':pars,'numpy_neural_path_microbenchmark_us':lat}
json.dump(info,open(f'{OUT}/situation_D_model_info.json','w'),indent=2)

# figures
import matplotlib.pyplot as plt
sel=['Geo','LMPC-H40','Geo+AE-CeNN']+props;main=sm[(sm.n==40)&sm.Controller.isin(sel)].pivot(index='Scenario',columns='Controller',values='rmse_mean').reindex(SCENARIOS)
ax=main.plot(kind='bar',figsize=(13,5.8));ax.set_yscale('log');ax.set_ylabel('Mean position RMSE [m]');ax.set_title('Situation D: 40-seed held-out comparison');ax.legend(title='',ncol=2,fontsize=8);plt.tight_layout();plt.savefig(f'{OUT}/figD1_rmse_comparison.png',dpi=220,bbox_inches='tight');plt.close()
g=[]
for scn in SCENARIOS:
 c=float(main.loc[scn,'Geo+AE-CeNN'])
 for pr in props:g.append([scn,pr,100*(c-float(main.loc[scn,pr]))/c])
g=pd.DataFrame(g,columns=['Scenario','Proposal','Improvement_vs_C_pct']);g.to_csv(f'{OUT}/situation_D_incremental_vs_C.csv',index=False);ax=g.pivot(index='Scenario',columns='Proposal',values='Improvement_vs_C_pct').reindex(SCENARIOS).plot(kind='bar',figsize=(12,5));ax.axhline(0,color='black',lw=.8);ax.set_ylabel('RMSE improvement vs Situation-C AE-CeNN [%]');ax.set_title('Incremental value of AE depth and history metadata');plt.tight_layout();plt.savefig(f'{OUT}/figD2_incremental_vs_C.png',dpi=220,bbox_inches='tight');plt.close()
# dropout relative to main, 30-seed matched main recompute matched means
dd=[]
for scn in SCENARIOS:
 for key,pr in [('D1','Geo+D1-3AE-CeNN'),('D2','Geo+D2-HistAE-CeNN'),('D3','Geo+D3-3HistAE-CeNN')]:
  a=tr[(tr.Scenario==scn)&(tr.Controller==pr)&(tr.SeedIndex<215)].pos_rmse_m.mean();b=tr[(tr.Scenario==scn)&(tr.Controller==pr+' dropout')].pos_rmse_m.mean();dd.append([scn,key,100*(b-a)/a])
dd=pd.DataFrame(dd,columns=['Scenario','Proposal','Dropout_penalty_pct']);dd.to_csv(f'{OUT}/situation_D_dropout_effect.csv',index=False);ax=dd.pivot(index='Scenario',columns='Proposal',values='Dropout_penalty_pct').reindex(SCENARIOS).plot(kind='bar',figsize=(12,5));ax.axhline(0,color='black',lw=.8);ax.set_ylabel('1 s dropout penalty [%]');ax.set_title('Situation D graceful-degradation diagnostic (15 paired seeds)');plt.tight_layout();plt.savefig(f'{OUT}/figD3_dropout_effect.png',dpi=220,bbox_inches='tight');plt.close()

# snapshot markdown
md=['# SITUATION D — serial autoencoders and history metadata','','Status: provisional frozen exploratory snapshot.','','## Definitions','D.1: three serial 3->6->2->6->3 denoising AEs before a bounded CeNN.','D.2: each of the 27 Situation-C temporal inputs is expanded to 8 metadata values: current, lag-10, lag-100, and [min,Q1,median,Q3,max] over a causal lag-100 window. For tractability the boxplot is estimated from 11 evenly spaced samples across that window. Each lattice cell therefore has 24 AE inputs. A 24->16->3->16->24 AE is used; its 3-D bottleneck drives the same CeNN.','D.3: D.2 history metadata plus three serial 24->16->3->16->24 AEs; the third bottleneck drives the same CeNN.','','## Held-out 40-seed mean position RMSE [m]','',main.round(5).to_markdown(),'','## Parameters','',pd.DataFrame({'Model':pars.keys(),'Parameters':pars.values()}).to_markdown(index=False),'','## Provenance','Independent AdaptiveQuadBench/RotorPy-derived outer-loop reproduction, not a native full AdaptiveQuadBench/acados execution. Within-D paired comparisons use identical seeds and disturbances. A/B/C snapshots are preserved separately and are not overwritten.']
open(f'{OUT}/SITUATION_D.md','w').write('\n'.join(md))
print('\nRESULTS\n',main.round(5).to_string());print('\nINFO\n',json.dumps(info,indent=2))

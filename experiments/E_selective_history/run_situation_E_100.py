import os, json, time, zipfile, hashlib
import numpy as np, pandas as pd, torch
from scipy.stats import wilcoxon
OUT='/mnt/data/situation_E_lag10/final100';os.makedirs(OUT,exist_ok=True)
BASE='/mnt/data/selective_history_e/run_d2_confirmatory_base.py'
text=open(BASE).read().replace('@njit(cache=True)','@njit(cache=False)').replace("OUT='/mnt/data/d2_confirmatory'",f"OUT='{OUT}'")
prefix=text.split("print('Collecting training data...'",1)[0];ns={};exec(prefix,ns)
SCENARIOS=ns['SCENARIOS'];TRAIN_SCENARIOS=ns['TRAIN_SCENARIOS'];TRAIN_SEEDS=ns['TRAIN_SEEDS'];STEPS=ns['STEPS'];WARMUP=ns['WARMUP'];DT=ns['DT'];UMAX=ns['UMAX'];RES_MAX=ns['RES_MAX'];AUTH=ns['AUTH'];SEED=ns['SEED'];TAU=ns['TAU'];PREF=ns['PREF'];VREF=ns['VREF'];scenario=ns['scenario'];collect=ns['collect'];AE=ns['AE'];CeNN=ns['CeNN'];train_ae=ns['train_ae'];train_pred=ns['train_pred'];enc_t=ns['enc_t'];latt=ns['latt'];FeatureState=ns['FeatureState'];geo=ns['geo'];gain=ns['gain'];KL=ns['KL']
np.random.seed(SEED+4500);torch.manual_seed(SEED+4500);torch.set_num_threads(1)
TEST=np.arange(1500,1600)
protocol={'name':'Situation E selective lag10 confirmatory-100','test_seeds':'1500-1599 (100 paired per scenario)','episode_s':12,'frequency_hz':100,'selected_history':'the three current temporal channels delayed by 10 samples/100 ms','active_params_current':494,'active_params_zero6':496,'active_params_selective':496,'claim_boundary':'exploratory pilot only; benchmark-derived simulation'};json.dump(protocol,open(f'{OUT}/protocol.json','w'),indent=2)
print('training',flush=True)
TT=[];YY=[]
for sc in TRAIN_SCENARIOS:
 t,y=collect(sc,TRAIN_SEEDS);TT.append(t);YY.append(y)
TT=np.concatenate(TT,1);YY=np.concatenate(YY,1);tm=TT.reshape(-1,3).mean(0);ts=TT.reshape(-1,3).std(0)+1e-6;TTn=(TT-tm)/ts
Fc=[];Fs=[];Y=[]
for k in range(WARMUP,STEPS,2):Fc.append(TTn[k]);Fs.append(np.concatenate([TTn[k],TTn[k-10]],axis=-1));Y.append(YY[k])
Fc=np.concatenate(Fc);Fs=np.concatenate(Fs);Y=np.concatenate(Y);rz=np.random.default_rng(SEED+1);ix=rz.choice(len(Fc),min(24000,len(Fc)),replace=False);Fc=Fc[ix];Fs=Fs[ix];Y=Y[ix];Fz=np.concatenate([Fc,np.zeros_like(Fc)],axis=-1)
def train(F,di,h,a,b):
 cells=F.reshape(-1,di);rr=np.random.default_rng(a);cells=cells[rr.choice(len(cells),min(50000,len(cells)),replace=False)];ae=train_ae(cells,di,h,3,a,4);z=enc_t(ae,F.reshape(-1,di)).reshape(len(F),9,3);zm=z.reshape(-1,3).mean(0);zs=z.reshape(-1,3).std(0)+1e-6;ce=train_pred(CeNN(),latt((z-zm)/zs),Y,b,5);return ae,zm,zs,ce
Mc=train(Fc,3,64,11,12);Mz=train(Fz,6,45,13,14);Ms=train(Fs,6,45,15,16)
def lin(l):return l.weight.detach().numpy(),l.bias.detach().numpy()
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
def pack(m):
 ae,zm,zs,ce=m;W1,b1=lin(ae.e1);W2,b2=lin(ae.e2);A=km(ce.A.detach().numpy());B=np.stack([km(ce.B.detach().numpy()[c]) for c in range(3)]);return W1,b1,W2,b2,zm,zs,A,B,float(ce.b.detach()),ce.g.detach().numpy(),ce.ob.detach().numpy()
PC,PZ,PS=pack(Mc),pack(Mz),pack(Ms)
def pred(P,F):
 W1,b1,W2,b2,zm,zs,A,B,bb,g,ob=P;z=np.tanh(np.tanh(F.reshape(-1,F.shape[-1])@W1.T+b1)@W2.T+b2).reshape(len(F),9,3);z=(z-zm)/zs;inp=np.transpose(z,(0,2,1));ff=np.zeros((len(F),9))
 for c in range(3):ff+=inp[:,c]@B[c].T
 x=np.zeros_like(ff)
 for _ in range(4):x+=.32*(-x+np.tanh(x)@A.T+ff+bb)
 return AUTH*np.tanh(np.tanh(x.reshape(-1,3,3))[:,2,:]*g+ob)*RES_MAX
cache={}
def getsc(name):
 if name not in cache:cache[name]=scenario(name,TEST)
 return cache[name]
def run(name,kind):
 sc=getsc(name);n=len(TEST);p=np.repeat(PREF[0][None],n,0);v=np.repeat(VREF[0][None],n,0)
 for j,s in enumerate(TEST):
  rr=np.random.default_rng(700000+int(s)*19+sum(map(ord,name))*5);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
 fs=FeatureState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];hist=[];err=np.zeros((STEPS,n));eff=np.zeros((STEPS,n));res=np.zeros((STEPS,n));sat=np.zeros((STEPS,n));perm=np.random.default_rng(999+sum(map(ord,name))).permutation(n)
 for k in range(STEPS):
  te=fs.feature(p,v,k,sc['pn'][k],sc['vn'][k]);tn=(te-tm)/ts;hist.append(tn.copy());
  if len(hist)>11:hist.pop(0)
  lag=hist[0] if len(hist)<11 else hist[-11]
  if kind=='LQR':base=gain(p,v,k,KL);r=np.zeros_like(base)
  else:
   base=geo(p,v,k)
   if kind=='Current':r=pred(PC,tn)
   elif kind=='Zero':r=pred(PZ,np.concatenate([tn,np.zeros_like(tn)],-1))
   elif kind=='Selective':r=pred(PS,np.concatenate([tn,lag],-1))
   elif kind=='Shuffled':r=pred(PS,np.concatenate([tn,lag[perm]],-1))
  raw=base+r;u=np.clip(raw,-UMAX,UMAX);fs.cmd(u);sat[k]=np.any(np.abs(raw)>=UMAX-1e-12,1);q.append(u.copy());q.pop(0);ud=np.empty_like(u)
  for dd in np.unique(sc['delay']):
   m=sc['delay']==dd;ud[m]=q[-1-int(dd)][m]
  act+=(DT/.055)*(ud-act);a=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);v+=DT*a;p+=DT*v;err[k]=np.linalg.norm(p-PREF[k],axis=1);eff[k]=np.sum(u*u,1);res[k]=np.linalg.norm(r,axis=1)
 lab={'LQR':'LQR-outer','Current':'CurrentOnly-AE-CeNN-494','Zero':'ZeroHistory6D-AE-CeNN-496','Selective':'SelectiveLag10-AE-CeNN-496','Shuffled':'SelectiveLag10-shuffled-at-test'}[kind];rows=[]
 for j,s in enumerate(TEST):
  e=err[WARMUP:,j];rows.append([name,lab,int(s),np.sqrt(np.mean(e*e)),np.quantile(e,.95),np.max(e),np.mean(eff[WARMUP:,j]),100*np.mean(sat[WARMUP:,j]),np.sqrt(np.mean(res[WARMUP:,j]**2))])
 return rows
print('evaluation',flush=True);rows=[]
for sc in SCENARIOS:
 for kind in ['LQR','Current','Zero','Selective','Shuffled']:rows+=run(sc,kind)
cols=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','control_effort','saturation_pct','residual_rms_ms2'];df=pd.DataFrame(rows,columns=cols);df.to_csv(f'{OUT}/trials_100seeds.csv',index=False);sm=df.groupby(['Scenario','Controller']).agg(n=('SeedIndex','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),p95_mean=('p95_error_m','mean'),max_mean=('max_error_m','mean'),effort_mean=('control_effort','mean'),sat_mean=('saturation_pct','mean')).reset_index();sm.to_csv(f'{OUT}/summary_100seeds.csv',index=False)
def stat(sc,a,b,seed):
 A=df[(df.Scenario==sc)&(df.Controller==a)].sort_values('SeedIndex');B=df[(df.Scenario==sc)&(df.Controller==b)].sort_values('SeedIndex');d=A.rmse_m.to_numpy()-B.rmse_m.to_numpy();rr=np.random.default_rng(seed);ii=rr.integers(0,len(d),(5000,len(d)));bs=d[ii].mean(1);return [sc,a,b,A.rmse_m.mean(),B.rmse_m.mean(),d.mean(),100*d.mean()/A.rmse_m.mean(),np.quantile(bs,.025),np.quantile(bs,.975),wilcoxon(d).pvalue]
st=[]
for ci,c in enumerate(['CurrentOnly-AE-CeNN-494','ZeroHistory6D-AE-CeNN-496','SelectiveLag10-shuffled-at-test','LQR-outer']):
 for si,sc in enumerate(SCENARIOS):st.append(stat(sc,c,'SelectiveLag10-AE-CeNN-496',1000+100*ci+si))
pd.DataFrame(st,columns=['Scenario','Comparator','Selected','Comparator_RMSE_m','Selected_RMSE_m','Improvement_m','Relative_improvement_pct','CI95_low_m','CI95_high_m','Wilcoxon_p']).to_csv(f'{OUT}/paired_stats_100seeds.csv',index=False)
piv=sm.pivot(index='Scenario',columns='Controller',values='rmse_mean').reindex(SCENARIOS);open(f'{OUT}/SITUATION_E_PILOT_REPORT.md','w').write('# Situation E — compact lag10 selective history (100 paired seeds)\n\n'+piv.round(5).to_markdown())
print(piv.round(5).to_string())

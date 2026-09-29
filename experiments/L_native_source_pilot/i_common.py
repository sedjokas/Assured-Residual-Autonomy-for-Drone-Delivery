import sys, os, numpy as np
sys.path.insert(0,'/mnt/data/h_work')
import h_common as H
R=np.load('/mnt/data/situation_I_riskaware/risk_monitor_pack.npz')
MW1,MB1,MW2,MB2,MD1,MDB1,MD2,MDB2=[R[k] for k in ['e1_w','e1_b','e2_w','e2_b','d1_w','d1_b','d2_w','d2_b']]
OOD_Q99=float(R['OOD_Q99'][0]);OOD_Q9995=float(R['OOD_Q9995'][0]);ACT_FULL=float(R['ACT_FULL'][0]);DYN_FULL=float(R['DYN_FULL'][0]);DYN_ZERO=float(R['DYN_ZERO'][0])
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def smoothstep01(x):x=np.clip(x,0,1);return x*x*(3-2*x)
def confidence_high_good(x,lo,hi):return smoothstep01((x-lo)/(hi-lo)) if hi>lo else np.ones_like(x)
def confidence_low_good(x,lo,hi):return 1-confidence_high_good(x,lo,hi)
def mon_sample_error(tn):
 x=tn.reshape(-1,3);h=np.tanh(x@MW1.T+MB1);z=np.tanh(h@MW2.T+MB2);h2=np.tanh(z@MD1.T+MDB1);y=(h2@MD2.T+MDB2).reshape(tn.shape);return np.sqrt(np.mean((y-tn)**2,axis=(1,2)))
def open_history_pred(tn,lag):
 z=H.cur_lat(tn);(w1,b1)=H.PG['h1'];(w2,b2)=H.PG['h2'];(wg,bg)=H.PG['gate'];zh=np.tanh(np.tanh(lag.reshape(-1,3)@w1.T+b1)@w2.T+b2).reshape(len(tn),9,3);glearn=H.PG['gmax']*sigmoid(np.concatenate([z,zh],-1)@wg.T+bg);return H.AUTH*H.cf(H.CC,z+glearn*zh),glearn
def base_scenario_I(kind,seeds):
 if kind.startswith('wind3_'):return H.scenario('wind3',seeds)
 if kind in ('soft_history_shift','soft_deadline_erosion','deadline_bursts','history_corruption','sensor_spikes','combined_faults'):return H.scenario('force_step',seeds)
 if kind=='severe_compound':
  sc=H.scenario('compound',seeds);sc['wind']*=1.8;sc['ext']*=1.6;return sc
 return H.scenario(kind,seeds)
def deadline_margin_schedule(kind,n,seeds):
 m=np.full((H.STEPS,n),.65)
 if 'deadline_bursts' in kind or 'combined_faults' in kind:m[400:425]=-.1;m[800:815]=-.1
 if 'soft_deadline_erosion' in kind:
  for a,b in [(380,450),(780,850)]:m[a:b]=np.linspace(.65,-.08,b-a)[:,None]
 return m
def custom_masks(kind,n,seeds):
 corrupt=np.zeros((H.STEPS,n),bool);spikes=np.zeros((H.STEPS,n),bool);softood=np.zeros((H.STEPS,n),bool)
 if 'history_corruption' in kind or 'combined_faults' in kind:corrupt[450:500]=True;corrupt[850:880]=True
 if 'sensor_spikes' in kind or 'combined_faults' in kind:
  for j,s in enumerate(seeds):
   rr=np.random.default_rng(1880000+int(s)*31)
   for kk in rr.choice(np.arange(300,1050),3,replace=False):spikes[kk,j]=True
 if 'soft_history_shift' in kind:softood[450:520]=True;softood[850:920]=True
 return corrupt,spikes,softood

def simulate_I(kind,seeds,controller='I',collect_trace=False):
 sc=base_scenario_I(kind,seeds);n=len(seeds);p=np.repeat(H.PREF[0][None],n,0);v=np.repeat(H.VREF[0][None],n,0)
 for j,s in enumerate(seeds):
  rr=np.random.default_rng(1990000+int(s)*23+sum(map(ord,kind))*7);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
 marginT=deadline_margin_schedule(kind,n,seeds);corrupt,spikes,softood=custom_masks(kind,n,seeds);fs=H.FeatureState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];hist=[]
 err=np.zeros((H.STEPS,n));sat=np.zeros((H.STEPS,n));scaleT=np.zeros((H.STEPS,n));alphaT=np.zeros((H.STEPS,n));oodT=np.ones((H.STEPS,n));deadT=np.ones((H.STEPS,n));actT=np.ones((H.STEPS,n));dynT=np.ones((H.STEPS,n));projT=np.ones((H.STEPS,n));hardT=np.ones((H.STEPS,n),bool);consecutive=np.zeros(n,dtype=np.int16)
 for k in range(H.STEPS):
  pn=sc['pn'][k].copy();vn=sc['vn'][k].copy()
  if spikes[k].any():
   for j in np.where(spikes[k])[0]:
    rr=np.random.default_rng(1991000+int(seeds[j])*13+k);pn[j]+=rr.normal(0,.8,3);vn[j]+=rr.normal(0,1.5,3)
  te=fs.feature(p,v,k,pn,vn);tn=(te-H.tm)/H.ts;hist.append(tn.copy());
  if len(hist)>11:hist.pop(0)
  lag=hist[0] if len(hist)<11 else hist[-11].copy()
  if corrupt[k].any():
   for j in np.where(corrupt[k])[0]:
    rr=np.random.default_rng(1992000+int(seeds[j])*17+k);lag[j]=20*lag[j]+rr.normal(0,10,lag[j].shape)
  if softood[k].any():lag[softood[k]]=1.55*lag[softood[k]]+0.35*np.tanh(lag[softood[k]])
  base=H.geo(p,v,k);rc=H.cur_pred(tn)
  if controller=='Current':scale=np.zeros(n);r=rc;alpha=np.zeros(n)
  elif controller=='G':re,_,alpha,_=H.event_pred(tn,lag);scale=np.ones(n);r=re
  else:
   score=H.detector_score(tn,lag);alpha=H.alpha_from_score(score);ropen,_=open_history_pred(tn,lag);delta=ropen-rc
   recerr=np.maximum(mon_sample_error(tn),mon_sample_error(lag));cood=confidence_low_good(recerr,OOD_Q99,OOD_Q9995);cdead=confidence_high_good(marginT[k],.05,.35);rem=(H.UMAX-H.ACT_MARGIN)-np.max(np.abs(base+rc),axis=1);cact=confidence_high_good(rem,0,ACT_FULL);enow=np.linalg.norm(p-H.PREF[k],axis=1);cdyn=confidence_low_good(enow,DYN_FULL,DYN_ZERO)
   detected=alpha>.5;consecutive=np.where(detected,consecutive+1,0);persist=consecutive>=H.PERSIST_N;finite=np.isfinite(tn).all((1,2))&np.isfinite(lag).all((1,2))&np.isfinite(delta).all(1);histok=np.max(np.abs(lag),axis=(1,2))<=H.HIST_ABS_ENV;track=np.sqrt(np.mean(tn[:,0:6,0]**2,axis=1));stateok=track<=H.TRACK_ENV;hard=persist&finite&histok&stateok&(marginT[k]>0);proj=H.action_scale(base+rc,delta);risk=alpha*np.minimum.reduce([cdead,cood,cact,cdyn]);scale=np.where(hard,np.minimum(proj,risk),0);r=rc+scale[:,None]*delta
   oodT[k]=cood;deadT[k]=cdead;actT[k]=cact;dynT[k]=cdyn;projT[k]=proj;hardT[k]=hard
  raw=base+r;u=np.clip(raw,-H.UMAX,H.UMAX);fs.cmd(u);sat[k]=np.any(np.abs(raw)>=H.UMAX-1e-12,1);q.append(u.copy());q.pop(0);ud=np.empty_like(u)
  for dd in np.unique(sc['delay']):m=sc['delay']==dd;ud[m]=q[-1-int(dd)][m]
  act+=(H.DT/.055)*(ud-act);acc=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);v+=H.DT*acc;p+=H.DT*v;err[k]=np.linalg.norm(p-H.PREF[k],axis=1);scaleT[k]=scale;alphaT[k]=alpha
 label={'Current':'CurrentOnly-AE-CeNN-494','G':'G-DetectorGate-noRTA','I':'I-ContinuousRiskGovernor'}[controller];rows=[]
 for j,s in enumerate(seeds):
  e=err[H.WARMUP:,j];rows.append([kind,label,int(s),np.sqrt(np.mean(e*e)),np.quantile(e,.95),np.max(e),100*np.mean(sat[H.WARMUP:,j]),100*np.mean(e>1),np.mean(scaleT[H.WARMUP:,j]),100*np.mean(scaleT[H.WARMUP:,j]>1e-4),np.mean(alphaT[H.WARMUP:,j]),np.mean(oodT[H.WARMUP:,j]),np.mean(deadT[H.WARMUP:,j]),np.mean(actT[H.WARMUP:,j]),np.mean(dynT[H.WARMUP:,j])])
 trace={'error':err,'scale':scaleT,'alpha':alphaT,'ood_conf':oodT,'deadline_conf':deadT,'act_conf':actT,'dyn_conf':dynT,'proj':projT,'hard_valid':hardT,'deadline_margin':marginT,'corrupt':corrupt,'spikes':spikes,'softood':softood} if collect_trace else None
 return rows,trace

def simulate_H_custom(kind,seeds,collect_trace=False):
 sc=base_scenario_I(kind,seeds);n=len(seeds);p=np.repeat(H.PREF[0][None],n,0);v=np.repeat(H.VREF[0][None],n,0)
 for j,s in enumerate(seeds):
  rr=np.random.default_rng(1990000+int(s)*23+sum(map(ord,kind))*7);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
 marginT=deadline_margin_schedule(kind,n,seeds);corrupt,spikes,softood=custom_masks(kind,n,seeds);fs=H.FeatureState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];hist=[]
 err=np.zeros((H.STEPS,n));sat=np.zeros((H.STEPS,n));scaleT=np.zeros((H.STEPS,n));consecutive=np.zeros(n,dtype=np.int16)
 for k in range(H.STEPS):
  pn=sc['pn'][k].copy();vn=sc['vn'][k].copy()
  if spikes[k].any():
   for j in np.where(spikes[k])[0]:
    rr=np.random.default_rng(1991000+int(seeds[j])*13+k);pn[j]+=rr.normal(0,.8,3);vn[j]+=rr.normal(0,1.5,3)
  te=fs.feature(p,v,k,pn,vn);tn=(te-H.tm)/H.ts;hist.append(tn.copy())
  if len(hist)>11:hist.pop(0)
  lag=hist[0] if len(hist)<11 else hist[-11].copy()
  if corrupt[k].any():
   for j in np.where(corrupt[k])[0]:
    rr=np.random.default_rng(1992000+int(seeds[j])*17+k);lag[j]=20*lag[j]+rr.normal(0,10,lag[j].shape)
  if softood[k].any():lag[softood[k]]=1.55*lag[softood[k]]+0.35*np.tanh(lag[softood[k]])
  base=H.geo(p,v,k);rc=H.cur_pred(tn);re,_,a,_=H.event_pred(tn,lag);delta=re-rc
  detected=a>.5;consecutive=np.where(detected,consecutive+1,0);persist=consecutive>=H.PERSIST_N;finite=np.isfinite(tn).all((1,2))&np.isfinite(lag).all((1,2))&np.isfinite(delta).all(1);histok=np.max(np.abs(lag),axis=(1,2))<=H.HIST_ABS_ENV;track=np.sqrt(np.mean(tn[:,0:6,0]**2,axis=1));stateok=track<=H.TRACK_ENV;deadline_ok=marginT[k]>0;proj=H.action_scale(base+rc,delta);admiss=persist&finite&histok&stateok&deadline_ok;scale=np.where(admiss,proj,0);r=rc+scale[:,None]*delta
  raw=base+r;u=np.clip(raw,-H.UMAX,H.UMAX);fs.cmd(u);sat[k]=np.any(np.abs(raw)>=H.UMAX-1e-12,1);q.append(u.copy());q.pop(0);ud=np.empty_like(u)
  for dd in np.unique(sc['delay']):m=sc['delay']==dd;ud[m]=q[-1-int(dd)][m]
  act+=(H.DT/.055)*(ud-act);acc=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);v+=H.DT*acc;p+=H.DT*v;err[k]=np.linalg.norm(p-H.PREF[k],axis=1);scaleT[k]=scale
 rows=[]
 for j,s in enumerate(seeds):
  e=err[H.WARMUP:,j];rows.append([kind,'H-RTA-SupervisedGate',int(s),np.sqrt(np.mean(e*e)),np.quantile(e,.95),np.max(e),100*np.mean(sat[H.WARMUP:,j]),100*np.mean(e>1),np.mean(scaleT[H.WARMUP:,j]),100*np.mean(scaleT[H.WARMUP:,j]>1e-4),np.nan,np.nan,np.nan,np.nan,np.nan])
 trace={'error':err,'scale':scaleT,'deadline_margin':marginT,'corrupt':corrupt,'softood':softood} if collect_trace else None
 return rows,trace

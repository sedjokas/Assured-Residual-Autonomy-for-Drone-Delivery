import sys,time,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data/qcenn_ablation')
from qcenn_eval_core import load_predictor,FeatureState,geo,PREF,STEPS,WARMUP,DT,OUT
sys.path.insert(0,'/mnt/data/AP_QI_CeNN_GitHub/experiments/L2_native_transfer')
import native_validation_l_core as N

def run(scn,seed,arch,steps=800):
 pred,tm,ts,*_=load_predictor(arch)
 params,delay,force_dir,windpar=N.scenario_params(scn,seed); rng=np.random.default_rng(1000000+seed); vr=N.NativeRotorPyCore(params,control_abstraction='cmd_acc',initial_hover=False)
 p0,v0,a0=N.source_flat(0); st=vr.initial_state.copy(); st['x']=p0+rng.normal(0,.012,3); st['v']=v0+rng.normal(0,.018,3)
 fs=FeatureState(1);q=[];err=[];latent=[]
 for k in range(steps):
  p=st['x'];v=st['v'];tn=(fs.feature(p[None],v[None],k,np.zeros((1,3)),np.zeros((1,3)))-tm)/ts;res,mx=pred(tn);base=geo(p[None],v[None],k);u=(base+res)[0];fs.cmd(u[None]);total=u+np.array([0.,0.,9.81]);q.append(total.copy());cmd=q[-1-delay] if len(q)>delay else q[0]
  if scn=='force_step' and 200<=k<400:st['ext_force']=.8*force_dir
  else:st['ext_force']=np.zeros(3)
  if windpar:
   if k==0:wg=N.DrydenWind(*windpar)
   st['wind']=wg.update(DT,rng)
  else:st['wind']=np.zeros(3)
  st=vr.step_rk4(st,{'cmd_acc':cmd},DT,rng);err.append(np.linalg.norm(st['x']-PREF[min(k+1,STEPS-1)]));latent.append(mx[0])
 e=np.array(err)[WARMUP:]
 return {'scenario':scn,'seed':seed,'architecture':arch,'rmse_m':float(np.sqrt(np.mean(e*e))),'p95_m':float(np.quantile(e,.95)),'max_m':float(np.max(e)),'latent_max':float(np.max(latent[WARMUP:]))}

if __name__=='__main__':
 scn=sys.argv[1];seeds=range(18000,18010);archs=['LinearCeNN-494','QCeNN-XX-503','QCeNN-XXXY-512'];rows=[];t=time.time()
 for s in seeds:
  for a in archs:rows.append(run(scn,s,a))
 d=pd.DataFrame(rows);p=OUT/f'native_{scn}_10seeds.csv';d.to_csv(p,index=False);print(p,'elapsed',time.time()-t);print(d.groupby('architecture').rmse_m.mean())

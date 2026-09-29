import os, sys, time, json, math
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp

# Frozen learned/RTA/HOCBF stack from Situations I/K
sys.path.insert(0,'/mnt/data/i_work')
import i_common as I
sys.path.insert(0,'/mnt/data/k_work')
import k_common as K
H=I.H

# AdaptiveQuadBench pinned vehicle parameters: commit d4c273861aa0ce6750818af0b1b63a2a40408e52
D=0.166
BASE_PARAMS={
 'mass':0.826,'Ixx':0.0047,'Iyy':0.005,'Izz':0.0074,'Ixy':0.,'Iyz':0.,'Ixz':0.,'arm_length':D,'com':np.zeros(3),
 'num_rotors':4,'rotor_pos':{
  'r1':D*np.array([np.sin(np.pi/4), np.cos(np.pi/4),0.]),
  'r2':D*np.array([np.sin(np.pi/4),-np.cos(np.pi/4),0.]),
  'r3':D*np.array([-np.sin(np.pi/4),-np.cos(np.pi/4),0.]),
  'r4':D*np.array([-np.sin(np.pi/4), np.cos(np.pi/4),0.])},
 'rotor_directions':np.array([1.,-1.,1.,-1.]), 'rotor_efficiency':np.ones(4),
 'c_Dx':0.,'c_Dy':0.,'c_Dz':0., 'cd1x':0.62,'cd1y':0.62,'cd1z':0.62,'cdz_h':0.,
 'k_eta':7.64e-6,'k_m':7.64e-6*0.014,'k_d':1.19e-4,'k_z':2.32e-4,'k_flap':0.,
 'tau_m':0.005,'rotor_speed_min':0.,'rotor_speed_max':1000.,'motor_noise_std':50.
}

def quat_dot(quat, omega):
 q0,q1,q2,q3=quat
 G=np.array([[q3,q2,-q1,-q0],[-q2,q3,q0,-q1],[q1,-q0,q3,-q2]])
 qd=.5*G.T@omega
 qe=np.sum(quat**2)-1
 return qd-qe*(2*quat)

class NativeRotorPyCore:
 """Source-exact core equations extracted from RotorPy submodule 07e6e2c5... ."""
 def __init__(self,p,control_abstraction='cmd_acc',initial_hover=False):
  self.p={k:(v.copy() if isinstance(v,np.ndarray) else ({kk:vv.copy() for kk,vv in v.items()} if isinstance(v,dict) else v)) for k,v in p.items()}
  self.mass=p['mass']; self.g=9.81
  self.inertia=np.array([[p['Ixx'],p['Ixy'],p['Ixz']],[p['Ixy'],p['Iyy'],p['Iyz']],[p['Ixz'],p['Iyz'],p['Izz']]])
  self.inv_inertia=np.linalg.inv(self.inertia)
  self.weight=np.array([0.,0.,-self.mass*self.g])
  self.rotor_pos=p['rotor_pos']; self.rotor_dir=p['rotor_directions']; self.num_rotors=p['num_rotors']
  self.com=p.get('com',np.zeros(3)); self.rotor_geometry=np.array([self.rotor_pos[k]-self.com for k in self.rotor_pos])
  self.k_eta=p['k_eta']; self.k_m=p['k_m']; self.k_d=p['k_d']; self.k_z=p['k_z']; self.k_flap=p['k_flap']
  self.rotor_efficiency=np.array(p.get('rotor_efficiency',np.ones(self.num_rotors)),float)
  self.rotor_speed_min=p['rotor_speed_min']; self.rotor_speed_max=p['rotor_speed_max']; self.tau_m=p['tau_m']; self.motor_noise=p['motor_noise_std']
  self.drag_matrix=np.diag([p.get('c_Dx',0.),p.get('c_Dy',0.),p.get('c_Dz',0.)])
  self.rotor_drag_matrix=np.diag([self.k_d,self.k_d,self.k_z])
  self.aero_model='other' if 'cd1x' in p else 'rotorpy'
  if self.aero_model=='other': self.drag_matrix=np.diag([p['cd1x'],p['cd1y'],p['cd1z']]); self.cdz_h=p.get('cdz_h',0.)
  self.aero=True; self.control_abstraction=control_abstraction
  self.k_w=1.; self.k_v=10.; self.kp_att=544.; self.kd_att=46.64
  k=self.k_m/self.k_eta
  self.f_to_TM=np.vstack((np.ones((1,self.num_rotors)),np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]),(k*self.rotor_dir).reshape(1,-1)))
  self.TM_to_f=np.linalg.inv(self.f_to_TM)
  hover=np.sqrt(self.mass*self.g/(self.num_rotors*self.k_eta))
  rs=np.full(4,hover if initial_hover else 1788.53)
  self.initial_state={'x':np.zeros(3),'v':np.zeros(3),'q':np.array([0.,0.,0.,1.]),'w':np.zeros(3),'wind':np.zeros(3),'rotor_speeds':rs,'ext_force':np.zeros(3),'ext_torque':np.zeros(3)}
 @staticmethod
 def hat(s): return np.array([[0,-s[2],s[1]],[s[2],0,-s[0]],[-s[1],s[0],0.]])
 @staticmethod
 def pack(st): return np.r_[st['x'],st['v'],st['q'],st['w'],st['wind'],st['rotor_speeds']]
 @staticmethod
 def unpack(s): return {'x':s[0:3].copy(),'v':s[3:6].copy(),'q':s[6:10].copy(),'w':s[10:13].copy(),'wind':s[13:16].copy(),'rotor_speeds':s[16:].copy()}
 def body_wrench(self,body_rates,rotor_speeds,body_airspeed):
  local=body_airspeed[:,None]+self.hat(body_rates)@self.rotor_geometry.T
  kes=self.k_eta*self.rotor_efficiency; kms=self.k_m*self.rotor_efficiency
  T=np.array([0.,0.,1.])[:,None]*(kes*rotor_speeds**2)
  if self.aero:
   if self.aero_model=='rotorpy': Df=-np.linalg.norm(body_airspeed)*self.drag_matrix@body_airspeed
   else:
    Df=-self.drag_matrix@body_airspeed; Df[-1]+=self.cdz_h*(body_airspeed[0]**2+body_airspeed[1]**2)
   Hf=-(rotor_speeds)*(self.rotor_drag_matrix@local)
   # k_flap = 0 in benchmark params; keep exact algebraic zero.
   Mfl=np.zeros_like(Hf)
  else: Df=np.zeros(3); Hf=np.zeros((3,self.num_rotors)); Mfl=np.zeros((3,self.num_rotors))
  Mforce=np.sum(np.stack([self.hat(self.rotor_geometry[i])@(T+Hf)[:,i] for i in range(self.num_rotors)],axis=0),axis=0)
  Myaw=self.rotor_dir*(np.array([0.,0.,1.])[:,None]*(kms*rotor_speeds**2))
  return np.sum(T+Hf,axis=1)+Df, Mforce+np.sum(Myaw+Mfl,axis=1)
 def cmd_motor_speeds(self,state,control):
  if self.control_abstraction=='cmd_motor_speeds': return control['cmd_motor_speeds']
  if self.control_abstraction=='cmd_acc':
   F_des=control['cmd_acc']*self.mass
   R=Rotation.from_quat(state['q']).as_matrix(); b3=R@np.array([0.,0.,1.]); cmd_thrust=np.dot(F_des,b3)
   n=np.linalg.norm(F_des); b3d=F_des/(n+1e-12); c1=np.array([1.,0.,0.]); cr=np.cross(b3d,c1)
   if np.linalg.norm(cr)<1e-8: c1=np.array([0.,1.,0.]); cr=np.cross(b3d,c1)
   b2d=cr/np.linalg.norm(cr); b1d=np.cross(b2d,b3d); Rd=np.stack([b1d,b2d,b3d]).T
   Serr=.5*(Rd.T@R-R.T@Rd); ae=np.array([-Serr[1,2],Serr[0,2],-Serr[0,1]])
   cmd_moment=self.inertia@(-self.kp_att*ae-self.kd_att*state['w'])+np.cross(state['w'],self.inertia@state['w'])
  else: raise ValueError(self.control_abstraction)
  TM=np.r_[cmd_thrust,cmd_moment]; mf=self.TM_to_f@TM; ms=mf/self.k_eta
  return np.sign(ms)*np.sqrt(np.abs(ms))
 def sdot(self,state,control):
  cmd=np.clip(self.cmd_motor_speeds(state,control),self.rotor_speed_min,self.rotor_speed_max)
  extf=state.get('ext_force',np.zeros(3)); extt=state.get('ext_torque',np.zeros(3))
  def fn(t,s):
   st=self.unpack(s); rs=st['rotor_speeds']; R=Rotation.from_quat(st['q']).as_matrix(); ra=(cmd-rs)/self.tau_m
   air=R.T@(st['v']-st['wind']); FB,MB=self.body_wrench(st['w'],rs,air); vdot=(self.weight+R@FB+extf)/self.mass
   wdot=self.inv_inertia@(MB+extt-self.hat(st['w'])@(self.inertia@st['w']))
   return np.r_[st['v'],vdot,quat_dot(st['q'],st['w']),wdot,np.zeros(3),ra]
  return fn

 def step_rk4(self,state,control,dt,rng):
  fn=self.sdot(state,control); s=self.pack(state)
  k1=fn(0,s); k2=fn(dt/2,s+dt*k1/2); k3=fn(dt/2,s+dt*k2/2); k4=fn(dt,s+dt*k3)
  sn=s+dt*(k1+2*k2+2*k3+k4)/6
  o=self.unpack(sn); o['q']/=np.linalg.norm(o['q']); o['rotor_speeds']+=rng.normal(scale=abs(self.motor_noise),size=4); o['rotor_speeds']=np.clip(o['rotor_speeds'],self.rotor_speed_min,self.rotor_speed_max)
  o['ext_force']=state.get('ext_force',np.zeros(3)).copy(); o['ext_torque']=state.get('ext_torque',np.zeros(3)).copy(); return o
 def step(self,state,control,dt,rng):
  fn=self.sdot(state,control); s=self.pack(state); sol=solve_ivp(fn,(0,dt),s,first_step=dt); sn=sol.y[:,-1]; o=self.unpack(sn)
  o['q']/=np.linalg.norm(o['q']); o['rotor_speeds']+=rng.normal(scale=abs(self.motor_noise),size=4); o['rotor_speeds']=np.clip(o['rotor_speeds'],self.rotor_speed_min,self.rotor_speed_max)
  o['ext_force']=state.get('ext_force',np.zeros(3)).copy(); o['ext_torque']=state.get('ext_torque',np.zeros(3)).copy(); return o

# Exact RotorPy Dryden equations at pinned submodule.
class GustModelBase:
 def __init__(self,V,L,sigma,dt=.05):
  self.dt=dt; b=2*np.sqrt(3)*L/V; c=2*L/V; self.alpha=sigma*np.sqrt(2*L/np.pi/V); self.beta=self.alpha*b; self.delta=2*c; self.gamma=c*c
  self.u1=self.u2=self.y1=self.y2=0.
 def run(self,dt,rng):
  C1=1+2*self.delta/dt+4*self.gamma/dt**2; C2=2-8*self.gamma/dt**2; C3=1-2*self.delta/dt+4*self.gamma/dt**2; C4=self.alpha+2*self.beta/dt; C5=2*self.alpha; C6=self.alpha-2*self.beta/dt
  uk=rng.uniform(-1,1); y=(C4*uk+C5*self.u1+C6*self.u2-C2*self.y1-C3*self.y2)/C1; self.u2,self.u1=self.u1,uk; self.y2,self.y1=self.y1,y; return y
 def integrate(self,dt,rng):
  if dt>self.dt:
   t=0.;y=0.
   while t<dt:
    inc=min(self.dt,dt-t); y=self.run(inc,rng); t+=inc
   return y
  return self.run(dt,rng)
class DrydenWind:
 def __init__(self,avg,sig,alt=2.):
  Lz=3.281*alt; Lx=Lz/((.177+.000823*Lz)**1.2); Ly=Lx
  self.nom=np.array(avg,float); self.g=[GustModelBase(1.,Lx/3.281,sig[0]),GustModelBase(1.,Ly/3.281,sig[1]),GustModelBase(1.,Lz/3.281,sig[2])]
 def update(self,dt,rng): return self.nom+np.array([g.integrate(dt,rng) for g in self.g])

# Native AdaptiveQuadBench circle definition from RandomizationConfig: radius=2, center=(-2,0,0), freq defaults 0.2 Hz.
def circle_flat(t):
 r=2.;cx=-2.;om=2*np.pi*.2
 return {'x':np.array([cx+r*np.cos(om*t),r*np.sin(om*t),0.]),'x_dot':np.array([-r*om*np.sin(om*t),r*om*np.cos(om*t),0.]),'x_ddot':np.array([-r*om**2*np.cos(om*t),-r*om**2*np.sin(om*t),0.])}

def source_flat(k):
 # h_common trajectory, continuous derivatives for source-to-native transfer
 t=k*H.DT; w=np.array([.72,.93,.51])
 p=np.array([1.05*np.sin(w[0]*t),.85*np.sin(w[1]*t+.35),1.25+.38*np.sin(w[2]*t)])
 v=np.array([1.05*w[0]*np.cos(w[0]*t),.85*w[1]*np.cos(w[1]*t+.35),.38*w[2]*np.cos(w[2]*t)])
 a=np.array([-1.05*w[0]**2*np.sin(w[0]*t),-.85*w[1]**2*np.sin(w[1]*t+.35),-.38*w[2]**2*np.sin(w[2]*t)])
 return p,v,a

def scenario_params(name,seed):
 rng=np.random.default_rng(seed*313+17)
 p={k:(v.copy() if isinstance(v,np.ndarray) else ({kk:vv.copy() for kk,vv in v.items()} if isinstance(v,dict) else v)) for k,v in BASE_PARAMS.items()}
 delay=0; force_dir=None; wind=None
 if name=='rotoreff30': p['rotor_efficiency']=1+rng.uniform(-.3,.3,4)
 if name=='model20':
  # plant-only parameter mismatch; controller/learned stack remains nominal
  for key in ['mass','Ixx','Iyy','Izz','k_eta','k_m','k_d','k_z','cd1x','cd1y','cd1z']:
   p[key]*=(1+rng.uniform(-.2,.2))
 if name=='payload50': p['mass']*=rng.uniform(1.35,1.50)
 if name=='latency40': delay=4
 if name=='force_step':
  d=rng.normal(size=3); force_dir=d/(np.linalg.norm(d)+1e-12)
 if name=='wind3':
  d=rng.normal(size=3); d/=np.linalg.norm(d)+1e-12; avg=3*d; sig=rng.uniform(30,60,3); wind=(avg,sig)
 return p,delay,force_dir,wind

def learned_command(fs,hist,p,v,k,mode,consecutive,deadline_margin=.65):
 # Uses frozen Situation-I models exactly; reference is H.PREF/H.VREF/H.AREF.
 tn=(fs.feature(p[None],v[None],k,np.zeros((1,3)),np.zeros((1,3)))-H.tm)/H.ts
 hist.append(tn.copy());
 if len(hist)>11: hist.pop(0)
 lag=hist[0] if len(hist)<11 else hist[-11].copy()
 base=H.geo(p[None],v[None],k); rc=H.cur_pred(tn)
 if mode=='Current': scale=np.array([0.]); r=rc
 else:
  score=H.detector_score(tn,lag); alpha=H.alpha_from_score(score); ropen,_=I.open_history_pred(tn,lag); delta=ropen-rc
  recerr=np.maximum(I.mon_sample_error(tn),I.mon_sample_error(lag)); cood=I.confidence_low_good(recerr,I.OOD_Q99,I.OOD_Q9995); cdead=I.confidence_high_good(np.array([deadline_margin]),.05,.35)
  rem=(H.UMAX-H.ACT_MARGIN)-np.max(np.abs(base+rc),axis=1); cact=I.confidence_high_good(rem,0,I.ACT_FULL); enow=np.linalg.norm(p[None]-H.PREF[k],axis=1); cdyn=I.confidence_low_good(enow,I.DYN_FULL,I.DYN_ZERO)
  detected=alpha>.5; consecutive[:] = np.where(detected,consecutive+1,0); persist=consecutive>=H.PERSIST_N; finite=np.isfinite(tn).all((1,2))&np.isfinite(lag).all((1,2))&np.isfinite(delta).all(1); histok=np.max(np.abs(lag),axis=(1,2))<=H.HIST_ABS_ENV; track=np.sqrt(np.mean(tn[:,0:6,0]**2,axis=1)); stateok=track<=H.TRACK_ENV; hard=persist&finite&histok&stateok&(deadline_margin>0); proj=H.action_scale(base+rc,delta); risk=alpha*np.minimum.reduce([cdead,cood,cact,cdyn]); scale=np.where(hard,np.minimum(proj,risk),0); r=rc+scale[:,None]*delta
 u=(base+r)[0]
 if mode=='K':
  cfg={'pmax':np.array([.75,.75,.60]),'vmax':np.array([2.2,2.2,1.7]),'dmax':np.array([1.6,1.6,1.6]),'lam1':3.,'lam2':3.,'lamv':4.,'cmd_lim':5.5}
  uk,act,feas,sv,corr,lo,hi=K.project_hocbf(u[None],p[None],v[None],k,cfg,H.geo(p[None],v[None],k)); u=uk[0]
  return u,float(scale[0]),bool(act[0]),bool(feas[0]),bool(sv[0])
 return u,float(scale[0]),False,True,True

def run_transfer(name,seed,mode='Current',steps=800,fast=True):
 params,delay,force_dir,windpar=scenario_params(name,seed); rng=np.random.default_rng(1000000+seed); vr=NativeRotorPyCore(params,control_abstraction='cmd_acc',initial_hover=False)
 p0,v0,a0=source_flat(0); st=vr.initial_state.copy(); st['x']=p0+rng.normal(0,.012,3); st['v']=v0+rng.normal(0,.018,3)
 fs=H.FeatureState(1); hist=[]; consecutive=np.zeros(1,dtype=np.int16); q=[]; err=[]; active=[]; infeas=[]; setbad=[]; lat=[]
 for k in range(steps):
  p=st['x'];v=st['v']; u,scale,act,fe,sv=learned_command(fs,hist,p,v,k,mode,consecutive)
  fs.cmd(u[None]); total=u+np.array([0.,0.,9.81]); q.append(total.copy());
  if len(q)>delay+1: cmd=q[-1-delay]
  else: cmd=q[0]
  if name=='force_step' and 200<=k<400: st['ext_force']=.8*force_dir
  else: st['ext_force']=np.zeros(3)
  if windpar:
   if k==0: wg=DrydenWind(*windpar)
   st['wind']=wg.update(H.DT,rng)
  else: st['wind']=np.zeros(3)
  t0=time.perf_counter_ns(); st=(vr.step_rk4(st,{'cmd_acc':cmd},H.DT,rng) if fast else vr.step(st,{'cmd_acc':cmd},H.DT,rng)); lat.append(time.perf_counter_ns()-t0)
  err.append(np.linalg.norm(st['x']-H.PREF[min(k+1,H.STEPS-1)])); active.append(act); infeas.append(not fe); setbad.append(not sv)
 e=np.array(err)[H.WARMUP:]; return {'scenario':name,'mode':mode,'seed':seed,'rmse_m':float(np.sqrt(np.mean(e*e))),'p95_m':float(np.quantile(e,.95)),'max_m':float(np.max(e)),'K_active_pct':100*np.mean(active[H.WARMUP:]),'K_infeasible_pct':100*np.mean(infeas[H.WARMUP:]),'K_set_invalid_pct':100*np.mean(setbad[H.WARMUP:]),'plant_step_p50_us':np.percentile(lat,50)/1e3,'plant_step_p99_us':np.percentile(lat,99)/1e3}

def run_circle_geo_sanity(seed,steps=500):
 # Uses a compact exact implementation of AdaptiveQuadBench GeoControl translational/attitude equations.
 rng=np.random.default_rng(2000000+seed); vr=NativeRotorPyCore(BASE_PARAMS,control_abstraction='cmd_acc',initial_hover=True); st=vr.initial_state.copy(); st['x']=np.zeros(3); st['v']=np.zeros(3)
 err=[]
 for k in range(steps):
  t=k*.01; f=circle_flat(t); ep=st['x']-f['x']; ev=st['v']-f['x_dot']; # exact Geo position gains for circle
  ades=f['x_ddot']-np.array([5.,5.,10.])*ep-np.array([4.,4.,8.])*ev
  total=ades+np.array([0.,0.,9.8]) # GeoControl source uses g=9.8 internally
  st=vr.step(st,{'cmd_acc':total},.01,rng); err.append(np.linalg.norm(st['x']-f['x']))
 e=np.array(err)[101:]; return {'seed':seed,'rmse_m':float(np.sqrt(np.mean(e*e))),'p95_m':float(np.quantile(e,.95)),'max_m':float(np.max(e))}

def main(outdir='/mnt/data/native_validation_L'):
 os.makedirs(outdir,exist_ok=True)
 # 1. native-core circle sanity
 sanity=[run_circle_geo_sanity(s) for s in range(30)]
 pd.DataFrame(sanity).to_csv(outdir+'/circle_geo_native_core_sanity.csv',index=False)
 # 2. native RotorPy core transfer of frozen I/K stack
 rows=[]; scenarios=['nominal','wind3','force_step','model20','latency40','payload50','rotoreff30']; modes=['Current','I','K']
 for sc in scenarios:
  for s in range(20):
   for m in modes: rows.append(run_transfer(sc,5000+s,m))
  print('done',sc,flush=True)
 df=pd.DataFrame(rows); df.to_csv(outdir+'/native_core_IK_trials.csv',index=False)
 sm=df.groupby(['scenario','mode']).agg(n=('seed','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),p95_mean=('p95_m','mean'),max_mean=('max_m','mean'),K_active_pct=('K_active_pct','mean'),K_infeasible_pct=('K_infeasible_pct','mean'),K_set_invalid_pct=('K_set_invalid_pct','mean'),plant_step_p50_us=('plant_step_p50_us','median'),plant_step_p99_us=('plant_step_p99_us','median')).reset_index(); sm.to_csv(outdir+'/native_core_IK_summary.csv',index=False)
 # paired K vs I / Current
 stat=[]
 for sc in scenarios:
  sub=df[df.scenario==sc]
  for comp in ['I','Current']:
   a=sub[sub['mode']==comp].sort_values('seed').rmse_m.to_numpy(); b=sub[sub['mode']=='K'].sort_values('seed').rmse_m.to_numpy(); d=a-b
   stat.append([sc,comp,a.mean(),b.mean(),d.mean(),100*d.mean()/a.mean()])
 pd.DataFrame(stat,columns=['scenario','comparator','comparator_rmse','K_rmse','improvement_m','improvement_pct']).to_csv(outdir+'/native_core_paired_effects.csv',index=False)
 # timing learned logic only on synthetic representative state batch
 # measure 50k single-state calls using frozen stack features from nominal rollout approximation
 p=H.PREF[300].copy();v=H.VREF[300].copy();fs=H.FeatureState(1);hist=[];con=np.zeros(1,dtype=np.int16)
 times=[]
 for k in range(300,1300):
  kk=min(k,H.STEPS-1); t0=time.perf_counter_ns(); learned_command(fs,hist,p,v,kk,'K',con); times.append(time.perf_counter_ns()-t0)
 timing={'learned_IK_logic_p50_us':float(np.percentile(times,50)/1e3),'learned_IK_logic_p95_us':float(np.percentile(times,95)/1e3),'learned_IK_logic_p99_us':float(np.percentile(times,99)/1e3),'platform':'container CPU Python/NumPy; not HIL/embedded'}
 json.dump(timing,open(outdir+'/software_timing.json','w'),indent=2)
 manifest={'adaptivequadbench_commit':'d4c273861aa0ce6750818af0b1b63a2a40408e52','rotorpy_submodule_commit':'07e6e2c57d55fc563ac55f9c44153c2769f0ae73','execution_scope':'source-exact native-core equations/controller-parameter extraction; NOT full package-native run_eval','acados_native_executed':False,'reason_acados':'acados_template and compiled libacados absent; container has no network/DNS to install them','n_seeds_per_native_transfer_scenario':20,'episode_s':8.0,'control_hz':100}
 json.dump(manifest,open(outdir+'/manifest.json','w'),indent=2)
 print(sm.to_string(index=False)); print('timing',timing)
if __name__=='__main__': main()

import os, sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0,'/mnt/data/native_validation_L2_py')
import native_validation_l_core as L
sys.path.insert(0,'/mnt/data/k_work')
import k_common as K
H=L.H

CFG_JSON='/mnt/data/native_validation_L2_py/L2_frozen_native_hocbf_config.json'
with open(CFG_JSON) as f:
    _cfg=json.load(f)
K2CFG={k:(np.asarray(v,float) if k in ('pmax','vmax','dmax') else v) for k,v in _cfg.items() if k in ('pmax','vmax','dmax','lam1','lam2','lamv','cmd_lim')}


def fault_addition(kind, seed, k):
    add=np.zeros(3); active=False
    if kind=='proposal_impulse':
        # Preserve the first original K impulse window that lies inside the 6-s L.2 episode.
        if 400 <= k < 450:
            rr=np.random.default_rng(4100000+int(seed)*37)
            d=rr.normal(size=3); d/=np.linalg.norm(d)+1e-12
            add=8.0*d; active=True
    elif kind=='proposal_stuck':
        if 450 <= k < 550:
            rr=np.random.default_rng(4200000+int(seed)*41)
            d=rr.normal(size=3); d/=np.linalg.norm(d)+1e-12
            add=5.5*d; active=True
    return add,active


def run_trial(base_scenario, seed, controller='I', steps=600, fault_kind=None, cfg=None):
    if cfg is None: cfg=K2CFG
    params,delay,force_dir,windpar=L.scenario_params(base_scenario,seed)
    rng=np.random.default_rng(1000000+seed)
    vr=L.NativeRotorPyCore(params,control_abstraction='cmd_acc',initial_hover=False)
    p0,v0,a0=L.source_flat(0)
    st=vr.initial_state.copy(); st['x']=p0+rng.normal(0,.012,3); st['v']=v0+rng.normal(0,.018,3)
    fs=H.FeatureState(1); hist=[]; consecutive=np.zeros(1,dtype=np.int16); q=[]
    err=[]; active=[]; infeas=[]; setbad=[]; corr=[]; auth=[]; fault=[]; dvals=[]
    for k in range(steps):
        p=st['x']; v=st['v']
        u_prop,scale,_,_,_=L.learned_command(fs,hist,p,v,k,'I',consecutive)
        add,fmask=fault_addition(fault_kind,seed,k) if fault_kind else (np.zeros(3),False)
        u_prop_fault=u_prop+add
        if controller=='K2':
            base=H.geo(p[None],v[None],k)
            uk,pa,fe,sv,cc,_,_=K.project_hocbf(u_prop_fault[None],p[None],v[None],k,cfg,base)
            u=uk[0]; active.append(bool(pa[0])); infeas.append(bool(not fe[0])); setbad.append(bool(not sv[0])); corr.append(float(cc[0]))
        else:
            u=np.clip(u_prop_fault,-H.UMAX,H.UMAX); active.append(False); infeas.append(False); setbad.append(False); corr.append(0.0)
        fs.cmd(u[None])
        total=u+np.array([0.,0.,9.81]); q.append(total.copy())
        if len(q)>delay+1: cmd=q[-1-delay]
        else: cmd=q[0]
        if base_scenario=='force_step' and 200<=k<400: st['ext_force']=.8*force_dir
        else: st['ext_force']=np.zeros(3)
        if windpar:
            if k==0: wg=L.DrydenWind(*windpar)
            st['wind']=wg.update(H.DT,rng)
        else: st['wind']=np.zeros(3)
        # Exact source-core instantaneous vdot before integration; compare to current filtered relative acceleration.
        fn=vr.sdot(st,{'cmd_acc':cmd}); svec=vr.pack(st); vdot=fn(0,svec)[3:6]
        dvals.append(vdot-u)
        st=vr.step_rk4(st,{'cmd_acc':cmd},H.DT,rng)
        err.append(np.linalg.norm(st['x']-H.PREF[min(k+1,H.STEPS-1)]))
        auth.append(float(scale)); fault.append(bool(fmask))
    sl=slice(H.WARMUP,None)
    e=np.asarray(err)[sl]; act=np.asarray(active)[sl]; inf=np.asarray(infeas)[sl]; sb=np.asarray(setbad)[sl]; co=np.asarray(corr)[sl]; au=np.asarray(auth)[sl]; fm=np.asarray(fault)[sl]; dv=np.asarray(dvals)[sl]
    exceed=np.any(np.abs(dv)>np.asarray(cfg['dmax'])[None,:],axis=1)
    scenario_name=base_scenario if fault_kind is None else f'{base_scenario}_{fault_kind}'
    return {
        'scenario':scenario_name,'base_scenario':base_scenario,'seed':int(seed),'controller':controller,
        'rmse_m':float(np.sqrt(np.mean(e*e))),'p95_m':float(np.quantile(e,.95)),'max_m':float(np.max(e)),
        'excursion_gt1m_pct':100*float(np.mean(e>1.0)),
        'filter_active_pct':100*float(np.mean(act)),'filter_infeasible_pct':100*float(np.mean(inf)),
        'filter_set_invalid_pct':100*float(np.mean(sb)),'filter_correction_mean':float(np.mean(co)),
        'I_authority_mean':float(np.mean(au)),'fault_window_pct':100*float(np.mean(fm)),
        'd_bound_exceed_pct':100*float(np.mean(exceed)),
        'd_abs_max_x':float(np.max(np.abs(dv[:,0]))),'d_abs_max_y':float(np.max(np.abs(dv[:,1]))),'d_abs_max_z':float(np.max(np.abs(dv[:,2])))
    }

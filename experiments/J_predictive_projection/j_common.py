import sys,os,numpy as np
sys.path.insert(0,'/mnt/data/i_work')
import i_common as I
H=I.H

# Independent predictive admissibility projection. No learned parameters.
def predictive_box(p,v,k,h,epos,evel,cmd_lim=5.5):
    e=p-H.PREF[k]; ev=v-H.VREF[k]; ar=H.AREF[k]
    epos=np.asarray(epos); evel=np.asarray(evel)
    lo=np.full_like(p,-cmd_lim,dtype=float); hi=np.full_like(p,cmd_lim,dtype=float)
    # velocity horizon constraints: |ev + h*(u-ar)| <= evel
    vlo=ar+(-evel-ev)/h; vhi=ar+(evel-ev)/h
    # position horizon constraints: |e + h*ev + 0.5*h^2*(u-ar)| <= epos
    plo=ar+2*(-epos-e-h*ev)/(h*h); phi=ar+2*(epos-e-h*ev)/(h*h)
    lo=np.maximum.reduce([lo,vlo,plo]); hi=np.minimum.reduce([hi,vhi,phi])
    feasible=np.all(lo<=hi,axis=1)
    return lo,hi,feasible

def project_command(u_prop,p,v,k,proj_cfg,base_nominal):
    lo,hi,feas=predictive_box(p,v,k,proj_cfg['h'],proj_cfg['epos'],proj_cfg['evel'],proj_cfg['cmd_lim'])
    u=np.minimum(np.maximum(u_prop,lo),hi)
    # Infeasible predictive boxes fall back to nominal stable command, with physical command margin only.
    fb=np.clip(base_nominal,-proj_cfg['cmd_lim'],proj_cfg['cmd_lim'])
    u=np.where(feas[:,None],u,fb)
    correction=np.linalg.norm(u-u_prop,axis=1)
    active=(correction>1e-10)|(~feas)
    return u,active,feas,correction,lo,hi

def base_scenario_J(kind,seeds):
    if kind.startswith('wind3_'): return I.base_scenario_I('wind3',seeds)
    if kind.startswith('compound_'): return I.base_scenario_I('compound',seeds)
    if kind.startswith('force_step_'): return I.base_scenario_I('force_step',seeds)
    if kind=='severe_compound': return I.base_scenario_I('severe_compound',seeds)
    return I.base_scenario_I(kind,seeds)

def fault_schedule(kind,n,seeds):
    add=np.zeros((H.STEPS,n,3)); fault=np.zeros((H.STEPS,n),bool)
    if 'proposal_impulse' in kind:
        for j,s in enumerate(seeds):
            rr=np.random.default_rng(3100000+int(s)*37)
            for a,b in [(400,450),(800,850)]:
                d=rr.normal(size=3); d/=np.linalg.norm(d)+1e-12
                add[a:b,j]=8.0*d; fault[a:b,j]=True
    if 'proposal_stuck' in kind:
        for j,s in enumerate(seeds):
            rr=np.random.default_rng(3200000+int(s)*41); d=rr.normal(size=3); d/=np.linalg.norm(d)+1e-12
            add[450:550,j]=5.5*d; fault[450:550,j]=True
    return add,fault

def simulate_J(kind,seeds,controller='J',proj_cfg=None,collect_trace=False):
    if proj_cfg is None: proj_cfg={'h':.25,'epos':np.array([.65,.65,.50]),'evel':np.array([1.7,1.7,1.3]),'cmd_lim':5.5}
    basekind=kind
    for suff in ['_proposal_impulse','_proposal_stuck']:
        if basekind.endswith(suff): basekind=basekind[:-len(suff)]
    sc=base_scenario_J(basekind,seeds); n=len(seeds)
    p=np.repeat(H.PREF[0][None],n,0);v=np.repeat(H.VREF[0][None],n,0)
    for j,s in enumerate(seeds):
        rr=np.random.default_rng(3300000+int(s)*23+sum(map(ord,kind))*7);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
    marginT=I.deadline_margin_schedule(basekind,n,seeds);corrupt,spikes,softood=I.custom_masks(basekind,n,seeds)
    addfault,faultmask=fault_schedule(kind,n,seeds)
    fs=H.FeatureState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];hist=[]
    err=np.zeros((H.STEPS,n));sat=np.zeros((H.STEPS,n));projA=np.zeros((H.STEPS,n),bool);feasT=np.ones((H.STEPS,n),bool);corrT=np.zeros((H.STEPS,n));scaleT=np.zeros((H.STEPS,n));consecutive=np.zeros(n,dtype=np.int16)
    for k in range(H.STEPS):
        pn=sc['pn'][k].copy();vn=sc['vn'][k].copy(); te=fs.feature(p,v,k,pn,vn);tn=(te-H.tm)/H.ts;hist.append(tn.copy())
        if len(hist)>11:hist.pop(0)
        lag=hist[0] if len(hist)<11 else hist[-11].copy();base=H.geo(p,v,k);rc=H.cur_pred(tn)
        # Frozen Situation-I authority law.
        score=H.detector_score(tn,lag);alpha=H.alpha_from_score(score);ropen,_=I.open_history_pred(tn,lag);delta=ropen-rc
        recerr=np.maximum(I.mon_sample_error(tn),I.mon_sample_error(lag));cood=I.confidence_low_good(recerr,I.OOD_Q99,I.OOD_Q9995);cdead=I.confidence_high_good(marginT[k],.05,.35);rem=(H.UMAX-H.ACT_MARGIN)-np.max(np.abs(base+rc),axis=1);cact=I.confidence_high_good(rem,0,I.ACT_FULL);enow=np.linalg.norm(p-H.PREF[k],axis=1);cdyn=I.confidence_low_good(enow,I.DYN_FULL,I.DYN_ZERO)
        detected=alpha>.5;consecutive=np.where(detected,consecutive+1,0);persist=consecutive>=H.PERSIST_N;finite=np.isfinite(tn).all((1,2))&np.isfinite(lag).all((1,2))&np.isfinite(delta).all(1);histok=np.max(np.abs(lag),axis=(1,2))<=H.HIST_ABS_ENV;track=np.sqrt(np.mean(tn[:,0:6,0]**2,axis=1));stateok=track<=H.TRACK_ENV;hard=persist&finite&histok&stateok&(marginT[k]>0);proj=H.action_scale(base+rc,delta);risk=alpha*np.minimum.reduce([cdead,cood,cact,cdyn]);scale=np.where(hard,np.minimum(proj,risk),0);rI=rc+scale[:,None]*delta;scaleT[k]=scale
        if controller=='Nominal': u_prop=base
        elif controller=='Current': u_prop=base+rc
        else: u_prop=base+rI
        # Post-governor proposal corruption challenge, simulating a bad learned/control proposal before independent J projection.
        if controller in ('I','J') and faultmask[k].any(): u_prop=u_prop+addfault[k]
        if controller=='J':
            u,pa,fe,corr,_,_=project_command(u_prop,p,v,k,proj_cfg,base);projA[k]=pa;feasT[k]=fe;corrT[k]=corr
        else: u=np.clip(u_prop,-H.UMAX,H.UMAX)
        fs.cmd(u);sat[k]=np.any(np.abs(u_prop)>=H.UMAX-1e-12,1);q.append(u.copy());q.pop(0);ud=np.empty_like(u)
        for dd in np.unique(sc['delay']):m=sc['delay']==dd;ud[m]=q[-1-int(dd)][m]
        act+=(H.DT/.055)*(ud-act);acc=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);v+=H.DT*acc;p+=H.DT*v;err[k]=np.linalg.norm(p-H.PREF[k],axis=1)
    label={'Nominal':'Nominal-Geo','Current':'CurrentOnly-AE-CeNN-494','I':'I-RiskGovernor-unprojected','J':'J-IndependentSafetyProjection'}[controller]
    rows=[]
    for j,s in enumerate(seeds):
        e=err[H.WARMUP:,j]
        rows.append([kind,label,int(s),np.sqrt(np.mean(e*e)),np.quantile(e,.95),np.max(e),100*np.mean(e>.5),100*np.mean(e>1.0),100*np.mean(sat[H.WARMUP:,j]),100*np.mean(projA[H.WARMUP:,j]),100*np.mean(~feasT[H.WARMUP:,j]),np.mean(corrT[H.WARMUP:,j]),np.mean(scaleT[H.WARMUP:,j])])
    tr=None
    if collect_trace: tr={'error':err,'projection_active':projA,'feasible':feasT,'projection_correction':corrT,'fault':faultmask,'authority_scale':scaleT}
    return rows,tr

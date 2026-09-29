import sys, numpy as np
sys.path.insert(0,'/mnt/data/i_work')
import i_common as I
H=I.H


def hocbf_box(p,v,k,cfg):
    """Robust HOCBF/CBF command interval for relative double-integrator surrogate.
    Guarantee is model-relative: e'' = u-a_ref+d, |d_j|<=dmax_j.
    Position set additionally requires psi1>=0 at the current state.
    """
    ep=p-H.PREF[k]; ev=v-H.VREF[k]; ar=H.AREF[k]
    pmax=np.asarray(cfg['pmax'],float); vmax=np.asarray(cfg['vmax'],float); dmax=np.asarray(cfg['dmax'],float)
    lam1=float(cfg['lam1']); lam2=float(cfg['lam2']); lamv=float(cfg['lamv']); ulim=float(cfg['cmd_lim'])
    k1=lam1+lam2; k0=lam1*lam2
    # robust HOCBF position intervals
    lo_p = ar + dmax - k1*ev - k0*(ep+pmax)
    hi_p = ar - dmax - k1*ev + k0*(pmax-ep)
    # robust first-order velocity CBF intervals
    lo_v = ar + dmax - lamv*(ev+vmax)
    hi_v = ar - dmax + lamv*(vmax-ev)
    lo=np.maximum.reduce([np.full_like(p,-ulim), lo_p, lo_v])
    hi=np.minimum.reduce([np.full_like(p, ulim), hi_p, hi_v])
    feasible=np.all(lo<=hi,axis=1)
    # Current-state membership in robust HOCBF admissible set.
    hpu=pmax-ep; hpl=ep+pmax
    psi_u=-ev+lam1*hpu; psi_l=ev+lam1*hpl
    hvu=vmax-ev; hvl=ev+vmax
    set_valid=np.all((hpu>=0)&(hpl>=0)&(psi_u>=0)&(psi_l>=0)&(hvu>=0)&(hvl>=0),axis=1)
    return lo,hi,feasible,set_valid


def project_hocbf(u_prop,p,v,k,cfg,base_nominal):
    lo,hi,feas,set_valid=hocbf_box(p,v,k,cfg)
    u=np.minimum(np.maximum(u_prop,lo),hi)
    # If robust constraints are infeasible, use bounded nominal backup.
    fb=np.clip(base_nominal,-cfg['cmd_lim'],cfg['cmd_lim'])
    u=np.where(feas[:,None],u,fb)
    corr=np.linalg.norm(u-u_prop,axis=1)
    active=(corr>1e-10)|(~feas)
    return u,active,feas,set_valid,corr,lo,hi



def predictive_box_J(p,v,k,cfg):
    ep=p-H.PREF[k]; ev=v-H.VREF[k]; ar=H.AREF[k]
    epos=np.asarray(cfg['epos'],float); evel=np.asarray(cfg['evel'],float); h=float(cfg['h']); ulim=float(cfg['cmd_lim'])
    lo=np.full_like(p,-ulim,dtype=float); hi=np.full_like(p,ulim,dtype=float)
    vlo=ar+(-evel-ev)/h; vhi=ar+(evel-ev)/h
    plo=ar+2*(-epos-ep-h*ev)/(h*h); phi=ar+2*(epos-ep-h*ev)/(h*h)
    lo=np.maximum.reduce([lo,vlo,plo]); hi=np.minimum.reduce([hi,vhi,phi])
    feasible=np.all(lo<=hi,axis=1)
    return lo,hi,feasible

def project_J(u_prop,p,v,k,cfg,base_nominal):
    lo,hi,feas=predictive_box_J(p,v,k,cfg)
    u=np.minimum(np.maximum(u_prop,lo),hi)
    fb=np.clip(base_nominal,-cfg['cmd_lim'],cfg['cmd_lim'])
    u=np.where(feas[:,None],u,fb)
    corr=np.linalg.norm(u-u_prop,axis=1); active=(corr>1e-10)|(~feas)
    return u,active,feas,corr

def base_scenario_K(kind,seeds):
    if kind.startswith('wind3_'): return I.base_scenario_I('wind3',seeds)
    if kind.startswith('compound_'): return I.base_scenario_I('compound',seeds)
    if kind.startswith('force_step_'): return I.base_scenario_I('force_step',seeds)
    if kind=='severe_compound': return I.base_scenario_I('severe_compound',seeds)
    return I.base_scenario_I(kind,seeds)


def fault_schedule(kind,n,seeds):
    add=np.zeros((H.STEPS,n,3)); fault=np.zeros((H.STEPS,n),bool)
    if 'proposal_impulse' in kind:
        for j,s in enumerate(seeds):
            rr=np.random.default_rng(4100000+int(s)*37)
            for a,b in [(400,450),(800,850)]:
                d=rr.normal(size=3);d/=np.linalg.norm(d)+1e-12
                add[a:b,j]=8.0*d; fault[a:b,j]=True
    if 'proposal_stuck' in kind:
        for j,s in enumerate(seeds):
            rr=np.random.default_rng(4200000+int(s)*41); d=rr.normal(size=3);d/=np.linalg.norm(d)+1e-12
            add[450:550,j]=5.5*d; fault[450:550,j]=True
    return add,fault


def simulate_K(kind,seeds,controller='K',cfg=None,collect_trace=False):
    if cfg is None:
        cfg={'pmax':np.array([.65,.65,.50]),'vmax':np.array([1.8,1.8,1.3]),'dmax':np.array([.9,.9,.75]),'lam1':3.0,'lam2':3.0,'lamv':4.0,'cmd_lim':5.5}
    basekind=kind
    for suff in ['_proposal_impulse','_proposal_stuck']:
        if basekind.endswith(suff): basekind=basekind[:-len(suff)]
    sc=base_scenario_K(basekind,seeds);n=len(seeds)
    p=np.repeat(H.PREF[0][None],n,0);v=np.repeat(H.VREF[0][None],n,0)
    for j,s in enumerate(seeds):
        rr=np.random.default_rng(4300000+int(s)*23+sum(map(ord,kind))*7);p[j]+=rr.normal(0,.012,3);v[j]+=rr.normal(0,.018,3)
    marginT=I.deadline_margin_schedule(basekind,n,seeds);corrupt,spikes,softood=I.custom_masks(basekind,n,seeds)
    addfault,faultmask=fault_schedule(kind,n,seeds)
    fs=H.FeatureState(n);act=np.zeros((n,3));q=[np.zeros((n,3)) for _ in range(5)];hist=[]
    err=np.zeros((H.STEPS,n));sat=np.zeros((H.STEPS,n));active=np.zeros((H.STEPS,n),bool);feasT=np.ones((H.STEPS,n),bool);setT=np.ones((H.STEPS,n),bool);corrT=np.zeros((H.STEPS,n));scaleT=np.zeros((H.STEPS,n));distT=np.zeros((H.STEPS,n,3));consecutive=np.zeros(n,dtype=np.int16)
    for k in range(H.STEPS):
        pn=sc['pn'][k].copy();vn=sc['vn'][k].copy();te=fs.feature(p,v,k,pn,vn);tn=(te-H.tm)/H.ts;hist.append(tn.copy())
        if len(hist)>11:hist.pop(0)
        lag=hist[0] if len(hist)<11 else hist[-11].copy();base=H.geo(p,v,k);rc=H.cur_pred(tn)
        score=H.detector_score(tn,lag);alpha=H.alpha_from_score(score);ropen,_=I.open_history_pred(tn,lag);delta=ropen-rc
        recerr=np.maximum(I.mon_sample_error(tn),I.mon_sample_error(lag));cood=I.confidence_low_good(recerr,I.OOD_Q99,I.OOD_Q9995);cdead=I.confidence_high_good(marginT[k],.05,.35);rem=(H.UMAX-H.ACT_MARGIN)-np.max(np.abs(base+rc),axis=1);cact=I.confidence_high_good(rem,0,I.ACT_FULL);enow=np.linalg.norm(p-H.PREF[k],axis=1);cdyn=I.confidence_low_good(enow,I.DYN_FULL,I.DYN_ZERO)
        detected=alpha>.5;consecutive=np.where(detected,consecutive+1,0);persist=consecutive>=H.PERSIST_N;finite=np.isfinite(tn).all((1,2))&np.isfinite(lag).all((1,2))&np.isfinite(delta).all(1);histok=np.max(np.abs(lag),axis=(1,2))<=H.HIST_ABS_ENV;track=np.sqrt(np.mean(tn[:,0:6,0]**2,axis=1));stateok=track<=H.TRACK_ENV;hard=persist&finite&histok&stateok&(marginT[k]>0);proj=H.action_scale(base+rc,delta);risk=alpha*np.minimum.reduce([cdead,cood,cact,cdyn]);scale=np.where(hard,np.minimum(proj,risk),0);rI=rc+scale[:,None]*delta;scaleT[k]=scale
        if controller=='Nominal':u_prop=base
        elif controller=='Current':u_prop=base+rc
        else:u_prop=base+rI
        if controller in ('I','J','K') and faultmask[k].any():u_prop=u_prop+addfault[k]
        if controller=='K':
            u,pa,fe,sv,corr,_,_=project_hocbf(u_prop,p,v,k,cfg,base);active[k]=pa;feasT[k]=fe;setT[k]=sv;corrT[k]=corr
        elif controller=='J':
            jcfg={'h':.30,'epos':np.array([.55,.55,.45]),'evel':np.array([1.5,1.5,1.1]),'cmd_lim':5.5}
            u,pa,fe,corr=project_J(u_prop,p,v,k,jcfg,base);active[k]=pa;feasT[k]=fe;corrT[k]=corr
        else:
            u=np.clip(u_prop,-H.UMAX,H.UMAX)
        fs.cmd(u);sat[k]=np.any(np.abs(u_prop)>=H.UMAX-1e-12,1);q.append(u.copy());q.pop(0);ud=np.empty_like(u)
        for dd in np.unique(sc['delay']):m=sc['delay']==dd;ud[m]=q[-1-int(dd)][m]
        act+=(H.DT/.055)*(ud-act);acc=np.einsum('nij,nj->ni',sc['E'],act)/sc['mass'][:,None]+sc['wind'][k]+sc['ext'][k]-sc['drag']*v*np.abs(v);distT[k]=acc-u;v+=H.DT*acc;p+=H.DT*v;err[k]=np.linalg.norm(p-H.PREF[k],axis=1)
    label={'Nominal':'Nominal-Geo','Current':'CurrentOnly-AE-CeNN-494','I':'I-RiskGovernor-unfiltered','J':'J-PredictiveProjection','K':'K-RobustHOCBF-Filter'}[controller]
    rows=[]
    for j,s in enumerate(seeds):
        e=err[H.WARMUP:,j]
        rows.append([kind,label,int(s),np.sqrt(np.mean(e*e)),np.quantile(e,.95),np.max(e),100*np.mean(e>.5),100*np.mean(e>1),100*np.mean(sat[H.WARMUP:,j]),100*np.mean(active[H.WARMUP:,j]),100*np.mean(~feasT[H.WARMUP:,j]),100*np.mean(~setT[H.WARMUP:,j]),np.mean(corrT[H.WARMUP:,j]),np.mean(scaleT[H.WARMUP:,j])])
    tr=None
    if collect_trace:tr={'error':err,'filter_active':active,'feasible':feasT,'set_valid':setT,'correction':corrT,'fault':faultmask,'authority_scale':scaleT,'model_disturbance':distT}
    return rows,tr

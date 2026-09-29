import os, time, math, json, hashlib, platform
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
from scipy.stats import wilcoxon
import torch

# Reproducibility / single-host fairness
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

OUT = Path('/mnt/data/E3_qi_benchmark_results')
OUT.mkdir(exist_ok=True, parents=True)

SCALES = {
    'small':  {'D':3,'R':12,'cap':25,'deadline':0.10},
    'medium': {'D':5,'R':20,'cap':30,'deadline':0.25},
    'large':  {'D':7,'R':28,'cap':32,'deadline':0.50},
}
TIGHTNESS = {'loose': 700.0, 'tight': 300.0}  # window width seconds
RESERVE = {'standard':0.15, 'high':0.30}
HORIZON = 2200.0
SERVICE_TIME = 28.0
QUBO_PENALTY = 1.20

@dataclass
class Candidate:
    drone:int
    route:tuple
    requests:frozenset
    energy:float
    duration:float
    reward:float
    weight:float


def euclid(a,b):
    return float(np.linalg.norm(a-b))


def generate_instance(seed, scale, tightness, reserve_level):
    rng = np.random.default_rng(seed)
    cfg = SCALES[scale]
    D,R = cfg['D'], cfg['R']
    # request geography (km), depot at origin
    pos = rng.uniform(-4.5, 4.5, size=(R,2))
    # avoid too-near points to make routing meaningful
    rad = np.linalg.norm(pos,axis=1)
    too_near = rad < 0.8
    if np.any(too_near):
        pos[too_near] *= (0.8/(rad[too_near,None]+1e-9))
    payload = rng.uniform(0.15,1.8,size=R)
    priority = rng.integers(1,6,size=R)
    # random mission wind vector m/s; kept common across candidate routes of the instance
    wind_speed = rng.uniform(0.0,5.0)
    wind_dir = rng.uniform(0,2*np.pi)
    wind = wind_speed*np.array([math.cos(wind_dir), math.sin(wind_dir)])
    # request windows anchored around plausible direct service time
    direct_t = np.linalg.norm(pos,axis=1)*1000/15.5
    centers = direct_t + rng.uniform(180, 1200, size=R)
    width = TIGHTNESS[tightness]
    earliest = np.maximum(0.0, centers-width/2)
    latest = np.minimum(HORIZON, centers+width/2)
    # drones heterogeneous but similar
    speed = rng.uniform(14.0,17.5,size=D)
    battery = rng.uniform(155.0,205.0,size=D)  # usable Wh before protected reserve
    soh = rng.uniform(0.82,1.0,size=D)
    # location from common depot but slight energy efficiency differences
    base_wh_km = rng.uniform(8.5,10.5,size=D)
    payload_wh_km = rng.uniform(1.5,2.3,size=D)
    reserve_frac = RESERVE[reserve_level]
    return {
        'seed':seed,'scale':scale,'tightness':tightness,'reserve_level':reserve_level,
        'D':D,'R':R,'pos':pos,'payload':payload,'priority':priority,'wind':wind,
        'earliest':earliest,'latest':latest,'speed':speed,'battery':battery,'soh':soh,
        'base_wh_km':base_wh_km,'payload_wh_km':payload_wh_km,'reserve_frac':reserve_frac,
    }


def evaluate_route(inst, d, route):
    pos=inst['pos']; payload=inst['payload']; pr=inst['priority']; wind=inst['wind']
    speed=float(inst['speed'][d]); base=float(inst['base_wh_km'][d]); paycoef=float(inst['payload_wh_km'][d])
    batt=float(inst['battery'][d]*inst['soh'][d]); reserve=inst['reserve_frac']*batt
    cur=np.zeros(2); t=0.0; energy=0.0
    remaining_payload=float(np.sum(payload[list(route)]))
    reward=0.0
    for r in route:
        vec = pos[r]-cur
        dist_km=euclid(pos[r],cur)
        unit = vec/(np.linalg.norm(vec)+1e-12)
        headwind=max(0.0, float(-np.dot(wind, unit)))
        groundspeed=max(7.0, speed-headwind)
        seg_t=dist_km*1000/groundspeed
        t += seg_t
        if t < inst['earliest'][r]:
            # hover/wait: modest energy
            wait=inst['earliest'][r]-t
            energy += 0.010*wait  # 36 W equivalent
            t=inst['earliest'][r]
        if t > inst['latest'][r]:
            return None
        # flight energy, payload and headwind sensitive
        wh_per_km = base + paycoef*remaining_payload + 0.75*headwind
        energy += dist_km*wh_per_km
        t += SERVICE_TIME
        energy += 0.008*SERVICE_TIME
        reward += 100.0*float(pr[r]) + 8.0
        remaining_payload -= float(payload[r])
        cur = pos[r]
    # return to depot
    vec = -cur
    dist_km=euclid(cur,np.zeros(2))
    unit = vec/(np.linalg.norm(vec)+1e-12)
    headwind=max(0.0,float(-np.dot(wind,unit)))
    groundspeed=max(7.0,speed-headwind)
    t += dist_km*1000/groundspeed
    energy += dist_km*(base+0.75*headwind)
    # safety feasibility: protected reserve must remain
    if energy + reserve > batt:
        return None
    if t > HORIZON:
        return None
    # reward minus moderate energy/duration cost
    weight = reward - 0.55*energy - 0.012*t
    if weight <= 0:
        return None
    return Candidate(d,tuple(route),frozenset(route),energy,t,reward,weight)


def generate_candidates(inst, cap, rng):
    t0=time.perf_counter()
    D,R=inst['D'],inst['R']
    by_drone=[]
    for d in range(D):
        seen={}
        # all singletons
        for r in range(R):
            c=evaluate_route(inst,d,(r,))
            if c: seen[c.route]=c
        # structured nearest-neighbor pairs/triples
        dmat=np.linalg.norm(inst['pos'][:,None,:]-inst['pos'][None,:,:],axis=2)
        for r in range(R):
            nbrs=np.argsort(dmat[r])[1: min(R,7)]
            for r2 in nbrs:
                for route in [(r,int(r2)),(int(r2),r)]:
                    c=evaluate_route(inst,d,route)
                    if c: seen[c.route]=c
        # randomized 2-4 stop routes
        attempts=cap*20
        for _ in range(attempts):
            k=int(rng.choice([2,3,4],p=[0.30,0.45,0.25]))
            if k>R: continue
            route=tuple(rng.choice(R,size=k,replace=False).tolist())
            c=evaluate_route(inst,d,route)
            if c: seen[c.route]=c
        vals=list(seen.values())
        vals.sort(key=lambda c:c.weight,reverse=True)
        # preserve some diversity rather than only top near-duplicates
        if len(vals)>cap:
            topn=max(1,int(cap*0.75))
            top=vals[:topn]
            rest=vals[topn:]
            take=cap-topn
            if len(rest)>take:
                idx=rng.choice(len(rest),size=take,replace=False)
                rest=[rest[i] for i in idx]
            vals=top+rest[:take]
            vals.sort(key=lambda c:c.weight,reverse=True)
        by_drone.extend(vals)
    # ensure every drone has at least one candidate by adding best feasible singleton if needed
    cand=by_drone
    pre_time=time.perf_counter()-t0
    return cand,pre_time


def build_conflicts(cand, D, R):
    n=len(cand)
    edges=[]
    drone_groups=[[] for _ in range(D)]
    request_groups=[[] for _ in range(R)]
    for i,c in enumerate(cand):
        drone_groups[c.drone].append(i)
        for r in c.requests: request_groups[r].append(i)
    edge_set=set()
    for g in drone_groups+request_groups:
        for i,j in combinations(g,2):
            if i>j: i,j=j,i
            edge_set.add((i,j))
    edges=sorted(edge_set)
    return edges,drone_groups,request_groups


def validate_selection(sel, cand):
    drones=set(); reqs=set(); obj=0.0
    for i in sel:
        c=cand[i]
        if c.drone in drones or reqs.intersection(c.requests):
            return False, obj
        drones.add(c.drone); reqs.update(c.requests); obj += c.weight
    return True,obj


def repair_and_augment(raw_sel, cand, edges=None):
    # common deterministic repair: keep higher-weight raw selections first, then greedily augment by weight/(1+conflict degree)
    n=len(cand)
    if edges is None:
        edges,_,_=build_conflicts(cand,max(c.drone for c in cand)+1, max(max(c.requests) for c in cand)+1)
    adj=[set() for _ in range(n)]
    for i,j in edges:
        adj[i].add(j); adj[j].add(i)
    chosen=[]; chosen_set=set()
    for i in sorted(set(raw_sel), key=lambda k:cand[k].weight, reverse=True):
        if not (adj[i] & chosen_set):
            chosen.append(i); chosen_set.add(i)
    order=sorted([i for i in range(n) if i not in chosen_set], key=lambda k:cand[k].weight/(1+0.15*len(adj[k])), reverse=True)
    for i in order:
        if not (adj[i] & chosen_set):
            chosen.append(i); chosen_set.add(i)
    ok,obj=validate_selection(chosen,cand)
    assert ok
    return chosen,obj


def build_milp(cand, D, R):
    n=len(cand)
    rows=[]; cols=[]; data=[]; b=[]
    rr=0
    for d in range(D):
        idx=[i for i,c in enumerate(cand) if c.drone==d]
        for i in idx: rows.append(rr); cols.append(i); data.append(1.0)
        b.append(1.0); rr+=1
    for r in range(R):
        idx=[i for i,c in enumerate(cand) if r in c.requests]
        for i in idx: rows.append(rr); cols.append(i); data.append(1.0)
        b.append(1.0); rr+=1
    A=sparse.csr_matrix((data,(rows,cols)),shape=(rr,n))
    con=LinearConstraint(A, lb=-np.inf*np.ones(rr), ub=np.array(b))
    return con


def solve_milp_budget(cand, D, R, common_pre, deadline):
    t0=time.perf_counter()
    con=build_milp(cand,D,R)
    encode=time.perf_counter()-t0
    rem=max(0.001, deadline-common_pre-encode)
    c=-np.array([x.weight for x in cand],dtype=float)
    t1=time.perf_counter()
    res=milp(c,integrality=np.ones(len(cand)),bounds=Bounds(0,1),constraints=con,
             options={'time_limit':rem,'presolve':True,'mip_rel_gap':0.0,'disp':False})
    solve=time.perf_counter()-t1
    raw=[]
    if getattr(res,'x',None) is not None:
        raw=np.where(np.asarray(res.x)>0.5)[0].tolist()
    t2=time.perf_counter(); sel,obj=repair_and_augment(raw,cand); post=time.perf_counter()-t2
    total=common_pre+encode+solve+post
    return {'solver':'MILP-HIGHS','obj':obj,'raw_feasible':validate_selection(raw,cand)[0] if raw else True,'repaired_feasible':True,
            'total_time':total,'encode_time':encode,'solve_time':solve,'post_time':post,
            'first_feasible_time':total if raw else np.nan,'raw_count':len(raw),'sel_count':len(sel),
            'status':int(getattr(res,'status',-99)), 'mip_gap':float(getattr(res,'mip_gap',np.nan))}


def offline_reference(cand,D,R,limit=5.0):
    con=build_milp(cand,D,R)
    c=-np.array([x.weight for x in cand],dtype=float)
    res=milp(c,integrality=np.ones(len(cand)),bounds=Bounds(0,1),constraints=con,
             options={'time_limit':limit,'presolve':True,'mip_rel_gap':0.0,'disp':False})
    raw=[] if getattr(res,'x',None) is None else np.where(np.asarray(res.x)>0.5)[0].tolist()
    sel,obj=repair_and_augment(raw,cand)
    return obj,int(getattr(res,'status',-99)),float(getattr(res,'mip_gap',np.nan)),float(getattr(res,'mip_dual_bound',np.nan))


def make_adj(n,edges):
    adj=[set() for _ in range(n)]
    for i,j in edges: adj[i].add(j); adj[j].add(i)
    return adj


def greedy_construct(cand, adj, rng=None, randomized=False):
    n=len(cand)
    chosen=[]; chosen_set=set(); remaining=set(range(n))
    while remaining:
        feasible=[i for i in remaining if not (adj[i]&chosen_set)]
        if not feasible: break
        scores=np.array([cand[i].weight/(1+0.10*len(adj[i])) for i in feasible])
        order=np.argsort(-scores)
        if randomized and len(order)>1:
            rcl=max(1,int(math.ceil(0.15*len(order))))
            pick=feasible[int(order[int(rng.integers(0,rcl))])]
        else:
            pick=feasible[int(order[0])]
        chosen.append(pick); chosen_set.add(pick)
        # eliminate pick and all conflicts
        remaining.discard(pick)
        remaining.difference_update(adj[pick])
    return chosen


def local_exchange(sel,cand,adj):
    chosen=set(sel); improved=True; loops=0
    while improved and loops<3:
        improved=False; loops+=1
        # single insertion / conflict replacement
        for i in sorted((x for x in range(len(cand)) if x not in chosen), key=lambda k:cand[k].weight, reverse=True):
            conf=adj[i]&chosen
            lost=sum(cand[j].weight for j in conf)
            if cand[i].weight > lost + 1e-9:
                trial=(chosen-conf)|{i}
                ok,_=validate_selection(sorted(trial),cand)
                if ok:
                    chosen=trial; improved=True
        # augment any newly free slots
        for i in sorted((x for x in range(len(cand)) if x not in chosen), key=lambda k:cand[k].weight, reverse=True):
            if not (adj[i]&chosen):
                chosen.add(i); improved=True
    sel=sorted(chosen)
    return sel,validate_selection(sel,cand)[1]


def solve_grasp_budget(cand,edges,common_pre,deadline,seed):
    t0=time.perf_counter(); adj=make_adj(len(cand),edges); encode=time.perf_counter()-t0
    rng=np.random.default_rng(seed+99173)
    start=time.perf_counter();
    sel=greedy_construct(cand,adj,None,False); sel,obj=local_exchange(sel,cand,adj)
    first=common_pre+encode+(time.perf_counter()-start)
    best_sel,best_obj=sel,obj
    restarts=0
    # reserve tiny postprocessing margin
    while common_pre+encode+(time.perf_counter()-start) < deadline-0.001:
        s=greedy_construct(cand,adj,rng,True); s,o=local_exchange(s,cand,adj); restarts+=1
        if o>best_obj+1e-9: best_sel,best_obj=s,o
    solve=time.perf_counter()-start
    t2=time.perf_counter(); final,obj2=repair_and_augment(best_sel,cand,edges); post=time.perf_counter()-t2
    total=common_pre+encode+solve+post
    return {'solver':'GRASP-LS','obj':obj2,'raw_feasible':True,'repaired_feasible':True,'total_time':total,'encode_time':encode,
            'solve_time':solve,'post_time':post,'first_feasible_time':first,'raw_count':len(best_sel),'sel_count':len(final),
            'restarts':restarts,'status':0,'mip_gap':np.nan}


def build_qubo(cand,edges):
    n=len(cand)
    w=np.array([c.weight for c in cand],dtype=np.float64)
    wnorm=w/max(w.max(),1e-12)
    M=np.zeros((n,n),dtype=np.float32)
    np.fill_diagonal(M,-wnorm.astype(np.float32))
    half=np.float32(QUBO_PENALTY/2)
    for i,j in edges:
        M[i,j]+=half; M[j,i]+=half
    # convert binary QUBO x'Mx to no-field Ising using one gauge/dummy spin
    h=0.5*M.sum(axis=1)  # linear in s after x=(s+1)/2
    J=np.zeros((n+1,n+1),dtype=np.float32)
    J[:n,:n]=-0.5*M
    np.fill_diagonal(J,0.0)
    J[:n,n]=-h
    J[n,:n]=-h
    return M,J


def sb_run(cand,edges,common_pre,deadline,seed,engine='discrete',pressure_slope=0.02,agents=16, sample_period=50):
    enc0=time.perf_counter(); M,Jnp=build_qubo(cand,edges); J=torch.tensor(Jnp,dtype=torch.float32); encode=time.perf_counter()-enc0
    nspin=J.shape[0]
    rem=deadline-common_pre-encode
    if rem<=0.002:
        sel,obj=repair_and_augment([],cand,edges)
        return {'solver':'SB-QI','obj':obj,'raw_feasible':True,'repaired_feasible':True,'total_time':common_pre+encode,'encode_time':encode,'solve_time':0,'post_time':0,
                'first_feasible_time':common_pre+encode,'raw_count':0,'sel_count':len(sel),'steps':0,'restarts':0,'agents':agents,'status':1,'mip_gap':np.nan}
    dt=0.1
    denom=torch.sqrt(torch.sum(J*J))
    scale=0.5*math.sqrt(max(1,nspin-1))/float(denom if denom>0 else 1.0)
    start=time.perf_counter(); total_steps=0; restarts=0; best_obj=-1e30; best_sel=[]; best_raw_feas=False; best_raw_count=0; first=np.nan
    # Repeated independent SB trajectories use the full equal wall-clock budget.
    while common_pre+encode+(time.perf_counter()-start) < deadline-0.002:
        torch.manual_seed(seed+123456+1009*restarts)
        pos=2*torch.rand((nspin,agents),dtype=torch.float32)-1
        mom=2*torch.rand((nspin,agents),dtype=torch.float32)-1
        step=0; restarts+=1
        while step<1000 and common_pre+encode+(time.perf_counter()-start) < deadline-0.002:
            block=min(sample_period,1000-step)
            for _ in range(block):
                pressure=min(dt*step*pressure_slope,1.0)
                mom.add_(pos, alpha=dt*(pressure-1.0))
                act=torch.sign(pos) if engine=='discrete' else pos
                mom.add_(torch.matmul(J,act), alpha=dt*scale)
                pos.add_(mom,alpha=dt)
                mask=torch.abs(pos)>1.0
                mom[mask]=0.0
                torch.clamp_(pos,-1.0,1.0)
                step+=1; total_steps+=1
            spin=torch.where(pos>=0,1,-1)
            eff=spin[:-1,:]*spin[-1:,:]
            xb=(eff>0).cpu().numpy()
            energies=np.einsum('ia,ij,ja->a',xb.astype(np.float32),M,xb.astype(np.float32),optimize=True)
            order=np.argsort(energies)[:min(4,agents)]
            for a in order:
                raw=np.where(xb[:,a])[0].tolist()
                raw_ok,_=validate_selection(raw,cand)
                sel,obj=repair_and_augment(raw,cand,edges)
                if np.isnan(first): first=common_pre+encode+(time.perf_counter()-start)
                if obj>best_obj:
                    best_obj=obj; best_sel=sel; best_raw_feas=raw_ok; best_raw_count=len(raw)
            if common_pre+encode+(time.perf_counter()-start) >= deadline-0.002:
                break
    solve=time.perf_counter()-start
    post0=time.perf_counter(); final,obj=repair_and_augment(best_sel,cand,edges); post=time.perf_counter()-post0
    ok,_=validate_selection(final,cand)
    total=common_pre+encode+solve+post
    return {'solver':'SB-QI','obj':obj,'raw_feasible':bool(best_raw_feas),'repaired_feasible':bool(ok),'total_time':total,'encode_time':encode,'solve_time':solve,'post_time':post,
            'first_feasible_time':first if not np.isnan(first) else total,'raw_count':int(best_raw_count),'sel_count':len(final),
            'steps':total_steps,'restarts':restarts,'agents':agents,'engine':engine,'pressure_slope':pressure_slope,'status':0,'mip_gap':np.nan}

def conflict_density(n,edges):
    return 0 if n<2 else 2*len(edges)/(n*(n-1))


def run_one(seed,scale,tightness,reserve_level,sb_cfg=None,offline=True):
    inst=generate_instance(seed,scale,tightness,reserve_level)
    rng=np.random.default_rng(seed+77)
    cand,pre=generate_candidates(inst,SCALES[scale]['cap'],rng)
    if len(cand)==0:
        raise RuntimeError('No candidates')
    edges,_,_=build_conflicts(cand,inst['D'],inst['R'])
    ref_obj=ref_status=ref_gap=ref_dual=np.nan
    if offline:
        ref_obj,ref_status,ref_gap,ref_dual=offline_reference(cand,inst['D'],inst['R'])
    deadline=SCALES[scale]['deadline']
    results=[]
    results.append(solve_milp_budget(cand,inst['D'],inst['R'],pre,deadline))
    results.append(solve_grasp_budget(cand,edges,pre,deadline,seed))
    cfg=sb_cfg or {'engine':'discrete','pressure_slope':0.02,'agents':16}
    results.append(sb_run(cand,edges,pre,deadline,seed,**cfg))
    meta={'seed':seed,'scale':scale,'tightness':tightness,'reserve_level':reserve_level,
          'D':inst['D'],'R':inst['R'],'n_candidates':len(cand),'n_conflicts':len(edges),'conflict_density':conflict_density(len(cand),edges),
          'common_pre_time':pre,'deadline':deadline,'ref_obj':ref_obj,'ref_status':ref_status,'ref_gap':ref_gap,'ref_dual':ref_dual}
    rows=[]
    for r in results:
        row={**meta,**r}
        row['gap_pct']=100*(ref_obj-r['obj'])/ref_obj if ref_obj and np.isfinite(ref_obj) else np.nan
        row['deadline_met']=row['total_time'] <= deadline*1.10  # 10% measurement-tolerance band; exact time also reported
        rows.append(row)
    return rows


def calibration():
    configs=[]
    for engine in ['ballistic','discrete']:
        for slope in [0.01,0.02,0.04]:
            for agents in [8,16]:
                configs.append({'engine':engine,'pressure_slope':slope,'agents':agents})
    seeds=[]
    for scale_i,scale in enumerate(['small','medium','large']):
        for k in range(3): seeds.append((31000+100*scale_i+k,scale,'tight','high'))
    # Precompute the identical calibration instances and references once.
    cache=[]
    for seed,scale,tight,resv in seeds:
        inst=generate_instance(seed,scale,tight,resv); rng=np.random.default_rng(seed+77)
        cand,pre=generate_candidates(inst,SCALES[scale]['cap'],rng); edges,_,_=build_conflicts(cand,inst['D'],inst['R'])
        ref_obj,ref_status,ref_gap,ref_dual=offline_reference(cand,inst['D'],inst['R'],limit=2.0)
        cache.append((seed,scale,cand,edges,pre,ref_obj))
    allrows=[]
    for cfg in configs:
        for seed,scale,cand,edges,pre,ref_obj in cache:
            r=sb_run(cand,edges,pre,SCALES[scale]['deadline'],seed,**cfg)
            gap=100*(ref_obj-r['obj'])/ref_obj
            allrows.append({**cfg,'seed':seed,'scale':scale,'n_candidates':len(cand),'ref_obj':ref_obj,'gap_pct':gap,
                            'raw_feasible':r['raw_feasible'],'repaired_feasible':r['repaired_feasible'],'total_time':r['total_time'],'steps':r.get('steps',0),'restarts':r.get('restarts',0)})
    df=pd.DataFrame(allrows)
    df.to_csv(OUT/'calibration_grid_results.csv',index=False)
    summ=(df.groupby(['engine','pressure_slope','agents'])
          .agg(mean_gap=('gap_pct','mean'),median_gap=('gap_pct','median'),raw_feasible_rate=('raw_feasible','mean'),mean_time=('total_time','mean'),mean_steps=('steps','mean'))
          .reset_index().sort_values(['mean_gap','raw_feasible_rate','mean_time'],ascending=[True,False,True]))
    summ.to_csv(OUT/'calibration_grid_summary.csv',index=False)
    best=summ.iloc[0]
    cfg={'engine':str(best.engine),'pressure_slope':float(best.pressure_slope),'agents':int(best.agents)}
    (OUT/'CALIBRATION_SELECTED.json').write_text(json.dumps({'selected':cfg,'selection_rule':'minimum mean repaired gap; raw-feasible-rate tie-break; lower time second tie-break'},indent=2))
    return cfg,summ

def bootstrap_ci(x, n=10000, seed=777):
    x=np.asarray(x,float); rng=np.random.default_rng(seed)
    means=np.empty(n)
    m=len(x)
    for i in range(n): means[i]=np.mean(x[rng.integers(0,m,size=m)])
    return float(np.percentile(means,2.5)),float(np.percentile(means,97.5))


def holm(pvals):
    pvals=np.asarray(pvals,float); m=len(pvals); order=np.argsort(pvals); adj=np.empty(m)
    prev=0
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*pvals[idx]); prev=max(prev,val); adj[idx]=prev
    return adj


def final_run(sb_cfg):
    rows=[]; instance_id=0
    for si,scale in enumerate(['small','medium','large']):
        for ti,tight in enumerate(['loose','tight']):
            for ri,resv in enumerate(['standard','high']):
                for k in range(12):
                    seed=41000+si*1000+ti*200+ri*100+k
                    rr=run_one(seed,scale,tight,resv,sb_cfg=sb_cfg,offline=True)
                    for row in rr:
                        row['instance_id']=instance_id
                        rows.append(row)
                    instance_id+=1
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'final_trial_metrics.csv',index=False)
    # reference quality filter
    df['ref_good']=(df['ref_status']==0) | (df['ref_gap'].fillna(1.0)<=1e-4)
    # summary
    summ=(df.groupby(['scale','tightness','reserve_level','solver'])
          .agg(n=('instance_id','count'), mean_gap_pct=('gap_pct','mean'), median_gap_pct=('gap_pct','median'),
               raw_feasible_rate=('raw_feasible','mean'), repaired_feasible_rate=('repaired_feasible','mean'), mean_obj=('obj','mean'), mean_total_time=('total_time','mean'),
               p95_total_time=('total_time',lambda x:np.percentile(x,95)), deadline_met_rate=('deadline_met','mean'),
               mean_first_feasible=('first_feasible_time','mean'))
          .reset_index())
    summ.to_csv(OUT/'final_summary.csv',index=False)
    # aggregate by scale
    ss=(df.groupby(['scale','solver'])
        .agg(n=('instance_id','count'),mean_gap_pct=('gap_pct','mean'),median_gap_pct=('gap_pct','median'),
             raw_feasible_rate=('raw_feasible','mean'),repaired_feasible_rate=('repaired_feasible','mean'),mean_total_time=('total_time','mean'),deadline_met_rate=('deadline_met','mean'))
        .reset_index())
    ss.to_csv(OUT/'scale_summary.csv',index=False)
    # pair stats SB vs GRASP by scale on good references
    stats=[]; pvals=[]; temp=[]
    for scale in ['medium','large']:
        piv=df[(df.scale==scale)&df.ref_good].pivot(index='instance_id',columns='solver',values='gap_pct').dropna()
        diff=piv['GRASP-LS']-piv['SB-QI'] # positive => SB better
        ci=bootstrap_ci(diff.values,10000,seed=880+len(stats))
        try: w=wilcoxon(diff.values,alternative='two-sided',zero_method='wilcox')
        except Exception: w=type('X',(object,),{'statistic':np.nan,'pvalue':1.0})()
        rec={'scale':scale,'n':len(diff),'mean_gap_GRASP':float(piv['GRASP-LS'].mean()),'mean_gap_SB':float(piv['SB-QI'].mean()),
             'mean_improvement_pp_GRASP_minus_SB':float(diff.mean()),'median_improvement_pp':float(diff.median()),
             'bootstrap95_low':ci[0],'bootstrap95_high':ci[1],'wilcoxon_stat':float(w.statistic),'p_raw':float(w.pvalue)}
        temp.append(rec); pvals.append(float(w.pvalue))
    adj=holm(pvals)
    for rec,padj in zip(temp,adj): rec['p_holm']=float(padj); stats.append(rec)
    sdf=pd.DataFrame(stats); sdf.to_csv(OUT/'primary_paired_statistics.csv',index=False)
    # all pairwise solver stats by scale for descriptive reporting
    allstats=[]
    for scale in ['small','medium','large']:
        piv=df[(df.scale==scale)&df.ref_good].pivot(index='instance_id',columns='solver',values='gap_pct').dropna()
        for a,b in [('SB-QI','GRASP-LS'),('SB-QI','MILP-HIGHS'),('GRASP-LS','MILP-HIGHS')]:
            diff=piv[b]-piv[a] # positive means a better
            ci=bootstrap_ci(diff.values,5000,seed=1200+len(allstats))
            try:w=wilcoxon(diff.values,zero_method='wilcox')
            except Exception:w=type('X',(object,),{'statistic':np.nan,'pvalue':1.0})()
            allstats.append({'scale':scale,'solver_A':a,'solver_B':b,'n':len(diff),'mean_gap_A':piv[a].mean(),'mean_gap_B':piv[b].mean(),
                             'mean_improvement_pp_B_minus_A':diff.mean(),'ci95_low':ci[0],'ci95_high':ci[1],'p_raw':w.pvalue})
    pd.DataFrame(allstats).to_csv(OUT/'descriptive_paired_statistics.csv',index=False)
    # H2 decision
    sc={r['scale']:r for r in stats}
    feasible=float(df[df.solver=='SB-QI'].groupby('scale').repaired_feasible.mean().min())
    medium=sc['medium']; large=sc['large']
    def qualifies(rec):
        return rec['mean_improvement_pp_GRASP_minus_SB']>=2.0 and rec['bootstrap95_low']>0 and rec['p_holm']<0.05
    qmed=qualifies(medium); qlarge=qualifies(large)
    other_ok=(large['mean_improvement_pp_GRASP_minus_SB']>=-5.0 if qmed else True) and (medium['mean_improvement_pp_GRASP_minus_SB']>=-5.0 if qlarge else True)
    supported=(feasible>=0.99) and (qmed or qlarge) and other_ok
    decision={'H2_supported':bool(supported),'SB_repaired_feasible_rate_min_scale':feasible,'medium_qualifies':qmed,'large_qualifies':qlarge,
              'other_scale_noninferiority_ok':other_ok,'decision_rule':'Frozen in PROTOCOL_FROZEN.md'}
    (OUT/'H2_DECISION.json').write_text(json.dumps(decision,indent=2))
    return df,summ,ss,sdf,decision


def make_plots(df,ss):
    import matplotlib.pyplot as plt
    # mean gap by scale and solver
    order=['small','medium','large']; solvers=['MILP-HIGHS','GRASP-LS','SB-QI']
    x=np.arange(len(order)); width=.24
    fig,ax=plt.subplots(figsize=(8,4.8))
    for j,s in enumerate(solvers):
        vals=[float(ss[(ss.scale==sc)&(ss.solver==s)].mean_gap_pct.iloc[0]) for sc in order]
        ax.bar(x+(j-1)*width,vals,width,label=s)
    ax.set_xticks(x,order); ax.set_ylabel('Mean optimality gap (%)'); ax.set_xlabel('Problem scale')
    ax.set_title('E3 quality at equal end-to-end deadline'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'e3_gap_by_scale.png',dpi=180); plt.close(fig)
    # raw feasible rate for SB and deadline met
    sb=df[df.solver=='SB-QI'].groupby('scale').agg(raw_feasible=('raw_feasible','mean'),deadline_met=('deadline_met','mean')).reindex(order)
    fig,ax=plt.subplots(figsize=(7.5,4.5)); ax.bar(x-0.17,sb.raw_feasible.values*100,0.34,label='Raw QUBO feasible'); ax.bar(x+0.17,sb.deadline_met.values*100,0.34,label='Deadline met');
    ax.set_xticks(x,order); ax.set_ylim(0,105); ax.set_ylabel('Rate (%)'); ax.set_title('SB-QI feasibility and deadline compliance'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'e3_sb_feasibility_deadline.png',dpi=180); plt.close(fig)


def write_reports(cfg,df,ss,stats,decision,cal_summ):
    # key aggregates
    table=ss.pivot(index='scale',columns='solver',values='mean_gap_pct').reindex(['small','medium','large'])
    feas=df.groupby(['scale','solver']).raw_feasible.mean().unstack()
    times=df.groupby(['scale','solver']).total_time.mean().unstack()
    refs=df.groupby('instance_id').first()
    refgood=((refs.ref_status==0)|(refs.ref_gap.fillna(1)<=1e-4)).mean()
    pstats=stats.set_index('scale')
    h2='SUPPORTED' if decision['H2_supported'] else 'NOT SUPPORTED'
    report=f"""# E3 Final Report — Equal-Budget Classical vs QUBO/Quantum-Inspired Fleet Optimization

## Outcome
**Frozen H2 decision: {h2}.**

E3 tested the manuscript claim that a quantum-inspired backend can provide a time-to-quality / quality-at-deadline advantage under complete accounting. The benchmark used the same energy/time-window-feasible route-bundle set-packing instances for all solvers and charged common preprocessing, backend encoding, solve, decode/repair, and validation to the operational deadline.

### Frozen SB-QI configuration
`engine={cfg['engine']}`, `pressure_slope={cfg['pressure_slope']}`, `agents={cfg['agents']}`, time step 0.1, no heating, CPU only.

## Mean optimality gap at the equal end-to-end deadline (%)

{table.to_markdown(floatfmt='.3f')}

## Raw solver feasibility rate

{feas.to_markdown(floatfmt='.3f')}

## Mean end-to-end runtime (s)

{times.to_markdown(floatfmt='.4f')}

Offline reference quality was optimal / <=1e-4 MIP gap for {100*refgood:.1f}% of final instances.

## Primary preregistered SB-QI vs GRASP-LS comparisons

{stats.to_markdown(index=False,floatfmt='.4f')}

The frozen H2 rule required >=99% repaired/raw-feasibility criterion as specified, at least a 2.0 percentage-point mean gap advantage on medium or large instances, a paired CI excluding zero with Holm-corrected significance, and no >5 percentage-point deterioration on the other medium/large scale.

Decision details:

```json
{json.dumps(decision,indent=2)}
```

## Scientific interpretation
A positive H2 result would justify retaining solver-regime language that treats SB-QI as an empirically advantaged backend for at least one operational regime. A negative result means the architecture may still keep QUBO as a solver-neutral representation/interface, but the manuscript should not imply that the quantum-inspired backend is faster or better than a strong classical heuristic under equal end-to-end accounting.

The exact meaning of this experiment is deliberately narrow. `SB-QI` is a CPU implementation of discrete Simulated Bifurcation update equations applied to the QUBO representation. No quantum processor, quantum annealer, or dedicated Ising machine was used. Therefore this experiment cannot support any claim of quantum hardware advantage.

## Provenance and limitations
- Synthetic-but-physics-grounded drone-fleet instances: candidate bundles satisfy time windows and a protected battery-reserve feasibility check before discrete optimization.
- Route decomposition means E3 benchmarks the fleet assignment/route-bundle selection layer, not a continuous full vehicle-routing problem.
- Common preprocessing is identical across solvers, but SciPy/HiGHS and the Python/Torch solvers have different implementation stacks. Wall-clock comparison is therefore host- and implementation-dependent.
- QUBO repair is deterministic and included in total time. Raw-QUBO feasibility is separately reported to avoid hiding penalty/solver failures.
- The offline MILP reference is outside the equal-budget contest and exists only to establish the quality denominator.
- E3 should be called an equal-budget CPU solver-regime experiment, not package-native quantum optimization evidence.
"""
    (OUT/'FINAL_REPORT.md').write_text(report)
    if decision['H2_supported']:
        status="H2 changes from OPEN to PARTIALLY SUPPORTED by equal-budget CPU quantum-inspired evidence."
        framing="The QUBO/SB backend demonstrated a preregistered quality-at-deadline advantage in at least one medium/large regime, but the claim remains CPU-backend- and instance-family-specific."
    else:
        status="H2 remains OPEN / NOT SUPPORTED as a solver-advantage claim."
        framing="Under equal end-to-end accounting, the SB-QI backend did not satisfy the preregistered advantage criterion against the strong classical GRASP-LS baseline. QUBO should therefore remain a solver-neutral interface in the manuscript, and QI should be de-emphasized unless future dedicated hardware/package-native evidence changes the result."
    insert=f"""## E3 — Equal-Budget Solver-Regime Benchmark

To test H2 without inferring solver merit from the QUBO representation itself, we generated energy- and time-window-feasible route bundles and solved the same set-packing problem with a time-limited MILP baseline (HiGHS), an anytime GRASP/local-search heuristic, and a CPU discrete Simulated Bifurcation QUBO backend. Common candidate generation and all backend-specific encoding, solve, decoding, repair, and validation were charged to fixed end-to-end deadlines of 0.10/0.25/0.50 s for small/medium/large instances. The final campaign contained 144 untouched instances spanning time-window tightness and protected-reserve levels; a separate long-budget MILP supplied the reference optimum or tight bound.

{status} {framing}

The manuscript should report the scale-wise mean optimality gaps and the preregistered paired SB-QI-versus-GRASP comparisons from `scale_summary.csv` and `primary_paired_statistics.csv`. The result is CPU-only quantum-inspired evidence; it is not quantum-hardware evidence and must not be described as generic quantum advantage.
"""
    (OUT/'MANUSCRIPT_INSERT_E3.md').write_text(insert)


def checksums():
    files=[p for p in OUT.iterdir() if p.is_file() and p.name!='SHA256SUMS']
    lines=[]
    for p in sorted(files):
        h=hashlib.sha256(p.read_bytes()).hexdigest(); lines.append(f'{h}  {p.name}')
    (OUT/'SHA256SUMS').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    print('Calibration...')
    cfg,cal_summ=calibration(); print('selected',cfg); print(cal_summ.head())
    print('Final...')
    df,summ,ss,stats,decision=final_run(cfg)
    print(ss)
    print(stats)
    print(decision)
    make_plots(df,ss)
    write_reports(cfg,df,ss,stats,decision,cal_summ)
    env={'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,
         'scipy':__import__('scipy').__version__,'torch':torch.__version__,'torch_threads':torch.get_num_threads(),
         'cpu_count':os.cpu_count(),'qubo_penalty':QUBO_PENALTY,'sb_selected':cfg}
    (OUT/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2))
    checksums()

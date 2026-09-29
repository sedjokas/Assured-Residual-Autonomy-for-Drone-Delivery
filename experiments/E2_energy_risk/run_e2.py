from __future__ import annotations
import math, json, hashlib, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
import matplotlib.pyplot as plt

OUT = Path('/mnt/data/E2_energy_risk_results')
OUT.mkdir(parents=True, exist_ok=True)

# Frozen physical / operational constants
M0 = 0.50                 # kg, source-informed Hummingbird base mass
G = 9.81
RHO20 = 1.204             # kg/m3 around 20 C
NROT = 4
RROTOR = 0.10             # m
AREA = NROT * math.pi * RROTOR**2
C_NOM = 30.0              # Wh, synthetic mission battery assumption
RES_FRAC = 0.15
V_GROUND = 8.0            # m/s
AUX_NOM = 5.0             # W
PROFILE_NOM = 18.0        # W
ETA_NOM = 0.64
CDA_NOM = 0.030           # m2
CLIMB_RATE = 2.0          # m/s
HOVER_SERVICE_BASE = 16.0 # s
DIVERT_FACTOR = 0.45      # fraction of original remaining true/predicted energy to nearest contingency site
TARGET_COVERAGE = 0.975

SCENARIOS = [
    'nominal','heavy_payload','headwind','aged_battery',
    'cold_temperature','vertical_motion','long_route','compound'
]


def temp_capacity_factor(Tc):
    Tc = np.asarray(Tc)
    # Mildly conservative synthetic capacity derating, clipped to avoid nonphysical collapse.
    f = 1.0 - 0.0045*np.maximum(20.0-Tc,0.0) - 0.0010*np.maximum(Tc-35.0,0.0)
    return np.clip(f, 0.72, 1.02)


def air_density(Tc):
    # Sea-level ideal-gas temperature correction, sufficient for this mechanism experiment.
    return RHO20 * (293.15 / (273.15 + np.asarray(Tc)))


def phase_energy(distance_m, payload, w_out, w_ret, vert_m, hover_s, temp_c,
                 eta_mult=1.0, cda_mult=1.0, aux_extra=0.0, soh=1.0):
    """Return 4 phase energies [Wh]: climb+45% outbound, 55% outbound+hover, 50% return, 50% return+landing.
    Energy is physics-grounded but intentionally lightweight, not an electrochemical battery model.
    """
    distance_m = np.asarray(distance_m); payload=np.asarray(payload)
    w_out=np.asarray(w_out); w_ret=np.asarray(w_ret); vert_m=np.asarray(vert_m)
    hover_s=np.asarray(hover_s); temp_c=np.asarray(temp_c); soh=np.asarray(soh)
    rho = air_density(temp_c)
    eta = np.clip(ETA_NOM*eta_mult*(1.0 - 0.06*(1.0-soh)), 0.45, 0.78)
    cda = CDA_NOM*cda_mult*(1.0 + 0.20*payload/0.5)
    aux = AUX_NOM + aux_extra

    # Loaded outbound and unloaded return masses.
    m_out = M0 + payload
    m_ret = np.full_like(m_out, M0)

    def cruise_power(mass, headwind):
        T = mass*G
        pind = (T**1.5) / np.sqrt(2.0*rho*AREA)
        vair = np.clip(V_GROUND + headwind, 1.0, 18.0)
        pdrag = 0.5*rho*cda*(vair**3) / 0.78
        # Forward flight relieves induced load slightly; profile/aux remain.
        return PROFILE_NOM + 0.84*pind/eta + pdrag + aux

    def hover_power(mass):
        T = mass*G
        pind = (T**1.5) / np.sqrt(2.0*rho*AREA)
        return PROFILE_NOM + pind/eta + aux

    Pout = cruise_power(m_out, w_out)
    Pret = cruise_power(m_ret, w_ret)
    Phov = hover_power(m_out)
    # Climb work plus hover-like support during ascent; descent/landing smaller but nonzero.
    tclimb = np.maximum(vert_m,0.0)/CLIMB_RATE
    Eclimb = (Phov*1.10 + m_out*G*CLIMB_RATE/eta) * tclimb / 3600.0
    Eland  = (hover_power(m_ret)*0.78) * np.maximum(6.0, 0.18*tclimb) / 3600.0

    d_out = 0.5*distance_m
    d_ret = 0.5*distance_m
    Eout = Pout*(d_out/V_GROUND)/3600.0
    Eret = Pret*(d_ret/V_GROUND)/3600.0
    Ehov = Phov*hover_s/3600.0

    # Four operational segments with delivery at end of segment 2.
    s0 = Eclimb + 0.45*Eout
    s1 = 0.55*Eout + Ehov
    s2 = 0.50*Eret
    s3 = 0.50*Eret + Eland
    return np.vstack([s0,s1,s2,s3]).T


def ranges_for(s):
    # total round-trip distance in meters, payload kg, mean headwind m/s, SoH, temp C, vertical climb m
    if s=='nominal':
        return dict(d=(1800,4200), p=(0.05,0.30), w=(-1.5,1.8), soh=(0.90,1.00), T=(15,30), v=(8,35))
    if s=='heavy_payload':
        return dict(d=(1800,4800), p=(0.32,0.55), w=(-1.0,2.0), soh=(0.88,1.00), T=(12,30), v=(10,45))
    if s=='headwind':
        return dict(d=(2000,5000), p=(0.10,0.40), w=(3.0,7.0), soh=(0.88,1.00), T=(10,30), v=(10,45))
    if s=='aged_battery':
        return dict(d=(1800,4700), p=(0.10,0.40), w=(-1.0,2.5), soh=(0.65,0.80), T=(10,30), v=(10,45))
    if s=='cold_temperature':
        return dict(d=(1800,4700), p=(0.10,0.40), w=(-0.5,3.0), soh=(0.80,1.00), T=(-10,5), v=(10,45))
    if s=='vertical_motion':
        return dict(d=(1500,4200), p=(0.10,0.40), w=(-1.0,2.5), soh=(0.85,1.00), T=(8,28), v=(60,150))
    if s=='long_route':
        return dict(d=(4400,7200), p=(0.08,0.42), w=(-0.5,3.5), soh=(0.82,1.00), T=(8,30), v=(10,55))
    if s=='compound':
        return dict(d=(3500,6500), p=(0.32,0.55), w=(3.5,8.0), soh=(0.65,0.82), T=(-6,10), v=(50,125))
    raise KeyError(s)


def generate_missions(n, seed, scenario=None, calibration_mix=False):
    rng = np.random.default_rng(seed)
    if calibration_mix:
        ss = rng.choice(SCENARIOS, size=n, replace=True)
    else:
        ss = np.full(n, scenario, dtype=object)
    # Allocate arrays
    d=np.empty(n); p=np.empty(n); wmean=np.empty(n); soh=np.empty(n); T=np.empty(n); vert=np.empty(n)
    for s in np.unique(ss):
        idx=np.where(ss==s)[0]; rr=ranges_for(s)
        d[idx]=rng.uniform(*rr['d'], size=len(idx))
        p[idx]=rng.uniform(*rr['p'], size=len(idx))
        wmean[idx]=rng.uniform(*rr['w'], size=len(idx))
        soh[idx]=rng.uniform(*rr['soh'], size=len(idx))
        T[idx]=rng.uniform(*rr['T'], size=len(idx))
        vert[idx]=rng.uniform(*rr['v'], size=len(idx))
    hover = rng.uniform(12,30,size=n)
    soc = rng.uniform(0.62,1.0,size=n)

    # Wind differs on outbound/return and can evolve after forecast. Return projection is correlated but not exact opposite.
    w_out_true = wmean + rng.normal(0, 0.8, size=n)
    w_ret_true = -0.45*wmean + rng.normal(0, 1.0, size=n)
    # Scenario-dependent forecast errors; compound/headwind intentionally harder.
    ferr = np.where(np.isin(ss,['headwind','compound']), 1.8, 0.95)
    w_out_fc = w_out_true + rng.normal(0, ferr, size=n)
    w_ret_fc = w_ret_true + rng.normal(0, ferr, size=n)

    # Estimated payload / SoH / temperature and SoC from onboard/preflight data.
    p_est = np.clip(p + rng.normal(0,0.018,size=n),0,0.60)
    soh_est = np.clip(soh + rng.normal(0,0.012,size=n),0.60,1.02)
    T_est = T + rng.normal(0,1.6,size=n)
    soc_est = np.clip(soc + rng.normal(0,0.006,size=n),0.55,1.0)

    # True plant / propulsion uncertainty.
    eta_mult = np.exp(rng.normal(0,0.065,size=n))
    cda_mult = np.exp(rng.normal(0,0.10,size=n))
    aux_extra = np.clip(rng.normal(2.0,2.0,size=n),0,8)
    # Extra unmodeled stochastic energy multiplier (e.g. path corrections / gust response).
    extra_mult = np.exp(rng.normal(0, np.where(np.isin(ss,['compound','headwind']),0.075,0.045), size=n))

    true_ph = phase_energy(d,p,w_out_true,w_ret_true,vert,hover,T,eta_mult,cda_mult,aux_extra,soh)
    true_ph *= extra_mult[:,None]
    true_E = true_ph.sum(axis=1)

    # Nominal preflight physics prediction with forecast / estimated conditions, no hidden true multipliers.
    pred_ph = phase_energy(d,p_est,w_out_fc,w_ret_fc,vert,hover,T_est,
                           np.ones(n),np.ones(n),np.zeros(n),soh_est)
    E_nom = pred_ph.sum(axis=1)

    cap_true = C_NOM*soh*temp_capacity_factor(T)
    cap_est  = C_NOM*soh_est*temp_capacity_factor(T_est)
    avail_true = cap_true*soc
    avail_est = cap_est*soc_est
    reserve_true = RES_FRAC*cap_true
    reserve_est = RES_FRAC*cap_est

    return pd.DataFrame({
        'scenario':ss,'distance_m':d,'payload_true':p,'payload_est':p_est,
        'w_out_true':w_out_true,'w_ret_true':w_ret_true,'w_out_fc':w_out_fc,'w_ret_fc':w_ret_fc,
        'soh_true':soh,'soh_est':soh_est,'temp_true':T,'temp_est':T_est,'vert_m':vert,'hover_s':hover,
        'soc_true':soc,'soc_est':soc_est,'E_true':true_E,'E_nom':E_nom,
        'avail_true':avail_true,'avail_est':avail_est,'reserve_true':reserve_true,'reserve_est':reserve_est,
        'true_s0':true_ph[:,0],'true_s1':true_ph[:,1],'true_s2':true_ph[:,2],'true_s3':true_ph[:,3],
        'pred_s0':pred_ph[:,0],'pred_s1':pred_ph[:,1],'pred_s2':pred_ph[:,2],'pred_s3':pred_ph[:,3],
    })


def feature_matrix(df):
    # Features frozen before final test. Scaling improves conditioning.
    cold=np.maximum(20-df.temp_est.values,0)/20.0
    age=(1-df.soh_est.values)/0.35
    wind=(np.abs(df.w_out_fc.values)+np.abs(df.w_ret_fc.values))/8.0
    return np.column_stack([
        np.ones(len(df)),
        df.E_nom.values,
        df.distance_m.values/5000.0,
        df.payload_est.values/0.5,
        wind,
        df.vert_m.values/100.0,
        cold,
        age,
    ])


def fit_models(fitdf):
    X=feature_matrix(fitdf); y=fitdf.E_true.values
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    mu=X@beta
    absres=np.abs(y-mu)
    # Heteroscedastic log-absolute-residual model with ridge stabilization.
    Z=X.copy(); target=np.log(absres+0.08)
    lam=1e-3
    gamma=np.linalg.solve(Z.T@Z + lam*np.eye(Z.shape[1]), Z.T@target)
    # Distance-only strong geometric calibration.
    Xg=np.column_stack([np.ones(len(fitdf)), fitdf.distance_m.values/5000.0])
    bg=np.linalg.lstsq(Xg,y,rcond=None)[0]
    return beta,gamma,bg


def predict_models(df,beta,gamma,bg):
    X=feature_matrix(df)
    mu=X@beta
    # convert predicted mean absolute residual to approximately sigma; floor avoids zero uncertainty.
    sig=np.maximum(np.exp(X@gamma)*math.sqrt(math.pi/2),0.10)
    Xg=np.column_stack([np.ones(len(df)), df.distance_m.values/5000.0])
    geo=Xg@bg
    return mu,sig,geo


def wilson(k,n,z=1.959963984540054):
    if n==0: return (np.nan,np.nan)
    p=k/n; den=1+z*z/n
    c=(p+z*z/(2*n))/den
    h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return c-h,c+h


def holm_adjust(pvals):
    pvals=np.asarray(pvals,float); m=len(pvals)
    order=np.argsort(pvals); out=np.empty(m); running=0.0
    for rank,idx in enumerate(order):
        adj=(m-rank)*pvals[idx]
        running=max(running,adj)
        out[idx]=min(running,1.0)
    return out


def exact_mcnemar(a,b):
    # a,b boolean outcomes where True is event (e.g., violation)
    b01=np.sum((~a)&b); b10=np.sum(a&(~b)); n=b01+b10
    if n==0: return 1.0,int(b01),int(b10)
    return float(binomtest(min(b01,b10), n=n, p=0.5, alternative='two-sided').pvalue),int(b01),int(b10)


def evaluate_policy(df, policy, mu,sig,geo,kappa):
    Eavail=df.avail_est.values; R=df.reserve_est.values
    trueE=df.E_true.values; trueAvail=df.avail_true.values; trueR=df.reserve_true.values
    feasible=trueE+trueR <= trueAvail
    if policy=='GEO': pred=geo
    elif policy=='DET': pred=mu
    elif policy in ('RISK','ADAPT'): pred=mu+kappa*sig
    else: raise KeyError(policy)
    admit=pred+R <= Eavail

    # Defaults for non-admitted missions
    energy_used=np.zeros(len(df)); completed=np.zeros(len(df),dtype=bool); delivered=np.zeros(len(df),dtype=bool)
    diverted=np.zeros(len(df),dtype=bool); final_energy=np.full(len(df),np.nan)
    if policy!='ADAPT':
        energy_used[admit]=trueE[admit]
        completed[admit]=True; delivered[admit]=True
        final_energy[admit]=trueAvail[admit]-trueE[admit]
    else:
        # Shared preflight gate with RISK, then online adaptation at 3 checkpoints.
        trueph=df[['true_s0','true_s1','true_s2','true_s3']].values
        predph=df[['pred_s0','pred_s1','pred_s2','pred_s3']].values
        for i in np.where(admit)[0]:
            consumed=0.0; pred_cum=0.0; scale=1.0; stopped=False
            for j in range(4):
                consumed += trueph[i,j]
                pred_cum += predph[i,j]
                # Delivery occurs at end of segment 1.
                if j>=1: delivered[i]=True
                if j<3:
                    ratio=consumed/max(pred_cum,1e-9)
                    # Bounded EWMA update; deliberately simple and transparent.
                    scale=np.clip(0.35*scale + 0.65*ratio, 0.65, 1.75)
                    pred_rem=predph[i,j+1:].sum()*scale
                    frac=max(predph[i,j+1:].sum()/max(predph[i].sum(),1e-9),0.0)
                    sig_rem=sig[i]*math.sqrt(frac)
                    projected=consumed + pred_rem + kappa*sig_rem + df.reserve_est.iloc[i]
                    if projected > df.avail_est.iloc[i]:
                        diverted[i]=True
                        true_remaining=trueph[i,j+1:].sum()
                        # Contingency site requires 45% of original remaining route energy.
                        diverted_energy=DIVERT_FACTOR*true_remaining
                        consumed += diverted_energy
                        energy_used[i]=consumed
                        final_energy[i]=trueAvail[i]-consumed
                        stopped=True
                        break
            if not stopped:
                energy_used[i]=consumed
                completed[i]=True; delivered[i]=True
                final_energy[i]=trueAvail[i]-consumed

    reserve_violation=np.zeros(len(df),dtype=bool)
    reserve_violation[admit]=final_energy[admit] < trueR[admit]
    unsafe_accept=admit & (~feasible)
    false_reject=(~admit) & feasible
    terminal_soc=np.full(len(df),np.nan)
    terminal_soc[admit]=final_energy[admit]/(C_NOM*df.soh_true.values[admit]*temp_capacity_factor(df.temp_true.values[admit]))
    margin=np.full(len(df),np.nan)
    margin[admit]=final_energy[admit]-trueR[admit]
    return pd.DataFrame({
        'policy':policy,'admit':admit,'true_feasible':feasible,'unsafe_accept':unsafe_accept,
        'false_reject':false_reject,'reserve_violation':reserve_violation,'completed':completed,
        'delivered':delivered,'diverted':diverted,'energy_used':energy_used,'terminal_soc':terminal_soc,
        'reserve_margin_Wh':margin,'pred_bound':pred,'mu':mu,'sigma':sig,
    })


def summarize(df, res):
    n=len(df); admit=res.admit.values; feasible=res.true_feasible.values
    ninf=np.sum(~feasible); nfeas=np.sum(feasible); nad=np.sum(admit)
    def rate(mask, denom): return float(np.sum(mask)/denom) if denom else np.nan
    rv=res.reserve_violation.values
    delivered=res.delivered.values; completed=res.completed.values
    eper=np.nan
    if np.sum(delivered)>0: eper=float(res.energy_used.values[delivered].sum()/np.sum(delivered))
    return dict(
        n=n, admission_rate=rate(admit,n),
        unsafe_admission_rate=rate(res.unsafe_accept.values,ninf),
        false_rejection_rate=rate(res.false_reject.values,nfeas),
        reserve_violation_rate_admitted=rate(rv & admit,nad),
        completion_rate_all=rate(completed,n), delivery_success_rate_all=rate(delivered,n),
        diversion_rate_admitted=rate(res.diverted.values & admit,nad),
        mean_terminal_soc=float(np.nanmean(res.terminal_soc.values)),
        mean_reserve_margin_Wh=float(np.nanmean(res.reserve_margin_Wh.values)),
        energy_per_delivered_Wh=eper,
        n_admitted=int(nad), n_true_feasible=int(nfeas), n_true_infeasible=int(ninf)
    )


def main():
    # Calibration generation and model fitting
    fitdf=generate_missions(12000,21000,calibration_mix=True)
    caldf=generate_missions(12000,22000,calibration_mix=True)
    beta,gamma,bg=fit_models(fitdf)
    mu_cal,sig_cal,geo_cal=predict_models(caldf,beta,gamma,bg)
    z=(caldf.E_true.values-mu_cal)/sig_cal
    kappa=float(np.quantile(z,TARGET_COVERAGE,method='higher'))
    cal_cov=float(np.mean(caldf.E_true.values <= mu_cal+kappa*sig_cal))
    fit_meta={
        'target_coverage':TARGET_COVERAGE,'kappa':kappa,'calibration_coverage':cal_cov,
        'mean_beta':beta.tolist(),'sigma_gamma':gamma.tolist(),'geo_beta':bg.tolist(),
        'fit_seed':21000,'risk_cal_seed':22000,'fit_n':12000,'risk_cal_n':12000,
    }
    (OUT/'CALIBRATION_PARAMETERS.json').write_text(json.dumps(fit_meta,indent=2))

    # Final independent test generated only now.
    all_base=[]; all_res=[]
    for si,s in enumerate(SCENARIOS):
        base=generate_missions(6000,30000+si,scenario=s)
        base['mission_id']=[f'{s}_{j:05d}' for j in range(len(base))]
        mu,sig,geo=predict_models(base,beta,gamma,bg)
        base['mu']=mu; base['sigma']=sig; base['geo_pred']=geo
        base['risk_bound']=mu+kappa*sig
        base['risk_covered']=base.E_true.values <= base.risk_bound.values
        all_base.append(base)
        for pol in ['GEO','DET','RISK','ADAPT']:
            rr=evaluate_policy(base,pol,mu,sig,geo,kappa)
            rr.insert(0,'mission_id',base.mission_id.values); rr.insert(1,'scenario',s)
            all_res.append(rr)
    base=pd.concat(all_base,ignore_index=True)
    res=pd.concat(all_res,ignore_index=True)

    # Summary by scenario/policy and pooled
    rows=[]
    for s in SCENARIOS+['POOLED']:
        b=base if s=='POOLED' else base[base.scenario==s]
        for pol in ['GEO','DET','RISK','ADAPT']:
            r=res[(res.policy==pol) & (res.mission_id.isin(b.mission_id))]
            sm=summarize(b.reset_index(drop=True),r.reset_index(drop=True))
            sm.update(scenario=s,policy=pol)
            rows.append(sm)
    summary=pd.DataFrame(rows)
    # Risk coverage separately
    covrows=[]
    for s in SCENARIOS+['POOLED']:
        b=base if s=='POOLED' else base[base.scenario==s]
        k=int(b.risk_covered.sum()); n=len(b); lo,hi=wilson(k,n)
        covrows.append(dict(scenario=s,n=n,coverage=k/n,ci_low=lo,ci_high=hi,
                            mean_abs_error=float(np.mean(np.abs(b.E_true-b.mu))),
                            rmse=float(np.sqrt(np.mean((b.E_true-b.mu)**2))),
                            mean_sigma=float(np.mean(b.sigma))))
    coverage=pd.DataFrame(covrows)

    # Add Wilson intervals to main rates.
    for col,denomcol in [
        ('admission_rate','n'),('unsafe_admission_rate','n_true_infeasible'),('false_rejection_rate','n_true_feasible'),
        ('reserve_violation_rate_admitted','n_admitted'),('completion_rate_all','n'),('delivery_success_rate_all','n'),
        ('diversion_rate_admitted','n_admitted')]:
        lows=[]; highs=[]
        for _,r in summary.iterrows():
            den=int(r[denomcol]); val=r[col]; k=int(round(val*den)) if den and np.isfinite(val) else 0
            lo,hi=wilson(k,den) if den else (np.nan,np.nan); lows.append(lo); highs.append(hi)
        summary[col+'_ci_low']=lows; summary[col+'_ci_high']=highs

    # Principal paired statistical tests on pooled missions.
    piv={p:res[res.policy==p].sort_values('mission_id').reset_index(drop=True) for p in ['GEO','DET','RISK','ADAPT']}
    tests=[]
    # Reserve violation DET vs RISK on missions admitted by DET or RISK; paired outcomes across all missions (rejected treated no violation would bias),
    # so primary test restricts common admitted missions.
    common=piv['DET'].admit.values & piv['RISK'].admit.values
    p,b01,b10=exact_mcnemar(piv['DET'].reserve_violation.values[common],piv['RISK'].reserve_violation.values[common])
    tests.append(dict(test='Reserve violation DET vs RISK (common admitted)',p_raw=p,n=len(np.where(common)[0]),discordant_a0_b1=b01,discordant_a1_b0=b10))
    common_ra=piv['RISK'].admit.values & piv['ADAPT'].admit.values
    p,b01,b10=exact_mcnemar(piv['RISK'].reserve_violation.values[common_ra],piv['ADAPT'].reserve_violation.values[common_ra])
    tests.append(dict(test='Reserve violation RISK vs ADAPT (shared gate)',p_raw=p,n=int(common_ra.sum()),discordant_a0_b1=b01,discordant_a1_b0=b10))
    # Delivery loss RISK vs ADAPT paired over all missions
    p,b01,b10=exact_mcnemar(piv['RISK'].delivered.values,piv['ADAPT'].delivered.values)
    tests.append(dict(test='Delivery success RISK vs ADAPT',p_raw=p,n=len(base),discordant_a0_b1=b01,discordant_a1_b0=b10))
    # Unsafe acceptance DET vs RISK among ground-truth infeasible
    infeas=~piv['DET'].true_feasible.values
    p,b01,b10=exact_mcnemar(piv['DET'].admit.values[infeas],piv['RISK'].admit.values[infeas])
    tests.append(dict(test='Unsafe admission DET vs RISK (true infeasible)',p_raw=p,n=int(infeas.sum()),discordant_a0_b1=b01,discordant_a1_b0=b10))
    stats=pd.DataFrame(tests); stats['p_holm']=holm_adjust(stats.p_raw.values)

    # Continuous paired comparison: reserve margin on common completed missions DET vs RISK is identical actual execution when both complete, not informative.
    # RISK vs ADAPT energy used among missions delivered by both captures diversion savings/costs.
    common_del=piv['RISK'].delivered.values & piv['ADAPT'].delivered.values
    diff=piv['ADAPT'].energy_used.values[common_del]-piv['RISK'].energy_used.values[common_del]
    if len(diff)>0 and np.any(np.abs(diff)>1e-12):
        w=wilcoxon(diff,alternative='two-sided',zero_method='wilcox')
        cont=dict(test='Energy used ADAPT-RISK among mutually delivered',n=len(diff),median_difference_Wh=float(np.median(diff)),mean_difference_Wh=float(np.mean(diff)),p_raw=float(w.pvalue))
    else:
        cont=dict(test='Energy used ADAPT-RISK among mutually delivered',n=len(diff),median_difference_Wh=float(np.median(diff)) if len(diff) else np.nan,mean_difference_Wh=float(np.mean(diff)) if len(diff) else np.nan,p_raw=1.0)
    pd.DataFrame([cont]).to_csv(OUT/'continuous_paired_statistics.csv',index=False)

    # Predeclared success criteria using pooled rows.
    pool=summary[summary.scenario=='POOLED'].set_index('policy')
    cov=float(coverage.loc[coverage.scenario=='POOLED','coverage'].iloc[0])
    det_rv=float(pool.loc['DET','reserve_violation_rate_admitted'])
    risk_rv=float(pool.loc['RISK','reserve_violation_rate_admitted'])
    adapt_rv=float(pool.loc['ADAPT','reserve_violation_rate_admitted'])
    risk_fr=float(pool.loc['RISK','false_rejection_rate'])
    adapt_div=float(pool.loc['ADAPT','diversion_rate_admitted'])
    risk_del=float(pool.loc['RISK','delivery_success_rate_all'])
    adapt_del=float(pool.loc['ADAPT','delivery_success_rate_all'])
    crit=[
        ('RISK coverage >= 96.0%', cov>=0.96, cov, 0.96),
        ('RISK reserve violations reduced >=50% vs DET', (det_rv>0 and risk_rv<=0.5*det_rv), 1-risk_rv/det_rv if det_rv>0 else np.nan, 0.50),
        ('RISK false rejection <=20%', risk_fr<=0.20, risk_fr,0.20),
        ('ADAPT reserve violations reduced >=40% vs RISK', (risk_rv>0 and adapt_rv<=0.60*risk_rv), 1-adapt_rv/risk_rv if risk_rv>0 else np.nan,0.40),
        ('ADAPT diversion <=15% admitted', adapt_div<=0.15, adapt_div,0.15),
        ('ADAPT delivery loss <=8 percentage points vs RISK', (risk_del-adapt_del)<=0.08, risk_del-adapt_del,0.08),
    ]
    criteria=pd.DataFrame(crit,columns=['criterion','pass','observed','threshold'])

    # Persist data.
    base.to_csv(OUT/'final_mission_truth_and_predictions.csv',index=False)
    res.to_csv(OUT/'final_policy_trial_metrics.csv',index=False)
    summary.to_csv(OUT/'final_summary.csv',index=False)
    coverage.to_csv(OUT/'final_coverage.csv',index=False)
    stats.to_csv(OUT/'final_paired_statistics.csv',index=False)
    criteria.to_csv(OUT/'SUCCESS_CRITERIA.csv',index=False)

    # Figures — separate plots, default matplotlib colors/styles.
    ptab=pool.loc[['GEO','DET','RISK','ADAPT']]
    fig,ax=plt.subplots(figsize=(7,4.4))
    ax.bar(ptab.index,100*ptab.reserve_violation_rate_admitted.values)
    ax.set_ylabel('Reserve violations among admitted missions (%)')
    ax.set_title('E2 pooled reserve-risk outcome')
    ax.grid(axis='y',alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT/'e2_reserve_violations.png',dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4.4))
    ax.bar(ptab.index,100*ptab.false_rejection_rate.values)
    ax.set_ylabel('False rejection of truly feasible missions (%)')
    ax.set_title('E2 conservatism of preflight admission')
    ax.grid(axis='y',alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT/'e2_false_rejection.png',dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,4.8))
    c=coverage[coverage.scenario!='POOLED']
    ax.bar(c.scenario,100*c.coverage.values)
    ax.axhline(96.0,linestyle='--',linewidth=1,label='predeclared minimum 96%')
    ax.axhline(97.5,linestyle=':',linewidth=1,label='target 97.5%')
    ax.set_ylabel('Risk-bound coverage (%)'); ax.set_ylim(90,100)
    ax.set_title('E2 independent-test upper-bound coverage by scenario')
    ax.tick_params(axis='x',rotation=35); ax.legend(); ax.grid(axis='y',alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT/'e2_coverage_by_scenario.png',dpi=180); plt.close(fig)

    # Manuscript-ready results text and final report.
    risk_reduction=100*(1-risk_rv/det_rv) if det_rv>0 else np.nan
    adapt_reduction=100*(1-adapt_rv/risk_rv) if risk_rv>0 else np.nan
    report=f'''# E2 Final Report — Cross-Layer Risk-Bounded Energy Admission and In-Flight Mission Adaptation\n\n## Evidence level\nIndependent physics-grounded / source-informed simulation. This experiment tests the manuscript's Eq. (2)-(3) risk-bounded admission contract, but it is not package-native AdaptiveQuadBench energy validation, HIL, flight, or an electrochemical battery experiment.\n\n## Calibration\n- Model-fit missions: 12,000 (seed 21000).\n- Independent risk-calibration missions: 12,000 (seed 22000).\n- One-sided target coverage: {TARGET_COVERAGE*100:.1f}%.\n- Frozen calibration multiplier: **kappa = {kappa:.3f}**.\n- Calibration-split upper-bound coverage: **{cal_cov*100:.2f}%**.\n\n## Independent final test\n48,000 untouched missions across eight stress scenarios (6,000 per scenario).\n\n### Pooled principal outcomes\n- Risk-bound coverage: **{cov*100:.2f}%**.\n- DET reserve-violation rate among admitted missions: **{det_rv*100:.2f}%**.\n- RISK reserve-violation rate among admitted missions: **{risk_rv*100:.2f}%** (**{risk_reduction:.1f}% reduction vs DET**).\n- RISK false-rejection rate among truly feasible missions: **{risk_fr*100:.2f}%**.\n- ADAPT reserve-violation rate among admitted missions: **{adapt_rv*100:.2f}%** (**{adapt_reduction:.1f}% reduction vs RISK**).\n- ADAPT diversion rate among admitted missions: **{adapt_div*100:.2f}%**.\n- RISK delivery-success rate over all candidate missions: **{risk_del*100:.2f}%**.\n- ADAPT delivery-success rate over all candidate missions: **{adapt_del*100:.2f}%** (loss **{(risk_del-adapt_del)*100:.2f} percentage points**).\n\n## Frozen success criteria\n{criteria.to_markdown(index=False)}\n\n## Interpretation\nThe experiment should be interpreted through both safety and conservatism. A risk bound that trivially rejects almost everything would not support H1. Conversely, a deterministic mean model that accepts many missions but frequently consumes the protected reserve would not satisfy R1. The paired comparison isolates the uncertainty-margin effect because DET and RISK share the same calibrated mean model; ADAPT then tests whether in-flight measured energy can repair residual preflight mismatch without changing the preflight gate.\n\n## Claim boundary\nA positive result supports the narrow statement that, in this physics-grounded simulation family, calibrated uncertainty margins and online energy feedback improve reserve protection relative to distance-only or deterministic admission while maintaining bounded conservatism. It does not establish flight-certified battery safety, package-native benchmark validation, or universal optimality of the selected energy model.\n'''
    (OUT/'FINAL_REPORT.md').write_text(report)

    insert=f'''### Cross-layer risk-bounded energy admission and in-flight adaptation (E2)\n\nTo move H1/R1 beyond architecture-only evidence, we conducted an independent physics-grounded mission-energy experiment implementing the admission contract $E_{{risk}}=\\mu_E+\\kappa\\sigma_E$ with a protected reserve. A calibrated deterministic mean-energy twin (DET), the same twin with a one-sided uncertainty margin (RISK), and RISK augmented with in-flight measured-energy adaptation/diversion (ADAPT) were compared against a calibrated distance-only baseline on 48,000 untouched missions spanning nominal, payload, headwind, aged-battery, cold-temperature, vertical-motion, long-route, and compound conditions. The uncertainty multiplier was frozen on a separate 12,000-mission calibration split for 97.5% target upper-bound coverage. On the independent test, coverage was {cov*100:.2f}%. RISK reduced admitted-mission reserve violations from {det_rv*100:.2f}% with DET to {risk_rv*100:.2f}% ({risk_reduction:.1f}% relative reduction), with a {risk_fr*100:.2f}% false-rejection rate among truly feasible missions. ADAPT further changed the reserve-violation rate to {adapt_rv*100:.2f}% while diverting {adapt_div*100:.2f}% of admitted missions; delivery success changed by {(risk_del-adapt_del)*100:.2f} percentage points relative to RISK. These results provide direct simulation evidence for the manuscript's energy-risk contract, but remain source-informed rather than package-native or flight-validated.\n'''
    (OUT/'MANUSCRIPT_INSERT_E2.md').write_text(insert)

    # SHA manifest
    manifest=[]
    for fn in sorted(OUT.iterdir()):
        if fn.is_file() and fn.name!='SHA256SUMS':
            h=hashlib.sha256(fn.read_bytes()).hexdigest(); manifest.append(f'{h}  {fn.name}')
    (OUT/'SHA256SUMS').write_text('\n'.join(manifest)+'\n')

    print(json.dumps({
        'kappa':kappa,'cal_cov':cal_cov,'final_cov':cov,
        'det_rv':det_rv,'risk_rv':risk_rv,'risk_reduction':risk_reduction,
        'risk_false_reject':risk_fr,'adapt_rv':adapt_rv,'adapt_reduction':adapt_reduction,
        'adapt_diversion':adapt_div,'risk_delivery':risk_del,'adapt_delivery':adapt_del,
        'criteria_passed':int(criteria['pass'].sum()),'criteria_total':len(criteria)
    }, indent=2))

if __name__=='__main__':
    main()

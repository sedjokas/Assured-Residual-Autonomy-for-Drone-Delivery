import importlib.util, pandas as pd, numpy as np, json, hashlib, time, math
from pathlib import Path
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
spec=importlib.util.spec_from_file_location('e1','/mnt/data/e1_fast.py'); e1=importlib.util.module_from_spec(spec);spec.loader.exec_module(e1)
OUT=e1.OUT
st=time.time(); e1.TAUH,e1.DML,e1.DEA,e1.NOBS=e1.calibration_data()
cal={'tau_hat':e1.TAUH.tolist(),'dmax_lumped':e1.DML.tolist(),'dmax_ext':e1.DEA.tolist(),'nobs':e1.NOBS}
json.dump(cal,open(OUT/'calibration_final.json','w'),indent=2)
# Independent ordinary tests
rows=e1.run_set(e1.TEST,0)
# Fresh faults on critical scenarios
for fid in [1,2]: rows += e1.run_set(e1.FAULT,fid,scenarios=['dryden3','compound'])
cols=['seed','scenario','filter','fault','rmse','max_error','gt1','violation','active','infeasible','du']
df=pd.DataFrame(rows,columns=cols);df.to_csv(OUT/'final_trial_metrics.csv',index=False)
s=e1.summarize(df);s.to_csv(OUT/'final_summary.csv',index=False)
# paired stats helper
statrows=[]
def paired(fault,scenario,metric,a,b):
    x=df[(df.fault==fault)&(df.scenario==scenario)&(df['filter']==a)].sort_values('seed')[metric].to_numpy()
    y=df[(df.fault==fault)&(df.scenario==scenario)&(df['filter']==b)].sort_values('seed')[metric].to_numpy()
    d=x-y
    if np.allclose(d,0): p=1.;stat=0.
    else:
        try: stat,p=wilcoxon(d,zero_method='wilcox',alternative='two-sided')
        except: stat,p=np.nan,1.
    # deterministic bootstrap CI seed fixed separately from experiment
    rng=np.random.default_rng(20260912); means=[]
    n=len(d)
    idx=rng.integers(0,n,size=(4000,n)); means=d[idx].mean(axis=1); lo,hi=np.quantile(means,[.025,.975])
    return dict(fault=fault,scenario=scenario,metric=metric,a=a,b=b,n=n,mean_a=x.mean(),mean_b=y.mean(),mean_diff=d.mean(),ci95_lo=lo,ci95_hi=hi,p=p,stat=stat)
for fault in ['stuck']:
  for sc in ['dryden3','compound']:
    for metric in ['rmse','violation','gt1','infeasible']:
      for b in ['LM','P']:
        statrows.append(paired(fault,sc,metric,'AA',b))
stats=pd.DataFrame(statrows)
# Holm adjustment for primary AA-vs-LM RMSE tests (2 tests)
mask=(stats.metric=='rmse')&(stats.b=='LM'); inds=list(stats[mask].index); ps=stats.loc[inds,'p'].to_numpy(); order=np.argsort(ps); adj=np.empty(len(ps));run=0
for rank,ii in enumerate(order):
    val=(len(ps)-rank)*ps[ii];run=max(run,val);adj[ii]=min(1,run)
stats['p_holm_primary']=np.nan
for j,idx in enumerate(inds):stats.loc[idx,'p_holm_primary']=adj[j]
stats.to_csv(OUT/'final_paired_statistics.csv',index=False)
# Criteria
get=lambda f,sc,fi:s[(s.fault==f)&(s.scenario==sc)&(s['filter']==fi)].iloc[0]
crit={}
for sc in e1.SCEN_NAMES:
    aa=get('none',sc,'AA');pp=get('none',sc,'P')
    crit[f'healthy_transparency_{sc}']=bool(aa.rmse<=1.02*pp.rmse+1e-12)
crit['nominal_intervention_lt_1pct']=bool(get('none','nominal','AA').active<.01)
for sc in ['dryden3','compound']:
    aa=get('stuck',sc,'AA');lm=get('stuck',sc,'LM');pp=get('stuck',sc,'P')
    crit[f'stuck_{sc}_AA_better_RMSE_than_P']=bool(aa.rmse<pp.rmse)
    crit[f'stuck_{sc}_AA_lower_violation_than_P']=bool(aa.violation<pp.violation)
    crit[f'stuck_{sc}_AA_no_worse_RMSE_than_LM']=bool(aa.rmse<=lm.rmse)
    crit[f'stuck_{sc}_AA_no_worse_violation_than_LM']=bool(aa.violation<=lm.violation)
    crit[f'stuck_{sc}_AA_no_worse_infeasible_than_LM']=bool(aa.infeasible<=lm.infeasible)
for sc in ['dryden3','compound']:
    rr=stats[(stats.fault=='stuck')&(stats.scenario==sc)&(stats.metric=='rmse')&(stats.b=='LM')].iloc[0]
    crit[f'stuck_{sc}_AA_vs_LM_RMSE_Holm_p_lt_0.05']=bool(rr.p_holm_primary<.05)
json.dump(crit,open(OUT/'final_acceptance.json','w'),indent=2)
# Summary report
lines=['# E1 Final Results — Actuator-Aware Runtime Safety Validation','',
'**Evidence label:** independent source-informed simulation; not package-native AdaptiveQuadBench/acados.','',
'## Calibration-only frozen model parameters',
f"- Identified command-realization time constants: {np.round(e1.TAUH,4).tolist()} s",
f"- Lumped-model mismatch bounds: {np.round(e1.DML,4).tolist()} m/s²",
f"- Actuator-aware external-disturbance bounds: {np.round(e1.DEA,4).tolist()} m/s²",'',
'## Independent ordinary-test results','',s[s.fault=='none'][['scenario','filter','rmse','violation','active','infeasible','gt1']].to_markdown(index=False,floatfmt='.5f'),'',
'## Fresh post-proposal challenges','',s[s.fault!='none'][['fault','scenario','filter','rmse','max_error','gt1','violation','active','infeasible']].to_markdown(index=False,floatfmt='.5f'),'',
'## Confirmatory paired statistics: AA versus LM','',stats[(stats.b=='LM')][['fault','scenario','metric','mean_a','mean_b','mean_diff','ci95_lo','ci95_hi','p','p_holm_primary']].to_markdown(index=False,floatfmt='.6g'),'',
'## Frozen acceptance criteria']
for k,v in crit.items():lines.append(f"- {'PASS' if v else 'FAIL'} — {k}")
lines += ['',f'Runtime: {time.time()-st:.1f} s','',
'## Interpretation',
'In ordinary operation all three controllers are expected to be nearly identical because the safety envelopes are not approached. That is a transparency test. The short impulse is intentionally retained even if the filters remain inactive: if the predicted envelope stays admissible, non-intervention is correct. The severe stuck-proposal challenge is the decisive filter stress test. The main comparison is whether AA improves containment over P and over the lumped instantaneous-command LM filter while reducing or not increasing filter infeasibility.', '',
'## Claim boundary',
'These results can support a new manuscript subsection as a mechanism-level actuator-aware confirmation. They cannot be called package-native AdaptiveQuadBench/acados evidence or HIL/flight validation. A final package-native rerun using the authors’ frozen CeNN/governor wrapper would remain the strongest next confirmation.' ]
(OUT/'FINAL_REPORT.md').write_text('\n'.join(lines))
# Plots: default matplotlib colors
# 1 stuck challenge RMSE
plot=s[s.fault=='stuck'].copy(); scenarios=['dryden3','compound'];filters=['P','LM','AA']
fig,ax=plt.subplots(figsize=(7.2,4.2));x=np.arange(len(scenarios));width=.24
for i,f in enumerate(filters):
    vals=[plot[(plot.scenario==sc)&(plot['filter']==f)].rmse.iloc[0] for sc in scenarios]
    ax.bar(x+(i-1)*width,vals,width,label=f)
ax.set_xticks(x);ax.set_xticklabels(['Severe Dryden','Compound']);ax.set_ylabel('Mean position RMSE [m]');ax.set_title('E1: Severe stuck-proposal containment');ax.legend();fig.tight_layout();fig.savefig(OUT/'e1_stuck_rmse.png',dpi=180);plt.close(fig)
# 2 violation
fig,ax=plt.subplots(figsize=(7.2,4.2))
for i,f in enumerate(filters):
    vals=[100*plot[(plot.scenario==sc)&(plot['filter']==f)].violation.iloc[0] for sc in scenarios]
    ax.bar(x+(i-1)*width,vals,width,label=f)
ax.set_xticks(x);ax.set_xticklabels(['Severe Dryden','Compound']);ax.set_ylabel('Envelope violation [% cycles]');ax.set_title('E1: Safety-envelope violations');ax.legend();fig.tight_layout();fig.savefig(OUT/'e1_stuck_violations.png',dpi=180);plt.close(fig)
# manifest
manifest={}
for fp in sorted(OUT.glob('*')):
    if fp.is_file():manifest[fp.name]=hashlib.sha256(fp.read_bytes()).hexdigest()
json.dump(manifest,open(OUT/'SHA256SUMS.json','w'),indent=2)
print((OUT/'FINAL_REPORT.md').read_text())

import os,json,glob,hashlib,zipfile,time,shutil
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from qcenn_eval_core import load_predictor,OUT
SCENARIOS=['nominal','wind3','force_step','model20','latency40','payload50','rotoreff30','compound']
STRESSED=SCENARIOS[1:]
ARCHS=['LinearCeNN-494','QCeNN-XX-503','QCeNN-XXXY-512','LinearCeNN-Capacity-515']
# Final trial files.
files=[]
for a in ARCHS:
 for s in SCENARIOS:
  p=OUT/f'final_{a}_{s}_200seeds.csv'
  if not p.exists(): raise FileNotFoundError(p)
  files.append(p)
final=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
final.to_csv(OUT/'final_trials_200seeds.csv',index=False)
summary=final.groupby(['scenario','architecture'],sort=False).agg(n=('seed','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),p95_mean=('p95_m','mean'),max_mean=('max_m','mean'),excursion_gt1m_mean_pct=('excursion_gt1m_pct','mean'),control_effort_mean=('control_effort_mean','mean'),command_saturation_mean_pct=('command_saturation_pct','mean'),residual_rms_mean=('residual_rms_mps2','mean'),latent_max=('latent_max','max'),latent_p999_mean=('latent_p999','mean'),finite_rate=('finite','mean'),residual_bound_violation_mean_pct=('residual_bound_violation_pct','mean')).reset_index()
summary.to_csv(OUT/'final_summary_200seeds.csv',index=False)

def paired_stat(sc,prop,comp,nboot=20000):
 a=final[(final.scenario==sc)&(final.architecture==comp)].sort_values('seed');b=final[(final.scenario==sc)&(final.architecture==prop)].sort_values('seed');assert np.array_equal(a.seed.values,b.seed.values)
 d=a.rmse_m.values-b.rmse_m.values;rng=np.random.default_rng(88000+sum(map(ord,sc+prop+comp)));boot=d[rng.integers(0,len(d),(nboot,len(d)))].mean(1)
 try:p=float(wilcoxon(d,zero_method='wilcox').pvalue) if not np.allclose(d,0) else 1.0
 except:p=1.0
 return {'scenario':sc,'proposal':prop,'comparator':comp,'n':len(d),'comparator_rmse':float(a.rmse_m.mean()),'proposal_rmse':float(b.rmse_m.mean()),'improvement_m':float(d.mean()),'improvement_pct':float(100*d.mean()/a.rmse_m.mean()),'ci95_low_m':float(np.quantile(boot,.025)),'ci95_high_m':float(np.quantile(boot,.975)),'wilcoxon_p':p,'proposal_better_fraction':float(np.mean(d>0)),'excursion_delta_pp':float(b.excursion_gt1m_pct.mean()-a.excursion_gt1m_pct.mean()),'saturation_delta_pp':float(b.command_saturation_pct.mean()-a.command_saturation_pct.mean()),'residual_rms_delta':float(b.residual_rms_mps2.mean()-a.residual_rms_mps2.mean())}
rows=[]
for sc in SCENARIOS:
 for prop,comp in [('QCeNN-XX-503','LinearCeNN-494'),('QCeNN-XXXY-512','LinearCeNN-494'),('QCeNN-XXXY-512','QCeNN-XX-503'),('QCeNN-XXXY-512','LinearCeNN-Capacity-515')]:rows.append(paired_stat(sc,prop,comp))
stats=pd.DataFrame(rows);stats['holm_p_stressed']=np.nan
for _,grp in stats[stats.scenario!='nominal'].groupby(['proposal','comparator']):
 p=grp.wilcoxon_p.values;idxs=grp.index.values;order=np.argsort(p);adj=np.empty(len(p));running=0.;m=len(p)
 for rank,ix in enumerate(order):running=max(running,(m-rank)*p[ix]);adj[ix]=min(1.,running)
 stats.loc[idxs,'holm_p_stressed']=adj
stats.to_csv(OUT/'paired_statistics.csv',index=False)
qbase=stats[(stats.proposal=='QCeNN-XXXY-512')&(stats.comparator=='LinearCeNN-494')]
qcap=stats[(stats.proposal=='QCeNN-XXXY-512')&(stats.comparator=='LinearCeNN-Capacity-515')]
qxx=stats[(stats.proposal=='QCeNN-XXXY-512')&(stats.comparator=='QCeNN-XX-503')]
mean_base=float(qbase[qbase.scenario!='nominal'].improvement_pct.mean());mean_cap=float(qcap[qcap.scenario!='nominal'].improvement_pct.mean());mean_xx=float(qxx[qxx.scenario!='nominal'].improvement_pct.mean())
nom=float(qbase[qbase.scenario=='nominal'].improvement_pct.iloc[0]);pos_ci=int(((qbase.scenario!='nominal')&(qbase.ci95_low_m>0)).sum());pos_ci_cap=int(((qcap.scenario!='nominal')&(qcap.ci95_low_m>0)).sum());max_exc=float(qbase[qbase.scenario!='nominal'].excursion_delta_pp.max());max_sat=float(qbase[qbase.scenario!='nominal'].saturation_delta_pp.max())
criteria={'q_vs_linear_positive_CI_scenarios':pos_ci,'q_vs_linear_mean_stressed_improvement_pct':mean_base,'q_vs_linear_nominal_degradation_pct':float(max(0,-nom)),'q_vs_linear_nominal_improvement_pct':float(max(0,nom)),'q_vs_linear_max_excursion_increase_pp':max_exc,'q_vs_linear_max_saturation_increase_pp':max_sat,'q_vs_linear_pass':bool(pos_ci>=4 and mean_base>=.25 and max(0,-nom)<.2 and max_exc<=.05 and max_sat<=.05),'structure_vs_capacity_positive_CI_scenarios':pos_ci_cap,'structure_vs_capacity_mean_stressed_improvement_pct':mean_cap,'structure_vs_capacity_pass':bool(pos_ci_cap>=4 or mean_cap>=.10),'xy_increment_mean_stressed_improvement_pct':mean_xx,'xy_increment_pass':bool(mean_xx>0),'all_missions_finite':bool(final.finite.all()),'residual_bound_violations_observed':int((final.residual_bound_violation_pct>0).sum())}
criteria['primary_overall_pass']=bool(criteria['q_vs_linear_pass'] and criteria['structure_vs_capacity_pass'] and criteria['xy_increment_pass'] and criteria['all_missions_finite'] and criteria['residual_bound_violations_observed']==0)
json.dump(criteria,open(OUT/'completion_criteria.json','w'),indent=2)
# Latency.
rng=np.random.default_rng(444);tn=rng.normal(size=(1,9,3));lat=[]
for a in ARCHS:
 pred,*_=load_predictor(a)
 for _ in range(300):pred(tn)
 tt=[]
 for _ in range(5000):
  t=time.perf_counter_ns();pred(tn);tt.append(time.perf_counter_ns()-t)
 lat.append({'architecture':a,'batch':1,'p50_us':np.percentile(tt,50)/1e3,'p95_us':np.percentile(tt,95)/1e3,'p99_us':np.percentile(tt,99)/1e3,'max_us':np.max(tt)/1e3})
latdf=pd.DataFrame(lat);latdf.to_csv(OUT/'inference_latency_python.csv',index=False)
# Native transfer aggregation.
nfiles=[OUT/f'native_{s}_10seeds.csv' for s in ['nominal','wind3','rotoreff30']]
native=pd.concat([pd.read_csv(p) for p in nfiles],ignore_index=True);native.to_csv(OUT/'native_source_transfer_trials_10seeds.csv',index=False)
nsum=native.groupby(['scenario','architecture']).agg(n=('seed','size'),rmse_mean=('rmse_m','mean'),rmse_sd=('rmse_m','std'),p95_mean=('p95_m','mean'),max_mean=('max_m','mean'),latent_max=('latent_max','max')).reset_index();nsum.to_csv(OUT/'native_source_transfer_summary_10seeds.csv',index=False)
ne=[]
for s in ['nominal','wind3','rotoreff30']:
 for prop in ['QCeNN-XX-503','QCeNN-XXXY-512']:
  a=native[(native.scenario==s)&(native.architecture=='LinearCeNN-494')].sort_values('seed');b=native[(native.scenario==s)&(native.architecture==prop)].sort_values('seed');d=a.rmse_m.values-b.rmse_m.values
  ne.append({'scenario':s,'proposal':prop,'comparator':'LinearCeNN-494','n':len(d),'improvement_m':float(d.mean()),'improvement_pct':float(100*d.mean()/a.rmse_m.mean()),'proposal_better_fraction':float(np.mean(d>0))})
ne=pd.DataFrame(ne);ne.to_csv(OUT/'native_source_transfer_effects.csv',index=False)
# Figures.
piv=summary.pivot(index='scenario',columns='architecture',values='rmse_mean').reindex(SCENARIOS)[ARCHS]
ax=piv.plot(kind='bar',figsize=(13,5.8));ax.set_yscale('log');ax.set_ylabel('Mean position RMSE [m]');ax.set_xlabel('Scenario');ax.set_title('Structured quadratic CeNN ablation — 200 paired seeds');ax.legend(title='',fontsize=8,ncol=2);plt.tight_layout();plt.savefig(OUT/'figQ1_rmse.png',dpi=220,bbox_inches='tight');plt.close()
eff=qbase.set_index('scenario').reindex(SCENARIOS);ax=eff.improvement_pct.plot(kind='bar',figsize=(10.5,4.8));ax.axhline(0,linewidth=1);ax.set_ylabel('QCeNN-XXXY improvement vs linear CeNN [%]');ax.set_xlabel('Scenario');ax.set_title('Paired RMSE effect of bounded XX+XY CeNN');plt.tight_layout();plt.savefig(OUT/'figQ2_relative_effect.png',dpi=220,bbox_inches='tight');plt.close()
ax=latdf.set_index('architecture').reindex(ARCHS)[['p50_us','p99_us']].plot(kind='bar',figsize=(10.5,4.8));ax.set_ylabel('Python/NumPy inference latency [µs], batch=1');ax.set_xlabel('Architecture');ax.set_title('Inference-cost diagnostic (not embedded/HIL)');plt.tight_layout();plt.savefig(OUT/'figQ3_latency.png',dpi=220,bbox_inches='tight');plt.close()
pp=nsum.pivot(index='scenario',columns='architecture',values='rmse_mean').reindex(['nominal','wind3','rotoreff30'])[['LinearCeNN-494','QCeNN-XX-503','QCeNN-XXXY-512']];ax=pp.plot(kind='bar',figsize=(9.8,4.8));ax.set_ylabel('Mean position RMSE [m]');ax.set_xlabel('Pinned RotorPy-source-core scenario');ax.set_title('Zero-shot transfer spot check — 10 paired seeds');ax.legend(title='',fontsize=8);plt.tight_layout();plt.savefig(OUT/'figQ4_native_transfer.png',dpi=220,bbox_inches='tight');plt.close()
# Concise key table.
key=summary[summary.architecture.isin(['LinearCeNN-494','QCeNN-XX-503','QCeNN-XXXY-512'])].pivot(index='scenario',columns='architecture',values='rmse_mean').reindex(SCENARIOS)
key['QXX_improvement_vs_linear_pct']=100*(key['LinearCeNN-494']-key['QCeNN-XX-503'])/key['LinearCeNN-494'];key['QXXXY_improvement_vs_linear_pct']=100*(key['LinearCeNN-494']-key['QCeNN-XXXY-512'])/key['LinearCeNN-494'];key.reset_index().to_csv(OUT/'key_effects.csv',index=False)
# Report.
train=pd.read_csv(OUT/'training_replicates.csv');sel=pd.read_csv(OUT/'validation_model_selection.csv')
report=f'''# Structured Quadratic CeNN Nonlinearity Ablation — FINAL REPORT

## Question

This experiment tests a single proposed change: replace the linear Chua–Yang-style CeNN state block in the frozen current-only AE–CeNN residual path by a bounded structured second-order block inspired by

`Xdot = -X + B Y + Bxx XX + Cxy XY + D U`.

The tested implementation deliberately bounds the nonlinear state factors: `xbar=tanh(X/2)` and `Y=tanh(X)`. A shared 3x3 local template acts on `xbar*xbar` (XX); the full variant adds a second 3x3 local template on `xbar*Y` (XY). This preserves local CeNN structure and avoids an unconstrained quadratic growth term that could dominate the `-X` leakage.

## Experimental discipline

- Same source-derived dynamics, trajectory, nominal controller, feature normalization, training corpus and ideal residual targets.
- Same 3→64→3 encoder for the three primary models.
- Same four CeNN relaxation steps, integration step 0.32, output tanh, residual limit and authority 0.35.
- Five initialization replicates per architecture.
- A documented pre-final protocol amendment selected the **median** model by held-out offline residual MSE on seeds 14000–14039; it did not choose the best replicate and occurred before any final seed was run.
- Final confirmation: 200 untouched paired seeds 16000–16199 in all eight scenarios.
- Secondary 515-parameter linear capacity control to check whether additional parameter count alone explains an effect.
- Zero-shot pinned RotorPy-source-core spot check: 10 untouched paired seeds 18000–18009 for nominal, native Dryden wind and rotor-efficiency. No retuning.

## Models

{train.groupby('architecture').active_inference_params.first().reset_index().to_markdown(index=False)}

## Frozen replicate selection

{sel.to_markdown(index=False)}

## Final source-derived mean results

{summary.round(7).to_markdown(index=False)}

## Key RMSE effects

{pd.read_csv(OUT/'key_effects.csv').round(6).to_markdown(index=False)}

Positive percentages mean the quadratic model has lower RMSE than the baseline linear CeNN.

## Paired statistics

{stats.round(8).to_markdown(index=False)}

## Pre-specified criteria

```json
{json.dumps(criteria,indent=2)}
```

**Overall confirmatory result: {'PASS' if criteria['primary_overall_pass'] else 'FAIL'}.**

The XX+XY model does **not** satisfy the pre-specified replacement criterion. Its average effect across the seven stressed source-derived regimes is {mean_base:+.4f}% relative to the linear CeNN. It improves nominal tracking by {max(0,nom):.3f}% and improves model-mismatch and latency modestly, but it is worse in wind, force-step, payload, rotor-efficiency and compound conditions. The additional XY term also has an average stressed effect of {mean_xx:+.4f}% relative to the XX-only model, so the proposed XX+XY extension is not supported as the preferred block.

The 515-parameter linear capacity control performs substantially worse in most stressed regimes despite a lower nominal RMSE. Therefore parameter count alone does not explain the primary baseline's robustness, but this does **not** rescue the quadratic hypothesis: the relevant comparison is QCeNN against the original linear CeNN, and that comparison fails the pre-specified criterion.

## Stability and boundedness diagnostics

All {len(final)} final controller–mission trials remained finite. The residual output bound was never violated. The maximum observed internal CeNN state magnitudes are reported in the main summary; no post-hoc stability threshold was introduced. The bounded nonlinear construction therefore avoided numerical blow-up in the tested domain, but numerical boundedness here is not a formal stability proof.

## Software latency diagnostic

{latdf.round(3).to_markdown(index=False)}

These are Python/NumPy batch-1 timings on the current CPU, not embedded or HIL latency measurements.

## Zero-shot pinned RotorPy-source-core spot check

{nsum.round(7).to_markdown(index=False)}

{ne.round(7).to_markdown(index=False)}

The zero-shot transfer differences are very small. XX+XY is slightly better under the 10-seed native-Dryden spot check, but slightly worse for nominal and rotor-efficiency. With N=10 and effects near zero, this is descriptive only and does not overturn the 200-seed confirmatory source-derived result.

## Scientific conclusion

The structured quadratic hypothesis is **plausible but not confirmed as a performance upgrade**. Adding bounded local XX and XY interactions changes the regime dependence of the residual controller: it can improve nominal/model/latency cases, yet it slightly degrades several multiplicative or compound stress cases that were hypothesized to benefit most. In other words, a more nonlinear CeNN is not automatically a better closed-loop residual controller, even when the nonlinearity is physically motivated and numerically bounded.

The most defensible manuscript decision is therefore **not to replace the existing linear CeNN block**. Preserve this experiment as a high-value negative ablation. It strengthens the paper's broader finding that additional representational expressivity must earn its place through closed-loop evidence; offline fit and structural plausibility are insufficient.

A future quadratic-CeNN study could consider physics-selective cross terms, explicit regularization/contraction constraints, or identification on native actuator-aware dynamics, but such retuning would constitute a new research stage and should not be introduced post-hoc into this confirmatory ablation.

## Claim boundary

This is source-derived simulation evidence plus a small pinned-RotorPy-source-core transfer spot check. It is not full package-native AdaptiveQuadBench/acados, HIL, or flight validation, and it does not support generic CeNN superiority.
'''
open(OUT/'QCeNN_ABLATION_FINAL_REPORT.md','w').write(report)
# manifest and bundle
manifest=[]
for p in sorted(OUT.iterdir()):
 if p.is_file() and p.suffix!='.zip':manifest.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
pd.DataFrame(manifest).to_csv(OUT/'MANIFEST_SHA256.csv',index=False)
bundle=OUT/'QCeNN_structured_nonlinearity_ablation_FINAL.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.iterdir()):
  if p.is_file() and p!=bundle:z.write(p,arcname=p.name)
 for extra in [Path('/mnt/data/qcenn_ablation/run_qcenn_ablation.py'),Path('/mnt/data/qcenn_ablation/qcenn_eval_core.py'),Path('/mnt/data/qcenn_ablation/eval_one.py'),Path('/mnt/data/qcenn_ablation/native_eval.py'),Path('/mnt/data/qcenn_ablation/aggregate_qcenn.py')]:z.write(extra,arcname='code/'+extra.name)
with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
sha=hashlib.sha256(bundle.read_bytes()).hexdigest();open(OUT/'bundle_sha256.txt','w').write(sha+'  '+bundle.name+'\n')
print(json.dumps(criteria,indent=2));print('\nKEY EFFECTS\n',pd.read_csv(OUT/'key_effects.csv').to_string(index=False));print('\nNATIVE\n',ne.to_string(index=False));print('\nBUNDLE',sha)

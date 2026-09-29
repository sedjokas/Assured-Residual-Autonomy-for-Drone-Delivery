import sys,os,argparse,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data/h_work')
from h_common import *
ap=argparse.ArgumentParser();ap.add_argument('family');args=ap.parse_args();fam=args.family
OUT='/mnt/data/situation_H_final';seeds=np.arange(5400,5500);rows=[];trH=None
for ctrl in ['Current','G','H']:
 rr,tr=simulate_H(fam,seeds,ctrl,collect_trace=(ctrl=='H'));rows+=rr
 if ctrl=='H':trH=tr
df=pd.DataFrame(rows,columns=['FaultFamily','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','saturation_pct','excursion_gt1m_pct','gate_mean','gate_open_pct','governor_scale_mean']);df.to_csv(f'{OUT}/fault_{fam}_trials.csv',index=False)
S=trH['scale']
if 'combined_faults' in fam:mask=(~trH['deadline_ok'])|trH['corrupt_mask']|trH['spike_mask']
elif 'deadline_bursts' in fam:mask=~trH['deadline_ok']
elif 'history_corruption' in fam:mask=trH['corrupt_mask']
elif 'sensor_spikes' in fam:mask=trH['spike_mask']
else:mask=np.zeros_like(S,dtype=bool)
rej=100*np.mean(S[mask]==0) if mask.any() else np.nan
pd.DataFrame([{'FaultFamily':fam,'Injected_fault_cycles':int(mask.sum()),'H_rejection_rate_pct':rej,'H_acceptance_rate_pct':100-rej if mask.any() else np.nan}]).to_csv(f'{OUT}/fault_{fam}_rejection.csv',index=False)
print(fam,'rejection',rej)
print(df.groupby('Controller')[['rmse_m','excursion_gt1m_pct']].mean())

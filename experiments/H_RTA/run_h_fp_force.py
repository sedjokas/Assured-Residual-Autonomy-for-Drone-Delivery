import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'/mnt/data/h_work')
from h_common import *
OUT='/mnt/data/situation_H_final';os.makedirs(OUT,exist_ok=True)
FP_SEEDS=np.arange(5000,5200);_,tr=simulate_H('nominal',FP_SEEDS,'H',collect_trace=True);v=tr['scale'][WARMUP:]
fp_cycle=100*np.mean(v>0);fp_episode=100*np.mean(np.any(v>0,axis=0));raw=100*np.mean(tr['alpha'][WARMUP:]>.5)
pd.DataFrame({'metric':['RTA false activation cycle rate pct','RTA episode any activation pct','raw detector alpha>0.5 cycle rate pct'],'value':[fp_cycle,fp_episode,raw]}).to_csv(f'{OUT}/nominal_false_positive_metrics.csv',index=False)
seeds=np.arange(5200,5300);rows=[];onset=300
for lev in [0.10,0.20,0.40,0.80]:
 _,tr=simulate_H('force_intensity',seeds,'H',intensity=lev,collect_trace=True);S=tr['scale'];A=tr['alpha'];l=[];la=[]
 for j in range(len(seeds)):
  ii=np.where(S[onset:onset+100,j]>0)[0];jj=np.where(A[onset:onset+100,j]>.5)[0];l.append(np.nan if len(ii)==0 else ii[0]*DT);la.append(np.nan if len(jj)==0 else jj[0]*DT)
 l=np.array(l);la=np.array(la);rows.append(['force',lev,100*np.mean(np.isfinite(l)),100*np.mean(l<=.2),np.nanmedian(l),100*np.mean(np.isfinite(la)),np.nanmedian(la)])
pd.DataFrame(rows,columns=['family','intensity','RTA_detection_rate_1s_pct','RTA_detection_within_200ms_pct','RTA_median_latency_s','raw_detector_rate_1s_pct','raw_detector_median_latency_s']).to_csv(f'{OUT}/detector_force_intensity.csv',index=False)
print('FP',fp_cycle,fp_episode,raw);print(pd.DataFrame(rows))

import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'/mnt/data/h_work')
from h_common import *
OUT='/mnt/data/situation_H_final';os.makedirs(OUT,exist_ok=True)
seeds=np.arange(5200,5300);rows=[];onset=300
for lev in [0.5,1.0,1.5,2.0,3.0,4.0]:
 _,tr=simulate_H('wind_onset',seeds,'H',intensity=lev,collect_trace=True);S=tr['scale'];A=tr['alpha'];l=[];la=[]
 for j in range(len(seeds)):
  ii=np.where(S[onset:onset+100,j]>0)[0];jj=np.where(A[onset:onset+100,j]>.5)[0];l.append(np.nan if len(ii)==0 else ii[0]*DT);la.append(np.nan if len(jj)==0 else jj[0]*DT)
 l=np.array(l);la=np.array(la);rows.append(['wind',lev,100*np.mean(np.isfinite(l)),100*np.mean(l<=.2),np.nanmedian(l),100*np.mean(np.isfinite(la)),np.nanmedian(la)])
df=pd.DataFrame(rows,columns=['family','intensity','RTA_detection_rate_1s_pct','RTA_detection_within_200ms_pct','RTA_median_latency_s','raw_detector_rate_1s_pct','raw_detector_median_latency_s']);df.to_csv(f'{OUT}/detector_wind_intensity.csv',index=False);print(df)

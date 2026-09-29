import sys,os,json,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data/k_work');import k_common as K
CFG=dict(pmax=np.array([.75,.75,.60]),vmax=np.array([2.2,2.2,1.7]),dmax=np.array([1.6,1.6,1.6]),lam1=3.,lam2=3.,lamv=4.,cmd_lim=5.5)
COLS=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','excursion_gt0p5_pct','excursion_gt1m_pct','saturation_pct','filter_active_pct','filter_infeasible_pct','hocbf_set_invalid_pct','filter_correction_mean','I_authority_scale_mean']
out='/mnt/data/situation_K_final';os.makedirs(out,exist_ok=True)
kind=sys.argv[1];start=int(sys.argv[2]);n=int(sys.argv[3]);mode=sys.argv[4]
seeds=np.arange(start,start+n);controllers=['Current','I','J','K'] if mode=='main' else ['Nominal','Current','I','J','K'];rows=[];dist=[]
for c in controllers:
    rr,tr=K.simulate_K(kind,seeds,c,CFG,collect_trace=(c=='K'));rows.extend(rr)
    if c=='K' and tr is not None:
        d=np.abs(tr['model_disturbance'][K.H.WARMUP:])
        dist.append(dict(Scenario=kind,Controller='K-RobustHOCBF-Filter',d_exceed_pct=100*np.mean(d>CFG['dmax']),d_p95_x=np.quantile(d[:,:,0],.95),d_p95_y=np.quantile(d[:,:,1],.95),d_p95_z=np.quantile(d[:,:,2],.95),d_max_x=d[:,:,0].max(),d_max_y=d[:,:,1].max(),d_max_z=d[:,:,2].max()))
pd.DataFrame(rows,columns=COLS).to_csv(f'{out}/{mode}_{kind}.csv',index=False)
if dist:pd.DataFrame(dist).to_csv(f'{out}/{mode}_{kind}_disturbance_diagnostic.csv',index=False)
print(kind,mode,'done')

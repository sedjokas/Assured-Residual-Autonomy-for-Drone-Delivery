import sys,numpy as np,pandas as pd,json
sys.path.insert(0,'/mnt/data/i_work');import j_common as J
kind=sys.argv[1];start=int(sys.argv[2]);n=int(sys.argv[3]);mode=sys.argv[4]
cfg={'h':.30,'epos':np.array([.55,.55,.45]),'evel':np.array([1.5,1.5,1.1]),'cmd_lim':5.5};seeds=np.arange(start,start+n)
cols=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','excursion_gt0p5_pct','excursion_gt1m_pct','saturation_pct','projection_active_pct','projection_infeasible_pct','projection_correction_mean','I_authority_scale_mean']
controllers=['Current','I','J'] if mode=='main' else ['Nominal','Current','I','J']
rows=[]
for c in controllers:
 r,_=J.simulate_J(kind,seeds,c,cfg);rows.extend(r)
pd.DataFrame(rows,columns=cols).to_csv(f'/mnt/data/situation_J_final/{mode}_{kind}.csv',index=False)
print(kind,mode,'done')

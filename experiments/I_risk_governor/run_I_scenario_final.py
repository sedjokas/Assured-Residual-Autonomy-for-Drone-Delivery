import sys,os,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data/i_work');import i_common as I
sc=sys.argv[1];start=int(sys.argv[2]) if len(sys.argv)>2 else 7400;seeds=np.arange(start,start+100);cols=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','saturation_pct','excursion_gt1m_pct','authority_scale_mean','authority_active_pct','detector_conf_mean','ood_conf_mean','deadline_conf_mean','actuator_conf_mean','dynamic_conf_mean'];rows=[]
for c in ['Current','G']:
 r,_=I.simulate_I(sc,seeds,c);rows.extend(r)
r,_=I.simulate_H_custom(sc,seeds);rows.extend(r);r,_=I.simulate_I(sc,seeds,'I');rows.extend(r)
pd.DataFrame(rows,columns=cols).to_csv(f'/mnt/data/situation_I_final/final_main_{sc}.csv',index=False);print(sc,'done')

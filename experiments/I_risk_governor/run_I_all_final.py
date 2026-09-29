import sys,os,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data/i_work');import i_common as I
scens=['nominal','wind3','force_step','model20','latency40','payload50','rotoreff30','compound'];seeds=np.arange(8000,8100);cols=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','saturation_pct','excursion_gt1m_pct','authority_scale_mean','authority_active_pct','detector_conf_mean','ood_conf_mean','deadline_conf_mean','actuator_conf_mean','dynamic_conf_mean'];rows=[]
for sc in scens:
 print(sc,flush=True)
 for c in ['Current','G']:
  r,_=I.simulate_I(sc,seeds,c);rows.extend(r)
 r,_=I.simulate_H_custom(sc,seeds);rows.extend(r);r,_=I.simulate_I(sc,seeds,'I');rows.extend(r)
pd.DataFrame(rows,columns=cols).to_csv('/mnt/data/situation_I_final/FINAL_main_trials_100seeds.csv',index=False)

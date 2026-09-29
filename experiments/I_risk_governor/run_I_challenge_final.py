import sys,os,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data/i_work');import i_common as I
kind=sys.argv[1];start=int(sys.argv[2]) if len(sys.argv)>2 else 7600;H=I.H;seeds=np.arange(start,start+100);cols=['Scenario','Controller','SeedIndex','rmse_m','p95_error_m','max_error_m','saturation_pct','excursion_gt1m_pct','authority_scale_mean','authority_active_pct','detector_conf_mean','ood_conf_mean','deadline_conf_mean','actuator_conf_mean','dynamic_conf_mean'];rows=[];summ=[]
for ctrl in ['Current','G','I']:
 r,t=I.simulate_I(kind,seeds,ctrl,collect_trace=(ctrl=='I'));rows.extend(r)
 if ctrl=='I':
  if 'soft_deadline' in kind:mask=((np.arange(H.STEPS)>=380)&(np.arange(H.STEPS)<450))|((np.arange(H.STEPS)>=780)&(np.arange(H.STEPS)<850))
  elif 'soft_history' in kind:mask=t['softood'].any(1)
  elif 'deadline_bursts' in kind:mask=t['deadline_margin'][:,0]<=0
  elif 'history_corruption' in kind:mask=t['corrupt'].any(1)
  else:mask=np.arange(H.STEPS)>=H.WARMUP
  summ.append([kind,'I',t['scale'][mask].mean(),t['ood_conf'][mask].mean(),t['deadline_conf'][mask].mean(),t['act_conf'][mask].mean(),t['dyn_conf'][mask].mean(),100*(~t['hard_valid'][mask]).mean()])
r,th=I.simulate_H_custom(kind,seeds,collect_trace=True);rows.extend(r)
if 'soft_deadline' in kind:mask=((np.arange(H.STEPS)>=380)&(np.arange(H.STEPS)<450))|((np.arange(H.STEPS)>=780)&(np.arange(H.STEPS)<850))
elif 'soft_history' in kind:mask=th['softood'].any(1)
elif 'deadline_bursts' in kind:mask=th['deadline_margin'][:,0]<=0
elif 'history_corruption' in kind:mask=th['corrupt'].any(1)
else:mask=np.arange(H.STEPS)>=H.WARMUP
summ.append([kind,'H',th['scale'][mask].mean(),np.nan,np.nan,np.nan,np.nan,np.nan])
pd.DataFrame(rows,columns=cols).to_csv(f'/mnt/data/situation_I_final/final_challenge_{kind}.csv',index=False);pd.DataFrame(summ,columns=['Challenge','Controller','target_window_authority_mean','OOD_conf_mean','deadline_conf_mean','actuator_conf_mean','dynamic_conf_mean','hard_veto_pct']).to_csv(f'/mnt/data/situation_I_final/final_challenge_{kind}_trace.csv',index=False);print(kind,'done',summ)

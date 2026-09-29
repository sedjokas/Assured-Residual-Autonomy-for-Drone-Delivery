import sys,os,argparse,pandas as pd
sys.path.insert(0,'/mnt/data');import native_validation_l_core as L
ap=argparse.ArgumentParser();ap.add_argument('scenario');a=ap.parse_args();rows=[]
for s in range(5):
 for m in ['Current','I','K']:
  rows.append(L.run_transfer(a.scenario,7000+s,m,steps=600,fast=True))
out='/mnt/data/native_validation_L';os.makedirs(out,exist_ok=True);pd.DataFrame(rows).to_csv(f'{out}/{a.scenario}_pilot.csv',index=False)
print(pd.DataFrame(rows).groupby('mode').rmse_m.agg(['mean','std']))

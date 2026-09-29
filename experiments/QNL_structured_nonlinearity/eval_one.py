import sys,time,numpy as np
from qcenn_eval_core import simulate,OUT
arch=sys.argv[1]; scn=sys.argv[2]; n=int(sys.argv[3]) if len(sys.argv)>3 else 100
seeds=np.arange(16000,16000+n); t=time.time(); d=simulate(scn,seeds,arch); p=OUT/f'final_{arch}_{scn}_{n}seeds.csv'; d.to_csv(p,index=False); print(p, 'elapsed',time.time()-t, 'rmse',d.rmse_m.mean())


import os, sys, pandas as pd
sys.path.insert(0, "/mnt/data/native_validation_L2_py")
from l2_core import run_trial

out="/mnt/data/native_validation_L2_py"
rows=[]
for seed in range(12000,12010):
    for ctrl in ("I","K2"):
        rows.append(run_trial("rotoreff30", seed, ctrl))
pd.DataFrame(rows).to_csv(f"{out}/test_rotoreff30.csv", index=False)
print("completed test_rotoreff30", flush=True)

for fault in ("proposal_impulse","proposal_stuck"):
    rows=[]
    for seed in range(13000,13010):
        for ctrl in ("I","K2"):
            rows.append(run_trial("wind3", seed, ctrl, fault_kind=fault))
    pd.DataFrame(rows).to_csv(f"{out}/challenge_wind3_{fault}.csv", index=False)
    print("completed", fault, flush=True)

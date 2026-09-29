#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "tables"
PROC = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

rows = []

# Situation A
a = pd.read_csv(RAW/"A_synthetic_micro"/"cenn_feasibility_summary.csv")
for c in ["Nominal","Compact MLP + RTA","CeNN + RTA","CeNN dropout -> nominal + RTA"]:
    r=a[a["Controller"]==c].iloc[0]
    rows.append(["A","synthetic",c,"paired missions",r["RMSE mean [m]"],r["Corridor violation [% missions]"],np.nan,np.nan])

# Safe-Control-Gym-derived hard compound comparison.
# The final manuscript uses the same N=20 compound seeds for every comparator
# because constrained LMPC-H40 was archived only on that matched subset.
public = pd.read_csv(RAW/"SCG_safe_control_gym_derived"/"public_benchmark_trials.csv")
public = public[(public["Scenario"]=="compound") & (public["SeedIndex"]<20)]
for c in ["PD","LQR","PD+MLP+RTA","PD+CeNN+RTA"]:
    q=public[public["Controller"]==c]
    rows.append(["SCG","Safe-Control-Gym-derived",c,"compound matched N=20",q["pos_rmse_m"].mean(),0.0,np.nan,np.nan])
mpc = pd.read_csv(RAW/"SCG_safe_control_gym_derived"/"constrained_lmpc_compound_trials_20.csv")
rows.append(["SCG","Safe-Control-Gym-derived","Constrained LMPC-H40 (OSQP)","compound matched N=20",mpc["pos_rmse_m"].mean(),0.0,np.nan,np.nan])

# H/I
for exp, folder, fname, sc, ctrls in [
    ("H","H_RTA","main_summary_100seeds.csv","compound",["CurrentOnly-AE-CeNN-494","G-DetectorGate","H-RTA-SupervisedGate"]),
    ("I","I_risk_governor","main_summary_100seeds.csv","compound",["CurrentOnly-AE-CeNN-494","H-RTA-SupervisedGate","I-ContinuousRiskGovernor"]),
]:
    df=pd.read_csv(RAW/folder/fname)
    for c in ctrls:
        r=df[(df["Scenario"]==sc)&(df["Controller"]==c)].iloc[0]
        rows.append([exp,"AdaptiveQuadBench-derived",c,sc,r["rmse_mean"],r.get("excursion_mean",np.nan),r.get("gate_open_pct",r.get("authority_active_pct",np.nan)),np.nan])

# J/K challenges
for exp, folder, fname, sc, ctrls in [
    ("J","J_projection","challenge_summary_100seeds.csv","wind3_proposal_impulse",["I-RiskGovernor-unprojected","J-IndependentSafetyProjection"]),
    ("K","K_HOCBF","challenge_summary_100seeds.csv","wind3_proposal_impulse",["I-RiskGovernor-unfiltered","K-RobustHOCBF-Filter"]),
]:
    df=pd.read_csv(RAW/folder/fname)
    for c in ctrls:
        r=df[(df["Scenario"]==sc)&(df["Controller"]==c)].iloc[0]
        rows.append([exp,"AdaptiveQuadBench-derived",c,sc,r["rmse_mean"],r.get("excursion_gt1m_mean_pct",r.get("exc_gt1m_pct",np.nan)),r.get("filter_active_mean_pct",r.get("projection_active_pct",np.nan)),r.get("infeasible_mean_pct",r.get("projection_infeasible_pct",np.nan))])


# QNL structured nonlinearity ablation (negative replacement result)
q = pd.read_csv(RAW/"QNL_structured_nonlinearity"/"final_summary_200seeds.csv")
for sc in ["nominal","compound"]:
    for c in ["LinearCeNN-494","QCeNN-XXXY-512"]:
        r=q[(q["scenario"]==sc)&(q["architecture"]==c)].iloc[0]
        rows.append(["QNL","source-derived",c,sc,r["rmse_mean"],r["excursion_gt1m_mean_pct"],np.nan,np.nan])

# L2 transfer
df=pd.read_csv(RAW/"L2_native_recalibration"/"independent_test_summary.csv")
for c in ["I","K2"]:
    r=df[(df["scenario"]=="wind3")&(df["controller"]==c)].iloc[0]
    rows.append(["L2","RotorPy pinned-source",c,"wind3",r["rmse_mean"],r["excursion_gt1m_mean_pct"],r["filter_active_mean_pct"],r["filter_infeasible_mean_pct"]])

out=pd.DataFrame(rows,columns=["experiment","provenance","controller","scenario","rmse_m","violation_or_gt1m_pct","filter_or_projection_active_pct","infeasible_pct"])
out.to_csv(OUT/"PAPER_KEY_RESULTS.csv",index=False)
out.to_csv(PROC/"PAPER_KEY_RESULTS.csv",index=False)

# Compact criteria snapshot.
criteria={}
for name,path in [
    ("H", RAW/"H_RTA"/"H_completion_criteria.json"),
    ("I", RAW/"I_risk_governor"/"I_completion_criteria.json"),
    ("J", RAW/"J_projection"/"J_completion_criteria.json"),
    ("K", RAW/"K_HOCBF"/"K_completion_criteria.json"),
    ("L2", RAW/"L2_native_recalibration"/"L2_final_assessment.json"),
    ("QNL", RAW/"QNL_structured_nonlinearity"/"completion_criteria.json"),
]:
    criteria[name]=json.load(open(path))
json.dump(criteria,open(PROC/"criteria_snapshot.json","w"),indent=2)

print(out.to_string(index=False))

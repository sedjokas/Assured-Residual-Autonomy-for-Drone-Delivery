#!/usr/bin/env python3
"""Regenerate the E1-E3 numbers reported in the manuscript (Tables IX-XII).

Reads only the frozen outputs under experiments/E1_*, E2_*, E3_* and checks
that they reproduce the values printed in the paper. It does not rerun the
simulations (see experiments/<E*>/ and REPRODUCIBILITY.md for that).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

rows = []

# E1 - severe stuck-proposal challenge (Table IX)
e1 = pd.read_csv(EXP / "E1_actuator_aware" / "results" / "final_summary.csv")
stuck = e1[e1.fault == "stuck"]
for scen, label in [("dryden3", "Severe Dryden"), ("compound", "Compound")]:
    for flt in ["P", "LM", "AA"]:
        r = stuck[(stuck.scenario == scen) & (stuck["filter"] == flt)].iloc[0]
        rows.append(["E1", label, flt, "rmse_m", r.rmse])
        rows.append(["E1", label, flt, "violation_pct", 100 * r.violation])
        rows.append(["E1", label, flt, "infeasible_pct", 100 * r.infeasible])
        rows.append(["E1", label, flt, "gt1m_pct", 100 * r.gt1])

# E2 - pooled results on 48,000 untouched missions (Table X) and coverage (Table XI)
e2 = pd.read_csv(EXP / "E2_energy_risk" / "final_summary.csv")
pooled = e2[e2.scenario == "POOLED"].set_index("policy")
for pol in ["GEO", "DET", "RISK", "ADAPT"]:
    r = pooled.loc[pol]
    for m in ["admission_rate", "unsafe_admission_rate", "reserve_violation_rate_admitted",
              "false_rejection_rate", "delivery_success_rate_all", "diversion_rate_admitted"]:
        rows.append(["E2", "pooled", pol, m + "_pct", 100 * r[m]])
cov = pd.read_csv(EXP / "E2_energy_risk" / "final_coverage.csv")
for _, r in cov.iterrows():
    rows.append(["E2", r.scenario, "RISK", "coverage_pct", 100 * r.coverage])

# E3 - equal-deadline optimality gap and raw feasibility (Table XII)
e3 = pd.read_csv(EXP / "E3_equal_budget_qi" / "scale_summary.csv")
for _, r in e3.iterrows():
    rows.append(["E3", r.scale, r.solver, "mean_gap_pct", r.mean_gap_pct])
    rows.append(["E3", r.scale, r.solver, "raw_feasible_pct", 100 * r.raw_feasible_rate])
e3p = pd.read_csv(EXP / "E3_equal_budget_qi" / "primary_paired_statistics.csv")
for _, r in e3p.iterrows():
    rows.append(["E3", r.scale, "GRASP-LS minus SB-QI", "mean_diff_pp", r.mean_improvement_pp_GRASP_minus_SB])
    rows.append(["E3", r.scale, "GRASP-LS minus SB-QI", "bootstrap95_low", r.bootstrap95_low])
    rows.append(["E3", r.scale, "GRASP-LS minus SB-QI", "bootstrap95_high", r.bootstrap95_high])

out = pd.DataFrame(rows, columns=["experiment", "scenario_or_scale", "arm", "metric", "value"])
out.to_csv(OUT / "E1_E3_KEY_RESULTS.csv", index=False)

# Values printed in the manuscript (rounded as in the paper).
def v(exp, scen, arm, metric):
    q = out[(out.experiment == exp) & (out.scenario_or_scale == scen) & (out.arm == arm) & (out.metric == metric)]
    return float(q.value.iloc[0])

expected = [
    (("E1", "Severe Dryden", "P", "rmse_m"), 1.4331, 4),
    (("E1", "Severe Dryden", "LM", "rmse_m"), 0.5264, 4),
    (("E1", "Severe Dryden", "AA", "rmse_m"), 0.5002, 4),
    (("E1", "Severe Dryden", "AA", "violation_pct"), 1.17, 2),
    (("E1", "Compound", "AA", "rmse_m"), 0.7376, 4),
    (("E2", "pooled", "DET", "reserve_violation_rate_admitted_pct"), 2.11, 2),
    (("E2", "pooled", "RISK", "reserve_violation_rate_admitted_pct"), 0.112, 3),
    (("E2", "pooled", "ADAPT", "reserve_violation_rate_admitted_pct"), 0.0, 3),
    (("E2", "pooled", "RISK", "false_rejection_rate_pct"), 14.43, 2),
    (("E2", "pooled", "ADAPT", "diversion_rate_admitted_pct"), 1.34, 2),
    (("E2", "POOLED", "RISK", "coverage_pct"), 97.46, 2),
    (("E2", "headwind", "RISK", "coverage_pct"), 92.62, 2),
    (("E2", "compound", "RISK", "coverage_pct"), 92.97, 2),
    (("E3", "small", "SB-QI", "mean_gap_pct"), 1.052, 3),
    (("E3", "medium", "SB-QI", "mean_gap_pct"), 2.614, 3),
    (("E3", "large", "SB-QI", "mean_gap_pct"), 4.365, 3),
    (("E3", "large", "GRASP-LS", "mean_gap_pct"), 0.817, 3),
    (("E3", "small", "SB-QI", "raw_feasible_pct"), 2.1, 1),
    (("E3", "medium", "GRASP-LS minus SB-QI", "mean_diff_pp"), -2.436, 3),
    (("E3", "large", "GRASP-LS minus SB-QI", "mean_diff_pp"), -3.548, 3),
]
bad = []
for key, paper, nd in expected:
    got = round(v(*key), nd)
    if abs(got - paper) > 10 ** (-nd) / 2 + 1e-12:
        bad.append((key, paper, got))
print(out.to_string(index=False))
if bad:
    print("\nMISMATCH with manuscript values:")
    for b in bad:
        print("  ", b)
    raise SystemExit(1)
print(f"\nOK - {len(expected)} manuscript values for E1-E3 reproduced from frozen outputs.")

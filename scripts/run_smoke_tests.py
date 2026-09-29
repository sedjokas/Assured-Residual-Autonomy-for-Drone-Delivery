#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, json
ROOT=Path(__file__).resolve().parents[1]
assert len(pd.read_csv(ROOT/"EXPERIMENT_REGISTRY.csv")) >= 16
assert (ROOT/"data/raw/A_synthetic_micro/cenn_feasibility_trials.csv").exists()
assert json.load(open(ROOT/"data/raw/L2_native_recalibration/L2_final_assessment.json"))["simple_scalar_bound_recalibration_accepted"] is False
assert json.load(open(ROOT/"data/raw/QNL_structured_nonlinearity/completion_criteria.json"))["primary_overall_pass"] is False
assert len(pd.read_csv(ROOT/"EXPERIMENT_REGISTRY.csv")) >= 19
assert json.load(open(ROOT/"experiments/E3_equal_budget_qi/H2_DECISION.json"))["H2_supported"] is False
for f in ["experiments/E1_actuator_aware/results/final_summary.csv",
          "experiments/E2_energy_risk/final_summary.csv",
          "experiments/E3_equal_budget_qi/scale_summary.csv"]:
    assert (ROOT/f).exists(), f
print("smoke tests passed")

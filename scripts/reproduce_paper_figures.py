#!/usr/bin/env python3
"""Regenerate the manuscript-facing figures from archived CSV/JSON evidence.

This script intentionally uses only archived results. It does not rerun the
simulators. The simulator runners are under experiments/ and have different
third-party environment requirements.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# The architecture and experimental-logic diagrams are the frozen manuscript
# figures in paper/figures/ (Fig. 1-3); only data-driven figures are regenerated here.

# 3. SCG strong-baseline comparison on the matched 20-seed subset.
public = pd.read_csv(RAW/"SCG_safe_control_gym_derived"/"public_benchmark_trials.csv")
public = public[(public["Scenario"]=="compound") & (public["SeedIndex"]<20)]
mpc = pd.read_csv(RAW/"SCG_safe_control_gym_derived"/"constrained_lmpc_compound_trials_20.csv")
labels = ["PD","LQR","PD + MLP","PD + CeNN","Constrained\nLMPC-H40"]
series = [
    public[public.Controller=="PD"].sort_values("SeedIndex").pos_rmse_m.to_numpy(),
    public[public.Controller=="LQR"].sort_values("SeedIndex").pos_rmse_m.to_numpy(),
    public[public.Controller=="PD+MLP+RTA"].sort_values("SeedIndex").pos_rmse_m.to_numpy(),
    public[public.Controller=="PD+CeNN+RTA"].sort_values("SeedIndex").pos_rmse_m.to_numpy(),
    mpc.sort_values("SeedIndex").pos_rmse_m.to_numpy(),
]
fig, ax = plt.subplots(figsize=(9,5.3))
ax.boxplot(series, tick_labels=labels, showmeans=True)
ax.set_ylabel("Position RMSE [m]")
ax.set_title("Hard compound case: matched strong-baseline subset (N=20)")
plt.tight_layout(); plt.savefig(OUT/"figSCG_strong_mpc.png",dpi=220,bbox_inches="tight"); plt.close()

# 4. D2 capacity-matched history ablation.
d2 = pd.read_csv(RAW/"D2_confirmatory"/"main_summary_200seeds.csv")
controllers = ["Geo","LQR-outer","LMPC-H40-surrogate","C: Geo+AE-CeNN","D2-History-AE-CeNN","History-MLP","ActiveCapacityMatched-noHistory-AE-CeNN"]
# Some archived versions used CapacityMatched-AE-CeNN as the final controller label.
if "ActiveCapacityMatched-noHistory-AE-CeNN" not in set(d2.Controller):
    controllers[-1] = "CapacityMatched-AE-CeNN"
scenarios = ["nominal","wind3","force_step","model20","latency40","payload50","rotoreff30","compound"]
p = d2[d2.Controller.isin(controllers)].pivot(index="Scenario",columns="Controller",values="rmse_mean").reindex(scenarios)
ax = p.plot(kind="bar",figsize=(12.5,5.4),logy=True)
ax.set_ylabel("Mean position RMSE [m]"); ax.set_xlabel("Scenario")
ax.set_title("D.2 confirmatory: exact-history architecture vs active-capacity-matched no-history control")
ax.legend(title="",fontsize=7,ncol=2)
plt.tight_layout(); plt.savefig(OUT/"figD2_capacity_vs_history.png",dpi=220,bbox_inches="tight"); plt.close()

# 5. G gate separation.
g = pd.read_csv(RAW/"G_event_gate"/"main_summary_100seeds.csv")
g = g[g.Controller=="EventDetected-GatedHistory"].set_index("Scenario").reindex(scenarios)
fig, ax = plt.subplots(figsize=(8.6,4.6))
ax.bar(g.index,g.gate_mean)
ax.set_ylabel("Mean effective gate"); ax.set_xlabel("Scenario")
ax.set_title("Disturbance-conditioned use of the history branch")
ax.tick_params(axis="x",rotation=38)
plt.tight_layout(); plt.savefig(OUT/"figG_gate_separation.png",dpi=220,bbox_inches="tight"); plt.close()

# 6. H detector envelope.
h = pd.read_csv(RAW/"H_RTA"/"detector_intensity_curve.csv")
fig, ax = plt.subplots(figsize=(8.6,4.7))
for fam in ["force","wind"]:
    q=h[h.family==fam]
    ax.plot(q.intensity,q.RTA_detection_within_200ms_pct,marker="o",label=fam)
ax.set_xlabel("Perturbation intensity (N for force; m/s for wind)")
ax.set_ylabel("RTA detection within 200 ms [%]")
ax.set_ylim(-2,102); ax.set_title("Situation H - sensitivity of causal detection"); ax.legend()
plt.tight_layout(); plt.savefig(OUT/"figH_detection.png",dpi=220,bbox_inches="tight"); plt.close()

# 7. I continuous authority vs binary H.
i = pd.read_csv(RAW/"I_risk_governor"/"challenge_trace_summary.csv")
order = ["deadline_bursts","history_corruption","severe_compound","soft_deadline_erosion","soft_history_shift"]
# severe_compound is not a trace row in every archive; use available rows while preserving the manuscript-relevant soft/hard challenges.
order = [x for x in order if x in set(i.Challenge)]
p = i.pivot(index="Challenge",columns="Controller",values="target_window_authority_mean").reindex(order)
ax = p.plot(kind="bar",figsize=(9.2,4.8))
ax.set_ylabel("Mean history authority in target window"); ax.set_xlabel("Challenge")
ax.set_title("Situation I - continuous attenuation versus binary H")
ax.legend(title="Controller"); ax.tick_params(axis="x",rotation=90)
plt.tight_layout(); plt.savefig(OUT/"figI_authority.png",dpi=220,bbox_inches="tight"); plt.close()

# 8. K source-derived fault containment.
k = pd.read_csv(RAW/"K_HOCBF"/"challenge_summary_100seeds.csv")
sel = ["I-RiskGovernor-unfiltered","J-PredictiveProjection","K-RobustHOCBF-Filter"]
kp = k[k.Controller.isin(sel)].pivot(index="Scenario",columns="Controller",values="rmse_mean")
ax = kp.plot(kind="bar",figsize=(9.4,4.8))
ax.set_ylabel("Mean position RMSE [m]"); ax.set_xlabel("Post-governor proposal fault")
ax.set_title("Situation K - containment of corrupted proposals")
ax.legend(title="",fontsize=8)
plt.tight_layout(); plt.savefig(OUT/"figK_containment.png",dpi=220,bbox_inches="tight"); plt.close()

# 9. L pinned-source transfer pilot.
l = pd.read_csv(RAW/"L_native_source_pilot"/"native_source_IK_summary_pilot.csv")
lp = l.pivot(index="scenario",columns="mode",values="rmse_mean")
# order if present
lo=[x for x in scenarios if x in lp.index]
lp=lp.reindex(lo)
ax=lp.plot(kind="bar",figsize=(10.5,5.0))
ax.set_ylabel("Mean position RMSE [m]"); ax.set_xlabel("Pinned RotorPy-source pilot scenario")
ax.set_title("Situation L - frozen I/K transfer pilot (N=5/scenario)")
ax.legend(title="")
plt.tight_layout(); plt.savefig(OUT/"figL_native_transfer.png",dpi=220,bbox_inches="tight"); plt.close()

# 10. L2 assumption failure.
l2 = pd.read_csv(RAW/"L2_native_recalibration"/"independent_test_summary.csv")
l2f = pd.read_csv(RAW/"L2_native_recalibration"/"fault_challenge_summary.csv")
q = pd.concat([
    l2[l2.controller=="K2"][["scenario","filter_infeasible_mean_pct","d_bound_exceed_mean_pct"]],
    l2f[l2f.controller=="K2"][["scenario","filter_infeasible_mean_pct","d_bound_exceed_mean_pct"]],
],ignore_index=True).set_index("scenario")
ax=q.plot(kind="bar",figsize=(10.6,4.9))
ax.set_ylabel("Fraction of evaluated cycles [%]"); ax.set_xlabel("Scenario")
ax.set_title("L.2 - mismatch-bound exceedance and HOCBF infeasibility")
ax.legend(title="")
plt.tight_layout(); plt.savefig(OUT/"figL2_assumption_failure.png",dpi=220,bbox_inches="tight"); plt.close()


# 11. QNL bounded second-order CeNN effect.
qnl = pd.read_csv(RAW/"QNL_structured_nonlinearity"/"key_effects.csv").set_index("scenario")
fig, ax = plt.subplots(figsize=(9.8,4.8))
ax.bar(qnl.index, qnl["QXXXY_improvement_vs_linear_pct"])
ax.axhline(0, linewidth=1)
ax.set_ylabel("QCeNN-XXXY improvement vs LinearCeNN-494 [%]")
ax.set_xlabel("Scenario")
ax.set_title("Structured quadratic CeNN ablation: regime-dependent effect")
ax.tick_params(axis="x",rotation=35)
plt.tight_layout(); plt.savefig(OUT/"figQNL_relative_effect.png",dpi=220,bbox_inches="tight"); plt.close()

print(f"Generated {len(list(OUT.glob('fig*.png')))} figure files in {OUT}")

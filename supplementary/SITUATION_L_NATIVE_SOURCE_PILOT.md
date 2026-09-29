# SITUATION L — Native-Source RotorPy/AdaptiveQuadBench Transfer Pilot

## Status and claim boundary

This is a **native-source core pilot**, not yet a full package-native `run_eval.py` execution. The execution environment has no DNS/network access and lacks `gymnasium`, `cvxopt`, `acados_template`, and the compiled acados library. Therefore a truthful full AdaptiveQuadBench + RotorPy + acados run cannot be claimed here.

What was executed is pinned to:
- AdaptiveQuadBench commit `d4c273861aa0ce6750818af0b1b63a2a40408e52`;
- its RotorPy submodule commit `07e6e2c57d55fc563ac55f9c44153c2769f0ae73`.

The pilot directly uses the pinned AdaptiveQuadBench vehicle parameters, RotorPy multirotor ODE/body-wrench/motor-lag equations, RotorPy `cmd_acc` low-level attitude/thrust mapping, and RotorPy Dryden equations. Frozen Situation-I learned/risk models and the Situation-K HOCBF filter were transferred without retraining.

The campaign uses a 10-ms fixed-step RK4 integration over the pinned RotorPy ODE to keep the experiment tractable. Three independent nominal seeds were also rerun with RotorPy's `solve_ivp` stepping logic: the mean RMSE difference between RK4 and solve_ivp was **0.653%** (max 1.269%).

## Transfer pilot

Five paired seeds per scenario, 6-s missions at 100 Hz. This is intentionally a pilot, not a confirmatory Monte-Carlo campaign.

| scenario   | mode    |   n |   rmse_mean |   rmse_sd |   p95_mean |   max_mean |   K_active_pct |   K_infeasible_pct |   K_set_invalid_pct |   plant_step_p50_us |   plant_step_p99_us |
|:-----------|:--------|----:|------------:|----------:|-----------:|-----------:|---------------:|-------------------:|--------------------:|--------------------:|--------------------:|
| force_step | Current |   5 |    0.147329 |  0.045602 |   0.22731  |   0.231923 |         0      |              0     |               0     |             1248.68 |             1935.96 |
| force_step | I       |   5 |    0.147001 |  0.04568  |   0.227067 |   0.231672 |         0      |              0     |               0     |             1220.49 |             2915.65 |
| force_step | K       |   5 |    0.147001 |  0.04568  |   0.227067 |   0.231672 |         0      |              0     |               0     |             1205.64 |             1993.09 |
| latency40  | Current |   5 |    0.105701 |  0.001464 |   0.134688 |   0.135443 |         0      |              0     |               0     |             1222.82 |             1725.59 |
| latency40  | I       |   5 |    0.105379 |  0.001461 |   0.134458 |   0.135213 |         0      |              0     |               0     |             1212.86 |             1878.1  |
| latency40  | K       |   5 |    0.105379 |  0.001461 |   0.134458 |   0.135213 |         0      |              0     |               0     |             1192.32 |             1717.67 |
| model20    | Current |   5 |    0.111293 |  0.028598 |   0.142814 |   0.143334 |         0      |              0     |               0     |             1214.09 |             1555.92 |
| model20    | I       |   5 |    0.110963 |  0.028607 |   0.142555 |   0.143073 |         0      |              0     |               0     |             1210.2  |             1734.75 |
| model20    | K       |   5 |    0.110963 |  0.028607 |   0.142555 |   0.143073 |         0      |              0     |               0     |             1188.14 |             1733.43 |
| nominal    | Current |   5 |    0.108208 |  0.001334 |   0.138437 |   0.139161 |         0      |              0     |               0     |             1237.35 |             3350.48 |
| nominal    | I       |   5 |    0.107872 |  0.001329 |   0.138208 |   0.138932 |         0      |              0     |               0     |             1219.51 |             2356.95 |
| nominal    | K       |   5 |    0.107872 |  0.001329 |   0.138208 |   0.138932 |         0      |              0     |               0     |             1198.49 |             3455.91 |
| payload50  | Current |   5 |    0.073922 |  0.002336 |   0.095397 |   0.096094 |         0      |              0     |               0     |             1239.2  |             2507.95 |
| payload50  | I       |   5 |    0.073581 |  0.002329 |   0.095129 |   0.095837 |         0      |              0     |               0     |             1227.51 |             2437.73 |
| payload50  | K       |   5 |    0.073581 |  0.002329 |   0.095129 |   0.095837 |         0      |              0     |               0     |             1196.02 |             1977.88 |
| rotoreff30 | Current |   5 |    0.176123 |  0.055467 |   0.222371 |   0.224826 |         0      |              0     |               0     |             1235.2  |             2958    |
| rotoreff30 | I       |   5 |    0.175698 |  0.05573  |   0.221971 |   0.224451 |         0      |              0     |               0     |             1211.21 |             1996.47 |
| rotoreff30 | K       |   5 |    0.175698 |  0.05573  |   0.221971 |   0.224451 |         0      |              0     |               0     |             1182.7  |             1862.51 |
| wind3      | Current |   5 |    0.889307 |  0.185285 |   1.11201  |   1.11651  |         0      |              0     |               0     |             1278.87 |             2896.45 |
| wind3      | I       |   5 |    0.889307 |  0.185285 |   1.11201  |   1.11651  |         0      |              0     |               0     |             1231.78 |             3358.81 |
| wind3      | K       |   5 |    0.927923 |  0.250517 |   1.19892  |   1.20329  |        22.1242 |             18.517 |              67.014 |             1233.98 |             3442.28 |

## Interpretation

The frozen continuous-risk I controller transfers cleanly to the pinned RotorPy core in nominal, force, model-mismatch, latency, payload and rotor-efficiency tests: it gives a small improvement over the frozen current-only residual in those pilot runs. In the pinned Dryden wind test, I collapses back to current-only behavior, indicating that the transferred detector/risk authority does not open usefully under this much stronger native wind process.

The K HOCBF layer is transparent whenever the transferred state stays inside its admissible surrogate envelope. In most non-wind pilot cases K and I are numerically identical. Under native Dryden wind, however, K becomes active and can become infeasible/set-invalid for some seeds; the mean K result is **worse** than I in this tiny pilot. This is an important transfer warning: the K envelope and its `|d|<=1.6 m/s^2` assumption were calibrated on the source-derived plant and cannot be treated as automatically portable to native RotorPy wind dynamics.

The correct conclusion is therefore **not** that native validation is complete. It is that the learned I path shows promising native-core transfer, while K must be re-audited/recalibrated against native RotorPy disturbance/actuator dynamics before any native safety claim.

## Benchmark sanity check

A separate circle sanity probe using the native benchmark vehicle parameters and a simplified source-exact translational Geo-control law gave mean RMSE 0.359 m. This does **not** reproduce the published AdaptiveQuadBench Geo result and must not be presented as package-native benchmark equivalence; it uses the RotorPy `cmd_acc` low-level abstraction rather than executing the repository's full `GeoControl` motor-speed implementation through `Environment.run`.

## acados status

AdaptiveQuadBench's `QuadOptimizer` imports `AcadosOcp`, `AcadosOcpSolver`, and `AcadosModel` from `acados_template`. In this execution environment `acados_template` and the compiled acados library are absent, and outbound DNS is unavailable. Native acados MPC was therefore **not executed**. The existing source-derived constrained MPC result remains a separate result and must not be relabeled as AdaptiveQuadBench/acados-native.

## Timing

{
  "learned_IK_logic_p50_us": 992.9905,
  "learned_IK_logic_p95_us": 1256.20635,
  "learned_IK_logic_p99_us": 2236.8577199999995,
  "native_source_RK4_plant_step_p50_us_median": 1219.5095,
  "native_source_RK4_plant_step_p99_us_median": 1996.469389999998,
  "note": "container CPU / Python+NumPy+SciPy; software timing only, NOT HIL/embedded timing"
}

These are software timings on the current container CPU, not HIL or embedded timing measurements.

## Next step

For a real native confirmation, run the prepared protocol in a Python 3.11 `quadbench` environment with the repository's pinned RotorPy submodule and acados installed, then execute the retained I/K wrapper through the actual `Environment`/`GeoControl` or native MPC loop. Native K should first be re-audited using the full RotorPy mismatch distribution; its robust disturbance bound and HOCBF envelope should be frozen only after that audit.

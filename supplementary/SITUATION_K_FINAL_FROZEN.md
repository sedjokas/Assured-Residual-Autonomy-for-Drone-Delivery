# SITUATION K — FINAL/FROZEN Robust HOCBF Safety Filter

## Purpose

Situation K replaces the heuristic predictive admissibility layer of J with an independent robust high-order control-barrier-function (HOCBF) filter. No new neural model is introduced. The upstream proposal remains the frozen Situation-I risk-governed learned controller.

## Surrogate model and barrier set

For each translational tracking axis, K assumes the relative surrogate

`e_p_dot = e_v`

`e_v_dot = u - a_ref + d`

with the componentwise bounded mismatch assumption

`|d_j| <= 1.6 m/s^2`.

The protected set uses:
- `|e_p| <= [0.75, 0.75, 0.60] m`
- `|e_v| <= [2.2, 2.2, 1.7] m/s`
- first HOCBF auxiliary constraints `psi1 >= 0`
- command bound `|u_j| <= 5.5 m/s^2`.

For an upper position boundary `h = p_max - e_p`, K uses

`psi1 = -e_v + lambda1*h`

and enforces

`psi1_dot + lambda2*psi1 >= 0`

robustly for the worst admissible disturbance. The lower boundary is handled symmetrically. Velocity limits use first-order robust CBF constraints. The frozen gains are `lambda1=lambda2=3` and `lambda_v=4`.

These inequalities generate a convex interval `[u_lower,u_upper]` per axis. K applies the Euclidean projection of the proposal into their Cartesian product. If the interval intersection is infeasible, the implementation falls back to the bounded nominal command.

## Conditional guarantee

Within the stated relative double-integrator surrogate, if:
1. the state starts in the HOCBF admissible set,
2. the robust command interval remains feasible,
3. the actual lumped mismatch satisfies `|d_j| <= 1.6 m/s^2`, and
4. the filtered acceleration command is realized according to the surrogate,

then the applied command satisfies the robust HOCBF/CBF inequalities by construction.

This is **not** a formal proof for the full quadrotor simulator or a real vehicle. Actuator lag, delay, attitude dynamics, estimator error and unmodelled dynamics are outside that conditional proof unless absorbed by the stated disturbance bound.

A 200,000-state randomized algebraic audit found no sampled violation of the implemented robust barrier inequalities for feasible states.

## Healthy final campaign — 100 new paired seeds per scenario

| Scenario   |   I_RMSE_m |   K_RMSE_m |   K_penalty_vs_I_pct |   K_filter_active_pct |   K_infeasible_pct |   K_hocbf_set_invalid_pct |
|:-----------|-----------:|-----------:|---------------------:|----------------------:|-------------------:|--------------------------:|
| nominal    |   0.009708 |   0.009708 |                    0 |                     0 |                  0 |                         0 |
| wind3      |   0.12042  |   0.12042  |                    0 |                     0 |                  0 |                         0 |
| force_step |   0.087156 |   0.087156 |                    0 |                     0 |                  0 |                         0 |
| model20    |   0.014147 |   0.014147 |                    0 |                     0 |                  0 |                         0 |
| latency40  |   0.011837 |   0.011837 |                    0 |                     0 |                  0 |                         0 |
| payload50  |   0.033221 |   0.033221 |                    0 |                     0 |                  0 |                         0 |
| rotoreff30 |   0.037206 |   0.037206 |                    0 |                     0 |                  0 |                         0 |
| compound   |   0.108614 |   0.108614 |                    0 |                     0 |                  0 |                         0 |

K was completely transparent in all eight healthy scenarios:
- filter activation: 0%
- infeasible robust intervals: 0%
- HOCBF-set invalidity: 0%
- RMSE penalty versus unfiltered Situation I: 0%.

The measured surrogate mismatch stayed within the frozen `1.6 m/s^2` componentwise bound in all healthy final runs:

| Scenario   |   d_exceed_pct |   d_p95_x |   d_p95_y |   d_p95_z |   d_max_x |   d_max_y |   d_max_z |   configured_dmax_x |   configured_dmax_y |   configured_dmax_z |
|:-----------|---------------:|----------:|----------:|----------:|----------:|----------:|----------:|--------------------:|--------------------:|--------------------:|
| force_step |              0 |  0.823271 |  0.839375 |  0.854833 |  1.20373  |  1.19231  |  1.19757  |                 1.6 |                 1.6 |                 1.6 |
| wind3      |              0 |  0.830641 |  0.822527 |  0.853693 |  1.51131  |  1.38106  |  1.43854  |                 1.6 |                 1.6 |                 1.6 |
| nominal    |              0 |  0.067331 |  0.071504 |  0.061866 |  0.161748 |  0.161427 |  0.151851 |                 1.6 |                 1.6 |                 1.6 |
| latency40  |              0 |  0.089534 |  0.104367 |  0.079814 |  0.210068 |  0.260648 |  0.19857  |                 1.6 |                 1.6 |                 1.6 |
| payload50  |              0 |  0.2879   |  0.375763 |  0.086696 |  0.404849 |  0.482349 |  0.172194 |                 1.6 |                 1.6 |                 1.6 |
| compound   |              0 |  0.82872  |  0.817201 |  0.775111 |  1.53347  |  1.48933  |  1.577    |                 1.6 |                 1.6 |                 1.6 |
| rotoreff30 |              0 |  0.348129 |  0.575623 |  0.15535  |  0.724109 |  0.880641 |  0.346756 |                 1.6 |                 1.6 |                 1.6 |
| model20    |              0 |  0.154351 |  0.184237 |  0.06598  |  0.315078 |  0.390784 |  0.153575 |                 1.6 |                 1.6 |                 1.6 |

This empirical audit supports the assumption for the tested operating envelope; it does not prove the bound for arbitrary flight.

## Corrupted-proposal challenges

| Scenario                | Controller                |   n |   rmse_mean |   rmse_sd |   p95_mean |   max_mean |   excursion_gt0p5_mean_pct |   excursion_gt1m_mean_pct |   filter_active_mean_pct |   infeasible_mean_pct |   hocbf_set_invalid_mean_pct |   correction_mean |
|:------------------------|:--------------------------|----:|------------:|----------:|-----------:|-----------:|---------------------------:|--------------------------:|-------------------------:|----------------------:|-----------------------------:|------------------:|
| wind3_proposal_impulse  | I-RiskGovernor-unfiltered | 100 |    0.347301 |  0.032238 |   0.820028 |   0.934805 |                   16.9991  |                  0.517743 |                  0       |                     0 |                            0 |          0        |
| wind3_proposal_impulse  | J-PredictiveProjection    | 100 |    0.261447 |  0.030019 |   0.598367 |   0.693658 |                   10.0073  |                  0        |                  9.10191 |                     0 |                            0 |          0.243727 |
| wind3_proposal_impulse  | K-RobustHOCBF-Filter      | 100 |    0.209957 |  0.02528  |   0.453769 |   0.524011 |                    2.85987 |                  0        |                  9.09918 |                     0 |                            0 |          0.37201  |
| compound_proposal_stuck | I-RiskGovernor-unfiltered | 100 |    0.314406 |  0.032245 |   0.875507 |   1.01962  |                   11.6806  |                  1.90628  |                  0       |                     0 |                            0 |          0        |
| compound_proposal_stuck | J-PredictiveProjection    | 100 |    0.239757 |  0.03029  |   0.645731 |   0.737012 |                    8.43312 |                  0        |                  7.17016 |                     0 |                            0 |          0.161794 |
| compound_proposal_stuck | K-RobustHOCBF-Filter      | 100 |    0.215095 |  0.031181 |   0.55272  |   0.634326 |                    6.24841 |                  0        |                  8.64604 |                     0 |                            0 |          0.20119  |

### Wind + proposal impulses

Unfiltered I:
- RMSE: 0.34730 m
- >1 m excursion cycles: 0.518%

K:
- RMSE: 0.20996 m
- improvement versus I: 39.55%
- >1 m excursion cycles: 0.000%.

For the same challenge, J obtains RMSE 0.26145 m while K obtains 0.20996 m, so the HOCBF filter is materially less permissive around the protected boundary.

### Compound + stuck proposal

Unfiltered I:
- RMSE: 0.31441 m
- >1 m excursion cycles: 1.906%

K:
- RMSE: 0.21509 m
- improvement versus I: 31.59%
- >1 m excursion cycles: 0.000%.

J obtains RMSE 0.23976 m and K obtains 0.21509 m.

## Important limitation exposed by the fault tests

During deliberately corrupted proposal windows, the measured quantity `actual acceleration - filtered command` can transiently exceed the frozen `1.6 m/s^2` surrogate mismatch bound because the simulator contains actuator lag and delayed actuation. Therefore the **formal HOCBF implication cannot be claimed during every instant of those adversarial challenge windows**.

The empirical containment result remains valid: K eliminates >1 m excursions in both challenges and substantially reduces RMSE. But this must be described as empirical full-simulator containment, not as a theorem about the full plant.

## Completion criteria

```json
{
  "healthy_max_RMSE_penalty_vs_I_pct": 0.0,
  "healthy_transparency_pass_le_0p5pct": true,
  "healthy_max_filter_infeasible_pct": 0.0,
  "healthy_feasibility_pass_le_0p1pct": true,
  "healthy_max_hocbf_set_invalid_pct": 0.0,
  "healthy_set_membership_pass_le_0p1pct": true,
  "healthy_max_measured_disturbance_bound_exceedance_pct": 0.0,
  "healthy_disturbance_assumption_empirically_pass_le_0p1pct": true,
  "wind_proposal_impulse_RMSE_improvement_vs_I_pct": 39.54615624505855,
  "wind_proposal_impulse_gt1m_pct_K": 0.0,
  "wind_challenge_pass": true,
  "compound_stuck_RMSE_improvement_vs_I_pct": 31.586903768134977,
  "compound_stuck_gt1m_pct_K": 0.0,
  "compound_challenge_pass": true,
  "all_preregistered_thresholds_pass": true
}
```

All pre-registered K thresholds pass.

## Scientific conclusion

Situation K strengthens J in two ways:

1. the admissibility constraints now come from explicit robust HOCBF/CBF inequalities rather than a hand-selected finite-horizon error box;
2. the resulting filter remains exactly transparent on healthy runs while containing deliberately corrupted proposals more strongly than J in the tested challenges.

The reviewer-safe claim is:

> An independent robust HOCBF filter can be placed after the risk-governed learned controller without altering healthy closed-loop performance in the tested operating envelope. Under the stated double-integrator and bounded-mismatch assumptions, its projected command satisfies explicit robust barrier inequalities; in the fuller benchmark-derived simulator, the same filter empirically contains severe post-governor proposal faults, while the formal guarantee remains conditional on the surrogate assumptions.

## Next step

The next priority is no longer another learned or safety layer. It is **native benchmark validation** of the retained I/K stack in AdaptiveQuadBench/RotorPy/acados, followed by hardware/HIL timing and fault-injection measurements.

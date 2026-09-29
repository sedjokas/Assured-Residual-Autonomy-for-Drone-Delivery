# SITUATION L.2 — FINAL: Native-Source Scalar-Bound HOCBF Recalibration

Status: **FINAL NEGATIVE RESULT / REJECTED CONFIGURATION**.

## Scope and protocol

L.2 tests whether the Situation-K robust HOCBF can be transferred to the pinned RotorPy-source dynamics by recalibrating only the constant componentwise mismatch bound and the HOCBF state envelope. It does **not** add or retrain any neural component.

This remains a **pinned RotorPy-source-core experiment**, not a full package-native `run_eval.py`/acados execution.

Calibration and final testing were separated:

- calibration: 20 seeds per scenario, 10000–10019;
- ordinary independent test: 10 seeds per scenario, 12000–12009;
- post-governor fault challenges: 10 fresh seeds, 13000–13009.

No L.2 parameter was changed after looking at the independent test split.

## Frozen native-source calibration

The calibration-only rule produced:

- `dmax = [4.491898, 5.930514, 6.859521] m/s^2`
- `pmax = [1.416344, 1.52002, 1.461986] m`
- `vmax = [3.08, 3.08, 2.520874] m/s`
- envelope scale factor = **1.40**

The scale factor was the smallest member of the frozen 1.00–2.00 grid that gave zero infeasible calibration samples.

The calibrated disturbance bound is much larger than Situation K's `[1.6,1.6,1.6] m/s²`; notably, the calibrated vertical component `6.860 m/s²` exceeds the fixed command limit `5.5 m/s²`. This is an early warning that a globally robust box may be difficult or impossible to control for every admissible state.

## Independent ordinary tests

| scenario   | controller   |   n |   rmse_mean |   rmse_sd |   p95_mean |   max_mean |   excursion_gt1m_mean_pct |   filter_active_mean_pct |   filter_infeasible_mean_pct |   filter_set_invalid_mean_pct |   d_bound_exceed_mean_pct |
|:-----------|:-------------|----:|------------:|----------:|-----------:|-----------:|--------------------------:|-------------------------:|-----------------------------:|------------------------------:|--------------------------:|
| force_step | I            |  10 |    0.128416 |  0.039421 |   0.191018 |   0.194955 |                   0       |                   0      |                      0       |                             0 |                   0       |
| force_step | K2           |  10 |    0.128416 |  0.039421 |   0.191018 |   0.194955 |                   0       |                   0      |                      0       |                             0 |                   0       |
| latency40  | I            |  10 |    0.105347 |  0.001658 |   0.133527 |   0.134135 |                   0       |                   0      |                      0       |                             0 |                   0       |
| latency40  | K2           |  10 |    0.105347 |  0.001658 |   0.133527 |   0.134135 |                   0       |                   0      |                      0       |                             0 |                   0       |
| model20    | I            |  10 |    0.099086 |  0.010922 |   0.127257 |   0.127853 |                   0       |                   0      |                      0       |                             0 |                   0       |
| model20    | K2           |  10 |    0.099086 |  0.010922 |   0.127257 |   0.127853 |                   0       |                   0      |                      0       |                             0 |                   0       |
| nominal    | I            |  10 |    0.107777 |  0.001657 |   0.137146 |   0.137721 |                   0       |                   0      |                      0       |                             0 |                   0       |
| nominal    | K2           |  10 |    0.107777 |  0.001657 |   0.137146 |   0.137721 |                   0       |                   0      |                      0       |                             0 |                   0       |
| payload50  | I            |  10 |    0.071723 |  0.003937 |   0.091719 |   0.092283 |                   0       |                   0      |                      0       |                             0 |                   0       |
| payload50  | K2           |  10 |    0.071723 |  0.003937 |   0.091719 |   0.092283 |                   0       |                   0      |                      0       |                             0 |                   0       |
| rotoreff30 | I            |  10 |    0.187713 |  0.060031 |   0.242669 |   0.244964 |                   0       |                   0      |                      0       |                             0 |                   0       |
| rotoreff30 | K2           |  10 |    0.187713 |  0.060031 |   0.242669 |   0.244964 |                   0       |                   0      |                      0       |                             0 |                   0       |
| wind3      | I            |  10 |    0.733366 |  0.116506 |   0.906608 |   0.917431 |                   9.21844 |                   0      |                      0       |                             0 |                   5.27054 |
| wind3      | K2           |  10 |    0.773523 |  0.149917 |   0.953771 |   0.96779  |                  20.9619  |                  28.9379 |                      5.67134 |                             0 |                   5.59118 |

All six non-wind scenarios, including the previously missing rotor-efficiency split, are exactly transparent: K2 never activates and reproduces I's RMSE.

The native-Dryden result does **not** transfer cleanly:

- I RMSE: **0.73337 m**
- K2 RMSE: **0.77352 m**
- K2 RMSE change versus I: **+5.48%**
- K2 filter active: **28.94%**
- K2 robust interval infeasible: **5.67%**
- mismatch bound exceeded: **5.59%** of evaluated cycles.

Thus calibration-split feasibility does not generalize to the independent native-Dryden split.

## Post-governor proposal-fault challenges

Both final challenges use the difficult native-Dryden regime. The first preserves the original K impulse magnitude/window that fits inside the 6-s L.2 episode; the second uses the original stuck-proposal magnitude/window. These challenge seeds are disjoint from both calibration and ordinary tests.

| scenario               | controller   |   n |   rmse_mean |   rmse_sd |   p95_mean |   max_mean |   excursion_gt1m_mean_pct |   filter_active_mean_pct |   filter_infeasible_mean_pct |   filter_set_invalid_mean_pct |   filter_correction_mean |   d_bound_exceed_mean_pct |
|:-----------------------|:-------------|----:|------------:|----------:|-----------:|-----------:|--------------------------:|-------------------------:|-----------------------------:|------------------------------:|-------------------------:|--------------------------:|
| wind3_proposal_impulse | I            |  10 |    0.950773 |  0.391448 |    1.39436 |    1.41792 |                   26.8938 |                   0      |                       0      |                         0     |                  0       |                   9.23848 |
| wind3_proposal_impulse | K2           |  10 |    0.972429 |  0.534893 |    1.29188 |    1.30756 |                   27.7355 |                  41.2425 |                      23.8878 |                        13.988 |                  1.41651 |                   6.29258 |
| wind3_proposal_stuck   | I            |  10 |    0.929155 |  0.374055 |    1.38975 |    1.41726 |                   23.6473 |                   0      |                       0      |                         0     |                  0       |                   7.51503 |
| wind3_proposal_stuck   | K2           |  10 |    0.990592 |  0.531628 |    1.35646 |    1.37008 |                   30.8818 |                  44.4289 |                      24.3687 |                        13.988 |                  1.41703 |                   5.9519  |

### Proposal impulse

K2 lowers the mean maximum error from **1.418 m** to **1.308 m**, but this local clipping does not translate into better mission tracking:

- I RMSE: **0.95077 m**
- K2 RMSE: **0.97243 m**
- RMSE change: **+2.28%**
- cycles above 1 m: **26.89% → 27.74%**
- K2 infeasible: **23.89%**.

### Stuck proposal

Again, K2 slightly lowers mean maximum error but worsens aggregate tracking and large-error duty cycle:

- I RMSE: **0.92915 m**
- K2 RMSE: **0.99059 m**
- RMSE change: **+6.61%**
- cycles above 1 m: **23.65% → 30.88%**
- K2 infeasible: **24.37%**.

These are negative confirmatory results: the scalar-bound K2 filter cannot be claimed to improve post-governor fault containment on this native-Dryden split.

## Paired statistics

| scenario               | comparator   | proposal   |   n |   comparator_rmse_m |   proposal_rmse_m |   improvement_m |   relative_improvement_pct |   ci95_low_m |   ci95_high_m |   wilcoxon_p |   comparator_gt1m_pct |   proposal_gt1m_pct |   holm_adjusted_p_central3 |
|:-----------------------|:-------------|:-----------|----:|--------------------:|------------------:|----------------:|---------------------------:|-------------:|--------------:|-------------:|----------------------:|--------------------:|---------------------------:|
| nominal                | I            | K2         |  10 |           0.107777  |         0.107777  |       0         |                    0       |     0        |     0         |     1        |               0       |              0      |                    nan     |
| wind3                  | I            | K2         |  10 |           0.733366  |         0.773523  |      -0.0401569 |                   -5.4757  |    -0.079838 |    -0.0083331 |     0.125    |               9.21844 |             20.9619 |                      0.375 |
| force_step             | I            | K2         |  10 |           0.128416  |         0.128416  |       0         |                    0       |     0        |     0         |     1        |               0       |              0      |                    nan     |
| model20                | I            | K2         |  10 |           0.0990859 |         0.0990859 |       0         |                    0       |     0        |     0         |     1        |               0       |              0      |                    nan     |
| latency40              | I            | K2         |  10 |           0.105347  |         0.105347  |       0         |                    0       |     0        |     0         |     1        |               0       |              0      |                    nan     |
| payload50              | I            | K2         |  10 |           0.0717227 |         0.0717227 |       0         |                    0       |     0        |     0         |     1        |               0       |              0      |                    nan     |
| rotoreff30             | I            | K2         |  10 |           0.187713  |         0.187713  |       0         |                    0       |     0        |     0         |     1        |               0       |              0      |                    nan     |
| wind3_proposal_impulse | I            | K2         |  10 |           0.950773  |         0.972429  |      -0.0216556 |                   -2.27769 |    -0.154464 |     0.1075    |     1        |              26.8938  |             27.7355 |                      1     |
| wind3_proposal_stuck   | I            | K2         |  10 |           0.929155  |         0.990592  |      -0.0614371 |                   -6.61215 |    -0.183605 |     0.0343424 |     0.845703 |              23.6473  |             30.8818 |                      1     |

With only 10 paired seeds per final condition, the inferential statistics are intentionally treated as secondary to the mechanistic evidence. The central finding is the repeated combination of bound exceedance, interval infeasibility, and degraded RMSE.

## Scientific conclusion

L.2 **rejects simple scalar-bound recalibration as the native transfer solution**.

A calibration-only enlargement of `dmax`, `pmax`, and `vmax` restores feasibility on the calibration split and remains transparent in non-wind tests. It does not cover the tails of independent native Dryden operation, and the resulting filter remains intermittently infeasible. In the two fresh post-governor fault challenges, K2 clips some peak errors but worsens RMSE and the duty cycle of >1 m errors.

The reviewer-safe conclusion is:

> Recalibrating a constant lumped disturbance bound is insufficient to transfer the Situation-K HOCBF to the native-source Dryden/actuator regime. The full mismatch mixes environmental forcing with command-realization dynamics, producing heavy-tailed and state-dependent residuals that violate the frozen bound and can make the robust HOCBF interval infeasible.

## Next design implication

The next safety-filter model should be **actuator-aware**, not merely more conservative. A suitable direction is to augment the barrier model with a command-realization state, e.g.

`e_p_dot = e_v`

`e_v_dot = a_act - a_ref + d_ext`

`tau_a * a_act_dot = a_cmd - a_act`

so that motor/attitude realization lag is modeled explicitly and the residual robust bound is reserved for external/model uncertainty. A smaller certified operating domain may also be necessary.

This result should be preserved as a negative ablation. It prevents a misleading claim that Situation K's original or simply rescaled HOCBF envelope is native-transferable.

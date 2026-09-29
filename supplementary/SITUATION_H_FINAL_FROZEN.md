# SITUATION H — FINAL Confirmatory Runtime-Assurance Study

Status: **frozen**. Situations A–G remain preserved separately.

## Scope

Situation H completes the runtime-assurance study started after Situation G. The frozen current-only AE-CeNN remains the permanent fallback. The 100-ms history branch inherited from G is permitted only when the causal detector is persistently positive, data remain finite and inside frozen training-derived envelopes, the branch deadline is met, incremental history authority is bounded, and the resulting command preserves an actuator margin.

The implemented form is:

`r_H = r_current + s_RTA * (r_G - r_current)`, with `0 <= s_RTA <= 1`.

Therefore `s_RTA = 0` returns exactly to the frozen current-only learned residual.

## Main 100-seed campaign

The earlier H main campaign used 100 paired seeds per scenario. H preserves essentially all of G's tracking benefit. Relative to G, the RMSE differences are extremely small: less than about 0.12% in every scenario, with the largest small penalty in model mismatch. This is the intended result: H is an assurance wrapper, not a new performance network.

| Scenario   |   CurrentOnly-AE-CeNN-494 |   G-DetectorGate |   H-RTA-SupervisedGate |   LQR-outer |
|:-----------|--------------------------:|-----------------:|-----------------------:|------------:|
| nominal    |                   0.00964 |          0.00964 |                0.00964 |     0.00964 |
| wind3      |                   0.11995 |          0.1193  |                0.1193  |     0.14105 |
| force_step |                   0.08768 |          0.08739 |                0.08739 |     0.09821 |
| model20    |                   0.01555 |          0.01551 |                0.01553 |     0.0205  |
| latency40  |                   0.01184 |          0.01184 |                0.01184 |     0.01295 |
| payload50  |                   0.03332 |          0.03306 |                0.03306 |     0.04809 |
| rotoreff30 |                   0.03938 |          0.03917 |                0.03918 |     0.04868 |
| compound   |                   0.11166 |          0.11118 |                0.11122 |     0.12393 |

## False positives in nominal operation

Over 200 new nominal seeds:

- raw detector-positive cycle rate: **0.7971%**
- final RTA history-authority cycle rate: **0.0100%**
- episodes with at least one brief RTA opening: **7.5%**

The 3-cycle persistence and admissibility checks reduce the nominal duty cycle by roughly two orders of magnitude. The episode-level number is not zero: rare brief activations still occur, so the detector should not be described as false-positive-free.

## False negatives / intensity envelope

| family   |   intensity |   RTA_detection_rate_1s_pct |   RTA_detection_within_200ms_pct |   RTA_median_latency_s |   raw_detector_rate_1s_pct |   raw_detector_median_latency_s |
|:---------|------------:|----------------------------:|---------------------------------:|-----------------------:|---------------------------:|--------------------------------:|
| force    |         0.1 |                          38 |                                4 |                  0.34  |                        100 |                            0.15 |
| force    |         0.2 |                         100 |                               97 |                  0.15  |                        100 |                            0.08 |
| force    |         0.4 |                         100 |                              100 |                  0.07  |                        100 |                            0.04 |
| force    |         0.8 |                         100 |                              100 |                  0.05  |                        100 |                            0.03 |
| wind     |         0.5 |                          82 |                               25 |                  0.305 |                         99 |                            0.12 |
| wind     |         1   |                          99 |                               69 |                  0.15  |                        100 |                            0.08 |
| wind     |         1.5 |                         100 |                               99 |                  0.09  |                        100 |                            0.05 |
| wind     |         2   |                         100 |                              100 |                  0.07  |                        100 |                            0.04 |
| wind     |         3   |                         100 |                              100 |                  0.05  |                        100 |                            0.03 |
| wind     |         4   |                         100 |                              100 |                  0.04  |                        100 |                            0.02 |

The detector has a clear sensitivity boundary. A 0.1 N force is usually too weak for fast reliable RTA activation (4% within 200 ms), while 0.2 N reaches 97% within 200 ms. Wind shows a similar transition: 0.5 m/s gives 25% within 200 ms, 1.0 m/s gives 69%, and 1.5 m/s already gives 99%.

For the pre-registered strong cases:
- **0.8 N force: 100% within 200 ms, median RTA latency 50 ms**
- **3 m/s wind: 100% within 200 ms, median RTA latency 50 ms**

The RTA latency is intentionally slightly larger than the raw detector latency because H requires three consecutive positive cycles.

## Fault rejection

| FaultFamily              |   Injected_fault_cycles |   H_rejection_rate_pct |   H_acceptance_rate_pct |
|:-------------------------|------------------------:|-----------------------:|------------------------:|
| deadline_bursts          |                    4000 |                100     |                   0     |
| history_corruption       |                    8000 |                 99.975 |                   0.025 |
| sensor_spikes            |                     300 |                100     |                   0     |
| combined_faults          |                   12249 |                100     |                   0     |
| compound_faults          |                       0 |                nan     |                 nan     |
| wind3_deadline_bursts    |                    4000 |                100     |                   0     |
| wind3_history_corruption |                    8000 |                100     |                   0     |
| wind3_combined_faults    |                   12249 |                100     |                   0     |

Deadline bursts are rejected at 100%. Large history corruption is rejected at 99.975% in the nominal-base test and 100% when injected while the wind-disturbance branch is active. Sensor-spike cycles are rejected at 100%. The combined injected-fault campaigns are also rejected at 100%.

## Fault containment

| FaultFamily              | Controller              |   n |   rmse_mean |   rmse_sd |   p95_mean |   max_mean |   saturation_mean_pct |   excursion_gt1m_mean_pct |   gate_open_mean_pct |   governor_scale_mean |
|:-------------------------|:------------------------|----:|------------:|----------:|-----------:|-----------:|----------------------:|--------------------------:|---------------------:|----------------------:|
| combined_faults          | CurrentOnly-AE-CeNN-494 | 100 |    0.00988  |  0.000522 |   0.013459 |   0.014383 |                     0 |                         0 |              0       |              0        |
| combined_faults          | G-DetectorGate-noRTA    | 100 |    0.009887 |  0.000527 |   0.013477 |   0.0144   |                     0 |                         0 |            100       |              1        |
| combined_faults          | H-RTA-SupervisedGate    | 100 |    0.00988  |  0.000521 |   0.013456 |   0.014382 |                     0 |                         0 |              1.76797 |              0.01768  |
| wind3_deadline_bursts    | CurrentOnly-AE-CeNN-494 | 100 |    0.121054 |  0.011331 |   0.156831 |   0.162163 |                     0 |                         0 |              0       |              0        |
| wind3_deadline_bursts    | G-DetectorGate-noRTA    | 100 |    0.120413 |  0.011379 |   0.156196 |   0.161519 |                     0 |                         0 |            100       |              1        |
| wind3_deadline_bursts    | H-RTA-SupervisedGate    | 100 |    0.120439 |  0.011377 |   0.156221 |   0.161543 |                     0 |                         0 |             96.3603  |              0.963603 |
| wind3_history_corruption | CurrentOnly-AE-CeNN-494 | 100 |    0.121048 |  0.01134  |   0.156828 |   0.162166 |                     0 |                         0 |              0       |              0        |
| wind3_history_corruption | G-DetectorGate-noRTA    | 100 |    0.120414 |  0.01139  |   0.156214 |   0.161555 |                     0 |                         0 |            100       |              1        |
| wind3_history_corruption | H-RTA-SupervisedGate    | 100 |    0.12046  |  0.011382 |   0.15625  |   0.161591 |                     0 |                         0 |             92.7207  |              0.927207 |
| wind3_combined_faults    | CurrentOnly-AE-CeNN-494 | 100 |    0.122087 |  0.011281 |   0.158207 |   0.163643 |                     0 |                         0 |              0       |              0        |
| wind3_combined_faults    | G-DetectorGate-noRTA    | 100 |    0.121452 |  0.011334 |   0.157584 |   0.163021 |                     0 |                         0 |            100       |              1        |
| wind3_combined_faults    | H-RTA-SupervisedGate    | 100 |    0.121529 |  0.011322 |   0.15765  |   0.163097 |                     0 |                         0 |             87.4832  |              0.874832 |

The active-wind tests are the more discriminating fault-containment experiments because the history branch is normally open there. H disables invalid history authority during the injected windows and remains very close to both G and current-only in aggregate RMSE. Importantly, the present fault magnitudes did **not** generate >1 m tracking excursions in any controller. Consequently, the original excursion-based containment criterion technically passes but is non-discriminating; it should not be presented as evidence of a formal safety margin.

Likewise, H is not systematically lower-RMSE than unsupervised G during these faults. The side branch is already bounded and small, so fault rejection mainly changes *authority provenance* rather than mission-averaged RMSE. The evidence therefore supports deterministic containment/fallback behavior, not a claim that the governor rescues a catastrophically unstable learned controller.

## Pre-registered completion criteria

```json
{
  "nominal_false_activation_cycle_rate_pct": 0.0100090991810737,
  "nominal_false_activation_pass_lt_0p2pct": true,
  "force_0p8N_detection_within_200ms_pct": 100.0,
  "force_detection_pass_ge_95pct": true,
  "wind_3mps_detection_within_200ms_pct": 100.0,
  "wind_detection_pass_ge_90pct": true,
  "deadline_rejection_pct": 100.0,
  "deadline_rejection_pass_ge_99pct": true,
  "history_corruption_rejection_pct": 99.975,
  "history_corruption_rejection_pass_ge_99pct": true,
  "sensor_spike_rejection_pct": 100.0,
  "sensor_spike_rejection_pass_ge_95pct": true,
  "wind3_deadline_rejection_pct": 100.0,
  "wind3_history_corruption_rejection_pct": 100.0,
  "wind3_combined_fault_rejection_pct": 100.0,
  "all_pre_registered_thresholds_pass": true
}
```

All numerical threshold criteria pre-registered for H pass in this benchmark-derived study.

## Scientific conclusion

Situation H supports the following bounded claim:

> A disturbance-gated learned history residual can be wrapped by an independent runtime-assurance governor that suppresses nearly all nominal history authority, detects sufficiently strong force/wind disturbances with short causal latency, rejects injected deadline/data-integrity faults, and falls back exactly to the frozen current-only AE-CeNN path when the side branch is inadmissible.

It does **not** establish a formal safety proof. It also reveals an important limitation: detection probability falls sharply for weak perturbations. H therefore defines an empirical detection envelope rather than universal disturbance observability.

## Next architectural implication

H is now sufficiently frozen to justify moving to Situation I. The next useful change should be a risk-aware authority governor, not a larger neural model: use detector confidence, deadline margin, AE/reconstruction confidence or OOD score, and safety margin to continuously allocate residual authority while an independent admissibility/projection layer remains capable of forcing the authority to zero.

## Claim boundary

All H results remain source-derived benchmark simulation. They do not establish native full AdaptiveQuadBench/acados execution, formal CBF/reachability guarantees, hardware real-time certification, HIL, or flight validation.

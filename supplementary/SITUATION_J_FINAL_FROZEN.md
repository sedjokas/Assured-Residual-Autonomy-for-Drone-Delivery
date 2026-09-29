# SITUATION J — FINAL Independent Predictive Admissibility Projection

Status: **frozen targeted confirmatory study**. Situations A–I remain preserved. No new neural network is introduced.

## Architecture
Situation J takes the frozen Situation-I proposed command and projects it through an independent deterministic admissibility layer. For each axis, the layer intersects three interval constraints: (i) a 5.5 m/s² command-component limit, (ii) a predicted 0.30-s relative-position envelope of ±[0.55,0.55,0.45] m, and (iii) a predicted 0.30-s relative-velocity envelope of ±[1.5,1.5,1.1] m/s. The prediction is a constant-command relative-motion surrogate. The Euclidean projection onto this orthotope is closed-form clipping. If the intersection is infeasible, J falls back to the nominal geometric command.

The selected envelope was frozen on validation seeds 8800–8819 before final testing. It was the tightest tested candidate that produced zero projection activity and zero RMSE change in healthy nominal, wind3, and compound validation cases.

## Healthy final campaign — 100 paired seeds 9000–9099
| Scenario   |   CurrentOnly-AE-CeNN-494 |   I-RiskGovernor-unprojected |   J-IndependentSafetyProjection |
|:-----------|--------------------------:|-----------------------------:|--------------------------------:|
| nominal    |                  0.009682 |                     0.009681 |                        0.009681 |
| wind3      |                  0.120024 |                     0.119411 |                        0.119411 |
| force_step |                  0.087845 |                     0.08754  |                        0.08754  |
| model20    |                  0.013743 |                     0.013729 |                        0.013729 |
| latency40  |                  0.011857 |                     0.011857 |                        0.011857 |
| payload50  |                  0.034071 |                     0.033801 |                        0.033801 |
| rotoreff30 |                  0.040274 |                     0.040081 |                        0.040081 |
| compound   |                  0.108501 |                     0.108065 |                        0.108065 |

Across the seven stressed healthy scenarios, mean J-vs-I RMSE effect is **0.000000%**. Nominal projection activity is **0.0000%**, maximum healthy main projection activity is **0.0000%**, and maximum predictive-box infeasibility is **0.0000%**. Thus the independent projection is transparent in the validated healthy envelope.

## Corrupted-proposal challenges — 100 paired seeds 9200–9299
| Scenario                |   I_RMSE_m |   J_RMSE_m |   J_improvement_pct |   CI95_low_m |   CI95_high_m |   Wilcoxon_p |   I_max_mean_m |   J_max_mean_m |   I_exc_gt1m_pct |   J_exc_gt1m_pct |   J_projection_active_pct |   J_infeasible_pct |
|:------------------------|-----------:|-----------:|--------------------:|-------------:|--------------:|-------------:|---------------:|---------------:|-----------------:|-----------------:|--------------------------:|-------------------:|
| wind3_proposal_impulse  |   0.346965 |   0.260718 |             24.8576 |     0.081725 |      0.090592 |            0 |       0.932402 |       0.688171 |         0.500455 |                0 |                   9.10828 |                  0 |
| compound_proposal_stuck |   0.316386 |   0.237072 |             25.0688 |     0.073373 |      0.085381 |            0 |       1.03011  |       0.730152 |         2.08098  |                0 |                   7.26661 |                  0 |
| severe_compound         |   0.199488 |   0.199488 |              0      |     0        |      0        |          nan |       0.333091 |       0.333091 |         0        |                0 |                   0       |                  0 |

A post-governor, pre-projection command corruption was injected only to test whether an independent projection can contain a bad proposal. It is not presented as a model of a specific cyberattack.

Under the wind proposal-impulse challenge, J reduces mean RMSE by **24.86%** relative to unprojected I and does not increase the >1 m excursion fraction. Under the compound stuck-proposal challenge, J reduces mean RMSE by **25.07%** and likewise does not increase >1 m excursions.

The natural severe-compound case is also included as a non-corrupted stress check.

## Scientific interpretation
J demonstrates a qualitatively different property from H/I: learned/risk-governed modules may propose a command, but a deterministic external layer can constrain that proposal before it reaches the plant. This makes command admissibility independent of neural correctness.

However, this is **not a formal safety proof**. The projection uses a simplified constant-command prediction and fixed tracking envelopes, not a control-barrier certificate, invariant set, or reachability computation. The healthy final campaign also shows that the chosen projection is mostly inactive in the normal benchmark envelope; its value appears under deliberately corrupted proposals.

## Pre-registered criteria
```json
{
  "healthy_mean_stressed_J_vs_I_pct": 0.0,
  "healthy_transparency_pass": true,
  "nominal_projection_active_pct": 0.0,
  "main_max_projection_active_pct": 0.0,
  "main_max_infeasible_pct": 0.0,
  "projection_feasibility_pass": true,
  "wind_impulse_J_improvement_pct": 24.85760238360793,
  "wind_impulse_I_gt1m_pct": 0.5004549590536852,
  "wind_impulse_J_gt1m_pct": 0.0,
  "wind_impulse_pass": true,
  "compound_stuck_J_improvement_pct": 25.068781768459655,
  "compound_stuck_I_gt1m_pct": 2.08098271155596,
  "compound_stuck_J_gt1m_pct": 0.0,
  "compound_stuck_pass": true,
  "all_preregistered_J_criteria_pass": true
}
```

All pre-registered Situation-J numerical criteria pass.

## Claim boundary
The evidence supports independent predictive command containment in this source-derived benchmark simulation. It does not establish formal forward invariance, native AdaptiveQuadBench/acados execution, HIL/flight safety, or hardware real-time certification.

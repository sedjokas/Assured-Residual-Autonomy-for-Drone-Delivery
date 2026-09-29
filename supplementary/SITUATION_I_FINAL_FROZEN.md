# SITUATION I — FINAL/FROZEN Continuous Risk-Aware Authority Governor

Situation I is frozen after an independent 100-seed main campaign (8000–8099) and an independent 100-seed challenge rerun (8300–8399). A–H remain preserved.

## Architecture
The frozen H hard-RTA shell is retained. Inside that shell, history authority is continuously scaled by the minimum of detector confidence, deadline-margin confidence, a reconstruction/OOD confidence applied to both current and 100-ms-history inputs, actuator margin, dynamic tracking margin, and the frozen action-projection limit. Hard invalidity still forces authority to zero.

## Main result
| Scenario   |   CurrentOnly-AE-CeNN-494 |   H-RTA-SupervisedGate |   I-ContinuousRiskGovernor |
|:-----------|--------------------------:|-----------------------:|---------------------------:|
| nominal    |                  0.009712 |               0.009712 |                   0.009712 |
| wind3      |                  0.120138 |               0.119509 |                   0.119509 |
| force_step |                  0.087561 |               0.087253 |                   0.08726  |
| model20    |                  0.015842 |               0.015818 |                   0.015818 |
| latency40  |                  0.011829 |               0.011829 |                   0.011829 |
| payload50  |                  0.033874 |               0.033608 |                   0.033608 |
| rotoreff30 |                  0.038676 |               0.038473 |                   0.038474 |
| compound   |                  0.11094  |               0.110475 |                   0.110488 |

Mean stressed I-vs-H RMSE effect: **-0.0029%** (positive means I is better). Nominal I-vs-current effect: **0.0022%**. The continuous governor therefore preserves the H tracking envelope.

Nominal mean history authority is **0.000121**, while mean stressed authority is **0.6441**, a selectivity ratio of about **5311x**.

## Soft-risk challenges
Under an eroding but still positive deadline margin, I reduces mean authority from H=0.710 to I=0.422, a **40.5%** reduction. Under the moderate history-distribution shift, authority falls from H=0.976 to I=0.716, a **26.6%** reduction.

The original exploratory 50% attenuation targets are **not met** and were not retuned: deadline target pass=False; soft-OOD target pass=False. This is retained as a negative result.

## Hard faults
Deadline bursts and gross history corruption both force **100% hard veto** in the target windows. The hard RTA shell therefore retains final authority over the continuous risk governor.

## Severe compound stress
In severe compound stress, I reduces mean history authority from H=0.954 to I=0.788; mean OOD confidence is 0.809. This demonstrates graded attenuation rather than a universal hard shutdown.

## Conclusion
Situation I supports a bounded claim: continuous risk signals can reduce learned history authority before hard invalidity while leaving the H veto/fallback semantics intact. The strong 50% soft-attenuation targets were too ambitious for the frozen calibration and fail; hard-fault containment and main non-inferiority pass. I is therefore frozen without further tuning.

This remains benchmark-derived simulation evidence, not formal safety certification or native hardware timing evidence.

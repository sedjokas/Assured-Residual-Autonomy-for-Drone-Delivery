# SITUATION E — Compact Selective-History AE-CeNN

Status: **frozen 100-seed confirmatory extension**.

Situations A, B, C, D and D2-confirmatory remain preserved separately.

## Architecture tested

The strongest previous learned baseline was the current-only AE-CeNN with 494 active inference parameters.

Situation E keeps that instantaneous information and adds only one compact historical view per CeNN cell:

- `m_i(t)`
- `tau * dm_i(t)`
- `tau^2 * d2m_i(t)`
- `m_i(t - 0.10 s)`
- `tau * dm_i(t - 0.10 s)`
- `tau^2 * d2m_i(t - 0.10 s)`

The AE input therefore grows from 3 to 6 values per cell, while its hidden width is reduced from 64 to 45. The active path is **496 parameters**, versus **494** for current-only. An identical 496-parameter zero-history control is included, with the three historical slots clamped to zero.

A shuffled-at-test variant preserves the lagged-feature distribution but destroys temporal correspondence.

## Protocol

- 100 untouched paired test seeds per scenario: 1500–1599.
- 12 s episodes at 100 Hz.
- Metrics start after the 1.01 s warm-up.
- Same frozen training split and residual authority as D2-confirmatory.
- Eight scenarios: nominal, wind, force-step, model mismatch, 40 ms latency, payload, rotor-efficiency loss, and compound stress.
- Benchmark-derived simulation only; no native full AdaptiveQuadBench/acados, HIL, or flight claim.

## Mean position RMSE [m]

| Scenario   |   CurrentOnly-AE-CeNN-494 |   LQR-outer |   SelectiveLag10-AE-CeNN-496 |   SelectiveLag10-shuffled-at-test |   ZeroHistory6D-AE-CeNN-496 |
|:-----------|--------------------------:|------------:|-----------------------------:|----------------------------------:|----------------------------:|
| nominal    |                   0.00962 |     0.00963 |                      0.00971 |                           0.00968 |                     0.01026 |
| wind3      |                   0.12115 |     0.1414  |                      0.13006 |                           0.14229 |                     0.12239 |
| force_step |                   0.08739 |     0.0982  |                      0.09575 |                           0.10279 |                     0.08884 |
| model20    |                   0.01486 |     0.01966 |                      0.01636 |                           0.01938 |                     0.0168  |
| latency40  |                   0.01179 |     0.01295 |                      0.01226 |                           0.01225 |                     0.01243 |
| payload50  |                   0.03373 |     0.04885 |                      0.03907 |                           0.03897 |                     0.03777 |
| rotoreff30 |                   0.0402  |     0.04995 |                      0.04697 |                           0.04913 |                     0.04466 |
| compound   |                   0.1106  |     0.12305 |                      0.12136 |                           0.12997 |                     0.11619 |

## Central finding

The coherent 100-ms lag **does contain useful temporal information**, because it beats its shuffled-history version strongly under wind, force, model mismatch, rotor-efficiency loss, and compound stress.

However, direct concatenation of that lag into the AE bottleneck does **not** improve the strongest current-only architecture.

Across the seven stressed scenarios, selective-lag AE-CeNN is on average:

- **10.49% worse** than the 494-parameter current-only AE-CeNN;
- **3.31% worse** than its identical 496-parameter zero-history capacity control;
- **5.96% better** than the same selective model when the lag is shuffled;
- **8.57% better** than LQR on average in this benchmark-derived sweep.

The predefined success rule for direct history augmentation therefore **fails**.

## Regime-level result versus the identical 496-parameter zero-history control

Coherent lag10 improves:
- nominal: 5.35%;
- model mismatch: 2.59%;
- 40 ms latency: 1.35%.

It degrades:
- wind: 6.27%;
- force-step: 7.78%;
- payload: 3.43%;
- rotor-efficiency: 5.16%;
- compound: 4.45%.

## Scientific interpretation

This is not evidence that temporal history is useless. The shuffled-history test shows the opposite: temporally coherent lagged information clearly matters in several disturbance regimes.

The more likely problem is **fusion interference**. The current-only latent representation is already strong. Forcing current and lagged information through one shared three-dimensional AE bottleneck changes that representation and degrades closed-loop behavior.

The next architecture should therefore preserve the current-only latent path and add history through a separate bounded side branch, for example:

`z = z_current + gate * delta_z_history`

with the gate initialized near zero and bounded. This tests whether history can contribute only when useful without corrupting the current-state representation.

## Claim boundary

Situation E is benchmark-derived simulation evidence. It does not establish flight validation, native full AdaptiveQuadBench/acados performance, or generic CeNN superiority.

# SITUATION G — Disturbance-detected sparse history gating

Status: **frozen targeted confirmatory snapshot**. Situations A–F remain preserved.

## Architecture
The exact strong current-only AE-CeNN path is frozen. The 100-ms history branch remains separate. A nominal-calibrated causal anomaly detector multiplies the learned history gate, so the side branch can be nearly closed during nominal behavior and open only when the current observation departs from nominal dynamics.

## Detector
The detector uses four causal aggregate features: disturbance-observer magnitude, derivative of the disturbance observer, current-vs-100-ms-lag innovation, and tracking-state magnitude. Each is robustly standardized using only nominal training data. The scalar anomaly score is the RMS of positive robust z-scores.

## Protocol
- Hyperparameters selected only on validation seeds 2400–2439.
- Final test: 100 new paired seeds 3200–3299 per scenario, 12 s at 100 Hz.
- Main controls: current-only, always-open history side, detector-shuffled gate, and LQR.
- Force-step event analysis uses 50 final-test seeds and measures detector latency causally.

## Main mean RMSE [m]

| Scenario   |   AlwaysOpen-HistorySide |   CurrentOnly-AE-CeNN-494 |   DetectorShuffled-GatedHistory |   EventDetected-GatedHistory |   LQR-outer |
|:-----------|-------------------------:|--------------------------:|--------------------------------:|-----------------------------:|------------:|
| nominal    |                  0.00975 |                   0.00973 |                         0.00973 |                      0.00973 |     0.00963 |
| wind3      |                  0.11923 |                   0.11986 |                         0.11923 |                      0.11923 |     0.14112 |
| force_step |                  0.08752 |                   0.08783 |                         0.08752 |                      0.08752 |     0.09815 |
| model20    |                  0.01457 |                   0.01463 |                         0.01462 |                      0.0146  |     0.0192  |
| latency40  |                  0.01183 |                   0.01183 |                         0.01183 |                      0.01183 |     0.01295 |
| payload50  |                  0.03296 |                   0.03323 |                         0.03297 |                      0.03297 |     0.04796 |
| rotoreff30 |                  0.03902 |                   0.03923 |                         0.03904 |                      0.03902 |     0.04842 |
| compound   |                  0.10917 |                   0.10963 |                         0.10917 |                      0.10917 |     0.12205 |

## Pre-registered criteria
```json
{
  "selected_detector_q": 0.995,
  "selected_gmax": 0.35,
  "nominal_gate_mean": 0.0029313472889823528,
  "mean_stressed_gate": 0.10999703888248573,
  "stress_to_nominal_gate_ratio": 37.524396804974934,
  "positive_CI_vs_current_stressed": [
    "wind3",
    "force_step",
    "model20",
    "latency40",
    "payload50",
    "rotoreff30",
    "compound"
  ],
  "positive_CI_vs_alwaysopen_stressed": [],
  "positive_CI_vs_detector_shuffled_stressed": [
    "model20",
    "rotoreff30"
  ],
  "nominal_improvement_vs_current_pct": 0.011166921738548113,
  "median_force_detection_latency_s": 0.03,
  "force_detection_rate": 1.0,
  "control_benefit_pass": true,
  "detector_specificity_pass": false,
  "gate_separation_pass": true,
  "force_response_pass": true
}
```

## Claim boundary
benchmark-derived simulation only; no native full AdaptiveQuadBench/acados, HIL, flight-validation, or generic CeNN-superiority claim
# E1 Final Results — Actuator-Aware Runtime Safety Validation

**Evidence label:** independent source-informed simulation; not package-native AdaptiveQuadBench/acados.

## Calibration-only frozen model parameters
- Identified command-realization time constants: [0.0878, 0.0906, 0.0638] s
- Lumped-model mismatch bounds: [1.3996, 0.9827, 0.6743] m/s²
- Actuator-aware external-disturbance bounds: [1.2349, 0.8746, 0.6143] m/s²

## Independent ordinary-test results

| scenario   | filter   |    rmse |   violation |   active |   infeasible |     gt1 |
|:-----------|:---------|--------:|------------:|---------:|-------------:|--------:|
| compound   | AA       | 0.11993 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| compound   | LM       | 0.11993 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| compound   | P        | 0.11993 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| dryden3    | AA       | 0.01851 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| dryden3    | LM       | 0.01851 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| dryden3    | P        | 0.01851 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| force      | AA       | 0.08297 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| force      | LM       | 0.08297 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| force      | P        | 0.08297 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| lag2x      | AA       | 0.01214 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| lag2x      | LM       | 0.01214 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| lag2x      | P        | 0.01214 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| nominal    | AA       | 0.01133 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| nominal    | LM       | 0.01133 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| nominal    | P        | 0.01133 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| rotor_eff  | AA       | 0.01802 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| rotor_eff  | LM       | 0.01802 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |
| rotor_eff  | P        | 0.01802 |     0.00000 |  0.00000 |      0.00000 | 0.00000 |

## Fresh post-proposal challenges

| fault   | scenario   | filter   |    rmse |   max_error |     gt1 |   violation |   active |   infeasible |
|:--------|:-----------|:---------|--------:|------------:|--------:|------------:|---------:|-------------:|
| impulse | compound   | AA       | 0.18470 |     0.68964 | 0.00000 |     0.00000 |  0.00000 |      0.00000 |
| impulse | compound   | LM       | 0.18470 |     0.68964 | 0.00000 |     0.00000 |  0.00000 |      0.00000 |
| impulse | compound   | P        | 0.18470 |     0.68964 | 0.00000 |     0.00000 |  0.00000 |      0.00000 |
| impulse | dryden3    | AA       | 0.12602 |     0.60493 | 0.00000 |     0.00000 |  0.00000 |      0.00000 |
| impulse | dryden3    | LM       | 0.12602 |     0.60493 | 0.00000 |     0.00000 |  0.00000 |      0.00000 |
| impulse | dryden3    | P        | 0.12602 |     0.60493 | 0.00000 |     0.00000 |  0.00000 |      0.00000 |
| stuck   | compound   | AA       | 0.73763 |     3.14584 | 0.08639 |     0.05263 |  0.08663 |      0.06576 |
| stuck   | compound   | LM       | 0.78672 |     3.32226 | 0.09003 |     0.05867 |  0.09363 |      0.06916 |
| stuck   | compound   | P        | 1.22581 |     4.87265 | 0.11267 |     0.10242 |  0.00000 |      0.00000 |
| stuck   | dryden3    | AA       | 0.50025 |     2.31871 | 0.06800 |     0.01171 |  0.05374 |      0.03024 |
| stuck   | dryden3    | LM       | 0.52638 |     2.41689 | 0.07098 |     0.02178 |  0.06047 |      0.03546 |
| stuck   | dryden3    | P        | 1.43310 |     5.74445 | 0.12151 |     0.10813 |  0.00000 |      0.00000 |

## Confirmatory paired statistics: AA versus LM

| fault   | scenario   | metric     |   mean_a |   mean_b |   mean_diff |     ci95_lo |    ci95_hi |           p |   p_holm_primary |
|:--------|:-----------|:-----------|---------:|---------:|------------:|------------:|-----------:|------------:|-----------------:|
| stuck   | dryden3    | rmse       | 0.500247 | 0.526376 |  -0.0261292 | -0.0261452  | -0.0261129 | 3.89656e-18 |      7.79312e-18 |
| stuck   | dryden3    | violation  | 0.01171  | 0.021785 |  -0.010075  | -0.010145   | -0.01001   | 7.60701e-19 |    nan           |
| stuck   | dryden3    | gt1        | 0.068    | 0.07098  |  -0.00298   | -0.002995   | -0.00296   | 9.51933e-23 |    nan           |
| stuck   | dryden3    | infeasible | 0.03024  | 0.035465 |  -0.005225  | -0.00527012 | -0.005175  | 5.28611e-19 |    nan           |
| stuck   | compound   | rmse       | 0.737627 | 0.786718 |  -0.0490909 | -0.0491626  | -0.0490162 | 3.89656e-18 |      7.79312e-18 |
| stuck   | compound   | violation  | 0.052635 | 0.058665 |  -0.00603   | -0.006055   | -0.00601   | 1.48001e-19 |    nan           |
| stuck   | compound   | gt1        | 0.08639  | 0.09003  |  -0.00364   | -0.0037     | -0.00358   | 5.60165e-19 |    nan           |
| stuck   | compound   | infeasible | 0.06576  | 0.069155 |  -0.003395  | -0.00344    | -0.003345  | 8.16726e-20 |    nan           |

## Frozen acceptance criteria
- PASS — healthy_transparency_nominal
- PASS — healthy_transparency_dryden3
- PASS — healthy_transparency_lag2x
- PASS — healthy_transparency_rotor_eff
- PASS — healthy_transparency_force
- PASS — healthy_transparency_compound
- PASS — nominal_intervention_lt_1pct
- PASS — stuck_dryden3_AA_better_RMSE_than_P
- PASS — stuck_dryden3_AA_lower_violation_than_P
- PASS — stuck_dryden3_AA_no_worse_RMSE_than_LM
- PASS — stuck_dryden3_AA_no_worse_violation_than_LM
- PASS — stuck_dryden3_AA_no_worse_infeasible_than_LM
- PASS — stuck_compound_AA_better_RMSE_than_P
- PASS — stuck_compound_AA_lower_violation_than_P
- PASS — stuck_compound_AA_no_worse_RMSE_than_LM
- PASS — stuck_compound_AA_no_worse_violation_than_LM
- PASS — stuck_compound_AA_no_worse_infeasible_than_LM
- PASS — stuck_dryden3_AA_vs_LM_RMSE_Holm_p_lt_0.05
- PASS — stuck_compound_AA_vs_LM_RMSE_Holm_p_lt_0.05

Runtime: 32.0 s

## Interpretation
In ordinary operation all three controllers are expected to be nearly identical because the safety envelopes are not approached. That is a transparency test. The short impulse is intentionally retained even if the filters remain inactive: if the predicted envelope stays admissible, non-intervention is correct. The severe stuck-proposal challenge is the decisive filter stress test. The main comparison is whether AA improves containment over P and over the lumped instantaneous-command LM filter while reducing or not increasing filter infeasibility.

## Claim boundary
These results can support a new manuscript subsection as a mechanism-level actuator-aware confirmation. They cannot be called package-native AdaptiveQuadBench/acados evidence or HIL/flight validation. A final package-native rerun using the authors’ frozen CeNN/governor wrapper would remain the strongest next confirmation.
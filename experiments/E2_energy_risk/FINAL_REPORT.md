# E2 Final Report — Cross-Layer Risk-Bounded Energy Admission and In-Flight Mission Adaptation

## Scientific status
**Completed confirmatory simulation, with an important mixed result.** The pooled preregistered criteria all pass, but the global uncertainty calibration is not conditionally reliable in the hardest headwind/compound regimes and becomes substantially conservative on long-route/compound missions. Therefore E2 upgrades H1/R1 from *open / architecture only* to **partially supported by direct simulation evidence**, not fully validated.

## Evidence level
Independent physics-grounded / source-informed simulation. It implements the manuscript's risk-admission contract `E_risk = mu_E + kappa*sigma_E` with a protected reserve and explicit in-flight measured-energy feedback. Vehicle-scale parameters are source-informed, but the battery capacity and mission-energy environment are synthetic. This is **not** package-native AdaptiveQuadBench energy validation, an electrochemical battery model, HIL, or flight evidence.

## Frozen design
- 12,000 model-fit calibration missions (seed 21000).
- 12,000 separate risk-calibration missions (seed 22000).
- 48,000 untouched final missions: eight scenarios × 6,000 missions.
- Same calibrated mean twin for DET and RISK; RISK differs only by the uncertainty margin.
- RISK and ADAPT share the same preflight admission gate; ADAPT adds online measured-energy correction and contingency diversion.
- One-sided target bound coverage: 97.5%.
- Frozen calibration multiplier: **kappa = 2.689**.

## Pooled independent-test results
| Metric | GEO | DET | RISK | ADAPT |
|---|---:|---:|---:|---:|
| Admission rate | 75.77% | 77.73% | 66.99% | 66.99% |
| Unsafe admission among truly infeasible missions | 14.25% | 7.51% | **0.344%** | 0.344% preflight |
| Reserve violations among admitted missions | 4.10% | 2.11% | **0.112%** | **0.000%** |
| False rejection of truly feasible missions | 7.08% | 2.69% | **14.43%** | 14.43% |
| Delivery success over all candidate missions | 75.77% | 77.73% | 66.99% | **66.14%** |
| Diversion among admitted missions | 0% | 0% | 0% | **1.34%** |

The pooled calibrated upper bound covered **97.46%** of untouched true mission energies. RISK reduced the admitted-mission reserve-violation rate from **2.11% to 0.112%**, a **94.7% relative reduction**, while rejecting **14.43%** of ground-truth feasible missions. ADAPT eliminated the remaining reserve violations in this test set, with a **1.34%** diversion rate and only **0.85 percentage points** lower delivery success than RISK.

## Predeclared criteria
| criterion                                         | pass   |   observed |   threshold |
|:--------------------------------------------------|:-------|-----------:|------------:|
| RISK coverage >= 96.0%                            | True   | 0.974604   |        0.96 |
| RISK reserve violations reduced >=50% vs DET      | True   | 0.946853   |        0.5  |
| RISK false rejection <=20%                        | True   | 0.144319   |        0.2  |
| ADAPT reserve violations reduced >=40% vs RISK    | True   | 1          |        0.4  |
| ADAPT diversion <=15% admitted                    | True   | 0.0133731  |        0.15 |
| ADAPT delivery loss <=8 percentage points vs RISK | True   | 0.00845833 |        0.08 |

## Critical scenario-level result — must remain visible
Pooled calibration is not enough to claim uniformly calibrated risk. Coverage by scenario was:

| scenario         |   coverage |
|:-----------------|-----------:|
| nominal          |      97.43 |
| heavy_payload    |      99.03 |
| headwind         |      92.62 |
| aged_battery     |      99.58 |
| cold_temperature |      99.35 |
| vertical_motion  |      99.78 |
| long_route       |      98.92 |
| compound         |      92.97 |

The hardest regimes expose a limitation: **headwind coverage = 92.62%** and **compound coverage = 92.97%**, materially below the 97.5% target. The RISK policy also becomes conservative in the long-route and compound regimes:

| scenario         |   admission_rate |   false_rejection_rate |   reserve_violation_rate_admitted |
|:-----------------|-----------------:|-----------------------:|----------------------------------:|
| nominal          |            99.5  |                   0.48 |                              0    |
| heavy_payload    |            85.45 |                  10.06 |                              0.1  |
| headwind         |            75.55 |                  16.39 |                              0.62 |
| aged_battery     |            71    |                  19.5  |                              0.02 |
| cold_temperature |            83.37 |                  12.03 |                              0    |
| vertical_motion  |            91.03 |                   7.74 |                              0    |
| long_route       |            29.83 |                  46.77 |                              0.11 |
| compound         |             0.17 |                  94.92 |                              0    |

This is a useful negative result. A single global `kappa` and lightweight heteroscedastic scale model provide excellent pooled reserve protection, but they do **not** guarantee conditional calibration under strong distribution shift. A future energy twin should therefore use regime-/feature-conditional uncertainty calibration or conformal/quantile methods and should be independently confirmed on new seeds.

## Statistical interpretation
The most informative paired test is unsafe admission among ground-truth infeasible missions: RISK sharply reduces these admissions relative to DET on identical missions. Comparing DET and RISK reserve violations only on missions admitted by both is intentionally uninformative because both policies execute the same physical mission once admitted; the gain comes from the risk-aware admission decision itself. ADAPT then reduces the residual post-admission risk through online feedback/diversion.

## Claim supported by E2
> In the tested physics-grounded simulation family, a calibrated uncertainty margin around a common mean-energy twin substantially reduces unsafe mission admission and protected-reserve violations relative to deterministic admission, while online measured-energy adaptation can contain residual reserve risk with a low pooled diversion rate. However, a single global calibration is not conditionally reliable in the strongest headwind/compound regimes and can become overconservative on difficult long routes.

## Claim boundary
Do **not** claim flight-certified battery safety, package-native AdaptiveQuadBench energy validation, universal 97.5% coverage, or universal superiority. H1/R1 should be marked **partially supported**, with the conditional-calibration limitation explicit.

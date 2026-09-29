# E2 Frozen Protocol — Cross-Layer Risk-Bounded Energy Admission and In-Flight Mission Adaptation

**Status:** frozen before independent final-test generation.

## Research question
Can a calibrated per-drone energy digital twin with an explicit uncertainty margin reduce infeasible / reserve-violating mission admissions without unacceptable conservatism, and can in-flight energy feedback further contain reserve risk when realized conditions depart from the preflight forecast?

## Hypothesis targeted
**H1 / R1:** calibrated risk-bounded mission admission reduces infeasible or reserve-violating missions without unacceptable conservatism.

The experiment implements the manuscript contract

- `E_risk = mu_E + kappa * sigma_E`
- admit only if `E_risk + E_reserve <= E_available`.

## Evidence level
Independent **physics-grounded / source-informed simulation**. Vehicle mass, rotor radius, motor time constant, rotor layout, and related source context are aligned with the pinned Hummingbird/RotorPy material used by the manuscript, but this is **not** a package-native AdaptiveQuadBench energy benchmark and is **not** flight/HIL evidence.

## Vehicle / energy simulator
- Base vehicle mass: 0.50 kg.
- Four rotors, radius 0.10 m each.
- Nominal battery energy: 30 Wh (synthetic mission battery assumption; not claimed as the historical Hummingbird battery).
- Hard reserve: 15% of SoH-adjusted nominal capacity.
- Mission energy includes takeoff/climb, loaded outbound cruise, delivery hover, unloaded return cruise, landing, and auxiliary electrical load.
- Ground-truth energy depends on payload, route length, vertical motion, actual wind on both legs, air density / temperature, aerodynamic drag, propulsion efficiency, SoH-dependent loss, and unmodeled auxiliary load.
- Preflight features use forecast / estimated conditions and therefore contain realistic forecast/model errors.

## Frozen comparison families
**GEO** — calibrated distance-only admission. A linear energy-vs-distance model is fitted on the model-fit calibration split; it ignores payload, wind, SoH, temperature and vertical motion.

**DET** — calibrated deterministic energy twin. A multivariate mean-energy correction model is fitted on the same model-fit split; admission uses `mu_E` only, with no uncertainty margin.

**RISK** — calibrated risk-bounded admission. It uses the same `mu_E` as DET plus heteroscedastic `sigma_E` and a one-sided calibration multiplier `kappa` chosen on a separate calibration split to target 97.5% upper-bound coverage.

**ADAPT** — same preflight admission as RISK, followed by in-flight cumulative-energy monitoring. At 25%, 50%, and 75% mission progress, the remaining-energy prediction is multiplicatively updated from measured-vs-predicted consumption. If projected terminal energy threatens the hard reserve, the mission diverts to a predeclared contingency site. Diversion energy is simulated as 45% of the original remaining-route energy from that checkpoint. Diversions before package delivery count as undelivered; diversions after delivery count as delivered but not original-route completion.

## Calibration / test separation
- Model-fit calibration missions: 12,000, fixed RNG seed 21000.
- Risk-calibration missions: 12,000, fixed RNG seed 22000.
- Independent final test: 8 scenarios x 6,000 missions = 48,000 missions, scenario seed block 30000–30007.
- Final-test missions are generated only after the model family, metrics, risk target and acceptance criteria below are frozen.

## Stress scenarios
1. nominal
2. heavy_payload
3. headwind
4. aged_battery
5. cold_temperature
6. vertical_motion
7. long_route
8. compound

The compound condition combines heavy payload, strong headwind, aged battery, cold temperature, longer route and substantial vertical motion.

## Primary endpoints
1. **Unsafe admission rate:** fraction of ground-truth infeasible full missions that are admitted.
2. **Reserve-violation rate:** fraction of admitted/executed missions ending below the protected reserve.
3. **Mission completion rate:** original route completed.
4. **Delivery success rate:** package delivered (can include post-delivery diversion for ADAPT).
5. **Terminal SoC and minimum reserve margin.**
6. **Upper-bound coverage:** fraction of missions with `E_true <= mu_E + kappa*sigma_E`.
7. **False rejection rate:** fraction of ground-truth feasible missions rejected preflight.
8. **Diversion rate** for ADAPT.
9. **Energy per delivered mission** among delivered missions.

## Predeclared primary success criteria
E2 is considered to materially support H1/R1 only if, on the pooled independent final test:

1. RISK upper-bound coverage is at least **96.0%** (target 97.5%).
2. RISK reduces reserve violations among admitted missions by at least **50% relative to DET**.
3. RISK false rejection of truly feasible missions is at most **20%**.
4. ADAPT reduces reserve violations by at least **40% relative to RISK** among missions admitted by the shared RISK/ADAPT preflight gate.
5. ADAPT diversion rate is at most **15%** of admitted missions.
6. ADAPT delivery-success rate is no more than **8 percentage points below RISK**.

Failure of any criterion is preserved and reported; no final-test retuning is allowed.

## Statistics
- Common missions are used for all policies.
- Wilson 95% intervals for proportions.
- Bootstrap 95% intervals for continuous aggregate differences where useful.
- McNemar exact/binomial test for paired binary policy outcomes.
- Wilcoxon signed-rank test for paired continuous outcomes on mutually executed / delivered missions.
- Holm correction across the predeclared principal pairwise tests.

## Claim boundary
A positive result can support: **a calibrated uncertainty margin and online energy feedback can improve simulated reserve protection relative to distance-only or deterministic admission under the tested model family.**

It cannot support: flight-certified battery safety, package-native AdaptiveQuadBench energy validation, real battery electrochemistry, or universal superiority of the chosen energy model.

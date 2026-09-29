# E1 — Actuator-Aware Runtime Safety Validation under Source-Informed UAV Dynamics

**Protocol frozen before the independent final seeds (12000–12099 / 13000–13099) were evaluated.**

## Scientific question
Does replacing a lumped instantaneous-command safety model by an explicit actuator-state model improve downstream safety-filter behavior under command-realization lag, Dryden wind, rotor-efficiency loss, and compound stress, while remaining transparent during healthy operation?

## Provenance boundary
This is a new independent implementation derived from Eq. (18) of the manuscript and the pinned RotorPy Hummingbird parameters/Dryden equations. It is not package-native AdaptiveQuadBench/acados and does not reproduce the manuscript's trained CeNN. The proposal source is deliberately controller-agnostic: a stable PD tracking law plus a bounded residual; the experiment isolates the downstream assurance model.

## Compared systems
- **P**: frozen bounded-residual proposal, no final safety filter.
- **LM**: lumped-model robust predictive filter using an instantaneous double-integrator command model and a calibration-only lumped mismatch bound (structurally analogous to the failed scalar-bound family).
- **AA**: actuator-aware robust predictive filter using the explicit augmented state `[position error, velocity error, realized acceleration]`, a calibration-only command-realization time constant, and a separate external-disturbance bound.

## Fixed protocol
- dt = 0.01 s; mission duration = 20 s; robust prediction horizon = 0.30 s.
- command bound = ±5.5 m/s² per axis.
- position envelope = [1.40, 1.40, 1.25] m; velocity envelope = [3.0, 3.0, 2.5] m/s.
- Hummingbird mass/drag coefficients and RotorPy Dryden recursion are taken from pinned RotorPy source commit `07e6e2c57d55fc563ac55f9c44153c2769f0ae73`.
- Calibration seeds: 10000–10019, never used for final testing.
- Pilot seeds: 11000–11019, used only to verify implementation and choose realistic final success criteria.
- Ordinary independent final seeds: 12000–12099.
- Fresh proposal-fault seeds: 13000–13099.
- Six ordinary scenarios: nominal, severe Dryden, doubled command-realization lag, rotor-efficiency loss, external-force disturbance, and compound stress.
- Two fresh post-proposal challenges under Dryden and compound stress: 0.30-s impulse and 0.80-s stuck extreme proposal.

## Primary endpoints
Position RMSE, position-error cycles >1 m, position/velocity-envelope violation rate, filter intervention rate, robust-filter infeasibility rate, and command-variation RMS.

## Frozen success criteria
1. **Healthy transparency:** AA ordinary-test mean RMSE must be within +2% of P in every scenario; nominal AA intervention rate <1%.
2. **No unnecessary filtering:** under the short impulse challenge, AA may remain inactive if the predicted envelope remains admissible; this is interpreted as selectivity, not failure.
3. **Severe stuck-proposal containment:** in both Dryden and compound conditions, AA must improve mean RMSE and envelope-violation rate versus P.
4. **Actuator-model value:** in both severe stuck-proposal conditions, AA must have mean RMSE and envelope-violation rate no worse than LM, and AA infeasibility must not exceed LM.
5. **Statistical confirmation:** for the two stuck-proposal conditions, paired AA-vs-LM RMSE differences are tested with Wilcoxon signed-rank tests; Holm-adjusted p < 0.05 is confirmatory. Envelope-violation and >1 m duty are secondary paired endpoints.

## Claim boundary
Passing E1 supports the **mechanism claim** that explicitly modeling command realization can improve a downstream safety filter relative to a lumped instantaneous model in this source-informed simulation. It does not establish package-native AdaptiveQuadBench/acados validity, HIL/flight safety, or formal full-plant forward invariance.

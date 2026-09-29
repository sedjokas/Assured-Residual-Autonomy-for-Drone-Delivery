# Comprehensive Experimental Compendium

> **Scope note.** This compendium covers the historical experiments A-QNL. The cross-layer experiments of the current manuscript are documented in their own frozen packages: `experiments/E1_actuator_aware/results/FINAL_REPORT.md`, `experiments/E2_energy_risk/FINAL_REPORT.md` and `experiments/E3_equal_budget_qi/FINAL_REPORT.md`.

## Purpose

This document records the complete experimental chain used to turn the AP-QI-CeNN reference architecture from a research agenda into an evidence-bearing, falsifiable design. It is intentionally broader than the main manuscript. Positive, neutral, and negative results are retained because many of the architectural decisions in the final system were driven by failed ablations rather than by monotonic performance gains.

The experiments are not all at the same provenance level. The progression is:

`synthetic -> public-benchmark-derived -> AdaptiveQuadBench/RotorPy source-derived -> pinned RotorPy-source core`

No full package-native AdaptiveQuadBench/acados execution, HIL, or flight experiment is claimed in this archive.

---

## A. Synthetic micro-H3 feasibility

### Question
Can a very small Chua-Yang-style CeNN residual improve tracking under disturbances while a stable nominal controller and independent runtime-assurance layer remain available?

### Protocol
- 120 paired Monte-Carlo missions.
- 28 s per mission, 50 Hz.
- Gusts, model mismatch, measurement error, and a forced CeNN outage.
- Tiny 3x3 CeNN, 19 learned parameters.
- Compact MLP baseline, 22 parameters.
- Nominal controller, nominal+RTA, MLP+RTA, CeNN only, CeNN+RTA, and forced-dropout fallback.
- Preregistered metrics: RMSE, maximum error, corridor violation, control effort, RTA interventions and residual authority.

### Key results
- Nominal RMSE: **0.5095 m**, corridor violations **23.33%**.
- Nominal+RTA: **0.4483 m**, violations **5.0%**.
- MLP+RTA: **0.1269 m**, violations **3.33%**.
- CeNN only: **0.1025 m**, violations **2.5%**.
- CeNN+RTA: **0.1024 m**, violations **2.5%**, about **79.9%** lower RMSE than nominal.
- CeNN outage -> nominal+RTA: **0.1770 m**, violations **4.17%**.
- Paired CeNN+RTA vs nominal RMSE improvement about **0.407 m**, bootstrap 95% CI approximately **[0.386, 0.430] m**.

### Interpretation
This experiment supports *mechanism feasibility*, not external validity. The generator and learned residuals share related functional structure, the RTA is not a formal CBF/reachability proof, `u^2` is not propulsion energy, and software parameters are not hardware timing/power measurements.

---

## SCG. Safe-Control-Gym-derived public benchmark

### Question
Does the CeNN residual remain competitive on a model/task derived from an external public benchmark, and how does it compare with strong conventional control?

### Provenance
The experiment independently reimplements the published Safe-Control-Gym 2-D quadrotor symbolic/reference model and figure-eight tracking task. PyBullet/Gymnasium were unavailable in the original sandbox; this is therefore **benchmark-derived**, not package-native.

### Protocol
- 2-D quadrotor.
- 6 s figure-eight.
- 50 Hz controller update; 1 kHz plant integration.
- 100 paired seeds across nominal, inertial shift, gusts, sensor noise, and compound conditions.
- PD, LQR, diagnostic LMPC-H40, parameter-matched compact MLP+RTA, CeNN, CeNN+RTA, and forced CeNN dropout.
- Separate strong constrained LMPC-H40 (OSQP) compound subset, N=20.

### Key compound results
- PD: **0.3233 m** RMSE.
- LQR: **0.2629 m**.
- MLP+RTA: **0.2813 m**.
- CeNN(+RTA): **0.2761 m**.
- Strong constrained LMPC-H40 subset: **0.1762 +/- 0.0595 m**, **0%** state-bound violations.

### Interpretation
This experiment is central to claim discipline. The CeNN improves on PD in the hard compound condition but is about 5% worse than LQR, while the strong constrained MPC comparator is substantially better. The paper therefore does **not** claim that a CeNN replaces MPC. The learned residual is treated as a bounded adaptive complement.

---

## B. Derivative-aware CeNN

### Question
Do first and second derivatives of local feature cells improve residual learning?

### Design
Each base measurement cell `m` was expanded to `[m, tau dm/dt, tau^2 d2m/dt2]`, with `tau=0.10 s`, using a causal nine-sample quadratic differentiator plus smoothing. CeNN-D2 was compared with same-capacity repeated-current CeNN-V3 and a compact MLP.

### Outcome
Offline normalized target RMSE improved modestly with derivatives, but closed-loop behavior usually did not. Examples:
- wind: V3 **0.0803**, D2 **0.0837 m**;
- force: V3 **0.0680**, D2 **0.0731 m**;
- model mismatch: V3 **0.1733**, D2 **0.1792 m**;
- latency: V3 **0.0736**, D2 **0.0786 m**.

### Interpretation
Derivative augmentation is feasible but creates noise/redundancy and should not be added indiscriminately.

---

## C. Denoising autoencoder before derivative CeNN

### Question
Can a compact autoencoder make noisy derivative information more useful?

### Architecture
For each cell temporal triplet `[m, tau dm, tau^2 d2m]`, a `3->6->2->6->3` denoising autoencoder feeds the derivative-aware CeNN. The combined model has about 120 active parameters in this study.

### Representative results
The AE-CeNN improves several disturbed regimes relative to derivative CeNN:
- wind: about **0.1030 vs 0.1036 m**;
- force: **0.0532 vs 0.0549 m**;
- model mismatch: **0.0588 vs 0.0641 m**;
- compound: **0.1871 vs 0.1914 m**.

It does not solve:
- latency, where both learned variants can be worse than nominal;
- severe rotor-efficiency faults.

### Interpretation
Representation quality matters, but the AE is not itself a safety mechanism. Reconstruction error later becomes a useful candidate for confidence/OOD monitoring.

---

## D. History depth and serial autoencoders

### Question
Would deeper autoencoding or richer causal history outperform the compact current-state representation?

### Variants
- D.1: three serial AEs before the CeNN.
- D.2: 27 scalar features expanded with causal history metadata: current, lag10, lag100, min, Q1, median, Q3, max; one history AE.
- D.3: D.2 plus three serial history AEs.

### Outcome
On exploratory paired tests:
- D.1 gave only about **0.4%** average improvement over the C rerun.
- D.2 gave about **13.8%** average improvement.
- D.3 became roughly **44% worse**.

### Interpretation
History appeared promising, but D.2 had far greater capacity than C. This capacity confound motivated a separate confirmatory experiment.

---

## D2. Capacity-matched confirmatory history study

### Question
Was the D.2 gain caused by temporal history, or simply by a larger representation/model?

### Protocol
- 200 paired held-out seeds per main scenario, seeds 300-499.
- 12 s at 100 Hz.
- Exact 101-sample causal history with exact quartiles.
- Small C baseline, capacity-matched current-only AE-CeNN, D2 history model, History-MLP, shuffled/lag-only/stats-only ablations.

### Core result
A **494-active-parameter current-only AE-CeNN** beat the history D2 model in every tested condition. D2 could beat the smaller C model, but not the active-capacity-matched current-only control.

### Interpretation
The claim that history itself caused D.2's gain was rejected. Capacity and current representation were more important. Shuffled-history tests still showed that coherent temporal information exists, so the next design objective became adding history *without corrupting the current latent*.

---

## E. Selective direct lag concatenation

### Question
Does a single 100-ms lag add useful information when total capacity is approximately matched?

### Design
The strong 494-parameter current-only path was compared with:
- 496-parameter zero-history capacity control;
- 496-parameter direct lag10 concatenation;
- shuffled-lag test control.

### Key result
Across stressed conditions, direct lag concatenation was about:
- **10.5% worse** than current-only;
- **3.3% worse** than the zero-history matched-capacity control;
- yet about **6.0% better** than shuffled history.

### Interpretation
The lag contains information, but forcing current and lagged features through the same bottleneck creates **representational interference**. This directly motivates a separate history branch.

---

## F. Dual-branch bounded history residual

### Question
Can history be added as a separate bounded latent residual while preserving the current-only path exactly?

### Architecture
- Frozen 494-parameter current-only AE-CeNN.
- 100-ms history branch with a small encoder.
- Learned bounded gate.
- Total active inference parameters: 574.
- If the side branch fails, the model returns exactly to current-only.

### Result
Mean stressed RMSE improvement over current-only: about **0.115%**. Six of seven stressed comparisons have positive paired confidence intervals versus current-only, but the absolute effect is tiny. The mean gate is about **0.049**, nearly constant and only a small fraction of its allowed maximum.

### Interpretation
The protected side branch is a viable architecture, but F does **not** prove sophisticated adaptive gating. Its strongest contribution is graceful compositionality: disabling the history branch returns the known current-only path.

---

## G. Event-detected history authorization

### Question
Can a causal disturbance detector keep the history branch nearly closed in nominal operation and authorize it under actual disturbances?

### Architecture
A robust median/MAD-calibrated detector uses disturbance-observer level, derivative, current-vs-lag innovation, and tracking state. It scales the history gate.

### Key results
Mean gate:
- nominal: **0.00293**;
- stressed mean: about **0.110**;
- stress/nominal separation: **37.5x**.

RMSE improvement over current-only:
- wind: **0.53%**;
- force: **0.36%**;
- payload: **0.79%**;
- compound: **0.42%**;
- stressed mean: about **0.4%**.

Force-step median detector latency: about **30 ms**, 100% detection in the characterized strong case.

### Negative attribution result
A detector-shuffling preregistered specificity criterion failed. The result therefore supports **selective authority**, not optimal event attribution.

---

## H. Independent runtime-assurance governor

### Question
Can learned history authority be independently supervised so invalid/late side-branch outputs are discarded while healthy tracking is preserved?

### Hard RTA checks
- three consecutive positive detector cycles;
- finite/plausible history and jump integrity monitor;
- state admissibility;
- bounded incremental history contribution;
- actuator margin;
- 5-ms side-branch deadline.

### Main result
H preserves essentially all of G's tracking performance; differences are below about 0.12% in all main scenarios.

### False-positive/false-negative characterization
- nominal RTA history-authority activation: **0.0100% of cycles**;
- 0.8 N force detected within 200 ms: **100%**;
- 3 m/s wind detected within 200 ms: **100%**.

The detector has an empirical sensitivity boundary:
- force 0.1 N: 4% within 200 ms;
- force 0.2 N: 97%;
- wind 0.5 m/s: 25%;
- wind 1.0 m/s: 69%;
- wind 1.5 m/s: 99%.

### Fault rejection
- deadline bursts: **100%**;
- history corruption: **99.975%** in nominal-base test and 100% in active-wind test;
- sensor spikes: **100%**;
- combined active-wind injected faults: **100%**.

### Interpretation
H demonstrates executable contract monitoring and exact fallback behavior. It is not a formal CBF/reachability proof and does not "rescue" an otherwise unstable controller; the learned side branch is already small and bounded.

---

## I. Continuous risk-aware authority

### Question
Can authority be continuously reduced as confidence degrades while the hard H veto remains supreme?

### Authority law
The history branch is scaled by the minimum of:
- detector confidence;
- deadline margin;
- OOD/reconstruction confidence;
- actuator margin;
- dynamic/tracking margin;
- plus the existing action projection.

Hard faults still force scale exactly to zero.

### Main results
- mean stressed I-vs-H RMSE difference: **-0.0029%** (practically identical);
- nominal degradation vs current-only: **+0.0022%**;
- nominal mean authority: **0.000121**;
- stressed mean authority: **0.644**;
- authority selectivity ratio: **>5300x**.

### Honest failures
Preregistered 50% soft-attenuation goals were not reached:
- soft deadline reduction: **40.5%**;
- soft-OOD/history-shift reduction: **26.6%**.

Hard deadline and gross history corruption still produce **100% veto**.

### Interpretation
I supports continuous risk allocation inside a hard supervisory shell. The two failed soft targets are retained rather than retuned post hoc.

---

## J. Independent predictive admissibility projection

### Question
Can a deterministic, non-neural layer constrain the final proposal without changing healthy behavior?

### Method
`u_safe = projection(u_prop)` into a hand-designed predictive admissibility set with command, predicted position, and predicted velocity bounds over a 0.30-s constant-command horizon.

### Healthy campaign
Across eight healthy scenarios:
- projection activity: **0%**;
- infeasibility: **0%**;
- RMSE penalty vs I: **0%**.

### Corrupted-proposal challenges
- wind + proposal impulse: RMSE improves by **24.86%**, cycles with error >1 m fall to **0%**;
- compound + stuck proposal: RMSE improves by **25.07%**, cycles >1 m fall to **0%**.

### Interpretation
J is an independent deterministic containment layer, but not a formal invariant-safety theorem.

---

## K. Robust HOCBF/CBF safety filter

### Question
Can J's heuristic predictive box be replaced by explicit robust barrier inequalities?

### Surrogate
Per translational axis:

`e_p_dot = e_v`

`e_v_dot = u - a_ref + d`, with `|d_j| <= 1.6 m/s^2`.

The protected set includes position, velocity, first HOCBF auxiliary constraints, and `|u| <= 5.5 m/s^2`.

### Healthy source-derived campaign
- filter activation: **0%**;
- infeasibility: **0%**;
- set invalidity: **0%**;
- RMSE penalty vs I: **0%**;
- measured surrogate mismatch bound exceedance in healthy final runs: **0%**.

A 200,000-state numerical algebraic audit found no sampled violation of the implemented robust inequalities for feasible sampled states.

### Proposal-fault challenges
- wind impulse: **39.55%** RMSE improvement vs I and 0% cycles >1 m;
- compound stuck proposal: **31.59%** improvement and 0% cycles >1 m.

### Claim boundary
The guarantee is conditional on the surrogate model, feasible intervals, command realization, and disturbance bound. It is not a theorem for the full multirotor dynamics.

---

## L. Pinned RotorPy-source transfer pilot

### Question
Do the frozen I/K mechanisms transfer to source-faithful RotorPy dynamics without retraining?

### Provenance
- AdaptiveQuadBench commit: `d4c273861aa0ce6750818af0b1b63a2a40408e52`.
- RotorPy submodule: `07e6e2c57d55fc563ac55f9c44153c2769f0ae73`.
- Source-faithful vehicle parameters, 6-DOF dynamics, aerodynamics, motor lag/noise, command allocation, and Dryden model.
- Pilot N=5 per scenario.

### Result
I transfers reasonably in several non-wind regimes. Under native-source Dryden:
- I: about **0.8893 m** RMSE;
- K: about **0.9279 m**;
- K active about **22.1%**;
- robust interval infeasible about **18.5%**;
- state outside K's HOCBF set about **67%**.

### Interpretation
K's source-derived bound/set is not directly portable. This is a transfer failure, not a reason to widen the bound after seeing the test.

---

## L2. Independent native-source HOCBF recalibration

### Question
Can K be rescued by recalibrating only its constant lumped disturbance bound and state envelope on a separate native-source calibration split?

### Strict split
- calibration: 20 seeds/scenario, seeds 10000-10019;
- ordinary test: 10 independent seeds/scenario, 12000-12009;
- post-governor fault tests: 10 new seeds, 13000-13009.

No L2 parameter is changed after examining the independent test split.

### Calibration-only result
Frozen recalibration:
- `dmax = [4.492, 5.931, 6.860] m/s^2`;
- `pmax = [1.416, 1.520, 1.462] m`;
- `vmax = [3.08, 3.08, 2.521] m/s`;
- state-envelope scale factor: **1.40**.

The vertical disturbance bound already exceeds the fixed command limit of 5.5 m/s^2, signaling a fundamental controllability/feasibility tension.

### Independent test
All non-wind conditions remain transparent. Native-source Dryden does not:
- I: **0.73337 m**;
- K2: **0.77352 m**;
- K2 degradation: **+5.48%**;
- filter infeasible: **5.67%**;
- frozen mismatch bound still exceeded: **5.59%**.

### Fresh fault challenges
Under native-source Dryden:
- proposal impulse: K2 RMSE **+2.28% worse** than I and >1 m duty increases by about 0.84 percentage points;
- proposal stuck: K2 RMSE **+6.61% worse** and >1 m duty increases by about 7.23 percentage points.

### Conclusion
Simple scalar-bound recalibration is **rejected** as a native transfer solution. The mismatch combines environmental forcing with command-realization/actuator dynamics, producing heavy-tailed and state-dependent residuals. The next theoretically justified model would make actuator realization explicit rather than hiding it inside an ever-larger constant disturbance bound.

---

---

## QNL. Structured Quadratic CeNN Nonlinearity Ablation

### Question
Does explicit bounded second-order CeNN state interaction improve closed-loop residual control beyond the intrinsic CeNN nonlinearity `Y=tanh(X)`?

### Models
Four closely matched variants were compared:
- LinearCeNN-494: retained bounded CeNN baseline;
- QCeNN-XX-503: adds bounded componentwise XX terms;
- QCeNN-XXXY-512: adds bounded XX and XY terms;
- LinearCeNN-Capacity-515: larger linear-dynamics CeNN capacity control.

All models retain `Y=tanh(X)`. For the quadratic variants, `xbar=tanh(X/2)` bounds the additional terms, so the test concerns extra explicit second-order expressivity rather than “linear versus nonlinear” control.

### Protocol
Five initialization replicates were trained per architecture. A documented pre-final amendment selected the median replicate using held-out offline residual MSE rather than final closed-loop RMSE. Final evaluation then used **200 untouched paired seeds per scenario** (16000-16199), eight scenarios and 6400 controller-missions.

### Main effects of QCeNN-XXXY versus the retained baseline
- nominal: **+3.053%** improvement;
- wind3: **-0.112%**;
- force step: **-0.020%**;
- model mismatch: **+0.730%**;
- latency 40 ms: **+1.223%**;
- payload: **-0.445%**;
- rotor efficiency: **-0.271%**;
- compound: **-0.075%**.

Only **2/7 stressed conditions** had a positive 95% confidence interval, and mean stressed improvement was only **+0.147%**. The preregistered replacement criterion therefore **failed**. All 6400 missions remained finite with zero residual-output-bound violations.

### Interpretation
The result does not show that CeNN nonlinearity is useless: the baseline is already nonlinear through `tanh(X)`. It shows that the tested bounded second-order interactions do not earn their additional complexity through sufficiently consistent closed-loop gains. This negative result reinforces the paper's design rule that additional model expressivity must justify itself under closed-loop, paired, out-of-sample evidence.

## Cross-experiment synthesis

The experimental sequence supports five high-level conclusions.

1. **CeNN value is regime-dependent and modest on public-style benchmarks.** The strongest constrained MPC comparator remains superior in the hard Safe-Control-Gym-derived compound condition.
2. **Representation design matters more than network depth.** Deeper AE stacks, indiscriminate derivatives, and direct history concatenation are not reliable improvements.
3. **History should be protected and selectively authorized.** A separate bounded branch allows exact fallback to a strong current-only path.
4. **Assurance should be independent of learned correctness.** H, I, J, and K progressively move authority decisions out of the learned residual itself.
5. **Safety arguments do not automatically transfer with the controller.** L/L2 show that a barrier model calibrated on one dynamics family can fail on a more source-faithful Dryden/actuator regime even when the learned controller itself transfers reasonably.

These conclusions motivate the final manuscript framing: the contribution is not a claim of universal CeNN superiority, but a reproducible architecture-and-assurance study showing where a bounded learned residual helps, how its authority can be constrained, and where the safety model itself breaks.

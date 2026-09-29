# Claim boundary

This file mirrors Appendix D (Tables D1 and D2) of the manuscript. It states what the archived evidence supports and which wording it does **not** support.

## Supported claims

1. **Architecture.** ARA separates proposal generation from execution authority. "Assured" is an architectural term (contracts, fallback, runtime assurance, independent final-command filtering); it does not mean certification.
2. **Residual CeNN (A, SCG, B-D2, QNL).** A small bounded CeNN residual can improve tracking in selected regimes, but it is **not** generally superior to LQR or constrained MPC. The retained CeNN is already nonlinear through `Y = tanh(X)`; bounded explicit `XX`/`XY` terms fail the preregistered replacement criterion.
3. **Representation (B-F).** Derivatives, deeper autoencoders, long history and direct lag concatenation are not automatically beneficial; capacity controls matter. A separate bounded history branch gives exact fallback to the current-only path.
4. **Selective authority and runtime assurance (G-I).** A causal detector and an independent RTA can suppress nominal learned authority and reject injected deadline/data-integrity faults. Two preregistered soft-attenuation targets of I fail and are reported as failures.
5. **Independent final-command layer (J, K).** Deterministic projection and a robust HOCBF filter empirically contain deliberately corrupted post-governor proposals in the source-derived simulator. The HOCBF property is conditional on its surrogate model and mismatch bound.
6. **Transfer failure (L, L2).** Scalar-bound HOCBF recalibration does not transfer to pinned-source Dryden/actuator dynamics.
7. **E1.** Modelling command realization explicitly (actuator-aware filter) improves severe stuck-proposal containment over the lumped filter, while remaining transparent in healthy cases.
8. **E2 (H1 partially supported).** Risk-bounded admission reduces reserve violations among admitted missions from 2.11% to 0.112% on 48,000 untouched missions; online adaptation removes the remaining violations in the tested set. Headwind/compound coverage (~93%) and long/compound conservatism remain open.
9. **E3 (H2 not supported).** The tested CPU Simulated-Bifurcation backend does not beat MILP-HiGHS or GRASP-LS under equal end-to-end accounting. QUBO remains a solver-neutral interface; the QI backend has no privileged status.

## Hypothesis status

| Hypothesis | Status | Evidence boundary |
|---|---|---|
| H1 energy-risk admission | Partially supported | E2 pooled reserve protection is strong; conditional calibration and conservatism remain open. |
| H2 solver-regime benefit | Not supported for the tested SB-QI backend | E3 CPU benchmark only; no claim about quantum hardware. |
| H3 bounded CeNN residual | Partially supported | Synthetic feasibility and some source-derived utility; no generic superiority over LQR/MPC. |
| H4 compositional assurance | Strengthened but partial | H-K containment; L/L2 transfer failure; E1 reformulation; no full-plant/HIL/flight proof. |

## Not supported by this release

- Universal CeNN superiority, or CeNN as a replacement for LQR/MPC/robust control.
- Quantum or quantum-inspired advantage; any statement that all quantum methods are inferior.
- A universal 97.5% battery-safety or conditional-coverage guarantee; a first energy-risk planner.
- Formal full-plant safety, native RotorPy invariance, or a first actuator-aware barrier method.
- Package-native AdaptiveQuadBench/acados results, embedded timing, HIL or flight validation.
- One integrated end-to-end ARA run: E1-E3 are coordinated but separate experiments.
- That every historical simulation can be rerun bit-for-bit (see `REPRODUCIBILITY.md`, "Known gaps").

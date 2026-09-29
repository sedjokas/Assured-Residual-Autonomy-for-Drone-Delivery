# PROVENANCE

## Evidence levels

### Level P0 - Synthetic feasibility
A uses a compact synthetic closed-loop generator. It establishes mechanism feasibility only.

### Level P1 - Public-benchmark-derived independent reproduction
The SCG campaign independently reimplements the published Safe-Control-Gym 2-D quadrotor task because the original PyBullet/Gymnasium stack was not available in the execution environment. It is **not** a package-native Safe-Control-Gym run.

### Level P2 - AdaptiveQuadBench/RotorPy source-derived reproduction
B-K and the QNL nonlinearity ablation use a source-derived 6-DOF plant/controller framework informed by AdaptiveQuadBench/RotorPy. They do not execute AdaptiveQuadBench `run_eval.py` or native acados MPC. `LMPC-H40-surrogate` is a simplified diagnostic comparator, not the benchmark acados controller.

The QNL campaign keeps the standard CeNN output nonlinearity `Y=tanh(X)` and tests whether bounded explicit second-order `XX` and `XY` interactions add closed-loop value. Its final negative result concerns **additional explicit second-order expressivity**, not whether nonlinear CeNN dynamics are useful at all.

### Level P3 - Pinned RotorPy-source core
L/L2 use equations and parameters pinned to:
- AdaptiveQuadBench commit `d4c273861aa0ce6750818af0b1b63a2a40408e52`;
- RotorPy submodule commit `07e6e2c57d55fc563ac55f9c44153c2769f0ae73`.

This improves source fidelity but remains a source-core execution, not full package-native `run_eval.py`. QNL also includes a small zero-shot pinned-source check that is treated as exploratory only.

### Level P3b - Independent coordinated cross-layer experiments (E1-E3)
- **E1** is an independent source-informed mechanism simulation: Hummingbird mass/drag and the Dryden recursion are transcribed from the pinned RotorPy source, but the runner is not AdaptiveQuadBench/acados.
- **E2** is an independent physics-grounded mission-energy simulation with source-informed vehicle-scale parameters and a synthetic battery/mission environment. It is not an electrochemical battery model, HIL or flight evidence.
- **E3** is a synthetic but physics-grounded equal-budget fleet-optimization benchmark. The QI backend is CPU Simulated Bifurcation, not quantum hardware or a dedicated Ising machine.

E1-E3 share the manuscript's hypotheses and authority logic but are separate experiments, not one integrated execution of the ARA stack.

### Level P4 - Full package-native benchmark
Not executed in the archived sandbox. `acados_template/libacados` and several runtime dependencies were unavailable and outbound DNS/network installation was blocked. The repository includes an explicit future-run recipe rather than substituting another optimizer and labeling it “native.”

### Level P5 - HIL / embedded / flight
Not executed. Reported timing is software timing on the execution CPU, not an embedded certification result.

## Why the distinctions matter

Claims are bounded to the strongest evidence level actually achieved. In particular:
- the strong constrained MPC comparator wins the hard SCG-derived compound subset;
- K's HOCBF statement is conditional on its translational surrogate, feasible intervals, command realization and mismatch bound;
- L2 shows that simple scalar-bound recalibration does not transfer K cleanly to pinned-source Dryden/actuator dynamics;
- QNL fails its replacement criterion, so the retained CeNN remains the simpler bounded `tanh`-nonlinear residual;
- E2 pooled calibration is not conditional calibration (headwind/compound coverage ~93%);
- E3 rejects solver advantage for the tested CPU SB-QI backend only.

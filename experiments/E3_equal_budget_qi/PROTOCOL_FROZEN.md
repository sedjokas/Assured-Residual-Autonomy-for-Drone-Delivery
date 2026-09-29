# E3 Frozen Protocol — Equal-Budget Benchmarking of Classical and QUBO/Quantum-Inspired Drone-Fleet Optimization

Frozen before calibration/final outcome inspection: 2026-09-12.

## Scientific target
Test manuscript hypothesis H2 / requirement R2: whether a quantum-inspired QUBO backend provides a preregistered quality-at-deadline advantage after complete encoding, preprocessing, post-processing, feasibility checking, and hardware accounting.

## Optimization problem
Each instance is an energy- and time-window-feasible drone route-bundle set-packing problem. Common preprocessing generates physically feasible candidate route bundles for each drone. A bundle contains 1–4 customer requests and is retained only if the route satisfies request time windows and the drone's energy reserve constraint. The discrete optimization chooses at most one bundle per drone and serves each request at most once.

Objective: maximize total request priority/value minus route energy and duration costs.

This bundle formulation is deliberately solver-neutral: the same candidate set and the same physical feasibility validator are supplied to all backends.

## Solver families
1. `MILP-HIGHS`: SciPy/HiGHS binary set-packing model; strong/exact classical baseline subject to the same operational deadline.
2. `GRASP-LS`: anytime randomized greedy + conflict-aware local exchange classical heuristic.
3. `SB-QI`: CPU-only discrete Simulated Bifurcation QUBO backend (quantum-inspired, no quantum hardware), using the published/open-source-style symplectic update equations. The QUBO uses the same objective and pairwise conflict constraints. A common deterministic feasibility repair/augmentation is applied after decoding; its time is included.

No hardware/cloud quantum backend is used because identical end-to-end accounting is not available in this environment.

## Fairness contract
- Identical candidate bundles, objective weights, constraints, and final feasibility validator.
- Identical wall-clock deadline within each scale.
- Common preprocessing time is charged to every backend.
- Backend-specific model/QUBO encoding time is charged to that backend.
- Solver time, decoding, repair, validation, and augmentation are charged.
- Sequential CPU execution on the same host; no concurrent solver runs.
- No backend receives relaxed feasibility constraints.
- Offline long-budget MILP is used only to establish the reference optimum/bound; it is not part of the equal-budget contest.

## Scaling grid and deadlines
- Small: 3 drones, 12 requests, <=25 candidate bundles/drone; end-to-end deadline 0.10 s.
- Medium: 5 drones, 20 requests, <=30 candidate bundles/drone; deadline 0.25 s.
- Large: 7 drones, 28 requests, <=32 candidate bundles/drone; deadline 0.50 s.

Stress grid within each scale:
- time-window tightness: loose / tight;
- protected battery reserve: standard (15%) / high (30%).

Final confirmation: 12 untouched instances per scale × tightness × reserve cell = 144 instances total.
Final seeds: deterministic range beginning at 41000; calibration uses separate seeds beginning at 31000.

## QUBO construction
For binary bundle variables z, minimize

    H(z) = -sum_i w_i z_i + A sum_(i,j in conflict) z_i z_j

with normalized bundle weights and A = 1.20. Because max normalized single-bundle weight is 1, A>1 makes any isolated conflict energetically removable in the exact QUBO ground state. Off-diagonal symmetric Q entries use A/2 so z^T Q z counts each conflict once.

## SB-QI calibration (calibration seeds only)
Predeclared grid:
- engine: ballistic / discrete;
- pressure slope: 0.01 / 0.02 / 0.04;
- agents: 8 / 16.
Time step fixed at 0.1; no heating. Calibration score is mean repaired optimality gap over calibration instances, with feasible-rate tie-break, then lower mean total time. The selected configuration is frozen before final seeds.

## Primary endpoints
1. Repaired feasible rate by deadline.
2. Objective optimality gap at deadline relative to offline MILP reference.
3. End-to-end wall-clock time including common preprocessing and backend-specific encode/solve/postprocess.
4. Quality-at-deadline scaling with problem size.

Secondary endpoints:
- raw QUBO feasibility before repair;
- time to first returned feasible solution;
- candidate count and conflict density;
- offline reference optimality status/mip gap.

## Preregistered H2 decision rule
H2 is upgraded from OPEN to SUPPORTED only if SB-QI satisfies all of the following on untouched final instances:
1. repaired feasible rate >= 99%;
2. on MEDIUM or LARGE scale, paired mean optimality-gap improvement over GRASP-LS is at least 2.0 percentage points;
3. the paired 95% bootstrap CI for that improvement excludes 0 after Holm correction of the two scale-level Wilcoxon tests (medium and large);
4. SB-QI is not more than 5 percentage points worse in mean gap than GRASP-LS on the other of the two medium/large scales.

If these conditions fail, H2 remains OPEN/NOT SUPPORTED and the manuscript should de-emphasize any implication of QI solver advantage. A negative result is retained.

## Reference and statistical analysis
Offline MILP reference budget: up to 5 s per instance, outside the equal-budget contest. Primary comparisons use only instances with an offline optimal solution or mip gap <= 1e-4. Paired bootstrap CIs (10,000 resamples) and paired Wilcoxon tests are reported. Holm correction is applied across medium and large SB-QI vs GRASP-LS primary comparisons.

## Claim boundary
This is a CPU simulation/optimization benchmark of a quantum-inspired QUBO method. It is not quantum hardware evidence, not a package-native quantum annealer result, and not evidence of generic quantum advantage.

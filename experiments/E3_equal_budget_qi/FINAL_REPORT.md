# E3 Final Report — Equal-Budget Classical vs QUBO/Quantum-Inspired Fleet Optimization

## Outcome
**Frozen H2 decision: NOT SUPPORTED.**

E3 tested the manuscript claim that a quantum-inspired backend can provide a time-to-quality / quality-at-deadline advantage under complete accounting. The benchmark used the same energy/time-window-feasible route-bundle set-packing instances for all solvers and charged common preprocessing, backend encoding, solve, decode/repair, and validation to the operational deadline.

### Frozen SB-QI configuration
`engine=discrete`, `pressure_slope=0.01`, `agents=8`, time step 0.1, no heating, CPU only.

## Mean optimality gap at the equal end-to-end deadline (%)

| scale   |   GRASP-LS |   MILP-HIGHS |   SB-QI |
|:--------|-----------:|-------------:|--------:|
| small   |      0.149 |        0.000 |   1.052 |
| medium  |      0.177 |        0.000 |   2.614 |
| large   |      0.817 |        0.000 |   4.365 |

## Raw solver feasibility rate

| scale   |   GRASP-LS |   MILP-HIGHS |   SB-QI |
|:--------|-----------:|-------------:|--------:|
| large   |      1.000 |        1.000 |   0.250 |
| medium  |      1.000 |        1.000 |   0.229 |
| small   |      1.000 |        1.000 |   0.021 |

## Mean end-to-end runtime (s)

| scale   |   GRASP-LS |   MILP-HIGHS |   SB-QI |
|:--------|-----------:|-------------:|--------:|
| large   |     0.5008 |       0.2865 |  0.5029 |
| medium  |     0.2502 |       0.1719 |  0.2508 |
| small   |     0.0993 |       0.0732 |  0.0991 |

Offline reference quality was optimal / <=1e-4 MIP gap for 100.0% of final instances.

## Primary preregistered SB-QI vs GRASP-LS comparisons

| scale   |   n |   mean_gap_GRASP |   mean_gap_SB |   mean_improvement_pp_GRASP_minus_SB |   median_improvement_pp |   bootstrap95_low |   bootstrap95_high |   wilcoxon_stat |   p_raw |   p_holm |
|:--------|----:|-----------------:|--------------:|-------------------------------------:|------------------------:|------------------:|-------------------:|----------------:|--------:|---------:|
| medium  |  48 |           0.1775 |        2.6139 |                              -2.4364 |                 -2.3361 |           -3.0794 |            -1.8368 |          8.0000 |  0.0000 |   0.0000 |
| large   |  48 |           0.8173 |        4.3649 |                              -3.5476 |                 -3.5227 |           -4.2600 |            -2.8715 |          5.0000 |  0.0000 |   0.0000 |

The frozen H2 rule required a >=99% repaired-feasibility rate, at least a 2.0 percentage-point mean gap advantage on medium or large instances, a paired CI excluding zero with Holm-corrected significance, and no >5 percentage-point deterioration on the other medium/large scale.

Decision details:

```json
{
  "H2_supported": false,
  "SB_repaired_feasible_rate_min_scale": 1.0,
  "medium_qualifies": false,
  "large_qualifies": false,
  "other_scale_noninferiority_ok": true,
  "decision_rule": "Frozen in PROTOCOL_FROZEN.md"
}
```

## Scientific interpretation
A positive H2 result would justify retaining solver-regime language that treats SB-QI as an empirically advantaged backend for at least one operational regime. A negative result means the architecture may still keep QUBO as a solver-neutral representation/interface, but the manuscript should not imply that the quantum-inspired backend is faster or better than a strong classical heuristic under equal end-to-end accounting.

The exact meaning of this experiment is deliberately narrow. `SB-QI` is a CPU implementation of discrete Simulated Bifurcation update equations applied to the QUBO representation. No quantum processor, quantum annealer, or dedicated Ising machine was used. Therefore this experiment cannot support any claim of quantum hardware advantage.

## Provenance and limitations
- Synthetic-but-physics-grounded drone-fleet instances: candidate bundles satisfy time windows and a protected battery-reserve feasibility check before discrete optimization.
- Route decomposition means E3 benchmarks the fleet assignment/route-bundle selection layer, not a continuous full vehicle-routing problem.
- Common preprocessing is identical across solvers, but SciPy/HiGHS and the Python/Torch solvers have different implementation stacks. Wall-clock comparison is therefore host- and implementation-dependent.
- QUBO repair is deterministic and included in total time. Raw-QUBO feasibility is separately reported to avoid hiding penalty/solver failures.
- The offline MILP reference is outside the equal-budget contest and exists only to establish the quality denominator.
- E3 should be called an equal-budget CPU solver-regime experiment, not package-native quantum optimization evidence.

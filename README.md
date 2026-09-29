# Assured Residual Autonomy (ARA) for Drone Delivery — code and data

Reproducibility package for the manuscript

> K. Kyamakya, T. Benarbia, and S. K. Kasereka, **"Assured Residual Autonomy for Drone Delivery: Coordinated Evidence on Energy Risk, Runtime Safety, and Solver Regimes,"** submitted to *IEEE Access*, 2026.

ARA separates **proposal generation** from **execution authority**: optimizers and learned modules may propose actions, but route/resource contracts, fallback control, runtime assurance and an independent final-command filter decide what is executed. This repository contains the code, frozen protocols, trial-level data and reports behind every result in the paper — including the negative ones.

## Quick start (≈ 10 minutes)

```bash
pip install -r environments/requirements-core.txt
python scripts/verify_checksums.py        # file integrity
python scripts/run_smoke_tests.py         # registry and key decisions
python scripts/reproduce_e1_e3_tables.py  # paper Tables IX-XII, checked against the manuscript
```

Reviewers: see [`REVIEWER_START_HERE.md`](REVIEWER_START_HERE.md). Rerunning campaigns: see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Main results

| Experiment | Question | Result |
|---|---|---|
| **E1** actuator-aware assurance | Does modelling command realization repair the assurance model after transfer failure? | Yes, for severe stuck proposals: Severe Dryden RMSE 0.5264 → 0.5002 m and envelope violation 2.18% → 1.17% vs. the lumped filter; transparent in healthy cases. |
| **E2** risk-bounded energy admission | Does an uncertainty margin protect the energy reserve without excessive conservatism? | Partially: reserve violations among admitted missions 2.11% → 0.112% (0% with online adaptation) on 48,000 untouched missions; coverage only ~93% under headwind/compound. |
| **E3** equal-budget solver benchmark | Does the tested quantum-inspired backend beat classical solvers at equal deadline? | No: large-instance gap MILP 0%, GRASP-LS 0.817%, SB-QI 4.365%. |
| **A-QNL** residual control and assurance chain | Does a bounded CeNN residual help, and can independent assurance contain bad proposals? | CeNN helps in favourable regimes but loses to LQR/MPC; independent layers contain corrupted proposals; the scalar-bound HOCBF fails to transfer (L2). |

## Paper → evidence map

| In the paper | Evidence in this repository |
|---|---|
| Fig. 1-3 (architecture, layers, experimental logic) | `paper/figures/fig01-03_*.png` (diagrams) |
| Table VI (A) | `data/raw/A_synthetic_micro/` |
| Table VII, Fig. 4 (SCG) | `data/raw/SCG_safe_control_gym_derived/` |
| Fig. 5 (D2) | `data/raw/D2_confirmatory/` |
| Table VIII, Fig. 6 (QNL) | `data/raw/QNL_structured_nonlinearity/`, `experiments/QNL_structured_nonlinearity/frozen_models/` |
| Fig. 7 (G) | `data/raw/G_event_gate/` |
| Fig. 8 (H) | `data/raw/H_RTA/` |
| Fig. 9 (I) | `data/raw/I_risk_governor/` |
| Fig. 10 (J, K) | `data/raw/J_projection/`, `data/raw/K_HOCBF/` |
| Fig. 11 (L2) | `data/raw/L2_native_recalibration/` |
| Table IX, Fig. 12 (E1) | `experiments/E1_actuator_aware/results/` |
| Tables X-XI, Fig. 13 (E2) | `experiments/E2_energy_risk/` |
| Table XII, Fig. 14 (E3) | `experiments/E3_equal_budget_qi/` |
| Appendix A (registry) | `EXPERIMENT_REGISTRY.csv` |
| Appendix B (decisive metrics) | `CLAIM_EVIDENCE_MATRIX.csv` |
| Appendix D (claim boundary) | `CLAIM_BOUNDARY.md` |

## Repository structure

```
.
├── README.md, REVIEWER_START_HERE.md, REPRODUCIBILITY.md
├── CLAIM_BOUNDARY.md, PROVENANCE.md          # what is claimed, at which evidence level
├── EXPERIMENT_REGISTRY.csv                   # one row per experiment (A ... QNL, E1-E3)
├── CLAIM_EVIDENCE_MATRIX.csv                 # claim -> experiment -> numbers -> status
├── MANIFEST_SHA256.csv                       # checksums of every file
├── paper/figures/                            # the 14 manuscript figures
├── experiments/<ID>/                         # runners, frozen protocols, E1-E3 full packages
├── data/raw/<ID>/                            # trial-level outputs of A ... QNL
├── data/processed/                           # consolidated manuscript-facing values
├── results/tables/, results/figures/         # regenerated tables and figures
├── supplementary/                            # per-experiment reports (A ... QNL)
├── scripts/                                  # checksum, smoke test and regeneration utilities
├── environments/                             # Python requirements, native-run environment
└── docs/                                     # experiment compendium, native-run recipe
```

## Evidence levels

Results are never pooled into a generic "benchmark" claim. From weakest to strongest:

1. **Synthetic** — A.
2. **Public-benchmark-derived reimplementation** — SCG (Safe-Control-Gym task, not package-native).
3. **AdaptiveQuadBench/RotorPy source-derived** — B-K, QNL.
4. **Pinned RotorPy source core** — L, L2.
5. **Independent coordinated cross-layer experiments** — E1 (source-informed), E2 and E3 (physics-grounded).
6. Package-native AdaptiveQuadBench/acados — **not executed**.
7. HIL, embedded or flight — **not executed**.

Details: [`PROVENANCE.md`](PROVENANCE.md). "Assured" is an architectural term and does not mean certification. The claims this evidence does and does not support are listed in [`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md).

## Requirements

Python ≥ 3.10; `numpy`, `pandas`, `scipy` (≥ 1.9, for HiGHS through `scipy.optimize.milp`), `matplotlib`, `numba`, `torch` — see `environments/requirements-core.txt`. The native-run environment for a future AdaptiveQuadBench/acados confirmation is in `environments/environment-native.yml`.

## Citation

If you use this code or data, please cite the manuscript (see [`CITATION.cff`](CITATION.cff)).

## License

Original code and data: MIT License ([`LICENSE`](LICENSE)). Third-party components keep their own licenses ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

## Contact

Selain K. Kasereka — selain.kasereka@aau.at

# Reviewer start path

The shortest route from the manuscript to the evidence, in about ten minutes.

## 1. Check integrity

```bash
pip install -r environments/requirements-core.txt
python scripts/verify_checksums.py
```

Expected: `OK - all manifest hashes match`.

## 2. Reproduce the headline numbers

```bash
python scripts/run_smoke_tests.py
python scripts/reproduce_e1_e3_tables.py
```

The second script recomputes the E1-E3 values of Tables IX-XII from the frozen outputs and fails loudly if any differs from the paper.

## 3. Read the claim boundary before the results

1. `CLAIM_BOUNDARY.md` — what is and is not claimed (mirrors Appendix D).
2. `EXPERIMENT_REGISTRY.csv` — one row per experiment, with evidence level, code and data paths.
3. `CLAIM_EVIDENCE_MATRIX.csv` — claim → experiment → numbers → status.
4. `PROVENANCE.md` — definitions of the evidence levels.

## 4. Open the three cross-layer experiments

| Paper | Folder | Start with |
|---|---|---|
| Section XIV, Table IX, Fig. 12 (E1) | `experiments/E1_actuator_aware/` | `results/PROTOCOL_FROZEN.md`, `results/FINAL_REPORT.md` |
| Section XV, Tables X-XI, Fig. 13 (E2) | `experiments/E2_energy_risk/` | `PROTOCOL_FROZEN.md`, `FINAL_REPORT.md` |
| Section XVI, Table XII, Fig. 14 (E3) | `experiments/E3_equal_budget_qi/` | `PROTOCOL_FROZEN.md`, `FINAL_REPORT.md`, `H2_DECISION.json` |

## 5. Historical chain (A-QNL)

Reports: `supplementary/SITUATION_*.md` and `docs/EXPERIMENTS_COMPENDIUM.md`. Trial data: `data/raw/<experiment>/`.

E1-E3 are coordinated by common hypotheses and authority logic; they are **not** one integrated package-native, HIL or flight demonstration.

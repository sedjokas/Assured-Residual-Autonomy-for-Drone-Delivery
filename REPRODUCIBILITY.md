# Reproducibility

## Tier 1 — verify the paper's numbers from frozen outputs (minutes)

Requires Python ≥ 3.10 with the packages in `environments/requirements-core.txt`.

```bash
pip install -r environments/requirements-core.txt
python scripts/verify_checksums.py        # integrity of every file in the release
python scripts/run_smoke_tests.py         # registry and key decisions present
python scripts/reproduce_e1_e3_tables.py  # Tables IX-XII (E1-E3), checked against the paper
python scripts/reproduce_paper_tables.py  # historical A-QNL key results
python scripts/reproduce_paper_figures.py # data-driven historical figures
```

`reproduce_e1_e3_tables.py` recomputes 20 values printed in the manuscript from the frozen E1-E3 outputs and exits with an error if any of them differs.

## Tier 2 — rerun a campaign

Each experiment folder under `experiments/` contains its runner, frozen protocol and (where available) its own SHA-256 list. Calibration/validation seeds are always separated from final test seeds.

The runners are kept **exactly as executed**. They write to the absolute paths of the original execution environment (`/mnt/data/...`). To rerun without altering the frozen files, work on a copy:

```bash
mkdir -p rerun && cp experiments/E3_equal_budget_qi/run_e3.py rerun/
sed -i.bak "s|/mnt/data/|$(pwd)/rerun/out_|g" rerun/run_e3.py
python rerun/run_e3.py
```

The same applies to `E2_energy_risk/run_e2.py`. For E1, copy both `e1_fast.py` and `e1_final.py` (the second imports the first from `/mnt/data/e1_fast.py`), apply the same substitution to both, then run `e1_final.py`.

| Experiment | Runner | Main outputs |
|---|---|---|
| E1 | `experiments/E1_actuator_aware/code/e1_final.py` (+ `e1_fast.py`) | `results/final_summary.csv`, `final_paired_statistics.csv` |
| E2 | `experiments/E2_energy_risk/run_e2.py` | `final_summary.csv`, `final_coverage.csv` |
| E3 | `experiments/E3_equal_budget_qi/run_e3.py` | `scale_summary.csv`, `primary_paired_statistics.csv`, `H2_DECISION.json` |
| A-QNL | see `EXPERIMENT_REGISTRY.csv` (`code_path`) | `data/raw/<experiment>/` |

## Tier 3 — full package-native confirmation

Not executed. See `docs/FULL_NATIVE_RUN_RECIPE.md` and `environments/environment-native.yml`. It requires a complete AdaptiveQuadBench environment, the pinned RotorPy submodule and acados, which the original sandbox did not provide.

## Tier 4 — HIL / embedded / flight

Not part of this release. Reported timings are workstation software timings.

## Known gaps (stated, not hidden)

- **SCG:** the original simulation-generation runner was not recovered. `experiments/SCG_safe_control_gym_derived/reproduce_scg_analysis.py` re-analyses the archived trial data; bit-for-bit regeneration of the SCG plant runs is not claimed.
- **Historical runners (A-QNL):** several scripts reference intermediate folders of the original environment (`/mnt/data/...`). Trial-level outputs, frozen protocols and reports are archived for every experiment; missing runners were not reconstructed.
- **L:** the recovered runner executes the pinned RotorPy source core, not package-native AdaptiveQuadBench/acados.
- Some helper modules are duplicated across folders (e.g. `h_common.py`, `i_common.py`, `k_common.py` in `L_native_source_pilot/`) so that each frozen campaign stays self-contained.

## Integrity

`MANIFEST_SHA256.csv` lists the SHA-256 of every file in the release. E1, E2 and E3 additionally keep the checksum lists written when their protocols were frozen (`SHA256SUMS`, `SHA256SUMS.json`, `PROTOCOL_FROZEN.sha256`); their folders are distributed unchanged, with one exception: the three `MANUSCRIPT_INSERT_*.md` drafting excerpts of the manuscript text are not published. These are the only entries of the frozen lists that are absent from this release. All other listed files are present and match their frozen hash, except `experiments/E3_equal_budget_qi/FINAL_REPORT.md`, which was edited after its hash was recorded (the discrepancy is already present in the original archive). Its numbers are identical to `scale_summary.csv` and `primary_paired_statistics.csv`, which do match their frozen hashes and from which `scripts/reproduce_e1_e3_tables.py` recomputes Table XII.

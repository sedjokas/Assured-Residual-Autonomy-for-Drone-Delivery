# SCG — Strong-Baseline Public-Benchmark Check

The archived raw trial/summary/statistics CSV files for the Safe-Control-Gym-derived experiment are included under `../../data/raw/SCG_safe_control_gym_derived/`.

**Reproducibility scope:** the original simulator-generation runner for this independent Safe-Control-Gym-derived reproduction was not recovered from the archived working files. Therefore this repository does **not** claim bit-for-bit regeneration of the plant simulations for SCG. The included `reproduce_scg_analysis.py` deterministically reconstructs the paper-facing summaries and sanity checks from the archived raw trial data.

The constrained MPC subset is a strong comparator and must not be conflated with later simplified LMPC surrogates used in other source-derived studies.

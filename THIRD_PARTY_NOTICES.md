# THIRD_PARTY_NOTICES

This release contains research code and data generated for the Assured Residual Autonomy (ARA) study (historical label: AP-QI-CeNN).

Some code paths reproduce or adapt concepts/equations/parameters from third-party open-source projects. Their original licenses remain applicable.

- **AdaptiveQuadBench** - https://github.com/Dz298/AdaptiveQuadBench - MIT License.
- **RotorPy** - https://github.com/spencerfolk/rotorpy and the benchmark fork/submodule - MIT License.
- **Safe-Control-Gym** - public benchmark source and paper cited in the manuscript; the archived experiment here is an independent reference-model reproduction rather than copied package code.

`experiments/L2_native_transfer/native_validation_l_core.py` contains a source-faithful extraction/reimplementation of pinned RotorPy equations used for transfer testing. Preserve upstream attribution when redistributing.

E3 uses the HiGHS MILP solver through `scipy.optimize.milp` (HiGHS: MIT License).

No third-party trademark or certification status is implied.

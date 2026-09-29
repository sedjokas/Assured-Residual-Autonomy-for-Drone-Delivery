# SITUATION B — provisional frozen snapshot

Status: provisional experimental state preserved before starting Situation C.

Benchmark basis: AdaptiveQuadBench / RotorPy, reproduced independently from the public source configuration available in this conversation. The experiment uses the benchmark's disturbance families (wind, model uncertainty, external force, rotor-efficiency variation, payload) and compares conventional/model-based controllers with residual-learning variants.

CeNN input construction in Situation B:
- Nine primitive inputs m_i arranged as a 3x3 grid: position error (3), velocity error (3), and estimated unexplained acceleration/disturbance (3).
- Temporal augmentation for the D2 variant: [m_i, tau * dm_i/dt, tau^2 * d2m_i/dt2] with tau=0.1 s, giving three 3x3 input planes (27 scalar values).
- Derivatives are causal and filtered, not noncausal finite differences.
- CeNN-V3 is a parameter-matched ablation using repeated current-value planes rather than derivatives.

Provisional Situation-B mean position RMSE results (100 paired trials per condition), as previously reported:
- nominal: Geo 0.059 m; LMPC 0.062 m; CeNN-V3 0.065 m; CeNN-D2 0.069 m
- wind 3 m/s: Geo 0.196 m; LMPC 0.202 m; CeNN-V3 0.080 m; CeNN-D2 0.084 m
- external force step: Geo 0.101 m; LMPC 0.106 m; CeNN-V3 0.068 m; CeNN-D2 0.073 m
- model uncertainty +/-20%: Geo 0.174 m; LMPC 0.170 m; CeNN-V3 0.173 m; CeNN-D2 0.179 m
- latency 40 ms: Geo 0.076 m; LMPC 0.072 m; CeNN-V3 0.074 m; CeNN-D2 0.079 m
- payload 35-50%: Geo 1.824 m; LMPC 2.639 m; CeNN-V3 2.034 m; CeNN-D2 2.034 m
- rotor efficiency +/-30%: Geo 4.319 m; LMPC 4.751 m; CeNN-V3 4.953 m; CeNN-D2 4.961 m
- compound: Geo 1.065 m; LMPC 1.778 m; CeNN-V3 1.115 m; CeNN-D2 1.120 m

Main Situation-B conclusion: temporal derivatives are technically feasible and improve offline residual-fit quality, but the full [m, dm/dt, d2m/dt2] augmentation did not consistently improve closed-loop control. CeNN remained most convincing as a bounded residual for disturbance-rejection regimes (wind and external force), not as a universal replacement for MPC or classical robust control.

Source data folder preserved at: /mnt/data/situation_B_adaptivequadbench/

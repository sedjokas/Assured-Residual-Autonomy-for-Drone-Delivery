# SITUATION A — Safe-Control-Gym-derived benchmark extension

Status: preserved snapshot of the state reached before Situation B.

## Core experiment
- Public-benchmark basis: Safe-Control-Gym-derived 2-D quadrotor trajectory tracking.
- Main comparison: conventional controller(s), compact residual MLP, bounded residual CeNN, and a strong model-based constrained MPC comparator.
- Independent runtime-assurance/fallback semantics retained.
- The experiment was explicitly benchmark-derived rather than claimed as flight or HIL validation.

## Main scientific result
The CeNN residual was useful in several mismatch/disturbance regimes, but the strong constrained MPC remained superior in the difficult compound case. This was treated as a positive architectural result rather than a failure: CeNN was positioned as a bounded, low-authority complement and not as a universal MPC replacement.

Representative result previously frozen:
- constrained MPC compound RMSE approximately 0.176 m;
- CeNN compound RMSE on the paired strong-baseline subset approximately 0.299 m;
- the benchmark study identified regime-dependent CeNN benefit and reinforced the paper's bounded-authority/runtime-assurance framing.

## Status of paper integration
At the end of Situation A, the new benchmark experiment existed, but the final approximately 30-page manuscript integration and the requested final 7 x 20 quality gate had not yet been completed.

## Related preserved data
The earlier public-benchmark files remain under:
`/mnt/data/cenn_public_benchmark_v1/`

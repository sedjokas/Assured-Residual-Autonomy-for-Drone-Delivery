# SITUATION F — Protected current latent + gated history residual

Status: **frozen targeted confirmatory benchmark-derived simulation snapshot**. Situations A–E remain preserved separately.

## Motivation
Situation E showed that a coherent 100-ms lag contains information, but direct concatenation of current and lagged signals into one bottleneck damages the strong current-only representation. Situation F therefore protects the current path and introduces history only as a bounded latent residual.

## Architecture
The exact deterministic Situation-E current-only path is reconstructed and frozen:

`current temporal triplet -> AE encoder 3->64->3 -> normalized current latent -> CeNN`.

A separate history side branch receives only the 100-ms lagged temporal triplet:

`lagged temporal triplet -> history encoder 3->8->3 -> z_history`.

Fusion is:

`z_fused = z_current + g(z_current, z_history) * z_history`, with `0 <= g <= g_max`.

The gate bound was selected on held-out validation seeds 1200–1239 from {0.15, 0.35, 0.60}. The selected value was **g_max=0.35**. The final test used new seeds 2200–2299 only after that selection.

Active inference parameters: current-only **494**; direct-concat E control **496**; dual-branch gated history **574**; matched current-side capacity control **574**.

## Main protocol
- 100 paired test seeds per scenario, 12 s at 100 Hz.
- Eight scenarios: nominal, wind, force-step, model mismatch, 40-ms latency, payload, rotor-efficiency loss, compound stress.
- Primary endpoint: position RMSE after the 1.01-s warm-up.
- Matched-current-side branch isolates added capacity from historical information.
- Shuffled-history control destroys temporal correspondence while preserving the lag distribution.
- A 30-seed 1-s side-branch outage test checks graceful return to the frozen current path.
- This remains an independent benchmark-derived simulator; it is not native full AdaptiveQuadBench/acados, HIL, or flight validation.

## Mean position RMSE [m]

| Scenario   |   CurrentOnly-AE-CeNN-494 |   DirectConcatLag10-AE-CeNN-496 |   DualBranch-GatedHistory |   DualBranch-MatchedCurrentSide |   LQR-outer |
|:-----------|--------------------------:|--------------------------------:|--------------------------:|--------------------------------:|------------:|
| nominal    |                   0.00968 |                         0.00976 |                   0.00969 |                         0.00966 |     0.00964 |
| wind3      |                   0.12048 |                         0.12935 |                   0.1203  |                         0.12086 |     0.14068 |
| force_step |                   0.08852 |                         0.09688 |                   0.08841 |                         0.08879 |     0.09902 |
| model20    |                   0.01468 |                         0.01619 |                   0.01467 |                         0.01472 |     0.01934 |
| latency40  |                   0.01179 |                         0.01219 |                   0.0118  |                         0.01178 |     0.01295 |
| payload50  |                   0.03377 |                         0.03915 |                   0.0337  |                         0.03399 |     0.0489  |
| rotoreff30 |                   0.03589 |                         0.04194 |                   0.03582 |                         0.03605 |     0.04518 |
| compound   |                   0.10918 |                         0.1195  |                   0.10901 |                         0.10947 |     0.1214  |

## Pre-registered outcomes
All three mechanism-level criteria **passed**:

1. **Incremental-history criterion:** gated history has a positive paired 95% CI versus current-only in 6/7 stressed regimes (wind, force, model mismatch, payload, rotor-efficiency, compound). It is slightly worse under 40-ms latency. Nominal degradation is only **0.127%**.
2. **History-specificity criterion:** gated history beats the parameter-matched current-side branch in 6/7 stressed regimes.
3. **Temporal-coherence criterion:** gated history beats shuffled history in 5/7 stressed regimes.

## Effect size
The key result is statistically robust but **small in absolute magnitude**.

Across the seven stressed regimes, gated history improves RMSE by only **0.115% on average versus the already strong current-only controller**. The scenario-level gains versus current-only are approximately 0.153% (wind), 0.126% (force), 0.035% (model mismatch), -0.089% (latency), 0.219% (payload), 0.200% (rotor-efficiency), and 0.163% (compound).

By contrast, the protected dual-branch architecture improves the older direct-concatenation architecture by **9.38% on average across stressed regimes**. This is the main architectural finding: isolating history avoids the representational interference seen in Situation E.

The gated-history model also beats the matched current-side capacity control by **0.422% on average** and shuffled history by **0.136% on average** across stressed regimes. Thus the side branch is not acting only as extra capacity; temporally coherent lag information contributes, although modestly.

## Gate behavior
The learned gate stays conservative. Its mean value over stressed regimes is approximately **0.0494**, only **14.1% of the allowed g_max**. Mean and p95 gates are close in every scenario. Therefore, this experiment supports a **small bounded history correction**, but does **not** yet support the stronger claim that the gate learns pronounced regime-dependent switching.

## Strong-controller comparison
In this benchmark-derived sweep, gated history is about **17.2% better than LQR on average across stressed scenarios**, while LQR remains slightly better nominally. The LMPC-H40 quantity in this study is explicitly a **surrogate diagnostic**, not the constrained MPC implementation from Situation A; it must not be presented as evidence of generic MPC inferiority.

## Graceful fallback
The forced 1-s history-side outage changes whole-episode RMSE by at most **0.030%** in the 30-seed diagnostic. This is expected by construction: side-branch removal returns exactly to the frozen current-only path rather than removing the full residual controller.

## Scientific conclusion
Situation F resolves the mechanism suggested by D and E. **History is useful, but only as a small isolated correction to a protected current-state representation.** The direct-concatenation approach in E damaged the current latent; the dual-branch design removes that interference and recovers a small but repeatable additional benefit from temporally coherent history.

The publication-safe claim is therefore:

> A protected current-state AE-CeNN with a separately encoded, bounded 100-ms history residual consistently preserved the strong current representation and produced small but statistically repeatable improvements in most stressed benchmark-derived regimes. Parameter-matched and shuffled-history controls indicate that the effect is not explained solely by added capacity, while the learned gate remained conservative. The results support selective residual use of recent history, not a claim that temporal conditioning or CeNN universally outperforms model-based control.

## Claim boundary
No flight-validation, HIL, native full AdaptiveQuadBench/acados, embedded real-time, or universal CeNN-superiority claim is made.

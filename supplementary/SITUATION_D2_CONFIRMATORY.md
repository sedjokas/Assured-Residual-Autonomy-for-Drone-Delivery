# D.2-CONFIRMATORY — FINAL INTERPRETATION

Status: frozen post-hoc fairness correction of the confirmatory experiment. Situations A, B, C and D remain preserved.

## Why a fairness correction was required

The first capacity control counted all autoencoder parameters, including decoder parameters. In D2, however, the decoder is used for denoising training but the 3-D encoder bottleneck is what drives the CeNN at inference. Therefore total trainable-parameter matching is not the cleanest control for inference capacity. A second no-history control was constructed with an encoder 3→64→3 whose **active encoder parameters are exactly 451**, matching the D2 history encoder 24→16→3 (also 451). Adding the same 43-parameter CeNN gives **494 active inference parameters for both controllers**.

## 200-seed confirmatory result

| Scenario   |     Geo |   LQR-outer |   LMPC-H40-surrogate |   C: Geo+AE-CeNN |   D2-History-AE-CeNN |   History-MLP |   ActiveCapacityMatched-NoHistory-AE-CeNN |
|:-----------|--------:|------------:|---------------------:|-----------------:|---------------------:|--------------:|------------------------------------------:|
| nominal    | 0.00993 |     0.00964 |              0.01181 |          0.00972 |              0.01325 |       0.01002 |                                   0.01044 |
| wind3      | 0.16997 |     0.14028 |              0.31395 |          0.15314 |              0.13062 |       0.13059 |                                   0.11717 |
| force_step | 0.12062 |     0.09811 |              0.22829 |          0.1107  |              0.09389 |       0.09638 |                                   0.08478 |
| model20    | 0.02403 |     0.01986 |              0.04467 |          0.02142 |              0.01941 |       0.01924 |                                   0.01782 |
| latency40  | 0.01424 |     0.01294 |              0.02092 |          0.01343 |              0.01531 |       0.01291 |                                   0.01313 |
| payload50  | 0.06462 |     0.0497  |              0.13489 |          0.05713 |              0.0415  |       0.0461  |                                   0.04029 |
| rotoreff30 | 0.05967 |     0.04629 |              0.12344 |          0.05461 |              0.04498 |       0.0475  |                                   0.04026 |
| compound   | 0.14948 |     0.12184 |              0.27947 |          0.1387  |              0.11978 |       0.12232 |                                   0.10784 |

### Result 1 — the D2 composite architecture is genuinely better than the small Situation-C architecture

D2 retains a positive paired RMSE confidence interval against C in 6 of the 7 stressed conditions: wind, external force, model mismatch, payload, rotor-efficiency, and compound stress. It is worse under explicit 40 ms latency and worse nominally. This confirms that the larger D2 composite architecture was not a 40-seed accident.

### Result 2 — the history-specific hypothesis is **not confirmed**

The active-capacity-matched current-only AE-CeNN is better than full D2 in **all eight** scenarios. The D2 disadvantage relative to that fair no-history control is:
- nominal: D2 is 26.96% worse (paired mean-difference CI [-0.00301, -0.00262] m; Holm p=1.15e-33).
- wind3: D2 is 11.48% worse (paired mean-difference CI [-0.01456, -0.01238] m; Holm p=2.57e-33).
- force_step: D2 is 10.74% worse (paired mean-difference CI [-0.00967, -0.00855] m; Holm p=1.15e-33).
- model20: D2 is 8.90% worse (paired mean-difference CI [-0.00185, -0.00132] m; Holm p=3.42e-20).
- latency40: D2 is 16.65% worse (paired mean-difference CI [-0.00236, -0.00201] m; Holm p=3.66e-33).
- payload50: D2 is 3.01% worse (paired mean-difference CI [-0.00140, -0.00102] m; Holm p=2.33e-22).
- rotoreff30: D2 is 11.73% worse (paired mean-difference CI [-0.00505, -0.00441] m; Holm p=1.15e-33).
- compound: D2 is 11.07% worse (paired mean-difference CI [-0.01275, -0.01114] m; Holm p=1.25e-33).

This is the central confirmatory conclusion. The improvement of D2 over the original 120-parameter C architecture cannot be attributed to temporal history. A simpler current-only encoder with the same active inference capacity performs better.

### Result 3 — temporally coherent history is still informative, but the full 8-metadata representation is over-complete

D2 strongly outperforms the shuffled-history control in all seven stressed scenarios, so destroying temporal coherence is harmful. However, lag-only and statistics-only variants each outperform full D2 in every tested scenario. This indicates that **history contains useful information, but the full concatenation of current + two explicit lags + five-number summary is not the best way to expose it to the AE/CeNN**.

On the 100-seed matched comparison against the active-capacity current-only control, lag-only helps modestly in nominal, model mismatch, latency, payload and rotor-efficiency, whereas statistics-only helps most clearly for model mismatch and payload. Neither history subset dominates across all disturbance types.

### Result 4 — the simpler active-capacity control becomes the strongest learned residual candidate

The 494-active-parameter current-only AE-CeNN beats D2 in every condition and is also better than LQR in wind, external force, model mismatch, payload, rotor-efficiency and compound stress in this surrogate. LQR remains better nominally and slightly better under 40 ms latency. This result shifts the design recommendation away from “more temporal metadata” toward **compact current-state conditioning with enough encoder capacity**, with selective history added only where justified.

### Result 5 — latency still exposes the same architecture rule

Full D2 is worse than C and LQR under 40 ms delay. The exact-history + neural microbenchmark had a measured p99 of 4.922 ms in this Python/NumPy implementation, below the 10 ms control period but with nontrivial software jitter. These timings are implementation measurements, not hardware certification.

## Reviewer-safe conclusion

> The 200-seed confirmatory study verifies that the enlarged D2 history-conditioned AE–CeNN architecture improves substantially over the original small one-AE CeNN under several stressed conditions. However, an active-inference-parameter-matched current-only AE–CeNN performs better than D2 in every tested regime. Consequently, the study does **not** support attributing the improvement specifically to temporal-history metadata. Shuffled-history and partial-history ablations show that temporal coherence contains useful information, but the full 8-metadata-per-signal representation is over-complete. The strongest current evidence therefore favors a compact higher-capacity current-state encoder, with selective rather than exhaustive history augmentation.

## Implication for the manuscript

D2 should not be inserted into the paper as the proposed final architecture. If this confirmatory study is included, it is better presented as an ablation that **rejects exhaustive history augmentation** and motivates a simpler resource-bounded encoder. The original AP-QI-CeNN assurance argument is strengthened rather than weakened: learned residual modules should remain replaceable, measured, bounded, and evidence-driven rather than enlarged by assumption.

# SITUATION D — Serial Autoencoders and Lag-Metadata Autoencoding

Status: **provisional frozen exploratory snapshot**.

Situations A, B and C are preserved separately and are not overwritten by D.

## D.1 — three serial autoencoders
Three shared cell-wise denoising AEs are cascaded before the bounded CeNN:

`3 -> 6 -> 2 -> 6 -> 3` × 3.

Total trainable parameters: **274**.

## D.2 — history metadata before one autoencoder
Each of the 27 Situation-C temporal scalar inputs is expanded into 8 causal values:
current value, lag-10, lag-100, and the five-number boxplot summary
[min, Q1, median, Q3, max] over the recent lag-100 window.

At 100 Hz, lag-10 = 0.10 s and lag-100 = 1.00 s.

For computational tractability in this exploratory sweep, the boxplot summary is estimated
from 11 evenly spaced causal samples spanning lags 0,10,...,100. Current, lag-10 and lag-100
themselves are exact.

Each lattice cell therefore has 24 AE inputs. The history AE is:

`24 -> 16 -> 3 -> 16 -> 24`.

Its 3-D bottleneck drives the same bounded 3-plane CeNN.

Total trainable parameters: **966**.

## D.3 — history metadata plus three serial history autoencoders
D.3 combines D.2 with three serial:

`24 -> 16 -> 3 -> 16 -> 24`

autoencoders. The bottleneck of the third stage drives the CeNN.

Total trainable parameters: **2812**.

## Protocol
- 40 held-out paired seeds per main condition: 200–239.
- 15 held-out paired seeds for the forced 1 s residual-outage diagnostic: 200–214.
- Control frequency: 100 Hz.
- Exploratory D episode duration: 2.5 s.
- Same stress families as Situation C: nominal, wind, external force, model mismatch,
  40 ms latency, payload, rotor-efficiency loss, and compound stress.
- This is an independent AdaptiveQuadBench/RotorPy-derived outer-loop reproduction,
  not a native full AdaptiveQuadBench/acados run.

The clean comparisons are therefore within Situation D. In particular, the Situation-C
AE-CeNN architecture is re-run inside D under exactly the same seeds and disturbances.

## Held-out mean position RMSE [m]

| Scenario   |     Geo |   LQR-outer |   LMPC-H40 |   Geo+AE-CeNN |   Geo+D1-3AE-CeNN |   Geo+D2-HistAE-CeNN |   Geo+D3-3HistAE-CeNN |
|:-----------|--------:|------------:|-----------:|--------------:|------------------:|---------------------:|----------------------:|
| nominal    | 0.01474 |     0.01432 |    0.01703 |       0.016   |           0.01624 |              0.01511 |               0.03518 |
| wind3      | 0.14384 |     0.11914 |    0.25085 |       0.14549 |           0.14337 |              0.11524 |               0.15059 |
| force_step | 0.06923 |     0.06015 |    0.08952 |       0.07057 |           0.07001 |              0.05673 |               0.07669 |
| model20    | 0.02765 |     0.02362 |    0.04288 |       0.02855 |           0.02804 |              0.02473 |               0.04102 |
| latency40  | 0.01754 |     0.01641 |    0.02084 |       0.01871 |           0.01908 |              0.01724 |               0.03902 |
| payload50  | 0.07341 |     0.05897 |    0.11957 |       0.07522 |           0.07444 |              0.06578 |               0.09827 |
| rotoreff30 | 0.06038 |     0.04957 |    0.09549 |       0.06177 |           0.06151 |              0.05464 |               0.08112 |
| compound   | 0.11457 |     0.09557 |    0.18497 |       0.11645 |           0.11516 |              0.09458 |               0.12273 |

## D.2 relative to the C architecture re-run

| Scenario   |   D.2 improvement vs C-rerun [%] |   Holm-adjusted p |
|:-----------|---------------------------------:|------------------:|
| nominal    |                             5.58 |           0.12934 |
| wind3      |                            20.79 |           0       |
| force_step |                            19.61 |           0       |
| model20    |                            13.38 |           5e-06   |
| latency40  |                             7.89 |           0.00103 |
| payload50  |                            12.55 |           0       |
| rotoreff30 |                            11.54 |           2.3e-05 |
| compound   |                            18.78 |           0       |

Mean relative RMSE improvement of D.2 over the C-rerun across the eight regimes:
**13.76%**.

The nominal gain does not remain significant after the global Holm correction.
The stressed-regime improvements do in this exploratory sweep.

## Main conclusions

1. **D.2 is the clear winner among D.1–D.3.** It has the lowest RMSE of the three D variants in every tested regime.

2. **History is more useful than AE depth.** D.1 changes the one-AE C-rerun by only
   **0.40% on average** across regimes. Three serial AEs,
   without richer information, mainly add complexity.

3. **D.2 materially improves the C-rerun under stress.** The paired improvement is approximately:
   - wind: 20.8%;
   - external force: 19.6%;
   - model mismatch: 13.4%;
   - latency: 7.9%;
   - payload: 12.5%;
   - rotor-efficiency: 11.5%;
   - compound: 18.8%.

4. **D.3 is counterproductive.** Its average change relative to C-rerun is
   **-44.0%**, i.e. substantially worse.
   Repeated low-dimensional bottlenecks appear to over-compress/over-smooth the dynamic information.

5. **D.2 is not universally superior to model-based control.** LQR remains better in several
   regimes, notably nominal, model-mismatch, latency, payload and rotor-efficiency in this D simulator.
   This preserves the paper's preferred interpretation: the learned residual is a bounded complement,
   not a replacement for strong classical/model-based control.

6. **The dropout behavior remains graceful.** Removing D.2 residual authority for one second causes
   little change in several regimes, but costs more under wind, force and compound disturbances.
   This indicates that the history-conditioned residual is genuinely active where external disturbances
   are present, while the nominal controller remains available as fallback.

7. **Capacity is now a major confound.** D.2 has 966 parameters versus 120 for C.
   A parameter-matched no-history AE, shuffled-history ablation and reduced-width D.2 are required before
   claiming that all of the improvement comes specifically from history metadata rather than additional capacity.

8. **The rotor-efficiency result is not yet a general actuator-fault claim.** D.2 helps the outer-loop
   surrogate here, whereas the earlier Situation-C snapshot showed severe rotor-efficiency failure.
   This difference reinforces the need for a native full RotorPy/AdaptiveQuadBench rerun before publication.

## Reviewer-safe conclusion

> Adding temporal-history metadata before a single compact autoencoder is substantially more useful
> than merely stacking additional autoencoders. In the paired benchmark-derived sweep, the history-conditioned
> D.2 architecture improved the one-AE CeNN residual across most stressed regimes, whereas a three-AE cascade
> without richer inputs offered little benefit and a three-stage history-AE cascade degraded performance.
> The result supports history-aware feature conditioning as a promising direction, but parameter-count and
> native-benchmark ablations are still required before attributing the gain specifically to temporal-history encoding.

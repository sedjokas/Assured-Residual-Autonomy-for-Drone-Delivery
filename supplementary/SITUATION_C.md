# SITUATION C — Autoencoder + CeNN front-end experiment

Status: provisional experimental snapshot, frozen after the 100-seed held-out evaluation.

## Relation to Situation B
Situation C keeps the Situation-B temporal input construction m_i, tau dm_i/dt, tau^2 d2m_i/dt2 and the same AdaptiveQuadBench/RotorPy-derived stress families, but inserts a small denoising autoencoder before the residual CeNN. A CeNN-D2 branch without the autoencoder is rerun inside the same Situation-C simulator and with the same seeds, so the incremental autoencoder effect is paired and directly measurable within Situation C.

## Benchmark basis and provenance
- AdaptiveQuadBench/RotorPy public source model and disturbance families are used as the basis.
- This run is an independent benchmark-derived reproduction, not a native execution of the full AdaptiveQuadBench stack (the complete RotorPy/acados dependency stack is not available in the execution environment).
- The numerical comparison within Situation C is valid because every controller sees the same simulator, scenario realization and holdout seed. Absolute Situation-C values should not be mixed numerically with the older provisional Situation-B table until A/B/C are rerun in one consolidated codebase.

## Inputs
Nine primitive inputs are arranged as a 3x3 grid:
  row 1: position errors e_x,e_y,e_z
  row 2: velocity errors e_vx,e_vy,e_vz
  row 3: causal unexplained-acceleration estimates a_hat_dx,a_hat_dy,a_hat_dz
For every m_i, the CeNN temporal stack is [m_i, tau*m_i_dot, tau^2*m_i_ddot] with tau=0.1 s. Derivatives use a causal 9-sample quadratic regression and low-pass filtering.

## Autoencoder front-end
The autoencoder is shared cell-wise over the 3x3 lattice. For each cell it maps the three temporal features through 3 -> 6 -> 2 -> 6 -> 3 with tanh nonlinearities. Thus the 27 scalar temporal features pass through an 18-scalar bottleneck before being reconstructed to a denoised 27-scalar tensor that is supplied to the same 3-input-plane CeNN-D2 topology.
- AE parameters: 77
- CeNN-D2 parameters: 43
- AE+CeNN total: 120
- compact MLP-D2 comparator: 127
The AE is first trained as a denoising reconstruction model, then frozen while the downstream CeNN is trained on the reconstructed temporal tensor.

## Residual authority contract
All learned residuals use a common conservative authority multiplier rho=0.35, after a small pilot sweep. The resulting effective acceleration residual is bounded to about +/-1.05 m/s^2. Pilot seeds are not used in the final holdout test.

## Data split
- calibration/training episodes: distinct seeds in the 100-119 range on nominal, wind, force and model-mismatch conditions;
- pilot authority sweep: only a few low-numbered pilot seeds;
- final held-out evaluation: 100 paired seeds, 200-299, for every scenario and controller.

## Controllers
- Geo
- LQR-outer
- LMPC-H40 (outer-loop predictive model-based comparator)
- Geo+MLP-D2
- Geo+CeNN-V3 (current-value-only temporal ablation)
- Geo+CeNN-D2 (Situation-B-style derivative-aware CeNN)
- Geo+AE-CeNN (Situation-C proposal)
- Geo+AE-CeNN dropout (1 s forced residual outage)

## Held-out mean position RMSE [m]
Scenario       Geo      LMPC-H40  MLP-D2   CeNN-V3  CeNN-D2  AE-CeNN  AE-CeNN dropout
nominal        0.03295  0.03368   0.02400  0.03610   0.02870  0.02435  0.02589
wind3          0.14010  0.14710   0.10499  0.10650   0.10361  0.10298  0.10940
force_step     0.06895  0.07165   0.05330  0.05926   0.05492  0.05319  0.06011
model20        0.08488  0.11059   0.06187  0.06907   0.06411  0.05877  0.06360
latency40      0.03385  0.03314   0.06609  0.09095   0.07902  0.07309  0.06507
payload50      0.20169  0.27940   0.17422  0.18654   0.17434  0.16931  0.18077
rotoreff30     4.80983  4.79999   4.70274  4.56125   4.54401  4.69482  4.70408
compound       0.24180  0.26171   0.19655  0.19845   0.19141  0.18708  0.19973

## Main paired conclusions
Relative to the same CeNN-D2 without an AE, AE-CeNN reduces mean RMSE by about:
- 15.2% nominal;
- 0.6% wind (not significant after Holm correction);
- 3.1% external force;
- 8.3% model mismatch;
- 7.5% latency stress, although both learned residuals are still substantially worse than Geo/MPC in this regime;
- 2.9% payload;
- 2.3% compound;
- it is about 3.3% worse in the severe rotor-efficiency case and does not change the 78% success rate.

Relative to Geo, AE-CeNN improves RMSE by roughly 16-31% in nominal, wind, force, model-mismatch, payload and compound conditions, but degrades strongly under the explicit 40 ms command-delay test. This delay result is architecturally important: the AE does not remove the need for deadline monitoring and deterministic fallback.

Against the compact MLP-D2, AE-CeNN is essentially tied on force, slightly worse nominally and under latency, but better under wind, model mismatch, payload and compound uncertainty. This supports a regime-dependent rather than universal advantage claim.

The forced 1 s AE-CeNN outage produces moderate degradation in most conditions while the nominal controller continues to operate. Under the latency stress, dropout actually improves the result, reinforcing the need for the runtime assurance/governor to withdraw residual authority when timing assumptions are violated.

## Autoencoder interpretation
The front-end does not win because it makes a better pointwise residual predictor: the frozen AE slightly worsens held-out one-step residual-fit RMSE compared with the direct CeNN. Its useful effect is instead consistent with regularization/denoising of the derivative-rich temporal stack. On held-out calibration data, the reconstructed first- and second-derivative planes have about 9-10% lower standard deviation than the normalized raw planes, while the current-value plane is changed much less.

## Runtime microbenchmark on this CPU / compiled NumPy-Numba path
- MLP-D2 residual: about 1.15 us/call
- CeNN-D2 residual: about 2.49 us/call
- AE-CeNN residual: about 7.30 us/call
These are software microbenchmarks only, not hardware claims. The AE front-end adds roughly 3x CeNN-only inference time in this implementation but remains far below a 10-20 ms control-cycle budget.

## Scientific interpretation
Situation C strengthens the architectural hypothesis in a qualified way: a small denoising AE can make derivative-augmented CeNN residual control more robust in several uncertainty regimes, especially model mismatch and compound disturbances. It does not make CeNN universally superior, does not solve actuator-efficiency failures, and is actively harmful if residual authority is retained through significant command latency. The strongest reviewer-safe claim is therefore that AE+CeNN is a promising bounded residual module whose authority should be conditional on timing and confidence, not a replacement for strong model-based or safety controllers.

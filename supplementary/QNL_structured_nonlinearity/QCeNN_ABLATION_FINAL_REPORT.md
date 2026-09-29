# Structured Quadratic CeNN Nonlinearity Ablation — FINAL REPORT

## Question

This experiment tests a single proposed change: replace the linear Chua–Yang-style CeNN state block in the frozen current-only AE–CeNN residual path by a bounded structured second-order block inspired by

`Xdot = -X + B Y + Bxx XX + Cxy XY + D U`.

The tested implementation deliberately bounds the nonlinear state factors: `xbar=tanh(X/2)` and `Y=tanh(X)`. A shared 3x3 local template acts on `xbar*xbar` (XX); the full variant adds a second 3x3 local template on `xbar*Y` (XY). This preserves local CeNN structure and avoids an unconstrained quadratic growth term that could dominate the `-X` leakage.

## Experimental discipline

- Same source-derived dynamics, trajectory, nominal controller, feature normalization, training corpus and ideal residual targets.
- Same 3→64→3 encoder for the three primary models.
- Same four CeNN relaxation steps, integration step 0.32, output tanh, residual limit and authority 0.35.
- Five initialization replicates per architecture.
- A documented pre-final protocol amendment selected the **median** model by held-out offline residual MSE on seeds 14000–14039; it did not choose the best replicate and occurred before any final seed was run.
- Final confirmation: 200 untouched paired seeds 16000–16199 in all eight scenarios.
- Secondary 515-parameter linear capacity control to check whether additional parameter count alone explains an effect.
- Zero-shot pinned RotorPy-source-core spot check: 10 untouched paired seeds 18000–18009 for nominal, native Dryden wind and rotor-efficiency. No retuning.

## Models

| architecture            |   active_inference_params |
|:------------------------|--------------------------:|
| LinearCeNN-494          |                       494 |
| LinearCeNN-Capacity-515 |                       515 |
| QCeNN-XX-503            |                       503 |
| QCeNN-XXXY-512          |                       512 |

## Frozen replicate selection

| architecture            |   replicate_seed |   validation_offline_normalized_mse | selected_median_validation_model   |
|:------------------------|-----------------:|------------------------------------:|:-----------------------------------|
| LinearCeNN-494          |               12 |                          0.00760386 | False                              |
| LinearCeNN-494          |              112 |                          0.00543235 | False                              |
| LinearCeNN-494          |              212 |                          0.00748162 | False                              |
| LinearCeNN-494          |              312 |                          0.00630722 | False                              |
| LinearCeNN-494          |              412 |                          0.00713933 | True                               |
| QCeNN-XX-503            |               12 |                          0.00754703 | False                              |
| QCeNN-XX-503            |              112 |                          0.0054377  | False                              |
| QCeNN-XX-503            |              212 |                          0.00747013 | False                              |
| QCeNN-XX-503            |              312 |                          0.0062839  | False                              |
| QCeNN-XX-503            |              412 |                          0.00712393 | True                               |
| QCeNN-XXXY-512          |               12 |                          0.00747472 | False                              |
| QCeNN-XXXY-512          |              112 |                          0.00544661 | False                              |
| QCeNN-XXXY-512          |              212 |                          0.00743851 | False                              |
| QCeNN-XXXY-512          |              312 |                          0.00624204 | False                              |
| QCeNN-XXXY-512          |              412 |                          0.00709204 | True                               |
| LinearCeNN-Capacity-515 |               12 |                          0.0136528  | False                              |
| LinearCeNN-Capacity-515 |              112 |                          0.0048055  | False                              |
| LinearCeNN-Capacity-515 |              212 |                          0.00711433 | True                               |
| LinearCeNN-Capacity-515 |              312 |                          0.00441973 | False                              |
| LinearCeNN-Capacity-515 |              412 |                          0.00765123 | False                              |

## Final source-derived mean results

| scenario   | architecture            |   n |   rmse_mean |   rmse_sd |   p95_mean |   max_mean |   excursion_gt1m_mean_pct |   control_effort_mean |   command_saturation_mean_pct |   residual_rms_mean |   latent_max |   latent_p999_mean |   finite_rate |   residual_bound_violation_mean_pct |
|:-----------|:------------------------|----:|------------:|----------:|-----------:|-----------:|--------------------------:|----------------------:|------------------------------:|--------------------:|-------------:|-------------------:|--------------:|------------------------------------:|
| nominal    | LinearCeNN-494          | 200 |   0.0076373 | 0.000614  |  0.0111311 |  0.0120268 |                         0 |              0.671003 |                             0 |           0.128538  |     0.733665 |           0.622521 |             1 |                                   0 |
| wind3      | LinearCeNN-494          | 200 |   0.114465  | 0.0118508 |  0.150716  |  0.156507  |                         0 |              0.982588 |                             0 |           0.292951  |     1.07893  |           0.821139 |             1 |                                   0 |
| force_step | LinearCeNN-494          | 200 |   0.0867963 | 0.0054147 |  0.165898  |  0.170102  |                         0 |              0.872591 |                             0 |           0.220928  |     1.10505  |           0.807298 |             1 |                                   0 |
| model20    | LinearCeNN-494          | 200 |   0.0124316 | 0.0042887 |  0.0196132 |  0.0208102 |                         0 |              0.682287 |                             0 |           0.14096   |     0.75076  |           0.635679 |             1 |                                   0 |
| latency40  | LinearCeNN-494          | 200 |   0.0072358 | 0.0006666 |  0.0118421 |  0.013039  |                         0 |              0.68321  |                             0 |           0.176441  |     0.766114 |           0.644113 |             1 |                                   0 |
| payload50  | LinearCeNN-494          | 200 |   0.0318254 | 0.004224  |  0.0474579 |  0.0497285 |                         0 |              0.977173 |                             0 |           0.196877  |     0.780212 |           0.685984 |             1 |                                   0 |
| rotoreff30 | LinearCeNN-494          | 200 |   0.0350797 | 0.0194395 |  0.0518305 |  0.0538129 |                         0 |              0.912747 |                             0 |           0.184574  |     0.863874 |           0.744899 |             1 |                                   0 |
| compound   | LinearCeNN-494          | 200 |   0.10643   | 0.0170995 |  0.173936  |  0.183871  |                         0 |              1.07841  |                             0 |           0.265224  |     1.08743  |           0.861715 |             1 |                                   0 |
| nominal    | QCeNN-XX-503            | 200 |   0.0075527 | 0.0006149 |  0.0110423 |  0.0119392 |                         0 |              0.670989 |                             0 |           0.128292  |     0.755891 |           0.643951 |             1 |                                   0 |
| wind3      | QCeNN-XX-503            | 200 |   0.114509  | 0.0118533 |  0.150764  |  0.156557  |                         0 |              0.982569 |                             0 |           0.292657  |     1.088    |           0.833011 |             1 |                                   0 |
| force_step | QCeNN-XX-503            | 200 |   0.0868016 | 0.0054254 |  0.165928  |  0.170145  |                         0 |              0.872596 |                             0 |           0.220743  |     1.11905  |           0.821394 |             1 |                                   0 |
| model20    | QCeNN-XX-503            | 200 |   0.0123983 | 0.0043143 |  0.01956   |  0.020763  |                         0 |              0.68228  |                             0 |           0.140702  |     0.775604 |           0.65731  |             1 |                                   0 |
| latency40  | QCeNN-XX-503            | 200 |   0.0072011 | 0.0006628 |  0.011791  |  0.0129889 |                         0 |              0.683112 |                             0 |           0.175995  |     0.783194 |           0.665547 |             1 |                                   0 |
| payload50  | QCeNN-XX-503            | 200 |   0.0318774 | 0.0042273 |  0.0474594 |  0.0497274 |                         0 |              0.977226 |                             0 |           0.196586  |     0.807918 |           0.707793 |             1 |                                   0 |
| rotoreff30 | QCeNN-XX-503            | 200 |   0.0351167 | 0.0194438 |  0.0518723 |  0.0538479 |                         0 |              0.912753 |                             0 |           0.184326  |     0.89352  |           0.768434 |             1 |                                   0 |
| compound   | QCeNN-XX-503            | 200 |   0.106458  | 0.0170968 |  0.173975  |  0.183894  |                         0 |              1.07838  |                             0 |           0.264895  |     1.10475  |           0.877824 |             1 |                                   0 |
| nominal    | QCeNN-XXXY-512          | 200 |   0.0074041 | 0.0006162 |  0.0108847 |  0.0117834 |                         0 |              0.670954 |                             0 |           0.127777  |     0.793397 |           0.682124 |             1 |                                   0 |
| wind3      | QCeNN-XXXY-512          | 200 |   0.114593  | 0.0118576 |  0.150861  |  0.156652  |                         0 |              0.982532 |                             0 |           0.292084  |     1.09766  |           0.852253 |             1 |                                   0 |
| force_step | QCeNN-XXXY-512          | 200 |   0.0868136 | 0.0054472 |  0.165997  |  0.170225  |                         0 |              0.872602 |                             0 |           0.220367  |     1.13685  |           0.845457 |             1 |                                   0 |
| model20    | QCeNN-XXXY-512          | 200 |   0.0123409 | 0.0043597 |  0.0194691 |  0.0206809 |                         0 |              0.682257 |                             0 |           0.140178  |     0.818134 |           0.696055 |             1 |                                   0 |
| latency40  | QCeNN-XXXY-512          | 200 |   0.0071473 | 0.0006573 |  0.0117159 |  0.0129042 |                         0 |              0.682897 |                             0 |           0.175108  |     0.817994 |           0.703639 |             1 |                                   0 |
| payload50  | QCeNN-XXXY-512          | 200 |   0.031967  | 0.0042321 |  0.0474722 |  0.0497376 |                         0 |              0.977311 |                             0 |           0.196053  |     0.856511 |           0.749944 |             1 |                                   0 |
| rotoreff30 | QCeNN-XXXY-512          | 200 |   0.0351749 | 0.0194494 |  0.0519338 |  0.0539025 |                         0 |              0.91275  |                             0 |           0.183824  |     0.94537  |           0.81092  |             1 |                                   0 |
| compound   | QCeNN-XXXY-512          | 200 |   0.106509  | 0.0170893 |  0.17404   |  0.183936  |                         0 |              1.07834  |                             0 |           0.26425   |     1.12917  |           0.904518 |             1 |                                   0 |
| nominal    | LinearCeNN-Capacity-515 | 200 |   0.0048993 | 0.0002165 |  0.0065063 |  0.0114112 |                         0 |              0.665405 |                             0 |           0.0739343 |     1.34117  |           1.18161  |             1 |                                   0 |
| wind3      | LinearCeNN-Capacity-515 | 200 |   0.148915  | 0.011915  |  0.185518  |  0.191781  |                         0 |              0.982629 |                             0 |           0.128741  |     1.86046  |           1.62703  |             1 |                                   0 |
| force_step | LinearCeNN-Capacity-515 | 200 |   0.102787  | 0.0037923 |  0.191099  |  0.199761  |                         0 |              0.872681 |                             0 |           0.138324  |     1.85777  |           1.6021   |             1 |                                   0 |
| model20    | LinearCeNN-Capacity-515 | 200 |   0.0161958 | 0.0077491 |  0.0220467 |  0.0250694 |                         0 |              0.677974 |                             0 |           0.0794902 |     1.55959  |           1.25128  |             1 |                                   0 |
| latency40  | LinearCeNN-Capacity-515 | 200 |   0.0068558 | 0.0002299 |  0.0095057 |  0.0118956 |                         0 |              0.666901 |                             0 |           0.0766569 |     1.35803  |           1.18898  |             1 |                                   0 |
| payload50  | LinearCeNN-Capacity-515 | 200 |   0.0487138 | 0.0055092 |  0.063475  |  0.0650264 |                         0 |              0.986302 |                             0 |           0.105901  |     1.6449   |           1.48526  |             1 |                                   0 |
| rotoreff30 | LinearCeNN-Capacity-515 | 200 |   0.0442958 | 0.022517  |  0.0648331 |  0.0671433 |                         0 |              0.911995 |                             0 |           0.110136  |     1.79739  |           1.46615  |             1 |                                   0 |
| compound   | LinearCeNN-Capacity-515 | 200 |   0.126888  | 0.0190915 |  0.196885  |  0.207025  |                         0 |              1.07645  |                             0 |           0.158002  |     1.98848  |           1.68212  |             1 |                                   0 |

## Key RMSE effects

| scenario   |   LinearCeNN-494 |   QCeNN-XX-503 |   QCeNN-XXXY-512 |   QXX_improvement_vs_linear_pct |   QXXXY_improvement_vs_linear_pct |
|:-----------|-----------------:|---------------:|-----------------:|--------------------------------:|----------------------------------:|
| nominal    |         0.007637 |       0.007553 |         0.007404 |                        1.1082   |                          3.05308  |
| wind3      |         0.114465 |       0.114509 |         0.114593 |                       -0.038439 |                         -0.112211 |
| force_step |         0.086796 |       0.086802 |         0.086814 |                       -0.006157 |                         -0.019998 |
| model20    |         0.012432 |       0.012398 |         0.012341 |                        0.267888 |                          0.729535 |
| latency40  |         0.007236 |       0.007201 |         0.007147 |                        0.479405 |                          1.22308  |
| payload50  |         0.031825 |       0.031877 |         0.031967 |                       -0.163296 |                         -0.44474  |
| rotoreff30 |         0.03508  |       0.035117 |         0.035175 |                       -0.105398 |                         -0.271309 |
| compound   |         0.10643  |       0.106458 |         0.106509 |                       -0.026923 |                         -0.074752 |

Positive percentages mean the quadratic model has lower RMSE than the baseline linear CeNN.

## Paired statistics

| scenario   | proposal       | comparator              |   n |   comparator_rmse |   proposal_rmse |   improvement_m |   improvement_pct |   ci95_low_m |   ci95_high_m |   wilcoxon_p |   proposal_better_fraction |   excursion_delta_pp |   saturation_delta_pp |   residual_rms_delta |   holm_p_stressed |
|:-----------|:---------------|:------------------------|----:|------------------:|----------------:|----------------:|------------------:|-------------:|--------------:|-------------:|---------------------------:|---------------------:|----------------------:|---------------------:|------------------:|
| nominal    | QCeNN-XX-503   | LinearCeNN-494          | 200 |        0.0076373  |      0.00755267 |      8.464e-05  |        1.1082     |   8.373e-05  |    8.555e-05  |    0         |                      1     |                    0 |                     0 |          -0.00024554 |       nan         |
| nominal    | QCeNN-XXXY-512 | LinearCeNN-494          | 200 |        0.0076373  |      0.00740413 |      0.00023317 |        3.05308    |   0.00023041 |    0.0002359  |    0         |                      1     |                    0 |                     0 |          -0.00076118 |       nan         |
| nominal    | QCeNN-XXXY-512 | QCeNN-XX-503            | 200 |        0.00755267 |      0.00740413 |      0.00014854 |        1.96668    |   0.00014672 |    0.00015038 |    0         |                      1     |                    0 |                     0 |          -0.00051564 |       nan         |
| nominal    | QCeNN-XXXY-512 | LinearCeNN-Capacity-515 | 200 |        0.00489929 |      0.00740413 |     -0.00250484 |      -51.1266     |  -0.00258059 |   -0.0024293  |    0         |                      0     |                    0 |                     0 |           0.0538423  |       nan         |
| wind3      | QCeNN-XX-503   | LinearCeNN-494          | 200 |        0.114465   |      0.114509   |     -4.4e-05    |       -0.0384387  |  -5.824e-05  |   -2.994e-05  |    0         |                      0.315 |                    0 |                     0 |          -0.00029412 |         1e-08     |
| wind3      | QCeNN-XXXY-512 | LinearCeNN-494          | 200 |        0.114465   |      0.114593   |     -0.00012844 |       -0.112211   |  -0.00016739 |   -8.855e-05  |    0         |                      0.285 |                    0 |                     0 |          -0.00086726 |         0         |
| wind3      | QCeNN-XXXY-512 | QCeNN-XX-503            | 200 |        0.114509   |      0.114593   |     -8.444e-05  |       -0.0737434  |  -0.00010942 |   -5.808e-05  |    0         |                      0.285 |                    0 |                     0 |          -0.00057313 |         0         |
| wind3      | QCeNN-XXXY-512 | LinearCeNN-Capacity-515 | 200 |        0.148915   |      0.114593   |      0.0343222  |       23.0481     |   0.0334898  |    0.0351542  |    0         |                      1     |                    0 |                     0 |           0.163343   |         0         |
| force_step | QCeNN-XX-503   | LinearCeNN-494          | 200 |        0.0867963  |      0.0868016  |     -5.34e-06   |       -0.00615706 |  -1.686e-05  |    6.62e-06   |    0.101788  |                      0.415 |                    0 |                     0 |          -0.0001854  |         0.101788  |
| force_step | QCeNN-XXXY-512 | LinearCeNN-494          | 200 |        0.0867963  |      0.0868136  |     -1.736e-05  |       -0.0199976  |  -4.904e-05  |    1.54e-05   |    0.0699935 |                      0.43  |                    0 |                     0 |          -0.00056076 |         0.0699935 |
| force_step | QCeNN-XXXY-512 | QCeNN-XX-503            | 200 |        0.0868016  |      0.0868136  |     -1.201e-05  |       -0.0138397  |  -3.224e-05  |    9.04e-06   |    0.0582643 |                      0.425 |                    0 |                     0 |          -0.00037535 |         0.0582643 |
| force_step | QCeNN-XXXY-512 | LinearCeNN-Capacity-515 | 200 |        0.102787   |      0.0868136  |      0.0159737  |       15.5405     |   0.0153203  |    0.0166206  |    0         |                      1     |                    0 |                     0 |           0.0820434  |         0         |
| model20    | QCeNN-XX-503   | LinearCeNN-494          | 200 |        0.0124316  |      0.0123983  |      3.33e-05   |        0.267888   |   2.897e-05  |    3.768e-05  |    0         |                      0.84  |                    0 |                     0 |          -0.00025833 |         0         |
| model20    | QCeNN-XXXY-512 | LinearCeNN-494          | 200 |        0.0124316  |      0.0123409  |      9.069e-05  |        0.729535   |   7.846e-05  |    0.00010278 |    0         |                      0.84  |                    0 |                     0 |          -0.00078199 |         0         |
| model20    | QCeNN-XXXY-512 | QCeNN-XX-503            | 200 |        0.0123983  |      0.0123409  |      5.739e-05  |        0.462887   |   4.962e-05  |    6.516e-05  |    0         |                      0.84  |                    0 |                     0 |          -0.00052366 |         0         |
| model20    | QCeNN-XXXY-512 | LinearCeNN-Capacity-515 | 200 |        0.0161958  |      0.0123409  |      0.00385486 |       23.8017     |   0.00333633 |    0.00438073 |    0         |                      0.815 |                    0 |                     0 |           0.060688   |         0         |
| latency40  | QCeNN-XX-503   | LinearCeNN-494          | 200 |        0.00723579 |      0.0072011  |      3.469e-05  |        0.479405   |   2.966e-05  |    3.925e-05  |    0         |                      0.935 |                    0 |                     0 |          -0.00044612 |         0         |
| latency40  | QCeNN-XXXY-512 | LinearCeNN-494          | 200 |        0.00723579 |      0.00714729 |      8.85e-05   |        1.22308    |   7.387e-05  |    0.00010217 |    0         |                      0.925 |                    0 |                     0 |          -0.00133317 |         0         |
| latency40  | QCeNN-XXXY-512 | QCeNN-XX-503            | 200 |        0.0072011  |      0.00714729 |      5.381e-05  |        0.747261   |   4.381e-05  |    6.272e-05  |    0         |                      0.915 |                    0 |                     0 |          -0.00088705 |         0         |
| latency40  | QCeNN-XXXY-512 | LinearCeNN-Capacity-515 | 200 |        0.00685577 |      0.00714729 |     -0.00029152 |       -4.25212    |  -0.00038223 |   -0.00020192 |    1e-08     |                      0.33  |                    0 |                     0 |           0.0984508  |         1e-08     |
| payload50  | QCeNN-XX-503   | LinearCeNN-494          | 200 |        0.0318254  |      0.0318774  |     -5.197e-05  |       -0.163296   |  -5.286e-05  |   -5.108e-05  |    0         |                      0     |                    0 |                     0 |          -0.00029075 |         0         |
| payload50  | QCeNN-XXXY-512 | LinearCeNN-494          | 200 |        0.0318254  |      0.031967   |     -0.00014154 |       -0.44474    |  -0.00014409 |   -0.00013899 |    0         |                      0     |                    0 |                     0 |          -0.0008238  |         0         |
| payload50  | QCeNN-XXXY-512 | QCeNN-XX-503            | 200 |        0.0318774  |      0.031967   |     -8.957e-05  |       -0.280985   |  -9.124e-05  |   -8.791e-05  |    0         |                      0     |                    0 |                     0 |          -0.00053305 |         0         |
| payload50  | QCeNN-XXXY-512 | LinearCeNN-Capacity-515 | 200 |        0.0487138  |      0.031967   |      0.0167469  |       34.3781     |   0.0165287  |    0.016963   |    0         |                      1     |                    0 |                     0 |           0.090152   |         0         |
| rotoreff30 | QCeNN-XX-503   | LinearCeNN-494          | 200 |        0.0350797  |      0.0351167  |     -3.697e-05  |       -0.105398   |  -3.941e-05  |   -3.45e-05   |    0         |                      0.04  |                    0 |                     0 |          -0.00024813 |         0         |
| rotoreff30 | QCeNN-XXXY-512 | LinearCeNN-494          | 200 |        0.0350797  |      0.0351749  |     -9.517e-05  |       -0.271309   |  -0.00010214 |   -8.792e-05  |    0         |                      0.05  |                    0 |                     0 |          -0.0007492  |         0         |
| rotoreff30 | QCeNN-XXXY-512 | QCeNN-XX-503            | 200 |        0.0351167  |      0.0351749  |     -5.82e-05   |       -0.165736   |  -6.282e-05  |   -5.344e-05  |    0         |                      0.065 |                    0 |                     0 |          -0.00050106 |         0         |
| rotoreff30 | QCeNN-XXXY-512 | LinearCeNN-Capacity-515 | 200 |        0.0442958  |      0.0351749  |      0.00912091 |       20.5909     |   0.00862302 |    0.00962311 |    0         |                      0.99  |                    0 |                     0 |           0.0736883  |         0         |
| compound   | QCeNN-XX-503   | LinearCeNN-494          | 200 |        0.10643    |      0.106458   |     -2.865e-05  |       -0.0269226  |  -3.962e-05  |   -1.736e-05  |    2.1e-07   |                      0.315 |                    0 |                     0 |          -0.00032824 |         4.2e-07   |
| compound   | QCeNN-XXXY-512 | LinearCeNN-494          | 200 |        0.10643    |      0.106509   |     -7.956e-05  |       -0.0747525  |  -0.00011007 |   -4.811e-05  |    3.7e-07   |                      0.305 |                    0 |                     0 |          -0.00097397 |         7.3e-07   |
| compound   | QCeNN-XXXY-512 | QCeNN-XX-503            | 200 |        0.106458   |      0.106509   |     -5.091e-05  |       -0.0478169  |  -7.109e-05  |   -3.049e-05  |    5.4e-07   |                      0.305 |                    0 |                     0 |          -0.00064573 |         1.08e-06  |
| compound   | QCeNN-XXXY-512 | LinearCeNN-Capacity-515 | 200 |        0.126888   |      0.106509   |      0.0203793  |       16.0608     |   0.0198749  |    0.0208832  |    0         |                      1     |                    0 |                     0 |           0.106248   |         0         |

## Pre-specified criteria

```json
{
  "q_vs_linear_positive_CI_scenarios": 2,
  "q_vs_linear_mean_stressed_improvement_pct": 0.14708700519840523,
  "q_vs_linear_nominal_degradation_pct": 0.0,
  "q_vs_linear_nominal_improvement_pct": 3.0530841237471167,
  "q_vs_linear_max_excursion_increase_pp": 0.0,
  "q_vs_linear_max_saturation_increase_pp": 0.0,
  "q_vs_linear_pass": false,
  "structure_vs_capacity_positive_CI_scenarios": 6,
  "structure_vs_capacity_mean_stressed_improvement_pct": 18.45256414552809,
  "structure_vs_capacity_pass": true,
  "xy_increment_mean_stressed_improvement_pct": 0.08971817297055791,
  "xy_increment_pass": true,
  "all_missions_finite": true,
  "residual_bound_violations_observed": 0,
  "primary_overall_pass": false
}
```

**Overall confirmatory result: FAIL.**

The XX+XY model does **not** satisfy the pre-specified replacement criterion. Its average effect across the seven stressed source-derived regimes is +0.1471% relative to the linear CeNN. It improves nominal tracking by 3.053% and improves model-mismatch and latency modestly, but it is worse in wind, force-step, payload, rotor-efficiency and compound conditions. The additional XY term also has an average stressed effect of +0.0897% relative to the XX-only model, so the proposed XX+XY extension is not supported as the preferred block.

The 515-parameter linear capacity control performs substantially worse in most stressed regimes despite a lower nominal RMSE. Therefore parameter count alone does not explain the primary baseline's robustness, but this does **not** rescue the quadratic hypothesis: the relevant comparison is QCeNN against the original linear CeNN, and that comparison fails the pre-specified criterion.

## Stability and boundedness diagnostics

All 6400 final controller–mission trials remained finite. The residual output bound was never violated. The maximum observed internal CeNN state magnitudes are reported in the main summary; no post-hoc stability threshold was introduced. The bounded nonlinear construction therefore avoided numerical blow-up in the tested domain, but numerical boundedness here is not a formal stability proof.

## Software latency diagnostic

| architecture            |   batch |   p50_us |   p95_us |   p99_us |   max_us |
|:------------------------|--------:|---------:|---------:|---------:|---------:|
| LinearCeNN-494          |       1 |   55.684 |   70.366 |   99.492 | 1073.98  |
| QCeNN-XX-503            |       1 |   60.541 |   75.346 |   97.78  |  321.775 |
| QCeNN-XXXY-512          |       1 |   67.818 |   83.056 |  105.531 | 1654.37  |
| LinearCeNN-Capacity-515 |       1 |   55.473 |   70.556 |  122.016 | 2238.01  |

These are Python/NumPy batch-1 timings on the current CPU, not embedded or HIL latency measurements.

## Zero-shot pinned RotorPy-source-core spot check

| scenario   | architecture   |   n |   rmse_mean |   rmse_sd |   p95_mean |   max_mean |   latent_max |
|:-----------|:---------------|----:|------------:|----------:|-----------:|-----------:|-------------:|
| nominal    | LinearCeNN-494 |  10 |    0.106307 | 0.0008884 |   0.131754 |   0.133513 |     0.800542 |
| nominal    | QCeNN-XX-503   |  10 |    0.106318 | 0.0008829 |   0.131674 |   0.133432 |     0.825295 |
| nominal    | QCeNN-XXXY-512 |  10 |    0.106318 | 0.0008724 |   0.131519 |   0.133249 |     0.873398 |
| rotoreff30 | LinearCeNN-494 |  10 |    0.222495 | 0.0788577 |   0.28217  |   0.284105 |     1.04553  |
| rotoreff30 | QCeNN-XX-503   |  10 |    0.22251  | 0.0788025 |   0.282205 |   0.284144 |     1.0611   |
| rotoreff30 | QCeNN-XXXY-512 |  10 |    0.222545 | 0.0787182 |   0.282292 |   0.284224 |     1.08211  |
| wind3      | LinearCeNN-494 |  10 |    0.844441 | 0.195173  |   1.08051  |   1.08737  |     1.06357  |
| wind3      | QCeNN-XX-503   |  10 |    0.844375 | 0.195064  |   1.08043  |   1.08729  |     1.08312  |
| wind3      | QCeNN-XXXY-512 |  10 |    0.844284 | 0.19489   |   1.08027  |   1.08715  |     1.11194  |

| scenario   | proposal       | comparator     |   n |   improvement_m |   improvement_pct |   proposal_better_fraction |
|:-----------|:---------------|:---------------|----:|----------------:|------------------:|---------------------------:|
| nominal    | QCeNN-XX-503   | LinearCeNN-494 |  10 |      -1.09e-05  |        -0.0102227 |                        0.4 |
| nominal    | QCeNN-XXXY-512 | LinearCeNN-494 |  10 |      -1.09e-05  |        -0.0102316 |                        0.5 |
| wind3      | QCeNN-XX-503   | LinearCeNN-494 |  10 |       6.58e-05  |         0.0077929 |                        0.6 |
| wind3      | QCeNN-XXXY-512 | LinearCeNN-494 |  10 |       0.0001563 |         0.0185052 |                        0.6 |
| rotoreff30 | QCeNN-XX-503   | LinearCeNN-494 |  10 |      -1.51e-05  |        -0.0067969 |                        0.2 |
| rotoreff30 | QCeNN-XXXY-512 | LinearCeNN-494 |  10 |      -5.02e-05  |        -0.0225543 |                        0.3 |

The zero-shot transfer differences are very small. XX+XY is slightly better under the 10-seed native-Dryden spot check, but slightly worse for nominal and rotor-efficiency. With N=10 and effects near zero, this is descriptive only and does not overturn the 200-seed confirmatory source-derived result.

## Scientific conclusion

The structured quadratic hypothesis is **plausible but not confirmed as a performance upgrade**. Adding bounded local XX and XY interactions changes the regime dependence of the residual controller: it can improve nominal/model/latency cases, yet it slightly degrades several multiplicative or compound stress cases that were hypothesized to benefit most. In other words, a more nonlinear CeNN is not automatically a better closed-loop residual controller, even when the nonlinearity is physically motivated and numerically bounded.

The most defensible manuscript decision is therefore **not to replace the existing linear CeNN block**. Preserve this experiment as a high-value negative ablation. It strengthens the paper's broader finding that additional representational expressivity must earn its place through closed-loop evidence; offline fit and structural plausibility are insufficient.

A future quadratic-CeNN study could consider physics-selective cross terms, explicit regularization/contraction constraints, or identification on native actuator-aware dynamics, but such retuning would constitute a new research stage and should not be introduced post-hoc into this confirmatory ablation.

## Claim boundary

This is source-derived simulation evidence plus a small pinned-RotorPy-source-core transfer spot check. It is not full package-native AdaptiveQuadBench/acados, HIL, or flight validation, and it does not support generic CeNN superiority.

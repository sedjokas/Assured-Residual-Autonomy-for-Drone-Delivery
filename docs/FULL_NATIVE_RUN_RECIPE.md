# Full native execution recipe (not executed in this container)

1. Checkout AdaptiveQuadBench at d4c273861aa0ce6750818af0b1b63a2a40408e52.
2. Initialize RotorPy submodule at 07e6e2c57d55fc563ac55f9c44153c2769f0ae73.
3. Create the repository Python 3.11 conda environment from environment.yaml.
4. Build/install acados and `acados_template` as described in the repository README.
5. Verify `python run_eval.py --controller geo --experiment no --num_trials 5 --trajectory circle --serial`.
6. Verify native MPC with `--controller mpc`.
7. Insert the frozen I/K wrapper after the translational command proposal and before motor allocation; keep K parameters frozen only after a native disturbance-bound audit.

The current container cannot perform steps 1–4 because outbound DNS/network is disabled and acados is not preinstalled.

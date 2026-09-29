#!/usr/bin/env python3
"""Reproduce SCG paper-facing summaries from archived raw CSV data.

This is an analysis/reduction script, not the original simulation generator.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "SCG_safe_control_gym_derived"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(RAW / "public_benchmark_summary.csv")
paired = pd.read_csv(RAW / "public_benchmark_paired_tests.csv")
cmpc = pd.read_csv(RAW / "constrained_lmpc_summary.csv")
cmpc_pairs = pd.read_csv(RAW / "constrained_lmpc_paired_tests_20.csv")

summary.to_csv(OUT / "SCG_PUBLIC_BENCHMARK_SUMMARY_REPRODUCED.csv", index=False)
paired.to_csv(OUT / "SCG_PUBLIC_BENCHMARK_PAIRED_REPRODUCED.csv", index=False)
cmpc.to_csv(OUT / "SCG_CONSTRAINED_MPC_SUMMARY_REPRODUCED.csv", index=False)
cmpc_pairs.to_csv(OUT / "SCG_CONSTRAINED_MPC_PAIRED_REPRODUCED.csv", index=False)

print("SCG archived-data analysis reproduced successfully.")
print(f"public summary rows: {len(summary)}")
print(f"constrained-MPC summary rows: {len(cmpc)}")

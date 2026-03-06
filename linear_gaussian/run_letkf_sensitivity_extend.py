#!/usr/bin/env python
"""
Run additional LETKF sensitivity experiments for new (hscale, covinflate1)
ranges and merge into the existing results JSON.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from linear_forward_run_letkf_sensitivity import run_letkf_single

basedir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(basedir, "linear_gaussian_data.npz")
kf_file = os.path.join(basedir, "linear_gaussian_kf.npz")
results_file = os.path.join(basedir, "letkf_sensitivity_results.json")

K = 50
seed = 42

# New grid: hscale in [1, 1.5, 2, 2.5, 3], covinflate in [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
hscale_new = [1.0, 1.5, 2.0, 2.5, 3.0]
covinflate_new = [0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95]

# Load existing results
if os.path.exists(results_file):
    with open(results_file) as f:
        existing = json.load(f)
    results = existing['results']
    best_params = existing['best']
    best_rmse = best_params['rmse']
else:
    results = []
    best_params = {}
    best_rmse = np.inf

# Build set of already-run (hscale, covinflate1) pairs
done = {(e['hscale'], e['covinflate1']) for e in results}

# Count new experiments
todo = [(hs, ci) for hs in hscale_new for ci in covinflate_new
        if (hs, ci) not in done]

print(f"[Extend] {len(todo)} new experiments to run (K={K})")
print(f"  hscale_new: {hscale_new}")
print(f"  covinflate_new: {covinflate_new}")
print(f"  Already done: {len(done)} entries")

for idx, (hs, ci) in enumerate(todo):
    print(f"\n  [{idx+1}/{len(todo)}] hscale={hs}, covinflate1={ci:.2f} ...", end="", flush=True)
    try:
        rmse, pct, elapsed = run_letkf_single(
            data_file, kf_file, K, hs, ci, seed=seed)
        print(f"  RMSE={rmse:.6f}  pct={pct:.2f}%  time={elapsed:.1f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        rmse, pct, elapsed = float('nan'), 0.0, 0.0

    entry = {
        'hscale': hs, 'covinflate1': ci,
        'K': K,
        'rmse': float(rmse), 'pct_lt_sigy2': float(pct),
        'elapsed': float(elapsed),
    }
    results.append(entry)

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = entry.copy()

    # Save after each experiment (checkpoint)
    with open(results_file, 'w') as f:
        json.dump({'results': results, 'best': best_params}, f, indent=2)

print(f"\n[Extend] Done. Total entries: {len(results)}")
print(f"[Extend] Best overall: hscale={best_params['hscale']}, "
      f"covinflate1={best_params['covinflate1']:.2f}, "
      f"RMSE={best_params['rmse']:.6f}")
print(f"[Extend] Saved to {results_file}")

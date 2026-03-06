#!/usr/bin/env python
"""
Run a fine-grid LETKF sensitivity for hscale in [0.7..1.4] and
covinflate1 in [0.7..1.4], merge into existing JSON, and produce
a heatmap with an 'x' on the best cell.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from linear_forward_run_letkf_sensitivity import run_letkf_single

basedir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(basedir, "linear_gaussian_data.npz")
kf_file   = os.path.join(basedir, "linear_gaussian_kf.npz")
results_file = os.path.join(basedir, "letkf_sensitivity_results.json")

K, seed = 50, 42

hscale_vals    = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
covinflate_vals = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                   1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]

# Load existing
if os.path.exists(results_file):
    with open(results_file) as f:
        existing = json.load(f)
    results = existing['results']
    best_params = existing['best']
    best_rmse = best_params['rmse']
else:
    results, best_params, best_rmse = [], {}, float('inf')

done = {(round(e['hscale'],4), round(e['covinflate1'],4)) for e in results}
todo = [(hs, ci) for hs in hscale_vals for ci in covinflate_vals
        if (round(hs,4), round(ci,4)) not in done]

print(f"[Zoom] {len(todo)} new experiments (K={K}), {len(done)} already done")

for idx, (hs, ci) in enumerate(todo):
    print(f"  [{idx+1}/{len(todo)}] hs={hs}, ci={ci:.2f} ...", end="", flush=True)
    try:
        rmse, pct, elapsed = run_letkf_single(data_file, kf_file, K, hs, ci, seed=seed)
        print(f"  RMSE={rmse:.6f}  pct={pct:.2f}%  t={elapsed:.0f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        rmse, pct, elapsed = float('nan'), 0.0, 0.0

    entry = {'hscale': hs, 'covinflate1': ci, 'K': K,
             'rmse': float(rmse), 'pct_lt_sigy2': float(pct),
             'elapsed': float(elapsed)}
    results.append(entry)
    if rmse < best_rmse:
        best_rmse = rmse; best_params = entry.copy()
    with open(results_file, 'w') as f:
        json.dump({'results': results, 'best': best_params}, f, indent=2)

print(f"\n[Zoom] Best: hs={best_params['hscale']}, ci={best_params['covinflate1']:.2f}, "
      f"RMSE={best_params['rmse']:.6f}")

# ---- Heatmap ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Build lookup
lookup = {}
for e in results:
    key = (round(e['hscale'],4), round(e['covinflate1'],4))
    lookup[key] = e['rmse']

nh = len(hscale_vals)
nc = len(covinflate_vals)
rmse_grid = np.full((nh, nc), np.nan)
for i, hs in enumerate(hscale_vals):
    for j, ci in enumerate(covinflate_vals):
        rmse_grid[i, j] = lookup.get((round(hs,4), round(ci,4)), np.nan)

best_i, best_j = np.unravel_index(np.nanargmin(rmse_grid), rmse_grid.shape)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(rmse_grid, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(nc))
ax.set_xticklabels([f"{v:.2f}" for v in covinflate_vals], rotation=45, ha='right')
ax.set_yticks(range(nh))
ax.set_yticklabels([f"{v:.1f}" for v in hscale_vals])
ax.set_xlabel('Covariance inflation / deflation', fontsize=12)
ax.set_ylabel('Localization scale (hscale)', fontsize=12)
ax.set_title(f'LETKF RMSE vs KF  (K={K})', fontsize=14)
plt.colorbar(im, ax=ax, label='RMSE')

# Annotate cells
for i in range(nh):
    for j in range(nc):
        if np.isfinite(rmse_grid[i, j]):
            color = 'white' if rmse_grid[i,j] > np.nanmedian(rmse_grid) else 'black'
            ax.text(j, i, f"{rmse_grid[i,j]:.5f}", ha='center', va='center',
                    fontsize=6, color=color)

# Mark best with X
ax.plot(best_j, best_i, 'rx', markersize=18, markeredgewidth=3)
ax.text(best_j, best_i + 0.35, 'BEST', ha='center', va='bottom',
        fontsize=8, color='red', fontweight='bold')

plt.tight_layout()
figpath = os.path.join(basedir, "letkf_sensitivity_zoom_heatmap.png")
plt.savefig(figpath, dpi=150)
plt.close()
print(f"[Zoom] Heatmap saved to {figpath}")

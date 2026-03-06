#!/usr/bin/env python
"""
1. Fill missing LETKF sensitivity cells
2. Rerun LSMCMC V1 and V2 with higher Na
3. Run LETKF with best params
4. Plot RMSE(t) comparison: LSMCMC V1, V2, LETKF vs KF
"""
import os, sys, json, time
import numpy as np

basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(basedir, ".."))

data_file = os.path.join(basedir, "linear_gaussian_data.npz")
kf_file   = os.path.join(basedir, "linear_gaussian_kf.npz")
results_file = os.path.join(basedir, "letkf_sensitivity_results.json")

# =====================================================================
# STEP 1: Fill missing sensitivity cells
# =====================================================================
print("=" * 60)
print("STEP 1: Fill missing LETKF sensitivity cells")
print("=" * 60)

from linear_forward_run_letkf_sensitivity import run_letkf_single

hscale_vals = [1.0, 1.5, 2.0, 2.5, 3, 5, 8, 10, 15, 20, 30, 40]
covinflate_vals = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.02, 1.05, 1.08, 1.10, 1.15]
K, seed = 50, 42

with open(results_file) as f:
    existing = json.load(f)
results = existing['results']
best_params = existing['best']
best_rmse = best_params['rmse']

done = {(round(e['hscale'], 4), round(e['covinflate1'], 4)) for e in results}
todo = [(hs, ci) for hs in hscale_vals for ci in covinflate_vals
        if (round(hs, 4), round(ci, 4)) not in done]

print(f"  {len(todo)} missing cells to fill")

for idx, (hs, ci) in enumerate(todo):
    print(f"  [{idx+1}/{len(todo)}] hs={hs}, ci={ci:.2f} ...", end="", flush=True)
    try:
        rmse, pct, elapsed = run_letkf_single(data_file, kf_file, K, hs, ci, seed=seed)
        print(f"  RMSE={rmse:.6f}  t={elapsed:.0f}s")
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

print(f"  Best: hs={best_params['hscale']}, ci={best_params['covinflate1']}, RMSE={best_params['rmse']:.6f}")

# =====================================================================
# STEP 2: Rerun LSMCMC V1 (M=50, Na=2000)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 2: LSMCMC V1 (Na=2000, Nf=50, M=1)")
print("=" * 60)

from linear_forward_run_lsmcmc_v1 import run_lsmcmc_v1

dat = np.load(data_file)
T = int(dat['T']); d = int(dat['d'])
Na_new = 2000
Nf_new = 50

print("\n--- LSMCMC-V1 ---")
outfile_v1_raw = run_lsmcmc_v1(data_file=data_file, outdir=basedir,
                               seed=1000, Gamma=900, Na=Na_new, Nf=Nf_new)
res_v1 = np.load(outfile_v1_raw)
outfile_v1 = os.path.join(basedir, "linear_gaussian_lsmcmc_v1_avg.npz")
np.savez(outfile_v1, lsmcmc_mean=res_v1['lsmcmc_mean'],
         T=T, d=d, M=1, Na=Na_new, Nf=Nf_new,
         elapsed_mean=float(res_v1['elapsed']))
print(f"  V1 saved: {outfile_v1}")

# =====================================================================
# STEP 3: Rerun LSMCMC V2 (M=50, Na=2000)
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3: LSMCMC V2 (Na=2000, Nf=50, M=1)")
print("=" * 60)

from linear_forward_run_lsmcmc_v2 import run_lsmcmc_v2

print("\n--- LSMCMC-V2 ---")
outfile_v2_raw = run_lsmcmc_v2(data_file=data_file, outdir=basedir,
                               seed=2000, Gamma=900, Na=Na_new,
                               Nf=Nf_new, r_loc=10.0)
res_v2 = np.load(outfile_v2_raw)
outfile_v2 = os.path.join(basedir, "linear_gaussian_lsmcmc_v2_avg.npz")
np.savez(outfile_v2, lsmcmc_mean=res_v2['lsmcmc_mean'],
         T=T, d=d, M=1, Na=Na_new, Nf=Nf_new,
         elapsed_mean=float(res_v2['elapsed']))
print(f"  V2 saved: {outfile_v2}")

# =====================================================================
# STEP 4: Run LETKF with best params
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4: LETKF with best params")
print("=" * 60)

from linear_forward_run_letkf import run_letkf

best_hs = best_params['hscale']
best_ci = best_params['covinflate1']
letkf_outfile = run_letkf(data_file=data_file, outdir=basedir, seed=42,
                           K=50, hscale=best_hs, covinflate1=best_ci)
print(f"  LETKF saved: {letkf_outfile}")

# =====================================================================
# STEP 5: Heatmap + RMSE(t) comparison plot
# =====================================================================
print("\n" + "=" * 60)
print("STEP 5: Generate plots")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 5a: Updated heatmap ---
with open(results_file) as f:
    d_sens = json.load(f)

lookup = {}
for e in d_sens['results']:
    lookup[(round(e['hscale'],4), round(e['covinflate1'],4))] = e['rmse']

nh, nc = len(hscale_vals), len(covinflate_vals)
rmse_grid = np.full((nh, nc), np.nan)
for i, hs in enumerate(hscale_vals):
    for j, ci in enumerate(covinflate_vals):
        rmse_grid[i, j] = lookup.get((round(hs,4), round(ci,4)), np.nan)

best_i, best_j = np.unravel_index(np.nanargmin(rmse_grid), rmse_grid.shape)

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(rmse_grid, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(nc))
ax.set_xticklabels([f'{v:.2f}' for v in covinflate_vals], rotation=45, ha='right')
ax.set_yticks(range(nh))
ax.set_yticklabels([str(v) for v in hscale_vals])
ax.set_xlabel('Covariance inflation / deflation', fontsize=13)
ax.set_ylabel('Localization scale (hscale, grid points)', fontsize=13)
ax.set_title(f'LETKF Sensitivity: RMSE vs KF  (K={K})', fontsize=14)
plt.colorbar(im, ax=ax, label='RMSE', shrink=0.85)
for i in range(nh):
    for j in range(nc):
        v = rmse_grid[i, j]
        if np.isfinite(v):
            color = 'white' if v > np.nanmedian(rmse_grid) else 'black'
            ax.text(j, i, f'{v:.5f}', ha='center', va='center', fontsize=5.5, color=color)
ax.plot(best_j, best_i, 'rx', markersize=20, markeredgewidth=3)
ax.text(best_j, best_i + 0.4, 'BEST', ha='center', va='bottom',
        fontsize=9, color='red', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(basedir, 'letkf_sensitivity_heatmap.png'), dpi=150)
plt.close()
print("  Heatmap saved")

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)

#!/usr/bin/env python3
"""
Plot combined (real + synthetic) SWOT SSH observations at selected cycles.
  - RED  = real SWOT data
  - BLUE = synthetic (HYCOM-based swath fill)

Generates a 3×3 grid of subplots with coastlines, bathymetry contours,
and observation points overlaid.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset

NX, NY = 80, 70
LON_MIN, LON_MAX = -60.0, -20.0
LAT_MIN, LAT_MAX = 10.0, 45.0
lon_grid = np.linspace(LON_MIN, LON_MAX, NX)
lat_grid = np.linspace(LAT_MIN, LAT_MAX, NY)

# ---- Load combined file ----
combined_file = './data/swot_2024aug_new/swot_ssh_combined_80x70.nc'
nc = Dataset(combined_file, 'r')
ssha_all = nc.variables['ssha'][:]
cell_all = nc.variables['cell_index'][:]
src_all  = nc.variables['source'][:]
nobs_arr = nc.variables['n_obs'][:]
nc.close()

# ---- Load bathymetry for contours ----
bathy = np.load('./data/etopo_bathy_-60.0_-20.0_10.0_45.0_70x80.npy')
bathy = np.abs(bathy)

# ---- Select cycles to plot ----
# Aim for a mix: some with real SWOT only, some with synthetic only,
# some at different phases of the orbit
cycles_to_plot = [1, 7, 20, 50, 80, 100, 150, 200, 236]
# filter to valid range
cycles_to_plot = [c for c in cycles_to_plot if 0 <= c < 240]

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for idx, cyc in enumerate(cycles_to_plot):
    ax = axes[idx]
    n = int(nobs_arr[cyc])
    
    if n == 0:
        ax.set_title(f'Cycle {cyc}: No observations', fontsize=11)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        continue
    
    cells = cell_all[cyc, :n].astype(int)
    ssha  = ssha_all[cyc, :n]
    src   = src_all[cyc, :n]
    
    iy = cells // NX
    ix = cells % NX
    lons = lon_grid[ix]
    lats = lat_grid[iy]
    
    real_mask = (src == 0)
    synth_mask = (src == 1)
    
    # Bathymetry contour (light grey)
    ax.contour(lon_grid, lat_grid, bathy, levels=[200, 1000, 3000],
               colors='lightgrey', linewidths=0.5, linestyles='-')
    
    # Plot synthetic first (blue), then real on top (red)
    if synth_mask.any():
        sc = ax.scatter(lons[synth_mask], lats[synth_mask],
                        c=ssha[synth_mask], cmap='RdBu_r',
                        vmin=-1.0, vmax=1.0,
                        s=8, alpha=0.7, marker='s',
                        edgecolors='blue', linewidths=0.3)
    
    if real_mask.any():
        sc = ax.scatter(lons[real_mask], lats[real_mask],
                        c=ssha[real_mask], cmap='RdBu_r',
                        vmin=-1.0, vmax=1.0,
                        s=12, alpha=0.9, marker='o',
                        edgecolors='red', linewidths=0.5)
    
    n_real = real_mask.sum()
    n_synth = synth_mask.sum()
    title_parts = [f'Cycle {cyc}']
    if n_real > 0:
        title_parts.append(f'Real: {n_real}')
    if n_synth > 0:
        title_parts.append(f'Synth: {n_synth}')
    ax.set_title(' | '.join(title_parts), fontsize=10)
    
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect('equal')
    if idx >= 6:
        ax.set_xlabel('Longitude')
    if idx % 3 == 0:
        ax.set_ylabel('Latitude')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
           markeredgecolor='red', markersize=8, label='Real SWOT'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
           markeredgecolor='blue', markersize=8, label='Synthetic (HYCOM)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.01))

plt.suptitle('SWOT SSH Observations: Real (red edge) vs Synthetic Swaths (blue edge)',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])

out_path = './ssh_obs_combined.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# ---- Also: orbit progression plot ----
fig2, ax2 = plt.subplots(figsize=(14, 5))
for c in range(240):
    n = int(nobs_arr[c])
    if n == 0:
        continue
    cells = cell_all[c, :n].astype(int)
    src   = src_all[c, :n]
    ix = cells % NX
    lons = lon_grid[ix]
    
    real_mask = (src == 0)
    synth_mask = (src == 1)
    
    if real_mask.any():
        ax2.scatter(np.full(real_mask.sum(), c), lons[real_mask],
                    c='red', s=0.2, alpha=0.5, rasterized=True)
    if synth_mask.any():
        ax2.scatter(np.full(synth_mask.sum(), c), lons[synth_mask],
                    c='blue', s=0.2, alpha=0.3, rasterized=True)

ax2.set_xlabel('Cycle')
ax2.set_ylabel('Longitude (°)')
ax2.set_title('SWOT Observation Coverage: Longitude vs Cycle\n'
              'Red = Real SWOT, Blue = Synthetic Swath')
ax2.set_xlim(0, 240)
ax2.set_ylim(LON_MIN, LON_MAX)
ax2.grid(True, alpha=0.3)

out_path2 = './ssh_obs_orbit_progression.png'
fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path2}")

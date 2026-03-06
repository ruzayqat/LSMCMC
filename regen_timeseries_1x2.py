#!/usr/bin/env python
"""
Regenerate the 3 timeseries figures (vel, SST, SSH) with 1x2 panels
from the saved V2 output, then copy into paper/figures and paper_mwr/figures.
"""
import os
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Global font sizes ─────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
})
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

# ── Paths ────────────────────────────────────────────────────────
OUTDIR   = './output_lsmcmc_ldata_V2'
NC_PATH  = os.path.join(OUTDIR, 'mlswe_lsmcmc_out.nc')
OBS_PATH = os.path.join(OUTDIR, 'mlswe_merged_obs.nc')
PLOT_DIR = os.path.join(OUTDIR, 'plots')
PREFIX   = 'mlswe_results'

# ── Load analysis ────────────────────────────────────────────────
print("Loading analysis ...")
with Dataset(NC_PATH, 'r') as nc:
    raw = np.asarray(nc.variables['lsmcmc_mean'][:])
    obs_times = np.asarray(nc.variables['obs_times'][:])
    nlayers = int(nc.nlayers)
    ny = int(nc.ny) if hasattr(nc, 'ny') else len(nc.dimensions['y'])
    nx = int(nc.nx) if hasattr(nc, 'nx') else len(nc.dimensions['x'])
    H_b = np.asarray(nc.variables['H_b'][:]) if 'H_b' in nc.variables else None

lsmcmc_mean = raw.reshape(raw.shape[0], -1)
nassim = lsmcmc_mean.shape[0] - 1
ncells = ny * nx
print(f"  ny={ny}, nx={nx}, ncells={ncells}, nassim={nassim}")

# ── Load observations ────────────────────────────────────────────
print("Loading observations ...")
with Dataset(OBS_PATH, 'r') as nc:
    yobs  = np.asarray(nc.variables['yobs_all'][:])
    yind  = np.asarray(nc.variables['yobs_ind_all'][:])
    yind0 = np.asarray(nc.variables['yobs_ind_level0_all'][:])

# ── Grid ─────────────────────────────────────────────────────────
lon_grid = np.linspace(-60.0, -20.0, nx)
lat_grid = np.linspace(10.0, 45.0, ny)

def get_fields(step):
    vec = lsmcmc_mean[step]
    h_total = vec[0*ncells:1*ncells].reshape(ny, nx)
    u0 = vec[1*ncells:2*ncells].reshape(ny, nx)
    v0 = vec[2*ncells:3*ncells].reshape(ny, nx)
    T0 = vec[3*ncells:4*ncells].reshape(ny, nx)
    return h_total, u0, v0, T0

# ================================================================
#  Figure 1: Velocity time series  (1 x 2)
# ================================================================
print("\n--- Velocity timeseries (1x2) ---")
obs_count = np.zeros(ncells, dtype=int)
for t in range(nassim):
    ind0t = yind0[t]
    valid = (ind0t >= 0) & (ind0t < ncells)
    for idx in ind0t[valid]:
        obs_count[int(idx)] += 1
top_cells = np.argsort(obs_count)[-2:][::-1]  # top 2

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
for panel, cell_flat in enumerate(top_cells):
    ax = axes[panel]
    iy_c, ix_c = divmod(cell_flat, nx)
    lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
    u_idx = ncells + cell_flat
    v_idx = 2 * ncells + cell_flat

    anal_u = lsmcmc_mean[:, u_idx]
    anal_v = lsmcmc_mean[:, v_idx]

    obs_u = np.full(nassim, np.nan)
    obs_v = np.full(nassim, np.nan)
    for t in range(nassim):
        ind_t = yind[t]; y_t = yobs[t]
        mu = np.where((ind_t == u_idx) & np.isfinite(y_t))[0]
        if mu.size > 0: obs_u[t] = np.mean(y_t[mu])
        mv = np.where((ind_t == v_idx) & np.isfinite(y_t))[0]
        if mv.size > 0: obs_v[t] = np.mean(y_t[mv])
    ax.plot(np.arange(nassim), obs_u, 'rx', ms=7, label='Obs u (east)')
    ax.plot(np.arange(nassim), obs_v, 'bx', ms=7, label='Obs v (north)')
    ax.plot(np.arange(nassim + 1), anal_u, 'r-', alpha=0.8, lw=1.5,
            label='Analysis u')
    ax.plot(np.arange(nassim + 1), anal_v, 'b-', alpha=0.8, lw=1.5,
            label='Analysis v')
    ax.set_title(f'({lon_c:.1f}°E, {lat_c:.1f}°N) — {obs_count[cell_flat]} obs',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Velocity (m/s)')
    ax.legend(fontsize=12, ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle('Analysis vs Observed Velocity at Most-Observed Grid Cells', fontsize=16)
vel_path = os.path.join(PLOT_DIR, f'{PREFIX}_vel_timeseries.png')
fig.savefig(vel_path, dpi=150)
print(f"Saved: {vel_path}")
plt.close(fig)

# ================================================================
#  Figure 2: SST time series  (1 x 2)
# ================================================================
print("\n--- SST timeseries (1x2) ---")
sst_obs_count = np.zeros(ncells, dtype=int)
for t in range(nassim):
    ind_t = yind[t]
    sst_mask_t = (ind_t >= 3*ncells) & (ind_t < 4*ncells) & np.isfinite(yobs[t])
    for idx in ind_t[sst_mask_t]:
        cell = int(idx) - 3*ncells
        if 0 <= cell < ncells:
            sst_obs_count[cell] += 1
top_sst_cells = np.argsort(sst_obs_count)[-2:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
for panel in range(2):
    cell_flat = top_sst_cells[panel]
    iy_c, ix_c = divmod(cell_flat, nx)
    lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
    t_idx = 3 * ncells + cell_flat

    anal_T = lsmcmc_mean[:, t_idx] - 273.15

    ax = axes[panel]
    obs_T = np.full(nassim, np.nan)
    for t in range(nassim):
        match = np.where((yind[t] == t_idx) & np.isfinite(yobs[t]))[0]
        if match.size > 0:
            obs_T[t] = np.mean(yobs[t][match]) - 273.15
    ax.plot(np.arange(nassim), obs_T, 'rx', ms=7, label='Obs SST')
    ax.plot(np.arange(nassim + 1), anal_T, 'k-', lw=1.5,
            label='Analysis SST')
    ax.set_title(f'SST at ({lon_c:.1f}°E, {lat_c:.1f}°N) — {sst_obs_count[cell_flat]} SST obs',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Temperature (°C)')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

fig.suptitle('SST: Analysis vs Drifter Observations', fontsize=16)
sst_path = os.path.join(PLOT_DIR, f'{PREFIX}_sst_timeseries.png')
fig.savefig(sst_path, dpi=150)
print(f"Saved: {sst_path}")
plt.close(fig)

# ================================================================
#  Figure 3: SSH time series  (1 x 2)
# ================================================================
print("\n--- SSH timeseries (1x2) ---")
from datetime import datetime as _dt9

swot_epoch = _dt9(2000, 1, 1)

# -- Load SWOT SSH data from merged obs NC --
# The merged obs NC has embedded SWOT data; let's extract SSH obs from yind
ssh_obs_count = np.zeros(ncells, dtype=int)
ssh_obs_per_cycle = {}

# Load raw SWOT data for SSH
swot_ssh_all = None
swot_lons_all = None
swot_lats_all = None
swot_times_all = None

# Try to load SWOT data from the data directory
swot_file_candidates = [
    './data/obs_2024aug/swot_ssh_binned_70x80.nc',
    './data/obs_2024aug/swot_ssh_na_2024aug.nc',
]
swot_file = None
for c in swot_file_candidates:
    if os.path.exists(c):
        swot_file = c
        break

if swot_file:
    print(f"Loading SWOT from: {swot_file}")
    with Dataset(swot_file, 'r') as nc:
        varnames = list(nc.variables.keys())
        print(f"  Variables: {varnames}")
        if 'ssh' in nc.variables:
            swot_ssh_all = np.asarray(nc.variables['ssh'][:]).ravel()
        elif 'ssh_karin_2' in nc.variables:
            swot_ssh_all = np.asarray(nc.variables['ssh_karin_2'][:]).ravel()
        if 'longitude' in nc.variables:
            swot_lons_all = np.asarray(nc.variables['longitude'][:]).ravel()
        elif 'lon' in nc.variables:
            swot_lons_all = np.asarray(nc.variables['lon'][:]).ravel()
        if 'latitude' in nc.variables:
            swot_lats_all = np.asarray(nc.variables['latitude'][:]).ravel()
        elif 'lat' in nc.variables:
            swot_lats_all = np.asarray(nc.variables['lat'][:]).ravel()
        if 'time' in nc.variables:
            swot_times_all = np.asarray(nc.variables['time'][:]).ravel()

has_swot = (swot_ssh_all is not None and swot_lons_all is not None
            and swot_lats_all is not None and swot_times_all is not None
            and len(swot_ssh_all) > 0)

if has_swot:
    print(f"  Loaded {len(swot_ssh_all):,} SWOT points")
    dlon = abs(lon_grid[1] - lon_grid[0]) / 2.0
    dlat_g = abs(lat_grid[1] - lat_grid[0]) / 2.0
    time_tol = 3600.0

    ix_all = np.searchsorted(lon_grid, swot_lons_all) - 1
    ix_all = np.clip(ix_all, 0, nx - 1)
    ix_p1 = np.clip(ix_all + 1, 0, nx - 1)
    closer_right = np.abs(lon_grid[ix_p1] - swot_lons_all) < np.abs(lon_grid[ix_all] - swot_lons_all)
    ix_all = np.where(closer_right, ix_p1, ix_all)

    iy_all = np.searchsorted(lat_grid, swot_lats_all) - 1
    iy_all = np.clip(iy_all, 0, ny - 1)
    iy_p1 = np.clip(iy_all + 1, 0, ny - 1)
    closer_up = np.abs(lat_grid[iy_p1] - swot_lats_all) < np.abs(lat_grid[iy_all] - swot_lats_all)
    iy_all = np.where(closer_up, iy_p1, iy_all)

    within_grid = ((np.abs(lon_grid[ix_all] - swot_lons_all) <= dlon) &
                   (np.abs(lat_grid[iy_all] - swot_lats_all) <= dlat_g))
    ix_g = ix_all[within_grid]
    iy_g = iy_all[within_grid]
    ssh_grid = swot_ssh_all[within_grid]
    time_grid = swot_times_all[within_grid]
    cell_flat_all = iy_g * nx + ix_g

    unix_to_swot_offset = (_dt9(1970, 1, 1) - swot_epoch).total_seconds()

    sort_idx = np.argsort(time_grid)
    time_sorted = time_grid[sort_idx]
    cell_sorted = cell_flat_all[sort_idx]
    ssh_sorted = ssh_grid[sort_idx]

    for t_idx in range(nassim):
        t_unix = obs_times[t_idx + 1] if t_idx + 1 < len(obs_times) else obs_times[-1]
        if t_unix <= 0:
            continue
        t_swot = t_unix + unix_to_swot_offset
        lo = np.searchsorted(time_sorted, t_swot - time_tol)
        hi = np.searchsorted(time_sorted, t_swot + time_tol)
        if lo >= hi:
            continue
        cells_t = cell_sorted[lo:hi]
        ssh_t = ssh_sorted[lo:hi]
        cell_means = {}
        for c, s in zip(cells_t, ssh_t):
            if c not in cell_means:
                cell_means[c] = [s]
            else:
                cell_means[c].append(s)
        for c in cell_means:
            cell_means[c] = np.mean(cell_means[c])
            ssh_obs_count[c] += 1
        if cell_means:
            ssh_obs_per_cycle[t_idx] = cell_means

    print(f"  SWOT: {ssh_obs_count.sum()} obs across {len(ssh_obs_per_cycle)} cycles")
else:
    # Fallback: extract SSH obs from yind (indices in [0, ncells) → SSH)
    print("No raw SWOT file found; extracting SSH obs from yind...")
    for t in range(nassim):
        ind_t = yind[t]
        y_t = yobs[t]
        ssh_mask = (ind_t >= 0) & (ind_t < ncells) & np.isfinite(y_t)
        for i in np.where(ssh_mask)[0]:
            cell = int(ind_t[i])
            ssh_obs_count[cell] += 1
            if t not in ssh_obs_per_cycle:
                ssh_obs_per_cycle[t] = {}
            if cell not in ssh_obs_per_cycle[t]:
                ssh_obs_per_cycle[t][cell] = y_t[i]
            else:
                ssh_obs_per_cycle[t][cell] = 0.5*(ssh_obs_per_cycle[t][cell] + y_t[i])
    print(f"  SSH from yind: {ssh_obs_count.sum()} obs across {len(ssh_obs_per_cycle)} cycles")

top_ssh_cells = np.argsort(ssh_obs_count)[-2:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
for panel, cell_flat in enumerate(top_ssh_cells[:2]):
    ax = axes[panel]
    iy_c, ix_c = divmod(cell_flat, nx)
    lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
    h_b_cell = H_b[iy_c, ix_c] if H_b is not None else 0.0

    anal_ssh = np.zeros(nassim + 1)
    for t in range(nassim + 1):
        h_total_t, _, _, _ = get_fields(t)
        anal_ssh[t] = h_total_t[iy_c, ix_c] - h_b_cell

    ax.plot(np.arange(nassim + 1), anal_ssh, 'b-', alpha=0.8, lw=1.5,
            label='Analysis SSH')
    obs_ssh = np.full(nassim, np.nan)
    for t_idx, cell_dict in ssh_obs_per_cycle.items():
        if cell_flat in cell_dict:
            obs_ssh[t_idx] = cell_dict[cell_flat] - h_b_cell
    n_obs_cell = np.sum(np.isfinite(obs_ssh))
    ax.plot(np.arange(nassim), obs_ssh, 'gx', ms=7,
            label=f'SWOT SSH obs ({n_obs_cell})')
    ax.set_title(f'({lon_c:.1f}°E, {lat_c:.1f}°N) — '
                 f'{ssh_obs_count[cell_flat]} SSH obs',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('SSH (m)')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

fig.suptitle('SSH: Analysis vs SWOT Observations at Most-Observed Grid Cells', fontsize=16)
ssh_path = os.path.join(PLOT_DIR, f'{PREFIX}_ssh_timeseries.png')
fig.savefig(ssh_path, dpi=150)
print(f"Saved: {ssh_path}")
plt.close(fig)

# ================================================================
#  Copy figures to paper directories
# ================================================================
for dest_dir in ['paper/figures', 'paper_mwr/figures']:
    for fname in [f'{PREFIX}_vel_timeseries.png',
                  f'{PREFIX}_sst_timeseries.png',
                  f'{PREFIX}_ssh_timeseries.png']:
        src = os.path.join(PLOT_DIR, fname)
        dst = os.path.join(dest_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")

print("\nDone.")

#!/usr/bin/env python3
"""
Pre-bin SWOT L2 LR SSH data onto the model grid (one obs per cell per cycle).

Reads all 80 raw SWOT NetCDF files (~2 km resolution), bins into the
80x70 model grid (0.5 deg cells), averages SSH per cell, and saves as a
compact NetCDF file.  This avoids the slow runtime binning.

Corrections applied:
  1. height_cor_xover  - crossover calibration (removes +/-3 m orbit bias)
  2. MSS - geoid       - converts SSHA to ADT (absolute dynamic topography)

ADT = ssha_karin + height_cor_xover + (mean_sea_surface_cnescls - geoid)

ADT is comparable to HYCOM SSH and model eta = h_total - H_b (all ~O(1) m).

The raw SWOT files are kept for future use at different resolutions.
"""
import os, glob
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta

# ---- Configuration (must match example_input_mlswe_ldata_V1.yml) ----
NX, NY = 80, 70
LON_MIN, LON_MAX = -60.0, -20.0
LAT_MIN, LAT_MAX = 10.0, 45.0
lon_grid = np.linspace(LON_MIN, LON_MAX, NX)
lat_grid = np.linspace(LAT_MIN, LAT_MAX, NY)
dlon = (lon_grid[1] - lon_grid[0]) / 2.0  # half cell width
dlat = (lat_grid[1] - lat_grid[0]) / 2.0

NASSIM = 240
T_START = datetime(2024, 8, 1, 0, 0, 0)
DT = 75.0
T_FREQ = 48
obs_dt = timedelta(seconds=T_FREQ * DT)
# Cycles: t0 + (i+1)*obs_dt for i=0..239
obs_times = [T_START + (i + 1) * obs_dt for i in range(NASSIM)]
SWOT_EPOCH = datetime(2000, 1, 1)
obs_epoch_sec = np.array([(t - SWOT_EPOCH).total_seconds() for t in obs_times])
TIME_TOLERANCE_S = 3600.0  # ±1 hour

SWOT_DIR = './data/swot_2024aug_new'
OUT_FILE = './data/swot_2024aug_new/swot_ssh_binned_80x70.nc'

# ---- Load all files, bin per cycle ----
files = sorted(glob.glob(os.path.join(SWOT_DIR, 'SWOT_*.nc')))
print(f"Processing {len(files)} SWOT files → {NX}×{NY} grid, {NASSIM} cycles")

# Accumulators: per (cycle, cell) → list of SSH values
from collections import defaultdict
cell_data = defaultdict(list)  # key=(cycle_idx, cell_flat) → [ssha values]

n_total_pts = 0
n_valid_pts = 0
n_in_domain = 0
n_matched = 0

n_no_xover = 0

for fi, fpath in enumerate(files):
    nc = Dataset(fpath, 'r')
    lon_sw = np.asarray(nc.variables['longitude'][:])   # (nlines, npix)
    lat_sw = np.asarray(nc.variables['latitude'][:])
    ssha_sw = np.asarray(nc.variables['ssha_karin'][:])
    time_sw = np.asarray(nc.variables['time'][:])        # (nlines,) seconds since 2000-01-01
    qual_sw = np.asarray(nc.variables['ssha_karin_qual'][:]) if 'ssha_karin_qual' in nc.variables else None
    # Crossover correction (removes +/-3 m orbit bias)
    xover_sw = np.asarray(nc.variables['height_cor_xover'][:])
    xover_qual_sw = np.asarray(nc.variables['height_cor_xover_qual'][:])
    # Mean Sea Surface and geoid for SSHA -> ADT conversion
    mss_sw = np.asarray(nc.variables['mean_sea_surface_cnescls'][:])
    geoid_sw = np.asarray(nc.variables['geoid'][:])
    nc.close()

    nlines, npix = lon_sw.shape
    n_total_pts += nlines * npix

    # Broadcast time to 2D
    time_2d = np.repeat(time_sw[:, np.newaxis], npix, axis=1)

    # Flatten
    lon_f = lon_sw.ravel()
    lat_f = lat_sw.ravel()
    ssha_f = ssha_sw.ravel()
    time_f = time_2d.ravel()
    qual_f = qual_sw.ravel() if qual_sw is not None else None
    xover_f = xover_sw.ravel()
    xover_qual_f = xover_qual_sw.ravel()
    mss_f = mss_sw.ravel()
    geoid_f = geoid_sw.ravel()

    # Valid: finite SSHA + quality=0 + valid xover + finite MSS/geoid
    valid = np.isfinite(ssha_f) & np.isfinite(lon_f) & np.isfinite(lat_f)
    valid &= np.isfinite(xover_f) & (xover_qual_f <= 1)  # good or suspect
    valid &= np.isfinite(mss_f) & np.isfinite(geoid_f)
    if qual_f is not None:
        valid &= (qual_f == 0)
    n_valid_pts += valid.sum()

    # Count points without xover for reporting
    n_no_xover += int((np.isfinite(ssha_f) & (~np.isfinite(xover_f) | (xover_qual_f > 1))).sum())

    # Convert lon to [-180, 180]
    lon_f = np.where(lon_f > 180, lon_f - 360, lon_f)

    # Domain filter
    valid &= (lon_f >= LON_MIN - dlon) & (lon_f <= LON_MAX + dlon)
    valid &= (lat_f >= LAT_MIN - dlat) & (lat_f <= LAT_MAX + dlat)

    idx = np.where(valid)[0]
    n_in_domain += len(idx)

    if len(idx) == 0:
        if (fi + 1) % 20 == 0:
            print(f"  File {fi+1}/{len(files)}: 0 pts in domain")
        continue

    # For each valid point, find nearest grid cell and nearest cycle
    lon_v = lon_f[idx]
    lat_v = lat_f[idx]
    # ADT = ssha + xover + MDT,  where MDT = MSS - geoid
    adt_v = ssha_f[idx] + xover_f[idx] + (mss_f[idx] - geoid_f[idx])
    time_v = time_f[idx]

    # Nearest grid cell indices (vectorized)
    ix = np.searchsorted(lon_grid, lon_v) - 1
    ix = np.clip(ix, 0, NX - 1)
    # Refine: check if ix or ix+1 is closer
    ix_alt = np.minimum(ix + 1, NX - 1)
    closer = np.abs(lon_grid[ix_alt] - lon_v) < np.abs(lon_grid[ix] - lon_v)
    ix[closer] = ix_alt[closer]

    iy = np.searchsorted(lat_grid, lat_v) - 1
    iy = np.clip(iy, 0, NY - 1)
    iy_alt = np.minimum(iy + 1, NY - 1)
    closer = np.abs(lat_grid[iy_alt] - lat_v) < np.abs(lat_grid[iy] - lat_v)
    iy[closer] = iy_alt[closer]

    # Check within half-cell
    in_cell = (np.abs(lon_grid[ix] - lon_v) <= dlon) & \
              (np.abs(lat_grid[iy] - lat_v) <= dlat)

    # Nearest cycle (vectorized)
    cyc_idx = np.searchsorted(obs_epoch_sec, time_v) - 1
    cyc_idx = np.clip(cyc_idx, 0, NASSIM - 1)
    # Check if cyc_idx or cyc_idx+1 is closer
    cyc_alt = np.minimum(cyc_idx + 1, NASSIM - 1)
    closer_c = np.abs(obs_epoch_sec[cyc_alt] - time_v) < np.abs(obs_epoch_sec[cyc_idx] - time_v)
    cyc_idx[closer_c] = cyc_alt[closer_c]
    in_time = np.abs(obs_epoch_sec[cyc_idx] - time_v) <= TIME_TOLERANCE_S

    keep = in_cell & in_time
    kept = np.where(keep)[0]
    n_matched += len(kept)

    for k in kept:
        c = int(cyc_idx[k])
        cell = int(iy[k]) * NX + int(ix[k])
        cell_data[(c, cell)].append(adt_v[k])

    if (fi + 1) % 10 == 0:
        print(f"  File {fi+1}/{len(files)}: {len(kept)} pts binned "
              f"(cumulative {n_matched})")

print(f"\nTotal SWOT pixels: {n_total_pts:,}")
print(f"Valid (finite+qual0+xover): {n_valid_pts:,}")
print(f"Rejected (no xover): {n_no_xover:,}")
print(f"In domain: {n_in_domain:,}")
print(f"Matched to grid+cycle: {n_matched:,}")
print(f"Unique (cycle, cell) pairs: {len(cell_data):,}")

# ---- Average per cell, build per-cycle arrays ----
ssh_per_cycle = [[] for _ in range(NASSIM)]   # (cell_flat, mean_adt, n_pts)

for (c, cell), vals in cell_data.items():
    ssh_per_cycle[c].append((cell, np.mean(vals), len(vals)))

# Stats
nobs_per_cycle = np.array([len(s) for s in ssh_per_cycle])
n_with_obs = np.sum(nobs_per_cycle > 0)
print(f"\nCycles with obs: {n_with_obs}/{NASSIM}")
print(f"Max obs/cycle (cells): {nobs_per_cycle.max()}")
if n_with_obs > 0:
    nz = nobs_per_cycle[nobs_per_cycle > 0]
    print(f"Mean obs/cycle (non-empty): {nz.mean():.1f}")
    print(f"Median obs/cycle (non-empty): {np.median(nz):.0f}")

# ---- Write compact NetCDF ----
max_nobs = max(nobs_per_cycle.max(), 1)

ssha_arr = np.full((NASSIM, max_nobs), np.nan, dtype=np.float32)
cell_arr = np.full((NASSIM, max_nobs), -1, dtype=np.int32)
npts_arr = np.full((NASSIM, max_nobs), 0, dtype=np.int32)

for c in range(NASSIM):
    for i, (cell, mean_ssh, npts) in enumerate(ssh_per_cycle[c]):
        ssha_arr[c, i] = mean_ssh
        cell_arr[c, i] = cell
        npts_arr[c, i] = npts

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with Dataset(OUT_FILE, 'w', format='NETCDF4') as nc:
    nc.createDimension('cycle', NASSIM)
    nc.createDimension('max_obs', max_nobs)

    v = nc.createVariable('ssha', 'f4', ('cycle', 'max_obs'), zlib=True)
    v[:] = ssha_arr
    v.long_name = 'cell-averaged absolute dynamic topography (ADT)'
    v.units = 'm'
    v.comment = ('ADT = ssha_karin + height_cor_xover + '
                 '(mean_sea_surface_cnescls - geoid). '
                 'Comparable to HYCOM SSH / model eta.')

    v2 = nc.createVariable('cell_index', 'i4', ('cycle', 'max_obs'), zlib=True)
    v2[:] = cell_arr
    v2.long_name = 'flat grid cell index (iy*nx + ix)'

    v3 = nc.createVariable('n_raw_pts', 'i4', ('cycle', 'max_obs'), zlib=True)
    v3[:] = npts_arr
    v3.long_name = 'number of raw SWOT pixels averaged into this cell'

    v4 = nc.createVariable('n_obs', 'i4', ('cycle',), zlib=True)
    v4[:] = nobs_per_cycle
    v4.long_name = 'number of observed cells this cycle'

    # Store grid info
    nc.nx = NX
    nc.ny = NY
    nc.lon_min = LON_MIN
    nc.lon_max = LON_MAX
    nc.lat_min = LAT_MIN
    nc.lat_max = LAT_MAX
    nc.nassim = NASSIM
    nc.time_tolerance_s = TIME_TOLERANCE_S
    nc.quality_filter = 'ssha_karin_qual == 0, height_cor_xover_qual <= 1'
    nc.description = ('SWOT L2 LR SSH binned to 80x70 grid as ADT '
                       '(= ssha_karin + xover + MDT). '
                       'Comparable to HYCOM SSH / model eta.')

    # Store obs times as epoch seconds
    vt = nc.createVariable('obs_times', 'f8', ('cycle',), zlib=True)
    vt[:] = obs_epoch_sec + (SWOT_EPOCH - datetime(1970, 1, 1)).total_seconds()
    vt.units = 'seconds since 1970-01-01'

print(f"\nSaved: {OUT_FILE}")
print(f"  Shape: ({NASSIM}, {max_nobs})")
sz = os.path.getsize(OUT_FILE)
print(f"  Size: {sz/1024:.0f} KB")

#!/usr/bin/env python3
"""
Generate swe_drifter_obs.nc for the 80x70 grid domain [-60,-20]x[10,45].

Reads the 6-hourly GDP drifter CSV, **interpolates each drifter track to
hourly resolution** (cubic spline for positions, finite-diff for velocities,
linear for SST), then converts to obs arrays and writes the NetCDF.
"""
import sys, os
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.interpolate import CubicSpline

# drifter_data module (local copy)
from drifter_data import drifters_to_obs_arrays, build_obs_netcdf

# ---- Configuration ----
CSV_PATH = './data/osmc_drifters_2024-08-01_2024-08-11.csv'
OUT_DIR  = './data/obs_2024aug'
OUT_FILE = os.path.join(OUT_DIR, 'swe_drifter_obs.nc')

# Grid: must match example_input_mlswe_ldata_V1.yml
NX = 80   # dgx
NY = 70   # dgy
LON_MIN, LON_MAX = -60.0, -20.0
LAT_MIN, LAT_MAX =  10.0,  45.0

lon_grid = np.linspace(LON_MIN, LON_MAX, NX)
lat_grid = np.linspace(LAT_MIN, LAT_MAX, NY)

# Assimilation times: 240 hourly cycles starting 2024-08-01
NASSIM = 240
T_START = datetime(2024, 8, 1, 0, 0, 0)
obs_times = [T_START + timedelta(hours=i) for i in range(NASSIM)]

# After interpolation, data is hourly → tight tolerance
TIME_TOLERANCE_S = 1800  # +/-30 min (hourly data → exact match)

OBS_TYPE = 'velocity_sst'
SIG_SST = 0.4   # SST obs noise (K)

R_EARTH = 6.371e6  # metres

# ---- Load CSV into per-drifter records ----
import csv as csv_module

raw_records = defaultdict(list)  # drifter_id -> list of (time, lat, lon, ve, vn, sst)

with open(CSV_PATH, 'r') as f:
    reader = csv_module.reader(f)
    header = next(reader)
    _units = next(reader)  # units row
    col = {name.strip(): i for i, name in enumerate(header)}

    for row in reader:
        if len(row) < max(col.get('vn', 10), col.get('ve', 9)) + 1:
            continue
        try:
            drifter_id = row[col['ID']].strip()
            t_str      = row[col['time']].strip()
            lat_v      = float(row[col['latitude']])
            lon_v      = float(row[col['longitude']])
            sst_str    = row[col['sst']].strip()
            ve_str     = row[col['ve']].strip()
            vn_str     = row[col['vn']].strip()
        except (ValueError, KeyError):
            continue

        # Parse time
        try:
            t = datetime.strptime(t_str[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue

        # Parse velocities
        try:
            ve_val = float(ve_str)
            vn_val = float(vn_str)
        except ValueError:
            continue

        # Parse SST (optional)
        try:
            sst_val = float(sst_str) if sst_str and sst_str != 'NaN' else np.nan
        except ValueError:
            sst_val = np.nan

        # Filter to domain
        if not (LON_MIN <= lon_v <= LON_MAX and LAT_MIN <= lat_v <= LAT_MAX):
            continue

        raw_records[drifter_id].append((t, lat_v, lon_v, ve_val, vn_val, sst_val))

n_raw = sum(len(v) for v in raw_records.values())
print(f"Loaded {n_raw} raw records from {len(raw_records)} drifters (6-hourly)")

# ---- Interpolate each drifter track to hourly resolution ----
print("\nInterpolating drifter tracks to hourly resolution...")
print("  Positions: cubic spline interpolation")
print("  Velocities: recomputed from hourly positions (centered FD)")
print("  SST: linear interpolation")

interp_lons, interp_lats = [], []
interp_ves, interp_vns = [], []
interp_ssts, interp_times, interp_ids = [], [], []

n_interp_total = 0
n_drifters_used = 0

for did, pts in raw_records.items():
    pts.sort(key=lambda x: x[0])
    n = len(pts)
    if n < 3:
        # Need at least 3 points for cubic spline; skip short tracks
        continue

    # Extract arrays
    times_raw = np.array([p[0] for p in pts])
    lats_raw  = np.array([p[1] for p in pts])
    lons_raw  = np.array([p[2] for p in pts])
    ssts_raw  = np.array([p[5] for p in pts])

    # Time in hours since first record
    t0 = times_raw[0]
    t_hours = np.array([(t - t0).total_seconds() / 3600.0 for t in times_raw])

    # Skip if time span is too short (< 6h)
    if t_hours[-1] - t_hours[0] < 6.0:
        continue

    # Remove duplicate times (keep first)
    _, unique_idx = np.unique(t_hours, return_index=True)
    t_hours = t_hours[unique_idx]
    lats_raw = lats_raw[unique_idx]
    lons_raw = lons_raw[unique_idx]
    ssts_raw = ssts_raw[unique_idx]
    times_raw = times_raw[unique_idx]

    if len(t_hours) < 3:
        continue

    # Hourly target times (only interpolate, no extrapolation)
    t_start_h = np.ceil(t_hours[0])   # first full hour at or after first record
    t_end_h   = np.floor(t_hours[-1]) # last full hour at or before last record
    t_hourly = np.arange(t_start_h, t_end_h + 0.5, 1.0)

    if len(t_hourly) < 2:
        continue

    # Cubic spline interpolation for positions
    cs_lon = CubicSpline(t_hours, lons_raw, bc_type='natural')
    cs_lat = CubicSpline(t_hours, lats_raw, bc_type='natural')

    lon_hourly = cs_lon(t_hourly)
    lat_hourly = cs_lat(t_hourly)

    # Recompute velocities from hourly positions via centered finite differences
    ve_hourly = np.zeros(len(t_hourly))
    vn_hourly = np.zeros(len(t_hourly))

    for i in range(len(t_hourly)):
        if i == 0:
            ia, ib = 0, 1
        elif i == len(t_hourly) - 1:
            ia, ib = len(t_hourly) - 2, len(t_hourly) - 1
        else:
            ia, ib = i - 1, i + 1

        dt_s = (t_hourly[ib] - t_hourly[ia]) * 3600.0  # hours -> seconds
        dlat = np.radians(lat_hourly[ib] - lat_hourly[ia])
        dlon = np.radians(lon_hourly[ib] - lon_hourly[ia])
        cos_lat = np.cos(np.radians(lat_hourly[i]))

        ve_hourly[i] = R_EARTH * cos_lat * dlon / dt_s
        vn_hourly[i] = R_EARTH * dlat / dt_s

    # Linear interpolation for SST (avoid cubic oscillation on noisy SST)
    finite_sst = np.isfinite(ssts_raw)
    if np.sum(finite_sst) >= 2:
        sst_hourly = np.interp(t_hourly, t_hours[finite_sst], ssts_raw[finite_sst])
    else:
        sst_hourly = np.full(len(t_hourly), np.nan)

    # Reject unrealistic interpolated velocities
    speed = np.sqrt(ve_hourly**2 + vn_hourly**2)
    valid = speed < 3.0  # < 3 m/s

    # Also ensure positions are still within domain
    valid &= (lon_hourly >= LON_MIN) & (lon_hourly <= LON_MAX)
    valid &= (lat_hourly >= LAT_MIN) & (lat_hourly <= LAT_MAX)

    idx = np.where(valid)[0]
    if len(idx) == 0:
        continue

    plat_id = int(did) if did.isdigit() else hash(did) % (10**9)

    for i in idx:
        t_actual = t0 + timedelta(hours=float(t_hourly[i]))
        interp_times.append(t_actual)
        interp_lats.append(lat_hourly[i])
        interp_lons.append(lon_hourly[i])
        interp_ves.append(ve_hourly[i])
        interp_vns.append(vn_hourly[i])
        interp_ssts.append(sst_hourly[i])
        interp_ids.append(plat_id)

    n_interp_total += len(idx)
    n_drifters_used += 1

print(f"\n  Drifters used (>=3 records): {n_drifters_used}/{len(raw_records)}")
print(f"  Hourly records after interpolation: {n_interp_total} (was {n_raw} at 6h)")
print(f"  Expansion factor: {n_interp_total/n_raw:.1f}x")

# ---- Build drifter_data dict ----
times_arr = np.array(interp_times, dtype=object)
lat_arr   = np.array(interp_lats)
lon_arr   = np.array(interp_lons)
ve_arr    = np.array(interp_ves)
vn_arr    = np.array(interp_vns)
sst_arr   = np.array(interp_ssts)
ids_arr   = np.array(interp_ids)

# Convert SST from Celsius to Kelvin
finite_sst = np.isfinite(sst_arr)
sst_arr[finite_sst] += 273.15

# Velocity errors: slightly larger than raw GDP (interpolation adds uncertainty)
# Raw GDP hourly ~ 0.01 m/s; here we interpolate from 6h data -> ~0.02 m/s
ve_err = np.full_like(ve_arr, 0.02)
vn_err = np.full_like(vn_arr, 0.02)

# Sort by time
sort_idx = np.argsort(times_arr)
drifter_data = {
    'lon':      lon_arr[sort_idx],
    'lat':      lat_arr[sort_idx],
    've':       ve_arr[sort_idx],
    'vn':       vn_arr[sort_idx],
    've_error': ve_err[sort_idx],
    'vn_error': vn_err[sort_idx],
    'sst':      sst_arr[sort_idx],
    'time':     times_arr[sort_idx],
    'ids':      ids_arr[sort_idx],
}

n_drifters = len(np.unique(ids_arr))
print(f"\n  {n_interp_total} hourly records from {n_drifters} drifters")
print(f"  Time range: {drifter_data['time'][0]} -> {drifter_data['time'][-1]}")
print(f"  SST range (K): {np.nanmin(drifter_data['sst']):.1f} -> {np.nanmax(drifter_data['sst']):.1f}")
print(f"  Velocity range: ve=[{ve_arr.min():.3f}, {ve_arr.max():.3f}], "
      f"vn=[{vn_arr.min():.3f}, {vn_arr.max():.3f}] m/s")

# ---- Convert to obs arrays ----
print(f"\nConverting to obs arrays (grid {NX}x{NY}, {NASSIM} cycles)...")
print(f"  obs_type={OBS_TYPE}, time_tolerance={TIME_TOLERANCE_S}s, sig_sst={SIG_SST}")

yobs_all, yobs_ind_all, yobs_ind0_all, sig_y_all = drifters_to_obs_arrays(
    drifter_data,
    lon_grid,
    lat_grid,
    obs_times,
    obs_type=OBS_TYPE,
    time_tolerance_s=TIME_TOLERANCE_S,
    sig_sst=SIG_SST,
)

# ---- Stats ----
nobs_per_cycle = [len(y) for y in yobs_all]
nonempty = sum(1 for n in nobs_per_cycle if n > 0)
print(f"\n  Cycles with obs: {nonempty}/{NASSIM}")
print(f"  Max obs/cycle: {max(nobs_per_cycle)}")
if nonempty > 0:
    nobs_nonempty = [n for n in nobs_per_cycle if n > 0]
    print(f"  Mean obs/cycle (non-empty): {np.mean(nobs_nonempty):.1f}")
    print(f"  Min obs/cycle (non-empty): {min(nobs_nonempty)}")
total_obs = sum(nobs_per_cycle)
print(f"  Total observations: {total_obs}")

# Show first few cycles
print("\n  First 12 cycles obs counts:")
for i in range(min(12, NASSIM)):
    print(f"    Cycle {i} ({obs_times[i]}): {nobs_per_cycle[i]} obs")

# ---- Write NetCDF ----
os.makedirs(OUT_DIR, exist_ok=True)
build_obs_netcdf(
    OUT_FILE,
    yobs_all,
    yobs_ind_all,
    yobs_ind0_all,
    obs_times,
    sig_y=0.02,
    sig_y_all=sig_y_all,
    attrs={
        'obs_type': OBS_TYPE,
        'time_tolerance_s': TIME_TOLERANCE_S,
        'grid_nx': NX,
        'grid_ny': NY,
        'lon_range': f'{LON_MIN},{LON_MAX}',
        'lat_range': f'{LAT_MIN},{LAT_MAX}',
        'nassim': NASSIM,
        'source': 'GDP drifter CSV, 6h->1h cubic spline interpolation, velocity + SST obs',
        'interpolation': 'cubic spline (positions), centered FD (velocities), linear (SST)',
    },
)

print(f"\nDone! Obs file: {OUT_FILE}")

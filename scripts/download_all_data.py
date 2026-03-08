#!/usr/bin/env python3
"""
download_all_data.py
====================
Download and prepare **all** external data needed to run the MLSWE experiments.

This script is the single entry point for data preparation.  It downloads:

  1. **HYCOM boundary conditions** (SSH, velocity, SST) via OPeNDAP
  2. **ETOPO bathymetry** via OPeNDAP
  3. **OSMC drifter observations** via ERDDAP
  4. **SWOT L2 LR SSH** via NASA Earthdata (requires ``earthaccess``)
  5. **HYCOM SST reference** (derived from the BC file — no extra download)

After completion the ``data/`` folder will contain everything required by
the runner scripts. The total download is around **200–600 MB** depending
on SWOT coverage during the chosen window.

Usage
-----
::

    python3 download_all_data.py          # default: Aug 2024 window
    python3 download_all_data.py --help   # all options

Requirements
------------
    pip install netCDF4 scipy earthaccess   # earthaccess only for real SWOT
"""
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
#  Paths (relative to this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)       # repository root
DATA_DIR   = os.path.join(REPO_ROOT, 'data')

# ---------------------------------------------------------------------------
#  1.  HYCOM Boundary Conditions
# ---------------------------------------------------------------------------
def step1_hycom_bc(lon_min, lon_max, lat_min, lat_max,
                   time_start, time_end, stride=6):
    """Download HYCOM GOFS 3.1 surface fields (SSH, u, v, SST)."""
    from download_hycom_bc import download_hycom_bc

    out_path = os.path.join(DATA_DIR, 'hycom_bc_2024aug.nc')
    print("\n" + "=" * 65)
    print("  STEP 1 / 6 — HYCOM Boundary Conditions")
    print("=" * 65)

    download_hycom_bc(
        lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max,
        time_start=time_start, time_end=time_end,
        output_path=out_path,
        buffer_deg=2.0,
        include_sst=True,
        stride=stride,
    )
    print(f"  → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
#  2.  ETOPO Bathymetry
# ---------------------------------------------------------------------------
def step2_etopo(lon_grid, lat_grid):
    """Download ETOPO 2022 60-arc-second bathymetry via OPeNDAP."""
    from scipy.interpolate import RegularGridInterpolator

    ny, nx = len(lat_grid), len(lon_grid)
    cache_file = os.path.join(
        DATA_DIR,
        f"etopo_bathy_{lon_grid[0]:.1f}_{lon_grid[-1]:.1f}_"
        f"{lat_grid[0]:.1f}_{lat_grid[-1]:.1f}_{ny}x{nx}.npy")

    print("\n" + "=" * 65)
    print("  STEP 2 / 6 — ETOPO Bathymetry")
    print("=" * 65)

    if os.path.exists(cache_file):
        print(f"  Already cached: {cache_file}")
        return np.load(cache_file)

    from netCDF4 import Dataset
    url = ("https://www.ngdc.noaa.gov/thredds/dodsC/"
           "global/ETOPO2022/60s/60s_bed_elev_netcdf/"
           "ETOPO_2022_v1_60s_N90W180_bed.nc")
    print(f"  Downloading from NOAA NCEI OPeNDAP ...")
    nc = Dataset(url, 'r')
    etopo_lon = np.asarray(nc.variables['lon'][:])
    etopo_lat = np.asarray(nc.variables['lat'][:])

    lon_min_buf = lon_grid[0] - 0.5
    lon_max_buf = lon_grid[-1] + 0.5
    lat_min_buf = lat_grid[0] - 0.5
    lat_max_buf = lat_grid[-1] + 0.5
    ix0 = max(0, np.searchsorted(etopo_lon, lon_min_buf) - 1)
    ix1 = min(len(etopo_lon), np.searchsorted(etopo_lon, lon_max_buf) + 1)
    iy0 = max(0, np.searchsorted(etopo_lat, lat_min_buf) - 1)
    iy1 = min(len(etopo_lat), np.searchsorted(etopo_lat, lat_max_buf) + 1)

    z = np.asarray(nc.variables['z'][iy0:iy1, ix0:ix1], dtype=np.float64)
    sub_lon = etopo_lon[ix0:ix1]
    sub_lat = etopo_lat[iy0:iy1]
    nc.close()

    interp = RegularGridInterpolator(
        (sub_lat, sub_lon), z,
        method='linear', bounds_error=False, fill_value=None)
    pts = np.array(np.meshgrid(lat_grid, lon_grid, indexing='ij'))
    pts = pts.reshape(2, -1).T
    H_b = -interp(pts).reshape(ny, nx)
    H_b = np.maximum(H_b, 10.0)

    np.save(cache_file, H_b)
    print(f"  → {cache_file}  shape={H_b.shape}")
    return H_b


# ---------------------------------------------------------------------------
#  3.  OSMC Drifter Observations
# ---------------------------------------------------------------------------
def step3_drifters(lon_min, lon_max, lat_min, lat_max,
                   time_start, time_end):
    """Download OSMC drifter data from ERDDAP."""
    t0 = time_start[:10]  # '2024-08-01'
    t1 = time_end[:10]    # '2024-08-11'
    filename = f"osmc_drifters_{t0}_{t1}.csv"
    dest = os.path.join(DATA_DIR, filename)

    print("\n" + "=" * 65)
    print("  STEP 3 / 6 — OSMC Drifter Observations")
    print("=" * 65)

    if os.path.exists(dest):
        print(f"  Already cached: {dest}")
        return dest

    # Build OSMC ERDDAP query
    base_url = (
        "https://osmc.noaa.gov/erddap/tabledap/OSMC_RealTime.csv"
    )
    variables = "platform_code,time,latitude,longitude,sst,platform_type"
    constraints = (
        f"&latitude>={lat_min}&latitude<={lat_max}"
        f"&longitude>={lon_min}&longitude<={lon_max}"
        f"&time>={t0}T00:00:00Z&time<={t1}T23:59:59Z"
        f"&platform_type=%22drifting%20buoy%22"
    )
    url = f"{base_url}?{variables}{constraints}"

    print(f"  Downloading from OSMC ERDDAP ...")
    print(f"  Region: lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]")
    print(f"  Period: {t0} to {t1}")

    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        import urllib.request
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Python/download_all_data')
        with urllib.request.urlopen(req, timeout=180) as resp:
            with open(dest, 'wb') as fh:
                fh.write(resp.read())
    except Exception as e:
        print(f"  WARNING: ERDDAP download failed: {e}")
        print(f"  You can manually download from:")
        print(f"    {url}")
        print(f"  and save to: {dest}")
        return None

    print(f"  → {dest}")
    return dest


# ---------------------------------------------------------------------------
#  4.  SWOT SSH Data
# ---------------------------------------------------------------------------
def step4_swot(lon_min, lon_max, lat_min, lat_max,
               time_start, time_end, swot_dir=None):
    """Download SWOT L2 LR SSH from PO.DAAC via earthaccess."""
    if swot_dir is None:
        swot_dir = os.path.join(DATA_DIR, 'swot_2024aug_new')

    print("\n" + "=" * 65)
    print("  STEP 4 / 6 — SWOT L2 LR SSH (via NASA Earthdata)")
    print("=" * 65)

    existing = [f for f in os.listdir(swot_dir) if f.startswith('SWOT_')]  \
        if os.path.isdir(swot_dir) else []
    if len(existing) >= 20:
        print(f"  Already have {len(existing)} SWOT files in {swot_dir}")
        return swot_dir

    try:
        from swot_ssh_data import download_swot_ssh
    except ImportError:
        print("  WARNING: swot_ssh_data.py not found. Skipping SWOT download.")
        print("  Install earthaccess: pip install earthaccess")
        return swot_dir

    try:
        download_swot_ssh(
            lon_range=(lon_min, lon_max),
            lat_range=(lat_min, lat_max),
            time_range=(time_start, time_end),
            dest_dir=swot_dir,
        )
    except Exception as e:
        print(f"  WARNING: SWOT download failed: {e}")
        print("  You may need to: pip install earthaccess")
        print("  and authenticate with NASA Earthdata Login.")

    print(f"  → {swot_dir}/")
    return swot_dir


# ---------------------------------------------------------------------------
#  5.  HYCOM SSH Reference (from BC file)
# ---------------------------------------------------------------------------
def step5_ssh_ref(bc_file, lon_grid, lat_grid, nassim, dt, t_freq):
    """Extract SSH reference time series from the HYCOM BC file."""
    from netCDF4 import Dataset
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt

    ny, nx = len(lat_grid), len(lon_grid)
    cache_file = os.path.join(
        DATA_DIR, f"hycom_ssh_ref_{ny}x{nx}_{nassim}steps.npy")

    print("\n" + "=" * 65)
    print("  STEP 5 / 6 — HYCOM SSH Reference")
    print("=" * 65)

    if os.path.exists(cache_file):
        print(f"  Already cached: {cache_file}")
        return cache_file

    print(f"  Extracting SSH reference from {bc_file} ...")
    with Dataset(bc_file, 'r') as nc:
        bc_lon = np.asarray(nc.variables['lon'][:], dtype=np.float64)
        bc_lat = np.asarray(nc.variables['lat'][:], dtype=np.float64)
        ssh_raw = np.asarray(nc.variables['ssh'][:], dtype=np.float64)

    # Fill NaNs
    def fill_nan(a):
        mask = np.isnan(a)
        if not mask.any():
            return a
        ind = distance_transform_edt(mask, return_distances=False,
                                     return_indices=True)
        a[mask] = a[tuple(ind[:, mask])]
        return a

    for k in range(ssh_raw.shape[0]):
        ssh_raw[k] = fill_nan(ssh_raw[k])

    if bc_lat[0] > bc_lat[-1]:
        bc_lat = bc_lat[::-1]
        ssh_raw = ssh_raw[:, ::-1, :]
    if np.any(np.diff(bc_lon) < 0):
        sort_ix = np.argsort(bc_lon)
        bc_lon = bc_lon[sort_ix]
        ssh_raw = ssh_raw[:, :, sort_ix]

    pts = np.array(np.meshgrid(lat_grid, lon_grid, indexing='ij'))
    pts = pts.reshape(2, -1).T
    ntime = ssh_raw.shape[0]
    ssh_ref = np.empty((ntime, ny, nx), dtype=np.float64)
    for k in range(ntime):
        interp = RegularGridInterpolator(
            (bc_lat, bc_lon), ssh_raw[k],
            method='linear', bounds_error=False, fill_value=None)
        ssh_ref[k] = interp(pts).reshape(ny, nx)

    np.save(cache_file, ssh_ref)
    print(f"  → {cache_file}  shape={ssh_ref.shape}")
    return cache_file


# ---------------------------------------------------------------------------
#  6.  HYCOM SST Reference (from BC file)
# ---------------------------------------------------------------------------
def step6_sst_ref(bc_file, lon_grid, lat_grid, time_start, time_end):
    """Extract SST reference time series from the HYCOM BC file."""
    from netCDF4 import Dataset, num2date
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt

    ny, nx = len(lat_grid), len(lon_grid)
    safe_start = time_start.replace(':', '').replace('-', '').replace('T', '')
    safe_end = time_end.replace(':', '').replace('-', '').replace('T', '')
    cache_3d = os.path.join(
        DATA_DIR, f"hycom_sst_ref_{safe_start}_{safe_end}_{ny}x{nx}_3d.npy")
    cache_times = os.path.join(
        DATA_DIR, f"hycom_sst_ref_{safe_start}_{safe_end}_{ny}x{nx}_times.npy")

    print("\n" + "=" * 65)
    print("  STEP 6 / 6 — HYCOM SST Reference")
    print("=" * 65)

    if os.path.exists(cache_3d) and os.path.exists(cache_times):
        print(f"  Already cached: {cache_3d}")
        return cache_3d, cache_times

    print(f"  Extracting SST reference from {bc_file} ...")
    t_start = datetime.fromisoformat(time_start.replace('Z', ''))

    with Dataset(bc_file, 'r') as nc:
        if 'sst' not in nc.variables:
            raise ValueError("HYCOM BC file does not contain 'sst' variable.")

        bc_lon = np.asarray(nc.variables['lon'][:], dtype=np.float64)
        bc_lat = np.asarray(nc.variables['lat'][:], dtype=np.float64)
        t_var = nc.variables['time']
        t_units = t_var.units
        t_cal = t_var.calendar if hasattr(t_var, 'calendar') else 'standard'
        t_dates = num2date(t_var[:], units=t_units, calendar=t_cal)
        T_ref_times = np.array(
            [(d - t_start).total_seconds() for d in t_dates],
            dtype=np.float64)

        raw = nc.variables['sst'][:]
        if hasattr(raw, 'filled'):
            sst = np.asarray(raw.filled(np.nan), dtype=np.float64)
        else:
            sst = np.asarray(raw, dtype=np.float64)
        sst[sst < -1000.0] = np.nan

    # Fill NaNs
    def fill_nan(a):
        mask = np.isnan(a)
        if not mask.any():
            return a
        ind = distance_transform_edt(mask, return_distances=False,
                                     return_indices=True)
        a[mask] = a[tuple(ind[:, mask])]
        return a

    for k in range(sst.shape[0]):
        sst[k] = fill_nan(sst[k])

    if np.nanmean(sst) < 100.0:
        sst = sst + 273.15

    if bc_lat[0] > bc_lat[-1]:
        bc_lat = bc_lat[::-1]
        sst = sst[:, ::-1, :]
    if np.any(np.diff(bc_lon) < 0):
        sort_ix = np.argsort(bc_lon)
        bc_lon = bc_lon[sort_ix]
        sst = sst[:, :, sort_ix]

    pts = np.array(np.meshgrid(lat_grid, lon_grid, indexing='ij'))
    pts = pts.reshape(2, -1).T
    ntime = sst.shape[0]
    T_ref_3d = np.empty((ntime, ny, nx), dtype=np.float64)
    for k in range(ntime):
        interp = RegularGridInterpolator(
            (bc_lat, bc_lon), sst[k],
            method='linear', bounds_error=False, fill_value=None)
        T_ref_3d[k] = interp(pts).reshape(ny, nx)

    np.save(cache_3d, T_ref_3d)
    np.save(cache_times, T_ref_times)
    print(f"  → {cache_3d}  shape={T_ref_3d.shape}"
          f"  range=[{T_ref_3d.min():.1f}, {T_ref_3d.max():.1f}] K")
    return cache_3d, cache_times


# =========================================================================
#  Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download all data for MLSWE experiments.")
    parser.add_argument('--lon-min', type=float, default=-60.0)
    parser.add_argument('--lon-max', type=float, default=-20.0)
    parser.add_argument('--lat-min', type=float, default=10.0)
    parser.add_argument('--lat-max', type=float, default=45.0)
    parser.add_argument('--time-start', default='2024-08-01T00:00:00',
                        help='ISO start time (default: 2024-08-01)')
    parser.add_argument('--time-end', default='2024-08-12T00:00:00',
                        help='ISO end time (default: 2024-08-12, covers 240 cycles)')
    parser.add_argument('--nassim', type=int, default=240,
                        help='Number of assimilation cycles')
    parser.add_argument('--nx', type=int, default=80)
    parser.add_argument('--ny', type=int, default=70)
    parser.add_argument('--stride', type=int, default=6,
                        help='HYCOM spatial subsample (6 ≈ 0.5° matching model grid)')
    parser.add_argument('--skip-swot', action='store_true',
                        help='Skip SWOT download (requires earthaccess + NASA login)')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    lon_grid = np.linspace(args.lon_min, args.lon_max, args.nx)
    lat_grid = np.linspace(args.lat_min, args.lat_max, args.ny)
    dt, t_freq = 75.0, 48  # model timestep and obs frequency

    print("=" * 65)
    print("  MLSWE Data Download — All Steps")
    print(f"  Domain : [{args.lon_min}, {args.lon_max}] x "
          f"[{args.lat_min}, {args.lat_max}]")
    print(f"  Grid   : {args.nx} x {args.ny}")
    print(f"  Window : {args.time_start} → {args.time_end}")
    print(f"  Output : {DATA_DIR}/")
    print("=" * 65)

    # 1. HYCOM BC
    bc_file = step1_hycom_bc(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max,
        args.time_start, args.time_end, stride=args.stride)

    # 2. Bathymetry
    step2_etopo(lon_grid, lat_grid)

    # 3. Drifters
    step3_drifters(args.lon_min, args.lon_max,
                   args.lat_min, args.lat_max,
                   args.time_start, args.time_end)

    # 4. SWOT (optional — requires earthaccess)
    if not args.skip_swot:
        step4_swot(args.lon_min, args.lon_max,
                   args.lat_min, args.lat_max,
                   args.time_start, args.time_end)
    else:
        print("\n  STEP 4 / 6 — SWOT download SKIPPED (--skip-swot)")

    # 5. SSH reference (from HYCOM BC)
    step5_ssh_ref(bc_file, lon_grid, lat_grid,
                  args.nassim, dt, t_freq)

    # 6. SST reference (from HYCOM BC)
    step6_sst_ref(bc_file, lon_grid, lat_grid,
                  args.time_start, args.time_end)

    print("\n" + "=" * 65)
    print("  ALL DONE — data saved in data/")
    print("=" * 65)
    print("\nNext steps:")
    print("  1. Generate drifter obs:    python3 generate_drifter_obs.py")
    print("  2. Pre-bin SWOT SSH:        python3 prebin_swot_ssh.py")
    print("  3. Fill synthetic SWOT:     python3 generate_synthetic_swot.py")
    print("  4. Run experiments — see LSMCMC.ipynb")


if __name__ == '__main__':
    main()

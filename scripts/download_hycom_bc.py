#!/usr/bin/env python
"""
download_hycom_bc.py
====================
Download HYCOM GOFS 3.1 surface reanalysis data for SWE boundary conditions.

Source
------
HYCOM GLBy0.08 / expt_93.0  via OPeNDAP  (free, no registration required)

- Resolution : 1/12° (~8 km) horizontal, 3-hourly temporal
- Coverage   : 2018-12-04 to present
- Variables  : surf_el (SSH), water_u, water_v, water_temp (all surface)

The output NetCDF is directly compatible with ``BoundaryHandler(nc_path=...)``.

Usage
-----
From Python::

    from download_hycom_bc import download_hycom_bc

    download_hycom_bc(
        lon_min=-80, lon_max=-20,
        lat_min=10,  lat_max=50,
        time_start='2019-07-01T00:00:00',
        time_end='2019-07-06T00:00:00',
        output_path='./data/hycom_bc.nc',
    )

Or from the command line::

    python download_hycom_bc.py
"""
from __future__ import print_function

import os
import re
import sys
import time as _time
import numpy as np
from datetime import datetime, timedelta

try:
    from netCDF4 import Dataset, num2date, date2num
except ImportError:
    raise ImportError("netCDF4 is required: pip install netCDF4")

# -----------------------------------------------------------------------
#  HYCOM GOFS 3.1 OPeNDAP endpoint
# -----------------------------------------------------------------------
HYCOM_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"


# -----------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------
def _lon_to_hycom(lon):
    """Convert longitude from -180..180 to HYCOM 0..360 convention."""
    return lon % 360.0


def _parse_time_units(units_str):
    """
    Parse CF-style time units like 'hours since 2000-01-01 00:00:00'.
    Returns (multiplier_to_seconds, reference_datetime).
    """
    m = re.match(r'(\w+)\s+since\s+(.+)', units_str.strip())
    if not m:
        raise ValueError(f"Cannot parse time units: {units_str}")

    unit = m.group(1).lower().rstrip('s')   # 'hour', 'second', 'day', ...
    ref_str = m.group(2).strip()

    # Try parsing reference date
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M', '%Y-%m-%d'):
        try:
            ref_dt = datetime.strptime(ref_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Cannot parse reference date: {ref_str}")

    multiplier = {'second': 1.0, 'minute': 60.0, 'hour': 3600.0,
                  'day': 86400.0}.get(unit)
    if multiplier is None:
        raise ValueError(f"Unknown time unit: {unit}")

    return multiplier, ref_dt


def _raw_to_unix(raw_times, units_str):
    """
    Convert raw numeric times (as stored in the NetCDF) to seconds since
    1970-01-01 00:00:00 UTC (Unix epoch).  This avoids cftime dependencies.
    """
    multiplier, ref_dt = _parse_time_units(units_str)
    epoch = datetime(1970, 1, 1)
    offset_s = (ref_dt - epoch).total_seconds()
    return raw_times * multiplier + offset_s


def _unix_to_datetime(unix_s):
    """Convert a Unix timestamp (seconds) to a Python datetime."""
    return datetime(1970, 1, 1) + timedelta(seconds=float(unix_s))


# -----------------------------------------------------------------------
#  Main download function
# -----------------------------------------------------------------------
def download_hycom_bc(
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    time_start,
    time_end,
    output_path,
    buffer_deg=2.0,
    include_sst=True,
    stride=1,
    chunk_size=8,
    max_retries=3,
):
    """
    Download HYCOM surface fields and save as BoundaryHandler-compatible
    NetCDF with variables ``ssh``, ``uo``, ``vo`` (and optionally ``sst``)
    on ``(time, lat, lon)`` dimensions.

    Parameters
    ----------
    lon_min, lon_max : float
        Longitude bounds of the model domain (degrees E, –180..180).
    lat_min, lat_max : float
        Latitude bounds (degrees N).
    time_start, time_end : str
        ISO-format timestamps, e.g. ``'2019-07-01T00:00:00'``.
    output_path : str
        Where to save the output NetCDF.
    buffer_deg : float
        Extra degrees around the domain to download (default 2).
    include_sst : bool
        Also download surface temperature (water_temp at depth=0).
    stride : int
        Spatial subsampling factor (1 = full 1/12° resolution,
        3 = ~0.25°, 6 = ~0.5° matching model grid).
    chunk_size : int
        Time steps per OPeNDAP request (to avoid server timeouts).
    max_retries : int
        Retries per chunk on download failure.

    Returns
    -------
    output_path : str
    """
    if os.path.exists(output_path):
        print(f"[HYCOM] BC file already exists: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    t_start = datetime.fromisoformat(time_start.replace('Z', '+00:00')
                                     .replace('+00:00', ''))
    t_end = datetime.fromisoformat(time_end.replace('Z', '+00:00')
                                   .replace('+00:00', ''))

    print(f"[HYCOM] ──────────────────────────────────────────────")
    print(f"[HYCOM] Downloading boundary conditions from HYCOM")
    print(f"[HYCOM] URL : {HYCOM_URL}")
    print(f"[HYCOM] Domain : lon=[{lon_min}, {lon_max}], "
          f"lat=[{lat_min}, {lat_max}]")
    print(f"[HYCOM] Time   : {time_start} → {time_end}")
    print(f"[HYCOM] Buffer : ±{buffer_deg}°,  stride={stride}")
    print(f"[HYCOM] ──────────────────────────────────────────────")

    # ---- Open remote HYCOM dataset (metadata only) ----
    print("[HYCOM] Opening remote dataset (metadata) ...")
    try:
        ds = Dataset(HYCOM_URL, 'r')
    except Exception as e:
        raise RuntimeError(
            f"Cannot open HYCOM OPeNDAP endpoint.\n"
            f"  URL: {HYCOM_URL}\n"
            f"  Error: {e}\n"
            f"  Check your internet connection or try again later."
        ) from e

    # ---- Read coordinate arrays ----
    hycom_lon = np.asarray(ds.variables['lon'][:], dtype=np.float64)  # 0..360
    hycom_lat = np.asarray(ds.variables['lat'][:], dtype=np.float64)
    hycom_time_var = ds.variables['time']
    hycom_time_units = hycom_time_var.units

    # ---- Spatial subsetting ----
    hlon_min = _lon_to_hycom(lon_min - buffer_deg)
    hlon_max = _lon_to_hycom(lon_max + buffer_deg)
    hlat_min = lat_min - buffer_deg
    hlat_max = lat_max + buffer_deg

    # Longitude mask (handles wraparound)
    if hlon_min < hlon_max:
        lon_mask = (hycom_lon >= hlon_min) & (hycom_lon <= hlon_max)
    else:
        lon_mask = (hycom_lon >= hlon_min) | (hycom_lon <= hlon_max)

    lat_mask = (hycom_lat >= hlat_min) & (hycom_lat <= hlat_max)

    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]

    if len(lon_idx) == 0 or len(lat_idx) == 0:
        ds.close()
        raise ValueError(
            f"No HYCOM grid points in domain "
            f"lon=[{lon_min-buffer_deg}, {lon_max+buffer_deg}], "
            f"lat=[{lat_min-buffer_deg}, {lat_max+buffer_deg}]"
        )

    # Apply stride (subsample)
    lon_idx = lon_idx[::stride]
    lat_idx = lat_idx[::stride]

    i_lon_start = int(lon_idx[0])
    i_lon_end = int(lon_idx[-1])
    i_lat_start = int(lat_idx[0])
    i_lat_end = int(lat_idx[-1])

    sel_lon = hycom_lon[i_lon_start:i_lon_end + 1:stride]
    sel_lat = hycom_lat[i_lat_start:i_lat_end + 1:stride]
    nlon = len(sel_lon)
    nlat = len(sel_lat)

    # Convert to -180..180 for output
    sel_lon_out = np.where(sel_lon > 180.0, sel_lon - 360.0, sel_lon)

    print(f"[HYCOM] Spatial grid : {nlon} lon × {nlat} lat")
    print(f"[HYCOM]   lon : {sel_lon_out[0]:.3f} → {sel_lon_out[-1]:.3f}")
    print(f"[HYCOM]   lat : {sel_lat[0]:.3f} → {sel_lat[-1]:.3f}")

    # ---- Temporal subsetting ----
    # Read ALL HYCOM times (typically ~50 k entries for multi-year run)
    # This is fast because it's just one 1-D variable.
    print("[HYCOM] Reading time coordinate ...")
    hycom_times_raw = np.asarray(hycom_time_var[:], dtype=np.float64)
    hycom_times_unix = _raw_to_unix(hycom_times_raw, hycom_time_units)

    epoch = datetime(1970, 1, 1)
    t_start_unix = (t_start - timedelta(hours=6) - epoch).total_seconds()
    t_end_unix = (t_end + timedelta(hours=6) - epoch).total_seconds()

    time_mask = (hycom_times_unix >= t_start_unix) & \
                (hycom_times_unix <= t_end_unix)
    time_idx = np.where(time_mask)[0]

    if len(time_idx) == 0:
        ds.close()
        raise ValueError(
            f"No HYCOM time steps in [{time_start}, {time_end}].  "
            f"HYCOM covers "
            f"{_unix_to_datetime(hycom_times_unix[0]):%Y-%m-%d} to "
            f"{_unix_to_datetime(hycom_times_unix[-1]):%Y-%m-%d}."
        )

    i_time_start = int(time_idx[0])
    i_time_end = int(time_idx[-1])
    ntime = i_time_end - i_time_start + 1

    t0_str = _unix_to_datetime(hycom_times_unix[i_time_start])
    t1_str = _unix_to_datetime(hycom_times_unix[i_time_end])
    dt_hrs = (hycom_times_unix[i_time_start + 1] -
              hycom_times_unix[i_time_start]) / 3600.0 if ntime > 1 else 0
    print(f"[HYCOM] Time range  : {t0_str:%Y-%m-%d %H:%M} → "
          f"{t1_str:%Y-%m-%d %H:%M}")
    print(f"[HYCOM] Time steps  : {ntime}  (Δt = {dt_hrs:.1f} h)")

    # ---- Allocate arrays ----
    ssh_all = np.full((ntime, nlat, nlon), np.nan, dtype=np.float32)
    uo_all = np.full((ntime, nlat, nlon), np.nan, dtype=np.float32)
    vo_all = np.full((ntime, nlat, nlon), np.nan, dtype=np.float32)
    sst_all = (np.full((ntime, nlat, nlon), np.nan, dtype=np.float32)
               if include_sst else None)

    # ---- Download in time-chunks ----
    n_chunks = int(np.ceil(ntime / chunk_size))
    vars_to_get = ['surf_el', 'water_u', 'water_v']
    if include_sst:
        vars_to_get.append('water_temp')
    print(f"[HYCOM] Variables   : {vars_to_get}")
    print(f"[HYCOM] Downloading in {n_chunks} chunk(s) of ≤{chunk_size} "
          f"time steps ...")

    t_wall_start = _time.time()

    for ic in range(n_chunks):
        t0_idx = i_time_start + ic * chunk_size
        t1_idx = min(t0_idx + chunk_size, i_time_end + 1)
        local_start = ic * chunk_size
        local_end = local_start + (t1_idx - t0_idx)

        label = (f"  chunk {ic + 1}/{n_chunks}  "
                 f"(time indices {t0_idx}–{t1_idx - 1})")

        for attempt in range(max_retries):
            try:
                print(f"[HYCOM]{label}  downloading ...", end='', flush=True)

                def _to_clean(raw):
                    """Convert masked array → float32 with NaN for fill."""
                    if hasattr(raw, 'filled'):
                        a = np.asarray(raw.filled(np.nan), dtype=np.float32)
                    else:
                        a = np.asarray(raw, dtype=np.float32)
                    a[a < -1000.0] = np.nan  # HYCOM fill ≈ -30000
                    return a

                # SSH  (time, lat, lon)
                ssh_chunk = _to_clean(
                    ds.variables['surf_el'][
                        t0_idx:t1_idx,
                        i_lat_start:i_lat_end + 1:stride,
                        i_lon_start:i_lon_end + 1:stride])
                ssh_all[local_start:local_end] = ssh_chunk

                # u-velocity  (time, depth, lat, lon) → surface = depth index 0
                uo_chunk = _to_clean(
                    ds.variables['water_u'][
                        t0_idx:t1_idx, 0,
                        i_lat_start:i_lat_end + 1:stride,
                        i_lon_start:i_lon_end + 1:stride])
                uo_all[local_start:local_end] = uo_chunk

                # v-velocity
                vo_chunk = _to_clean(
                    ds.variables['water_v'][
                        t0_idx:t1_idx, 0,
                        i_lat_start:i_lat_end + 1:stride,
                        i_lon_start:i_lon_end + 1:stride])
                vo_all[local_start:local_end] = vo_chunk

                # SST  (water_temp at depth=0)
                if include_sst:
                    sst_chunk = _to_clean(
                        ds.variables['water_temp'][
                            t0_idx:t1_idx, 0,
                            i_lat_start:i_lat_end + 1:stride,
                            i_lon_start:i_lon_end + 1:stride])
                    sst_all[local_start:local_end] = sst_chunk

                elapsed = _time.time() - t_wall_start
                print(f"  OK  ({elapsed:.0f}s elapsed)")
                break  # success

            except Exception as e:
                print(f"  FAILED ({e})")
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"[HYCOM]    retrying in {wait}s ...")
                    _time.sleep(wait)
                else:
                    ds.close()
                    raise RuntimeError(
                        f"Failed to download chunk {ic + 1} after "
                        f"{max_retries} attempts: {e}"
                    ) from e

    ds.close()
    elapsed_total = _time.time() - t_wall_start
    print(f"[HYCOM] Download complete in {elapsed_total:.0f}s")

    # ---- Compute Unix times for the selected range ----
    sel_times_unix = hycom_times_unix[i_time_start:i_time_end + 1]

    # ---- Write output NetCDF ----
    print(f"[HYCOM] Writing {output_path} ...")

    with Dataset(output_path, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', ntime)
        nc.createDimension('lat', nlat)
        nc.createDimension('lon', nlon)

        # Time — stored as seconds since epoch for easy use
        vt = nc.createVariable('time', 'f8', ('time',))
        vt[:] = sel_times_unix
        vt.units = 'seconds since 1970-01-01 00:00:00'
        vt.calendar = 'standard'
        vt.long_name = 'time'

        # Latitude
        vlat = nc.createVariable('lat', 'f8', ('lat',))
        vlat[:] = sel_lat
        vlat.units = 'degrees_north'

        # Longitude (–180..180)
        vlon = nc.createVariable('lon', 'f8', ('lon',))
        vlon[:] = sel_lon_out
        vlon.units = 'degrees_east'

        # SSH
        vs = nc.createVariable('ssh', 'f4', ('time', 'lat', 'lon'),
                               zlib=True, complevel=4)
        vs[:] = ssh_all
        vs.units = 'm'
        vs.long_name = 'sea surface height'

        # Eastward velocity
        vu = nc.createVariable('uo', 'f4', ('time', 'lat', 'lon'),
                               zlib=True, complevel=4)
        vu[:] = uo_all
        vu.units = 'm/s'
        vu.long_name = 'eastward sea water velocity'

        # Northward velocity
        vv = nc.createVariable('vo', 'f4', ('time', 'lat', 'lon'),
                               zlib=True, complevel=4)
        vv[:] = vo_all
        vv.units = 'm/s'
        vv.long_name = 'northward sea water velocity'

        # SST (optional)
        if include_sst:
            vT = nc.createVariable('sst', 'f4', ('time', 'lat', 'lon'),
                                   zlib=True, complevel=4)
            vT[:] = sst_all
            vT.units = 'degC'
            vT.long_name = 'sea surface temperature'

        # Global attributes
        nc.source = 'HYCOM GOFS 3.1 GLBy0.08/expt_93.0'
        nc.url = HYCOM_URL
        nc.domain = (f'lon=[{lon_min}, {lon_max}], '
                     f'lat=[{lat_min}, {lat_max}]')
        nc.time_range = f'{time_start} to {time_end}'
        nc.resolution_deg = f'{stride / 12.0:.4f}'
        nc.include_sst = int(include_sst)

    fsize_mb = os.path.getsize(output_path) / 1e6
    print(f"[HYCOM] Saved {output_path}  ({fsize_mb:.1f} MB)")
    print(f"[HYCOM]   {ntime} time × {nlat} lat × {nlon} lon")
    return output_path


# -----------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(
        description='Download HYCOM GOFS 3.1 surface fields for SWE BC')
    p.add_argument('--lon-min', type=float, default=-80.0)
    p.add_argument('--lon-max', type=float, default=-20.0)
    p.add_argument('--lat-min', type=float, default=10.0)
    p.add_argument('--lat-max', type=float, default=50.0)
    p.add_argument('--time-start', default='2019-07-01T00:00:00')
    p.add_argument('--time-end', default='2019-07-06T00:00:00')
    p.add_argument('--output', '-o', default='./data/hycom_bc.nc')
    p.add_argument('--no-sst', action='store_true',
                   help='Skip SST download')
    p.add_argument('--stride', type=int, default=1,
                   help='Spatial subsampling (1=full 1/12°, 3=0.25°, 6=0.5°)')
    p.add_argument('--chunk-size', type=int, default=8)
    args = p.parse_args()

    download_hycom_bc(
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        time_start=args.time_start,
        time_end=args.time_end,
        output_path=args.output,
        include_sst=not args.no_sst,
        stride=args.stride,
        chunk_size=args.chunk_size,
    )

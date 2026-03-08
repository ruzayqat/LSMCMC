"""
drifter_data.py  –  Download and read NOAA Global Drifter Program hourly data
================================================================================

The GDP hourly dataset provides:
    - longitude, latitude  (degrees)
    - ve, vn               (eastward / northward velocity, m/s)
    - sst                  (sea surface temperature, Kelvin)
    - time                 (datetime64)
    - ID                   (drifter buoy ID)

Data source:  https://www.aoml.noaa.gov/phod/gdp/hourly_data.php
Direct NetCDF: https://www.ncei.noaa.gov/data/oceans/archive/arc0199/0248584/2.2/data/0-data/gdp_hourly.nc

Supported input formats:
    A. Full GDP NetCDF (``gdp_hourly.nc``)    → ``load_drifters_in_region()``
    B. Per-drifter ``.npy`` files             → ``load_drifters_from_npy()``
       Layout: ``drifters/floater_x00000.npy``, ``floater_y00000.npy``, etc.
    C. Interpolated ``.h5`` (1 dataset/step)  → ``load_drifters_from_h5()``
       Layout: ``floater_real.h5`` with keys ``t_00000000``, ...

This module provides:
    1. download_gdp_hourly()          – fetch the (big ~4 GB) NetCDF once.
    2. load_drifters_in_region()      – extract drifters from full GDP NetCDF.
    3. load_drifters_from_npy()       – read per-drifter .npy files.
    4. load_drifters_from_h5()        – read interpolated .h5 file.
    5. drifters_to_obs_arrays()       – convert to observation arrays for DA.
    6. build_obs_netcdf()             – write observations to NetCDF.
"""
from __future__ import print_function
import os
import glob
import warnings
import numpy as np
from datetime import datetime, timedelta

try:
    import h5py
except ImportError:
    h5py = None

try:
    from netCDF4 import Dataset, num2date
except ImportError:
    Dataset = None


# ---------------------------------------------------------------------------
#  1.  Download
# ---------------------------------------------------------------------------
GDP_HOURLY_URL = (
    "https://www.ncei.noaa.gov/data/oceans/archive/"
    "arc0199/0248584/2.2/data/0-data/gdp_hourly.nc"
)


def download_gdp_hourly(dest_dir=".", filename="gdp_hourly.nc", overwrite=False):
    """Download the full GDP hourly NetCDF (~4 GB).  Needs ``wget`` or ``urllib``."""
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and not overwrite:
        print(f"[drifter_data] File already exists: {dest}")
        return dest
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[drifter_data] Downloading GDP hourly data to {dest} ...")
    print(f"[drifter_data] URL: {GDP_HOURLY_URL}")
    print("[drifter_data] This file is ~4 GB; download may take a while.")
    try:
        import urllib.request
        urllib.request.urlretrieve(GDP_HOURLY_URL, dest)
    except Exception:
        # fallback: try wget
        ret = os.system(f'wget -c -O "{dest}" "{GDP_HOURLY_URL}"')
        if ret != 0:
            raise RuntimeError("Download failed.  Install wget or check URL.")
    print(f"[drifter_data] Download complete: {dest}")
    return dest


# ---------------------------------------------------------------------------
#  1b.  Download drifter subset from AOML ERDDAP (no ~4 GB download!)
# ---------------------------------------------------------------------------
ERDDAP_BASE = (
    "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_hourly_qc"
)


def download_gdp_erddap(
    lon_range=(-80.0, -20.0),
    lat_range=(10.0, 50.0),
    time_range=("2019-07-01", "2019-07-06"),
    dest_dir=".",
    filename=None,
    overwrite=False,
):
    """
    Download GDP hourly drifter data for a specific region and time window
    from the AOML ERDDAP server.  Returns a **small** CSV file (typically
    a few MB) instead of the 4 GB full NetCDF.

    Parameters
    ----------
    lon_range : (float, float)
        (lon_min, lon_max) in degrees East.
    lat_range : (float, float)
        (lat_min, lat_max) in degrees North.
    time_range : (str, str)
        ISO-8601 date strings, e.g. ('2019-07-01', '2019-07-06').
    dest_dir : str
        Directory for the downloaded file.
    filename : str or None
        Output filename.  If None, auto-generated from query bounds.
    overwrite : bool
        If False, skip download if file already exists.

    Returns
    -------
    dest : str
        Path to the downloaded CSV file.
    """
    if filename is None:
        filename = (
            f"gdp_hourly_{lon_range[0]}_{lon_range[1]}_"
            f"{lat_range[0]}_{lat_range[1]}_"
            f"{time_range[0]}_{time_range[1]}.csv"
        )
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and not overwrite:
        print(f"[drifter_data] ERDDAP cache exists: {dest}")
        return dest

    # Build ERDDAP query URL
    # Variables available: longitude, latitude, ve, vn, err_ve, err_vn,
    #   sst, time, ID
    # NOTE: ERDDAP constraint operators (>=, <=) must be percent-encoded
    #       because Python's urllib does not auto-encode them unlike browsers.
    variables = "longitude%2Clatitude%2Cve%2Cvn%2Cerr_ve%2Cerr_vn%2Csst%2Ctime%2CID"
    constraints = (
        f"&longitude%3E={lon_range[0]}&longitude%3C={lon_range[1]}"
        f"&latitude%3E={lat_range[0]}&latitude%3C={lat_range[1]}"
        f"&time%3E={time_range[0]}T00%3A00%3A00Z"
        f"&time%3C={time_range[1]}T23%3A59%3A59Z"
    )
    url = f"{ERDDAP_BASE}.csv?{variables}{constraints}"

    print(f"[drifter_data] Downloading drifters from AOML ERDDAP ...")
    print(f"[drifter_data] Region: lon={lon_range}, lat={lat_range}")
    print(f"[drifter_data] Period: {time_range[0]} to {time_range[1]}")
    print(f"[drifter_data] URL (len={len(url)}): {url[:120]}...")

    try:
        import urllib.request
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Python/drifter_data')
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(dest, 'wb') as fh:
                fh.write(resp.read())
    except Exception as e1:
        # fallback: try wget (wget handles encoding itself)
        wget_url = (
            f"{ERDDAP_BASE}.csv?"
            f"longitude,latitude,ve,vn,err_ve,err_vn,sst,time,ID"
            f"&longitude>={lon_range[0]}&longitude<={lon_range[1]}"
            f"&latitude>={lat_range[0]}&latitude<={lat_range[1]}"
            f"&time>={time_range[0]}T00:00:00Z"
            f"&time<={time_range[1]}T23:59:59Z"
        )
        ret = os.system(f'wget -q -O "{dest}" "{wget_url}"')
        if ret != 0:
            raise RuntimeError(
                f"ERDDAP download failed: {e1}\n"
                "Check internet connection or try the full GDP download instead."
            )

    # Check file is not an error page
    with open(dest, 'r') as f:
        header = f.readline()
    if 'Error' in header or '<!DOCTYPE' in header:
        err_txt = open(dest).read()[:500]
        os.remove(dest)
        raise RuntimeError(f"ERDDAP returned error:\n{err_txt}")

    # Count records
    n_lines = sum(1 for _ in open(dest)) - 2  # minus header + units rows
    print(f"[drifter_data] Downloaded {n_lines} drifter records to {dest}")
    return dest


def load_drifters_from_erddap_csv(csv_path, lon_range=None, lat_range=None):
    """
    Read a CSV file downloaded from AOML ERDDAP and return drifter records
    in the standard dict format.

    Parameters
    ----------
    csv_path : str
        Path to the ERDDAP CSV file.
    lon_range, lat_range : (float, float) or None
        Optional additional spatial filter.

    Returns
    -------
    dict  – same format as ``load_drifters_in_region``.
    """
    import csv
    from datetime import datetime

    lons, lats, ves, vns, ve_errs, vn_errs, ssts = [], [], [], [], [], [], []
    times_list, ids_list = [], []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)   # column names
        _units = next(reader)   # units row (skip)
        for row in reader:
            if len(row) < 9:
                continue
            try:
                lon_val = float(row[0])
                lat_val = float(row[1])
                ve_val  = float(row[2]) if row[2] != '' else np.nan
                vn_val  = float(row[3]) if row[3] != '' else np.nan
                eve_val = float(row[4]) if row[4] != '' else 0.01
                evn_val = float(row[5]) if row[5] != '' else 0.01
                sst_val = float(row[6]) if row[6] != '' else np.nan
                time_str = row[7]
                drifter_id = int(float(row[8])) if row[8] != '' else 0
            except (ValueError, IndexError):
                continue

            # Parse time
            try:
                t = datetime.strptime(time_str[:19], "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                try:
                    t = datetime.fromisoformat(time_str[:19])
                except ValueError:
                    continue

            lons.append(lon_val)
            lats.append(lat_val)
            ves.append(ve_val)
            vns.append(vn_val)
            ve_errs.append(eve_val)
            vn_errs.append(evn_val)
            ssts.append(sst_val)
            times_list.append(t)
            ids_list.append(drifter_id)

    all_lon = np.array(lons)
    all_lat = np.array(lats)
    all_ve  = np.array(ves)
    all_vn  = np.array(vns)
    all_ve_err = np.array(ve_errs)
    all_vn_err = np.array(vn_errs)
    # Replace remaining NaN errors with the finite median (robust default)
    finite_ve = all_ve_err[np.isfinite(all_ve_err)]
    finite_vn = all_vn_err[np.isfinite(all_vn_err)]
    if finite_ve.size > 0:
        all_ve_err[~np.isfinite(all_ve_err)] = np.median(finite_ve)
    if finite_vn.size > 0:
        all_vn_err[~np.isfinite(all_vn_err)] = np.median(finite_vn)
    all_sst = np.array(ssts)
    all_ids = np.array(ids_list)
    all_time = np.array(times_list, dtype=object)

    # Filter NaN velocity
    valid = np.isfinite(all_ve) & np.isfinite(all_vn)
    if lon_range is not None:
        valid &= (all_lon >= lon_range[0]) & (all_lon <= lon_range[1])
    if lat_range is not None:
        valid &= (all_lat >= lat_range[0]) & (all_lat <= lat_range[1])

    idx = np.where(valid)[0]
    # Sort by time
    sort_order = np.argsort(all_time[idx])
    idx = idx[sort_order]

    result = {
        'lon':      all_lon[idx],
        'lat':      all_lat[idx],
        've':       all_ve[idx],
        'vn':       all_vn[idx],
        've_error': all_ve_err[idx],
        'vn_error': all_vn_err[idx],
        'sst':      all_sst[idx],
        'time':     all_time[idx],
        'ids':      all_ids[idx],
    }
    n = idx.size
    n_ids = len(np.unique(result['ids'])) if n > 0 else 0
    print(f"[drifter_data] Loaded {n} records from {n_ids} drifters (ERDDAP CSV).")
    return result


# ---------------------------------------------------------------------------
#  1c.  Load drifters from OSMC RealTime CSV (velocities from positions)
# ---------------------------------------------------------------------------
def load_drifters_from_osmc_csv(csv_path, lon_range=None, lat_range=None,
                                 sst_celsius_to_kelvin=True):
    """
    Read a CSV file downloaded from the OSMC_RealTime ERDDAP dataset.

    The OSMC dataset typically has NO velocity (uo/vo) for drifting buoys.
    This function computes velocities from consecutive position reports
    using centred finite differences, exactly as the GDP QC pipeline does.

    Parameters
    ----------
    csv_path : str
        Path to the OSMC CSV file.
    lon_range, lat_range : (float, float) or None
        Optional spatial filter.
    sst_celsius_to_kelvin : bool
        If True, convert SST from °C to Kelvin (add 273.15).

    Returns
    -------
    dict  –  same format as ``load_drifters_from_erddap_csv``.
    """
    import csv
    from datetime import datetime

    # --- Parse CSV ---
    records = {}  # platform_code → list of (time, lat, lon, sst)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        _units = next(reader)

        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            if len(row) < max(col.values()) + 1:
                continue
            try:
                plat = row[col['platform_code']].strip()
                t_str = row[col['time']]
                lat_v = float(row[col['latitude']])
                lon_v = float(row[col['longitude']])
                sst_str = row[col['sst']]
                sst_v = float(sst_str) if sst_str and sst_str != 'NaN' else np.nan
            except (ValueError, KeyError):
                continue
            try:
                t = datetime.strptime(t_str[:19], "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                continue
            records.setdefault(plat, []).append((t, lat_v, lon_v, sst_v))

    # --- Sort each drifter by time ---
    for plat in records:
        records[plat].sort(key=lambda x: x[0])

    # --- Compute velocities via centred finite differences ---
    R_earth = 6.371e6  # metres
    lons, lats, ves, vns, ssts = [], [], [], [], []
    times_list, ids_list = [], []

    for plat, pts in records.items():
        n = len(pts)
        if n < 2:
            continue
        plat_id = int(plat) if plat.isdigit() else hash(plat) % (10**9)

        for i in range(n):
            t_i, lat_i, lon_i, sst_i = pts[i]

            # Centred difference where possible, else forward/backward
            if i == 0:
                t_a, lat_a, lon_a, _ = pts[0]
                t_b, lat_b, lon_b, _ = pts[1]
            elif i == n - 1:
                t_a, lat_a, lon_a, _ = pts[n - 2]
                t_b, lat_b, lon_b, _ = pts[n - 1]
            else:
                t_a, lat_a, lon_a, _ = pts[i - 1]
                t_b, lat_b, lon_b, _ = pts[i + 1]

            dt_s = (t_b - t_a).total_seconds()
            if dt_s <= 0 or dt_s > 86400:
                # Skip if time gap > 24 h or non-positive
                continue

            dlat = np.radians(lat_b - lat_a)
            dlon = np.radians(lon_b - lon_a)
            cos_lat = np.cos(np.radians(lat_i))
            ve_val = R_earth * cos_lat * dlon / dt_s   # eastward m/s
            vn_val = R_earth * dlat / dt_s              # northward m/s

            # Reject unrealistic velocities (> 3 m/s)
            if abs(ve_val) > 3.0 or abs(vn_val) > 3.0:
                continue

            lons.append(lon_i)
            lats.append(lat_i)
            ves.append(ve_val)
            vns.append(vn_val)
            ssts.append(sst_i)
            times_list.append(t_i)
            ids_list.append(plat_id)

    all_lon = np.array(lons)
    all_lat = np.array(lats)
    all_ve  = np.array(ves)
    all_vn  = np.array(vns)
    all_sst = np.array(ssts)
    all_ids = np.array(ids_list)
    all_time = np.array(times_list, dtype=object)

    if sst_celsius_to_kelvin:
        finite_sst = np.isfinite(all_sst)
        all_sst[finite_sst] += 273.15

    # Velocity error estimate: ~0.01 m/s for hourly positions
    all_ve_err = np.full_like(all_ve, 0.01)
    all_vn_err = np.full_like(all_vn, 0.01)

    # Filter by region
    valid = np.isfinite(all_ve) & np.isfinite(all_vn)
    if lon_range is not None:
        valid &= (all_lon >= lon_range[0]) & (all_lon <= lon_range[1])
    if lat_range is not None:
        valid &= (all_lat >= lat_range[0]) & (all_lat <= lat_range[1])

    idx = np.where(valid)[0]
    sort_order = np.argsort(all_time[idx])
    idx = idx[sort_order]

    result = {
        'lon':      all_lon[idx],
        'lat':      all_lat[idx],
        've':       all_ve[idx],
        'vn':       all_vn[idx],
        've_error': all_ve_err[idx],
        'vn_error': all_vn_err[idx],
        'sst':      all_sst[idx],
        'time':     all_time[idx],
        'ids':      all_ids[idx],
    }
    n = idx.size
    n_ids = len(np.unique(result['ids'])) if n > 0 else 0
    print(f"[drifter_data] Loaded {n} records from {n_ids} drifters (OSMC CSV, "
          f"velocities from positions).")
    return result


# ---------------------------------------------------------------------------
#  2.  Load drifters inside a region / time window
# ---------------------------------------------------------------------------
def load_drifters_in_region(
    nc_path,
    lon_range=(-80, -40),
    lat_range=(20, 50),
    time_range=None,
    max_drifters=None,
):
    """
    Read the GDP hourly NetCDF and return drifter records inside the
    specified geographic and temporal bounding box.

    Parameters
    ----------
    nc_path : str
        Path to ``gdp_hourly.nc``.
    lon_range : (float, float)
        (lon_min, lon_max) in degrees East  (use negative for West).
    lat_range : (float, float)
        (lat_min, lat_max) in degrees North.
    time_range : (datetime, datetime) or None
        If given, only keep records within this period.
    max_drifters : int or None
        Cap on number of unique drifter IDs to load (for memory).

    Returns
    -------
    dict with keys: 'lon', 'lat', 've', 'vn', 'time', 'ids'
        Each value is a 1-D array / list.  Records are sorted by time.
    """
    if Dataset is None:
        raise ImportError("netCDF4 is required.  pip install netCDF4")
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"GDP NetCDF not found: {nc_path}")

    print(f"[drifter_data] Opening {nc_path} ...")
    nc = Dataset(nc_path, 'r')

    # Variable names in the GDP hourly NetCDF (v2.01)
    lon  = np.asarray(nc.variables['longitude'][:])
    lat  = np.asarray(nc.variables['latitude'][:])
    ve   = np.asarray(nc.variables['ve'][:])
    vn   = np.asarray(nc.variables['vn'][:])
    t_var = nc.variables['time']
    times = num2date(t_var[:], units=t_var.units, calendar=getattr(t_var, 'calendar', 'standard'))
    ids  = np.asarray(nc.variables['ID'][:])
    nc.close()

    # Spatial mask
    mask = ((lon >= lon_range[0]) & (lon <= lon_range[1]) &
            (lat >= lat_range[0]) & (lat <= lat_range[1]))

    # Temporal mask
    if time_range is not None:
        t0, t1 = time_range
        tmask = np.array([(t0 <= t <= t1) for t in times])
        mask = mask & tmask

    idx = np.where(mask)[0]
    if idx.size == 0:
        warnings.warn("No drifter records found in the requested region/time window.")
        return {'lon': np.array([]), 'lat': np.array([]),
                've': np.array([]), 'vn': np.array([]),
                'time': np.array([]), 'ids': np.array([])}

    # Optional cap on number of unique drifters
    sel_ids = ids[idx]
    if max_drifters is not None:
        unique_ids = np.unique(sel_ids)
        if len(unique_ids) > max_drifters:
            unique_ids = unique_ids[:max_drifters]
            id_mask = np.isin(sel_ids, unique_ids)
            idx = idx[id_mask]

    # Sort by time
    sort_order = np.argsort([times[i] for i in idx])
    idx = idx[sort_order]

    result = {
        'lon':  lon[idx],
        'lat':  lat[idx],
        've':   ve[idx],
        'vn':   vn[idx],
        'time': np.array([times[i] for i in idx]),
        'ids':  ids[idx],
    }
    n = idx.size
    n_ids = len(np.unique(result['ids']))
    print(f"[drifter_data] Loaded {n} records from {n_ids} drifters.")
    return result


# ---------------------------------------------------------------------------
#  2b.  Load drifters from per-drifter .npy files
# ---------------------------------------------------------------------------
def load_drifters_from_npy(
    data_dir,
    lon_range=None,
    lat_range=None,
    dt_hours=1.0,
    t0=None,
):
    """
    Read per-drifter ``.npy`` files produced by preprocessing scripts.

    Expected layout inside *data_dir*::

        floater_x00000.npy   –  longitudes, shape (ntime,)
        floater_y00000.npy   –  latitudes,  shape (ntime,)
        floater_ve00000.npy  –  east velocity (m/s)
        floater_vn00000.npy  –  north velocity (m/s)
        floater_ve_error00000.npy  –  error / std of ve
        floater_vn_error00000.npy  –  error / std of vn
        ...  (different suffix numbers = different drifters)

    Parameters
    ----------
    data_dir : str
        Folder containing the ``.npy`` files.
    lon_range, lat_range : (float, float) or None
        Optional geographic filter (applied per-record).
    dt_hours : float
        Time spacing between consecutive entries (hours).  Default 1.
    t0 : datetime or None
        Start time for the first record.  If None, uses 2020-01-01 00:00.

    Returns
    -------
    dict  – same format as ``load_drifters_in_region``.
    """
    if t0 is None:
        t0 = datetime(2020, 1, 1)

    # Discover drifters by x-files
    x_files = sorted(glob.glob(os.path.join(data_dir, "floater_x*.npy")))
    if not x_files:
        raise FileNotFoundError(
            f"No floater_x*.npy files found in {data_dir}")

    all_lon, all_lat, all_ve, all_vn = [], [], [], []
    all_ve_err, all_vn_err = [], []
    all_time, all_ids = [], []

    for xf in x_files:
        # Extract the drifter suffix (e.g. "00000")
        base = os.path.basename(xf)              # floater_x00000.npy
        suffix = base.replace("floater_x", "").replace(".npy", "")

        yf      = os.path.join(data_dir, f"floater_y{suffix}.npy")
        vef     = os.path.join(data_dir, f"floater_ve{suffix}.npy")
        vnf     = os.path.join(data_dir, f"floater_vn{suffix}.npy")
        vee_f   = os.path.join(data_dir, f"floater_ve_error{suffix}.npy")
        vne_f   = os.path.join(data_dir, f"floater_vn_error{suffix}.npy")

        x_arr = np.load(xf)
        y_arr = np.load(yf) if os.path.exists(yf) else np.full_like(x_arr, np.nan)
        ve_arr = np.load(vef) if os.path.exists(vef) else np.zeros_like(x_arr)
        vn_arr = np.load(vnf) if os.path.exists(vnf) else np.zeros_like(x_arr)
        ve_err = np.load(vee_f) if os.path.exists(vee_f) else np.full_like(x_arr, 0.01)
        vn_err = np.load(vne_f) if os.path.exists(vne_f) else np.full_like(x_arr, 0.01)

        ntime = len(x_arr)
        times = [t0 + timedelta(hours=dt_hours * k) for k in range(ntime)]
        drifter_id = int(suffix) if suffix.isdigit() else hash(suffix) % (10**8)

        all_lon.append(x_arr)
        all_lat.append(y_arr)
        all_ve.append(ve_arr)
        all_vn.append(vn_arr)
        all_ve_err.append(ve_err)
        all_vn_err.append(vn_err)
        all_time.extend(times)
        all_ids.extend([drifter_id] * ntime)

    all_lon = np.concatenate(all_lon)
    all_lat = np.concatenate(all_lat)
    all_ve  = np.concatenate(all_ve)
    all_vn  = np.concatenate(all_vn)
    all_ve_err = np.concatenate(all_ve_err)
    all_vn_err = np.concatenate(all_vn_err)
    all_ids = np.array(all_ids)

    # Optional spatial filter
    mask = np.ones(len(all_lon), dtype=bool)
    if lon_range is not None:
        mask &= (all_lon >= lon_range[0]) & (all_lon <= lon_range[1])
    if lat_range is not None:
        mask &= (all_lat >= lat_range[0]) & (all_lat <= lat_range[1])

    # Remove NaNs
    mask &= np.isfinite(all_lon) & np.isfinite(all_lat)

    idx = np.where(mask)[0]
    all_time = np.array(all_time, dtype=object)

    result = {
        'lon':      all_lon[idx],
        'lat':      all_lat[idx],
        've':       all_ve[idx],
        'vn':       all_vn[idx],
        've_error': all_ve_err[idx],
        'vn_error': all_vn_err[idx],
        'time':     all_time[idx],
        'ids':      all_ids[idx],
    }
    n = idx.size
    n_ids = len(np.unique(result['ids']))
    print(f"[drifter_data] Loaded {n} records from {n_ids} drifters (npy).")
    return result


# ---------------------------------------------------------------------------
#  2c.  Load drifters from interpolated .h5 file
# ---------------------------------------------------------------------------
def load_drifters_from_h5(
    h5_path,
    lon_grid=None,
    lat_grid=None,
    dt_hours=1.0,
    t0=None,
    fields_per_drifter=6,
):
    """
    Read an HDF5 file with one dataset per time step, produced by the
    preprocessing pipeline (e.g. ``floater_real.h5``).

    Expected structure::

        /t_00000000   →  1-D array of length (n_drifters * fields_per_drifter)
        /t_00000001   →  ...
        ...

    The 1-D array stores, for each drifter at that time step:
        [x, y, ve, vn, ve_error, vn_error]   (if fields_per_drifter == 6)
    or  [x, y, ve, vn]                        (if fields_per_drifter == 4)

    Parameters
    ----------
    h5_path : str
        Path to the ``.h5`` file.
    lon_grid, lat_grid : 1-D arrays or None
        If provided, only keep drifters inside the grid bounding box.
    dt_hours : float
        Time spacing between consecutive time steps (hours).
    t0 : datetime or None
        Time for step 0.  Default: 2020-01-01 00:00.
    fields_per_drifter : int
        Number of fields per drifter per time step (4 or 6).

    Returns
    -------
    dict  – same format as ``load_drifters_in_region``, with extra
            ``ve_error`` / ``vn_error`` keys when available.
    """
    if h5py is None:
        raise ImportError("h5py is required.  pip install h5py")
    if t0 is None:
        t0 = datetime(2020, 1, 1)

    all_lon, all_lat, all_ve, all_vn = [], [], [], []
    all_ve_err, all_vn_err = [], []
    all_time, all_ids = [], []

    with h5py.File(h5_path, 'r') as hf:
        keys = sorted(hf.keys())
        for step_idx, key in enumerate(keys):
            arr = np.asarray(hf[key])
            if arr.size == 0:
                continue
            n_drifters = arr.size // fields_per_drifter
            if n_drifters == 0:
                continue

            data = arr.reshape(n_drifters, fields_per_drifter)
            xs   = data[:, 0]
            ys   = data[:, 1]
            ves  = data[:, 2]
            vns  = data[:, 3]
            if fields_per_drifter >= 6:
                ve_errs = data[:, 4]
                vn_errs = data[:, 5]
            else:
                ve_errs = np.full(n_drifters, 0.01)
                vn_errs = np.full(n_drifters, 0.01)

            t_step = t0 + timedelta(hours=dt_hours * step_idx)

            all_lon.append(xs)
            all_lat.append(ys)
            all_ve.append(ves)
            all_vn.append(vns)
            all_ve_err.append(ve_errs)
            all_vn_err.append(vn_errs)
            all_time.extend([t_step] * n_drifters)
            all_ids.extend(range(n_drifters))  # id = index within step

    if len(all_lon) == 0:
        warnings.warn("No drifter records found in h5 file.")
        return {'lon': np.array([]), 'lat': np.array([]),
                've': np.array([]), 'vn': np.array([]),
                'time': np.array([]), 'ids': np.array([])}

    all_lon    = np.concatenate(all_lon)
    all_lat    = np.concatenate(all_lat)
    all_ve     = np.concatenate(all_ve)
    all_vn     = np.concatenate(all_vn)
    all_ve_err = np.concatenate(all_ve_err)
    all_vn_err = np.concatenate(all_vn_err)
    all_ids    = np.array(all_ids)
    all_time   = np.array(all_time, dtype=object)

    # Optional spatial filter
    mask = np.ones(len(all_lon), dtype=bool)
    if lon_grid is not None:
        mask &= (all_lon >= lon_grid.min()) & (all_lon <= lon_grid.max())
    if lat_grid is not None:
        mask &= (all_lat >= lat_grid.min()) & (all_lat <= lat_grid.max())
    mask &= np.isfinite(all_lon) & np.isfinite(all_lat)

    idx = np.where(mask)[0]
    result = {
        'lon':      all_lon[idx],
        'lat':      all_lat[idx],
        've':       all_ve[idx],
        'vn':       all_vn[idx],
        've_error': all_ve_err[idx],
        'vn_error': all_vn_err[idx],
        'time':     all_time[idx],
        'ids':      all_ids[idx],
    }
    n = idx.size
    n_ids = len(np.unique(result['ids'])) if n > 0 else 0
    print(f"[drifter_data] Loaded {n} records from {n_ids} drifters (h5).")
    return result


# ---------------------------------------------------------------------------
#  3.  Convert drifters to gridded observation arrays
# ---------------------------------------------------------------------------
def drifters_to_obs_arrays(
    drifter_data,
    lon_grid,
    lat_grid,
    obs_times,
    obs_type="velocity",
    time_tolerance_s=1800,
    **kwargs,
):
    """
    Snap drifter observations onto the model grid and group by observation
    time.

    Parameters
    ----------
    drifter_data : dict
        Output of ``load_drifters_in_region``.
    lon_grid : 1-D array (nx,)
        Longitude values of grid columns (degrees E).
    lat_grid : 1-D array (ny,)
        Latitude values of grid rows (degrees N), **south-to-north**.
    obs_times : array of datetime-like
        Assimilation times.
    obs_type : str
        'velocity' →  observe (u, v);   'position' →  observe (lon, lat) diffs
        (velocity derived from position displacement).  Default: 'velocity'.
    time_tolerance_s : float
        Maximum time difference (seconds) to snap a drifter record to an
        obs_time.

    Returns
    -------
    yobs_all : list of 1-D arrays
        Observation vectors at each assimilation time.  For velocity type:
        [ve_0, vn_0, ve_1, vn_1, ...] flattened over drifters present.
    yobs_ind_all : list of 1-D int arrays
        Flat indices into the model state vector [h, u, v] indicating
        which state components are observed.  Velocity drifters observe
        u- and v-components at the nearest grid cell.
    yobs_ind_level0_all : list of 1-D int arrays
        The grid-cell (spatial, single-field) flat index for each observation.
    sig_y_all : list of 1-D arrays
        Per-observation noise std dev (from drifter error fields).
        Same length as yobs_all.  The ERDDAP fields ``err_ve``, ``err_vn``
        are 95 % confidence intervals →  σ = CI / 1.96.
    """
    ny = len(lat_grid)
    nx = len(lon_grid)
    ncells = ny * nx
    d_lon = drifter_data['lon']
    d_lat = drifter_data['lat']
    d_ve  = drifter_data['ve']
    d_vn  = drifter_data['vn']
    d_time = drifter_data['time']

    # Per-drifter error (95% CI → std dev)
    has_errors = ('ve_error' in drifter_data and 'vn_error' in drifter_data)
    if has_errors:
        d_eve = drifter_data['ve_error'] / 1.96
        d_evn = drifter_data['vn_error'] / 1.96
        default_sig = max(np.nanmedian(d_eve), np.nanmedian(d_evn), 0.005)
    else:
        d_eve = d_evn = None
        default_sig = 0.01

    # SST data (for 'velocity_sst' obs_type)
    has_sst = ('sst' in drifter_data)
    if has_sst:
        d_sst = drifter_data['sst']
    else:
        d_sst = None
    # SST observation noise: use caller-provided value, or data-driven default
    sig_sst = float(kwargs.get('sig_sst', 0.1))
    if isinstance(drifter_data, dict) and 'sig_sst' in drifter_data:
        sig_sst = float(drifter_data['sig_sst'])

    yobs_all = []
    yobs_ind_all = []
    yobs_ind_level0_all = []
    sig_y_all = []

    for t_obs in obs_times:
        # find drifter records within time_tolerance of t_obs
        dt_sec = np.array([(d_time[k] - t_obs).total_seconds()
                           if hasattr(d_time[k] - t_obs, 'total_seconds')
                           else 0.0
                           for k in range(len(d_time))])
        sel = np.where(np.abs(dt_sec) <= time_tolerance_s)[0]
        if sel.size == 0:
            yobs_all.append(np.array([], dtype=np.float64))
            yobs_ind_all.append(np.array([], dtype=int))
            yobs_ind_level0_all.append(np.array([], dtype=int))
            sig_y_all.append(np.array([], dtype=np.float64))
            continue

        # Nearest-grid-point mapping
        obs_vals = []
        obs_state_inds = []
        obs_cell_inds = []
        obs_sigs = []

        for k in sel:
            ix = int(np.argmin(np.abs(lon_grid - d_lon[k])))
            iy = int(np.argmin(np.abs(lat_grid - d_lat[k])))
            cell_flat = iy * nx + ix

            if obs_type in ("velocity", "velocity_sst"):
                # observe u → offset 1*ncells;  v → offset 2*ncells
                obs_vals.append(d_ve[k])
                obs_state_inds.append(1 * ncells + cell_flat)  # u
                obs_cell_inds.append(cell_flat)
                s_ve = d_eve[k] if (d_eve is not None and np.isfinite(d_eve[k])) else default_sig
                obs_sigs.append(s_ve)

                obs_vals.append(d_vn[k])
                obs_state_inds.append(2 * ncells + cell_flat)  # v
                obs_cell_inds.append(cell_flat)
                s_vn = d_evn[k] if (d_evn is not None and np.isfinite(d_evn[k])) else default_sig
                obs_sigs.append(s_vn)

                # SST observation → offset 3*ncells
                if obs_type == "velocity_sst" and d_sst is not None:
                    sst_val = d_sst[k]
                    if np.isfinite(sst_val):
                        obs_vals.append(sst_val)
                        obs_state_inds.append(3 * ncells + cell_flat)  # T
                        obs_cell_inds.append(cell_flat)
                        obs_sigs.append(sig_sst)

            elif obs_type == "sst":
                # SST-only observations → offset 3*ncells
                if d_sst is not None:
                    sst_val = d_sst[k]
                    if np.isfinite(sst_val):
                        obs_vals.append(sst_val)
                        obs_state_inds.append(3 * ncells + cell_flat)  # T
                        obs_cell_inds.append(cell_flat)
                        obs_sigs.append(sig_sst)

            else:
                # position-type: treat (lon, lat) displacement as pseudo-obs
                obs_vals.append(d_lon[k])
                obs_state_inds.append(cell_flat)
                obs_cell_inds.append(cell_flat)
                obs_sigs.append(default_sig)

                obs_vals.append(d_lat[k])
                obs_state_inds.append(cell_flat)
                obs_cell_inds.append(cell_flat)
                obs_sigs.append(default_sig)

        yobs_all.append(np.array(obs_vals, dtype=np.float64))
        yobs_ind_all.append(np.array(obs_state_inds, dtype=int))
        yobs_ind_level0_all.append(np.array(obs_cell_inds, dtype=int))
        sig_y_all.append(np.array(obs_sigs, dtype=np.float64))

    return yobs_all, yobs_ind_all, yobs_ind_level0_all, sig_y_all


# ---------------------------------------------------------------------------
#  4.  Build observation NetCDF  (for offline use by the DA code)
# ---------------------------------------------------------------------------
def build_obs_netcdf(
    outpath,
    yobs_all,
    yobs_ind_all,
    yobs_ind_level0_all,
    obs_times,
    sig_y=0.01,
    sig_y_all=None,
    attrs=None,
):
    """
    Write observation arrays to a NetCDF that the local-SMCMC filter can read.

    Parameters
    ----------
    outpath : str
    yobs_all, yobs_ind_all, yobs_ind_level0_all :
        Lists returned by ``drifters_to_obs_arrays``.
    obs_times : list/array of floats or datetimes
        Observation times (will be written as float seconds if datetime).
    sig_y : float
        Observation noise standard deviation.
    attrs : dict or None
        Extra global attributes to store.
    """
    if Dataset is None:
        raise ImportError("netCDF4 required")

    nassim = len(yobs_all)
    # Pad ragged arrays to the same length (fill with -1 for indices, NaN for values)
    # Note: yobs_all and yobs_ind_all have the same length per time step
    # (e.g. 2 entries per cell for u,v), but yobs_ind_level0_all may be
    # shorter (1 entry per observed cell).  Use separate max sizes.
    max_nobs = max((len(y) for y in yobs_all), default=0)
    max_nobs_ind0 = max((len(y) for y in yobs_ind_level0_all), default=0)
    if max_nobs == 0:
        warnings.warn("No observations to write!")
        return

    yobs_arr  = np.full((nassim, max_nobs), np.nan, dtype=np.float64)
    yind_arr  = np.full((nassim, max_nobs), -1, dtype=np.int32)
    yind0_arr = np.full((nassim, max_nobs_ind0), -1, dtype=np.int32)

    for i in range(nassim):
        n = len(yobs_all[i])
        yobs_arr[i, :n] = yobs_all[i]
        n_ind = len(yobs_ind_all[i])
        yind_arr[i, :n_ind] = yobs_ind_all[i]
        n_ind0 = len(yobs_ind_level0_all[i])
        yind0_arr[i, :n_ind0] = yobs_ind_level0_all[i]

    with Dataset(outpath, 'w', format='NETCDF4') as nc:
        nc.createDimension('nassim', nassim)
        nc.createDimension('max_nobs', max_nobs)
        nc.createDimension('max_nobs_ind0', max_nobs_ind0)
        nc.sig_y = sig_y

        v1 = nc.createVariable('yobs_all', 'f8', ('nassim', 'max_nobs'), zlib=True)
        v1[:] = yobs_arr
        v2 = nc.createVariable('yobs_ind_all', 'i4', ('nassim', 'max_nobs'), zlib=True)
        v2[:] = yind_arr
        v3 = nc.createVariable('yobs_ind_level0_all', 'i4', ('nassim', 'max_nobs_ind0'), zlib=True)
        v3[:] = yind0_arr

        # Per-observation noise std dev (if provided)
        if sig_y_all is not None:
            sigy_arr = np.full((nassim, max_nobs), np.nan, dtype=np.float64)
            for i in range(nassim):
                n = len(sig_y_all[i])
                sigy_arr[i, :n] = sig_y_all[i]
            v4 = nc.createVariable('sig_y_all', 'f8', ('nassim', 'max_nobs'), zlib=True)
            v4[:] = sigy_arr
            v4.long_name = "per-observation noise std dev (sigma_y)"
            v4.units = "m/s"
            # Also store the scalar mean for backward compatibility
            flat_sigs = np.concatenate([s for s in sig_y_all if len(s) > 0])
            nc.sig_y = float(np.nanmean(flat_sigs))
            nc.sig_y_median = float(np.nanmedian(flat_sigs))
        else:
            nc.sig_y = sig_y

        # store times as float seconds since epoch
        t_arr = np.zeros(nassim, dtype=np.float64)
        for i, t in enumerate(obs_times):
            if isinstance(t, (datetime,)):
                t_arr[i] = (t - datetime(1970, 1, 1)).total_seconds()
            else:
                t_arr[i] = float(t)
        vt = nc.createVariable('obs_times', 'f8', ('nassim',), zlib=True)
        vt[:] = t_arr
        vt.units = "seconds since 1970-01-01"

        if attrs:
            for k, val in attrs.items():
                setattr(nc, k, val)

    print(f"[drifter_data] Wrote observation file: {outpath}  ({nassim} times, max_nobs={max_nobs})")

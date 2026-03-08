# -*- coding: utf-8 -*-
"""
SWOT SSH observation module.

Provides two modes:
  1. **Real SWOT**: Download SWOT L2 LR SSH data from PO.DAAC via earthaccess,
     interpolate swath data onto the model grid.
  2. **Synthetic SWOT**: Generate OSSE-style SSH observations from HYCOM
     reference data with realistic Gaussian noise and optional swath-like
     coverage patterns.

Both modes produce observation arrays compatible with the existing
drifter_data.build_obs_netcdf format:
  - yobs_all        : list of 1-D arrays (SSH values in metres)
  - yobs_ind_all    : list of 1-D int arrays (state-vector indices, 0..ncells-1 for h)
  - yobs_ind0_all   : list of 1-D int arrays (grid-cell flat indices)
  - sig_y_all       : list of 1-D arrays (per-obs noise σ)
"""

import os
import warnings
import numpy as np
from datetime import datetime, timedelta

try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None


# ============================================================================
#  0.  Fill empty SWOT cycles with HYCOM SSH + noise
# ============================================================================

def _generate_swath_schedule(nassim, lon_grid, track_a_start_lon=-23.5,
                              track_sep=25.0, first_cycle=7,
                              repeat_periods=(11, 13), drift_per=-1.5):
    """Generate a schedule of synthetic SWOT swath center longitudes.

    Returns dict mapping cycle -> list of center longitudes.
    """
    lon_min, lon_max = lon_grid[0], lon_grid[-1]
    lon_range = lon_max - lon_min

    schedule = {}
    current_lon_a = track_a_start_lon
    current_cycle = first_cycle
    period_idx = 0

    track_a_events = []
    while current_cycle < nassim:
        track_a_events.append((current_cycle, current_lon_a))
        gap = repeat_periods[period_idx % len(repeat_periods)]
        current_cycle += gap
        current_lon_a += drift_per
        if current_lon_a < lon_min:
            current_lon_a += lon_range
        period_idx += 1

    for cyc_a, lon_a in track_a_events:
        cyc_b = cyc_a + 2
        lon_b = lon_a - track_sep
        if lon_b < lon_min:
            lon_b += lon_range
        schedule.setdefault(cyc_a, []).append(lon_a)
        if cyc_b < nassim:
            schedule.setdefault(cyc_b, []).append(lon_b)

    return schedule


def _make_swath_cells(center_lon, lon_grid, nx, ny,
                      swath_width=3, tilt=0.10):
    """Return flat cell indices for a thin, tilted SWOT-like swath.

    Parameters
    ----------
    center_lon : float
        Longitude of swath centre at iy=0.
    swath_width : int
        Width in grid cells (default 3 ≈ 1.5° ≈ 120 km).
    tilt : float
        Longitude shift per latitude row (degrees).
        Positive = swath leans eastward with increasing latitude.
    """
    half_w = swath_width // 2
    cells = []
    for iy in range(ny):
        lon_c = center_lon + iy * tilt
        ix_center = int(np.argmin(np.abs(lon_grid - lon_c)))
        for dix in range(-half_w, half_w + 1):
            ix = ix_center + dix
            if 0 <= ix < nx:
                cells.append(iy * nx + ix)
    return np.array(cells, dtype=int)


def fill_empty_cycles_with_hycom(
    ssh_yobs,
    ssh_ind,
    ssh_ind0,
    ssh_sigy,
    ssh_ref_3d,
    ssh_ref_times,
    obs_epoch_times,
    H_b,
    lon_grid,
    lat_grid,
    sig_ssh=0.5,
    obs_fraction=0.15,
    seed=99,
    min_obs_frac=0.5,
    swath_width=3,
    swath_tilt=0.10,
):
    """
    For assimilation cycles that have NO or FEW real SWOT SSH observations,
    generate synthetic SSH obs from the HYCOM reference field + small noise.

    Synthetic observations are placed in **thin, tilted SWOT-like swath
    bands** mimicking real satellite passes (~200 obs per cycle, 3 cells
    wide, slightly tilted from vertical).

    A cycle is considered "sparse" if its observation count is below
    ``min_obs_frac`` times the median count of cycles that *do* have
    observations.  Both empty and sparse cycles are replaced with
    HYCOM-based synthetic observations to keep SSH anchored.

    Parameters
    ----------
    ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy : lists of arrays
        Existing SWOT SSH obs arrays (length nassim).
    ssh_ref_3d : (ntime_ref, ny, nx) array
        HYCOM SSH reference fields (metres) from the BC file.
    ssh_ref_times : 1-D array
        Epoch times (seconds) for each ssh_ref_3d snapshot.
    obs_epoch_times : 1-D array or list
        Epoch time (seconds) for each assimilation cycle.
    H_b : (ny, nx) array
        Bathymetry (total depth, positive).
    lon_grid, lat_grid : 1-D arrays
        Model grid coordinates.
    sig_ssh : float
        Observation noise σ (metres) for synthetic obs.
    obs_fraction : float
        Fraction of grid cells to observe per filled cycle (unused now,
        kept for API compatibility).
    seed : int
        Random seed for reproducibility.
    min_obs_frac : float
        Cycles with fewer than ``min_obs_frac * median(non-zero counts)``
        observations are replaced.  Default 0.5 = 50 % of median.
    swath_width : int
        Width of synthetic SWOT swath in grid cells.  Default 3 (~1.5°
        ≈ 120 km on a 0.5° grid), matching real KaRIn swath width.
    swath_tilt : float
        Longitude shift per latitude row (degrees).  Default 0.10
        gives ~12° tilt from vertical, mimicking orbit inclination.

    Returns
    -------
    ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy : lists of arrays  (modified in-place
        and returned)
    n_filled : int
        Number of cycles that were filled.
    """
    ny, nx = len(lat_grid), len(lon_grid)
    ncells = ny * nx
    H_b_flat = np.asarray(H_b, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    nassim = len(ssh_yobs)

    # Compute threshold: cycles with fewer obs than this get filled
    counts = np.array([len(a) for a in ssh_yobs])
    nonzero = counts[counts > 0]
    if len(nonzero) > 0:
        threshold = int(min_obs_frac * np.median(nonzero))
    else:
        threshold = 0  # no real obs at all — fill everything

    # Build swath schedule (same orbit model as generate_synthetic_swot.py)
    schedule = _generate_swath_schedule(nassim, lon_grid)
    sched_cycles = sorted(schedule.keys())
    lon_drift = -1.5  # per appearance

    n_filled = 0
    for c in range(nassim):
        if len(ssh_yobs[c]) >= max(threshold, 1):
            continue  # this cycle has enough real SWOT data

        # Determine swath centre longitude(s) for this cycle
        if c in schedule:
            swath_lons = schedule[c]
        else:
            # Interpolate between nearest scheduled cycles
            before = [s for s in sched_cycles if s <= c]
            after  = [s for s in sched_cycles if s > c]
            if before and after:
                cb, ca = before[-1], after[0]
                frac = (c - cb) / (ca - cb)
                interp_lon = schedule[cb][0] + frac * (schedule[ca][0] - schedule[cb][0])
                swath_lons = [interp_lon]
            elif before:
                swath_lons = [schedule[before[-1]][0] + lon_drift * 0.1]
            else:
                swath_lons = [schedule[after[0]][0]]

        # Collect cells from all swaths in this cycle
        all_cells = []
        for sl in swath_lons:
            cells = _make_swath_cells(sl, lon_grid, nx, ny,
                                     swath_width=swath_width,
                                     tilt=swath_tilt)
            all_cells.append(cells)
        obs_cells = np.unique(np.concatenate(all_cells))

        # Find nearest HYCOM SSH snapshot
        t_c = float(obs_epoch_times[c])
        idx = int(np.searchsorted(ssh_ref_times, t_c, side='right')) - 1
        idx = max(0, min(idx, len(ssh_ref_times) - 1))
        ssh_field = ssh_ref_3d[idx]  # (ny, nx)  SSH in metres
        ssh_flat = ssh_field.ravel()

        # h_obs = H_b(cell) + SSH_hycom + noise
        h_obs = H_b_flat[obs_cells] + ssh_flat[obs_cells] + \
                rng.normal(0.0, sig_ssh, size=len(obs_cells))

        ssh_yobs[c] = h_obs.astype(np.float64)
        ssh_ind[c] = obs_cells.astype(int)
        ssh_ind0[c] = obs_cells.astype(int)
        ssh_sigy[c] = np.full(len(obs_cells), sig_ssh, dtype=np.float64)
        n_filled += 1

    return ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy, n_filled


# ============================================================================
#  1.  Synthetic SWOT SSH observations (OSSE mode)
# ============================================================================

def generate_synthetic_ssh_obs(
    lon_grid,
    lat_grid,
    obs_times,
    ssh_reference,
    H_mean=4000.0,
    H_b=None,
    sig_ssh=0.05,
    obs_fraction=0.15,
    swath_width_km=120.0,
    seed=42,
):
    """
    Generate synthetic SWOT-like SSH observations from a reference SSH field.

    In real SWOT, the satellite observes narrow swaths (120 km wide, two sides
    of the nadir track) with a 21-day repeat cycle.  For the OSSE, we
    approximate this by randomly observing a fraction of grid cells at each
    assimilation time.

    The model state stores total water column depth h = H_b(x,y) + SSH.
    Observations are of h directly:  y = H_b(cell) + SSH_true + ε.
    Because the observation operator is *linear* — y = H·x  where H selects
    h at the observed cell — the DA filter handles this natively.

    **Important**: ``H_b`` (spatially varying bathymetry) should be provided
    so that observed h values match the model's definition of h.  If ``H_b``
    is None, falls back to the constant ``H_mean`` (only correct for flat
    bathymetry).

    To integrate with the existing obs indexing:
      obs_state_ind[i] = cell_index     (h is field 0, offset 0 in state vector)
      obs_cell_ind[i]  = cell_index     (spatial position for localisation)

    Parameters
    ----------
    lon_grid : 1-D array (nx,)
    lat_grid : 1-D array (ny,)
    obs_times : list  (length nassim)
        Assimilation times (only length matters for synthetic obs).
    ssh_reference : callable or (nassim, ny, nx) array
        If callable: ssh_reference(t_idx) → (ny, nx) SSH field (metres).
        If array: ssh_reference[t_idx] is the SSH field at assimilation time t_idx.
        SSH = h - H_b, so values should be O(0.01 – 1 m).
    H_mean : float
        Fallback mean total depth (m) when H_b is not provided.
    H_b : ndarray (ny, nx) or None
        Spatially varying bathymetry (ocean depth, positive downward).
        When provided, obs value = H_b(cell) + SSH_true + noise.
        When None, obs value = H_mean + SSH_true + noise.
    sig_ssh : float
        SWOT SSH observation noise standard deviation (metres).
        Typical values: 0.02 – 0.05 m for the 2 km KaRIn product.
    obs_fraction : float in (0, 1]
        Fraction of grid cells observed per assimilation cycle.
        E.g. 0.15 ≈ one passage per cycle for a 21-day repeat orbit
        over a large Atlantic domain.
    swath_width_km : float
        Not used in the simple random mode but reserved for future
        realistic swath pattern generation.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    yobs_all, yobs_ind_all, yobs_ind0_all, sig_y_all :
        Same format as drifters_to_obs_arrays output.
    """
    ny, nx = len(lat_grid), len(lon_grid)
    ncells = ny * nx
    rng = np.random.default_rng(seed)

    # Build flat H_b array for per-cell depth lookup
    if H_b is not None:
        H_b_flat = np.asarray(H_b, dtype=np.float64).ravel()
        assert H_b_flat.shape[0] == ncells, (
            f"H_b has {H_b_flat.shape[0]} cells, expected {ncells}")
    else:
        # Fallback: use constant H_mean everywhere
        H_b_flat = np.full(ncells, H_mean, dtype=np.float64)

    yobs_all = []
    yobs_ind_all = []
    yobs_ind0_all = []
    sig_y_all = []

    nassim = len(obs_times)
    nobs_per_cycle = max(1, int(obs_fraction * ncells))

    for t_idx in range(nassim):
        # Get reference SSH field at this time
        if callable(ssh_reference):
            ssh_field = ssh_reference(t_idx)
        else:
            ssh_field = ssh_reference[t_idx]

        ssh_flat = ssh_field.ravel()  # (ncells,)

        # Select random subset of cells to observe (simulating swath coverage)
        obs_cells = rng.choice(ncells, size=nobs_per_cycle, replace=False)
        obs_cells.sort()

        # Observed value = H_b(cell) + SSH_true + noise
        # State vector stores h (total depth) = H_b + SSH, so the observation
        # is of h directly at each cell, using the ACTUAL bathymetry at that cell.
        ssh_true = ssh_flat[obs_cells]
        h_true = H_b_flat[obs_cells] + ssh_true
        noise = rng.normal(0.0, sig_ssh, size=nobs_per_cycle)
        h_obs = h_true + noise

        # State-vector indices: h field is at offset 0 → index = cell_index
        obs_state_inds = obs_cells.copy()
        obs_cell_inds = obs_cells.copy()
        obs_sigs = np.full(nobs_per_cycle, sig_ssh)

        yobs_all.append(h_obs.astype(np.float64))
        yobs_ind_all.append(obs_state_inds.astype(int))
        yobs_ind0_all.append(obs_cell_inds.astype(int))
        sig_y_all.append(obs_sigs.astype(np.float64))

    return yobs_all, yobs_ind_all, yobs_ind0_all, sig_y_all


# ============================================================================
#  2.  Build SSH reference from HYCOM BC file
# ============================================================================

def build_ssh_reference_from_hycom(
    bc_file,
    lon_grid,
    lat_grid,
    obs_times,
    cache_dir=None,
):
    """
    Extract time-varying SSH reference fields from the HYCOM BC NetCDF.

    Parameters
    ----------
    bc_file : str
        Path to HYCOM BC NetCDF (must contain 'ssh', 'time', 'lon', 'lat').
    lon_grid, lat_grid : 1-D arrays
        Model grid coordinates.
    obs_times : list of datetime
        Assimilation times.
    cache_dir : str or None
        If set, cache interpolated fields to .npy files.

    Returns
    -------
    ssh_ref : (nassim, ny, nx) array
        SSH reference in metres at each assimilation time.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    import cftime

    ny, nx = len(lat_grid), len(lon_grid)
    nassim = len(obs_times)

    # Check cache
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, f'hycom_ssh_ref_{ny}x{nx}_{nassim}steps.npy')
        if os.path.exists(cache_path):
            print(f"[swot_ssh_data] Loading cached SSH reference: {cache_path}")
            return np.load(cache_path)
    else:
        cache_path = None

    if Dataset is None:
        raise ImportError("netCDF4 required for HYCOM BC access")

    nc = Dataset(bc_file, 'r')
    bc_lon = np.asarray(nc.variables['lon'][:], dtype=np.float64)
    bc_lat = np.asarray(nc.variables['lat'][:], dtype=np.float64)

    # Read times
    time_var = nc.variables['time']
    time_units = time_var.units
    time_cal = getattr(time_var, 'calendar', 'standard')
    bc_times_raw = np.asarray(time_var[:])
    bc_datetimes = cftime.num2date(bc_times_raw, time_units, time_cal)
    # Convert cftime to standard datetime for comparison
    bc_dt_std = np.array([datetime(d.year, d.month, d.day,
                                   d.hour, d.minute, d.second)
                          for d in bc_datetimes])

    # Read all SSH snapshots
    ssh_all = np.asarray(nc.variables['ssh'][:], dtype=np.float64)
    if hasattr(nc.variables['ssh'][0], 'filled'):
        ssh_all = np.where(ssh_all < -1000, np.nan, ssh_all)
    nc.close()

    ntime_bc = ssh_all.shape[0]

    def fill_nan(arr2d):
        mask = np.isnan(arr2d)
        if not mask.any():
            return arr2d
        if mask.all():
            arr2d[:] = 0.0
            return arr2d
        ind = distance_transform_edt(mask, return_distances=False,
                                     return_indices=True)
        arr2d[mask] = arr2d[tuple(ind[:, mask])]
        return arr2d

    def interp_to_grid(field_2d):
        interp = RegularGridInterpolator(
            (bc_lat, bc_lon), field_2d,
            method='linear', bounds_error=False, fill_value=None)
        pts = np.array(np.meshgrid(lat_grid, lon_grid, indexing='ij'))
        pts = pts.reshape(2, -1).T
        return interp(pts).reshape(ny, nx)

    # For each obs_time, find nearest HYCOM BC time and interpolate
    ssh_ref = np.zeros((nassim, ny, nx), dtype=np.float64)

    for t_idx in range(nassim):
        t_obs = obs_times[t_idx]
        if isinstance(t_obs, datetime):
            diffs = np.array([(t_obs - bdt).total_seconds() for bdt in bc_dt_std])
        else:
            diffs = np.array([float(t_obs - bdt) for bdt in bc_dt_std])

        nearest_bc_idx = int(np.argmin(np.abs(diffs)))
        ssh_snap = ssh_all[nearest_bc_idx].copy()
        fill_nan(ssh_snap)
        ssh_ref[t_idx] = interp_to_grid(ssh_snap)

    if cache_path is not None:
        np.save(cache_path, ssh_ref)
        print(f"[swot_ssh_data] Cached SSH reference: {cache_path}")

    print(f"[swot_ssh_data] Built SSH reference from HYCOM: "
          f"shape={ssh_ref.shape}, "
          f"range=[{ssh_ref.min():.3f}, {ssh_ref.max():.3f}] m")

    return ssh_ref


# ============================================================================
#  3.  Merge SSH observations with drifter observations
# ============================================================================

def merge_obs_arrays(
    drift_yobs, drift_ind, drift_ind0, drift_sigy,
    ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy,
):
    """
    Merge drifter (u, v, T) and SWOT SSH observations into a single set
    of observation arrays, per assimilation cycle.

    Both inputs must have the same length (nassim).

    Parameters
    ----------
    drift_yobs, drift_ind, drift_ind0, drift_sigy : lists of arrays
        Drifter observations (from drifters_to_obs_arrays).
    ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy : lists of arrays
        SSH observations (from generate_synthetic_ssh_obs or download_swot_ssh).

    Returns
    -------
    merged_yobs, merged_ind, merged_ind0, merged_sigy : lists of arrays
    """
    nassim = len(drift_yobs)
    assert len(ssh_yobs) == nassim, "Drifter and SSH obs must have same nassim"

    merged_yobs = []
    merged_ind = []
    merged_ind0 = []
    merged_sigy = []

    for i in range(nassim):
        # Concatenate obs from both sources
        merged_yobs.append(np.concatenate([drift_yobs[i], ssh_yobs[i]]))
        merged_ind.append(np.concatenate([drift_ind[i], ssh_ind[i]]))
        merged_ind0.append(np.concatenate([drift_ind0[i], ssh_ind0[i]]))
        merged_sigy.append(np.concatenate([drift_sigy[i], ssh_sigy[i]]))

    return merged_yobs, merged_ind, merged_ind0, merged_sigy


# ============================================================================
#  4.  Download real SWOT L2 LR SSH data from PO.DAAC
# ============================================================================

def download_swot_ssh(
    lon_range,
    lat_range,
    time_range,
    dest_dir='data/swot',
    short_name='SWOT_L2_LR_SSH_2.0',
    granule_filter='*Basic*',
):
    """
    Download SWOT L2 LR Sea Surface Height granules from PO.DAAC.

    Requires:
      - earthaccess package: ``pip install earthaccess``
      - NASA Earthdata Login account (set up via earthaccess.login())

    Parameters
    ----------
    lon_range : (lon_min, lon_max)
    lat_range : (lat_min, lat_max)
    time_range : (start_str, end_str)
        ISO-format date strings, e.g. ('2024-02-01', '2024-02-02').
    dest_dir : str
        Local directory to store downloaded files.
    short_name : str
        PO.DAAC collection short name.
        Version C (2.0): 'SWOT_L2_LR_SSH_2.0'
        Version D:       'SWOT_L2_LR_SSH'
        Sub-collections can be filtered via granule_filter.
    granule_filter : str
        Wildcard filter for granule names.
        Options: '*Basic*', '*Expert*', '*Windwave*', '*Unsmoothed*'.

    Returns
    -------
    file_list : list of str
        Paths to downloaded NetCDF files.
    """
    try:
        import earthaccess
    except ImportError:
        raise ImportError("earthaccess package required: pip install earthaccess")

    os.makedirs(dest_dir, exist_ok=True)

    # Authenticate
    earthaccess.login()

    # Search for granules
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=time_range,
        bounding_box=(lon_range[0], lat_range[0], lon_range[1], lat_range[1]),
        granule_name=granule_filter,
    )
    print(f"[swot_ssh_data] Found {len(results)} SWOT granules")

    if len(results) == 0:
        warnings.warn("No SWOT granules found for the specified query.")
        return []

    # Download
    downloaded = earthaccess.download(results, dest_dir)
    file_list = [str(p) for p in downloaded]
    print(f"[swot_ssh_data] Downloaded {len(file_list)} files to {dest_dir}")
    return file_list


def load_swot_ssh_to_grid(
    file_list,
    lon_grid,
    lat_grid,
    obs_times,
    H_mean=4000.0,
    H_b=None,
    sig_ssh=0.05,
    quality_threshold=0,
    time_tolerance_s=3600,
    ssha_max=None,
):
    """
    Load downloaded SWOT L2 LR SSH files and bin observations onto the
    model grid.

    SWOT L2 data comes on an irregular swath grid (num_lines × num_pixels).
    We bin swath pixels into model grid cells and average.

    Parameters
    ----------
    file_list : list of str
        Paths to SWOT L2 LR SSH NetCDF files.
    lon_grid, lat_grid : 1-D arrays
        Model grid coordinates.
    obs_times : list of datetime
        Assimilation times.
    H_mean : float
        Mean depth for SSH → h conversion.
    sig_ssh : float
        Observation noise σ (metres) for SSH.
    quality_threshold : int
        Maximum acceptable ssh_karin_qual flag value (0 = best quality only).
    time_tolerance_s : float
        Max time difference to associate a SWOT measurement with an obs_time.

    Returns
    -------
    yobs_all, yobs_ind_all, yobs_ind0_all, sig_y_all
    """
    import xarray as xr

    ny, nx = len(lat_grid), len(lon_grid)
    ncells = ny * nx
    nassim = len(obs_times)

    # Grid spacing for binning
    dlon = np.abs(lon_grid[1] - lon_grid[0]) / 2.0
    dlat = np.abs(lat_grid[1] - lat_grid[0]) / 2.0

    # Epoch for SWOT time: seconds since 2000-01-01
    swot_epoch = datetime(2000, 1, 1)

    yobs_all = [np.array([], dtype=np.float64) for _ in range(nassim)]
    yobs_ind_all = [np.array([], dtype=int) for _ in range(nassim)]
    yobs_ind0_all = [np.array([], dtype=int) for _ in range(nassim)]
    sig_y_all = [np.array([], dtype=np.float64) for _ in range(nassim)]

    if not file_list:
        return yobs_all, yobs_ind_all, yobs_ind0_all, sig_y_all

    # Load all SWOT files
    ds = xr.open_mfdataset(file_list, combine='nested', concat_dim='num_lines',
                           decode_times=False)

    # Extract relevant variables
    lon_swot = ds['longitude'].values  # (num_lines, num_pixels)
    lat_swot = ds['latitude'].values
    ssh_swot = ds['ssha_karin'].values  # SSH anomaly (m)
    time_swot = ds['time'].values      # seconds since 2000-01-01 (per num_line)
    qual = ds['ssha_karin_qual'].values if 'ssha_karin_qual' in ds else None

    ds.close()

    # Convert SWOT times to datetime
    if time_swot.ndim == 1:
        # time is per-line → broadcast to (num_lines, num_pixels)
        time_swot_2d = np.repeat(time_swot[:, np.newaxis], lon_swot.shape[1], axis=1)
    else:
        time_swot_2d = time_swot

    # Flatten for processing
    lon_flat = lon_swot.ravel()
    lat_flat = lat_swot.ravel()
    ssh_flat = ssh_swot.ravel()
    time_flat = time_swot_2d.ravel()
    qual_flat = qual.ravel() if qual is not None else None

    # Filter valid data
    valid = np.isfinite(ssh_flat) & np.isfinite(lon_flat) & np.isfinite(lat_flat)
    if qual_flat is not None:
        valid &= (qual_flat <= quality_threshold)
    # Reject outliers: |ssha| > ssha_max (real SWOT data often has spurious values)
    if ssha_max is not None and ssha_max > 0:
        n_before = valid.sum()
        valid &= (np.abs(ssh_flat) <= ssha_max)
        n_rejected = n_before - valid.sum()
        if n_rejected > 0:
            print(f"[swot_ssh_data] Rejected {n_rejected} SWOT points with "
                  f"|ssha| > {ssha_max:.1f}m ({100*n_rejected/max(n_before,1):.1f}%)")

    lon_v = lon_flat[valid]
    lat_v = lat_flat[valid]
    ssh_v = ssh_flat[valid]
    time_v = time_flat[valid]

    # Convert SWOT longitudes to [-180,180] if needed
    lon_v = np.where(lon_v > 180, lon_v - 360, lon_v)

    # Match to assimilation times and grid cells
    for t_idx, t_obs in enumerate(obs_times):
        t_obs_sec = (t_obs - swot_epoch).total_seconds()
        time_mask = np.abs(time_v - t_obs_sec) <= time_tolerance_s
        if not time_mask.any():
            continue

        lon_t = lon_v[time_mask]
        lat_t = lat_v[time_mask]
        ssh_t = ssh_v[time_mask]

        # Bin into grid cells
        obs_dict = {}  # cell_flat → list of ssh values
        for k in range(len(lon_t)):
            ix = int(np.argmin(np.abs(lon_grid - lon_t[k])))
            iy = int(np.argmin(np.abs(lat_grid - lat_t[k])))
            # Check within grid bounds
            if abs(lon_grid[ix] - lon_t[k]) > dlon or abs(lat_grid[iy] - lat_t[k]) > dlat:
                continue
            cell_flat = iy * nx + ix
            if cell_flat not in obs_dict:
                obs_dict[cell_flat] = []
            obs_dict[cell_flat].append(ssh_t[k])

        if not obs_dict:
            continue

        # Average SSH per cell, convert to h = H_b(cell) + SSH
        cells = sorted(obs_dict.keys())
        if H_b is not None:
            ny_g = len(lat_grid)
            nx_g = len(lon_grid)
            H_b_flat = np.asarray(H_b).reshape(-1)
            obs_vals = np.array([H_b_flat[c] + np.mean(obs_dict[c]) for c in cells])
        else:
            obs_vals = np.array([H_mean + np.mean(obs_dict[c]) for c in cells])
        obs_inds = np.array(cells, dtype=int)
        obs_sigs = np.full(len(cells), sig_ssh)

        yobs_all[t_idx] = obs_vals
        yobs_ind_all[t_idx] = obs_inds
        yobs_ind0_all[t_idx] = obs_inds.copy()
        sig_y_all[t_idx] = obs_sigs

    total_obs = sum(len(y) for y in yobs_all)
    n_with_obs = sum(1 for y in yobs_all if len(y) > 0)
    print(f"[swot_ssh_data] Loaded real SWOT SSH: "
          f"{total_obs} obs across {n_with_obs}/{nassim} cycles")

    return yobs_all, yobs_ind_all, yobs_ind0_all, sig_y_all

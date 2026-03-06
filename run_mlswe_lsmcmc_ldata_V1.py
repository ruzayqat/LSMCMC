#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_mlswe_lsmcmc_ldata_V1.py
============================
Runner script for the 3-layer MLSWE model with LSMCMC data assimilation (V1).

Usage
-----
    python run_mlswe_lsmcmc_ldata_V1.py [config.yml]
"""
import os
import sys
import time
import yaml
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))
_SWE_DIR = os.path.join(os.path.dirname(__file__), '..', 'SWE_LSMCMC')
if os.path.isdir(_SWE_DIR):
    sys.path.insert(0, os.path.abspath(_SWE_DIR))

from mlswe.model import MLSWE, coriolis_array, lonlat_to_dxdy
from mlswe.boundary_handler import MLBoundaryHandler
from mlswe.lsmcmc_V1 import Loc_SMCMC_MLSWE_Filter

from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


def _unix_to_datetime(unix_s):
    return datetime(1970, 1, 1) + timedelta(seconds=float(unix_s))


def _fill_nan_nearest(arr_2d):
    """Fill NaN in a 2-D array with nearest finite value."""
    mask = np.isnan(arr_2d)
    if not mask.any():
        return arr_2d
    if mask.all():
        arr_2d[:] = 0.0
        return arr_2d
    ind = distance_transform_edt(mask, return_distances=False,
                                  return_indices=True)
    arr_2d[mask] = arr_2d[tuple(ind[:, mask])]
    return arr_2d


def init_from_bc_handler(bc_handler, H_b, tstart,
                         H_rest=(100.0, 400.0, 3500.0),
                         T_rest=(298.15, 283.15, 275.15),
                         beta_vel=(1.0, 0.3, 0.05),
                         dx=55.6e3, dy=55.6e3,
                         geostrophic_blend=0.5):
    """
    Initialise all 3 layers from HYCOM via bc_handler.get_full_field().

    Uses the **same** approach as the LETKF runner to ensure identical ICs:
    bc_handler.get_full_field() → geostrophic blend → layer distribution
    → apply BC.

    Parameters
    ----------
    bc_handler : MLBoundaryHandler
        Boundary handler with full HYCOM fields still loaded.
    H_b : ndarray (ny, nx)
        Bathymetry (positive depth in metres).
    tstart : float
        Start time (Unix epoch seconds).
    beta_vel : tuple
        Velocity depth-decay factors per layer (default matches LETKF).
    geostrophic_blend : float
        Fraction of geostrophic velocity in the blend (0=pure HYCOM, 1=pure geo).

    Returns
    -------
    h0_list, u0_list, v0_list, T0_list : lists of 3 arrays (ny, nx)
    """
    lat_grid = bc_handler.model_lat
    lon_grid = bc_handler.model_lon
    ny = len(lat_grid)
    nx = len(lon_grid)
    H_rest = np.asarray(H_rest, dtype=np.float64)
    T_rest = np.asarray(T_rest, dtype=np.float64)
    beta_vel = np.asarray(beta_vel, dtype=np.float64)
    H_rest_total = float(H_rest.sum())

    # --- Get HYCOM fields via bc_handler (same as LETKF) ---
    ssh_full = bc_handler.get_full_field('ssh', tstart)
    uo_full  = bc_handler.get_full_field('uo',  tstart)
    vo_full  = bc_handler.get_full_field('vo',  tstart)
    try:
        sst_full = bc_handler.get_full_field('sst', tstart)
    except (ValueError, AttributeError):
        sst_full = None

    print(f"[init_from_bc_handler] HYCOM fields on ({ny}x{nx}) grid:")
    print(f"  SSH range: [{ssh_full.min():.3f}, {ssh_full.max():.3f}] m")
    print(f"  uo  range: [{uo_full.min():.3f}, {uo_full.max():.3f}] m/s")
    print(f"  vo  range: [{vo_full.min():.3f}, {vo_full.max():.3f}] m/s")
    if sst_full is not None:
        print(f"  SST range: [{sst_full.min()-273.15:.1f}, "
              f"{sst_full.max()-273.15:.1f}] deg C")

    # --- Geostrophic velocity blending (same as LETKF) ---
    if geostrophic_blend > 0:
        from mlswe.boundary_handler import geostrophic_velocities
        u_geo, v_geo = geostrophic_velocities(
            ssh_full, lat_grid, lon_grid, dx, dy)
        gb = float(geostrophic_blend)
        uo_full = (1.0 - gb) * uo_full + gb * u_geo
        vo_full = (1.0 - gb) * vo_full + gb * v_geo
        print(f"  Geostrophic blend ({gb:.2f}): "
              f"u=[{uo_full.min():.4f}, {uo_full.max():.4f}], "
              f"v=[{vo_full.min():.4f}, {vo_full.max():.4f}] m/s")

    # --- Distribute across 3 layers (same as LETKF) ---
    h_total = np.maximum(H_b + ssh_full, 10.0)

    h0_list, u0_list, v0_list, T0_list = [], [], [], []
    for k in range(3):
        h_k = (H_rest[k] / H_rest_total) * h_total
        h0_list.append(h_k)

        u0_list.append(beta_vel[k] * uo_full.copy())
        v0_list.append(beta_vel[k] * vo_full.copy())

        if k == 0 and sst_full is not None:
            T0_list.append(sst_full.copy())
        else:
            T0_list.append(np.full((ny, nx), T_rest[k], dtype=np.float64))

    # --- Apply BC to enforce boundary consistency (same as LETKF) ---
    if bc_handler is not None:
        state = {}
        for k in range(3):
            state[f'h{k}'] = h0_list[k]
            state[f'u{k}'] = u0_list[k]
            state[f'v{k}'] = v0_list[k]
            state[f'T{k}'] = T0_list[k]
        state = bc_handler(state, tstart)
        for k in range(3):
            h0_list[k] = state[f'h{k}']
            u0_list[k] = state[f'u{k}']
            v0_list[k] = state[f'v{k}']
            T0_list[k] = state[f'T{k}']

    h_total_check = h0_list[0] + h0_list[1] + h0_list[2]
    ssh_check = h_total_check - H_b
    print(f"  Layer h: [{h0_list[0].mean():.1f}, {h0_list[1].mean():.1f}, "
          f"{h0_list[2].mean():.1f}] m (mean)")
    print(f"  SSH check: [{ssh_check.min():.3f}, {ssh_check.max():.3f}] m "
          f"(should be ~HYCOM SSH)")

    return h0_list, u0_list, v0_list, T0_list


def load_bathymetry(params, ny, nx, lon_grid, lat_grid):
    """Load bathymetry, preferring files whose shape matches (ny, nx)."""
    from scipy.interpolate import RegularGridInterpolator
    data_dir = params.get('data_dir', './data')
    H_min = params.get('H_min', 100.0)

    # Collect all candidate files from data/ and SWE_LSMCMC/data
    candidates = []
    for d in [data_dir, os.path.join(_SWE_DIR, 'data')]:
        if d and os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith('etopo_bathy_') and f.endswith('.npy'):
                    candidates.append(os.path.join(d, f))

    if not candidates:
        print("[runner] No bathymetry found; using flat bottom")
        return None

    # First pass: prefer exact shape match
    for c in candidates:
        arr = np.load(c).astype(np.float64)
        if arr.shape == (ny, nx):
            H_b = np.maximum(np.abs(arr), H_min)
            # Smooth bathymetry (Beckmann-Haidvogel) if requested
            bathy_r_max = params.get('bathy_r_max', None)
            if bathy_r_max is not None:
                from mlswe.model import smooth_bathymetry
                H_b = smooth_bathymetry(H_b, r_max=float(bathy_r_max))
                H_b = np.maximum(H_b, H_min)  # re-apply floor
            print(f"[runner] Loaded bathymetry (exact match): {c}, "
                  f"shape={H_b.shape}, range=[{H_b.min():.0f}, {H_b.max():.0f}] m")
            return H_b

    # Second pass: interpolate the first available file
    arr = np.load(candidates[0]).astype(np.float64)
    if arr.min() < 0:
        arr = np.abs(arr)
    bathy_lat = np.linspace(lat_grid[0], lat_grid[-1], arr.shape[0])
    bathy_lon = np.linspace(lon_grid[0], lon_grid[-1], arr.shape[1])
    interp = RegularGridInterpolator(
        (bathy_lat, bathy_lon), arr,
        method='linear', bounds_error=False, fill_value=None)
    pts = np.array(np.meshgrid(lat_grid, lon_grid, indexing='ij'))
    pts = pts.reshape(2, -1).T
    H_b = interp(pts).reshape(ny, nx)
    H_b = np.maximum(H_b, H_min)
    # Smooth bathymetry (Beckmann-Haidvogel) if requested
    bathy_r_max = params.get('bathy_r_max', None)
    if bathy_r_max is not None:
        from mlswe.model import smooth_bathymetry
        H_b = smooth_bathymetry(H_b, r_max=float(bathy_r_max))
        H_b = np.maximum(H_b, H_min)  # re-apply floor
    print(f"[runner] Loaded bathymetry (interpolated from {candidates[0]}): "
          f"shape={H_b.shape}, range=[{H_b.min():.0f}, {H_b.max():.0f}] m")
    return H_b


# =====================================================================
#  M-run parallel averaging  (mirrors linear_forward_run_lsmcmc_v1.py)
# =====================================================================
# Module-level globals populated before forking; inherited via COW.
_g_params = None
_g_H_b = None
_g_bc_handler = None
_g_obs_file = None
_g_tstart = None
_g_ncores_per_worker = 1   # cores each worker may use for forecast


def _v1_worker(args):
    """Run one independent LSMCMC V1 simulation (for M-run averaging).

    Each worker gets `_g_ncores_per_worker` cores for its forecast pool.
    When total_cores >= M * nforecast, each worker fully parallelises
    its forecast step.  When total_cores < M * nforecast, each worker
    runs forecasts serially (ncores_per_worker = 1).
    """
    isim, seed = args
    params = dict(_g_params)       # shallow copy; numpy refs shared via COW
    params['ncores'] = _g_ncores_per_worker
    params['verbose'] = False      # suppress per-cycle output
    np.random.seed(seed)

    filt = Loc_SMCMC_MLSWE_Filter(isim, params)
    t0 = time.time()
    filt.run(_g_H_b, _g_bc_handler, _g_obs_file, _g_tstart)
    elapsed = time.time() - t0

    print(f"  [done] run {isim+1}  seed={seed}  elapsed={elapsed:.1f}s  "
          f"vel={np.nanmean(filt.rmse_vel):.6f}  "
          f"sst={np.nanmean(filt.rmse_sst):.4f}  "
          f"ssh={np.nanmean(filt.rmse_ssh):.4f}", flush=True)

    return (isim, filt.lsmcmc_mean.copy(),
            filt.rmse_vel.copy(), filt.rmse_sst.copy(), filt.rmse_ssh.copy(),
            elapsed)


def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'example_input_mlswe_ldata_V1.yml'
    with open(config_file) as f:
        params = yaml.safe_load(f)

    print("=" * 60)
    print("  MLSWE LSMCMC Runner -- 3-Layer Primitive Equations")
    print("=" * 60)

    nx = params['dgx']
    ny = params['dgy']
    lon = np.linspace(params['lon_min'], params['lon_max'], nx)
    lat = np.linspace(params['lat_min'], params['lat_max'], ny)

    # Bathymetry
    H_b = load_bathymetry(params, ny, nx, lon, lat)

    # Boundary conditions
    bc_file = params.get('bc_file', './data/hycom_bc.nc')
    if not os.path.exists(bc_file):
        # Try SWE_LSMCMC
        bc_file = os.path.join(_SWE_DIR, 'data', 'hycom_bc.nc')
    print(f"BC file: {bc_file}")

    bc_handler = MLBoundaryHandler(
        nc_path=bc_file,
        model_lon=lon,
        model_lat=lat,
        H_b=H_b,
        H_mean=params.get('H_mean', 4000.0),
        H_rest=params.get('H_rest', [100.0, 400.0, 3500.0]),
        T_rest=params.get('T_rest', [298.15, 283.15, 275.15]),
        alpha_h=params.get('alpha_h', [0.6, 0.3, 0.1]),
        beta_vel=params.get('beta_vel', [1.0, 1.0, 1.0]),
        n_ghost=params.get('bc_n_ghost', 2),
        sponge_width=params.get('sponge_width', 8),
        sponge_timescale=params.get('sponge_timescale', 3600.0),
    )

    # Observation file
    obs_file = params.get('obs_file',
                          '../SWE_LSMCMC/output_comparison_lsmcmc/swe_drifter_obs.nc')
    if not os.path.exists(obs_file):
        obs_file = os.path.join(_SWE_DIR, 'output_comparison_lsmcmc',
                                 'swe_drifter_obs.nc')
    print(f"Obs file: {obs_file}")

    # Get start time from obs
    with Dataset(obs_file, 'r') as nc:
        obs_times = np.asarray(nc.variables['obs_times'][:])
    tstart = obs_times[0]
    print(f"Start time: {_unix_to_datetime(tstart)} UTC")

    # Load SST reference for nudging (HYCOM SST on model grid)
    sst_nudging_rate = params.get('sst_nudging_rate', 0.0)
    if sst_nudging_rate > 0:
        data_dir = params.get('data_dir', './data')
        import glob
        # Match SST ref by grid size to pick correct domain
        grid_tag = f'{ny}x{nx}'
        sst_files = sorted(glob.glob(os.path.join(data_dir,
                                                   f'hycom_sst_ref_*_{grid_tag}_3d.npy')))
        sst_time_files = sorted(glob.glob(os.path.join(data_dir,
                                                        f'hycom_sst_ref_*_{grid_tag}_times.npy')))
        if not sst_files:
            # Fallback: try any SST ref file
            sst_files = sorted(glob.glob(os.path.join(data_dir,
                                                       'hycom_sst_ref_*_3d.npy')))
            sst_time_files = sorted(glob.glob(os.path.join(data_dir,
                                                            'hycom_sst_ref_*_times.npy')))
        if sst_files:
            # Pick the file whose end date covers the run period
            sst_ref = np.load(sst_files[-1])
            sst_ref_times_raw = np.load(sst_time_files[-1])
            # Times in SST ref are seconds relative to obs_time_start
            # Convert to same epoch as tstart (Unix seconds)
            t0_sst_str = params.get('obs_time_start', '2024-08-01T00:00:00')
            t0_sst_dt = datetime.strptime(t0_sst_str[:19], '%Y-%m-%dT%H:%M:%S')
            epoch_obs = (t0_sst_dt - datetime(1970, 1, 1)).total_seconds()
            sst_ref_times = sst_ref_times_raw + epoch_obs
            params['sst_nudging_ref'] = sst_ref
            params['sst_nudging_ref_times'] = sst_ref_times
            print(f"[runner] SST nudging: lam={sst_nudging_rate:.6f} s^-1 "
                  f"(tau={1.0/sst_nudging_rate:.0f}s), "
                  f"ref shape={sst_ref.shape}, "
                  f"T range=[{sst_ref.min():.1f}, {sst_ref.max():.1f}] K")
        else:
            print("[runner] WARNING: SST nudging enabled but no reference found!")

    # Load SSH reference for relaxation (from HYCOM BC)
    ssh_relax_rate = float(params.get('ssh_relax_rate', 0.0))
    if ssh_relax_rate > 0:
        try:
            from scipy.interpolate import RegularGridInterpolator
            with Dataset(bc_file, 'r') as nc:
                bc_lat = np.asarray(nc.variables['lat'][:], dtype=np.float64)
                bc_lon = np.asarray(nc.variables['lon'][:], dtype=np.float64)
                bc_times = np.asarray(nc.variables['time'][:], dtype=np.float64)
                bc_ssh = np.asarray(nc.variables['ssh'][:], dtype=np.float64)
            bc_ssh[np.isnan(bc_ssh)] = 0.0
            mg_lat, mg_lon = np.meshgrid(lat, lon, indexing='ij')
            ssh_ref_3d = np.zeros((len(bc_times), ny, nx), dtype=np.float64)
            for ti in range(len(bc_times)):
                interp = RegularGridInterpolator(
                    (bc_lat, bc_lon), bc_ssh[ti],
                    method='linear', bounds_error=False, fill_value=0.0)
                ssh_ref_3d[ti] = interp((mg_lat, mg_lon))
            params['ssh_relax_ref'] = ssh_ref_3d
            params['ssh_relax_ref_times'] = bc_times
            print(f"[runner] SSH relaxation: rate={ssh_relax_rate:.2e} s^-1 "
                  f"(tau={1.0/ssh_relax_rate:.0f}s), "
                  f"ref shape={ssh_ref_3d.shape}, "
                  f"SSH range=[{ssh_ref_3d.min():.3f}, {ssh_ref_3d.max():.3f}] m")
        except Exception as e:
            print(f"[runner] WARNING: SSH relax enabled but ref load failed: {e}")

    # Load T_air reference for surface heat flux (use SST nudging ref as proxy)
    sst_flux_type = params.get('sst_flux_type', None)
    if sst_flux_type is not None:
        # Use the same SST reference as T_air (HYCOM SST = best local air-sea T proxy)
        if 'sst_nudging_ref' in params and params['sst_nudging_ref'] is not None:
            params['sst_T_air'] = params['sst_nudging_ref']
            params['sst_T_air_times'] = params['sst_nudging_ref_times']
            alpha_val = float(params.get('sst_alpha', 15.0))
            hmix_val = float(params.get('sst_h_mix', 50.0))
            lam_flux = alpha_val / (1025.0 * 3990.0 * hmix_val)
            print(f"[runner] Surface heat flux: type={sst_flux_type}, "
                  f"alpha={alpha_val} W/(m2.K), h_mix={hmix_val}m, "
                  f"lambda_flux={lam_flux:.2e} s^-1")
        else:
            print(f"[runner] WARNING: heat flux enabled but no T_air ref; "
                  f"using SST nudging ref as fallback")


    # --- Initialise from HYCOM (same approach as LETKF) ---
    # Compute grid spacing in metres (at domain centre latitude)
    lat_centre = 0.5 * (params['lat_min'] + params['lat_max'])
    dx_m = np.deg2rad(abs(lon[1] - lon[0])) * 6.371e6 * np.cos(np.deg2rad(lat_centre))
    dy_m = np.deg2rad(abs(lat[1] - lat[0])) * 6.371e6

    h0_ic, u0_ic, v0_ic, T0_ic = init_from_bc_handler(
        bc_handler, H_b, tstart,
        H_rest=params.get('H_rest', [100.0, 400.0, 3500.0]),
        T_rest=params.get('T_rest', [298.15, 283.15, 275.15]),
        beta_vel=params.get('beta_vel', [1.0, 0.3, 0.05]),
        dx=dx_m,
        dy=dy_m,
        geostrophic_blend=params.get('geostrophic_blend', 0.5),
    )
    params['ic_h0'] = h0_ic
    params['ic_u0'] = u0_ic
    params['ic_v0'] = v0_ic
    params['ic_T0'] = T0_ic

    # ---- Release full HYCOM arrays (no longer needed after init) ----
    # Saves ~3.8 GB — the full (101, 1226, 988) arrays for ssh/uo/vo/sst
    if bc_handler is not None:
        bc_handler.release_full_fields()

    # --- SWOT SSH observations (optional — merge with drifter obs) ---
    use_swot_ssh = params.get('use_swot_ssh', False)
    if use_swot_ssh:
        from swot_ssh_data import (
            generate_synthetic_ssh_obs,
            build_ssh_reference_from_hycom,
            merge_obs_arrays,
            download_swot_ssh as _download_swot_ssh,
            load_swot_ssh_to_grid,
        )
        from drifter_data import build_obs_netcdf

        print("[runner] Adding SWOT SSH observations to existing drifter obs ...")
        sig_ssh = float(params.get('sig_y_ssh', params.get('sig_ssh', 0.05)))
        ssh_obs_fraction = float(params.get('ssh_obs_fraction', 0.15))
        H_mean = float(params.get('H_mean', 4000.0))
        nassim = params['nassim']
        ncells = nx * ny

        # Reconstruct assimilation datetimes
        t0_str = params.get('obs_time_start', '2019-07-01')
        t0 = datetime.fromisoformat(t0_str)
        t_freq = int(params.get('t_freq', params.get('assim_timesteps', 720)))
        dt = float(params['dt'])
        obs_dt_delta = timedelta(seconds=t_freq * dt)
        obs_datetimes_swot = [t0 + i * obs_dt_delta for i in range(1, nassim + 1)]

        swot_source = params.get('swot_source', 'synthetic')
        data_dir = params.get('data_dir', './data')
        if swot_source == 'real':
            t1_str = params.get('obs_time_end', '2024-08-12')
            swot_dir = params.get('swot_dir',
                                  os.path.join(data_dir, 'swot_2024aug'))

            # ---- Fast path: load from pre-binned or combined file ----
            combined_file = None
            binned_file = None
            for tag in [f'{ny}x{nx}', f'{nx}x{ny}']:
                c = os.path.join(swot_dir, f'swot_ssh_combined_{tag}.nc')
                b = os.path.join(swot_dir, f'swot_ssh_binned_{tag}.nc')
                if os.path.exists(c):
                    combined_file = c
                if os.path.exists(b):
                    binned_file = b
            fast_file = combined_file if combined_file else binned_file

            if fast_file is not None:
                print(f"[runner] Loading pre-binned SWOT from {fast_file}")
                with Dataset(fast_file, 'r') as nc_b:
                    ssha_bin = nc_b.variables['ssha'][:]
                    cell_bin = nc_b.variables['cell_index'][:]
                    nobs_bin = nc_b.variables['n_obs'][:]
                H_b_flat = np.asarray(H_b, dtype=np.float64).ravel()
                ssh_yobs = [np.array([], dtype=np.float64)] * nassim
                ssh_ind  = [np.array([], dtype=int)] * nassim
                ssh_ind0 = [np.array([], dtype=int)] * nassim
                ssh_sigy = [np.array([], dtype=np.float64)] * nassim
                n_loaded = 0
                for c in range(min(nassim, ssha_bin.shape[0])):
                    n = int(nobs_bin[c])
                    if n > 0:
                        cells = cell_bin[c, :n].astype(int)
                        ssha  = ssha_bin[c, :n].astype(np.float64)
                        h_obs = H_b_flat[cells] + ssha
                        ssh_yobs[c] = h_obs
                        ssh_ind[c]  = cells.copy()
                        ssh_ind0[c] = cells.copy()
                        ssh_sigy[c] = np.full(n, sig_ssh, dtype=np.float64)
                        n_loaded += 1
                print(f"[runner] Loaded {n_loaded}/{nassim} cycles "
                      f"from pre-binned file")
            else:
                # ---- Slow path: load from raw SWOT files ----
                import glob as _glob2
                existing_nc = sorted(_glob2.glob(os.path.join(
                    swot_dir, 'SWOT_*.nc')))
                if existing_nc:
                    print(f"[runner] Using {len(existing_nc)} pre-downloaded "
                          f"SWOT files from {swot_dir}")
                    swot_files = existing_nc
                else:
                    swot_files = _download_swot_ssh(
                        lon_range=(params['lon_min'], params['lon_max']),
                        lat_range=(params['lat_min'], params['lat_max']),
                        time_range=(t0_str[:10], t1_str[:10]),
                        dest_dir=swot_dir,
                    )
                _ssha_max_raw = params.get('ssha_max', 2.0)
                _ssha_max = None if _ssha_max_raw is None else float(_ssha_max_raw)
                ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy = load_swot_ssh_to_grid(
                    swot_files, lon, lat, obs_datetimes_swot,
                    H_mean=H_mean, H_b=H_b, sig_ssh=sig_ssh,
                    ssha_max=_ssha_max,
                )
        else:
            # Synthetic SWOT from HYCOM reference
            ssh_ref = build_ssh_reference_from_hycom(
                bc_file, lon, lat, obs_datetimes_swot,
                cache_dir=data_dir,
            )
            ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy = generate_synthetic_ssh_obs(
                lon, lat, obs_datetimes_swot,
                ssh_reference=ssh_ref,
                H_mean=H_mean,
                H_b=H_b,
                sig_ssh=sig_ssh,
                obs_fraction=ssh_obs_fraction,
                seed=params.get('ssh_obs_seed', 42),
            )

        # Load existing drifter obs, merge, write new file
        with Dataset(obs_file, 'r') as nc:
            drift_yobs_arr = np.asarray(nc.variables['yobs_all'][:])
            drift_ind_arr = np.asarray(nc.variables['yobs_ind_all'][:])
            drift_ind0_arr = np.asarray(nc.variables['yobs_ind_level0_all'][:])
            drift_sigy_arr = np.asarray(nc.variables['sig_y_all'][:]) if 'sig_y_all' in nc.variables else None
            drift_sig_y_scalar = float(nc.sig_y)

        drift_yobs, drift_ind, drift_ind0, drift_sigy = [], [], [], []
        for i in range(nassim):
            valid_y = ~np.isnan(drift_yobs_arr[i])
            drift_yobs.append(drift_yobs_arr[i, valid_y])
            valid_i = drift_ind_arr[i] >= 0
            drift_ind.append(drift_ind_arr[i, valid_i])
            valid_i0 = drift_ind0_arr[i] >= 0
            drift_ind0.append(drift_ind0_arr[i, valid_i0])
            if drift_sigy_arr is not None:
                valid_s = ~np.isnan(drift_sigy_arr[i])
                drift_sigy.append(drift_sigy_arr[i, valid_s])
            else:
                drift_sigy.append(np.full(int(valid_y.sum()), drift_sig_y_scalar))

        merged_yobs, merged_ind, merged_ind0, merged_sigy = merge_obs_arrays(
            drift_yobs, drift_ind, drift_ind0, drift_sigy,
            ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy,
        )

        outdir = params.get('lsmcmc_dir', './output_lsmcmc')
        os.makedirs(outdir, exist_ok=True)
        obs_file_merged = os.path.join(outdir, 'mlswe_merged_obs.nc')
        build_obs_netcdf(obs_file_merged, merged_yobs, merged_ind, merged_ind0,
                         obs_datetimes_swot, sig_y=drift_sig_y_scalar,
                         sig_y_all=merged_sigy)
        obs_file = obs_file_merged

        n_ssh = [len(y) for y in ssh_yobs]
        n_total = [len(y) for y in merged_yobs]
        print(f"[runner] SWOT SSH: sig={sig_ssh:.3f}m, frac={ssh_obs_fraction}, "
              f"source={swot_source}")
        print(f"[runner] SSH/cycle: mean={np.mean(n_ssh):.0f}, "
              f"total/cycle: mean={np.mean(n_total):.0f}")
        print(f"[runner] Merged obs file: {obs_file_merged}")

    # ---- Run filter(s) ----
    M = int(params.get('M', 1))
    ncores = int(params.get('ncores', 1))
    workers = int(params.get('workers', ncores))
    outdir = params.get('lsmcmc_dir', './output_lsmcmc_ldata_V1')
    os.makedirs(outdir, exist_ok=True)
    nassim = int(params['nassim'])
    dimx = 12 * ny * nx

    if M <= 1:
        # ============================================================
        #  Single run  (backward-compatible, uses ncores for forecast)
        # ============================================================
        filt = Loc_SMCMC_MLSWE_Filter(0, params)

        t_wall = time.time()
        filt.run(H_b, bc_handler, obs_file, tstart)
        elapsed = time.time() - t_wall

        filt.save_results(outdir, obs_times=obs_times, H_b=H_b)

        print(f"\n{'='*60}")
        print(f"  MLSWE LSMCMC V1 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Mean vel RMSE:  {np.nanmean(filt.rmse_vel):.6f} m/s")
        print(f"  Mean SST RMSE:  {np.nanmean(filt.rmse_sst):.4f} K")
        print(f"  Mean SSH RMSE:  {np.nanmean(filt.rmse_ssh):.4f} m")
        print(f"  Output: {outdir}/")
        print(f"{'='*60}")

    else:
        # ============================================================
        #  M independent runs averaged  (linear V1 pattern)
        #  Distribute total cores across M workers:
        #    ncores_per_worker = ncores // min(workers, M)
        #  Each worker uses its share for forecast parallelism.
        #  On a supercomputer with many nodes, set ncores = total
        #  cores available so each worker gets its fair share.
        # ============================================================
        global _g_params, _g_H_b, _g_bc_handler, _g_obs_file, _g_tstart
        global _g_ncores_per_worker
        _g_params = params
        _g_H_b = H_b
        _g_bc_handler = bc_handler
        _g_obs_file = obs_file
        _g_tstart = tstart

        n_workers = min(workers, M)
        ncores_per_worker = max(1, ncores // n_workers)
        _g_ncores_per_worker = ncores_per_worker
        Nf = int(params.get('nforecast', 25))
        Na = int(params.get('mcmc_N', 500))
        Gamma = int(params.get('num_subdomains', 50))

        print(f"\n[V1] Launching M={M} independent runs on {n_workers} workers")
        print(f"[V1] ncores_per_worker={ncores_per_worker}  "
              f"(total={ncores}, forecast parallel={'yes' if ncores_per_worker > 1 else 'no'})")
        print(f"[V1] Nf={Nf}, Na={Na}, Gamma={Gamma}")

        # Incremental (Welford) averaging to avoid O(M) memory
        avg_mean = np.zeros((nassim + 1, dimx), dtype=np.float64)
        avg_rmse_vel = np.zeros(nassim, dtype=np.float64)
        avg_rmse_sst = np.zeros(nassim, dtype=np.float64)
        avg_rmse_ssh = np.zeros(nassim, dtype=np.float64)
        all_elapsed = []
        count = 0

        worker_args = [(i, 1000 + i) for i in range(M)]

        t_wall = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_v1_worker, wa) for wa in worker_args]
            for fut in futures:
                isim, mean_arr, rv, rs, rh, elapsed = fut.result()
                count += 1
                # Welford incremental average
                avg_mean += (mean_arr - avg_mean) / count
                avg_rmse_vel += (rv - avg_rmse_vel) / count
                avg_rmse_sst += (rs - avg_rmse_sst) / count
                avg_rmse_ssh += (rh - avg_rmse_ssh) / count
                all_elapsed.append(elapsed)

        total_elapsed = time.time() - t_wall

        # Save averaged results via filter's save mechanism
        filt = Loc_SMCMC_MLSWE_Filter.__new__(Loc_SMCMC_MLSWE_Filter)
        filt.nassim = nassim
        filt.nlayers = 3
        filt.fields_per_layer = 4
        filt.ny = ny
        filt.nx = nx
        filt.ncells = ny * nx
        filt.lsmcmc_mean = avg_mean
        filt.rmse_vel = avg_rmse_vel
        filt.rmse_sst = avg_rmse_sst
        filt.rmse_ssh = avg_rmse_ssh
        filt.save_results(outdir, obs_times=obs_times, H_b=H_b)

        print(f"\n{'='*60}")
        print(f"  MLSWE LSMCMC V1 complete: M={M} runs averaged")
        print(f"  Wall time:      {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"  Mean run time:  {np.mean(all_elapsed):.1f}s")
        print(f"  Mean vel RMSE:  {np.nanmean(avg_rmse_vel):.6f} m/s")
        print(f"  Mean SST RMSE:  {np.nanmean(avg_rmse_sst):.4f} K")
        print(f"  Mean SSH RMSE:  {np.nanmean(avg_rmse_ssh):.4f} m")
        print(f"  Output: {outdir}/")
        print(f"{'='*60}")

    # ---- Auto-generate diagnostic plots ----
    try:
        from plot_mlswe_results import main as plot_results
        print("\n[runner] Generating diagnostic plots ...")
        plot_results(
            outdir=outdir,
            save_prefix='mlswe_results',
            config_file=config_file,
            method_label='LSMCMC',
        )
    except Exception as e:
        print(f"[runner] WARNING: Plotting failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()

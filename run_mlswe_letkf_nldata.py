#!/usr/bin/env python
"""
run_mlswe_letkf_nldata.py
=======================
Multiprocessing-parallelised LETKF for the NONLINEAR 3-layer multi-layer
shallow-water model (MLSWE_NL) with real drifter observations from the
NOAA Global Drifter Program.

Usage
-----
    python3 run_mlswe_letkf_nldata.py [example_input_mlswe_letkf_nldata.yml]

The script expects that the LSMCMC runner has already been executed (or
at least that the observation NetCDF ``swe_drifter_obs.nc`` exists).
It reads observations from there and writes its own output NetCDF in the
LETKF output directory.

Output NetCDF layout:
    - ``lsmcmc_mean`` -> (t, layer, field, y, x)   analysis ensemble mean
    - ``obs_times``   -> (t,)                       observation times
"""

import os
import sys
import time
import copy
import ctypes
import warnings
import multiprocessing as mp

import numpy as np
import yaml
from netCDF4 import Dataset
from datetime import datetime, timedelta

# ---- MLSWE model ----
sys.path.insert(0, os.path.dirname(__file__))
_SWE_DIR = os.path.join(os.path.dirname(__file__), '..', 'SWE_LSMCMC')
if os.path.isdir(_SWE_DIR):
    sys.path.insert(0, os.path.abspath(_SWE_DIR))

from mlswe.model import MLSWE, coriolis_array, lonlat_to_dxdy
from mlswe.boundary_handler import MLBoundaryHandler
from mlswe.letkf import (
    letkf_update_mlswe_nl_mpi,
    precompute_covlocal,
    precompute_covlocal_local,
)

N_LAYERS = 3
FIELDS_PER_LAYER = 4


# ===================================================================
#  Helpers
# ===================================================================

def _unix_to_datetime(epoch):
    return datetime(1970, 1, 1) + timedelta(seconds=float(epoch))


def _load_obs_netcdf(obs_file):
    """Read observation NetCDF produced by the LSMCMC runner."""
    nc = Dataset(obs_file, 'r')
    raw_obs = np.asarray(nc.variables['yobs_all'][:])
    raw_ind = np.asarray(nc.variables['yobs_ind_all'][:])
    raw_ind0 = np.asarray(nc.variables['yobs_ind_level0_all'][:])
    raw_sig = np.asarray(nc.variables['sig_y_all'][:])
    obs_times = np.asarray(nc.variables['obs_times'][:])
    nc.close()

    nassim = raw_obs.shape[0]
    yobs_list, ind_list, ind0_list, sig_list = [], [], [], []
    for n in range(nassim):
        valid = (raw_ind[n] >= 0) & np.isfinite(raw_obs[n])
        yobs_list.append(raw_obs[n, valid].astype(np.float64))
        # Obs indices map directly to layer 0 (first 4*ncells in state)
        ind_list.append(raw_ind[n, valid].astype(np.int64))
        ind0_list.append(raw_ind0[n, valid].astype(np.int64))
        sig_list.append(raw_sig[n, valid].astype(np.float64))
    return yobs_list, ind_list, ind0_list, sig_list, obs_times


def load_bathymetry(params, ny, nx, lon_grid, lat_grid):
    """Load bathymetry, preferring files whose shape matches (ny, nx)."""
    from scipy.interpolate import RegularGridInterpolator
    data_dir = params.get('data_dir', './data')
    H_min = params.get('H_min', 100.0)

    # Collect all candidate files
    candidates = []
    for d in [data_dir,
              os.path.join(_SWE_DIR, 'data') if os.path.isdir(_SWE_DIR) else '']:
        if d and os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith('etopo_bathy_') and f.endswith('.npy'):
                    candidates.append(os.path.join(d, f))

    if not candidates:
        return None

    # First pass: prefer exact shape match
    for c in candidates:
        arr = np.load(c).astype(np.float64)
        if arr.shape == (ny, nx):
            return np.maximum(np.abs(arr), H_min)

    # Second pass: interpolate from first available
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
    return np.maximum(H_b, H_min)


def _make_init_state(ny, nx, H_b, H_rest, T_rest, bc_handler, tstart,
                     beta_vel=None, lat_grid=None, lon_grid=None,
                     dx=55.6e3, dy=55.6e3, geostrophic_blend=0.5):
    """
    Create initial layer states from HYCOM boundary conditions.

    Now properly initializes u,v,T from HYCOM throughout the entire domain,
    not just at boundaries. Blends HYCOM velocities with geostrophic
    velocities computed from SSH to ensure pressure-velocity consistency.
    """
    H_rest_total = float(H_rest.sum())
    h0, u0, v0, T0 = [], [], [], []

    # Get HYCOM fields for entire domain if bc_handler available
    if bc_handler is not None:
        try:
            ssh_full = bc_handler.get_full_field('ssh', tstart)
            uo_full = bc_handler.get_full_field('uo', tstart)
            vo_full = bc_handler.get_full_field('vo', tstart)
            try:
                sst_full = bc_handler.get_full_field('sst', tstart)
            except (ValueError, AttributeError):
                sst_full = None
            hycom_available = True
        except (AttributeError, Exception) as e:
            print(f"[INIT] WARNING: Could not load HYCOM full fields: {e}")
            hycom_available = False
    else:
        hycom_available = False

    # Geostrophic velocity blending
    if hycom_available and geostrophic_blend > 0 and lat_grid is not None:
        from mlswe.boundary_handler import geostrophic_velocities
        u_geo, v_geo = geostrophic_velocities(ssh_full, lat_grid, lon_grid, dx, dy)
        gb = float(geostrophic_blend)
        uo_full = (1.0 - gb) * uo_full + gb * u_geo
        vo_full = (1.0 - gb) * vo_full + gb * v_geo
        print(f"[INIT] Geostrophic blend ({gb:.2f}): "
              f"u=[{uo_full.min():.4f}, {uo_full.max():.4f}], "
              f"v=[{vo_full.min():.4f}, {vo_full.max():.4f}] m/s")

    if beta_vel is None:
        beta_vel = [1.0, 0.3, 0.05]  # Default depth decay

    for k in range(N_LAYERS):
        # Initialize layer thickness
        h_k = np.full((ny, nx), H_rest[k], dtype=np.float64)
        if H_b is not None:
            if hycom_available:
                # Use HYCOM SSH to perturb layer thickness
                h_total = np.maximum(H_b + ssh_full, 10.0)
                h_k = (H_rest[k] / H_rest_total) * h_total
            else:
                # Fallback: scale by bathymetry only
                if k < N_LAYERS - 1:
                    ratio = np.where(H_b < H_rest_total,
                                     H_b / H_rest_total, 1.0)
                    h_k = np.maximum(H_rest[k] * ratio, 5.0)
                else:
                    h_above = sum(h0)
                    h_k = np.maximum(H_b - h_above, 5.0)
        h0.append(h_k)

        # Initialize velocities from HYCOM with depth decay
        if hycom_available:
            u0.append(beta_vel[k] * uo_full.copy())
            v0.append(beta_vel[k] * vo_full.copy())
        else:
            u0.append(np.zeros((ny, nx), dtype=np.float64))
            v0.append(np.zeros((ny, nx), dtype=np.float64))

        # Initialize temperature
        if hycom_available and k == 0 and sst_full is not None:
            T0.append(sst_full.copy())
        else:
            T0.append(np.full((ny, nx), T_rest[k], dtype=np.float64))

    # Still apply boundary conditions to ensure consistency at boundaries
    if bc_handler is not None:
        state = {}
        for k in range(N_LAYERS):
            state[f'h{k}'] = h0[k]
            state[f'u{k}'] = u0[k]
            state[f'v{k}'] = v0[k]
            state[f'T{k}'] = T0[k]
        state = bc_handler(state, tstart)
        for k in range(N_LAYERS):
            h0[k] = state[f'h{k}']
            u0[k] = state[f'u{k}']
            v0[k] = state[f'v{k}']
            T0[k] = state[f'T{k}']

    return h0, u0, v0, T0


# ===================================================================
#  Forecast pool  (shallow-copy pattern)
# ===================================================================
_fcst_template = None   # set in main() before pool creation


def _fcst_init_worker():
    global _fcst_model
    _fcst_model = copy.copy(_fcst_template)


def _fcst_advance(args):
    state_flat, t_current, assim_timesteps = args
    _fcst_model.state_flat = state_flat
    _fcst_model.tstart = t_current
    _fcst_model.timesteps = assim_timesteps
    try:
        _fcst_model.advance()
        return (_fcst_model.state_flat.copy(), False)
    except Exception:
        return (state_flat, True)


# ===================================================================
#  DA pool  (shared-memory pvens via mp.RawArray)
# ===================================================================
_da_pvens_buf = None
_da_nanals = _da_dimx = _da_nfields = _da_ncells = 0
_da_grid_lons = _da_grid_lats = None
_da_hcovlocal_scale = 0.0


def _da_chunk_worker(args):
    nstart, nend, obs_values, obs_indices, obs_errvar, obs_cells = args
    pvens = np.frombuffer(_da_pvens_buf, dtype=np.float64).reshape(
        _da_nanals, _da_dimx).copy()  # read-only snapshot
    covlocal_local = precompute_covlocal_local(
        obs_cells, _da_grid_lons, _da_grid_lats,
        _da_hcovlocal_scale, nstart, nend)
    xens_local = letkf_update_mlswe_nl_mpi(
        pvens, obs_values, obs_indices,
        obs_errvar, covlocal_local,
        _da_nfields, _da_ncells, nstart, nend)
    return (nstart, nend, xens_local)


# ===================================================================
#  Main
# ===================================================================

def main():
    # ---- Config ----
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='example_input_mlswe_letkf_best.yml')
    parser.add_argument('--hcovlocal_scale', type=float, default=None)
    parser.add_argument('--covinflate1', type=float, default=None)
    parser.add_argument('--nanals', type=int, default=None)
    parser.add_argument('--ncores', type=int, default=None)
    args = parser.parse_args()

    ncores = args.ncores or mp.cpu_count()

    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        params = yaml.safe_load(f)

    if args.hcovlocal_scale is not None:
        params['hcovlocal_scale'] = args.hcovlocal_scale
    if args.covinflate1 is not None:
        params['covinflate1'] = args.covinflate1
    if args.nanals is not None:
        params['nanals'] = args.nanals

    np.random.seed(42)

    # ---- Grid ----
    ny = int(params['dgy'])
    nx = int(params['dgx'])
    ncells = ny * nx
    nfields = N_LAYERS * FIELDS_PER_LAYER  # 12
    dimx = nfields * ncells

    lon_grid = np.linspace(params['lon_min'], params['lon_max'], nx)
    lat_grid = np.linspace(params['lat_min'], params['lat_max'], ny)
    lat_center = 0.5 * (params['lat_min'] + params['lat_max'])
    dlon = (params['lon_max'] - params['lon_min']) / (nx - 1)
    dlat = (params['lat_max'] - params['lat_min']) / (ny - 1)
    dx, dy = lonlat_to_dxdy(lat_center, dlon, dlat)

    lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
    grid_lons = lon_2d.ravel()
    grid_lats = lat_2d.ravel()

    # ---- LETKF parameters ----
    nanals = int(params.get('nanals', 25))
    hcovlocal_scale_km = float(params.get('hcovlocal_scale', 500.0))
    hcovlocal_scale = hcovlocal_scale_km * 1000.0
    covinflate1 = float(params.get('covinflate1', 0.5))
    covinflate2 = float(params.get('covinflate2', -1.0))

    # Noise per field - controlled by assimilate_fields:
    #   Options: 'uv', 'uv_ssh', 'uv_sst', 'uv_ssh_sst', 'ssh', 'ssh_sst'
    # Only add process noise to fields that are being assimilated.
    # Layers 1,2: never observed -> zero noise.
    sig_x_uv  = float(params.get('sig_x_uv', params.get('sig_x', 0.15)))
    sig_x_sst = float(params.get('sig_x_sst', 1.0))
    sig_x_ssh = float(params.get('sig_x_ssh', 0.0))
    use_swot_ssh = params.get('use_swot_ssh', False)
    assimilate_fields = str(params.get('assimilate_fields', 'uv_sst'))
    assim_uv  = 'uv' in assimilate_fields
    assim_ssh = use_swot_ssh          # SSH observed -> h must have noise
    assim_sst = 'sst' in assimilate_fields
    sig_x_vec = []
    for k in range(N_LAYERS):
        if k == 0:  # surface layer - observed
            sig_x_vec.extend([sig_x_ssh if assim_ssh else 0.0] * ncells)   # h
            sig_x_vec.extend([sig_x_uv  if assim_uv  else 0.0] * ncells)  # u
            sig_x_vec.extend([sig_x_uv  if assim_uv  else 0.0] * ncells)  # v
            sig_x_vec.extend([sig_x_sst if assim_sst else 0.0] * ncells)  # T
        else:       # deeper layers - never observed
            sig_x_vec.extend([0.0] * ncells)  # h
            sig_x_vec.extend([0.0] * ncells)  # u
            sig_x_vec.extend([0.0] * ncells)  # v
            sig_x_vec.extend([0.0] * ncells)  # T
    sig_x_vec = np.array(sig_x_vec)

    # ---- Time ----
    dt = float(params['dt'])
    assim_timesteps = int(params.get('assim_timesteps', params.get('t_freq', 48)))
    T_total = int(params['T'])
    nassim = T_total // assim_timesteps

    print("=" * 64)
    print("  MLSWE_NL LETKF - Multiprocessing-parallelised (3-layer)")
    print(f"  CPU cores: {ncores}")
    print("=" * 64)

    print(f"[LETKF] assimilate_fields='{assimilate_fields}' "
          f"-> uv={assim_uv}, ssh={assim_ssh}, sst={assim_sst}")
    print(f"[LETKF] Noise: h0={sig_x_ssh if assim_ssh else 0.0:.4f}, "
          f"u0,v0={sig_x_uv if assim_uv else 0.0:.4f}, "
          f"T0={sig_x_sst if assim_sst else 0.0:.4f}, layers 1,2=0")
    print(f"[LETKF] Grid: {nx}x{ny}, layers: {N_LAYERS}, nfields: {nfields}, "
          f"dimx: {dimx}")
    print(f"[LETKF] K={nanals}, loc={hcovlocal_scale_km}km, "
          f"inflate={covinflate1}")
    print(f"[LETKF] dt={dt}s, t_freq={assim_timesteps}, nassim={nassim}")

    # ---- Bathymetry ----
    H_b = load_bathymetry(params, ny, nx, lon_grid, lat_grid)
    if H_b is not None:
        print(f"[LETKF] Bathymetry: range=[{H_b.min():.0f}, {H_b.max():.0f}] m")

    # ---- CFL check ----
    g = float(params.get('g', 9.81))
    H_max = float(H_b.max()) if H_b is not None else float(params.get('H_mean', 4000.0))
    c_max = np.sqrt(g * H_max)
    dxy_min = min(dx, dy)
    cfl_dt = 0.5 * dxy_min / c_max
    if dt > cfl_dt:
        safe_dt = 0.9 * cfl_dt
        assim_interval = dt * assim_timesteps
        new_t_freq = max(1, int(np.ceil(assim_interval / safe_dt)))
        dt = assim_interval / new_t_freq
        nassim = T_total // new_t_freq
        assim_timesteps = new_t_freq
        print(f"[LETKF] CFL: dt->{dt:.2f}s, t_freq->{new_t_freq}, nassim->{nassim}")
    else:
        print(f"[LETKF] CFL OK: dt={dt}s (limit {cfl_dt:.1f}s)")

    # ---- Coriolis ----
    f_2d = coriolis_array(params['lat_min'], params['lat_max'], ny, nx)

    # ---- Boundary conditions ----
    bc_file = params.get('bc_file', './data/hycom_bc.nc')
    if not os.path.exists(bc_file):
        bc_file = os.path.join(_SWE_DIR, 'data', 'hycom_bc.nc')

    H_rest = params.get('H_rest', [100.0, 400.0, 3500.0])
    T_rest = params.get('T_rest', [298.15, 283.15, 275.15])

    bc_handler = None
    if os.path.exists(bc_file):
        bc_handler = MLBoundaryHandler(
            nc_path=bc_file,
            model_lon=lon_grid,
            model_lat=lat_grid,
            H_b=H_b,
            H_mean=float(params.get('H_mean', 4000.0)),
            H_rest=H_rest,
            T_rest=T_rest,
            alpha_h=params.get('alpha_h', [0.6, 0.3, 0.1]),
            beta_vel=params.get('beta_vel', [1.0, 0.3, 0.05]),
            n_ghost=params.get('bc_n_ghost', 2),
            sponge_width=params.get('sponge_width', 8),
            sponge_timescale=params.get('sponge_timescale', 3600.0),
            verbose=True,
        )
        print(f"[LETKF] BC from {bc_file}")

    # ---- Load SST nudging reference (HYCOM SST on model grid) ----
    sst_nudging_rate = float(params.get('sst_nudging_rate', 0.0))
    if sst_nudging_rate > 0:
        data_dir = params.get('data_dir', './data')
        import glob as _glob
        # Match SST ref by grid size to pick correct domain
        grid_tag = f'{ny}x{nx}'
        sst_files = sorted(_glob.glob(os.path.join(data_dir,
                                                    f'hycom_sst_ref_*_{grid_tag}_3d.npy')))
        sst_time_files = sorted(_glob.glob(os.path.join(data_dir,
                                                         f'hycom_sst_ref_*_{grid_tag}_times.npy')))
        if not sst_files:
            # Fallback: try any SST ref file
            sst_files = sorted(_glob.glob(os.path.join(data_dir,
                                                        'hycom_sst_ref_*_3d.npy')))
            sst_time_files = sorted(_glob.glob(os.path.join(data_dir,
                                                             'hycom_sst_ref_*_times.npy')))
        if sst_files:
            sst_ref = np.load(sst_files[-1])
            sst_ref_times_raw = np.load(sst_time_files[-1])
            # Derive epoch from obs_time_start (not hardcoded)
            t0_sst_str = params.get('obs_time_start', '2024-08-01T00:00:00')
            t0_sst_dt = datetime.strptime(t0_sst_str[:19], '%Y-%m-%dT%H:%M:%S')
            epoch_obs = (t0_sst_dt - datetime(1970, 1, 1)).total_seconds()
            sst_ref_times = sst_ref_times_raw + epoch_obs
            params['sst_nudging_ref'] = sst_ref
            params['sst_nudging_ref_times'] = sst_ref_times
            print(f"[LETKF] SST nudging: lam={sst_nudging_rate:.6f} s^-1 "
                  f"(tau={1.0/sst_nudging_rate:.0f}s), "
                  f"ref shape={sst_ref.shape}")
        else:
            print("[LETKF] WARNING: SST nudging enabled but no reference found!")

    # ---- Load SSH relaxation reference (from HYCOM BC) ----
    ssh_relax_rate = float(params.get('ssh_relax_rate', 0.0))
    if ssh_relax_rate > 0:
        try:
            from scipy.interpolate import RegularGridInterpolator
            with Dataset(bc_file, 'r') as nc:
                bc_lat_r = np.asarray(nc.variables['lat'][:], dtype=np.float64)
                bc_lon_r = np.asarray(nc.variables['lon'][:], dtype=np.float64)
                bc_ssh_r = np.asarray(nc.variables['ssh'][:], dtype=np.float64)
            bc_ssh_r[np.isnan(bc_ssh_r)] = 0.0
            # Use boundary handler's properly-parsed epoch times
            bc_times_epoch = bc_handler.bc_times
            mg_lat_r, mg_lon_r = np.meshgrid(lat_grid, lon_grid, indexing='ij')
            ssh_ref_3d = np.zeros((len(bc_times_epoch), ny, nx), dtype=np.float64)
            for ti in range(len(bc_times_epoch)):
                interp_r = RegularGridInterpolator(
                    (bc_lat_r, bc_lon_r), bc_ssh_r[ti],
                    method='linear', bounds_error=False, fill_value=0.0)
                ssh_ref_3d[ti] = interp_r((mg_lat_r, mg_lon_r))
            params['ssh_relax_ref'] = ssh_ref_3d
            params['ssh_relax_ref_times'] = bc_times_epoch
            print(f"[LETKF] SSH relaxation: rate={ssh_relax_rate:.2e} s^-1 "
                  f"(tau={1.0/ssh_relax_rate:.0f}s), "
                  f"ref times: [{bc_times_epoch[0]:.0f}, {bc_times_epoch[-1]:.0f}]")
        except Exception as e:
            print(f"[LETKF] WARNING: SSH relax ref load failed: {e}")

    # ---- Load T_air for surface heat flux ----
    sst_flux_type = params.get('sst_flux_type', None)
    if sst_flux_type is not None and params.get('sst_nudging_ref') is not None:
        params['sst_T_air'] = params['sst_nudging_ref']
        params['sst_T_air_times'] = params['sst_nudging_ref_times']
        alpha_val = float(params.get('sst_alpha', 15.0))
        hmix_val = float(params.get('sst_h_mix', 50.0))
        print(f"[LETKF] Heat flux: {sst_flux_type}, "
              f"alpha={alpha_val}, h_mix={hmix_val}m")

    # ---- Initial conditions ----
    # tstart must be in epoch seconds (UTC) so that BC handler, SST nudging,
    # SSH relaxation, and heat flux reference lookups work correctly.
    t0_str = params.get('obs_time_start', '2019-07-01T00:00:00')
    t0_dt = datetime.strptime(t0_str[:19], '%Y-%m-%dT%H:%M:%S')
    tstart = (t0_dt - datetime(1970, 1, 1)).total_seconds()  # UTC epoch
    print(f"[LETKF] tstart = {tstart:.0f} ({_unix_to_datetime(tstart)})")
    beta_vel = params.get('beta_vel', [1.0, 0.3, 0.05])

    # Compute grid spacing in metres (at domain centre latitude)
    lat_centre = 0.5 * (params['lat_min'] + params['lat_max'])
    dx_m_geo = np.deg2rad(abs(lon_grid[1] - lon_grid[0])) * 6.371e6 * np.cos(np.deg2rad(lat_centre))
    dy_m_geo = np.deg2rad(abs(lat_grid[1] - lat_grid[0])) * 6.371e6

    h0, u0, v0, T0 = _make_init_state(
        ny, nx, H_b, np.array(H_rest), np.array(T_rest),
        bc_handler, tstart, beta_vel=beta_vel,
        lat_grid=lat_grid, lon_grid=lon_grid,
        dx=dx_m_geo, dy=dy_m_geo,
        geostrophic_blend=params.get('geostrophic_blend', 0.5))

    # Diagnostic: check initial state statistics
    print(f"[INIT] Layer-0 initial state statistics:")
    print(f"  u: mean={np.mean(u0[0]):.4f} m/s, std={np.std(u0[0]):.4f} m/s, "
          f"range=[{np.min(u0[0]):.4f}, {np.max(u0[0]):.4f}]")
    print(f"  v: mean={np.mean(v0[0]):.4f} m/s, std={np.std(v0[0]):.4f} m/s, "
          f"range=[{np.min(v0[0]):.4f}, {np.max(v0[0]):.4f}]")
    print(f"  T: mean={np.mean(T0[0]):.2f} K, std={np.std(T0[0]):.2f} K, "
          f"range=[{np.min(T0[0]):.2f}, {np.max(T0[0]):.2f}]")
    print(f"  h: mean={np.mean(h0[0]):.2f} m, std={np.std(h0[0]):.2f} m")

    model_kw = dict(
        rho=params.get('rho', [1023, 1026, 1028]),
        dx=dx, dy=dy, dt=dt, f0=f_2d, g=g,
        H_b=H_b,
        H_mean=params.get('H_mean', 4000.0),
        H_rest=H_rest,
        bottom_drag=params.get('bottom_drag', 1e-6),
        diff_coeff=params.get('diff_coeff', 500.0),
        diff_order=params.get('diff_order', 1),
        tracer_diff=params.get('tracer_diff', 100.0),
        bc_handler=bc_handler,
        tstart=tstart,
        precision=params.get('precision', 'double'),
        sst_nudging_rate=params.get('sst_nudging_rate', 0.0),
        sst_nudging_ref=params.get('sst_nudging_ref', None),
        sst_nudging_ref_times=params.get('sst_nudging_ref_times', None),
        ssh_relax_rate=params.get('ssh_relax_rate', 0.0),
        ssh_relax_ref=params.get('ssh_relax_ref', None),
        ssh_relax_ref_times=params.get('ssh_relax_ref_times', None),
        sst_flux_type=params.get('sst_flux_type', None),
        sst_alpha=float(params.get('sst_alpha', 15.0)),
        sst_h_mix=float(params.get('sst_h_mix', 50.0)),
        sst_T_air=params.get('sst_T_air', None),
        sst_T_air_times=params.get('sst_T_air_times', None),
        ssh_relax_interior_floor=float(params.get('ssh_relax_interior_floor', 0.1)),
        shallow_drag_depth=float(params.get('shallow_drag_depth', 500.0)),
        shallow_drag_coeff=float(params.get('shallow_drag_coeff', 5.0e-4)),
    )

    # ---- Template model for forecast pool ----
    template_model = MLSWE(h0, u0, v0, T0=T0, **model_kw)
    ic_flat = template_model.state_flat.copy()

    # ---- Release full HYCOM arrays (no longer needed after init) ----
    # Saves ~3.8 GB - the full (101, 1226, 988) arrays for ssh/uo/vo/sst
    if bc_handler is not None:
        bc_handler.release_full_fields()

    # ---- Load observations ----
    obs_file = params.get('obs_file',
                          '../SWE_LSMCMC/output_comparison_lsmcmc/swe_drifter_obs.nc')
    if not os.path.exists(obs_file):
        obs_file = os.path.join(_SWE_DIR, 'output_comparison_lsmcmc',
                                'swe_drifter_obs.nc')
    if not os.path.exists(obs_file):
        print(f"[LETKF] ERROR: obs file not found: {obs_file}")
        sys.exit(1)

    print(f"[LETKF] Loading obs from {obs_file}")
    yobs_list, ind_list, ind0_list, sig_list, obs_times = _load_obs_netcdf(obs_file)
    nassim_obs = len(yobs_list)
    if nassim_obs < nassim:
        nassim = nassim_obs

    # ---- SWOT SSH observations (merge with drifter obs) ----
    if use_swot_ssh:
        from swot_ssh_data import (
            generate_synthetic_ssh_obs,
            build_ssh_reference_from_hycom,
            merge_obs_arrays,
            download_swot_ssh as _download_swot_ssh,
            load_swot_ssh_to_grid,
            fill_empty_cycles_with_hycom,
        )
        print("[LETKF] Adding SWOT SSH observations ...")
        sig_ssh = float(params.get('sig_y_ssh', params.get('sig_ssh', 0.05)))
        ssh_obs_fraction = float(params.get('ssh_obs_fraction', 0.15))
        H_mean = float(params.get('H_mean', 4000.0))

        # Reconstruct assimilation datetimes
        t0_str = params.get('obs_time_start', '2019-07-01')
        t0_dt = datetime.fromisoformat(t0_str)
        t_freq = int(params.get('t_freq', params.get('assim_timesteps', 48)))
        obs_dt_delta = timedelta(seconds=t_freq * dt)
        obs_datetimes_swot = [t0_dt + i * obs_dt_delta
                              for i in range(1, nassim + 1)]

        swot_source = params.get('swot_source', 'synthetic')
        data_dir = params.get('data_dir', './data')
        bc_file_swot = params.get('bc_file', './data/hycom_bc.nc')
        if not os.path.exists(bc_file_swot):
            bc_file_swot = os.path.join(_SWE_DIR, 'data', 'hycom_bc.nc')

        if swot_source == 'real':
            t1_str = params.get('obs_time_end', '2024-08-12')
            swot_dir = params.get('swot_dir',
                                  os.path.join(data_dir, 'swot_2024aug'))

            # ---- Fast path: load from pre-binned file ----
            binned_file = None
            for tag in [f'{ny}x{nx}', f'{nx}x{ny}']:
                for prefix in ['swot_ssh_combined', 'swot_ssh_binned']:
                    candidate = os.path.join(swot_dir, f'{prefix}_{tag}.nc')
                    if os.path.exists(candidate):
                        binned_file = candidate
                        break
                if binned_file:
                    break
            if binned_file is not None:
                print(f"[LETKF] Loading pre-binned SWOT from {binned_file}")
                with Dataset(binned_file, 'r') as nc_b:
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
                        # Convert ADT to total h: h = H_b + ADT
                        h_obs = H_b_flat[cells] + ssha
                        ssh_yobs[c] = h_obs
                        ssh_ind[c]  = cells.copy()
                        ssh_ind0[c] = cells.copy()
                        ssh_sigy[c] = np.full(n, sig_ssh, dtype=np.float64)
                        n_loaded += 1
                print(f"[LETKF] Loaded {n_loaded}/{nassim} cycles "
                      f"from pre-binned file")
            else:
                # ---- Slow path: load from raw SWOT files ----
                import glob as _glob2
                existing_nc = sorted(_glob2.glob(os.path.join(swot_dir,
                                                              'SWOT_*.nc')))
                if existing_nc:
                    print(f"[LETKF] Using {len(existing_nc)} pre-downloaded "
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
                    swot_files, lon_grid, lat_grid, obs_datetimes_swot,
                    H_mean=H_mean, H_b=H_b, sig_ssh=sig_ssh,
                    ssha_max=_ssha_max,
                )

            # Fill cycles without enough SWOT data with HYCOM SSH + noise
            ssh_ref_3d_fill = params.get('ssh_relax_ref', None)
            ssh_ref_times_fill = params.get('ssh_relax_ref_times', None)
            if ssh_ref_3d_fill is not None and ssh_ref_times_fill is not None:
                obs_epoch_fill = [tstart + (i+1) * t_freq * dt
                                  for i in range(nassim)]
                # Count real obs before filling
                real_counts = np.array([len(a) for a in ssh_yobs])
                nz = real_counts[real_counts > 0]
                real_median = int(np.median(nz)) if len(nz) > 0 else 0

                ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy, n_filled = \
                    fill_empty_cycles_with_hycom(
                        ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy,
                        ssh_ref_3d_fill, ssh_ref_times_fill,
                        obs_epoch_fill, H_b, lon_grid, lat_grid,
                        sig_ssh=sig_ssh,
                        obs_fraction=ssh_obs_fraction,
                    )
                print(f"[LETKF] Filled {n_filled}/{nassim} sparse/empty "
                      f"cycles with HYCOM SSH + noise "
                      f"(threshold < {int(0.5*real_median)} obs, "
                      f"median real={real_median}, "
                      f"sig={sig_ssh:.3f}m, frac={ssh_obs_fraction})")
        else:
            ssh_ref = build_ssh_reference_from_hycom(
                bc_file_swot, lon_grid, lat_grid, obs_datetimes_swot,
                cache_dir=data_dir,
            )
            ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy = generate_synthetic_ssh_obs(
                lon_grid, lat_grid, obs_datetimes_swot,
                ssh_reference=ssh_ref,
                H_mean=H_mean,
                H_b=H_b,
                sig_ssh=sig_ssh,
                obs_fraction=ssh_obs_fraction,
                seed=params.get('ssh_obs_seed', 42),
            )

        # Merge SWOT SSH into drifter obs
        drift_sigy = [s_arr for s_arr in sig_list]
        merged_yobs, merged_ind, merged_ind0, merged_sigy = merge_obs_arrays(
            yobs_list[:nassim], ind_list[:nassim],
            ind0_list[:nassim], drift_sigy[:nassim],
            ssh_yobs, ssh_ind, ssh_ind0, ssh_sigy,
        )
        yobs_list = merged_yobs
        ind_list = merged_ind
        ind0_list = merged_ind0
        sig_list = merged_sigy

        n_ssh = [len(y) for y in ssh_yobs]
        n_total = [len(y) for y in merged_yobs]
        print(f"[LETKF] SWOT SSH: sig={sig_ssh:.3f}m, "
              f"frac={ssh_obs_fraction}, "
              f"ssh/cycle={np.mean(n_ssh):.0f}, "
              f"total/cycle={np.mean(n_total):.0f}")

    nobs_per = [len(y) for y in yobs_list]
    print(f"[LETKF] Obs/cycle: min={min(nobs_per)}, max={max(nobs_per)}, "
          f"mean={np.mean(nobs_per):.0f}")

    # ---- Output directory ----
    outdir = params.get('letkf_dir', './output_letkf')
    os.makedirs(outdir, exist_ok=True)

    # ---- Initialise ensemble ----
    print(f"[LETKF] Initialising {nanals}-member ensemble ...")

    pvens = np.empty((nanals, dimx), dtype=np.float64)

    for k_ens in range(nanals):
        h_pert = [hk + np.random.normal(scale=sig_x_ssh, size=(ny, nx)) for hk in h0]
        u_pert = [uk + np.random.normal(scale=sig_x_uv, size=(ny, nx)) for uk in u0]
        v_pert = [vk + np.random.normal(scale=sig_x_uv, size=(ny, nx)) for vk in v0]
        T_pert = [Tk + np.random.normal(scale=sig_x_sst, size=(ny, nx)) for Tk in T0]
        tmp_mdl = MLSWE(h_pert, u_pert, v_pert, T0=T_pert, **model_kw)
        pvens[k_ens] = tmp_mdl.state_flat.copy()
        del tmp_mdl

    # ---- Create forecast pool ----
    global _fcst_template
    _fcst_template = template_model
    n_fcst_workers = nanals
    fcst_pool = mp.Pool(n_fcst_workers, initializer=_fcst_init_worker)
    print(f"[LETKF] Forecast pool: {n_fcst_workers} workers")

    # ---- Create DA shared memory & pool ----
    global _da_pvens_buf, _da_nanals, _da_dimx
    global _da_nfields, _da_ncells
    global _da_grid_lons, _da_grid_lats, _da_hcovlocal_scale

    _da_pvens_buf = mp.RawArray(ctypes.c_double, nanals * dimx)
    _da_nanals = nanals
    _da_dimx = dimx
    _da_nfields = nfields
    _da_ncells = ncells
    _da_grid_lons = grid_lons
    _da_grid_lats = grid_lats
    _da_hcovlocal_scale = hcovlocal_scale

    n_da_workers = min(ncores, ncells)
    da_pool = mp.Pool(n_da_workers)
    print(f"[LETKF] DA pool: {n_da_workers} workers")

    # Numpy view of shared DA buffer (parent side)
    pvens_shared = np.frombuffer(_da_pvens_buf, dtype=np.float64).reshape(
        nanals, dimx)

    # ---- DA cell chunks ----
    chunk_size = ncells // n_da_workers
    remainder = ncells % n_da_workers
    da_chunks = []
    s = 0
    for i in range(n_da_workers):
        n = chunk_size + (1 if i < remainder else 0)
        da_chunks.append((s, s + n - 1))
        s += n

    # ---- Storage ----
    analysis_mean = np.zeros((nassim + 1, dimx), dtype=np.float64)
    analysis_mean[0] = ic_flat
    rmse_vel_store = np.zeros(nassim)
    rmse_sst_store = np.zeros(nassim)
    rmse_ssh_store = np.zeros(nassim)

    # =================================================================
    #  CYCLING
    # =================================================================
    print(f"\n[LETKF] Starting {nassim} assimilation cycles ...")

    t_current = tstart

    for ntime in range(nassim):
        t_step = time.time()

        obs_values = yobs_list[ntime]
        obs_indices = ind_list[ntime]    # into [h,u,v,T] x ncells = layer 0
        obs_cells = ind0_list[ntime]
        obs_errvar = sig_list[ntime] ** 2
        nobs = len(obs_values)

        # ---- Ensemble forecast (mp.Pool) ----
        t_fcst = time.time()
        fcst_args = [(pvens[k].copy(), t_current, assim_timesteps)
                     for k in range(nanals)]
        fcst_results = fcst_pool.map(_fcst_advance, fcst_args)

        for k, (state_new, blew_up) in enumerate(fcst_results):
            if not blew_up:
                pvens[k] = state_new

        t_current += assim_timesteps * dt

        # Process noise
        noise = np.random.normal(size=pvens.shape) * sig_x_vec[np.newaxis, :]
        pvens += noise

        # Stability clamp
        # NOTE: state_flat stores h_total in slot 0; slots 4,8 (h1,h2) are
        #       zero placeholders.  Only clamp h_total (slot 0), not h1/h2.
        # Bathymetry-aware: limit SSH anomaly to ±20 m  (prevents h_total ≪ H_b
        # at shallow cells which cause extreme pressure gradients and NaN).
        pvens_3d = pvens.reshape(nanals, nfields, ncells)
        if H_b is not None:
            H_b_flat = H_b.ravel()
            h_min = np.maximum(5.0, H_b_flat - 20.0)   # SSH >= -20 m
            pvens_3d[:, 0, :] = np.maximum(pvens_3d[:, 0, :],
                                           h_min[np.newaxis, :])
        else:
            pvens_3d[:, 0, :] = np.clip(pvens_3d[:, 0, :], 5.0, None)
        for k_lay in range(N_LAYERS):
            off = k_lay * FIELDS_PER_LAYER
            pvens_3d[:, off+1, :] = np.clip(pvens_3d[:, off+1, :], -20, 20)    # u
            pvens_3d[:, off+2, :] = np.clip(pvens_3d[:, off+2, :], -20, 20)    # v
            pvens_3d[:, off+3, :] = np.clip(pvens_3d[:, off+3, :], 250, 320)   # T
        pvens = pvens_3d.reshape(nanals, dimx)

        t_fcst_elapsed = time.time() - t_fcst

        # Background spread (for RTPS)
        pvens_3d = pvens.reshape(nanals, nfields, ncells)
        pvensmean_b = pvens_3d.mean(axis=0)
        fsprd = np.sqrt(((pvens_3d - pvensmean_b) ** 2).sum(axis=0) / (nanals - 1))

        # ---- LETKF update (mp.Pool with shared-memory pvens) ----
        t_da = time.time()
        if nobs > 0:
            # Copy current ensemble into shared memory for DA workers
            pvens_shared[:] = pvens

            da_args = [(ns, ne, obs_values, obs_indices, obs_errvar,
                        obs_cells)
                       for ns, ne in da_chunks]
            da_results = da_pool.map(_da_chunk_worker, da_args)

            # Assemble updated cell chunks
            pvens_3d = pvens.reshape(nanals, nfields, ncells)
            for ns, ne, xens_local in da_results:
                pvens_3d[:, :, ns:ne + 1] = xens_local
            pvens = pvens_3d.reshape(nanals, dimx)

        t_da_elapsed = time.time() - t_da

        # ---- RTPS inflation ----
        if covinflate1 > 0.0 and nobs > 0:
            pvens_3d = pvens.reshape(nanals, nfields, ncells)
            pvensmean_a = pvens_3d.mean(axis=0)
            pvprime_a = pvens_3d - pvensmean_a
            asprd = np.sqrt((pvprime_a ** 2).sum(axis=0) / (nanals - 1))
            if covinflate2 < 0:
                safe_asprd = np.where(asprd > 1e-10, asprd, 1e-10)
                inflation = 1.0 + covinflate1 * (fsprd - asprd) / safe_asprd
            else:
                inflation = 1.0 + covinflate1
            pvprime_a *= inflation
            pvens_3d = pvprime_a + pvensmean_a
            pvens = pvens_3d.reshape(nanals, dimx)

        # ---- Post-analysis SSH relaxation ----
        ssh_analysis_relax_frac = float(params.get('ssh_analysis_relax_frac', 0.0))
        if ssh_analysis_relax_frac > 0.0 and H_b is not None:
            ssh_ref_3d_arr = params.get('ssh_relax_ref', None)
            ssh_ref_times = params.get('ssh_relax_ref_times', None)
            H_b_flat = H_b.ravel()  # (ncells,)

            # Get reference SSH at current time (epoch seconds)
            t_now = tstart + (ntime + 1) * assim_timesteps * dt
            if ssh_ref_3d_arr is not None and ssh_ref_times is not None:
                idx = np.searchsorted(ssh_ref_times, t_now, side='right') - 1
                idx = np.clip(idx, 0, len(ssh_ref_times) - 1)
                eta_ref = ssh_ref_3d_arr[idx].ravel()  # (ncells,)
            else:
                eta_ref = np.zeros(ncells, dtype=np.float64)

            pvens_3d = pvens.reshape(nanals, nfields, ncells)
            for k_ens in range(nanals):
                # h_total lives in slot 0 only (slots 4,8 are zero placeholders)
                h_total = pvens_3d[k_ens, 0, :]  # slot 0 = h_total
                eta = h_total - H_b_flat
                # Apply correction only to h_total (slot 0)
                correction = ssh_analysis_relax_frac * (eta - eta_ref)
                pvens_3d[k_ens, 0, :] -= correction
            pvens = pvens_3d.reshape(nanals, dimx)

        # ---- Post-analysis NaN / Inf recovery ----
        # LETKF update or RTPS inflation can produce NaN at cells with
        # extreme bathymetry gradients or near-zero spread.  Replace any
        # non-finite member values with the (finite) ensemble mean, and
        # re-apply the bathymetry-aware clamp.
        nan_members = ~np.isfinite(pvens).all(axis=1)          # (nanals,)
        if nan_members.any():
            n_bad = int(nan_members.sum())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ens_mean = np.nanmean(pvens, axis=0)           # (dimx,)
            still_nan = ~np.isfinite(ens_mean)
            if still_nan.any():
                # Fallback: use pre-update (background) ensemble mean
                ens_mean[still_nan] = pvensmean_b.ravel()[still_nan]
            for k_ens in np.where(nan_members)[0]:
                bad = ~np.isfinite(pvens[k_ens])
                pvens[k_ens, bad] = ens_mean[bad]
            # Re-apply bathymetry-aware stability clamp
            pvens_3d = pvens.reshape(nanals, nfields, ncells)
            if H_b is not None:
                H_b_flat2 = H_b.ravel()
                h_min2 = np.maximum(5.0, H_b_flat2 - 20.0)
                pvens_3d[:, 0, :] = np.maximum(pvens_3d[:, 0, :],
                                               h_min2[np.newaxis, :])
            for k_lay in range(N_LAYERS):
                off = k_lay * FIELDS_PER_LAYER
                pvens_3d[:, off+1, :] = np.clip(pvens_3d[:, off+1, :], -20, 20)
                pvens_3d[:, off+2, :] = np.clip(pvens_3d[:, off+2, :], -20, 20)
                pvens_3d[:, off+3, :] = np.clip(pvens_3d[:, off+3, :], 250, 320)
            pvens = pvens_3d.reshape(nanals, dimx)
            print(f"    [NaN recovery] cycle {ntime+1}: "
                  f"replaced {n_bad}/{nanals} members")

        # ---- RMSE ----
        z_a = pvens.mean(axis=0)
        analysis_mean[ntime + 1] = z_a

        if nobs > 0:
            obs_arr = np.asarray(obs_indices, dtype=int)
            residuals = z_a[obs_arr] - obs_values
            # SSH (h): field 0, indices in [0, ncells)
            ssh_mask = obs_arr < ncells
            # Velocity (u, v): fields 1-2, indices in [ncells, 3*ncells)
            vel_mask = (obs_arr >= ncells) & (obs_arr < 3 * ncells)
            # SST (T): field 3, indices in [3*ncells, 4*ncells)
            sst_mask = (obs_arr >= 3 * ncells) & (obs_arr < 4 * ncells)
            rmse_vel = np.sqrt(np.mean(residuals[vel_mask]**2)) if vel_mask.sum() > 0 else 0.0
            rmse_sst = np.sqrt(np.mean(residuals[sst_mask]**2)) if sst_mask.sum() > 0 else 0.0
            rmse_ssh = np.sqrt(np.mean(residuals[ssh_mask]**2)) if ssh_mask.sum() > 0 else 0.0
            rmse_vel_store[ntime] = rmse_vel
            rmse_sst_store[ntime] = rmse_sst
            rmse_ssh_store[ntime] = rmse_ssh
            sst_str = f"  sst={rmse_sst:.3f}K" if sst_mask.sum() > 0 else ""
            ssh_str = f"  ssh={rmse_ssh:.3f}m" if ssh_mask.sum() > 0 else ""
        else:
            rmse_vel_store[ntime] = np.nan
            rmse_sst_store[ntime] = np.nan
            rmse_ssh_store[ntime] = np.nan
            rmse_vel = np.nan
            sst_str = ""
            ssh_str = ""

        dt_wall = time.time() - t_step
        print(f"  [{ntime+1:04d}/{nassim}]  nobs={nobs:4d}  "
              f"vel={rmse_vel:.3e}{sst_str}{ssh_str}  "
              f"fcst={t_fcst_elapsed:.1f}s  da={t_da_elapsed:.1f}s  "
              f"wall={dt_wall:.1f}s")

    # =================================================================
    #  Cleanup pools
    # =================================================================
    fcst_pool.close()
    fcst_pool.join()
    da_pool.close()
    da_pool.join()

    # =================================================================
    #  Write output
    # =================================================================
    print(f"\n[LETKF] Writing output ...")
    outfn = os.path.join(outdir, 'mlswe_letkf_out.nc')
    with Dataset(outfn, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', nassim + 1)
        nc.createDimension('layer', N_LAYERS)
        nc.createDimension('field', FIELDS_PER_LAYER)
        nc.createDimension('y', ny)
        nc.createDimension('x', nx)
        nc.createDimension('cycle', nassim)

        v = nc.createVariable('lsmcmc_mean', 'f4',
                              ('time', 'layer', 'field', 'y', 'x'),
                              zlib=True)
        reshaped = analysis_mean.reshape(
            nassim + 1, N_LAYERS, FIELDS_PER_LAYER, ny, nx)
        v[:] = reshaped.astype(np.float32)

        rv = nc.createVariable('rmse_vel', 'f4', ('cycle',), zlib=True)
        rv[:] = rmse_vel_store.astype(np.float32)
        rs = nc.createVariable('rmse_sst', 'f4', ('cycle',), zlib=True)
        rs[:] = rmse_sst_store.astype(np.float32)
        rh = nc.createVariable('rmse_ssh', 'f4', ('cycle',), zlib=True)
        rh[:] = rmse_ssh_store.astype(np.float32)

        # Save bathymetry so plotter always uses the correct H_b
        if H_b is not None:
            vb = nc.createVariable('H_b', 'f4', ('y', 'x'), zlib=True)
            vb[:] = H_b.astype(np.float32)

        vt = nc.createVariable('obs_times', 'f8', ('time',))
        if obs_times is not None and len(obs_times) >= nassim:
            vt[0] = tstart          # initial time in epoch seconds
            vt[1:] = obs_times[:nassim]
        else:
            vt[:] = tstart + np.arange(nassim + 1) * assim_timesteps * dt

        nc.description = 'MLSWE (3-layer) LETKF analysis output'
        nc.nlayers = N_LAYERS
        nc.fields_per_layer = FIELDS_PER_LAYER
        nc.nanals = nanals
        nc.hcovlocal_scale_km = hcovlocal_scale_km
        nc.covinflate1 = covinflate1
        nc.dt = dt
        nc.nassim = nassim

    print(f"[LETKF] Wrote {outfn}")
    print(f"[LETKF] Mean RMSE_vel = {np.nanmean(rmse_vel_store):.5f}")
    print(f"[LETKF] Mean RMSE_sst = {np.nanmean(rmse_sst_store):.5f}")
    print(f"[LETKF] Mean RMSE_ssh = {np.nanmean(rmse_ssh_store):.5f}")

    # ---- Auto-generate diagnostic plots ----
    try:
        from plot_mlswe_results import main as plot_results
        print("\n[LETKF] Generating diagnostic plots ...")
        plot_results(
            outdir=outdir,
            save_prefix='mlswe_results',
            config_file=args.config,
            method_label='LETKF',
        )
    except Exception as e:
        print(f"[LETKF] WARNING: Plotting failed: {e}")
        import traceback; traceback.print_exc()

    print("[LETKF] All done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
run_nlgamma_twin_letkf.py
==========================
Multiprocessing-parallelised LETKF for the NL-Gamma twin experiment.

Uses ``multiprocessing.Pool`` for both forecast ensemble advancement and
LETKF analysis update — no MPI dependency.

Uses:
    * **Nonlinear obs operator:**  h(x) = arctan(x)
    * **Gaussian likelihood** (misspecified — truth is Student-t noise)
    * Pre-generated nature run + synthetic obs from V1 output directory

Usage
-----
    python3 nlgamma_ldata/run_nlgamma_twin_letkf.py [config.yml]

Expects:
    - Nature run in ``output_nlgamma_twin_V1/truth_trajectory.nc``
    - Synthetic obs in ``output_nlgamma_twin_V1/synthetic_nlgamma_obs.nc``
  (generate with:  python nlgamma_ldata/run_nlgamma_twin.py --truth-only)
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from mlswe.model import MLSWE, coriolis_array, lonlat_to_dxdy
from mlswe.boundary_handler import MLBoundaryHandler
from mlswe.letkf import calcwts_letkf, precompute_covlocal_local

N_LAYERS = 3
FIELDS_PER_LAYER = 4


# ===================================================================
#  Nonlinear LETKF update — h(x) = arctan(x)
# ===================================================================


def letkf_update_softclip_mpi(xens, obs_values, obs_indices,
                               obs_errvar, covlocal_local,
                               nfields, ncells, nstart, nend):
    """
    LETKF analysis update with h(x) = arctan(x) obs operator.
    """
    nanals = xens.shape[0]
    nobs = len(obs_values)
    nlocal = nend - nstart + 1

    xens_3d = xens.reshape(nanals, nfields, ncells)

    # Nonlinear obs operator: h(x) = arctan(x)
    raw = xens[:, obs_indices]
    hxens = np.arctan(raw)
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean

    innovations = obs_values - hxmean

    xmean_3d = xens_3d.mean(axis=0)
    xprime_local = xens_3d[:, :, nstart:nend+1] - \
                   xmean_3d[:, nstart:nend+1]
    xmean_local = xmean_3d[:, nstart:nend+1]

    inv_obs_errvar = 1.0 / obs_errvar
    xens_updated_local = xens_3d[:, :, nstart:nend+1].copy()

    for count in range(nlocal):
        loc_wt = covlocal_local[:, count]
        mask = loc_wt > 1.e-10
        nobs_local = int(mask.sum())
        if nobs_local == 0:
            continue

        rinv_diag = loc_wt[mask] * inv_obs_errvar[mask]
        hx_local = hxprime[:, mask]
        innov_local = innovations[mask]

        wts = calcwts_letkf(hx_local, rinv_diag, innov_local, nanals)
        if not np.isfinite(wts).all():
            continue

        xp = xprime_local[:, :, count]
        updated = xmean_local[:, count] + xp.T.dot(wts).T
        if np.isfinite(updated).all():
            xens_updated_local[:, :, count] = updated

    return xens_updated_local


# ===================================================================
#  Helpers
# ===================================================================
def _load_obs_netcdf(obs_file):
    nc = Dataset(obs_file, 'r')
    raw_obs = np.asarray(nc.variables['yobs_all'][:])
    raw_ind = np.asarray(nc.variables['yobs_ind_all'][:])
    raw_sig = np.asarray(nc.variables['sig_y_all'][:])
    obs_times = np.asarray(nc.variables['obs_times'][:])
    nc.close()
    nassim = raw_obs.shape[0]
    yobs_list, ind_list, sig_list = [], [], []
    for n in range(nassim):
        valid = (raw_ind[n] >= 0) & np.isfinite(raw_obs[n])
        yobs_list.append(raw_obs[n, valid].astype(np.float64))
        ind_list.append(raw_ind[n, valid].astype(np.int64))
        sig_list.append(raw_sig[n, valid].astype(np.float64))
    return yobs_list, ind_list, sig_list, obs_times


def _load_truth(truth_file, nassim, dimx):
    nc = Dataset(truth_file, 'r')
    truth_raw = np.asarray(nc.variables['truth'][:])
    nc.close()
    nt = truth_raw.shape[0]
    truth_flat = truth_raw.reshape(nt, -1)
    return truth_flat[:nassim + 1]


def load_bathymetry(params, ny, nx, lon, lat):
    """Load or download bathymetry (reuse from main runner)."""
    try:
        import run_mlswe_lsmcmc_ldata_V1 as _lv1
        return _lv1.load_bathymetry(params, ny, nx, lon, lat)
    except Exception:
        return None


def _make_init_state(ny, nx, H_b, H_rest, T_rest,
                     bc_handler, tstart, **kw):
    """Create initial conditions from BC handler."""
    try:
        from run_mlswe_lsmcmc_ldata_V1 import init_from_bc_handler
        lat_grid = kw.pop('lat_grid', None)
        lon_grid = kw.pop('lon_grid', None)
        return init_from_bc_handler(
            bc_handler, H_b, tstart,
            H_rest=list(H_rest), T_rest=list(T_rest), **kw)
    except Exception:
        nlayers = len(H_rest)
        h0 = [np.full((ny, nx), H_rest[k]) for k in range(nlayers)]
        u0 = [np.zeros((ny, nx)) for _ in range(nlayers)]
        v0 = [np.zeros((ny, nx)) for _ in range(nlayers)]
        T0 = [np.full((ny, nx), T_rest[k]) for k in range(nlayers)]
        return h0, u0, v0, T0


# ===================================================================
#  Forecast pool  (shallow-copy pattern)
# ===================================================================
_fcst_template = None
_fcst_worker_model = None


def _fcst_init_worker():
    """Create per-worker model via shallow copy + state array copies."""
    global _fcst_worker_model
    mdl = _fcst_template
    new_mdl = copy.copy(mdl)
    new_mdl.h = [hk.copy() for hk in mdl.h]
    new_mdl.u = [uk.copy() for uk in mdl.u]
    new_mdl.v = [vk.copy() for vk in mdl.v]
    if mdl.use_tracer:
        new_mdl.T = [Tk.copy() if Tk is not None else None for Tk in mdl.T]
    _fcst_worker_model = new_mdl


def _fcst_advance(args):
    """Advance one ensemble member: set state, run nsteps, return result."""
    state_flat, t_val, nsteps = args
    mdl = _fcst_worker_model
    mdl.state_flat = state_flat
    mdl.t = t_val
    for _ in range(nsteps):
        mdl._timestep()
        if not np.all(np.isfinite(mdl.state_flat)):
            return state_flat.copy(), True   # blow-up sentinel
    return mdl.state_flat.copy(), False


# ===================================================================
#  DA pool  (shared-memory pvens via mp.RawArray)
# ===================================================================
_da_pvens_buf = None
_da_nanals = None
_da_dimx = None
_da_nfields = None
_da_ncells = None
_da_grid_lons = None
_da_grid_lats = None
_da_hcovlocal_scale = None


def _da_chunk_worker(args):
    """LETKF update for a contiguous chunk of grid cells."""
    nstart, nend, obs_values, obs_indices, obs_errvar, obs_cells = args

    # Read ensemble from shared memory
    pvens = np.frombuffer(_da_pvens_buf, dtype=np.float64).reshape(
        _da_nanals, _da_dimx).copy()

    # Compute localization matrix for this cell chunk
    covlocal_local = precompute_covlocal_local(
        obs_cells, _da_grid_lons, _da_grid_lats,
        _da_hcovlocal_scale, nstart, nend)

    # Softclip (arctan) LETKF update
    xens_local = letkf_update_softclip_mpi(
        pvens, obs_values, obs_indices,
        obs_errvar, covlocal_local,
        _da_nfields, _da_ncells, nstart, nend)

    return nstart, nend, xens_local


# ===================================================================
#  Main
# ===================================================================
def main():
    print("=" * 64)
    print("  NL-Gamma LETKF Twin  (softclip obs + Gamma noise, multiprocessing)")
    print("=" * 64)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?',
                        default=os.path.join(
                            _HERE,
                            'input_nlgamma_twin_letkf.yml'))
    parser.add_argument('--ncores', type=int, default=None)
    args = parser.parse_args()

    ncores = args.ncores or mp.cpu_count()

    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        params = yaml.safe_load(f)

    np.random.seed(42)

    # ---- Grid ----
    ny = int(params['dgy'])
    nx = int(params['dgx'])
    ncells = ny * nx
    nfields = N_LAYERS * FIELDS_PER_LAYER
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
    hcovlocal_scale_km = float(params.get('hcovlocal_scale', 200.0))
    hcovlocal_scale = hcovlocal_scale_km * 1000.0
    covinflate1 = float(params.get('covinflate1', 0.95))

    sig_x_uv  = float(params.get('sig_x_uv', 0.15))
    sig_x_sst = float(params.get('sig_x_sst', 1.0))
    sig_x_ssh = float(params.get('sig_x_ssh', 0.5))

    assimilate_fields = str(params.get('assimilate_fields', 'uv_ssh_sst'))
    assim_uv  = 'uv' in assimilate_fields
    assim_ssh = 'ssh' in assimilate_fields
    assim_sst = 'sst' in assimilate_fields

    sig_x_vec = []
    for k in range(N_LAYERS):
        if k == 0:
            sig_x_vec.extend([sig_x_ssh if assim_ssh else 0.0] * ncells)
            sig_x_vec.extend([sig_x_uv  if assim_uv  else 0.0] * ncells)
            sig_x_vec.extend([sig_x_uv  if assim_uv  else 0.0] * ncells)
            sig_x_vec.extend([sig_x_sst if assim_sst else 0.0] * ncells)
        else:
            sig_x_vec.extend([0.0] * 4 * ncells)
    sig_x_vec = np.array(sig_x_vec)

    dt = float(params['dt'])
    assim_timesteps = int(params.get('assim_timesteps',
                                      params.get('t_freq', 48)))
    T_total = int(params['T'])
    nassim = T_total // assim_timesteps

    print(f"[NLG-LETKF] Grid: {nx}x{ny}, dimx={dimx}")
    print(f"[NLG-LETKF] K={nanals}, loc={hcovlocal_scale_km}km, "
          f"inflate={covinflate1}")
    print(f"[NLG-LETKF] nassim={nassim}, ncores={ncores}")

    # ---- Bathymetry ----
    H_b = load_bathymetry(params, ny, nx, lon_grid, lat_grid)

    # ---- Coriolis ----
    f_2d = coriolis_array(params['lat_min'], params['lat_max'], ny, nx)
    g = float(params.get('g', 9.81))

    # ---- Boundary conditions ----
    bc_file = params.get('bc_file', './data/hycom_bc_2024aug.nc')
    H_rest = params.get('H_rest', [100.0, 400.0, 3500.0])
    T_rest = params.get('T_rest', [298.15, 283.15, 275.15])

    bc_handler = None
    if os.path.exists(bc_file):
        bc_handler = MLBoundaryHandler(
            nc_path=bc_file,
            model_lon=lon_grid, model_lat=lat_grid, H_b=H_b,
            H_mean=float(params.get('H_mean', 4000.0)),
            H_rest=H_rest, T_rest=T_rest,
            alpha_h=params.get('alpha_h', [0.6, 0.3, 0.1]),
            beta_vel=params.get('beta_vel', [1.0, 1.0, 1.0]),
            n_ghost=params.get('bc_n_ghost', 2),
            sponge_width=params.get('sponge_width', 8),
            sponge_timescale=params.get('sponge_timescale', 3600.0),
            verbose=True,
        )

    # ---- SST / SSH references ----
    sst_nudging_rate = float(params.get('sst_nudging_rate', 0.0))
    if sst_nudging_rate > 0:
        data_dir = params.get('data_dir', './data')
        import glob as _glob
        grid_tag = f'{ny}x{nx}'
        sst_files = sorted(_glob.glob(os.path.join(
            data_dir, f'hycom_sst_ref_*_{grid_tag}_3d.npy')))
        sst_time_files = sorted(_glob.glob(os.path.join(
            data_dir, f'hycom_sst_ref_*_{grid_tag}_times.npy')))
        if not sst_files:
            sst_files = sorted(_glob.glob(os.path.join(
                data_dir, 'hycom_sst_ref_*_3d.npy')))
            sst_time_files = sorted(_glob.glob(os.path.join(
                data_dir, 'hycom_sst_ref_*_times.npy')))
        if sst_files:
            sst_ref = np.load(sst_files[-1])
            sst_ref_times_raw = np.load(sst_time_files[-1])
            t0_str = params.get('obs_time_start', '2024-08-01T00:00:00')
            t0_dt = datetime.strptime(t0_str[:19], '%Y-%m-%dT%H:%M:%S')
            epoch = (t0_dt - datetime(1970, 1, 1)).total_seconds()
            sst_ref_times = sst_ref_times_raw + epoch
            params['sst_nudging_ref'] = sst_ref
            params['sst_nudging_ref_times'] = sst_ref_times

    ssh_relax_rate = float(params.get('ssh_relax_rate', 0.0))
    if ssh_relax_rate > 0:
        try:
            from scipy.interpolate import RegularGridInterpolator
            with Dataset(bc_file, 'r') as nc:
                bc_lat_r = np.asarray(nc.variables['lat'][:])
                bc_lon_r = np.asarray(nc.variables['lon'][:])
                bc_ssh_r = np.asarray(nc.variables['ssh'][:])
            bc_ssh_r[np.isnan(bc_ssh_r)] = 0.0
            bc_times_epoch = bc_handler.bc_times
            mg_lat_r, mg_lon_r = np.meshgrid(lat_grid, lon_grid,
                                              indexing='ij')
            ssh_ref_3d = np.zeros((len(bc_times_epoch), ny, nx))
            for ti in range(len(bc_times_epoch)):
                interp_r = RegularGridInterpolator(
                    (bc_lat_r, bc_lon_r), bc_ssh_r[ti],
                    method='linear', bounds_error=False, fill_value=0.0)
                ssh_ref_3d[ti] = interp_r((mg_lat_r, mg_lon_r))
            params['ssh_relax_ref'] = ssh_ref_3d
            params['ssh_relax_ref_times'] = bc_times_epoch
        except Exception as e:
            print(f"[NLG-LETKF] WARNING: SSH ref failed: {e}")

    if params.get('sst_flux_type') is not None:
        if params.get('sst_nudging_ref') is not None:
            params['sst_T_air'] = params['sst_nudging_ref']
            params['sst_T_air_times'] = params['sst_nudging_ref_times']

    # ---- Initial conditions ----
    t0_str = params.get('obs_time_start', '2024-08-01T00:00:00')
    t0_dt = datetime.strptime(t0_str[:19], '%Y-%m-%dT%H:%M:%S')
    tstart = (t0_dt - datetime(1970, 1, 1)).total_seconds()

    h0, u0, v0, T0 = _make_init_state(
        ny, nx, H_b, np.array(H_rest), np.array(T_rest),
        bc_handler, tstart,
        beta_vel=params.get('beta_vel', [1.0, 1.0, 1.0]),
        lat_grid=lat_grid, lon_grid=lon_grid,
        dx=dx, dy=dy,
        geostrophic_blend=params.get('geostrophic_blend', 0.5))

    # ---- Model keyword dict ----
    model_kw = dict(
        rho=params.get('rho', [1023, 1026, 1028]),
        dx=dx, dy=dy, dt=dt, f0=f_2d, g=g,
        H_b=H_b, H_mean=params.get('H_mean', 4000.0),
        H_rest=H_rest,
        bottom_drag=params.get('bottom_drag', 1e-6),
        diff_coeff=params.get('diff_coeff', 500.0),
        diff_order=params.get('diff_order', 1),
        tracer_diff=params.get('tracer_diff', 100.0),
        bc_handler=bc_handler, tstart=tstart,
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
        ssh_relax_interior_floor=float(
            params.get('ssh_relax_interior_floor', 0.1)),
        shallow_drag_depth=float(
            params.get('shallow_drag_depth', 500.0)),
        shallow_drag_coeff=float(
            params.get('shallow_drag_coeff', 5.0e-4)),
    )

    # ---- Template model for forecast pool ----
    template_model = MLSWE(h0, u0, v0, T0=T0, **model_kw)
    ic_flat = template_model.state_flat.copy()

    if bc_handler is not None:
        bc_handler.release_full_fields()

    # ---- Load synthetic observations ----
    twin_dir = params.get('twin_dir', './output_nlgamma_twin_V1')
    obs_file = os.path.join(twin_dir, 'synthetic_nlgamma_obs.nc')
    truth_file = os.path.join(twin_dir, 'truth_trajectory.nc')

    if not os.path.exists(obs_file):
        print(f"ERROR: Synthetic obs not found: {obs_file}")
        print("Run V1 first:  python3 -u nlgamma_ldata/run_nlgamma_twin.py --truth-only")
        sys.exit(1)

    print(f"[NLG-LETKF] Obs: {obs_file}")
    print(f"[NLG-LETKF] Nature Run: {truth_file}")

    yobs_list, ind_list, sig_list, obs_times = \
        _load_obs_netcdf(obs_file)
    nassim_obs = len(yobs_list)
    if nassim_obs < nassim:
        nassim = nassim_obs
    truth_states = _load_truth(truth_file, nassim, dimx)
    nobs_per = [len(y) for y in yobs_list]
    print(f"[NLG-LETKF] Obs/cycle: min={min(nobs_per)}, "
          f"max={max(nobs_per)}, mean={np.mean(nobs_per):.0f}")

    # ---- Output directory ----
    outdir = params.get('letkf_dir', './output_nlgamma_twin_letkf')
    os.makedirs(outdir, exist_ok=True)

    # ---- Create forecast pool ----
    global _fcst_template
    _fcst_template = template_model
    n_fcst_workers = nanals
    fcst_pool = mp.Pool(n_fcst_workers, initializer=_fcst_init_worker)
    print(f"[NLG-LETKF] Forecast pool: {n_fcst_workers} workers")

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
    print(f"[NLG-LETKF] DA pool: {n_da_workers} workers")

    # ---- Initialise ensemble ----
    pvens = np.empty((nanals, dimx), dtype=np.float64)
    for k_ens in range(nanals):
        h_pert = [hk + np.random.normal(scale=sig_x_ssh, size=(ny, nx))
                  for hk in h0]
        u_pert = [uk + np.random.normal(scale=sig_x_uv, size=(ny, nx))
                  for uk in u0]
        v_pert = [vk + np.random.normal(scale=sig_x_uv, size=(ny, nx))
                  for vk in v0]
        T_pert = [Tk + np.random.normal(scale=sig_x_sst, size=(ny, nx))
                  for Tk in T0]
        tmp_mdl = MLSWE(h_pert, u_pert, v_pert, T0=T_pert, **model_kw)
        pvens[k_ens] = tmp_mdl.state_flat.copy()
        del tmp_mdl

    print(f"[NLG-LETKF] Ensemble: {nanals} members initialised")

    # Numpy view of shared DA buffer (parent side)
    pvens_shared = np.frombuffer(_da_pvens_buf, dtype=np.float64).reshape(
        nanals, dimx)

    # ---- DA cell chunks ----
    chunk_size = ncells // n_da_workers
    remainder_da = ncells % n_da_workers
    da_chunks = []
    s = 0
    for i in range(n_da_workers):
        n = chunk_size + (1 if i < remainder_da else 0)
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
    print(f"\n[NLG-LETKF] Starting {nassim} cycles ...")
    t_total_start = time.time()
    t_current = tstart

    for ntime in range(nassim):
        t_step = time.time()

        obs_values  = yobs_list[ntime]
        obs_indices = ind_list[ntime]
        obs_cells   = obs_indices % ncells
        obs_errvar  = sig_list[ntime] ** 2
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
        noise = np.random.normal(size=pvens.shape) * sig_x_vec
        pvens += noise

        # Stability clamp
        pvens_3d = pvens.reshape(nanals, nfields, ncells)
        if H_b is not None:
            H_b_flat = H_b.ravel()
            h_min = np.maximum(5.0, H_b_flat - 20.0)
            pvens_3d[:, 0, :] = np.maximum(pvens_3d[:, 0, :],
                                           h_min[np.newaxis, :])
        for k_lay in range(N_LAYERS):
            off = k_lay * FIELDS_PER_LAYER
            pvens_3d[:, off+1, :] = np.clip(pvens_3d[:, off+1, :],
                                            -20, 20)
            pvens_3d[:, off+2, :] = np.clip(pvens_3d[:, off+2, :],
                                            -20, 20)
            pvens_3d[:, off+3, :] = np.clip(pvens_3d[:, off+3, :],
                                            250, 320)
        pvens = pvens_3d.reshape(nanals, dimx)
        t_fcst_elapsed = time.time() - t_fcst

        # Background spread
        pvens_3d = pvens.reshape(nanals, nfields, ncells)
        pvensmean_b = pvens_3d.mean(axis=0)
        fsprd = np.sqrt(((pvens_3d - pvensmean_b) ** 2).sum(axis=0)
                        / (nanals - 1))

        # ---- LETKF update (softclip obs operator, mp.Pool) ----
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
            safe_asprd = np.where(asprd > 1e-10, asprd, 1e-10)
            inflation = 1.0 + covinflate1 * (fsprd - asprd) / safe_asprd
            pvprime_a *= inflation
            pvens_3d = pvprime_a + pvensmean_a
            pvens = pvens_3d.reshape(nanals, dimx)

        # ---- NaN recovery ----
        nan_members = ~np.isfinite(pvens).all(axis=1)
        if nan_members.any():
            n_bad = int(nan_members.sum())
            ens_mean = np.nanmean(pvens, axis=0)
            still_nan = ~np.isfinite(ens_mean)
            if still_nan.any():
                ens_mean[still_nan] = pvensmean_b.ravel()[still_nan]
            for k_ens in np.where(nan_members)[0]:
                bad = ~np.isfinite(pvens[k_ens])
                pvens[k_ens, bad] = ens_mean[bad]
            pvens_3d = pvens.reshape(nanals, nfields, ncells)
            if H_b is not None:
                H_b_flat2 = H_b.ravel()
                h_min2 = np.maximum(5.0, H_b_flat2 - 20.0)
                pvens_3d[:, 0, :] = np.maximum(pvens_3d[:, 0, :],
                                               h_min2[np.newaxis, :])
            for k_lay in range(N_LAYERS):
                off = k_lay * FIELDS_PER_LAYER
                pvens_3d[:, off+1, :] = np.clip(
                    pvens_3d[:, off+1, :], -20, 20)
                pvens_3d[:, off+2, :] = np.clip(
                    pvens_3d[:, off+2, :], -20, 20)
                pvens_3d[:, off+3, :] = np.clip(
                    pvens_3d[:, off+3, :], 250, 320)
            pvens = pvens_3d.reshape(nanals, dimx)
            print(f"    [NaN] cycle {ntime+1}: {n_bad}/{nanals}")

        # ---- RMSE against nature run ----
        z_a = pvens.mean(axis=0)
        analysis_mean[ntime + 1] = z_a
        z_t = truth_states[ntime + 1]
        diff = z_a - z_t

        vel_diff = diff[ncells:3*ncells]
        rmse_vel = np.sqrt(np.mean(vel_diff**2))
        sst_diff = diff[3*ncells:4*ncells]
        rmse_sst = np.sqrt(np.mean(sst_diff**2))
        ssh_diff = diff[0:ncells]
        rmse_ssh = np.sqrt(np.mean(ssh_diff**2))

        rmse_vel_store[ntime] = rmse_vel
        rmse_sst_store[ntime] = rmse_sst
        rmse_ssh_store[ntime] = rmse_ssh

        dt_wall = time.time() - t_step
        if (ntime + 1) % 10 == 0 or ntime == 0:
            print(f"  [{ntime+1:04d}/{nassim}]  nobs={nobs:4d}  "
                  f"vel={rmse_vel:.4f}  sst={rmse_sst:.3f}  "
                  f"ssh={rmse_ssh:.3f}  "
                  f"fcst={t_fcst_elapsed:.1f}s  "
                  f"da={t_da_elapsed:.1f}s  "
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
    t_total_elapsed = time.time() - t_total_start
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

        rv = nc.createVariable('rmse_vel', 'f4', ('cycle',))
        rv[:] = rmse_vel_store.astype(np.float32)
        rs = nc.createVariable('rmse_sst', 'f4', ('cycle',))
        rs[:] = rmse_sst_store.astype(np.float32)
        rh = nc.createVariable('rmse_ssh', 'f4', ('cycle',))
        rh[:] = rmse_ssh_store.astype(np.float32)

        if H_b is not None:
            vb = nc.createVariable('H_b', 'f4', ('y', 'x'), zlib=True)
            vb[:] = H_b.astype(np.float32)

        vt = nc.createVariable('obs_times', 'f8', ('time',))
        if obs_times is not None and len(obs_times) >= nassim:
            vt[0] = tstart
            vt[1:] = obs_times[:nassim]

        nc.description = ('NL-Gamma twin — LETKF '
                          '(softclip arctan obs, Gamma noise, multiprocessing)')
        nc.nanals = nanals
        nc.hcovlocal_scale_km = hcovlocal_scale_km
        nc.covinflate1 = covinflate1

    print(f"\n[NLG-LETKF] Wrote {outfn}")
    print(f"[NLG-LETKF] Total wall time: {t_total_elapsed:.1f}s "
          f"({t_total_elapsed/60:.1f} min)")
    print(f"[NLG-LETKF] Mean RMSE_vel = {np.nanmean(rmse_vel_store):.5f}")
    print(f"[NLG-LETKF] Mean RMSE_sst = {np.nanmean(rmse_sst_store):.4f}")
    print(f"[NLG-LETKF] Mean RMSE_ssh = {np.nanmean(rmse_ssh_store):.4f}")
    print(f"[NLG-LETKF] Done!")

    # ---- Symlink truth/obs for easy plotting ----
    try:
        import shutil
        src = os.path.join(outdir, 'mlswe_letkf_out.nc')
        dst = os.path.join(outdir, 'mlswe_lsmcmc_out.nc')
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
        for fname in ['truth_trajectory.nc',
                      'synthetic_nlgamma_obs.nc']:
            link = os.path.join(outdir, fname)
            target = os.path.join(twin_dir, fname)
            if not os.path.exists(link) and os.path.exists(target):
                os.symlink(os.path.abspath(target), link)
    except Exception:
        pass


if __name__ == "__main__":
    main()

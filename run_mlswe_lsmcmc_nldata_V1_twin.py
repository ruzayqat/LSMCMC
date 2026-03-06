#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_mlswe_lsmcmc_nldata_V1_twin.py
===================================
Synthetic twin experiment for the nonlinear LSMCMC (V1).

1. Run a "nature run" model forward (no noise, no DA).
2. At each cycle, generate synthetic observations:
       y = arctan(H · z_truth) + N(0, σ_y²)
   using the same observation locations as the real obs file.
3. Run the NL-LSMCMC filter on the synthetic observations.
4. Compute RMSE against the nature run state (not observations).

Usage
-----
    python run_mlswe_lsmcmc_nldata_V1_twin.py [config.yml]

Default config: ``example_input_mlswe_nldata_V1_twin.yml``
"""
import os
import sys
import time
import yaml
import warnings
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
_SWE_DIR = os.path.join(os.path.dirname(__file__), '..', 'SWE_LSMCMC')
if os.path.isdir(_SWE_DIR):
    sys.path.insert(0, os.path.abspath(_SWE_DIR))

from netCDF4 import Dataset
from mlswe.model import MLSWE
from mlswe.lsmcmc_nl_V1 import NL_SMCMC_MLSWE_Filter
import run_mlswe_lsmcmc_ldata_V1          # reuse data-loading helpers


# =====================================================================
#  Load SST / SSH reference fields (copied from run_mlswe_lsmcmc_ldata_V1.main)
# =====================================================================
def _load_sst_ssh_refs(params, bc_file, lon, lat, ny, nx, obs_times):
    """Populate params with sst_nudging_ref, ssh_relax_ref, etc."""
    import glob
    from datetime import datetime
    tstart = obs_times[0]
    data_dir = params.get('data_dir', './data')

    # SST nudging reference
    sst_rate = params.get('sst_nudging_rate', 0.0)
    if sst_rate > 0:
        grid_tag = f'{ny}x{nx}'
        sst_files = sorted(glob.glob(os.path.join(
            data_dir, f'hycom_sst_ref_*_{grid_tag}_3d.npy')))
        sst_time_files = sorted(glob.glob(os.path.join(
            data_dir, f'hycom_sst_ref_*_{grid_tag}_times.npy')))
        if not sst_files:
            sst_files = sorted(glob.glob(os.path.join(
                data_dir, 'hycom_sst_ref_*_3d.npy')))
            sst_time_files = sorted(glob.glob(os.path.join(
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
            print(f"[twin] SST nudging: lam={sst_rate:.6f} s^-1, "
                  f"ref shape={sst_ref.shape}")
        else:
            print("[twin] WARNING: SST nudging enabled but no ref found")

    # SSH relaxation reference
    ssh_rate = float(params.get('ssh_relax_rate', 0.0))
    if ssh_rate > 0:
        try:
            from scipy.interpolate import RegularGridInterpolator
            with Dataset(bc_file, 'r') as nc:
                bc_lat = np.asarray(nc.variables['lat'][:])
                bc_lon = np.asarray(nc.variables['lon'][:])
                bc_times = np.asarray(nc.variables['time'][:])
                bc_ssh = np.asarray(nc.variables['ssh'][:])
            bc_ssh[np.isnan(bc_ssh)] = 0.0
            mg_lat, mg_lon = np.meshgrid(lat, lon, indexing='ij')
            ssh_ref_3d = np.zeros((len(bc_times), ny, nx))
            for ti in range(len(bc_times)):
                interp = RegularGridInterpolator(
                    (bc_lat, bc_lon), bc_ssh[ti],
                    method='linear', bounds_error=False,
                    fill_value=0.0)
                ssh_ref_3d[ti] = interp((mg_lat, mg_lon))
            params['ssh_relax_ref'] = ssh_ref_3d
            params['ssh_relax_ref_times'] = bc_times
            print(f"[twin] SSH relaxation: rate={ssh_rate:.2e}, "
                  f"ref shape={ssh_ref_3d.shape}")
        except Exception as e:
            print(f"[twin] WARNING: SSH ref load failed: {e}")

    # T_air for surface heat flux
    if params.get('sst_flux_type') is not None:
        if 'sst_nudging_ref' in params:
            params['sst_T_air'] = params['sst_nudging_ref']
            params['sst_T_air_times'] = params['sst_nudging_ref_times']


# =====================================================================
#  Nonlinear observation operator:  h(z) = arctan(z)
# =====================================================================
def obs_operator_arctan(z_local, H_loc, obs_ind_local):
    """y_pred = arctan(H_loc @ z_local)"""
    Hz = H_loc @ z_local
    return np.arctan(Hz)


# =====================================================================
#  Wrapper that injects the obs operator AND nature run state
# =====================================================================
_OrigFilter = NL_SMCMC_MLSWE_Filter


class _TwinFilter(_OrigFilter):
    """NL filter with arctan obs-operator and nature-run-state RMSE."""

    _truth_state = None          # set externally before .run()

    def __init__(self, isim, params):
        super().__init__(isim, params, obs_operator=obs_operator_arctan)


# =====================================================================
#  M-run parallel worker for twin experiment
# =====================================================================
_g_twin_params = None
_g_twin_H_b = None
_g_twin_bc_handler = None
_g_twin_obs_file = None
_g_twin_tstart = None
_g_twin_truth = None
_g_twin_ncores_per_worker = 1


def _twin_v1_worker(args):
    """Run one independent NL-LSMCMC twin simulation."""
    isim, seed = args
    params = dict(_g_twin_params)
    params['ncores'] = _g_twin_ncores_per_worker
    params['verbose'] = False
    np.random.seed(seed)

    filt = _TwinFilter(isim, params)
    filt._truth_state = _g_twin_truth
    t0 = time.time()
    filt.run(_g_twin_H_b, _g_twin_bc_handler, _g_twin_obs_file, _g_twin_tstart)
    elapsed = time.time() - t0

    print(f"  [done] twin run {isim+1}  seed={seed}  elapsed={elapsed:.1f}s  "
          f"vel={np.nanmean(filt.rmse_vel):.6f}  "
          f"sst={np.nanmean(filt.rmse_sst):.4f}  "
          f"ssh={np.nanmean(filt.rmse_ssh):.4f}", flush=True)

    return (isim, filt.lsmcmc_mean.copy(),
            filt.rmse_vel.copy(), filt.rmse_sst.copy(), filt.rmse_ssh.copy(),
            elapsed)


# =====================================================================
#  Generate nature run trajectory + synthetic arctan observations
# =====================================================================
def generate_truth_and_obs(params, H_b, bc_handler, tstart,
                           real_obs_file, outdir, rng_seed=42):
    """
    Run a single "nature run" model forward and produce synthetic obs.

    Returns
    -------
    truth_states : (nassim+1, dimx) — nature run state at each cycle
    synth_obs_file : str — path to the written NetCDF
    """
    nassim = int(params['nassim'])
    t_freq = int(params.get('t_freq',
                             params.get('assim_timesteps', 48)))

    # --- Load real obs to get observation locations & structure --------
    nc_real = Dataset(real_obs_file, 'r')
    real_yobs = np.asarray(nc_real.variables['yobs_all'][:])
    real_yind = np.asarray(nc_real.variables['yobs_ind_all'][:])
    obs_times = np.asarray(nc_real.variables['obs_times'][:])
    if 'sig_y_all' in nc_real.variables:
        real_sigy = np.asarray(nc_real.variables['sig_y_all'][:])
    else:
        real_sigy = None
    sig_y_scalar = float(nc_real.sig_y) if hasattr(nc_real, 'sig_y') else 0.1
    nc_real.close()

    n_cycles = min(nassim, real_yobs.shape[0])
    max_nobs_drifter = real_yobs.shape[1]

    # --- SSH observation locations from linear-case merged obs --------
    # The linear case already merged drifter + SWOT (real + synthetic
    # swaths).  We reuse those SSH observation *indices* per cycle and
    # generate arctan synthetic obs at those locations.
    nx = int(params['dgx'])
    ny = int(params['dgy'])
    nc = nx * ny   # cells per field (5600)

    linear_merged = params.get(
        'linear_merged_obs',
        './output_lsmcmc_ldata_V1/mlswe_merged_obs.nc')
    nc_lin = Dataset(linear_merged, 'r')
    lin_yind = np.asarray(nc_lin.variables['yobs_ind_all'][:])
    nc_lin.close()

    # Extract SSH-only indices (idx < nc) per cycle
    n_lin_cycles = min(n_cycles, lin_yind.shape[0])
    ssh_ind_per_cycle = []
    for c in range(n_lin_cycles):
        inds_c = lin_yind[c]
        valid_c = inds_c[(inds_c >= 0) & (inds_c < nc)]
        ssh_ind_per_cycle.append(valid_c.astype(int))
    # Pad remaining cycles (if any) by reusing last available
    while len(ssh_ind_per_cycle) < n_cycles:
        ssh_ind_per_cycle.append(ssh_ind_per_cycle[-1].copy())

    max_ssh_obs = max(len(s) for s in ssh_ind_per_cycle)
    max_nobs = max_nobs_drifter + max_ssh_obs
    print(f"[twin] SSH obs from linear merged file: {linear_merged}")
    print(f"[twin] SSH obs per cycle: {min(len(s) for s in ssh_ind_per_cycle)}"
          f"–{max_ssh_obs}  (drifter uv/sst max={max_nobs_drifter}, "
          f"combined max_nobs={max_nobs})")

    if 'ic_h0' in params and params['ic_h0'] is not None:
        h0 = [np.array(hk, dtype=np.float64) for hk in params['ic_h0']]
        u0 = [np.array(uk, dtype=np.float64) for uk in params['ic_u0']]
        v0 = [np.array(vk, dtype=np.float64) for vk in params['ic_v0']]
        T0 = [np.array(Tk, dtype=np.float64) for Tk in params['ic_T0']]
    else:
        raise RuntimeError("ICs must be set before calling "
                           "generate_truth_and_obs")

    lon = np.linspace(params['lon_min'], params['lon_max'], nx)
    lat = np.linspace(params['lat_min'], params['lat_max'], ny)
    from mlswe.model import coriolis_array
    lat_centre = 0.5 * (params['lat_min'] + params['lat_max'])
    dx_m = np.deg2rad(abs(lon[1] - lon[0])) * 6.371e6 * np.cos(
        np.deg2rad(lat_centre))
    dy_m = np.deg2rad(abs(lat[1] - lat[0])) * 6.371e6
    f_2d = coriolis_array(params['lat_min'], params['lat_max'], ny, nx)

    model_kw = dict(
        rho=[float(r) for r in params.get('rho', [1023, 1026, 1028])],
        dx=dx_m, dy=dy_m,
        dt=float(params['dt']),
        f0=f_2d,
        g=params.get('g', 9.81),
        H_b=H_b,
        H_mean=float(params.get('H_mean', 4000.0)),
        H_rest=[float(x) for x in params.get('H_rest',
                                               [100, 400, 3500])],
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
        ssh_relax_interior_floor=float(
            params.get('ssh_relax_interior_floor', 0.1)),
        shallow_drag_depth=float(
            params.get('shallow_drag_depth', 500.0)),
        shallow_drag_coeff=float(
            params.get('shallow_drag_coeff', 5.0e-4)),
    )

    truth_mdl = MLSWE(
        [hk.copy() for hk in h0],
        [uk.copy() for uk in u0],
        [vk.copy() for vk in v0],
        T0=[Tk.copy() for Tk in T0],
        **model_kw)
    truth_mdl.timesteps = 1

    dimx = truth_mdl.state_flat.size
    truth_states = np.zeros((n_cycles + 1, dimx))
    truth_states[0] = truth_mdl.state_flat.copy()

    # --- Forward-run nature run & build synthetic obs ----------------------
    rng = np.random.default_rng(rng_seed)
    synth_yobs = np.full((n_cycles, max_nobs), np.nan)
    synth_yind = np.full((n_cycles, max_nobs), -1, dtype=np.int32)
    synth_sigy = np.full((n_cycles, max_nobs), np.nan)

    # Per-variable sig_y from config
    sig_y_uv  = float(params.get('sig_y_uv',  0.10))
    sig_y_sst = float(params.get('sig_y_sst', 0.40))
    sig_y_ssh = float(params.get('sig_y_ssh', 0.50))

    print(f"[twin] Generating nature run trajectory ({n_cycles} cycles, "
          f"t_freq={t_freq}) ...")
    t0_gen = time.time()

    for cycle in range(n_cycles):
        # Advance nature run model (NO noise)
        for _ in range(t_freq):
            truth_mdl._timestep()
            if not np.all(np.isfinite(truth_mdl.state_flat)):
                warnings.warn(f"[twin] Nature run model blew up at cycle "
                              f"{cycle+1}")
                truth_mdl.state_flat = truth_states[cycle].copy()
                break
        truth_states[cycle + 1] = truth_mdl.state_flat.copy()

        # --- Drifter obs (UV + SST only; no SSH in drifter file) ------
        ind_raw = real_yind[cycle]
        valid = ind_raw >= 0
        drifter_ind = ind_raw[valid].astype(int) if valid.any() else np.array([], dtype=int)

        # --- SSH obs from linear merged file --------------------------
        ssh_ind = ssh_ind_per_cycle[cycle]

        # Combine drifter (uv/sst) + SSH indices
        obs_ind = np.concatenate([drifter_ind, ssh_ind])
        nv = len(obs_ind)
        if nv == 0:
            continue

        # Nature run state at obs locations
        z_truth = truth_states[cycle + 1]
        z_at_obs = z_truth[obs_ind]

        # Apply nonlinear operator: y = arctan(z_truth_at_obs)
        y_clean = np.arctan(z_at_obs)

        # Per-obs noise σ (by variable type)
        sig_obs = np.empty(nv)
        for i_obs in range(nv):
            idx = obs_ind[i_obs]
            if idx < nc:          # SSH (h_total)
                sig_obs[i_obs] = sig_y_ssh
            elif idx < 3 * nc:    # u or v
                sig_obs[i_obs] = sig_y_uv
            elif idx < 4 * nc:    # SST
                sig_obs[i_obs] = sig_y_sst
            else:
                sig_obs[i_obs] = sig_y_scalar

        # Add Gaussian noise
        noise = rng.normal(scale=sig_obs)
        y_noisy = y_clean + noise

        # Store
        synth_yobs[cycle, :nv] = y_noisy
        synth_yind[cycle, :nv] = obs_ind
        synth_sigy[cycle, :nv] = sig_obs

        if (cycle + 1) % 50 == 0 or cycle == 0:
            n_ssh_c = len(ssh_ind)
            n_drft_c = len(drifter_ind)
            _ssh_truth = z_truth[:nc] - H_b.ravel()
            print(f"  [twin] cycle {cycle+1}/{n_cycles}  "
                  f"nobs={nv} (drifter={n_drft_c}, ssh={n_ssh_c})  "
                  f"y_clean=[{y_clean.min():.4f},{y_clean.max():.4f}]  "
                  f"SSH_nature_run=[{_ssh_truth.min():.2f},"
                  f"{_ssh_truth.max():.2f}]")

    elapsed_gen = time.time() - t0_gen
    print(f"[twin] Nature run trajectory generated in {elapsed_gen:.1f}s")

    # --- Write synthetic obs NetCDF -----------------------------------
    os.makedirs(outdir, exist_ok=True)
    synth_obs_file = os.path.join(outdir, 'synthetic_arctan_obs.nc')
    ds = Dataset(synth_obs_file, 'w', format='NETCDF4')
    ds.createDimension('n_cycles', n_cycles)
    ds.createDimension('max_nobs', max_nobs)

    vy = ds.createVariable('yobs_all', 'f8', ('n_cycles', 'max_nobs'))
    vy[:] = synth_yobs
    vi = ds.createVariable('yobs_ind_all', 'i4', ('n_cycles', 'max_nobs'))
    vi[:] = synth_yind
    # The filter also expects yobs_ind_level0_all (identity for layer0)
    vi0 = ds.createVariable('yobs_ind_level0_all', 'i4',
                            ('n_cycles', 'max_nobs'))
    vi0[:] = synth_yind
    vs = ds.createVariable('sig_y_all', 'f8', ('n_cycles', 'max_nobs'))
    vs[:] = synth_sigy
    vt = ds.createVariable('obs_times', 'f8', ('n_cycles',))
    vt[:] = obs_times[:n_cycles]
    ds.sig_y = float(sig_y_scalar)
    ds.obs_operator = 'arctan'
    ds.close()
    print(f"[twin] Wrote synthetic obs: {synth_obs_file}  "
          f"({n_cycles} cycles, max_nobs={max_nobs})")

    return truth_states, synth_obs_file


# =====================================================================
#  Main
# =====================================================================
def main():
    config_file = (sys.argv[1] if len(sys.argv) > 1
                   else 'example_input_mlswe_nldata_V1_twin.yml')
    with open(config_file) as f:
        params = yaml.safe_load(f)

    print("=" * 60)
    print("  Synthetic Twin Experiment — NL-LSMCMC (arctan)")
    print("=" * 60)

    nx = params['dgx']
    ny = params['dgy']
    lon = np.linspace(params['lon_min'], params['lon_max'], nx)
    lat = np.linspace(params['lat_min'], params['lat_max'], ny)

    # Bathymetry
    H_b = run_mlswe_lsmcmc_ldata_V1.load_bathymetry(params, ny, nx, lon, lat)

    # Boundary conditions
    bc_file = params.get('bc_file', './data/hycom_bc.nc')
    if not os.path.exists(bc_file):
        bc_file = os.path.join(_SWE_DIR, 'data', 'hycom_bc.nc')
    from mlswe.boundary_handler import MLBoundaryHandler
    bc_handler = MLBoundaryHandler(
        nc_path=bc_file,
        model_lon=lon, model_lat=lat, H_b=H_b,
        H_mean=params.get('H_mean', 4000.0),
        H_rest=params.get('H_rest', [100.0, 400.0, 3500.0]),
        T_rest=params.get('T_rest', [298.15, 283.15, 275.15]),
        alpha_h=params.get('alpha_h', [0.6, 0.3, 0.1]),
        beta_vel=params.get('beta_vel', [1.0, 1.0, 1.0]),
        n_ghost=params.get('bc_n_ghost', 2),
        sponge_width=params.get('sponge_width', 8),
        sponge_timescale=params.get('sponge_timescale', 3600.0),
    )

    # Real obs file (for observation locations)
    obs_file = params.get('obs_file',
                          './data/obs_2024aug/swe_drifter_obs.nc')
    print(f"Real obs file (for locations): {obs_file}")

    # Start time
    with Dataset(obs_file, 'r') as nc:
        obs_times = np.asarray(nc.variables['obs_times'][:])
    tstart = obs_times[0]

    # Load SST / SSH references (inline, same as run_mlswe_lsmcmc_ldata_V1.main)
    _load_sst_ssh_refs(params, bc_file, lon, lat, ny, nx, obs_times)

    # HYCOM IC
    lat_centre = 0.5 * (params['lat_min'] + params['lat_max'])
    dx_m = np.deg2rad(abs(lon[1] - lon[0])) * 6.371e6 * np.cos(
        np.deg2rad(lat_centre))
    dy_m = np.deg2rad(abs(lat[1] - lat[0])) * 6.371e6
    h0, u0, v0, T0 = run_mlswe_lsmcmc_ldata_V1.init_from_bc_handler(
        bc_handler, H_b, tstart,
        H_rest=params.get('H_rest', [100.0, 400.0, 3500.0]),
        T_rest=params.get('T_rest', [298.15, 283.15, 275.15]),
        beta_vel=params.get('beta_vel', [1.0, 0.3, 0.05]),
        dx=dx_m, dy=dy_m,
        geostrophic_blend=params.get('geostrophic_blend', 0.5),
    )
    params['ic_h0'] = h0
    params['ic_u0'] = u0
    params['ic_v0'] = v0
    params['ic_T0'] = T0

    # Release HYCOM memory
    if bc_handler is not None:
        bc_handler.release_full_fields()

    outdir = params.get('lsmcmc_dir', './output_lsmcmc_nldata_V1')
    os.makedirs(outdir, exist_ok=True)

    # ---- Generate nature run + synthetic obs ----
    # Synthetic obs already include SSH locations (taken from the
    # linear merged obs file), so no SWOT merge needed here.
    truth_states, synth_obs_file = generate_truth_and_obs(
        params, H_b, bc_handler, tstart,
        obs_file, outdir, rng_seed=42)

    # Also save nature run trajectory
    _save_truth(outdir, truth_states, H_b, ny, nx)

    # ---- Check for --truth-only mode ----
    truth_only = '--truth-only' in sys.argv
    if truth_only:
        print(f"\n{'='*60}")
        print(f"  Nature-run-only run complete.")
        print(f"  Nature run trajectory: {outdir}/truth_trajectory.nc")
        print(f"  Synthetic obs:    {synth_obs_file}")
        print(f"{'='*60}")
        return

    # ---- Run the NL-LSMCMC filter on synthetic obs ----
    # Inject nature run state for RMSE computation
    _TwinFilter._truth_state = truth_states

    # Monkey-patch into the main runner
    run_mlswe_lsmcmc_ldata_V1.Loc_SMCMC_MLSWE_Filter = _TwinFilter

    # Override obs_file in params to use the synthetic one
    params['obs_file'] = synth_obs_file
    # Don't re-merge SWOT; synthetic obs already generated
    params['use_swot_ssh'] = False

    print(f"\n{'='*60}")
    print(f"  Running NL-LSMCMC filter on synthetic arctan observations")
    print(f"{'='*60}")

    nassim = int(params['nassim'])
    dimx = 12 * ny * nx
    M = int(params.get('M', 1))
    ncores = int(params.get('ncores', 1))
    workers = int(params.get('workers', ncores))

    if M <= 1:
        # ---- Single run (backward-compatible) ----
        filt = _TwinFilter(0, params)
        filt._truth_state = truth_states

        t_wall = time.time()
        filt.run(H_b, bc_handler, synth_obs_file, tstart)
        elapsed = time.time() - t_wall

        filt.save_results(outdir, obs_times=obs_times, H_b=H_b)

        print(f"\n{'='*60}")
        print(f"  Synthetic twin experiment complete in "
              f"{elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Mean vel RMSE:  {np.nanmean(filt.rmse_vel):.6f} m/s")
        print(f"  Mean SST RMSE:  {np.nanmean(filt.rmse_sst):.4f} K")
        print(f"  Mean SSH RMSE:  {np.nanmean(filt.rmse_ssh):.4f} m")
        print(f"  Output: {outdir}/")
        print(f"{'='*60}")

    else:
        # ---- M independent runs averaged ----
        global _g_twin_params, _g_twin_H_b, _g_twin_bc_handler
        global _g_twin_obs_file, _g_twin_tstart, _g_twin_truth
        global _g_twin_ncores_per_worker
        _g_twin_params = params
        _g_twin_H_b = H_b
        _g_twin_bc_handler = bc_handler
        _g_twin_obs_file = synth_obs_file
        _g_twin_tstart = tstart
        _g_twin_truth = truth_states

        n_workers = min(workers, M)
        ncores_per_worker = max(1, ncores // n_workers)
        _g_twin_ncores_per_worker = ncores_per_worker

        print(f"\n[V1-twin] Launching M={M} independent runs on {n_workers} workers")
        print(f"[V1-twin] ncores_per_worker={ncores_per_worker}")

        avg_mean = np.zeros((nassim + 1, dimx), dtype=np.float64)
        avg_rmse_vel = np.zeros(nassim, dtype=np.float64)
        avg_rmse_sst = np.zeros(nassim, dtype=np.float64)
        avg_rmse_ssh = np.zeros(nassim, dtype=np.float64)
        all_elapsed = []
        count = 0

        worker_args = [(i, 1000 + i) for i in range(M)]

        t_wall = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_twin_v1_worker, wa) for wa in worker_args]
            for fut in futures:
                isim, mean_arr, rv, rs, rh, elapsed = fut.result()
                count += 1
                avg_mean += (mean_arr - avg_mean) / count
                avg_rmse_vel += (rv - avg_rmse_vel) / count
                avg_rmse_sst += (rs - avg_rmse_sst) / count
                avg_rmse_ssh += (rh - avg_rmse_ssh) / count
                all_elapsed.append(elapsed)

        total_elapsed = time.time() - t_wall

        # Save averaged results
        filt = _TwinFilter.__new__(_TwinFilter)
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
        print(f"  NL-twin V1 complete: M={M} runs averaged")
        print(f"  Wall time:      {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"  Mean run time:  {np.mean(all_elapsed):.1f}s")
        print(f"  Mean vel RMSE:  {np.nanmean(avg_rmse_vel):.6f} m/s")
        print(f"  Mean SST RMSE:  {np.nanmean(avg_rmse_sst):.4f} K")
        print(f"  Mean SSH RMSE:  {np.nanmean(avg_rmse_ssh):.4f} m")
        print(f"  Output: {outdir}/")
        print(f"{'='*60}")

    # ---- Auto-generate diagnostic plots ----
    try:
        from plot_nl_twin_results import main as plot_nl_results
        print("\n[runner] Generating NL twin diagnostic plots ...")
        plot_nl_results(
            outdir=outdir,
            save_prefix='mlswe_results',
            config_file=config_file,
            method_label='NL-LSMCMC',
        )
    except Exception as e:
        print(f"[runner] WARNING: Plotting failed: {e}")
        import traceback; traceback.print_exc()


def _save_truth(outdir, truth_states, H_b, ny, nx):
    """Save nature run trajectory to NetCDF for later plotting."""
    outfile = os.path.join(outdir, 'truth_trajectory.nc')
    ds = Dataset(outfile, 'w', format='NETCDF4')
    nt, dimx = truth_states.shape
    nlayers = 3
    fields_per_layer = 4
    ds.createDimension('time', nt)
    ds.createDimension('layer', nlayers)
    ds.createDimension('field', fields_per_layer)
    ds.createDimension('y', ny)
    ds.createDimension('x', nx)
    v = ds.createVariable('truth', 'f4',
                          ('time', 'layer', 'field', 'y', 'x'),
                          zlib=True)
    reshaped = truth_states.reshape(nt, nlayers, fields_per_layer, ny, nx)
    v[:] = reshaped.astype(np.float32)
    if H_b is not None:
        vb = ds.createVariable('H_b', 'f4', ('y', 'x'), zlib=True)
        vb[:] = H_b.astype(np.float32)
    ds.close()
    print(f"[twin] Saved nature run trajectory: {outfile}")



if __name__ == '__main__':
    main()

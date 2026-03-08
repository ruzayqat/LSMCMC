#!/usr/bin/env python
"""
plot_mlswe_results.py
======================
Analyse and plot results from the MLSWE Local SMCMC filter with real drifter obs.

Follows the same plotting logic as SWE_LSMCMC/plot_swe_results.py, adapted for
the 3-layer MLSWE state vector layout:
    [h0,u0,v0,T0,  h1,u1,v1,T1,  h2,u2,v2,T2]
    each block (ny * nx).  dimx = 12 * ny * nx.

Surface-only obs map to layer-0 fields (first 4 * ncells entries).

Plots:
    1. Full-grid SSH map (sum(hk) - H_b) at first and last assimilation cycle.
    2. Full-grid Eastward velocity (u0) map.
    3. Full-grid Northward velocity (v0) map.
    4. Full-grid SST map (T0).
    5. Drifter coverage map + locations.
    6. RMSE time series (velocity + SST, dual-axis, HYCOM baseline).
    6b. RMSE vs HYCOM reanalysis at obs locations.
    7. Analysis vs observed velocity time series at selected cells.
    8. SST time series at selected cells.
    4a-4d. Analysis vs HYCOM Reanalysis comparison (2x3 panels per variable).

RMSE is computed against the drifter observations:
    RMSE_t = sqrt( (1/d_y) || H_t * z^a_t - y_t ||^2 )
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset


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


# ================================================================
#  Helper functions (matching SWE_LSMCMC exactly)
# ================================================================

def load_analysis(nc_path):
    """Load MLSWE LSMCMC analysis output.

    Returns (lsmcmc_mean_flat, rmse_vel, rmse_sst, rmse_ssh, obs_times, nlayers, ny, nx, H_b_nc).
    lsmcmc_mean_flat has shape (nassim+1, dimx) where
    state is flattened as [h0,u0,v0,T0, h1,u1,v1,T1, h2,u2,v2,T2].
    """
    with Dataset(nc_path, 'r') as nc:
        raw = np.asarray(nc.variables['lsmcmc_mean'][:])     # (time, 3, 4, ny, nx)
        rmse_vel = np.asarray(nc.variables['rmse_vel'][:])
        rmse_sst = np.asarray(nc.variables['rmse_sst'][:])
        if 'rmse_ssh' in nc.variables:
            rmse_ssh = np.asarray(nc.variables['rmse_ssh'][:])
        else:
            rmse_ssh = None
        obs_times = np.asarray(nc.variables['obs_times'][:])
        nlayers = int(nc.nlayers)
        # Handle both LSMCMC (has ny/nx attributes) and LETKF (has y/x dimensions)
        ny = int(nc.ny) if hasattr(nc, 'ny') else len(nc.dimensions['y'])
        nx = int(nc.nx) if hasattr(nc, 'nx') else len(nc.dimensions['x'])
        # Read H_b if saved (ensures plotter uses exact model bathymetry)
        H_b_nc = None
        if 'H_b' in nc.variables:
            H_b_nc = np.asarray(nc.variables['H_b'][:])
    # Flatten to (time, dimx) in the order [h0,u0,v0,T0, h1,u1,v1,T1, h2,u2,v2,T2]
    nt = raw.shape[0]
    nfields = nlayers * 4
    dimx = nfields * ny * nx
    flat = raw.reshape(nt, dimx)
    return flat, rmse_vel, rmse_sst, rmse_ssh, obs_times, nlayers, ny, nx, H_b_nc


def load_obs(nc_path):
    with Dataset(nc_path, 'r') as nc:
        yobs  = np.asarray(nc.variables['yobs_all'][:])
        yind  = np.asarray(nc.variables['yobs_ind_all'][:])
        yind0 = np.asarray(nc.variables['yobs_ind_level0_all'][:])
        times = np.asarray(nc.variables['obs_times'][:])
        sig_y = float(nc.sig_y) if hasattr(nc, 'sig_y') else 0.01
        sig_y_arr = None
        if 'sig_y_all' in nc.variables:
            sig_y_arr = np.asarray(nc.variables['sig_y_all'][:])
    return yobs, yind, yind0, times, sig_y, sig_y_arr


def compute_rmse_vs_obs(lsmcmc_mean, yobs, yind, nassim, dimx, ncells):
    """Compute per-cycle RMSE of analysis vs drifter observations.

    Obs indices are in single-layer layout [h,u,v,T] x ncells which is
    identical to layer-0 in the multi-layer state vector.
    """
    rmse_total = np.full(nassim, np.nan)
    rmse_vel   = np.full(nassim, np.nan)
    rmse_sst   = np.full(nassim, np.nan)
    rmse_ssh   = np.full(nassim, np.nan)

    for t in range(nassim):
        y   = yobs[t]
        ind = yind[t]
        # Obs indices reference single-layer [h,u,v,T] which maps to
        # layer-0 of multi-layer state (identity mapping for surface obs).
        valid = (ind >= 0) & (ind < 4 * ncells) & np.isfinite(y)
        if valid.sum() == 0:
            continue
        y_valid   = y[valid]
        ind_valid = ind[valid].astype(int)

        z_a = lsmcmc_mean[t + 1]
        H_z = z_a[ind_valid]
        residuals = H_z - y_valid
        rmse_total[t] = np.sqrt(np.mean(residuals**2))

        ssh_mask = (ind_valid >= 0) & (ind_valid < ncells)
        vel_mask = (ind_valid >= ncells) & (ind_valid < 3 * ncells)
        sst_mask = (ind_valid >= 3 * ncells)

        if ssh_mask.sum() > 0:
            rmse_ssh[t] = np.sqrt(np.mean(residuals[ssh_mask]**2))
        if vel_mask.sum() > 0:
            rmse_vel[t] = np.sqrt(np.mean(residuals[vel_mask]**2))
        if sst_mask.sum() > 0:
            rmse_sst[t] = np.sqrt(np.mean(residuals[sst_mask]**2))

    return {'total': rmse_total, 'velocity': rmse_vel, 'sst': rmse_sst,
            'ssh': rmse_ssh}


def load_hycom_reanalysis(bc_path, model_lon, model_lat, model_time_sec):
    """Load HYCOM reanalysis and interpolate to model grid at a given time."""
    with Dataset(bc_path, 'r') as nc:
        bc_lat   = np.asarray(nc.variables['lat'][:], dtype=np.float64)
        bc_lon   = np.asarray(nc.variables['lon'][:], dtype=np.float64)
        bc_times = np.asarray(nc.variables['time'][:], dtype=np.float64)
        bc_ssh   = np.asarray(nc.variables['ssh'][:], dtype=np.float64)
        bc_uo    = np.asarray(nc.variables['uo'][:],  dtype=np.float64)
        bc_vo    = np.asarray(nc.variables['vo'][:],  dtype=np.float64)
        bc_sst   = np.asarray(nc.variables['sst'][:], dtype=np.float64)

    # NaN fill with nearest-neighbor (not zero!)
    for arr3d in [bc_ssh, bc_uo, bc_vo, bc_sst]:
        for t in range(arr3d.shape[0]):
            _fill_nan_nearest(arr3d[t])
    if np.nanmean(bc_sst) < 100.0:
        bc_sst += 273.15

    t_idx = np.interp(model_time_sec, bc_times, np.arange(len(bc_times)))
    t_lo  = int(np.floor(t_idx))
    t_hi  = min(t_lo + 1, len(bc_times) - 1)
    alpha = t_idx - t_lo

    result = {}
    mg_lat, mg_lon = np.meshgrid(model_lat, model_lon, indexing='ij')
    for name, field3d in [('ssh', bc_ssh), ('u', bc_uo),
                          ('v', bc_vo), ('sst', bc_sst)]:
        snap = ((1 - alpha) * field3d[t_lo] + alpha * field3d[t_hi]
                if t_lo != t_hi else field3d[t_lo])
        interp = RegularGridInterpolator(
            (bc_lat, bc_lon), snap,
            method='linear', bounds_error=False, fill_value=np.nan)
        result[name] = interp((mg_lat, mg_lon))
    return result


def compute_hycom_vs_obs(bc_path, model_lon, model_lat, obs_times,
                         yind, yobs, nassim, dimx, ncells):
    """RMSE: HYCOM reanalysis vs drifter observations (baseline)."""
    with Dataset(bc_path, 'r') as nc:
        bc_lat   = np.asarray(nc.variables['lat'][:], dtype=np.float64)
        bc_lon   = np.asarray(nc.variables['lon'][:], dtype=np.float64)
        bc_times = np.asarray(nc.variables['time'][:], dtype=np.float64)
        bc_ssh   = np.asarray(nc.variables['ssh'][:], dtype=np.float64)
        bc_uo    = np.asarray(nc.variables['uo'][:],  dtype=np.float64)
        bc_vo    = np.asarray(nc.variables['vo'][:],  dtype=np.float64)
        bc_sst   = np.asarray(nc.variables['sst'][:], dtype=np.float64)
    for arr3d in [bc_ssh, bc_uo, bc_vo, bc_sst]:
        for t in range(arr3d.shape[0]):
            _fill_nan_nearest(arr3d[t])
    if np.nanmean(bc_sst) < 100.0:
        bc_sst += 273.15

    mg_lat, mg_lon = np.meshgrid(model_lat, model_lon, indexing='ij')
    rmse_vel_h = np.full(nassim, np.nan)
    rmse_sst_h = np.full(nassim, np.nan)

    for t in range(nassim):
        ind = yind[t]; y = yobs[t]
        valid = (ind >= 0) & (ind < 4 * ncells) & np.isfinite(y)
        if valid.sum() == 0:
            continue
        ind_valid = ind[valid].astype(int)
        y_valid   = y[valid]

        t_unix = obs_times[t]
        t_idx = np.interp(t_unix, bc_times, np.arange(len(bc_times)))
        t_lo = int(np.floor(t_idx))
        t_hi = min(t_lo + 1, len(bc_times) - 1)
        alpha = t_idx - t_lo

        hycom_fields = []
        for field3d in [bc_ssh, bc_uo, bc_vo, bc_sst]:
            snap = ((1 - alpha) * field3d[t_lo] + alpha * field3d[t_hi]
                    if t_lo != t_hi else field3d[t_lo])
            interp = RegularGridInterpolator(
                (bc_lat, bc_lon), snap,
                method='linear', bounds_error=False, fill_value=np.nan)
            hycom_fields.append(interp((mg_lat, mg_lon)).ravel())
        hycom_state = np.concatenate(hycom_fields)

        # Skip obs where HYCOM is NaN (outside BC domain)
        hycom_valid = np.isfinite(hycom_state[ind_valid])
        if hycom_valid.sum() == 0:
            continue
        residuals = np.full(len(ind_valid), np.nan)
        residuals[hycom_valid] = hycom_state[ind_valid[hycom_valid]] - y_valid[hycom_valid]
        vel_mask = (ind_valid >= ncells) & (ind_valid < 3 * ncells) & hycom_valid
        sst_mask = (ind_valid >= 3 * ncells) & hycom_valid
        if vel_mask.sum() > 0:
            rmse_vel_h[t] = np.sqrt(np.mean(residuals[vel_mask]**2))
        if sst_mask.sum() > 0:
            rmse_sst_h[t] = np.sqrt(np.mean(residuals[sst_mask]**2))

    return {'velocity': rmse_vel_h, 'sst': rmse_sst_h}


def compute_rmse_vs_hycom_at_obs(bc_path, model_lon, model_lat, obs_times,
                                  lsmcmc_mean, yind, yobs, nassim, dimx, ncells):
    """RMSE: Analysis vs HYCOM reanalysis at observation locations."""
    with Dataset(bc_path, 'r') as nc:
        bc_lat   = np.asarray(nc.variables['lat'][:], dtype=np.float64)
        bc_lon   = np.asarray(nc.variables['lon'][:], dtype=np.float64)
        bc_times = np.asarray(nc.variables['time'][:], dtype=np.float64)
        bc_ssh   = np.asarray(nc.variables['ssh'][:], dtype=np.float64)
        bc_uo    = np.asarray(nc.variables['uo'][:],  dtype=np.float64)
        bc_vo    = np.asarray(nc.variables['vo'][:],  dtype=np.float64)
        bc_sst   = np.asarray(nc.variables['sst'][:], dtype=np.float64)
    for arr3d in [bc_ssh, bc_uo, bc_vo, bc_sst]:
        for t in range(arr3d.shape[0]):
            _fill_nan_nearest(arr3d[t])
    if np.nanmean(bc_sst) < 100.0:
        bc_sst += 273.15

    mg_lat, mg_lon = np.meshgrid(model_lat, model_lon, indexing='ij')
    rmse_vel_hycom = np.full(nassim, np.nan)
    rmse_sst_hycom = np.full(nassim, np.nan)

    for t in range(nassim):
        ind = yind[t]; y = yobs[t]
        valid = (ind >= 0) & (ind < 4 * ncells) & np.isfinite(y)
        if valid.sum() == 0:
            continue
        ind_valid = ind[valid].astype(int)

        z_a = lsmcmc_mean[t + 1]

        t_unix = obs_times[t]
        t_idx = np.interp(t_unix, bc_times, np.arange(len(bc_times)))
        t_lo = int(np.floor(t_idx))
        t_hi = min(t_lo + 1, len(bc_times) - 1)
        alpha = t_idx - t_lo

        hycom_fields = []
        for field3d in [bc_ssh, bc_uo, bc_vo, bc_sst]:
            snap = ((1 - alpha) * field3d[t_lo] + alpha * field3d[t_hi]
                    if t_lo != t_hi else field3d[t_lo])
            interp = RegularGridInterpolator(
                (bc_lat, bc_lon), snap,
                method='linear', bounds_error=False, fill_value=np.nan)
            hycom_fields.append(interp((mg_lat, mg_lon)).ravel())
        hycom_state = np.concatenate(hycom_fields)

        H_z_analysis = z_a[ind_valid]
        H_z_hycom    = hycom_state[ind_valid]

        # Skip obs where HYCOM is NaN (outside BC domain)
        hycom_valid = np.isfinite(H_z_hycom)
        vel_mask = (ind_valid >= ncells) & (ind_valid < 3 * ncells) & hycom_valid
        sst_mask = (ind_valid >= 3 * ncells) & hycom_valid
        if vel_mask.sum() > 0:
            rmse_vel_hycom[t] = np.sqrt(np.mean(
                (H_z_analysis[vel_mask] - H_z_hycom[vel_mask])**2))
        if sst_mask.sum() > 0:
            rmse_sst_hycom[t] = np.sqrt(np.mean(
                (H_z_analysis[sst_mask] - H_z_hycom[sst_mask])**2))

    return {'velocity': rmse_vel_hycom, 'sst': rmse_sst_hycom}


def find_bathy(data_dir, ny=None, nx=None):
    """Search for cached bathymetry .npy, preferring one that matches (ny, nx)."""
    candidates = []
    for d in [data_dir, '.', './data',
              '../SWE_LSMCMC/data', os.path.expanduser('~/SWE_LSMCMC/data')]:
        if d and os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith('etopo_bathy_') and f.endswith('.npy'):
                    candidates.append(os.path.join(d, f))
    if not candidates:
        return None
    # Prefer file whose shape matches (ny, nx)
    for c in candidates:
        arr = np.load(c)
        if ny is not None and nx is not None and arr.shape == (ny, nx):
            return arr
    # Fall back to first candidate
    return np.load(candidates[0])


# ================================================================
#  Main
# ================================================================
def main(outdir='./output_lsmcmc', save_prefix='mlswe_results',
         config_file=None, method_label='LSMCMC'):
    # --- Try to detect output nc file ---
    nc_candidates = [
        os.path.join(outdir, 'mlswe_lsmcmc_out.nc'),
        os.path.join(outdir, 'mlswe_letkf_out.nc'),
    ]
    nc_path = None
    for c in nc_candidates:
        if os.path.exists(c):
            nc_path = c
            break
    if nc_path is None:
        print(f"No output file found in: {outdir}")
        sys.exit(1)

    data_dir = './data'
    # Try merged obs first (has SWOT+drifter), then raw drifter
    obs_candidates = [
        os.path.join(outdir, 'mlswe_merged_obs.nc'),
        os.path.join(outdir, 'swe_drifter_obs.nc'),
        './data/obs_2024aug/swe_drifter_obs.nc',
        '../SWE_LSMCMC/data/obs_2024aug/swe_drifter_obs.nc',
    ]
    obs_file = None
    for c in obs_candidates:
        if os.path.exists(c):
            obs_file = c
            break

    if obs_file is None:
        print("WARNING: Could not find observation file in any known location.")
        print("Looked in:", obs_candidates)

    # --- Load analysis ---
    lsmcmc_mean, rmse_vel_saved, rmse_sst_saved, rmse_ssh_saved, obs_times, nlayers, ny, nx, H_b_nc = \
        load_analysis(nc_path)
    nassim_plus1 = lsmcmc_mean.shape[0]
    nassim = nassim_plus1 - 1
    ncells = ny * nx
    nfields = nlayers * 4       # 12
    dimx = nfields * ncells     # 115200
    has_sst = True
    print(f"Loaded analysis: shape={lsmcmc_mean.shape}, "
          f"nlayers={nlayers}, ny={ny}, nx={nx}, nassim={nassim}")
    print(f"  nfields={nfields}, dimx={dimx}  (with SST tracer)")

    # --- Load observations ---
    has_obs = obs_file is not None and os.path.exists(obs_file)
    if has_obs:
        yobs, yind, yind0, obs_times_obs, sig_y, sig_y_arr = load_obs(obs_file)
        print(f"Loaded obs from: {obs_file}")
        print(f"  {yobs.shape[0]} cycles, sig_y = {sig_y:.6f}")
    else:
        sig_y = 0.01
        yobs = yind = yind0 = obs_times_obs = sig_y_arr = None

    # --- Bathymetry ---
    # Prefer H_b saved in output NC (exact match with model)
    if H_b_nc is not None:
        H_b = H_b_nc.astype(np.float64)
        print(f"Loaded bathymetry from output NC: shape={H_b.shape}, "
              f"max depth={H_b.max():.0f} m")
    else:
        H_b = find_bathy(data_dir, ny=ny, nx=nx)
        if H_b is not None:
            H_b = np.maximum(np.abs(H_b), 100.0)
            print(f"Loaded bathymetry from file: shape={H_b.shape}, "
                  f"max depth={H_b.max():.0f} m")

    # --- Grid ---
    # Try to read from config file first, then fall back to defaults
    lon_min, lon_max = -60.0, -20.0
    lat_min, lat_max = 10.0, 45.0
    params_cfg = {}
    if config_file is not None and os.path.exists(config_file):
        import yaml
        with open(config_file) as _cf:
            params_cfg = yaml.safe_load(_cf)
        lon_min = params_cfg.get('lon_min', lon_min)
        lon_max = params_cfg.get('lon_max', lon_max)
        lat_min = params_cfg.get('lat_min', lat_min)
        lat_max = params_cfg.get('lat_max', lat_max)
    lon_grid = np.linspace(lon_min, lon_max, nx)
    lat_grid = np.linspace(lat_min, lat_max, ny)

    # Noise std values for RMSE reference lines (new config keys)
    sig_y_vel = params_cfg.get('sig_y_uv', params_cfg.get('sig_y', sig_y if has_obs else 0.01))
    sig_y_sst_cfg = params_cfg.get('sig_y_sst', 0.4)
    sig_ssh_cfg = params_cfg.get('sig_y_ssh', params_cfg.get('sig_ssh', 0.05))

    # --- Nonlinear obs operator detection & HYCOM truth loading ---
    nl_obs = str(params_cfg.get('obs_operator', '')).lower() == 'arctan'
    hycom_truth = None
    if nl_obs and params_cfg.get('use_hycom_truth', False):
        try:
            from run_mlswe_lsmcmc_nlrealdata_V1 import build_hycom_truth_states
            bc_file = params_cfg.get('bc_file', './data/hycom_bc_2024aug.nc')
            dt_ = float(params_cfg['dt'])
            t_freq_ = int(params_cfg.get('t_freq',
                          params_cfg.get('assim_timesteps', 48)))
            t0_str_ = params_cfg.get('obs_time_start',
                                     '2024-08-01T00:00:00')
            from datetime import datetime as _dt_hy
            t0_dt_ = _dt_hy.strptime(t0_str_[:19], '%Y-%m-%dT%H:%M:%S')
            epoch_ = (t0_dt_ - _dt_hy(1970, 1, 1)).total_seconds()
            cycle_dt_ = t_freq_ * dt_
            obs_times_hy = [epoch_ + (i + 1) * cycle_dt_
                            for i in range(nassim)]
            hycom_truth = build_hycom_truth_states(
                bc_file, params_cfg, H_b, obs_times_hy)
            print(f"[plotter] Loaded HYCOM truth for nonlinear case: "
                  f"shape={hycom_truth.shape}")
        except Exception as e:
            print(f"[plotter] WARNING: Could not load HYCOM truth: {e}")
            hycom_truth = None

    # --- Field extraction (surface = layer 0) ---
    def get_fields(step):
        """Extract (h_total, u0, v0, T0) for a given time step."""
        vec = lsmcmc_mean[step]
        # Multi-layer state: h0 slot stores h_total directly.
        # Layout: [h_total,u0,v0,T0, 0,u1,v1,T1, 0,u2,v2,T2]
        h_total = vec[0*ncells:1*ncells].reshape(ny, nx)
        u0 = vec[1*ncells:2*ncells].reshape(ny, nx)
        v0 = vec[2*ncells:3*ncells].reshape(ny, nx)
        T0 = vec[3*ncells:4*ncells].reshape(ny, nx)
        return h_total, u0, v0, T0

    # --- RMSE computation (from scratch, like SWE_LSMCMC) ---
    if has_obs:
        rmse_dict = compute_rmse_vs_obs(lsmcmc_mean, yobs, yind,
                                        nassim, dimx, ncells)
        rmse_obs = rmse_dict['total']
        rmse_vel = rmse_dict['velocity']
        rmse_sst = rmse_dict['sst']
        rmse_ssh = rmse_dict['ssh']
        
        # Use saved SSH RMSE if present (since obs file often lacks SSH)
        if rmse_ssh_saved is not None and np.any(np.isfinite(rmse_ssh_saved)):
            # Only override if the computed one is all NaN or largely missing
            if not np.any(np.isfinite(rmse_ssh)):
                rmse_ssh = rmse_ssh_saved
                print(f"Using saved SSH RMSE from NetCDF (mean={np.nanmean(rmse_ssh):.4f} m)")

        print(f"\nRMSE (analysis vs drifter obs):")
        print(f"  {'Cycle':>5s}  {'Total':>10s}  {'Velocity':>10s}  {'SST(K)':>10s}  {'SSH(m)':>10s}")
        for t in range(nassim):
            total_s = f"{rmse_obs[t]:.6f}" if np.isfinite(rmse_obs[t]) else "N/A"
            vel_s   = f"{rmse_vel[t]:.6f}" if np.isfinite(rmse_vel[t]) else "N/A"
            sst_s   = f"{rmse_sst[t]:.4f}" if np.isfinite(rmse_sst[t]) else "N/A"
            ssh_s   = f"{rmse_ssh[t]:.4f}" if np.isfinite(rmse_ssh[t]) else "N/A"
            print(f"  {t+1:5d}  {total_s:>10s}  {vel_s:>10s}  {sst_s:>10s}  {ssh_s:>10s}")

        # Observation statistics
        all_vel_obs = []
        all_sst_obs = []
        for t in range(nassim):
            ind_t = yind[t]; y_t = yobs[t]
            valid = (ind_t >= 0) & (ind_t < 4 * ncells) & np.isfinite(y_t)
            vel_mask = (ind_t[valid] >= ncells) & (ind_t[valid] < 3 * ncells)
            sst_mask = (ind_t[valid] >= 3 * ncells)
            all_vel_obs.extend(y_t[valid][vel_mask])
            all_sst_obs.extend(y_t[valid][sst_mask])
        data_std     = np.std(all_vel_obs) if len(all_vel_obs) > 0 else 0.25
        data_mean    = np.mean(all_vel_obs) if len(all_vel_obs) > 0 else 0.0
        data_std_sst = np.std(all_sst_obs) if len(all_sst_obs) > 0 else 1.0

        mean_rmse_vel = np.nanmean(rmse_vel)
        mean_rmse_sst = np.nanmean(rmse_sst)
        mean_rmse_ssh = np.nanmean(rmse_ssh)

        print(f"\n  Mean velocity RMSE: {mean_rmse_vel:.6f} m/s")
        if np.isfinite(mean_rmse_sst):
            print(f"  Mean SST RMSE:      {mean_rmse_sst:.4f} K")
        if np.isfinite(mean_rmse_ssh):
            print(f"  Mean SSH RMSE:      {mean_rmse_ssh:.4f} m")
        print(f"  sig_y (vel noise):  {sig_y:.6f}")
        print(f"  Data std (vel obs): {data_std:.4f} m/s")
        print(f"  Data mean (vel):    {data_mean:.4f} m/s")
        if mean_rmse_vel < data_std:
            print(f"  OK: Vel RMSE ({mean_rmse_vel:.4f}) < data std ({data_std:.4f})")
        else:
            print(f"  !! Vel RMSE ({mean_rmse_vel:.4f}) >= data std ({data_std:.4f}): needs tuning")
    else:
        rmse_obs = rmse_vel = rmse_sst = rmse_ssh = rmse_ssh = None
        data_std = 0.25
        data_std_sst = 1.0

    # --- Drifter lon/lat helper ---
    def get_drifter_lonlat(cycle_idx):
        if not has_obs:
            return np.array([]), np.array([])
        ind1 = yind[cycle_idx]
        valid1 = (ind1 >= 0) & (ind1 < dimx)
        u_inds = ind1[valid1]
        u_obs_mask = (u_inds >= ncells) & (u_inds < 2 * ncells)
        cell_flat_u = u_inds[u_obs_mask] - ncells
        iy_u = cell_flat_u // nx
        ix_u = cell_flat_u % nx
        return lon_grid[ix_u], lat_grid[iy_u]

    # --- Bathymetry contours ---
    def add_bathy_contour(ax):
        if H_b is not None:
            ax.contour(lon_grid, lat_grid, H_b, levels=[200], colors='0.4',
                       linewidths=0.8, linestyles='-')
            ax.contour(lon_grid, lat_grid, H_b, levels=[1000, 3000, 5000],
                       colors='0.6', linewidths=0.4, linestyles='--')

    # --- Ocean mask ---
    if H_b is not None:
        ocean_mask = H_b >= 200.0
    else:
        ocean_mask = np.ones((ny, nx), dtype=bool)

    def mask_land(field):
        return np.ma.masked_where(~ocean_mask, field)

    def smooth_ocean(field, sigma=2.0):
        filled = field.copy()
        filled[~ocean_mask] = np.nanmean(field[ocean_mask]) if ocean_mask.any() else 0.0
        smoothed = gaussian_filter(filled, sigma=sigma)
        return np.ma.masked_where(~ocean_mask, smoothed)

    # --- Load HYCOM reanalysis for comparison ---
    # Auto-detect the correct BC file (prefer 2024aug, fall back to generic)
    bc_path = None
    for bc_candidate in ['hycom_bc_2024aug.nc', 'hycom_bc.nc']:
        p = os.path.join(data_dir, bc_candidate)
        if os.path.exists(p):
            bc_path = p
            print(f"Using HYCOM BC file: {bc_path}")
            break
    if bc_path is None:
        bc_path = os.path.join(data_dir, 'hycom_bc.nc')  # fallback
    hycom_rean_init = None
    hycom_rean_final = None
    if os.path.exists(bc_path) and has_obs and obs_times is not None:
        try:
            init_time_unix = obs_times[0]
            hycom_rean_init = load_hycom_reanalysis(
                bc_path, lon_grid, lat_grid, init_time_unix)
            print(f"Loaded HYCOM reanalysis (initial) at t={init_time_unix:.0f}")
        except Exception as e:
            print(f"Could not load HYCOM reanalysis (initial): {e}")
        try:
            final_time_unix = obs_times[nassim - 1] if nassim <= len(obs_times) else obs_times[-1]
            hycom_rean_final = load_hycom_reanalysis(
                bc_path, lon_grid, lat_grid, final_time_unix)
            print(f"Loaded HYCOM reanalysis (final) at t={final_time_unix:.0f}")
        except Exception as e:
            print(f"Could not load HYCOM reanalysis (final): {e}")

    # --- RMSE: Analysis vs HYCOM at obs locations ---
    rmse_hycom = None
    if has_obs and os.path.exists(bc_path):
        try:
            rmse_hycom = compute_rmse_vs_hycom_at_obs(
                bc_path, lon_grid, lat_grid, obs_times_obs,
                lsmcmc_mean, yind, yobs, nassim, dimx, ncells)
            print(f"\nRMSE (analysis vs HYCOM) at obs locations:")
            print(f"  Mean velocity: {np.nanmean(rmse_hycom['velocity']):.6f} m/s")
            print(f"  Mean SST:      {np.nanmean(rmse_hycom['sst']):.4f} K")
        except Exception as e:
            print(f"Could not compute RMSE vs HYCOM: {e}")

    # --- RMSE: HYCOM vs Observations (baseline) ---
    rmse_hycom_vs_obs = None
    if has_obs and os.path.exists(bc_path):
        try:
            rmse_hycom_vs_obs = compute_hycom_vs_obs(
                bc_path, lon_grid, lat_grid, obs_times_obs,
                yind, yobs, nassim, dimx, ncells)
            print(f"\nRMSE (HYCOM vs observations) - BASELINE:")
            print(f"  Mean velocity: {np.nanmean(rmse_hycom_vs_obs['velocity']):.6f} m/s")
            print(f"  Mean SST:      {np.nanmean(rmse_hycom_vs_obs['sst']):.4f} K")
            if rmse_vel is not None:
                vel_imp = ((np.nanmean(rmse_hycom_vs_obs['velocity']) - np.nanmean(rmse_vel))
                           / np.nanmean(rmse_hycom_vs_obs['velocity']) * 100)
                print(f"  -> Analysis improves velocity RMSE by {vel_imp:.1f}% vs HYCOM")
            if rmse_sst is not None:
                sst_imp = ((np.nanmean(rmse_hycom_vs_obs['sst']) - np.nanmean(rmse_sst))
                           / np.nanmean(rmse_hycom_vs_obs['sst']) * 100)
                print(f"  -> Analysis improves SST RMSE by {sst_imp:.1f}% vs HYCOM")
        except Exception as e:
            print(f"Could not compute HYCOM vs obs RMSE: {e}")

    # --- Create plot output dir ---
    plot_dir = os.path.join(outdir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # ================================================================
    #  Figure 1: SSH (sum hk - H_b) -- ocean only, smoothed
    # ================================================================
    steps_ssh = [0, nassim]
    labels_ssh = ['Initial', f'Cycle {nassim}']

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for col, (step, label) in enumerate(zip(steps_ssh, labels_ssh)):
        h_tot, u, v, T = get_fields(step)
        ssh = (h_tot - H_b) if H_b is not None else (h_tot - 4000.0)
        ssh_smooth = smooth_ocean(ssh, sigma=3.0)
        ax = axes1[col]
        vals = ssh_smooth.compressed()
        vmax = max(abs(np.percentile(vals, 2)),
                   abs(np.percentile(vals, 98)), 0.01)
        ax.set_facecolor('0.88')
        im = ax.pcolormesh(lon_grid, lat_grid, ssh_smooth, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        add_bathy_contour(ax)
        if step > 0:
            dlons, dlats = get_drifter_lonlat(step - 1)
            ax.scatter(dlons, dlats, s=6, c='lime', edgecolors='k',
                       linewidths=0.3, zorder=5, label='Drifters')
            ax.legend(fontsize=8, loc='lower right')
        ax.set_title(f'SSH anomaly -- {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (deg E)')
        ax.set_ylabel('Latitude (deg N)')
        plt.colorbar(im, ax=ax, label='SSH (m)', shrink=0.85)

    fig1.suptitle('Sea Surface Height Anomaly (sum hk - H_b) -- Ocean Only', fontsize=14)
    fig1_path = os.path.join(plot_dir, f'{save_prefix}_ssh.png')
    fig1.savefig(fig1_path, dpi=150)
    print(f"\nSaved: {fig1_path}")

    # ================================================================
    #  Figure 2: Eastward velocity (u0) -- ocean only
    # ================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for col, (step, label) in enumerate(zip(steps_ssh, labels_ssh)):
        h_tot, u, v, T = get_fields(step)
        u_ocean = smooth_ocean(u, sigma=2.0)
        ax = axes2[col]
        ax.set_facecolor('0.88')
        vmax = max(np.percentile(np.abs(u_ocean.compressed()), 99), 0.01)
        im = ax.pcolormesh(lon_grid, lat_grid, u_ocean, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        add_bathy_contour(ax)
        if step > 0:
            dlons, dlats = get_drifter_lonlat(step - 1)
            ax.scatter(dlons, dlats, s=6, c='lime', edgecolors='k',
                       linewidths=0.3, zorder=5, label='Drifters')
            ax.legend(fontsize=8, loc='lower right')
        ax.set_title(f'Eastward Velocity (u0) -- {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (deg E)')
        ax.set_ylabel('Latitude (deg N)')
        plt.colorbar(im, ax=ax, label='u (m/s)', shrink=0.85)

    fig2.suptitle('Eastward Velocity (u0) -- Ocean Only', fontsize=14)
    fig2_path = os.path.join(plot_dir, f'{save_prefix}_u_velocity.png')
    fig2.savefig(fig2_path, dpi=150)
    print(f"Saved: {fig2_path}")

    # ================================================================
    #  Figure 3: Northward velocity (v0) -- ocean only
    # ================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for col, (step, label) in enumerate(zip(steps_ssh, labels_ssh)):
        h_tot, u, v, T = get_fields(step)
        v_ocean = smooth_ocean(v, sigma=2.0)
        ax = axes3[col]
        ax.set_facecolor('0.88')
        vmax = max(np.percentile(np.abs(v_ocean.compressed()), 99), 0.01)
        im = ax.pcolormesh(lon_grid, lat_grid, v_ocean, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        add_bathy_contour(ax)
        if step > 0:
            dlons, dlats = get_drifter_lonlat(step - 1)
            ax.scatter(dlons, dlats, s=6, c='lime', edgecolors='k',
                       linewidths=0.3, zorder=5, label='Drifters')
            ax.legend(fontsize=8, loc='lower right')
        ax.set_title(f'Northward Velocity (v0) -- {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (deg E)')
        ax.set_ylabel('Latitude (deg N)')
        plt.colorbar(im, ax=ax, label='v (m/s)', shrink=0.85)

    fig3.suptitle('Northward Velocity (v0) -- Ocean Only', fontsize=14)
    fig3_path = os.path.join(plot_dir, f'{save_prefix}_v_velocity.png')
    fig3.savefig(fig3_path, dpi=150)
    print(f"Saved: {fig3_path}")

    # ================================================================
    #  Figure 4: SST -- ocean only (same color scale for both panels)
    # ================================================================
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    # Pre-compute SST range across both panels for consistent color scale
    sst_vals = []
    for step in steps_ssh:
        _, _, _, T = get_fields(step)
        T_c = smooth_ocean(T - 273.15, sigma=2.0)
        sst_vals.extend(T_c.compressed())
    sst_vmin = np.percentile(sst_vals, 2)
    sst_vmax = np.percentile(sst_vals, 98)

    for col, (step, label) in enumerate(zip(steps_ssh, labels_ssh)):
        h_tot, u, v, T = get_fields(step)
        ax = axes4[col]
        ax.set_facecolor('0.88')
        T_c = smooth_ocean(T - 273.15, sigma=2.0)
        im = ax.pcolormesh(lon_grid, lat_grid, T_c, cmap='RdYlBu_r',
                           vmin=sst_vmin, vmax=sst_vmax, shading='auto')
        add_bathy_contour(ax)
        if step > 0:
            dlons, dlats = get_drifter_lonlat(step - 1)
            ax.scatter(dlons, dlats, s=6, c='lime', edgecolors='k',
                       linewidths=0.3, zorder=5, label='Drifters')
            ax.legend(fontsize=8, loc='lower right')
        ax.set_title(f'SST -- {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (deg E)')
        ax.set_ylabel('Latitude (deg N)')
        plt.colorbar(im, ax=ax, label='SST (deg C)', shrink=0.85)

    fig4.suptitle('Sea Surface Temperature -- Ocean Only', fontsize=14)
    fig4_path = os.path.join(plot_dir, f'{save_prefix}_sst_maps.png')
    fig4.savefig(fig4_path, dpi=150)
    print(f"Saved: {fig4_path}")

    # ================================================================
    #  Figures 4a-4d: Analysis vs HYCOM Reanalysis comparison
    #  Each figure: 2 rows (Initial, Final) x 3 cols (Analysis, Reanalysis, Diff)
    # ================================================================
    if hycom_rean_init is not None and hycom_rean_final is not None:

        def _plot_comparison(var_key, cmap, label_unit, title_prefix,
                             analysis_func, sigma, save_name, diverging=True):
            fig, axes = plt.subplots(2, 3, figsize=(21, 10), constrained_layout=True)
            rean_data = {0: hycom_rean_init, 1: hycom_rean_final}
            step_labels = ['Initial (Cycle 0)', f'Final (Cycle {nassim})']
            steps = [0, nassim]

            for row, (step, rlabel) in enumerate(zip(steps, step_labels)):
                anal_field = analysis_func(step)
                anal_smooth = smooth_ocean(anal_field, sigma=sigma)

                if step == 0:
                    # At the initial cycle the analysis IS the HYCOM
                    # initialisation, so use the analysis itself as the
                    # reference to avoid artefacts from a second,
                    # independent interpolation pipeline.
                    rean_smooth = anal_smooth.copy()
                else:
                    rean_field = rean_data[row][var_key].copy()
                    if var_key == 'sst':
                        rean_field = rean_field - 273.15
                    # Fill NaN from HYCOM interpolation with nearest finite value
                    _fill_nan_nearest(rean_field)
                    rean_smooth = smooth_ocean(rean_field, sigma=sigma)

                # Use only analysis values for color scale (HYCOM may have
                # fill artifacts at domain edges)
                anal_vals = anal_smooth.compressed()
                anal_vals = anal_vals[np.isfinite(anal_vals)]
                rean_vals = rean_smooth.compressed()
                rean_vals = rean_vals[np.isfinite(rean_vals)]
                all_vals = np.concatenate([anal_vals, rean_vals])
                if len(all_vals) == 0:
                    continue
                if diverging:
                    vmax = max(abs(np.percentile(all_vals, 2)),
                               abs(np.percentile(all_vals, 98)), 0.01)
                    vmin = -vmax
                else:
                    vmin = np.percentile(all_vals, 2)
                    vmax = np.percentile(all_vals, 98)

                # Column 0: Analysis
                ax = axes[row, 0]
                ax.set_facecolor('0.88')
                im = ax.pcolormesh(lon_grid, lat_grid, anal_smooth,
                                   cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
                add_bathy_contour(ax)
                ax.set_title(f'Analysis -- {rlabel}', fontsize=11, fontweight='bold')
                ax.set_xlabel('Longitude (deg E)')
                ax.set_ylabel('Latitude (deg N)')
                plt.colorbar(im, ax=ax, label=label_unit, shrink=0.85)

                # Column 1: Reanalysis
                ax = axes[row, 1]
                ax.set_facecolor('0.88')
                im = ax.pcolormesh(lon_grid, lat_grid, rean_smooth,
                                   cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
                add_bathy_contour(ax)
                ax.set_title(f'HYCOM Reanalysis -- {rlabel}', fontsize=11, fontweight='bold')
                ax.set_xlabel('Longitude (deg E)')
                ax.set_ylabel('Latitude (deg N)')
                plt.colorbar(im, ax=ax, label=label_unit, shrink=0.85)

                # Column 2: Difference
                diff = anal_smooth - rean_smooth
                diff_vals = diff.compressed()
                diff_vals = diff_vals[np.isfinite(diff_vals)]
                if len(diff_vals) == 0:
                    dmax = 0.1
                else:
                    dmax = max(abs(np.percentile(diff_vals, 2)),
                               abs(np.percentile(diff_vals, 98)), 0.001)
                ax = axes[row, 2]
                ax.set_facecolor('0.88')
                im = ax.pcolormesh(lon_grid, lat_grid, diff, cmap='RdBu_r',
                                   vmin=-dmax, vmax=dmax, shading='auto')
                add_bathy_contour(ax)
                ax.set_title(f'Difference (Analysis - Reanalysis) -- {rlabel}',
                             fontsize=11, fontweight='bold')
                ax.set_xlabel('Longitude (deg E)')
                ax.set_ylabel('Latitude (deg N)')
                plt.colorbar(im, ax=ax, label=label_unit, shrink=0.85)

            fig.suptitle(f'{title_prefix}: Analysis vs HYCOM Reanalysis', fontsize=14)
            fpath = os.path.join(plot_dir, f'{save_prefix}_{save_name}.png')
            fig.savefig(fpath, dpi=150)
            print(f"Saved: {fpath}")

        # SSH comparison
        def _ssh_field(step):
            h_tot, u, v, T = get_fields(step)
            return (h_tot - H_b) if H_b is not None else (h_tot - 4000.0)

        _plot_comparison('ssh', 'RdBu_r', 'SSH (m)', 'Sea Surface Height',
                         _ssh_field, 3.0, 'compare_ssh', diverging=True)

        # U velocity comparison
        _plot_comparison('u', 'RdBu_r', 'u (m/s)', 'Eastward Velocity (u0)',
                         lambda s: get_fields(s)[1], 2.0, 'compare_u', diverging=True)

        # V velocity comparison
        _plot_comparison('v', 'RdBu_r', 'v (m/s)', 'Northward Velocity (v0)',
                         lambda s: get_fields(s)[2], 2.0, 'compare_v', diverging=True)

        # SST comparison
        def _sst_field(step):
            return get_fields(step)[3] - 273.15

        _plot_comparison('sst', 'RdYlBu_r', 'SST (deg C)', 'Sea Surface Temperature',
                         _sst_field, 2.0, 'compare_sst', diverging=False)

    # ================================================================
    #  Figure 5: Observation coverage (drifter + SWOT SSH)
    # ================================================================
    if has_obs:
        # --- Load SWOT SSH obs from raw SWOT files (once, shared by Fig 5 & 9) ---
        import glob as _glob
        from datetime import datetime as _dt_swot
        swot_dirs = ['./data/swot_2024aug_new', './data/swot_2024aug', './data/swot']
        swot_files = []
        for sd in swot_dirs:
            swot_files = sorted(_glob.glob(os.path.join(sd, 'SWOT_*.nc')))
            if swot_files:
                break

        has_ssh_obs = False
        swot_lons_all, swot_lats_all = np.array([]), np.array([])
        swot_ssh_all, swot_times_all = np.array([]), np.array([])
        swot_lons_first, swot_lats_first = [], []
        swot_lons_last, swot_lats_last = [], []

        if swot_files:
            try:
                print(f"Loading {len(swot_files)} SWOT files (shared for coverage + timeseries) ...")
                _lons_c, _lats_c, _ssh_c, _times_c = [], [], [], []
                for fi, sf in enumerate(swot_files):
                    with Dataset(sf, 'r') as nc:
                        lats = np.array(nc.variables['latitude'][:])
                        lons_raw = np.array(nc.variables['longitude'][:])
                        ssha_raw = np.array(nc.variables['ssha_karin'][:])
                        time_s = np.array(nc.variables['time'][:])
                        # Apply crossover + MSS-geoid correction → ADT
                        # ADT = ssha_karin + height_cor_xover + (MSS - geoid)
                        if 'height_cor_xover' in nc.variables:
                            xover = np.array(nc.variables['height_cor_xover'][:])
                            mss = np.array(nc.variables['mean_sea_surface_cnescls'][:])
                            geoid = np.array(nc.variables['geoid'][:])
                            ssh = ssha_raw + xover + (mss - geoid)
                        else:
                            ssh = ssha_raw
                        # time is per-line (1D), broadcast to 2D
                        if time_s.ndim == 1 and lons_raw.ndim == 2:
                            time_s = np.repeat(time_s[:, np.newaxis],
                                               lons_raw.shape[1], axis=1)
                    # Flatten
                    lats = lats.ravel()
                    lons_raw = lons_raw.ravel()
                    ssh = ssh.ravel()
                    time_s = time_s.ravel()
                    # Convert lon from 0-360 to -180-180
                    lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
                    # Filter valid SSH and domain
                    valid = (np.isfinite(ssh) & np.isfinite(lons) &
                             np.isfinite(lats) & (np.abs(ssh) < 100) &
                             (lons >= lon_min) & (lons <= lon_max) &
                             (lats >= lat_min) & (lats <= lat_max))
                    lons_v, lats_v = lons[valid], lats[valid]
                    _lons_c.append(lons_v)
                    _lats_c.append(lats_v)
                    _ssh_c.append(ssh[valid])
                    _times_c.append(time_s[valid])
                    # First and last files for swath visualization
                    if fi < 3:
                        swot_lons_first.extend(lons_v)
                        swot_lats_first.extend(lats_v)
                    if fi >= len(swot_files) - 3:
                        swot_lons_last.extend(lons_v)
                        swot_lats_last.extend(lats_v)

                swot_lons_all = np.concatenate(_lons_c)
                swot_lats_all = np.concatenate(_lats_c)
                swot_ssh_all = np.concatenate(_ssh_c)
                swot_times_all = np.concatenate(_times_c)
                del _lons_c, _lats_c, _ssh_c, _times_c

                if len(swot_lons_all) > 0:
                    has_ssh_obs = True
                    print(f"  Loaded SWOT: {len(swot_lons_all):,} valid points "
                          f"from {len(swot_files)} files")
            except Exception as e:
                print(f"Could not read SWOT files: {e}")
                import traceback; traceback.print_exc()

        # Create figure: 2 panels (drifter heatmap, observation positions)
        fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # Panel 1: Drifter coverage heatmap
        drifter_coverage = np.zeros((ny, nx), dtype=int)
        for t in range(nassim):
            ind0 = yind0[t]
            valid = (ind0 >= 0) & (ind0 < ncells)
            for idx in ind0[valid]:
                iy, ix = divmod(int(idx), nx)
                drifter_coverage[iy, ix] += 1

        ax = axes5[0]
        im = ax.pcolormesh(lon_grid, lat_grid, drifter_coverage, cmap='YlOrRd', shading='auto')
        add_bathy_contour(ax)
        ax.set_title('Drifter observation count per grid cell',
                      fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (deg E)')
        ax.set_ylabel('Latitude (deg N)')
        plt.colorbar(im, ax=ax, label='count')

        # Panel 2: Drifter + SWOT positions at first and last cycle
        ax2 = axes5[1]
        if H_b is not None:
            ax2.contourf(lon_grid, lat_grid, H_b, levels=20, cmap='Blues', alpha=0.4)
        dlons, dlats = get_drifter_lonlat(0)
        ax2.scatter(dlons, dlats, s=12, c='red', edgecolors='white',
                    linewidths=0.3, label=f'Cycle 1 ({len(dlons)} drifters)')
        if nassim > 1:
            dlons2, dlats2 = get_drifter_lonlat(nassim - 1)
            ax2.scatter(dlons2, dlats2, s=12, c='blue', edgecolors='white',
                        linewidths=0.3, alpha=0.6,
                        label=f'Cycle {nassim} ({len(dlons2)} drifters)')
        # Show SWOT swath from first and last files
        if has_ssh_obs and len(swot_lons_first) > 0:
            ax2.scatter(swot_lons_first, swot_lats_first, s=1, c='green',
                        alpha=0.15, label=f'SWOT first passes')
        if has_ssh_obs and len(swot_lons_last) > 0:
            ax2.scatter(swot_lons_last, swot_lats_last, s=1, c='orange',
                        alpha=0.15, label=f'SWOT last passes')
        ax2.set_title('Observation positions', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Longitude (deg E)')
        ax2.set_ylabel('Latitude (deg N)')
        ax2.legend(fontsize=8, loc='upper right')

        fig5_path = os.path.join(plot_dir, f'{save_prefix}_drifter_coverage.png')
        fig5.savefig(fig5_path, dpi=150)
        print(f"Saved: {fig5_path}")

    # ================================================================
    #  Figure 6: RMSE time series (UV, SSH, SST — 3 subplots)
    #  With observation noise std as dotted horizontal lines.
    #  Goal: RMSE should be below the noise lines.
    # ================================================================
    if has_obs and rmse_vel is not None:
        # Use SSH RMSE from compute_rmse_vs_obs; fall back to saved in NC
        rmse_ssh_plot = rmse_ssh
        if rmse_ssh_plot is None or not np.any(np.isfinite(rmse_ssh_plot)):
            with Dataset(nc_path, 'r') as _nc_check:
                if 'rmse_ssh' in _nc_check.variables:
                    rmse_ssh_plot = np.asarray(_nc_check.variables['rmse_ssh'][:])

        fig6, axes6 = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                    constrained_layout=True)
        cycles = np.arange(1, nassim + 1)

        # --- Panel 1: UV RMSE ---
        ax = axes6[0]
        if np.any(np.isfinite(rmse_vel)):
            ax.plot(cycles, rmse_vel, 'b-', lw=1.2, alpha=0.8,
                    label=f'{method_label} RMSE')
        if hycom_truth is None and rmse_hycom_vs_obs is not None and np.any(np.isfinite(rmse_hycom_vs_obs['velocity'])):
            ax.plot(cycles, rmse_hycom_vs_obs['velocity'], color='gray', ls='--',
                    lw=1.2, alpha=0.7, label='HYCOM baseline')
        ax.axhline(sig_y_vel, color='k', ls=':', lw=1.5,
                   label=f'$\\sigma_y^{{vel}}$ = {sig_y_vel:.4f} m/s')
        ax.set_ylabel('UV RMSE (m/s)', fontsize=11)
        ax.set_title(f'Velocity (U,V) RMSE  —  mean = {np.nanmean(rmse_vel):.5f} m/s',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # --- Panel 2: SSH RMSE ---
        ax = axes6[1]
        if rmse_ssh_plot is not None and np.any(np.isfinite(rmse_ssh_plot)):
            ax.plot(cycles, rmse_ssh_plot, 'g-', lw=1.2, alpha=0.8,
                    label=f'{method_label} RMSE')
            ax.set_title(f'SSH RMSE  —  mean = {np.nanmean(rmse_ssh_plot):.4f} m',
                         fontsize=12, fontweight='bold')
        else:
            ax.set_title('SSH RMSE  —  (not available)', fontsize=12)
        ax.axhline(sig_ssh_cfg, color='k', ls=':', lw=1.5,
                   label=f'$\\sigma_{{ssh}}$ = {sig_ssh_cfg:.2f} m')
        ax.set_ylabel('SSH RMSE (m)', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # --- Panel 3: SST RMSE ---
        ax = axes6[2]
        if np.any(np.isfinite(rmse_sst)):
            ax.plot(cycles, rmse_sst, 'r-', lw=1.2, alpha=0.8,
                    label=f'{method_label} RMSE')
        if hycom_truth is None and rmse_hycom_vs_obs is not None and np.any(np.isfinite(rmse_hycom_vs_obs['sst'])):
            ax.plot(cycles, rmse_hycom_vs_obs['sst'], color='gray', ls='--',
                    lw=1.2, alpha=0.7, label='HYCOM baseline')
        ax.axhline(sig_y_sst_cfg, color='k', ls=':', lw=1.5,
                   label=f'$\\sigma_y^{{sst}}$ = {sig_y_sst_cfg:.2f} K')
        ax.set_ylabel('SST RMSE (K)', fontsize=11)
        ax.set_xlabel('Assimilation Cycle', fontsize=11)
        ax.set_title(f'SST RMSE  —  mean = {np.nanmean(rmse_sst):.4f} K',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        _rmse_suptitle = (f'{method_label}: RMSE vs HYCOM  (dotted = obs noise $\\sigma$)'
                         if hycom_truth is not None
                         else f'{method_label}: RMSE vs Observations  (dotted = obs noise $\\sigma$)')
        fig6.suptitle(_rmse_suptitle, fontsize=14, fontweight='bold')
        fig6_path = os.path.join(plot_dir, f'{save_prefix}_rmse.png')
        fig6.savefig(fig6_path, dpi=150)
        print(f"Saved: {fig6_path}")

    # ================================================================
    #  Figure 6b: Analysis RMSE vs Data (UV, SSH if observed, SST)
    # ================================================================
    if has_obs and rmse_vel is not None:
        has_vel_data = np.any(np.isfinite(rmse_vel))
        has_ssh_data = rmse_ssh is not None and np.any(np.isfinite(rmse_ssh))
        has_sst_data = rmse_sst is not None and np.any(np.isfinite(rmse_sst))
        n_panels = int(has_vel_data) + int(has_ssh_data) + int(has_sst_data)

        if n_panels > 0:
            fig6b, axes6b = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels),
                                          sharex=True, constrained_layout=True)
            if n_panels == 1:
                axes6b = [axes6b]
            cycles = np.arange(1, nassim + 1)
            panel_idx = 0

            if has_vel_data:
                ax = axes6b[panel_idx]
                ax.plot(cycles, rmse_vel, 'b-o', markersize=2, lw=1.2,
                        label=f'Vel RMSE vs data (mean = {np.nanmean(rmse_vel):.4f} m/s)')
                ax.axhline(sig_y_vel, color='k', ls=':', lw=1.5,
                           label=f'$\\sigma_y^{{vel}}$ = {sig_y_vel:.4f}')
                ax.set_ylabel('UV RMSE (m/s)', fontsize=11)
                ax.set_title(f'Velocity RMSE vs Data  —  mean = {np.nanmean(rmse_vel):.5f} m/s',
                             fontsize=12, fontweight='bold')
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
                panel_idx += 1

            if has_ssh_data:
                ax = axes6b[panel_idx]
                ax.plot(cycles, rmse_ssh, 'g-s', markersize=2, lw=1.2,
                        label=f'SSH RMSE vs data (mean = {np.nanmean(rmse_ssh):.4f} m)')
                ax.axhline(sig_ssh_cfg, color='k', ls=':', lw=1.5,
                           label=f'$\\sigma_{{ssh}}$ = {sig_ssh_cfg:.2f} m')
                ax.set_ylabel('SSH RMSE (m)', fontsize=11)
                ax.set_title(f'SSH RMSE vs Data  —  mean = {np.nanmean(rmse_ssh):.4f} m',
                             fontsize=12, fontweight='bold')
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
                panel_idx += 1

            if has_sst_data:
                ax = axes6b[panel_idx]
                ax.plot(cycles, rmse_sst, 'r-D', markersize=2, lw=1.2,
                        label=f'SST RMSE vs data (mean = {np.nanmean(rmse_sst):.3f} K)')
                ax.axhline(sig_y_sst_cfg, color='k', ls=':', lw=1.5,
                           label=f'$\\sigma_y^{{sst}}$ = {sig_y_sst_cfg:.2f} K')
                ax.set_ylabel('SST RMSE (K)', fontsize=11)
                ax.set_title(f'SST RMSE vs Data  —  mean = {np.nanmean(rmse_sst):.4f} K',
                             fontsize=12, fontweight='bold')
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
                panel_idx += 1

            axes6b[-1].set_xlabel('Assimilation Cycle', fontsize=11)
            fig6b.suptitle(f'{method_label}: Analysis RMSE vs Data',
                           fontsize=14, fontweight='bold')

            fig6b_path = os.path.join(plot_dir, f'{save_prefix}_rmse_hycom.png')
            fig6b.savefig(fig6b_path, dpi=150)
            print(f"Saved: {fig6b_path}")

    # ================================================================
    #  Figure 7: Analysis vs obs/HYCOM velocity time series
    # ================================================================
    if has_obs:
        obs_count = np.zeros(ncells, dtype=int)
        for t in range(nassim):
            ind0t = yind0[t]
            valid = (ind0t >= 0) & (ind0t < ncells)
            for idx in ind0t[valid]:
                obs_count[int(idx)] += 1
        top_cells = np.argsort(obs_count)[-4:][::-1]

        fig7, axes7 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for panel, cell_flat in enumerate(top_cells):
            ax = axes7.flat[panel]
            iy_c, ix_c = divmod(cell_flat, nx)
            lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
            u_idx = ncells + cell_flat
            v_idx = 2 * ncells + cell_flat

            anal_u = lsmcmc_mean[:, u_idx]
            anal_v = lsmcmc_mean[:, v_idx]

            if hycom_truth is not None:
                # Nonlinear case: plot HYCOM reference instead of obs
                hycom_u = hycom_truth[:, u_idx]
                hycom_v = hycom_truth[:, v_idx]
                ax.plot(np.arange(nassim + 1), hycom_u, 'r--', alpha=0.6, lw=1.2,
                        label='HYCOM u')
                ax.plot(np.arange(nassim + 1), hycom_v, 'b--', alpha=0.6, lw=1.2,
                        label='HYCOM v')
                ax.plot(np.arange(nassim + 1), anal_u, 'r-', alpha=0.8, lw=1.5,
                        label='Analysis u')
                ax.plot(np.arange(nassim + 1), anal_v, 'b-', alpha=0.8, lw=1.5,
                        label='Analysis v')
            else:
                obs_u = np.full(nassim, np.nan)
                obs_v = np.full(nassim, np.nan)
                for t in range(nassim):
                    ind_t = yind[t]; y_t = yobs[t]
                    mu = np.where((ind_t == u_idx) & np.isfinite(y_t))[0]
                    if mu.size > 0: obs_u[t] = np.mean(y_t[mu])
                    mv = np.where((ind_t == v_idx) & np.isfinite(y_t))[0]
                    if mv.size > 0: obs_v[t] = np.mean(y_t[mv])
                ax.plot(np.arange(nassim), obs_u, 'rx', ms=7, label='Obs u (east)')
                ax.plot(np.arange(nassim), obs_v, 'bx', ms=7, label='Obs v (north)')
                ax.plot(np.arange(nassim + 1), anal_u, 'r-', alpha=0.8, lw=1.5,
                        label='Analysis u')
                ax.plot(np.arange(nassim + 1), anal_v, 'b-', alpha=0.8, lw=1.5,
                        label='Analysis v')
            ax.set_title(f'({lon_c:.1f} deg E, {lat_c:.1f} deg N) -- {obs_count[cell_flat]} obs',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Velocity (m/s)')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

        _vel_title = ('Analysis vs HYCOM Velocity at Most-Observed Grid Cells'
                      if hycom_truth is not None
                      else 'Analysis vs Observed Velocity at Most-Observed Grid Cells')
        fig7.suptitle(_vel_title, fontsize=13)
        fig7_path = os.path.join(plot_dir, f'{save_prefix}_timeseries.png')
        fig7.savefig(fig7_path, dpi=150)
        print(f"Saved: {fig7_path}")

    # ================================================================
    #  Figure 8: SST time series at selected cells
    # ================================================================
    if has_obs:
        # Select cells with the most SST observations specifically
        sst_obs_count = np.zeros(ncells, dtype=int)
        for t in range(nassim):
            ind_t = yind[t]
            sst_mask_t = (ind_t >= 3*ncells) & (ind_t < 4*ncells) & np.isfinite(yobs[t])
            for idx in ind_t[sst_mask_t]:
                cell = int(idx) - 3*ncells
                if 0 <= cell < ncells:
                    sst_obs_count[cell] += 1
        top_sst_cells = np.argsort(sst_obs_count)[-4:][::-1]

        fig8, axes8 = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        for panel in range(min(2, len(top_sst_cells))):
            cell_flat = top_sst_cells[panel]
            iy_c, ix_c = divmod(cell_flat, nx)
            lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
            t_idx = 3 * ncells + cell_flat    # layer-0 T

            anal_T = lsmcmc_mean[:, t_idx] - 273.15

            ax = axes8[panel]
            if hycom_truth is not None:
                # Nonlinear case: plot HYCOM reference instead of obs
                hycom_T = hycom_truth[:, t_idx] - 273.15
                ax.plot(np.arange(nassim + 1), hycom_T, 'r--', alpha=0.6,
                        lw=1.2, label='HYCOM SST')
                ax.plot(np.arange(nassim + 1), anal_T, 'k-', lw=1.5,
                        label='Analysis SST')
            else:
                obs_T = np.full(nassim, np.nan)
                for t in range(nassim):
                    match = np.where((yind[t] == t_idx) & np.isfinite(yobs[t]))[0]
                    if match.size > 0:
                        obs_T[t] = np.mean(yobs[t][match]) - 273.15
                ax.plot(np.arange(nassim), obs_T, 'rx', ms=7, label='Obs SST')
                ax.plot(np.arange(nassim + 1), anal_T, 'k-', lw=1.5,
                        label='Analysis SST')
            ax.set_title(f'SST at ({lon_c:.1f} deg E, {lat_c:.1f} deg N) -- {sst_obs_count[cell_flat]} SST obs',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Temperature (deg C)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        _sst_title = ('SST: Analysis vs HYCOM' if hycom_truth is not None
                       else 'SST: Analysis vs Drifter Observations')
        fig8.suptitle(_sst_title, fontsize=13)
        fig8_path = os.path.join(plot_dir, f'{save_prefix}_sst_timeseries.png')
        fig8.savefig(fig8_path, dpi=150)
        print(f"Saved: {fig8_path}")

    # ================================================================
    #  Figure 9: SSH time series at most-observed cells (like Fig 7/8)
    # ================================================================
    if H_b is not None and has_obs:
        # --- Reuse SWOT data loaded for Figure 5, bin to grid per cycle ---
        from datetime import datetime as _dt9

        swot_epoch = _dt9(2000, 1, 1)

        # Load SWOT SSH and bin to grid per cycle
        ssh_obs_count = np.zeros(ncells, dtype=int)
        ssh_obs_per_cycle = {}   # cycle -> dict{cell_flat: mean_ssh_anomaly}

        if has_ssh_obs and len(swot_ssh_all) > 0:
            try:
                print(f"Binning {len(swot_ssh_all):,} cached SWOT points for SSH timeseries ...")

                # Pre-bin SWOT data to grid cells ONCE (vectorized)
                dlon = abs(lon_grid[1] - lon_grid[0]) / 2.0
                dlat_g = abs(lat_grid[1] - lat_grid[0]) / 2.0
                time_tol = 3600.0  # 1 hour

                ix_all = np.searchsorted(lon_grid, swot_lons_all) - 1
                ix_all = np.clip(ix_all, 0, nx - 1)
                ix_p1 = np.clip(ix_all + 1, 0, nx - 1)
                closer_right = np.abs(lon_grid[ix_p1] - swot_lons_all) < np.abs(lon_grid[ix_all] - swot_lons_all)
                ix_all = np.where(closer_right, ix_p1, ix_all)

                iy_all = np.searchsorted(lat_grid, swot_lats_all) - 1
                iy_all = np.clip(iy_all, 0, ny - 1)
                iy_p1 = np.clip(iy_all + 1, 0, ny - 1)
                closer_up = np.abs(lat_grid[iy_p1] - swot_lats_all) < np.abs(lat_grid[iy_all] - swot_lats_all)
                iy_all = np.where(closer_up, iy_p1, iy_all)

                # Filter points within grid cell tolerance
                within_grid = ((np.abs(lon_grid[ix_all] - swot_lons_all) <= dlon) &
                               (np.abs(lat_grid[iy_all] - swot_lats_all) <= dlat_g))
                ix_g = ix_all[within_grid]
                iy_g = iy_all[within_grid]
                ssh_grid = swot_ssh_all[within_grid]
                time_grid = swot_times_all[within_grid]
                cell_flat_all = iy_g * nx + ix_g

                # Convert obs_times (unix epoch) to SWOT epoch
                # offset = (1970 - 2000) in seconds = -946684800
                unix_to_swot_offset = (_dt9(1970, 1, 1) - swot_epoch).total_seconds()

                # Sort by time for efficient cycle matching
                sort_idx = np.argsort(time_grid)
                time_sorted = time_grid[sort_idx]
                cell_sorted = cell_flat_all[sort_idx]
                ssh_sorted = ssh_grid[sort_idx]

                print(f"  Pre-binned {len(ssh_sorted):,} SWOT points to grid")

                # Match to assimilation cycles using sorted time
                for t_idx in range(nassim):
                    t_unix = obs_times[t_idx + 1] if t_idx + 1 < len(obs_times) else obs_times[-1]
                    if t_unix <= 0:
                        continue
                    t_swot = t_unix + unix_to_swot_offset
                    lo = np.searchsorted(time_sorted, t_swot - time_tol)
                    hi = np.searchsorted(time_sorted, t_swot + time_tol)
                    if lo >= hi:
                        continue

                    cells_t = cell_sorted[lo:hi]
                    ssh_t = ssh_sorted[lo:hi]

                    # Average per cell
                    cell_means = {}
                    for c, s in zip(cells_t, ssh_t):
                        if c not in cell_means:
                            cell_means[c] = [s]
                        else:
                            cell_means[c].append(s)
                    for c in cell_means:
                        cell_means[c] = np.mean(cell_means[c])
                        ssh_obs_count[c] += 1
                    if cell_means:
                        ssh_obs_per_cycle[t_idx] = cell_means

                print(f"  SWOT: {ssh_obs_count.sum()} obs across {len(ssh_obs_per_cycle)} cycles")
            except Exception as e:
                print(f"Could not load SWOT SSH for timeseries: {e}")
                import traceback; traceback.print_exc()

        if ssh_obs_count.sum() > 0:
            top_ssh_cells = np.argsort(ssh_obs_count)[-4:][::-1]
        else:
            # Fallback: pick 4 ocean cells if no SSH obs
            ocean_flat = np.where(ocean_mask.ravel())[0]
            step_f = max(1, len(ocean_flat) // 5)
            top_ssh_cells = ocean_flat[step_f::step_f][:4]
            print("No SSH obs found; using representative ocean cells")

        fig9, axes9 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for panel, cell_flat in enumerate(top_ssh_cells[:4]):
            ax = axes9.flat[panel]
            iy_c, ix_c = divmod(cell_flat, nx)
            lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
            h_b_cell = H_b[iy_c, ix_c]

            # Analysis SSH at this cell
            anal_ssh = np.zeros(nassim + 1)
            for t in range(nassim + 1):
                h_total_t, _, _, _ = get_fields(t)
                anal_ssh[t] = h_total_t[iy_c, ix_c] - h_b_cell

            ax.plot(np.arange(nassim + 1), anal_ssh, 'b-', alpha=0.8, lw=1.5,
                    label='Analysis SSH')
            if hycom_truth is not None:
                # Nonlinear case: plot HYCOM SSH reference
                hycom_ssh = hycom_truth[:, cell_flat] - h_b_cell
                ax.plot(np.arange(nassim + 1), hycom_ssh, 'g--', alpha=0.6,
                        lw=1.2, label='HYCOM SSH')
            else:
                # SWOT SSH obs at this cell
                obs_ssh = np.full(nassim, np.nan)
                for t_idx, cell_dict in ssh_obs_per_cycle.items():
                    if cell_flat in cell_dict:
                        obs_ssh[t_idx] = cell_dict[cell_flat]
                n_obs_cell = np.sum(np.isfinite(obs_ssh))
                ax.plot(np.arange(nassim), obs_ssh, 'gx', ms=7,
                        label=f'SWOT SSH obs ({n_obs_cell})')
            ax.set_title(f'({lon_c:.1f}°E, {lat_c:.1f}°N) — '
                         f'{ssh_obs_count[cell_flat]} SSH obs',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('SSH (m)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        _ssh_ts_title = ('SSH: Analysis vs HYCOM at Most-Observed Grid Cells'
                         if hycom_truth is not None
                         else 'SSH: Analysis vs SWOT Observations at Most-Observed Grid Cells')
        fig9.suptitle(_ssh_ts_title, fontsize=13)
        fig9_path = os.path.join(plot_dir, f'{save_prefix}_ssh_timeseries.png')
        fig9.savefig(fig9_path, dpi=150)
        print(f"Saved: {fig9_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model type: 3-layer MLSWE (Boussinesq, hydrostatic)")
    print(f"  Assimilation cycles: {nassim}")
    print(f"  Grid: {ny} x {nx}  ({lon_min} deg to {lon_max} deg E, "
          f"{lat_min} deg to {lat_max} deg N)")
    print(f"  Fields: {nfields}  (state dim = {dimx})")
    if has_obs:
        nobs_per = [np.sum((yind[t] >= 0) & np.isfinite(yobs[t]))
                    for t in range(nassim)]
        print(f"  Obs per cycle: {min(nobs_per)} - {max(nobs_per)}")
        print(f"  sig_y (vel noise):  {sig_y:.6f}")
        if rmse_vel is not None and np.any(np.isfinite(rmse_vel)):
            print(f"  Vel RMSE range: {np.nanmin(rmse_vel):.6f} - "
                  f"{np.nanmax(rmse_vel):.6f} m/s")
            print(f"  Mean vel RMSE:  {np.nanmean(rmse_vel):.6f} m/s")
            print(f"  Data std (vel): {data_std:.4f} m/s")
            ratio = np.nanmean(rmse_vel) / data_std if data_std > 0 else np.inf
            print(f"  Vel RMSE/std:   {ratio:.3f}")
        if rmse_sst is not None and np.any(np.isfinite(rmse_sst)):
            print(f"  SST RMSE range: {np.nanmin(rmse_sst):.4f} - "
                  f"{np.nanmax(rmse_sst):.4f} K")
            print(f"  Mean SST RMSE:  {np.nanmean(rmse_sst):.4f} K")
        if rmse_ssh is not None and np.any(np.isfinite(rmse_ssh)):
            print(f"  SSH RMSE range: {np.nanmin(rmse_ssh):.4f} - "
                  f"{np.nanmax(rmse_ssh):.4f} m")
            print(f"  Mean SSH RMSE:  {np.nanmean(rmse_ssh):.4f} m")
    if H_b is not None:
        print(f"  Bathymetry: depth range {H_b.min():.0f} - {H_b.max():.0f} m")
    print("=" * 60)
    plt.close('all')
    print(f"\nAll plots saved to: {plot_dir}/")


if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else './output_lsmcmc'
    config = sys.argv[2] if len(sys.argv) > 2 else None
    label = sys.argv[3] if len(sys.argv) > 3 else 'LSMCMC'
    main(outdir=outdir, config_file=config, method_label=label)

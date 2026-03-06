#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_mlswe_lsmcmc_nlrealdata_V1.py
===================================
Nonlinear LSMCMC (V1) with REAL observations, RMSE vs HYCOM.

Unlike the twin experiment (run_mlswe_lsmcmc_nldata_V1.py) which
applies arctan to *synthetic* observations, here we:
  1. Use real drifter + SWOT observations directly.
  2. Apply the nonlinear obs operator h(z) = arctan(H·z) inside the
     filter (the filter "thinks" the observations come from arctan(state)).
  3. Compute RMSE by comparing each analysis field against HYCOM
     reanalysis interpolated to the model grid at each cycle time.

Usage
-----
    python run_mlswe_lsmcmc_nlrealdata_V1.py [config.yml]

Default config: ``example_input_mlswe_nlrealdata_V1.yml``
"""
import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
_SWE_DIR = os.path.join(os.path.dirname(__file__), '..', 'SWE_LSMCMC')
if os.path.isdir(_SWE_DIR):
    sys.path.insert(0, os.path.abspath(_SWE_DIR))

from netCDF4 import Dataset
from mlswe.lsmcmc_nl_V1 import NL_SMCMC_MLSWE_Filter
import run_mlswe_lsmcmc_ldata_V1


# ---- Nonlinear observation operator: h(z) = arctan(z) ----
def obs_operator_arctan(z_local, H_loc, obs_ind_local):
    """y_pred = arctan(H_loc @ z_local)"""
    Hz = H_loc @ z_local if hasattr(H_loc, 'dot') else H_loc @ z_local
    return np.arctan(Hz)


def build_hycom_truth_states(bc_file, params, H_b, obs_times):
    """
    Build HYCOM "truth" state vectors at each assimilation cycle.

    Constructs full state vectors [h0,u0,v0,T0, h1,u1,v1,T1, h2,u2,v2,T2]
    from HYCOM reanalysis at each cycle time, using the same layer
    distribution as the model initialisation.

    Parameters
    ----------
    bc_file : str
        Path to HYCOM boundary condition NetCDF file.
    params : dict
        Model parameters (grid, H_rest, alpha_h, beta_vel, etc.).
    H_b : ndarray (ny, nx)
        Bathymetry.
    obs_times : list of float
        Observation times (Unix epoch seconds), length = nassim.

    Returns
    -------
    truth_states : ndarray (nassim+1, dimx)
        HYCOM-derived state vectors. Index 0 = initial time,
        indices 1..nassim = after each assimilation cycle.
    """
    from scipy.interpolate import RegularGridInterpolator

    ny = int(params['dgy'])
    nx = int(params['dgx'])
    nc = ny * nx
    nlayers = 3
    fields_per_layer = 4
    dimx = nlayers * fields_per_layer * nc
    nassim = int(params['nassim'])

    H_rest = np.asarray(params.get('H_rest', [100.0, 400.0, 3500.0]),
                        dtype=np.float64)
    alpha_h = np.asarray(params.get('alpha_h', [0.6, 0.3, 0.1]),
                         dtype=np.float64)
    beta_vel = np.asarray(params.get('beta_vel', [1.0, 1.0, 1.0]),
                          dtype=np.float64)
    T_rest = np.asarray(params.get('T_rest', [298.15, 283.15, 275.15]),
                        dtype=np.float64)
    H_rest_total = float(H_rest.sum())

    # Load HYCOM fields
    with Dataset(bc_file, 'r') as nc_ds:
        bc_lat = np.asarray(nc_ds.variables['lat'][:])
        bc_lon = np.asarray(nc_ds.variables['lon'][:])
        bc_times = np.asarray(nc_ds.variables['time'][:])
        bc_ssh = np.asarray(nc_ds.variables['ssh'][:])
        bc_uo = np.asarray(nc_ds.variables['uo'][:])
        bc_vo = np.asarray(nc_ds.variables['vo'][:])
        # SST if available
        if 'sst' in nc_ds.variables:
            bc_sst = np.asarray(nc_ds.variables['sst'][:])
        else:
            bc_sst = None

    # Handle NaNs
    bc_ssh[np.isnan(bc_ssh)] = 0.0
    bc_uo[np.isnan(bc_uo)] = 0.0
    bc_vo[np.isnan(bc_vo)] = 0.0
    if bc_sst is not None:
        bc_sst[np.isnan(bc_sst)] = T_rest[0]

    # Model grid
    lon_min, lon_max = float(params['lon_min']), float(params['lon_max'])
    lat_min, lat_max = float(params['lat_min']), float(params['lat_max'])
    model_lat = np.linspace(lat_min, lat_max, ny)
    model_lon = np.linspace(lon_min, lon_max, nx)
    mg_lat, mg_lon = np.meshgrid(model_lat, model_lon, indexing='ij')
    pts = np.column_stack([mg_lat.ravel(), mg_lon.ravel()])

    # Time offset (HYCOM times are in seconds since some epoch)
    t0_str = params.get('obs_time_start', '2024-08-01T00:00:00')
    t0_dt = datetime.strptime(t0_str[:19], '%Y-%m-%dT%H:%M:%S')
    epoch = (t0_dt - datetime(1970, 1, 1)).total_seconds()
    # Detect time_offset: if bc_times are << epoch, they are relative
    if bc_times.max() < 1e8:
        time_offset = epoch
    else:
        time_offset = 0.0

    # Build all cycle times (including initial)
    dt = float(params['dt'])
    t_freq = int(params.get('t_freq', params.get('assim_timesteps', 48)))
    cycle_dt = t_freq * dt
    all_times = [epoch + i * cycle_dt for i in range(nassim + 1)]

    truth_states = np.zeros((nassim + 1, dimx), dtype=np.float64)

    print(f"[HYCOM truth] Building {nassim+1} truth states from {bc_file}")
    for ci, t_cycle in enumerate(all_times):
        # Time interpolation in HYCOM
        t_query = t_cycle - epoch + (bc_times[0] if bc_times[0] > 0 else 0.0)
        if time_offset > 0:
            t_query = t_cycle - time_offset + bc_times[0]
        else:
            t_query = t_cycle

        idx = np.searchsorted(bc_times, t_query) - 1
        idx = max(0, min(idx, len(bc_times) - 2))
        t0b = bc_times[idx]
        t1b = bc_times[idx + 1]
        dtb = t1b - t0b
        alpha_t = (t_query - t0b) / dtb if dtb > 0 else 0.0
        alpha_t = np.clip(alpha_t, 0.0, 1.0)

        # Interpolate fields
        ssh_t = (1.0 - alpha_t) * bc_ssh[idx] + alpha_t * bc_ssh[idx + 1]
        uo_t = (1.0 - alpha_t) * bc_uo[idx] + alpha_t * bc_uo[idx + 1]
        vo_t = (1.0 - alpha_t) * bc_vo[idx] + alpha_t * bc_vo[idx + 1]
        if bc_sst is not None:
            sst_t = (1.0 - alpha_t) * bc_sst[idx] + alpha_t * bc_sst[idx + 1]
        else:
            sst_t = None

        # Spatial interpolation to model grid
        def interp_field(field_2d):
            interp = RegularGridInterpolator(
                (bc_lat, bc_lon), field_2d,
                method='linear', bounds_error=False, fill_value=None)
            return interp(pts).reshape(ny, nx)

        ssh_grid = interp_field(ssh_t)
        uo_grid = interp_field(uo_t)
        vo_grid = interp_field(vo_t)
        sst_grid = interp_field(sst_t) if sst_t is not None else None

        # Build state vector matching model's state_flat convention:
        #   slot 0  = h_total (Σhₖ), NOT layer-0 thickness
        #   slots 1,2 = 0 (unused h-slots for layers 1,2)
        #   SST in Kelvin (model internal unit)
        h_total = np.maximum(H_b + ssh_grid, 10.0)

        z_truth = np.zeros(dimx, dtype=np.float64)
        for k in range(nlayers):
            offset = k * fields_per_layer * nc
            # h-slot: only k=0 gets h_total; others are 0
            if k == 0:
                z_truth[offset: offset + nc] = h_total.ravel()
            else:
                z_truth[offset: offset + nc] = 0.0
            # u_k
            z_truth[offset + nc: offset + 2*nc] = (
                beta_vel[k] * uo_grid).ravel()
            # v_k
            z_truth[offset + 2*nc: offset + 3*nc] = (
                beta_vel[k] * vo_grid).ravel()
            # T_k  (HYCOM SST is in °C → convert to Kelvin)
            if k == 0 and sst_grid is not None:
                z_truth[offset + 3*nc: offset + 4*nc] = (
                    sst_grid + 273.15).ravel()
            else:
                z_truth[offset + 3*nc: offset + 4*nc] = T_rest[k]

        truth_states[ci] = z_truth

    print(f"[HYCOM truth] Done. State dim={dimx}, "
          f"SSH range=[{ssh_grid.min():.3f}, {ssh_grid.max():.3f}] m "
          f"at final cycle")
    return truth_states


# ---- Wrapper: inject obs_operator + HYCOM truth state ----
_OrigFilter = NL_SMCMC_MLSWE_Filter


class _NLFilterWithArctanHYCOM(_OrigFilter):
    """NL filter with arctan obs-operator and HYCOM truth for RMSE."""

    _truth_state = None  # set externally before .run()
    _rmse_obs_only = True  # real data: RMSE at obs locations vs HYCOM

    def __init__(self, isim, params):
        super().__init__(isim, params, obs_operator=obs_operator_arctan)


# ---- Monkey-patch and run ----
run_mlswe_lsmcmc_ldata_V1.Loc_SMCMC_MLSWE_Filter = _NLFilterWithArctanHYCOM

# We need to intercept the runner to inject HYCOM truth BEFORE filt.run()
_original_main = run_mlswe_lsmcmc_ldata_V1.main


def main_with_hycom_truth():
    """
    Modified main that builds HYCOM truth states and injects them
    into the filter for RMSE computation.
    """
    import yaml

    # Load config
    config_file = sys.argv[1] if len(sys.argv) > 1 else \
        'example_input_mlswe_nlrealdata_V1.yml'
    with open(config_file, 'r') as f:
        params = yaml.safe_load(f)

    # Check if HYCOM truth is requested
    use_hycom = params.get('use_hycom_truth', False)

    if not use_hycom:
        # Fall back to normal run
        _original_main()
        return

    # We need to hook into the runner. The simplest approach:
    # Build HYCOM truth, then monkey-patch the filter class to
    # auto-inject it at construction.

    bc_file = params.get('bc_file', './data/hycom_bc_2024aug.nc')

    # Build obs_times for HYCOM interpolation
    nassim = int(params['nassim'])
    dt = float(params['dt'])
    t_freq = int(params.get('t_freq', params.get('assim_timesteps', 48)))
    t0_str = params.get('obs_time_start', '2024-08-01T00:00:00')
    t0_dt = datetime.strptime(t0_str[:19], '%Y-%m-%dT%H:%M:%S')
    epoch = (t0_dt - datetime(1970, 1, 1)).total_seconds()
    cycle_dt = t_freq * dt
    obs_times = [epoch + (i + 1) * cycle_dt for i in range(nassim)]

    # We need H_b — load it the same way the runner does
    ny = int(params['dgy'])
    nx = int(params['dgx'])
    lon_min, lon_max = float(params['lon_min']), float(params['lon_max'])
    lat_min, lat_max = float(params['lat_min']), float(params['lat_max'])
    model_lat = np.linspace(lat_min, lat_max, ny)
    model_lon = np.linspace(lon_min, lon_max, nx)
    H_b = run_mlswe_lsmcmc_ldata_V1.load_bathymetry(
        params, ny, nx, model_lon, model_lat)
    if H_b is None:
        H_b = np.full((ny, nx), float(params.get('H_mean', 4000.0)))

    # Build HYCOM truth states
    truth_states = build_hycom_truth_states(bc_file, params, H_b, obs_times)

    # Monkey-patch the filter class to auto-inject truth
    class _FilterWithTruth(_NLFilterWithArctanHYCOM):
        def __init__(self, isim, params):
            super().__init__(isim, params)
            self._truth_state = truth_states

    run_mlswe_lsmcmc_ldata_V1.Loc_SMCMC_MLSWE_Filter = _FilterWithTruth

    # Now run normally
    _original_main()


if __name__ == '__main__':
    # Default config
    if len(sys.argv) < 2:
        sys.argv.append('example_input_mlswe_nlrealdata_V1.yml')
    main_with_hycom_truth()

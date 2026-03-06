#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_mlswe_lsmcmc_nlrealdata_V2.py
===================================
Nonlinear LSMCMC (V2) with REAL observations, RMSE vs HYCOM.

Unlike the twin experiment, here we:
  1. Use real drifter + SWOT observations directly.
  2. Apply the nonlinear obs operator h(z) = arctan(H·z) inside the
     filter (the filter "thinks" the observations come from arctan(state)).
  3. Compute RMSE by comparing each analysis field against HYCOM
     reanalysis interpolated to the model grid at each cycle time.

Usage
-----
    python run_mlswe_lsmcmc_nlrealdata_V2.py [config.yml]

Default config: ``example_input_mlswe_nlrealdata_V2.yml``
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
from mlswe.lsmcmc_nl_V2 import NL_SMCMC_MLSWE_Filter_V2
import run_mlswe_lsmcmc_ldata_V1

# Reuse the HYCOM truth builder from the V1 real-data runner
from run_mlswe_lsmcmc_nlrealdata_V1 import (
    build_hycom_truth_states,
    obs_operator_arctan,
)

import yaml


# ---- Wrapper: inject obs_operator + HYCOM truth state ----
_OrigFilterV2 = NL_SMCMC_MLSWE_Filter_V2


class _NLFilterV2WithArctanHYCOM(_OrigFilterV2):
    """NL-V2 filter with arctan obs-operator and HYCOM truth for RMSE."""

    _truth_state = None  # set externally before .run()
    _rmse_obs_only = True  # real data: RMSE at obs locations vs HYCOM

    def __init__(self, isim, params):
        super().__init__(isim, params, obs_operator=obs_operator_arctan)


def main():
    # Load config
    config_file = sys.argv[1] if len(sys.argv) > 1 else \
        'example_input_mlswe_nlrealdata_V2.yml'
    with open(config_file, 'r') as f:
        params = yaml.safe_load(f)

    use_hycom = params.get('use_hycom_truth', False)

    if not use_hycom:
        # No HYCOM truth — just run with arctan obs operator
        run_mlswe_lsmcmc_ldata_V1.Loc_SMCMC_MLSWE_Filter = \
            _NLFilterV2WithArctanHYCOM
        run_mlswe_lsmcmc_ldata_V1.main()
        return

    bc_file = params.get('bc_file', './data/hycom_bc_2024aug.nc')

    nassim = int(params['nassim'])
    dt = float(params['dt'])
    t_freq = int(params.get('t_freq', params.get('assim_timesteps', 48)))
    t0_str = params.get('obs_time_start', '2024-08-01T00:00:00')
    t0_dt = datetime.strptime(t0_str[:19], '%Y-%m-%dT%H:%M:%S')
    epoch = (t0_dt - datetime(1970, 1, 1)).total_seconds()
    cycle_dt = t_freq * dt
    obs_times = [epoch + (i + 1) * cycle_dt for i in range(nassim)]

    # Load bathymetry
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

    # Build HYCOM truth
    truth_states = build_hycom_truth_states(bc_file, params, H_b, obs_times)

    # Create filter class with truth auto-injected
    class _FilterV2WithTruth(_NLFilterV2WithArctanHYCOM):
        def __init__(self, isim, params):
            super().__init__(isim, params)
            self._truth_state = truth_states

    run_mlswe_lsmcmc_ldata_V1.Loc_SMCMC_MLSWE_Filter = _FilterV2WithTruth
    run_mlswe_lsmcmc_ldata_V1.main()


if __name__ == '__main__':
    # Default config
    if len(sys.argv) < 2:
        sys.argv.append('example_input_mlswe_nlrealdata_V2.yml')
    main()

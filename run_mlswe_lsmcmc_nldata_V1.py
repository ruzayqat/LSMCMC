#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_mlswe_lsmcmc_nldata_V1.py
==============================
Runner for the nonlinear LSMCMC (V1) — block-partition localization
with MCMC (Gibbs-within-MH or joint RW-MH).

Reuses all data-loading logic from ``run_mlswe_lsmcmc_ldata_V1.py`` but
substitutes ``NL_SMCMC_MLSWE_Filter`` from ``mlswe.lsmcmc_nl_V1``.

Usage
-----
    python run_mlswe_lsmcmc_nldata_V1.py [config.yml]

Default config: ``example_input_mlswe_nldata_real_V1.yml``
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
_SWE_DIR = os.path.join(os.path.dirname(__file__), '..', 'SWE_LSMCMC')
if os.path.isdir(_SWE_DIR):
    sys.path.insert(0, os.path.abspath(_SWE_DIR))

from mlswe.lsmcmc_nl_V1 import NL_SMCMC_MLSWE_Filter
import run_mlswe_lsmcmc_ldata_V1


# ---- Nonlinear observation operator: h(z) = arctan(z) ----
def obs_operator_arctan(z_local, H_loc, obs_ind_local):
    """Apply arctan to the linearly-selected state components.

    H_loc selects the observed entries from z_local, then arctan is
    applied element-wise:  y_pred = arctan(H_loc @ z_local).
    """
    Hz = H_loc @ z_local if hasattr(H_loc, 'dot') else H_loc @ z_local
    return np.arctan(Hz)


# ---- Wrap the filter class to inject obs_operator ----
_OrigFilter = NL_SMCMC_MLSWE_Filter


class _NLFilterWithArctan(_OrigFilter):
    """Thin wrapper that injects obs_operator=arctan at construction."""
    def __init__(self, isim, params):
        super().__init__(isim, params, obs_operator=obs_operator_arctan)


run_mlswe_lsmcmc_ldata_V1.Loc_SMCMC_MLSWE_Filter = _NLFilterWithArctan

# Default to the NL config if no argument given
if len(sys.argv) < 2:
    sys.argv.append('example_input_mlswe_nldata_real_V1.yml')

run_mlswe_lsmcmc_ldata_V1.main()

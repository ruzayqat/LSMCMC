#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_lsmcmc_nongauss.py
======================
Runner for the localized LSMCMC (V2) variant with **non-Gaussian
observation noise** (ε-contaminated Gaussian mixture).

Reuses all data-loading logic from ``run_mlswe_lsmcmc_ldata_V1.py`` but
substitutes ``NonGaussianFilter`` from ``nongauss_ldata.lsmcmc_nongauss``.

The observation-noise distribution is a J-component Gaussian mixture::

    V_t ~ Σ_j  α_j  N(0, s_j² · R)

where α_j are mixture weights and s_j are scale factors on the base
observation noise R = diag(σ²_y).

The posterior remains a Gaussian mixture (J × N_f components) and is
sampled directly — no MCMC iteration needed.

Usage
-----
    python run_lsmcmc_nongauss.py [config.yml]

Default config: ``nongauss_ldata/input_nongauss.yml``
"""
import sys
import os

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ---- Import the non-Gaussian filter and monkey-patch into V1 runner ----
from nongauss_ldata.lsmcmc_nongauss import NonGaussianFilter
import run_mlswe_lsmcmc_ldata_V1

run_mlswe_lsmcmc_ldata_V1.Loc_SMCMC_MLSWE_Filter = NonGaussianFilter

# Default to the non-Gaussian config if no argument given
if len(sys.argv) < 2:
    sys.argv.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'input_nongauss.yml'))

run_mlswe_lsmcmc_ldata_V1.main()

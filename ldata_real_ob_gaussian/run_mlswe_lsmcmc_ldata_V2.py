#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_mlswe_lsmcmc_ldata_V2.py
=============================
Runner for the localized LSMCMC (V2) variant with per-block halo
localization and Gaspari-Cohn tapering.

Reuses all data-loading logic from ``run_mlswe_lsmcmc_ldata_V1.py`` but
substitutes ``Loc_SMCMC_MLSWE_Filter_V2`` from ``mlswe.lsmcmc_V2``.

Usage
-----
    python run_mlswe_lsmcmc_ldata_V2.py [config.yml]

Default config: ``example_input_mlswe_ldata_V2.yml``
"""
import sys
import os

# Ensure imports work — add repo root so 'mlswe' package is found
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# ---- Redirect the filter class used by the original runner ----
from mlswe.lsmcmc_V2 import Loc_SMCMC_MLSWE_Filter_V2
import run_mlswe_lsmcmc_ldata_V1

run_mlswe_lsmcmc_ldata_V1.Loc_SMCMC_MLSWE_Filter = Loc_SMCMC_MLSWE_Filter_V2

# Default to the localized config if no argument given
if len(sys.argv) < 2:
    sys.argv.append('example_input_mlswe_ldata_V2.yml')

run_mlswe_lsmcmc_ldata_V1.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_mlswe_lsmcmc_nldata_V2_twin.py
===================================
Synthetic twin experiment for the nonlinear LSMCMC **V2**
(per-block halo-based localization with Gaspari-Cohn tapering).

Identical workflow to ``run_mlswe_lsmcmc_nldata_V1_twin.py`` (V1) except it
uses ``NL_SMCMC_MLSWE_Filter_V2`` from ``mlswe.lsmcmc_nl_V2``.

Usage
-----
    python run_mlswe_lsmcmc_nldata_V2_twin.py [config.yml]

Default config: ``example_input_mlswe_nldata_V2_twin.yml``
"""
import os
import sys

# ---- Reuse everything from the V1 twin runner ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'ldata_real_ob_gaussian'))
import run_mlswe_lsmcmc_nldata_V1_twin as _v1_twin

# ---- Swap the filter class to V2 ----
from mlswe.lsmcmc_nl_V2 import NL_SMCMC_MLSWE_Filter_V2

# Nonlinear obs operator (same arctan as V1)
obs_operator_arctan = _v1_twin.obs_operator_arctan


class _TwinFilterV2(NL_SMCMC_MLSWE_Filter_V2):
    """V2 filter with arctan obs-operator and nature-run-state RMSE."""

    _truth_state = None

    def __init__(self, isim, params):
        super().__init__(isim, params, obs_operator=obs_operator_arctan)


def main():
    config_file = (sys.argv[1] if len(sys.argv) > 1
                   else 'example_input_mlswe_nldata_V2_twin.yml')

    # Monkey-patch V1 twin module to use V2 filter
    _v1_twin._TwinFilter = _TwinFilterV2
    _v1_twin._OrigFilter = NL_SMCMC_MLSWE_Filter_V2

    # Override default config in sys.argv
    if len(sys.argv) <= 1:
        sys.argv.append(config_file)
    else:
        sys.argv[1] = config_file

    _v1_twin.main()


if __name__ == '__main__':
    main()

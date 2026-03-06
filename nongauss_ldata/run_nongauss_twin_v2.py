#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_nongauss_twin_v2.py
========================
Non-Gaussian twin experiment using ``NonGaussianFilter`` (V2) which
knows the correct Gaussian-mixture observation noise model.

Reuses all infrastructure from ``run_nongauss_twin.py`` (nature run
generation, obs loading, RMSE computation) but substitutes the
``NonGaussianFilter`` which does exact direct sampling from the
J × N_f Gaussian mixture posterior.

Usage
-----
    python nongauss_ldata/run_nongauss_twin_v2.py [config.yml]

Default config: ``nongauss_ldata/input_nongauss_twin_v2.yml``
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# ---- Reuse everything from the V1 twin runner ----
import nongauss_ldata.run_nongauss_twin as _v1_twin

# ---- Swap the filter class to NonGaussianFilter (correct mixture model) ----
from nongauss_ldata.lsmcmc_nongauss import NonGaussianFilter


class _TwinFilterV2(NonGaussianFilter):
    """NonGaussianFilter with nature-run RMSE evaluation."""
    _truth_state = None

    def __init__(self, isim, params):
        super().__init__(isim, params)


def main():
    config_file = (sys.argv[1] if len(sys.argv) > 1
                   else os.path.join(_HERE, 'input_nongauss_twin_v2.yml'))

    # Monkey-patch the V1 twin module to use NonGaussianFilter
    _v1_twin._TwinFilter = _TwinFilterV2
    _v1_twin._OrigFilter = NonGaussianFilter

    # Override default config in sys.argv
    if len(sys.argv) <= 1:
        sys.argv.append(config_file)
    else:
        sys.argv[1] = config_file

    _v1_twin.main()


if __name__ == '__main__':
    main()

"""
mlswe – Multi-Layer Shallow Water Equation solver and utilities for
data assimilation on an Atlantic Ocean sub-domain with real drifter
observations.

This package implements a 3-layer primitive-equations model (Boussinesq,
hydrostatic, stacked isopycnal layers) with RK4 time integration, intended
to produce baroclinic dynamics comparable to HYCOM.
"""
from .model import MLSWE
from .model import coriolis_array, lonlat_to_dxdy

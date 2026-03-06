"""
mlswe.letkf  –  LETKF utilities adapted for the 3-layer MLSWE model
=====================================================================

The single-layer LETKF code (swe_letkf_utils.py) is generic enough to
work with any number of fields.  For the multi-layer model:

    nfields = 12  (h₀,u₀,v₀,T₀, h₁,u₁,v₁,T₁, h₂,u₂,v₂,T₂)
    ncells  = ny × nx
    dimx    = 12 × ncells

Observations map to layer-0 only (surface drifter data).

This module re-exports the needed functions, adding only the
covariance-inflation modifier for multi-layer state vectors.

Performance notes (v2):
  * covlocal_local: only local columns computed per rank.
  * calcwts_letkf: diagonal of R^{-1} (no dense diag matrix).
  * Inner loop: vectorised field update (matrix multiply instead of per-field loop).
"""
import sys
import os
import numpy as np

# Import from SWE_LSMCMC if available
_SWE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         '..', 'SWE_LSMCMC')
if os.path.isdir(_SWE_DIR):
    _SWE_DIR = os.path.abspath(_SWE_DIR)
    if _SWE_DIR not in sys.path:
        sys.path.insert(0, _SWE_DIR)

from swe_letkf_utils import (
    haversine_distance,
    gaspcohn,
    calcwts_letkf,
    precompute_covlocal,
    precompute_covlocal_local,
)


def letkf_update_mlswe_mpi(xens, obs_values, obs_indices,
                            obs_errvar, covlocal_local,
                            nfields, ncells, nstart, nend):
    """
    MPI-parallelised LETKF analysis update for the multi-layer model.

    Identical to the single-layer version except nfields = 12.
    The weight matrix computed from surface observations is applied
    to ALL 12 fields at each grid point, propagating information
    from surface obs to all layers through the ensemble covariances.

    Parameters
    ----------
    xens : (K, dimx)   where dimx = nfields × ncells
    obs_values : (nobs,)
    obs_indices : (nobs,)   into flat state vector
    obs_errvar : (nobs,)
    covlocal_local : (nobs, nlocal)   LOCAL localisation matrix
    nfields : int = 12
    ncells : int = ny × nx
    nstart, nend : int   grid-point range for this MPI rank

    Returns
    -------
    xens_updated_local : (K, nfields, nlocal)
    """
    nanals = xens.shape[0]
    nobs = len(obs_values)
    nlocal = nend - nstart + 1

    xens_3d = xens.reshape(nanals, nfields, ncells)

    hxens = xens[:, obs_indices]
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean

    innovations = obs_values - hxmean

    xmean_3d = xens_3d.mean(axis=0)                          # (nfields, ncells)
    xprime_local = xens_3d[:, :, nstart:nend + 1] - \
                   xmean_3d[:, nstart:nend + 1]               # (K, nfields, nlocal)
    xmean_local = xmean_3d[:, nstart:nend + 1]               # (nfields, nlocal)

    inv_obs_errvar = 1.0 / obs_errvar

    # Initialise with prior (handles no-obs case)
    xens_updated_local = xens_3d[:, :, nstart:nend + 1].copy()

    for count in range(nlocal):
        loc_wt = covlocal_local[:, count]
        mask = loc_wt > 1.e-10
        nobs_local = int(mask.sum())

        if nobs_local == 0:
            continue  # prior already in output

        # Diagonal of localised R^{-1} — no dense diag matrix
        rinv_diag = loc_wt[mask] * inv_obs_errvar[mask]

        hx_local = hxprime[:, mask]
        innov_local = innovations[mask]

        wts = calcwts_letkf(hx_local, rinv_diag, innov_local, nanals)

        # If weights contain NaN (ill-conditioned Pa_tilde_inv), keep prior
        if not np.isfinite(wts).all():
            continue

        # Apply same weights to ALL fields via matrix multiply
        xp = xprime_local[:, :, count]       # (K, nfields)
        updated = xmean_local[:, count] + xp.T.dot(wts).T

        # Final check: only accept if result is finite
        if np.isfinite(updated).all():
            xens_updated_local[:, :, count] = updated

    return xens_updated_local


def letkf_update_mlswe_nl_mpi(xens, obs_values, obs_indices,
                               obs_errvar, covlocal_local,
                               nfields, ncells, nstart, nend):
    """
    MPI-parallelised LETKF analysis with **nonlinear arctan** obs operator.

    Identical to ``letkf_update_mlswe_mpi`` except:
        hxens = arctan(xens[:, obs_indices])
    instead of the linear identity observation operator.
    """
    nanals = xens.shape[0]
    nobs = len(obs_values)
    nlocal = nend - nstart + 1

    xens_3d = xens.reshape(nanals, nfields, ncells)

    # ---- Nonlinear obs operator: h(x) = arctan(x) ----
    hxens = np.arctan(xens[:, obs_indices])
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean

    innovations = obs_values - hxmean

    xmean_3d = xens_3d.mean(axis=0)
    xprime_local = xens_3d[:, :, nstart:nend + 1] - \
                   xmean_3d[:, nstart:nend + 1]
    xmean_local = xmean_3d[:, nstart:nend + 1]

    inv_obs_errvar = 1.0 / obs_errvar

    xens_updated_local = xens_3d[:, :, nstart:nend + 1].copy()

    for count in range(nlocal):
        loc_wt = covlocal_local[:, count]
        mask = loc_wt > 1.e-10
        nobs_local = int(mask.sum())

        if nobs_local == 0:
            continue

        rinv_diag = loc_wt[mask] * inv_obs_errvar[mask]
        hx_local = hxprime[:, mask]
        innov_local = innovations[mask]

        wts = calcwts_letkf(hx_local, rinv_diag, innov_local, nanals)

        # If weights contain NaN (ill-conditioned Pa_tilde_inv), keep prior
        if not np.isfinite(wts).all():
            continue

        xp = xprime_local[:, :, count]
        updated = xmean_local[:, count] + xp.T.dot(wts).T

        # Final check: only accept if result is finite
        if np.isfinite(updated).all():
            xens_updated_local[:, :, count] = updated

    return xens_updated_local

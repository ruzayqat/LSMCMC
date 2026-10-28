"""Optional diagnostics hooks for the MLSWE LSMCMC filters.

These helpers support three ensemble-calibration diagnostics:

  * ``free_run``            -- no-DA forecast (free) ensemble baseline
  * ``save_spread``        -- forecast ensemble spread + rank histograms
  * ``cv_fraction``        -- cross-validation against withheld observations

Every function is side-effect free except for the in-place accumulation of
arrays that are explicitly passed in.  The filters only call these helpers
when the corresponding flag is enabled, so default filter behaviour is
unchanged.
"""
import numpy as np


def var_masks(obs_ind_ml, ncells):
    """Return boolean masks ``(ssh, vel, sst)`` for observation state indices.

    The flat multi-layer state stores SSH/h in ``[0, ncells)``, the two
    velocity components in ``[ncells, 3*ncells)`` and SST in
    ``[3*ncells, 4*ncells)``.
    """
    ssh = (obs_ind_ml >= 0) & (obs_ind_ml < ncells)
    vel = (obs_ind_ml >= ncells) & (obs_ind_ml < 3 * ncells)
    sst = (obs_ind_ml >= 3 * ncells) & (obs_ind_ml < 4 * ncells)
    return ssh, vel, sst


def cv_split(n_obs, cv_fraction, seed):
    """Random assimilate/holdout boolean masks over ``n_obs`` observations.

    A fraction ``cv_fraction`` of the (already valid) observations is
    withheld from assimilation so the analysis can be scored against an
    independent set.  At least one observation is always kept for
    assimilation.
    """
    assim = np.ones(n_obs, dtype=bool)
    hold = np.zeros(n_obs, dtype=bool)
    if cv_fraction <= 0.0 or n_obs == 0:
        return assim, hold
    rng = np.random.default_rng(seed)
    hold = rng.random(n_obs) < cv_fraction
    if hold.all():            # never withhold *every* observation
        hold[0] = False
    return ~hold, hold


def spread_and_rank(forecast, obs_ind_ml, y_valid, ncells, rankhist, rng,
                    transform=None, sig_y=None):
    """Forecast (prior) ensemble spread and rank-histogram accumulation.

    Parameters
    ----------
    forecast : ndarray, shape (dimx, Nf)
        Ensemble matrix BEFORE the analysis update (the forecast/prior).
    obs_ind_ml : ndarray
        State indices of the verifying observations.
    y_valid : ndarray
        Verifying observation values (observation space).
    ncells : int
        Number of grid cells (``ny * nx``).
    rankhist : dict
        Dict with ``'ssh'``/``'vel'``/``'sst'`` integer arrays of length
        ``Nf + 1`` that are mutated in place.
    rng : numpy.random.Generator
        Generator used for random tie breaking of ranks and for the
        observation-error perturbation of the rank histogram.
    transform : callable or None
        Optional observation operator applied elementwise to the ensemble
        state values so the comparison happens in observation space
        (e.g. ``numpy.arctan`` for the nonlinear experiments).
    sig_y : float, ndarray or None
        Observation-error standard deviation(s) in observation space.  When
        provided, an independent draw from ``N(0, sig_y)`` is added to each
        ensemble member before computing the rank of the verifying
        observation (Hamill, 2001).  This places the noisy observation and
        the ensemble on an equal footing; omitting it produces a spuriously
        U-shaped rank histogram even for a perfectly reliable ensemble.  The
        reported ensemble spread is always the *unperturbed* spread.

    Returns
    -------
    dict
        Per-cycle RMS ensemble spread for each variable group.
    """
    ens = forecast[obs_ind_ml, :]               # (nobs, Nf)
    if transform is not None:
        ens = transform(ens)
    Nf = ens.shape[1]
    std = np.std(ens, axis=1)                   # pure ensemble spread

    # Rank histogram: perturb each ensemble member with an independent
    # observation-error draw so the verifying (noisy) observation is ranked
    # against an equally noisy ensemble (Hamill, 2001).
    ens_r = ens
    if sig_y is not None:
        sig_col = np.atleast_1d(np.asarray(sig_y, dtype=float))
        if sig_col.size == 1:
            sig_col = np.full(ens.shape[0], float(sig_col[0]))
        else:
            sig_col = sig_col[:ens.shape[0]]
        ens_r = ens + rng.normal(size=ens.shape) * sig_col[:, None]

    below = np.sum(ens_r < y_valid[:, None], axis=1)
    equal = np.sum(ens_r == y_valid[:, None], axis=1)
    rank = (below + np.floor(rng.random(ens_r.shape[0]) * (equal + 1))
            ).astype(int)
    rank = np.clip(rank, 0, Nf)
    masks = var_masks(obs_ind_ml, ncells)
    out = {}
    for nm, m in zip(('ssh', 'vel', 'sst'), masks):
        if m.sum() == 0:
            out[nm] = np.nan
            continue
        out[nm] = float(np.sqrt(np.mean(std[m] ** 2)))
        np.add.at(rankhist[nm], rank[m], 1)
    return out


def state_spread(ensemble, ncells):
    """Ensemble spread in STATE space (physical units), per variable group.

    Unlike :func:`spread_and_rank`, which measures the spread at observed
    locations in *observation* space (and therefore saturates to ~0 for the
    nonlinear ``arctan`` observation operator on large-magnitude SST/thickness
    fields), this helper measures the raw model-state ensemble standard
    deviation over the *full field*.  The variable groupings match the
    layer-0 field layout used for the truth-based RMSE so that the resulting
    time-mean spread is directly comparable to the mean RMSE:

      * ``ssh`` / thickness ``h`` : state indices ``[0, ncells)``
      * velocity ``(u, v)``       : state indices ``[ncells, 3*ncells)``
      * ``sst``                   : state indices ``[3*ncells, 4*ncells)``

    Parameters
    ----------
    ensemble : ndarray, shape (dimx, Nf)
        Ensemble matrix in state space.  Pass the posterior (analysis)
        ensemble so the spread is comparable to the analysis RMSE.
    ncells : int
        Number of grid cells (``ny * nx``).

    Returns
    -------
    dict
        Per-cycle RMS ensemble spread for each variable group, in the same
        physical units as the corresponding RMSE.
    """
    nc = ncells
    std = np.std(ensemble, axis=1)              # (dimx,)

    def _rms(seg):
        return float(np.sqrt(np.mean(seg ** 2))) if seg.size else np.nan

    return {
        'ssh': _rms(std[0:nc]),
        'vel': _rms(std[nc:3 * nc]),
        'sst': _rms(std[3 * nc:4 * nc]),
    }


def obs_spread(ensemble, obs_ind_ml, ncells, transform=None):
    """Ensemble spread at observation locations, per variable group.

    Computes the RMS ensemble standard deviation of the (optionally
    transformed) ensemble values at the observed state indices.  Pass the
    posterior (analysis) ensemble to obtain a spread comparable to an
    analysis RMSE evaluated at the same observation locations.

    Parameters
    ----------
    ensemble : ndarray, shape (dimx, Nf)
        Ensemble matrix in state space.
    obs_ind_ml : ndarray
        State indices of the verifying observations.
    ncells : int
        Number of grid cells (``ny * nx``).
    transform : callable or None
        Optional observation operator applied elementwise (e.g.
        ``numpy.arctan``).  Leave ``None`` for linear observations.

    Returns
    -------
    dict
        Per-cycle RMS ensemble spread for each variable group.
    """
    ens = ensemble[obs_ind_ml, :]
    if transform is not None:
        ens = transform(ens)
    std = np.std(ens, axis=1)
    masks = var_masks(obs_ind_ml, ncells)
    out = {}
    for nm, m in zip(('ssh', 'vel', 'sst'), masks):
        out[nm] = (float(np.sqrt(np.mean(std[m] ** 2)))
                   if m.sum() else np.nan)
    return out


def cv_rmse(z_a, obs_ind_hold, y_hold, ncells, transform=None):
    """RMSE of the analysis mean against withheld observations, per variable.

    ``transform`` is an optional observation operator (e.g. ``numpy.arctan``)
    applied to the analysis state at the withheld locations so the residual
    is taken in observation space.
    """
    pred = z_a[obs_ind_hold]
    if transform is not None:
        pred = transform(pred)
    resid = pred - y_hold
    masks = var_masks(obs_ind_hold, ncells)
    out = {}
    for nm, m in zip(('ssh', 'vel', 'sst'), masks):
        out[nm] = (float(np.sqrt(np.mean(resid[m] ** 2)))
                   if m.sum() > 0 else np.nan)
    return out

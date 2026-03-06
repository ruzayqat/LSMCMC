#!/usr/bin/env python
"""
run_letkf_sensitivity.py
========================
LETKF sensitivity analysis for the linear Gaussian model.

Performs a grid search over:
  - hscale : localization scale (in grid points) for GC taper
  - covinflate1 : multiplicative covariance inflation factor

For each (hscale, covinflate1) pair, runs LETKF with the given ensemble
size K and computes RMSE vs the KF mean.

Outputs a heatmap of RMSE and saves the sensitivity results.

The LETKF here uses Euclidean distances on the grid (not haversine)
and operates on a single field (nfields=1).
"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ===================================================================
#  Gaspari-Cohn taper (standalone, same as in swe_letkf_utils.py)
# ===================================================================
def gaspcohn(r):
    r = np.asarray(r, dtype=np.float64)
    rr = 2.0 * r
    rr += 1.e-13
    taper = np.where(
        r <= 0.5,
        (((-0.25 * rr + 0.5) * rr + 0.625) * rr - 5.0 / 3.0) * rr ** 2 + 1.0,
        np.zeros(r.shape, r.dtype))
    taper = np.where(
        np.logical_and(r > 0.5, r < 1.0),
        (((((rr / 12.0 - 0.5) * rr + 0.625) * rr + 5.0 / 3.0) * rr - 5.0)
         * rr + 4.0 - 2.0 / (3.0 * rr)),
        taper)
    return taper


# ===================================================================
#  LETKF weight computation (Hunt et al. 2007)
# ===================================================================
def calcwts_letkf(hx, rinv_diag, ominusf, nanals):
    YbRinv = hx * rinv_diag[np.newaxis, :]
    Pa_tilde_inv = (nanals - 1) * np.eye(nanals) + YbRinv.dot(hx.T)
    evals, eigs = np.linalg.eigh(Pa_tilde_inv)
    evals = np.maximum(evals, 1.e-10)
    inv_sqrt_evals = np.sqrt(1.0 / evals)
    sqrt_Pa_tilde = (eigs * inv_sqrt_evals[np.newaxis, :]).dot(eigs.T)
    Pa_tilde = sqrt_Pa_tilde.dot(sqrt_Pa_tilde.T)
    wa_bar = Pa_tilde.dot(YbRinv.dot(ominusf))[:, np.newaxis]
    Wa = np.sqrt(nanals - 1) * sqrt_Pa_tilde
    return wa_bar + Wa


# ===================================================================
#  Precompute Euclidean localisation (single grid point → all obs)
# ===================================================================
def precompute_covlocal_euclidean(obs_cells, Ngy, Ngx, hscale, nstart, nend):
    """
    Compute GC taper weights for grid points [nstart..nend] vs all obs.

    Parameters
    ----------
    obs_cells : (nobs,) int  — flat indices into (Ngy, Ngx)
    hscale : float — GC taper cutoff in grid points
    nstart, nend : int — grid point range (inclusive)

    Returns
    -------
    covlocal_local : (nobs, nlocal) float32
    """
    nlocal = nend - nstart + 1
    nobs = len(obs_cells)

    # Grid positions
    obs_iy, obs_ix = np.unravel_index(obs_cells, (Ngy, Ngx))
    local_cells = np.arange(nstart, nend + 1)
    loc_iy, loc_ix = np.unravel_index(local_cells, (Ngy, Ngx))

    # Pairwise Euclidean distance (nobs, nlocal)
    dy = obs_iy[:, None].astype(float) - loc_iy[None, :].astype(float)
    dx = obs_ix[:, None].astype(float) - loc_ix[None, :].astype(float)
    dists = np.sqrt(dx**2 + dy**2)

    covlocal = gaspcohn(dists / hscale).astype(np.float32)
    return covlocal


# ===================================================================
#  LETKF update for linear Gaussian model (single field, no MPI)
# ===================================================================
def letkf_update(xens, obs_values, obs_indices, obs_errvar,
                 covlocal, ncells):
    """
    LETKF analysis update for single-field model.

    Parameters
    ----------
    xens : (K, d)
    obs_values : (nobs,)
    obs_indices : (nobs,) int  — flat indices into state vector
    obs_errvar : (nobs,)
    covlocal : (nobs, ncells)  — localisation weights for ALL grid points
    ncells : int

    Returns
    -------
    xens_updated : (K, d)
    """
    nanals = xens.shape[0]
    nobs = len(obs_values)

    hxens = xens[:, obs_indices]
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    innovations = obs_values - hxmean

    xmean = xens.mean(axis=0)
    xprime = xens - xmean

    inv_obs_errvar = 1.0 / obs_errvar
    xens_updated = xens.copy()

    for n in range(ncells):
        loc_wt = covlocal[:, n]
        mask = loc_wt > 1.e-10
        nobs_local = int(mask.sum())
        if nobs_local == 0:
            continue

        rinv_diag = loc_wt[mask] * inv_obs_errvar[mask]
        hx_local = hxprime[:, mask]
        innov_local = innovations[mask]

        wts = calcwts_letkf(hx_local, rinv_diag, innov_local, nanals)
        xens_updated[:, n] = xmean[n] + xprime[:, n] @ wts

    return xens_updated


# ===================================================================
#  Run one LETKF experiment
# ===================================================================
def run_letkf_single(data_file, kf_file, K, hscale, covinflate1, seed=42):
    """
    Run LETKF with given parameters and return RMSE vs KF.

    Returns
    -------
    rmse : float   — time-averaged RMSE(LETKF_mean, KF_mean)
    pct : float    — percentage of |errors| < sigma_y/2
    elapsed : float
    """
    np.random.seed(seed)

    dat = np.load(data_file)
    T = int(dat['T']); d = int(dat['d'])
    Ngx = int(dat['Ngx']); Ngy = int(dat['Ngy'])
    sigma_z = float(dat['sigma_z']); sigma_y = float(dat['sigma_y'])
    a_coeff = float(dat['a_coeff']); Z0 = dat['Z0']
    obs_inds_arr = dat['obs_inds']; y_obs_arr = dat['y_obs']
    nobs_arr = dat['nobs']

    kf_dat = np.load(kf_file)
    kf_mean = kf_dat['kf_mean']

    # Initialise ensemble by perturbing Z0
    xens = np.tile(Z0, (K, 1)) + 0.01 * np.random.randn(K, d)

    letkf_mean = np.zeros((T + 1, d))
    letkf_mean[0] = Z0.copy()

    obs_errvar = sigma_y ** 2

    t0 = time.time()

    for k in range(T):
        # Forecast
        for j in range(K):
            xens[j] = a_coeff * xens[j] + sigma_z * np.random.randn(d)

        # Covariance inflation / deflation
        if covinflate1 != 1.0:
            xmean = xens.mean(axis=0)
            xens = xmean + covinflate1 * (xens - xmean)

        # Get obs
        mk = int(nobs_arr[k])
        obs_idx = obs_inds_arr[k, :mk].astype(int)
        y_k = y_obs_arr[k, :mk]

        if mk == 0:
            letkf_mean[k + 1] = xens.mean(axis=0)
            continue

        # Localisation
        covlocal = precompute_covlocal_euclidean(
            obs_idx, Ngy, Ngx, hscale, 0, d - 1)
        obs_errvar_vec = np.full(mk, obs_errvar)

        # Update
        xens = letkf_update(xens, y_k, obs_idx, obs_errvar_vec,
                            covlocal, d)
        letkf_mean[k + 1] = xens.mean(axis=0)

    elapsed = time.time() - t0

    # Compute metrics vs KF
    abs_errors = np.abs(letkf_mean[1:] - kf_mean[1:])  # (T, d)
    rmse = np.sqrt(np.mean((letkf_mean[1:] - kf_mean[1:])**2))
    pct = 100.0 * np.mean(abs_errors < sigma_y / 2.0)

    return rmse, pct, elapsed


# ===================================================================
#  Main: sensitivity grid search
# ===================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--kf", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--K", type=int, default=50,
                        help="Ensemble size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    basedir = os.path.dirname(os.path.abspath(__file__))
    if args.outdir is None:
        args.outdir = basedir
    if args.data is None:
        args.data = os.path.join(basedir, "linear_gaussian_data.npz")
    if args.kf is None:
        args.kf = os.path.join(basedir, "linear_gaussian_kf.npz")

    # Sensitivity grid
    hscale_values = [0.5, 1, 1.5, 2, 3, 5, 8, 10, 15, 20]
    covinflate_values = [0.50, 0.70, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10]

    results = []
    best_rmse = np.inf
    best_params = {}

    print(f"[LETKF Sensitivity] K={args.K}")
    print(f"  hscale values: {hscale_values}")
    print(f"  covinflate values: {covinflate_values}")
    print(f"  Total experiments: {len(hscale_values) * len(covinflate_values)}")

    for hs in hscale_values:
        for ci in covinflate_values:
            print(f"\n  Running hscale={hs}, covinflate1={ci:.2f} ...", end="")
            try:
                rmse, pct, elapsed = run_letkf_single(
                    args.data, args.kf, args.K, hs, ci, seed=args.seed)
                print(f"  RMSE={rmse:.6f}  pct={pct:.2f}%  time={elapsed:.1f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
                rmse, pct, elapsed = np.nan, 0.0, 0.0

            entry = {
                'hscale': hs, 'covinflate1': ci,
                'K': args.K,
                'rmse': float(rmse), 'pct_lt_sigy2': float(pct),
                'elapsed': float(elapsed),
            }
            results.append(entry)

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = entry.copy()

    # Save results
    outfile = os.path.join(args.outdir, "letkf_sensitivity_results.json")
    with open(outfile, 'w') as f:
        json.dump({'results': results, 'best': best_params}, f, indent=2)
    print(f"\n[LETKF Sensitivity] Best: hscale={best_params['hscale']}, "
          f"covinflate1={best_params['covinflate1']:.2f}, "
          f"RMSE={best_params['rmse']:.6f}, pct={best_params['pct_lt_sigy2']:.2f}%")
    print(f"[LETKF Sensitivity] Saved to {outfile}")

    # ---- Plot heatmap ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        nh = len(hscale_values)
        nc = len(covinflate_values)
        rmse_grid = np.full((nh, nc), np.nan)
        pct_grid = np.full((nh, nc), np.nan)

        for entry in results:
            i = hscale_values.index(entry['hscale'])
            j = covinflate_values.index(entry['covinflate1'])
            rmse_grid[i, j] = entry['rmse']
            pct_grid[i, j] = entry['pct_lt_sigy2']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # RMSE heatmap
        ax = axes[0]
        im = ax.imshow(rmse_grid, origin='lower', aspect='auto',
                       cmap='viridis_r')
        ax.set_xticks(range(nc))
        ax.set_xticklabels([f"{v:.2f}" for v in covinflate_values])
        ax.set_yticks(range(nh))
        ax.set_yticklabels([str(v) for v in hscale_values])
        ax.set_xlabel('Covariance inflation')
        ax.set_ylabel('Localization scale (hscale)')
        ax.set_title(f'RMSE vs KF (K={args.K})')
        plt.colorbar(im, ax=ax, label='RMSE')
        # Annotate
        for i in range(nh):
            for j in range(nc):
                if np.isfinite(rmse_grid[i, j]):
                    ax.text(j, i, f"{rmse_grid[i,j]:.4f}",
                            ha='center', va='center', fontsize=6,
                            color='white' if rmse_grid[i,j] > np.nanmedian(rmse_grid) else 'black')

        # Pct heatmap
        ax = axes[1]
        im2 = ax.imshow(pct_grid, origin='lower', aspect='auto',
                        cmap='RdYlGn')
        ax.set_xticks(range(nc))
        ax.set_xticklabels([f"{v:.2f}" for v in covinflate_values])
        ax.set_yticks(range(nh))
        ax.set_yticklabels([str(v) for v in hscale_values])
        ax.set_xlabel('Covariance inflation')
        ax.set_ylabel('Localization scale (hscale)')
        ax.set_title(f'% of |errors| < σ_y/2 (K={args.K})')
        plt.colorbar(im2, ax=ax, label='%')
        for i in range(nh):
            for j in range(nc):
                if np.isfinite(pct_grid[i, j]):
                    ax.text(j, i, f"{pct_grid[i,j]:.1f}",
                            ha='center', va='center', fontsize=6)

        plt.tight_layout()
        figpath = os.path.join(args.outdir, "letkf_sensitivity_heatmap.png")
        plt.savefig(figpath, dpi=150)
        plt.close()
        print(f"[LETKF Sensitivity] Heatmap saved to {figpath}")

    except ImportError:
        print("[LETKF Sensitivity] matplotlib not available, skipping heatmap.")


if __name__ == "__main__":
    main()

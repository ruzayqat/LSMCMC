#!/usr/bin/env python
"""
run_enkf.py
===========
Ensemble Kalman Filter (EnKF) for the linear Gaussian model.

Uses the Sherman-Morrison-Woodbury formula when m_k > K (ensemble size)
as described in the paper.  When m_k <= K, uses standard EnKF update.

Settings from the paper:
  Test 1: K=1200, ~145s
  Test 2: K=5000, ~486s
"""
import os
import sys
import time
import numpy as np


def enkf_update(xens, y_k, obs_idx, sigma_y):
    """
    EnKF stochastic update with perturbed observations.
    Uses Woodbury identity when nobs > K.

    Parameters
    ----------
    xens : (K, d)
    y_k : (m,)
    obs_idx : (m,) int
    sigma_y : float

    Returns
    -------
    xens_updated : (K, d)
    """
    K, d = xens.shape
    m = len(y_k)
    R = sigma_y ** 2

    xmean = xens.mean(axis=0)
    X = xens - xmean                             # (K, d) perturbations

    # H(x) = x[obs_idx]
    HX = X[:, obs_idx]                            # (K, m)
    Hxm = xmean[obs_idx]                          # (m,)

    # Perturbed observations
    Y_pert = y_k[None, :] + sigma_y * np.random.randn(K, m)  # (K, m)
    D = Y_pert - xens[:, obs_idx]                 # (K, m) innovations

    if m <= K:
        # Standard: S = (1/(K-1)) HX^T HX + R*I,  K_gain = X^T HX S^{-1} / (K-1)
        S = (1.0 / (K - 1)) * HX.T @ HX + R * np.eye(m)
        SinvD = np.linalg.solve(S, D.T).T         # (K, m)
        # Update: x_a = x_f + (1/(K-1)) X^T (HX) S^{-1} D  ... but per member
        # More standard form:
        # K_gain (d, m) = (1/(K-1)) X^T HX S^{-1}
        K_gain = (1.0 / (K - 1)) * X.T @ HX @ np.linalg.inv(S)
        xens_updated = xens + D @ K_gain.T
    else:
        # Woodbury: avoid inverting m×m, use K×K instead
        # K_gain = X^T (HX) [R*(K-1)*I + HX^T HX]^{-1}  / (K-1)
        # But using Woodbury on  P H^T (H P H^T + R)^{-1}
        # with P = (1/(K-1)) X^T X, we get:
        # (H P H^T + R I)^{-1} = R^{-1} I - R^{-1} HX [R(K-1)I + HX^T HX]^{-1} HX^T R^{-1}
        # Simpler: work in ensemble space
        # C = HX^T HX / (K-1) + R I   (m×m) — too big
        # Instead: Tippett form: work with K×K matrix
        A = (1.0 / (K - 1)) * HX @ HX.T + R * np.eye(K)  # (K, K)
        AinvD_HX = np.linalg.solve(A, D @ HX.T.T)  # wrong dims...

        # Actually let's just use the direct form with solve
        # S (m,m), solve S x = D^T
        S = (1.0 / (K - 1)) * HX.T @ HX + R * np.eye(m)
        SinvDT = np.linalg.solve(S, D.T)            # (m, K)
        K_gain = (1.0 / (K - 1)) * X.T @ HX         # (d, m)
        xens_updated = xens + (K_gain @ SinvDT).T    # (K, d)

    return xens_updated


def run_enkf(data_file=None, outdir=None, seed=42, K=1200):
    """
    Run EnKF and save analysis mean trajectory.
    """
    np.random.seed(seed)

    basedir = os.path.dirname(os.path.abspath(__file__))
    if data_file is None:
        data_file = os.path.join(basedir, "linear_gaussian_data.npz")
    if outdir is None:
        outdir = basedir
    os.makedirs(outdir, exist_ok=True)

    dat = np.load(data_file)
    T = int(dat['T']); d = int(dat['d'])
    sigma_z = float(dat['sigma_z']); sigma_y = float(dat['sigma_y'])
    a_coeff = float(dat['a_coeff']); Z0 = dat['Z0']
    obs_inds_arr = dat['obs_inds']; y_obs_arr = dat['y_obs']
    nobs_arr = dat['nobs']

    print(f"[EnKF] d={d}, T={T}, K={K}")

    # Initialise
    xens = np.tile(Z0, (K, 1)) + 0.01 * np.random.randn(K, d)

    enkf_mean = np.zeros((T + 1, d))
    enkf_mean[0] = Z0.copy()

    t0 = time.time()

    for k in range(T):
        t_step = time.time()

        # Forecast
        for j in range(K):
            xens[j] = a_coeff * xens[j] + sigma_z * np.random.randn(d)

        t_fcst = time.time() - t_step

        # Get obs
        mk = int(nobs_arr[k])
        obs_idx = obs_inds_arr[k, :mk].astype(int)
        y_k = y_obs_arr[k, :mk]

        if mk == 0:
            enkf_mean[k + 1] = xens.mean(axis=0)
            continue

        # EnKF update
        xens = enkf_update(xens, y_k, obs_idx, sigma_y)
        enkf_mean[k + 1] = xens.mean(axis=0)

        t_da = time.time() - t_step - t_fcst

        if (k + 1) % 10 == 0 or k == 0:
            elapsed = time.time() - t0
            print(f"  Cycle {k+1:3d}/{T}  nobs={mk:4d}  "
                  f"fcst={t_fcst:.2f}s  DA={t_da:.2f}s  "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"[EnKF] Done in {elapsed:.1f}s")

    outfile = os.path.join(outdir, "linear_gaussian_enkf.npz")
    np.savez(outfile, enkf_mean=enkf_mean, T=T, d=d,
             K=K, elapsed=elapsed)
    print(f"[EnKF] Saved to {outfile}")
    return outfile


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    run_enkf(data_file=args.data, outdir=args.outdir,
             seed=args.seed, K=args.K)


if __name__ == "__main__":
    main()

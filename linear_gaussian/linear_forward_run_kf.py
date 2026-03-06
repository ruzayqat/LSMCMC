#!/usr/bin/env python
"""
run_kf.py
=========
Kalman Filter for the linear Gaussian model:

    Z_{k+1} = A * Z_k + sigma_z * W_{k+1}
    Y_k     = C_k * Z_k + sigma_y * V_k

Since A = a*I and Q = sigma_z^2 * I  the KF forecast is trivially:
    m_{k|k-1} = a * m_{k-1|k-1}
    P_{k|k-1} = a^2 * P_{k-1|k-1} + sigma_z^2 * I

The update uses C_k (selection matrix) and R_k = sigma_y^2 * I_{m_k}.
Because C_k selects a subset of coordinates:
    S_k = C_k P_{k|k-1} C_k^T + R_k  =  P_obs_obs + sigma_y^2 I_{m_k}
    K_k = P_{k|k-1} C_k^T S_k^{-1}
    m_{k|k}   = m_{k|k-1} + K_k (y_k - C_k m_{k|k-1})
    P_{k|k}   = P_{k|k-1} - K_k S_k K_k^T

Key optimisation: P is always diagonal at the start (P0=0), and after forecast
P stays diagonal. But the update P - K S K^T makes it dense IF C selects a
large fraction of states. For d=14400 and ~700-1400 obs per cycle, the
matrix S is ~m_k x m_k which is manageable. We only need to invert/solve
S and store P (d x d ~ 14400^2 ≈ 1.6 GB if float64). This is feasible.

To reduce memory, we exploit the structure: P is symmetric so we store full.
We also note that P will become dense after the first update, so we cannot
stay diagonal throughout.

Output: linear_gaussian_kf.npz  (KF mean trajectory)
"""
import os
import sys
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def main(data_file=None, outdir=None):
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "linear_gaussian_data.npz")
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)

    # ---- Load data ----
    dat = np.load(data_file)
    T = int(dat['T'])
    d = int(dat['d'])
    sigma_z = float(dat['sigma_z'])
    sigma_y = float(dat['sigma_y'])
    a_coeff = float(dat['a_coeff'])
    Z0 = dat['Z0']
    obs_inds_arr = dat['obs_inds']    # (T, max_nobs), -1 padded
    y_obs_arr = dat['y_obs']          # (T, max_nobs), NaN padded
    nobs_arr = dat['nobs']            # (T,)

    print(f"[KF] d={d}, T={T}, sigma_z={sigma_z}, sigma_y={sigma_y}, a={a_coeff}")
    print(f"[KF] Mean nobs per cycle: {np.mean(nobs_arr):.1f}")

    # ---- Initialise KF ----
    m = Z0.copy()                                        # (d,)
    P = np.zeros((d, d), dtype=np.float64)               # initial covariance = 0

    # Store KF means
    kf_mean = np.zeros((T + 1, d), dtype=np.float64)
    kf_mean[0] = m.copy()

    a2 = a_coeff ** 2
    sz2 = sigma_z ** 2
    sy2 = sigma_y ** 2

    t0 = time.time()

    for k in range(T):
        # ---- Forecast ----
        m = a_coeff * m
        P = a2 * P
        P[np.diag_indices(d)] += sz2    # P += sigma_z^2 * I

        # ---- Get observations ----
        mk = int(nobs_arr[k])
        obs_idx = obs_inds_arr[k, :mk].astype(int)
        y_k = y_obs_arr[k, :mk]

        if mk == 0:
            kf_mean[k + 1] = m.copy()
            continue

        # ---- Update ----
        # C_k selects rows obs_idx from P
        # S = P[obs_idx][:, obs_idx] + sigma_y^2 I
        P_obs = P[np.ix_(obs_idx, obs_idx)]               # (mk, mk)
        S = P_obs + sy2 * np.eye(mk)

        # Innovation
        innov = y_k - m[obs_idx]                           # (mk,)

        # Solve S^{-1} * innov  and  S^{-1} * (P C^T)^T  via Cholesky
        cho = cho_factor(S)

        # K = P[:, obs_idx] @ S^{-1}   — but compute via solve
        P_cross = P[:, obs_idx]                            # (d, mk)
        S_inv_innov = cho_solve(cho, innov)                # (mk,)
        S_inv_Pcross_T = cho_solve(cho, P_cross.T)         # (mk, d)

        # m_update = m + P_cross @ S^{-1} innov
        m = m + P_cross @ S_inv_innov

        # P_update = P - P_cross @ S^{-1} @ P_cross^T
        P = P - P_cross @ S_inv_Pcross_T

        # Symmetrise (numerical)
        P = 0.5 * (P + P.T)

        kf_mean[k + 1] = m.copy()

        if (k + 1) % 10 == 0 or k == 0:
            elapsed = time.time() - t0
            print(f"  KF cycle {k+1:3d}/{T}  nobs={mk:4d}  "
                  f"trace(P)={np.trace(P):.4e}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"[KF] Done in {elapsed:.1f}s")

    # ---- Save ----
    outfile = os.path.join(outdir, "linear_gaussian_kf.npz")
    np.savez(outfile, kf_mean=kf_mean, T=T, d=d,
             sigma_z=sigma_z, sigma_y=sigma_y, a_coeff=a_coeff,
             elapsed=elapsed)
    print(f"[KF] Saved to {outfile}")

    return outfile


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
generate_data.py
================
Generate synthetic truth trajectory and SWOT-like swath observations for the
linear Gaussian model:

    Z_{k+1} = A * Z_k + sigma_z * W_{k+1},   W ~ N(0, I_d)
    Y_k     = C_k * Z_k + sigma_y * V_k,      V ~ N(0, I_{m_k})

Settings (from the paper):
    T  = 100        assimilation cycles
    Ngx = Ngy = 120   => d = 14400
    sigma_z = sigma_y = 0.05
    A = 0.25 * I_d
    Z_0^{(j)} ~ -0.15 * U[0,1]  for j <= floor(d/3), else 0

Observations: SWOT-like dual swath, width 7 grid points, alternating angle,
moving east-to-west cyclically over 20 cycles.

Output: linear_gaussian_data.npz
"""
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
#  Import swath observation generator (local copy)
# ---------------------------------------------------------------------------
from generate_swath_observations import generate_swath_observations


def main(seed=42, outdir=None):
    np.random.seed(seed)
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)

    # ---- Model parameters ----
    T = 100                # number of assimilation cycles
    Ngx = 120
    Ngy = 120
    d = Ngx * Ngy          # 14400

    sigma_z = 0.05
    sigma_y = 0.05
    a_coeff = 0.25         # A = a_coeff * I_d

    # ---- Initial condition ----
    Z0 = np.zeros(d)
    n_nonzero = d // 3
    Z0[:n_nonzero] = -0.15 * np.random.rand(n_nonzero)

    # ---- Grid for observation generator ----
    # generate_swath_observations expects 2D arrays x(ny,nx), y(ny,nx)
    # and domain lengths Lx, Ly.  We use grid-point units: Lx = Ngx, Ly = Ngy,
    # so that swath_width=7 means 7 grid points.
    xv = np.linspace(0.5, Ngx - 0.5, Ngx)   # cell-centre coords
    yv = np.linspace(0.5, Ngy - 0.5, Ngy)
    xg, yg = np.meshgrid(xv, yv)             # shape (Ngy, Ngx)
    Lx = float(Ngx)
    Ly = float(Ngy)

    # Swath parameters (in grid-point units)
    swath_width = 7.0
    gap_width = 2.0       # nadir gap between the two swaths
    cycles_to_cross = 20  # cycles to traverse domain once

    # ---- Forward run: generate truth trajectory ----
    Z_truth = np.zeros((T + 1, d))
    Z_truth[0] = Z0.copy()

    for k in range(T):
        Z_truth[k + 1] = a_coeff * Z_truth[k] + sigma_z * np.random.randn(d)

    # ---- Generate observations at every cycle ----
    # obs_inds_all[k] contains the flat indices into (Ngy, Ngx) grid
    # y_obs_all[k] contains the observation values
    obs_inds_list = []
    y_obs_list = []
    nobs_list = []

    for k in range(1, T + 1):
        obs_inds, _, _, nobs = generate_swath_observations(
            xg, yg, Lx, Ly,
            frame=k - 1,
            nassim=T,
            swath_width=swath_width,
            gap_width=gap_width,
            fixed_nobs=None,
            cycles_to_cross=cycles_to_cross,
            oscillate=False,
            jitter=0.0,
            angle_even=70,
            angle_odd=110,
        )
        # Observation values: Y_k = C_k * Z_k + sigma_y * V_k
        z_true_k = Z_truth[k]
        y_k = z_true_k[obs_inds] + sigma_y * np.random.randn(len(obs_inds))

        obs_inds_list.append(obs_inds)
        y_obs_list.append(y_k)
        nobs_list.append(len(obs_inds))

    # ---- Pad to rectangular arrays for saving ----
    max_nobs = max(nobs_list)
    obs_inds_arr = -np.ones((T, max_nobs), dtype=np.int32)
    y_obs_arr = np.full((T, max_nobs), np.nan, dtype=np.float64)

    for k in range(T):
        n = nobs_list[k]
        obs_inds_arr[k, :n] = obs_inds_list[k]
        y_obs_arr[k, :n] = y_obs_list[k]

    nobs_arr = np.array(nobs_list, dtype=np.int32)

    # ---- Save ----
    outfile = os.path.join(outdir, "linear_gaussian_data.npz")
    np.savez(outfile,
             Z_truth=Z_truth,
             Z0=Z0,
             obs_inds=obs_inds_arr,
             y_obs=y_obs_arr,
             nobs=nobs_arr,
             T=T, Ngx=Ngx, Ngy=Ngy, d=d,
             sigma_z=sigma_z, sigma_y=sigma_y,
             a_coeff=a_coeff)

    print(f"[generate_data] Saved to {outfile}")
    print(f"  T={T}, d={d}, max_nobs={max_nobs}, "
          f"mean_nobs={np.mean(nobs_arr):.1f}")

    return outfile


if __name__ == "__main__":
    main()

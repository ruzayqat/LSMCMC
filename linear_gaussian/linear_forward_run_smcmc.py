#!/usr/bin/env python
"""
run_smcmc.py
============
SMCMC (NO localization) for the linear Gaussian model.

This is the plain SMCMC: no domain partitioning, the full state vector is
sampled jointly. This is included for comparison with LSMCMC.

Settings from the paper:
  Test 1: N=1000, burn_in=700, M=52, ~145s
  Test 2: N=5000, burn_in=3000, M=52, ~766s
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loc_smcmc_swe_exact_from_Gauss import (
    sample_posterior_mixture_sparse,
    gaussian_block_means,
    build_H_loc_from_global,
)


def run_smcmc(data_file=None, outdir=None, seed=42, Na=1000, Nf=52):
    """
    Run one SMCMC simulation (no localization).
    """
    np.random.seed(seed)

    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "linear_gaussian_data.npz")
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)

    # ---- Load data ----
    dat = np.load(data_file)
    T = int(dat['T']); d = int(dat['d'])
    Ngx = int(dat['Ngx']); Ngy = int(dat['Ngy'])
    sigma_z = float(dat['sigma_z']); sigma_y = float(dat['sigma_y'])
    a_coeff = float(dat['a_coeff']); Z0 = dat['Z0']
    obs_inds_arr = dat['obs_inds']; y_obs_arr = dat['y_obs']
    nobs_arr = dat['nobs']

    print(f"[SMCMC] d={d}, T={T}, Na={Na}, Nf={Nf}")

    # ---- Initialise ensemble (full state) ----
    forecast = np.tile(Z0[:, None], (1, Nf))  # (d, Nf)

    smcmc_mean = np.zeros((T + 1, d), dtype=np.float64)
    smcmc_mean[0] = Z0.copy()

    t0 = time.time()

    for k in range(T):
        t_step = time.time()

        # Forecast
        for j in range(Nf):
            forecast[:, j] = a_coeff * forecast[:, j] + \
                             sigma_z * np.random.randn(d)
        t_fcst = time.time() - t_step

        forecast_mean = np.mean(forecast, axis=1)

        # Get obs
        mk = int(nobs_arr[k])
        obs_idx = obs_inds_arr[k, :mk].astype(int)
        y_k = y_obs_arr[k, :mk]

        if mk == 0:
            smcmc_mean[k + 1] = forecast_mean
            continue

        # Build H for full state
        all_cells = np.arange(d)
        H_full = build_H_loc_from_global(obs_idx, all_cells)

        # Exact posterior sampling — full state
        m_list = forecast.T  # (Nf, d)
        anal_samples, mu = sample_posterior_mixture_sparse(
            m_list, H_full, sigma_z, sigma_y, y_k,
            n_samples=Na,
        )

        # Reduce to Nf
        anal_reduced, _ = gaussian_block_means(anal_samples, Nf)

        t_da = time.time() - t_step - t_fcst

        smcmc_mean[k + 1] = mu
        forecast = anal_reduced

        if (k + 1) % 10 == 0 or k == 0:
            elapsed = time.time() - t0
            print(f"  Cycle {k+1:3d}/{T}  nobs={mk:4d}  "
                  f"fcst={t_fcst:.2f}s  DA={t_da:.2f}s  "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"[SMCMC] Done in {elapsed:.1f}s")

    outfile = os.path.join(outdir, "linear_gaussian_smcmc.npz")
    np.savez(outfile, smcmc_mean=smcmc_mean, T=T, d=d,
             Na=Na, Nf=Nf, elapsed=elapsed)
    print(f"[SMCMC] Saved to {outfile}")
    return outfile


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--Na", type=int, default=1000)
    parser.add_argument("--Nf", type=int, default=52)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.dirname(os.path.abspath(__file__))
    if args.data is None:
        args.data = os.path.join(args.outdir, "linear_gaussian_data.npz")

    dat = np.load(args.data)
    T = int(dat['T']); d = int(dat['d'])

    all_means = np.zeros((args.M, T + 1, d), dtype=np.float64)
    all_elapsed = np.zeros(args.M)

    for isim in range(args.M):
        print(f"\n====== SMCMC run {isim+1}/{args.M} ======")
        seed = 3000 + isim
        outfile = run_smcmc(
            data_file=args.data, outdir=args.outdir, seed=seed,
            Na=args.Na, Nf=args.Nf)
        res = np.load(outfile)
        all_means[isim] = res['smcmc_mean']
        all_elapsed[isim] = float(res['elapsed'])

    avg_mean = np.mean(all_means, axis=0)
    avg_elapsed = np.mean(all_elapsed)

    outfile_avg = os.path.join(args.outdir, "linear_gaussian_smcmc_avg.npz")
    np.savez(outfile_avg,
             smcmc_mean=avg_mean,
             all_means=all_means,
             T=T, d=d,
             M=args.M, Na=args.Na, Nf=args.Nf,
             elapsed_mean=avg_elapsed, elapsed_all=all_elapsed)
    print(f"\n[SMCMC] Averaged over M={args.M} runs, "
          f"mean time={avg_elapsed:.1f}s")
    print(f"[SMCMC] Saved to {outfile_avg}")


if __name__ == "__main__":
    main()

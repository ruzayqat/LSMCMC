#!/usr/bin/env python
"""
run_lsmcmc_v1.py
================
V1 LSMCMC (joint observed-block localization) for the linear Gaussian model.

The domain (120 x 120 = 14400) is partitioned into Gamma = 900 blocks.
At each assimilation cycle:
  1. Forecast:  m_j = a * m_j + sigma_z * W  for j=1..Nf
  2. Find which blocks contain observations
  3. Build local state (union of all cells in observed blocks)
  4. Build sparse H mapping local state -> obs
  5. Sample exactly from the Gaussian mixture posterior
  6. Reduce Na samples to Nf via Gaussian block means
  7. Unobserved blocks keep the forecast mean

Settings from the paper (Test 1):
    Gamma = 900, N = 500, N_burn = 300, M = 50 (Nf = 52)
"""
import os
import sys
import time
import numpy as np
import scipy.sparse as sp

# Import from existing codebase
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loc_smcmc_swe_exact_from_Gauss import (
    partition_domain,
    build_H_loc_from_global,
    sample_posterior_mixture_sparse,
    gaussian_block_means,
)


def run_lsmcmc_v1(data_file=None, outdir=None, seed=42,
                   Gamma=900, Na=2000, burn_in=300, Nf=52):
    """
    Run one LSMCMC V1 simulation.

    Parameters
    ----------
    Gamma : int
        Number of blocks for domain partitioning.
    Na : int
        Number of MCMC (posterior) samples.
    burn_in : int
        Number of burn-in samples (not used in exact sampling, kept for
        consistency — we draw Na + burn_in then discard first burn_in).
        For exact sampling from Gaussian mixture, burn_in is irrelevant.
        We simply draw Na samples and reduce to Nf via block means.
    Nf : int
        Number of forecast ensemble members (= M + 2 overhead or just M).
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
    T = int(dat['T'])
    d = int(dat['d'])
    Ngx = int(dat['Ngx'])
    Ngy = int(dat['Ngy'])
    sigma_z = float(dat['sigma_z'])
    sigma_y = float(dat['sigma_y'])
    a_coeff = float(dat['a_coeff'])
    Z0 = dat['Z0']
    obs_inds_arr = dat['obs_inds']
    y_obs_arr = dat['y_obs']
    nobs_arr = dat['nobs']

    print(f"[LSMCMC-V1] d={d}, T={T}, Gamma={Gamma}, Na={Na}, Nf={Nf}")

    # ---- Load KF reference for per-cycle RMSE ----
    kf_file = os.path.join(os.path.dirname(data_file), "linear_gaussian_kf.npz")
    kf_mean = None
    if os.path.exists(kf_file):
        kf_mean = np.load(kf_file)['kf_mean']
        print(f"[LSMCMC-V1] Loaded KF reference from {kf_file}")
    else:
        print(f"[LSMCMC-V1] KF file not found at {kf_file}, skipping per-cycle RMSE.")

    # ---- Partition domain ----
    partitions, partition_labels, Gamma_actual, nby, nbx, bh, bw = \
        partition_domain(Ngy, Ngx, N=Gamma)
    print(f"[LSMCMC-V1] Partition: {nby}x{nbx} blocks of {bh}x{bw}, "
          f"Gamma={Gamma_actual}")

    # ---- Initialise ensemble ----
    # forecast[:, j] is the j-th ensemble member (d,)
    forecast = np.tile(Z0[:, None], (1, Nf))  # (d, Nf)

    # Storage
    lsmcmc_mean = np.zeros((T + 1, d), dtype=np.float64)
    lsmcmc_mean[0] = Z0.copy()

    t0 = time.time()

    for k in range(T):
        t_step = time.time()

        # ---- Forecast all members ----
        for j in range(Nf):
            forecast[:, j] = a_coeff * forecast[:, j] + \
                             sigma_z * np.random.randn(d)
        t_fcst = time.time() - t_step

        forecast_mean = np.mean(forecast, axis=1)

        # ---- Get observations ----
        mk = int(nobs_arr[k])
        obs_idx = obs_inds_arr[k, :mk].astype(int)
        y_k = y_obs_arr[k, :mk]

        if mk == 0:
            lsmcmc_mean[k + 1] = forecast_mean
            continue

        # ---- Find observed blocks and their cells ----
        obs_rows, obs_cols = np.unravel_index(obs_idx, (Ngy, Ngx))
        obs_block_ids = np.unique(partition_labels[obs_rows, obs_cols])
        mask = np.isin(partition_labels, obs_block_ids)
        ij = np.argwhere(mask)
        cells_flat = np.ravel_multi_index((ij[:, 0], ij[:, 1]), (Ngy, Ngx))
        cells_flat = np.sort(cells_flat)

        # ---- Build sparse H ----
        H_loc = build_H_loc_from_global(obs_idx, cells_flat)

        # ---- Exact posterior sampling ----
        # m_list: (Nf, d_local)  — forecast ensemble restricted to local cells
        m_list = forecast[cells_flat].T   # (Nf, d_local)

        # Per-obs y restricted to mapped obs
        # build_H_loc_from_global may drop obs not in cells — so
        # we need to figure out which obs are kept
        col_set = set(cells_flat.tolist())
        kept_mask = np.array([int(g) in col_set for g in obs_idx], dtype=bool)
        y_loc = y_k[kept_mask]

        anal_samples, mu_loc = sample_posterior_mixture_sparse(
            m_list, H_loc, sigma_z, sigma_y, y_loc,
            n_samples=Nf,
        )

        # Use posterior samples directly as new ensemble (no block-mean compression)
        anal_reduced = anal_samples

        t_da = time.time() - t_step - t_fcst

        # ---- Update stored state ----
        lsmcmc_mean[k + 1] = forecast_mean.copy()
        lsmcmc_mean[k + 1, cells_flat] = mu_loc

        # Update forecast ensemble for observed cells
        forecast[cells_flat, :] = anal_reduced

        if (k + 1) % 10 == 0 or k == 0:
            elapsed = time.time() - t0
            rmse_str = ""
            if kf_mean is not None:
                rmse_k = np.sqrt(np.mean((lsmcmc_mean[k + 1] - kf_mean[k + 1]) ** 2))
                rmse_str = f"  RMSE={rmse_k:.6f}"
            print(f"  Cycle {k+1:3d}/{T}  nobs={mk:4d}  "
                  f"d_local={len(cells_flat):5d}  "
                  f"fcst={t_fcst:.2f}s  DA={t_da:.2f}s  "
                  f"elapsed={elapsed:.1f}s{rmse_str}")
        else:
            if kf_mean is not None:
                rmse_k = np.sqrt(np.mean((lsmcmc_mean[k + 1] - kf_mean[k + 1]) ** 2))
                print(f"  Cycle {k+1:3d}/{T}  RMSE={rmse_k:.6f}")

    elapsed = time.time() - t0
    print(f"[LSMCMC-V1] Done in {elapsed:.1f}s")

    # ---- Compute & print RMSE vs KF ----
    if kf_mean is not None:
        rmse_per_cycle = np.sqrt(np.mean((lsmcmc_mean - kf_mean) ** 2, axis=1))
        rmse_overall = np.mean(rmse_per_cycle[1:])  # skip t=0
        print(f"[LSMCMC-V1] RMSE vs KF (per-cycle mean, excluding t=0): {rmse_overall:.6f}")
    else:
        print(f"[LSMCMC-V1] KF file not found, skipping RMSE.")

    # ---- Save ----
    outfile = os.path.join(outdir, "linear_gaussian_lsmcmc_v1.npz")
    np.savez(outfile, lsmcmc_mean=lsmcmc_mean, T=T, d=d,
             Gamma=Gamma_actual, Na=Na, Nf=Nf,
             elapsed=elapsed)
    print(f"[LSMCMC-V1] Saved to {outfile}")
    return outfile, lsmcmc_mean, elapsed


def _worker_v1(args_tuple):
    """Picklable worker for parallel V1 runs."""
    isim, data_file, outdir, Gamma, Na, burn_in, Nf = args_tuple
    seed = 1000 + isim          # unique seed per run
    # Each worker saves to a unique sub-directory to avoid file corruption
    sim_outdir = os.path.join(outdir, f"sim_{isim:03d}")
    os.makedirs(sim_outdir, exist_ok=True)
    print(f"\n====== LSMCMC-V1 run {isim+1} (seed={seed}) ======", flush=True)
    _outfile, mean_arr, elapsed = run_lsmcmc_v1(
        data_file=data_file, outdir=sim_outdir, seed=seed,
        Gamma=Gamma, Na=Na, burn_in=burn_in, Nf=Nf)
    return isim, mean_arr.copy(), float(elapsed)


def _load_yaml_config(path):
    """Load a YAML config file and return as dict."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    """Run M independent simulations (in parallel) and average."""
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--M", type=int, default=None, help="Independent runs")
    parser.add_argument("--Gamma", type=int, default=None)
    parser.add_argument("--Na", type=int, default=None)
    parser.add_argument("--burn_in", type=int, default=None)
    parser.add_argument("--Nf", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (0 = serial)")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    # Defaults
    defaults = dict(M=52, Gamma=900, Na=500, burn_in=300, Nf=52,
                    workers=52, data=None, outdir=None)
    # Load YAML config if provided
    if args.config:
        cfg = _load_yaml_config(args.config)
        defaults.update({k: v for k, v in cfg.items() if v is not None})
    # CLI args override YAML (only if explicitly provided)
    for key in defaults:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            defaults[key] = cli_val
    # Apply merged config back to args
    for key, val in defaults.items():
        setattr(args, key, val)

    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "lsmcmc_v1_output")
    os.makedirs(args.outdir, exist_ok=True)
    if args.data is None:
        args.data = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "linear_gaussian_data.npz")

    dat = np.load(args.data)
    T = int(dat['T'])
    d = int(dat['d'])

    all_means = np.zeros((args.M, T + 1, d), dtype=np.float64)
    all_elapsed = np.zeros(args.M)

    worker_args = [
        (isim, args.data, args.outdir, args.Gamma, args.Na,
         args.burn_in, args.Nf)
        for isim in range(args.M)
    ]

    if args.workers > 1:
        n_workers = min(args.workers, args.M)
        print(f"[LSMCMC-V1] Launching {args.M} runs on {n_workers} workers "
              f"(Na={args.Na}, Nf={args.Nf})")
        with mp.Pool(n_workers) as pool:
            for isim, mean_arr, elapsed in pool.imap_unordered(
                    _worker_v1, worker_args):
                all_means[isim] = mean_arr
                all_elapsed[isim] = elapsed
                print(f"  [done] run {isim+1}/{args.M}  elapsed={elapsed:.1f}s",
                      flush=True)
    else:
        for wa in worker_args:
            isim, mean_arr, elapsed = _worker_v1(wa)
            all_means[isim] = mean_arr
            all_elapsed[isim] = elapsed

    # Average over M runs
    avg_mean = np.mean(all_means, axis=0)
    avg_elapsed = np.mean(all_elapsed)

    outfile_avg = os.path.join(args.outdir, "linear_gaussian_lsmcmc_v1_avg.npz")
    np.savez(outfile_avg,
             lsmcmc_mean=avg_mean,
             all_means=all_means,
             T=T, d=d,
             M=args.M, Gamma=args.Gamma, Na=args.Na, Nf=args.Nf,
             elapsed_mean=avg_elapsed, elapsed_all=all_elapsed)
    # ---- Compute & print RMSE of averaged mean vs KF ----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kf_file = os.path.join(base_dir, "linear_gaussian_kf.npz")
    if os.path.exists(kf_file):
        kf_mean = np.load(kf_file)['kf_mean']
        rmse_per_cycle = np.sqrt(np.mean((avg_mean - kf_mean) ** 2, axis=1))
        rmse_overall = np.mean(rmse_per_cycle[1:])
        print(f"\n[LSMCMC-V1] Averaged RMSE vs KF (excluding t=0): {rmse_overall:.6f}")
    else:
        print(f"\n[LSMCMC-V1] KF file not found, skipping RMSE.")

    print(f"[LSMCMC-V1] Averaged over M={args.M} runs, "
          f"mean time={avg_elapsed:.1f}s")
    print(f"[LSMCMC-V1] Saved to {outfile_avg}")


if __name__ == "__main__":
    main()

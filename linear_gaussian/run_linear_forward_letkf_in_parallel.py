#!/usr/bin/env python
"""
run_linear_forward_letkf_in_parallel.py
=======================================
Run M independent LETKF runs (each with K=50 ensemble members) using
different random seeds, then average their analysis means and compute
RMSE vs the KF reference.

This tests whether averaging M independent LETKF analyses can reduce
the RMSE compared to a single run.

Best LETKF parameters (from sensitivity): hscale=1.0, covinflate1=1.02, K=50
"""
import os
import sys
import time
import json
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from linear_forward_run_letkf_sensitivity import (
    gaspcohn,
    calcwts_letkf,
    precompute_covlocal_euclidean,
    letkf_update,
)


def run_single_letkf(args):
    """
    Run a single LETKF with a given seed. Returns the analysis mean trajectory.
    Called by multiprocessing.Pool.
    """
    (run_id, seed, data_file, K, hscale, covinflate1) = args

    np.random.seed(seed)

    dat = np.load(data_file)
    T = int(dat['T']); d = int(dat['d'])
    Ngx = int(dat['Ngx']); Ngy = int(dat['Ngy'])
    sigma_z = float(dat['sigma_z']); sigma_y = float(dat['sigma_y'])
    a_coeff = float(dat['a_coeff']); Z0 = dat['Z0']
    obs_inds_arr = dat['obs_inds']; y_obs_arr = dat['y_obs']
    nobs_arr = dat['nobs']

    # Initialise ensemble
    xens = np.tile(Z0, (K, 1)) + 0.01 * np.random.randn(K, d)

    letkf_mean = np.zeros((T + 1, d))
    letkf_mean[0] = Z0.copy()

    obs_errvar = sigma_y ** 2

    t0 = time.time()

    for k in range(T):
        # Forecast
        for j in range(K):
            xens[j] = a_coeff * xens[j] + sigma_z * np.random.randn(d)

        # Covariance inflation
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

        # LETKF update
        xens = letkf_update(xens, y_k, obs_idx, obs_errvar_vec,
                            covlocal, d)
        letkf_mean[k + 1] = xens.mean(axis=0)

    elapsed = time.time() - t0
    print(f"  [Run {run_id+1:3d}] seed={seed}  elapsed={elapsed:.1f}s")

    return letkf_mean


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run M parallel LETKF runs and average their analyses")
    parser.add_argument("--M", type=int, default=52,
                        help="Number of independent LETKF runs")
    parser.add_argument("--K", type=int, default=50,
                        help="Ensemble size per run")
    parser.add_argument("--hscale", type=float, default=1.0,
                        help="Localisation half-scale")
    parser.add_argument("--covinflate1", type=float, default=1.02,
                        help="Covariance inflation factor")
    parser.add_argument("--base_seed", type=int, default=1000,
                        help="Base seed; run i uses seed = base_seed + i")
    parser.add_argument("--nprocs", type=int, default=None,
                        help="Number of parallel processes (default: cpu_count)")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--use_best", action="store_true",
                        help="Load best params from sensitivity analysis")
    args = parser.parse_args()

    basedir = os.path.dirname(os.path.abspath(__file__))

    if args.data is None:
        args.data = os.path.join(basedir, "linear_gaussian_data.npz")
    if args.outdir is None:
        args.outdir = os.path.join(basedir, "letkf_Mparallel_output")
    os.makedirs(args.outdir, exist_ok=True)

    # Load best params from sensitivity if requested
    if args.use_best:
        sens_file = os.path.join(basedir, "letkf_sensitivity_results.json")
        if os.path.exists(sens_file):
            with open(sens_file) as f:
                sens = json.load(f)
            best = sens.get('best', {})
            args.hscale = best.get('hscale', args.hscale)
            args.covinflate1 = best.get('covinflate1', args.covinflate1)
            args.K = best.get('K', args.K)
            print(f"[Parallel LETKF] Using best params: "
                  f"hscale={args.hscale}, covinflate1={args.covinflate1}, K={args.K}")

    nprocs = args.nprocs if args.nprocs else min(args.M, cpu_count())

    print(f"[Parallel LETKF] M={args.M} runs, K={args.K}, "
          f"hscale={args.hscale}, covinflate1={args.covinflate1}")
    print(f"[Parallel LETKF] Using {nprocs} processes")

    # ---- Load KF reference ----
    kf_file = os.path.join(basedir, "linear_gaussian_kf.npz")
    kf_mean = None
    if os.path.exists(kf_file):
        kf_mean = np.load(kf_file)['kf_mean']
        print(f"[Parallel LETKF] Loaded KF reference from {kf_file}")
    else:
        print(f"[Parallel LETKF] KF file not found, skipping RMSE vs KF.")

    # ---- Also load single-run LETKF reference for comparison ----
    single_file = os.path.join(basedir, "linear_gaussian_letkf.npz")
    single_rmse = None
    if os.path.exists(single_file) and kf_mean is not None:
        single_mean = np.load(single_file)['letkf_mean']
        single_rmse_per_cycle = np.sqrt(np.mean((single_mean - kf_mean) ** 2, axis=1))
        single_rmse = np.mean(single_rmse_per_cycle[1:])
        print(f"[Parallel LETKF] Single-run LETKF RMSE vs KF: {single_rmse:.6f}")

    # ---- Build worker arguments ----
    worker_args = [
        (i, args.base_seed + i, args.data, args.K, args.hscale, args.covinflate1)
        for i in range(args.M)
    ]

    # ---- Run in parallel ----
    t0 = time.time()
    print(f"\n[Parallel LETKF] Launching {args.M} runs ...")

    with Pool(processes=nprocs) as pool:
        all_means = pool.map(run_single_letkf, worker_args)

    elapsed = time.time() - t0
    print(f"\n[Parallel LETKF] All {args.M} runs finished in {elapsed:.1f}s")

    # ---- Stack and average ----
    # all_means is a list of (T+1, d) arrays
    stacked = np.stack(all_means, axis=0)  # (M, T+1, d)
    averaged_mean = stacked.mean(axis=0)   # (T+1, d)

    # ---- Compute RMSE for each individual run and for the average ----
    dat = np.load(args.data)
    T = int(dat['T']); d = int(dat['d'])

    if kf_mean is not None:
        # Per-run RMSE
        individual_rmses = []
        for i in range(args.M):
            rmse_per_cycle = np.sqrt(np.mean((stacked[i] - kf_mean) ** 2, axis=1))
            individual_rmses.append(np.mean(rmse_per_cycle[1:]))
        individual_rmses = np.array(individual_rmses)

        # Averaged-mean RMSE
        avg_rmse_per_cycle = np.sqrt(np.mean((averaged_mean - kf_mean) ** 2, axis=1))
        avg_rmse = np.mean(avg_rmse_per_cycle[1:])

        print(f"\n{'='*60}")
        print(f"  RESULTS  (M={args.M} parallel LETKF runs, K={args.K})")
        print(f"{'='*60}")
        print(f"  Individual run RMSE vs KF:")
        print(f"    mean = {individual_rmses.mean():.6f}")
        print(f"    std  = {individual_rmses.std():.6f}")
        print(f"    min  = {individual_rmses.min():.6f}")
        print(f"    max  = {individual_rmses.max():.6f}")
        print(f"")
        print(f"  Averaged-mean RMSE vs KF: {avg_rmse:.6f}")
        if single_rmse is not None:
            print(f"  Single-run LETKF RMSE:    {single_rmse:.6f}")
            reduction_pct = (1 - avg_rmse / single_rmse) * 100
            print(f"  Reduction from averaging:  {reduction_pct:+.2f}%")
        print(f"{'='*60}")

    # ---- Save results ----
    outfile = os.path.join(args.outdir, "letkf_Mparallel_results.npz")
    save_dict = dict(
        averaged_mean=averaged_mean,
        individual_rmses=individual_rmses if kf_mean is not None else np.array([]),
        avg_rmse=avg_rmse if kf_mean is not None else np.nan,
        M=args.M, K=args.K,
        hscale=args.hscale, covinflate1=args.covinflate1,
        elapsed=elapsed, T=T, d=d,
        base_seed=args.base_seed,
    )
    if single_rmse is not None:
        save_dict['single_rmse'] = single_rmse
    np.savez(outfile, **save_dict)
    print(f"\n[Parallel LETKF] Saved results to {outfile}")

    # Also save a JSON summary
    summary = {
        "M": args.M,
        "K": args.K,
        "hscale": args.hscale,
        "covinflate1": args.covinflate1,
        "base_seed": args.base_seed,
        "elapsed_s": round(elapsed, 1),
        "avg_rmse_vs_kf": round(avg_rmse, 8) if kf_mean is not None else None,
        "individual_rmse_mean": round(float(individual_rmses.mean()), 8) if kf_mean is not None else None,
        "individual_rmse_std": round(float(individual_rmses.std()), 8) if kf_mean is not None else None,
        "single_run_rmse": round(single_rmse, 8) if single_rmse is not None else None,
    }
    json_file = os.path.join(args.outdir, "letkf_Mparallel_summary.json")
    with open(json_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Parallel LETKF] Saved summary to {json_file}")


if __name__ == "__main__":
    main()

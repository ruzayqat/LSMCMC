#!/usr/bin/env python
"""
run_lsmcmc_v2.py
================
V2 LSMCMC (per-block halo localization with GC tapering) for the linear
Gaussian model.

For each block Q_i:
  1. Find observations within radius r_loc (in grid points)
  2. Build a halo of cells around Q_i within r_loc
  3. Apply Gaspari-Cohn tapering to the process noise variance based on
     distance from Q_i center
  4. Sample exactly from the Gaussian mixture posterior on the halo state
  5. Extract only the interior block cells from the posterior sample
  6. Reduce Na samples to Nf via Gaussian block means (with RTPS inflation)
  7. Blocks with no nearby observations keep the forecast

Settings from the paper:
    Gamma = 90, Na = 500, Nf = 52, r_loc = some value in grid points
"""
import os
import sys
import time

# ---- Pin BLAS/LAPACK to 1 thread per process BEFORE importing NumPy ----
# Without this, each multiprocessing worker spawns its own multi-threaded
# BLAS pool, causing massive thread oversubscription (ncores × BLAS_threads)
# that is *slower* than serial execution.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loc_smcmc_swe_exact_from_Gauss import (
    partition_domain,
    build_H_loc_from_global,
    sample_posterior_mixture_sparse,
    gaussian_block_means,
)


def gaspcohn(r):
    """Gaspari-Cohn compactly supported taper."""
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


def precompute_block_halo(partitions, partition_labels, Ngy, Ngx, r_loc):
    """Precompute halo cells, GC weights, interior masks for each block."""
    block_info = []
    all_iy, all_ix = np.mgrid[:Ngy, :Ngx]
    all_iy_f = all_iy.ravel().astype(np.float64)
    all_ix_f = all_ix.ravel().astype(np.float64)

    for b, ((y0, y1), (x0, x1)) in enumerate(partitions):
        cy = 0.5 * (y0 + y1 - 1)
        cx = 0.5 * (x0 + x1 - 1)

        iy_int, ix_int = np.mgrid[y0:y1, x0:x1]
        interior_flat = np.ravel_multi_index(
            (iy_int.ravel(), ix_int.ravel()), (Ngy, Ngx))

        dy = all_iy_f - cy
        dx = all_ix_f - cx
        dists = np.sqrt(dx * dx + dy * dy)
        halo_mask = dists <= r_loc

        # Ensure ALL interior cells are in the halo (large blocks may exceed r_loc)
        interior_set = set(interior_flat.tolist())
        for c in interior_flat:
            halo_mask[c] = True

        halo_flat = np.where(halo_mask)[0]
        # GC weights: taper only beyond r_loc; interior cells within r_loc get
        # their natural taper, those beyond get weight 1.0 (no taper).
        gc_raw = dists[halo_mask] / max(r_loc, 1e-15)
        gc_wts = gaspcohn(gc_raw)
        # Force interior cells that fell outside r_loc to have weight 1.0
        for i, c in enumerate(halo_flat):
            if c in interior_set and gc_wts[i] < 1e-15:
                gc_wts[i] = 1.0

        interior_in_halo = np.isin(halo_flat, interior_flat)
        halo_sorted = np.sort(halo_flat)

        block_info.append({
            'halo_cells': halo_flat,
            'halo_sorted': halo_sorted,
            'gc_weights': gc_wts,
            'interior_in_halo': interior_in_halo,
            'interior_flat': interior_flat,
        })
    return block_info


# ---- Module-level worker for multiprocessing (must be picklable) ----
_worker_shared = {}   # set by parent before Pool creation


def _init_block_worker(forecast_shape, block_info_list, Na, sigma_y, rtps_alpha):
    """Initialise per-worker globals."""
    _worker_shared['Na'] = Na
    _worker_shared['sigma_y'] = sigma_y
    _worker_shared['rtps_alpha'] = rtps_alpha
    _worker_shared['block_info'] = block_info_list


def _process_block(args):
    """Worker: run DA on one block. Returns (interior_flat, mu_interior, anal_reduced) or None."""
    b, halo_cells, gc_wts, interior_mask, interior_flat, halo_sorted, \
        obs_sel, obs_idx_full, y_k_full, forecast_halo, forecast_interior = args

    Na = _worker_shared['Na']
    sigma_y = _worker_shared['sigma_y']
    rtps_alpha = _worker_shared['rtps_alpha']
    Nf = forecast_interior.shape[1]

    local_obs_idx = obs_idx_full[obs_sel]
    local_y = y_k_full[obs_sel]

    H_loc = build_H_loc_from_global(local_obs_idx, halo_cells)
    sigma_x_loc = gc_wts  # already scaled by sigma_z in caller
    m_list = forecast_halo.T   # (Nf, d_halo)

    idx = np.searchsorted(halo_sorted, local_obs_idx)
    kept_mask = (idx < len(halo_sorted)) & (halo_sorted[np.minimum(idx, len(halo_sorted)-1)] == local_obs_idx)
    y_loc = local_y[kept_mask]

    if y_loc.size == 0:
        return None

    anal_samples, mu_loc = sample_posterior_mixture_sparse(
        m_list, H_loc, sigma_x_loc, sigma_y, y_loc, n_samples=Nf)

    anal_interior = anal_samples[interior_mask]
    mu_interior = mu_loc[interior_mask]
    anal_reduced = anal_interior  # use posterior samples directly (no block-mean compression)

    if rtps_alpha > 0:
        fc_std = forecast_interior.std(axis=1)
        an_std = anal_reduced.std(axis=1)
        an_mean = anal_reduced.mean(axis=1)
        scale = np.where(an_std > 1e-15,
                         1.0 + rtps_alpha * (fc_std - an_std) / an_std, 1.0)
        anal_reduced = an_mean[:, None] + scale[:, None] * (anal_reduced - an_mean[:, None])

    return (interior_flat, mu_interior, anal_reduced)


# Threshold: if a block has fewer cells than this, process all blocks serially.
# With BLAS threads pinned to 1, pool overhead is modest even for small blocks,
# so we keep this very low.  The halo (r_loc) dominates the per-block cost.
SERIAL_BLOCK_CELL_THRESHOLD = 4


def run_lsmcmc_v2(data_file=None, outdir=None, seed=42,
                   Gamma=90, Na=500, Nf=52, r_loc=10.0,
                   rtps_alpha=0.0, ncores=0):
    np.random.seed(seed)

    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "linear_gaussian_data.npz")
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)

    dat = np.load(data_file)
    T = int(dat['T']); d = int(dat['d'])
    Ngx = int(dat['Ngx']); Ngy = int(dat['Ngy'])
    sigma_z = float(dat['sigma_z']); sigma_y = float(dat['sigma_y'])
    a_coeff = float(dat['a_coeff'])
    Z0 = dat['Z0']
    obs_inds_arr = dat['obs_inds']; y_obs_arr = dat['y_obs']; nobs_arr = dat['nobs']

    print(f"[LSMCMC-V2] d={d}, T={T}, Gamma={Gamma}, Na={Na}, Nf={Nf}, r_loc={r_loc}")

    # ---- Load KF reference for per-cycle RMSE ----
    kf_file = os.path.join(os.path.dirname(data_file), "linear_gaussian_kf.npz")
    kf_mean = None
    if os.path.exists(kf_file):
        kf_mean = np.load(kf_file)['kf_mean']
        print(f"[LSMCMC-V2] Loaded KF reference from {kf_file}")
    else:
        print(f"[LSMCMC-V2] KF file not found at {kf_file}, skipping per-cycle RMSE.")

    partitions, partition_labels, Gamma_actual, nby, nbx, bh, bw = \
        partition_domain(Ngy, Ngx, N=Gamma)
    cells_per_block = bh * bw
    use_parallel = cells_per_block >= SERIAL_BLOCK_CELL_THRESHOLD
    print(f"[LSMCMC-V2] Partition: {nby}x{nbx} blocks of {bh}x{bw} "
          f"({cells_per_block} cells/block)")

    # Decide number of workers
    if ncores <= 0:
        ncores = max(1, mp.cpu_count() - 1)
    if not use_parallel:
        ncores = 1
    print(f"[LSMCMC-V2] {'Parallel' if use_parallel else 'Serial'} mode "
          f"(threshold={SERIAL_BLOCK_CELL_THRESHOLD} cells), ncores={ncores}")

    # Set up pool if parallel
    pool = None
    if use_parallel and ncores > 1:
        pool = mp.Pool(ncores, initializer=_init_block_worker,
                       initargs=(None, None, Na, sigma_y, rtps_alpha))
        print(f"[LSMCMC-V2] Pool created with {ncores} workers")
    else:
        # Initialise shared state in main process for serial path
        _init_block_worker(None, None, Na, sigma_y, rtps_alpha)

    block_info = precompute_block_halo(partitions, partition_labels, Ngy, Ngx, r_loc)

    # Precompute cell -> list of block ids whose halo contains that cell
    cell_to_blocks = [[] for _ in range(d)]
    for b, binfo in enumerate(block_info):
        for c in binfo['halo_cells']:
            cell_to_blocks[c].append(b)

    forecast = np.tile(Z0[:, None], (1, Nf))
    lsmcmc_mean = np.zeros((T + 1, d), dtype=np.float64)
    lsmcmc_mean[0] = Z0.copy()

    t0 = time.time()

    for k in range(T):
        t_step = time.time()

        # Forecast (vectorised)
        forecast = a_coeff * forecast + sigma_z * np.random.randn(d, Nf)
        t_fcst = time.time() - t_step

        forecast_mean = np.mean(forecast, axis=1)

        mk = int(nobs_arr[k])
        obs_idx = obs_inds_arr[k, :mk].astype(int)
        y_k = y_obs_arr[k, :mk]

        if mk == 0:
            lsmcmc_mean[k + 1] = forecast_mean
            continue

        # Route obs to blocks via precomputed lookup
        block_obs = [[] for _ in range(len(block_info))]
        for oi, cell in enumerate(obs_idx):
            for b in cell_to_blocks[cell]:
                block_obs[b].append(oi)

        new_mean = forecast_mean.copy()
        new_forecast = forecast.copy()

        # Build task list for blocks with observations
        block_tasks = []
        for b, binfo in enumerate(block_info):
            if not block_obs[b]:
                continue
            obs_sel = np.array(block_obs[b], dtype=int)
            halo_cells = binfo['halo_cells']
            gc_wts = sigma_z * binfo['gc_weights']
            interior_mask = binfo['interior_in_halo']
            interior_flat = binfo['interior_flat']
            halo_sorted = binfo['halo_sorted']

            block_tasks.append((
                b, halo_cells, gc_wts, interior_mask, interior_flat, halo_sorted,
                obs_sel, obs_idx, y_k,
                forecast[halo_cells].copy(),
                forecast[interior_flat].copy(),
            ))

        if block_tasks:
            if pool is not None:
                # Use chunksize to batch IPC for many small blocks
                chunksize = max(1, len(block_tasks) // (ncores * 4))
                results_list = pool.map(_process_block, block_tasks,
                                        chunksize=chunksize)
            else:
                results_list = [_process_block(t) for t in block_tasks]

            for res in results_list:
                if res is None:
                    continue
                interior_flat, mu_interior, anal_reduced = res
                new_mean[interior_flat] = mu_interior
                new_forecast[interior_flat, :] = anal_reduced

        t_da = time.time() - t_step - t_fcst
        lsmcmc_mean[k + 1] = new_mean
        forecast = new_forecast

        if (k + 1) % 10 == 0 or k == 0:
            elapsed = time.time() - t0
            rmse_str = ""
            if kf_mean is not None:
                rmse_k = np.sqrt(np.mean((lsmcmc_mean[k + 1] - kf_mean[k + 1]) ** 2))
                rmse_str = f"  RMSE={rmse_k:.6f}"
            print(f"  Cycle {k+1:3d}/{T}  nobs={mk:4d}  "
                  f"fcst={t_fcst:.2f}s  DA={t_da:.2f}s  elapsed={elapsed:.1f}s{rmse_str}")
        else:
            if kf_mean is not None:
                rmse_k = np.sqrt(np.mean((lsmcmc_mean[k + 1] - kf_mean[k + 1]) ** 2))
                print(f"  Cycle {k+1:3d}/{T}  RMSE={rmse_k:.6f}")

    # Clean up pool
    if pool is not None:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    print(f"[LSMCMC-V2] Done in {elapsed:.1f}s")

    # ---- Compute & print RMSE vs KF ----
    if kf_mean is not None:
        rmse_per_cycle = np.sqrt(np.mean((lsmcmc_mean - kf_mean) ** 2, axis=1))
        rmse_overall = np.mean(rmse_per_cycle[1:])
        print(f"[LSMCMC-V2] RMSE vs KF (per-cycle mean, excluding t=0): {rmse_overall:.6f}")

    outfile = os.path.join(outdir, "linear_gaussian_lsmcmc_v2.npz")
    np.savez(outfile, lsmcmc_mean=lsmcmc_mean, T=T, d=d,
             Gamma=Gamma_actual, Na=Na, Nf=Nf, r_loc=r_loc, elapsed=elapsed)
    print(f"[LSMCMC-V2] Saved to {outfile}")
    return outfile, lsmcmc_mean, elapsed


def _worker_v2(args_tuple):
    """Picklable worker for parallel V2 runs.

    Each worker forces ncores=1 inside run_lsmcmc_v2 so the inner
    block-level Pool is DISABLED.  The outer Pool already saturates
    all CPU cores with M-level parallelism — a nested pool would
    cause thread oversubscription.
    """
    isim, data_file, outdir, Gamma, Na, Nf, r_loc, rtps_alpha, ncores_inner = args_tuple
    seed = 2000 + isim
    sim_outdir = os.path.join(outdir, f"sim_{isim:03d}")
    os.makedirs(sim_outdir, exist_ok=True)
    print(f"\n====== LSMCMC-V2 run {isim+1} (seed={seed}) ======", flush=True)
    _outfile, mean_arr, elapsed = run_lsmcmc_v2(
        data_file=data_file, outdir=sim_outdir, seed=seed,
        Gamma=Gamma, Na=Na, Nf=Nf, r_loc=r_loc,
        rtps_alpha=rtps_alpha, ncores=ncores_inner)
    return isim, mean_arr.copy(), float(elapsed)


def _load_yaml_config(path):
    """Load a YAML config file and return as dict."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    """Run M independent simulations (in parallel) and average."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--M", type=int, default=None, help="Independent runs")
    parser.add_argument("--Gamma", type=int, default=None)
    parser.add_argument("--Na", type=int, default=None)
    parser.add_argument("--Nf", type=int, default=None)
    parser.add_argument("--r_loc", type=float, default=None)
    parser.add_argument("--rtps_alpha", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for M runs (0 = serial)")
    parser.add_argument("--ncores", type=int, default=None,
                        help="Number of cores for block-level parallelism within each run")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    # Defaults
    defaults = dict(M=52, Gamma=90, Na=500, Nf=52, r_loc=1.0,
                    rtps_alpha=0.02, workers=52, ncores=0, data=None, outdir=None)
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
                                   "lsmcmc_v2_output")
    os.makedirs(args.outdir, exist_ok=True)
    if args.data is None:
        args.data = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "linear_gaussian_data.npz")

    dat = np.load(args.data)
    T = int(dat['T']); d = int(dat['d'])

    all_means = np.zeros((args.M, T + 1, d), dtype=np.float64)
    all_elapsed = np.zeros(args.M)

    # Distribute cores across outer workers; 0 = auto-detect per worker
    if args.workers > 1 and args.ncores == 0:
        total_cores = max(1, mp.cpu_count())
        inner_ncores = max(1, total_cores // min(args.workers, args.M))
    elif args.workers > 1:
        inner_ncores = max(1, args.ncores // min(args.workers, args.M))
    else:
        inner_ncores = args.ncores

    worker_args = [
        (isim, args.data, args.outdir, args.Gamma, args.Na,
         args.Nf, args.r_loc, args.rtps_alpha,
         inner_ncores)
        for isim in range(args.M)
    ]

    if args.workers > 1:
        n_workers = min(args.workers, args.M)
        print(f"[LSMCMC-V2] Launching {args.M} runs on {n_workers} workers "
              f"(Na={args.Na}, Nf={args.Nf}, r_loc={args.r_loc}, "
              f"inner_ncores={inner_ncores})")
        # Use ProcessPoolExecutor (non-daemon) so inner mp.Pool works
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_v2, wa): wa[0]
                       for wa in worker_args}
            for fut in futures:
                isim, mean_arr, elapsed = fut.result()
                all_means[isim] = mean_arr
                all_elapsed[isim] = elapsed
                print(f"  [done] run {isim+1}/{args.M}  elapsed={elapsed:.1f}s",
                      flush=True)
    else:
        for wa in worker_args:
            isim, mean_arr, elapsed = _worker_v2(wa)
            all_means[isim] = mean_arr
            all_elapsed[isim] = elapsed

    # Average over M runs
    avg_mean = np.mean(all_means, axis=0)
    avg_elapsed = np.mean(all_elapsed)

    outfile_avg = os.path.join(args.outdir, "linear_gaussian_lsmcmc_v2_avg.npz")
    np.savez(outfile_avg,
             lsmcmc_mean=avg_mean,
             all_means=all_means,
             T=T, d=d,
             M=args.M, Gamma=args.Gamma, Na=args.Na, Nf=args.Nf,
             r_loc=args.r_loc,
             elapsed_mean=avg_elapsed, elapsed_all=all_elapsed)

    # ---- Compute & print RMSE of averaged mean vs KF ----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kf_file = os.path.join(base_dir, "linear_gaussian_kf.npz")
    if os.path.exists(kf_file):
        kf_mean = np.load(kf_file)['kf_mean']
        rmse_per_cycle = np.sqrt(np.mean((avg_mean - kf_mean) ** 2, axis=1))
        rmse_overall = np.mean(rmse_per_cycle[1:])
        print(f"\n[LSMCMC-V2] Averaged RMSE vs KF (excluding t=0): {rmse_overall:.6f}")
    else:
        print(f"\n[LSMCMC-V2] KF file not found, skipping RMSE.")

    print(f"[LSMCMC-V2] Averaged over M={args.M} runs, "
          f"mean time={avg_elapsed:.1f}s")
    print(f"[LSMCMC-V2] Saved to {outfile_avg}")


if __name__ == "__main__":
    main()

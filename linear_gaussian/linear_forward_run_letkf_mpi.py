#!/usr/bin/env python
"""
linear_forward_run_letkf_mpi.py
================================
Multiprocessing-parallelised LETKF for the linear Gaussian model.

Grid points are distributed across mp.Pool workers so the LETKF
analysis update runs in parallel.  The forecast step
(Z_{k+1} = a Z_k + sigma_z W) is trivially cheap and runs serially.

Usage
-----
    python linear_forward_run_letkf_mpi.py --config input_linear_letkf.yml

Model:  Z_{k+1} = a * Z_k + sigma_z * W,   d = 120x120 = 14400
Obs:    Y_k = C_k * Z_k + sigma_y * V  (SWOT-like swath)
"""
import os
import sys
import time
import json
import ctypes
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse LETKF helper functions from sensitivity script
from linear_forward_run_letkf_sensitivity import (
    gaspcohn,
    calcwts_letkf,
    precompute_covlocal_euclidean,
)


# ===================================================================
#  Shared-memory arrays for zero-copy parallel DA
# ===================================================================
_shm_xens_buf = None      # RawArray backing the ensemble
_shm_xens_shape = None     # (K, d)
_shm_Ngy = None
_shm_Ngx = None


def _init_da_worker(xens_buf, xens_shape, Ngy, Ngx):
    """Initialiser for DA pool workers — store shared references."""
    global _shm_xens_buf, _shm_xens_shape, _shm_Ngy, _shm_Ngx
    _shm_xens_buf = xens_buf
    _shm_xens_shape = xens_shape
    _shm_Ngy = Ngy
    _shm_Ngx = Ngx


def _da_worker(args):
    """
    LETKF update for grid-point chunk [nstart..nend].

    Reads ensemble from shared memory; returns (nstart, updated_local).
    """
    nstart, nend, obs_values, obs_indices, obs_errvar, hscale = args
    K, d = _shm_xens_shape

    # View into shared memory (zero-copy)
    xens = np.frombuffer(_shm_xens_buf, dtype=np.float64).reshape(K, d)

    nlocal = nend - nstart + 1

    # Localisation for this chunk
    covlocal_local = precompute_covlocal_euclidean(
        obs_indices, _shm_Ngy, _shm_Ngx, hscale, nstart, nend)

    # Observation-space quantities
    hxens = xens[:, obs_indices]
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    innovations = obs_values - hxmean

    xmean = xens.mean(axis=0)
    xprime = xens - xmean

    inv_obs_errvar = 1.0 / obs_errvar

    xens_local = xens[:, nstart:nend + 1].copy()

    for count in range(nlocal):
        loc_wt = covlocal_local[:, count]
        mask = loc_wt > 1.e-10
        nobs_local = int(mask.sum())
        if nobs_local == 0:
            continue

        rinv_diag = loc_wt[mask] * inv_obs_errvar[mask]
        hx_local = hxprime[:, mask]
        innov_local = innovations[mask]

        wts = calcwts_letkf(hx_local, rinv_diag, innov_local, K)

        if not np.isfinite(wts).all():
            continue

        n_global = nstart + count
        updated = xmean[n_global] + xprime[:, n_global] @ wts

        if np.isfinite(updated).all():
            xens_local[:, count] = updated.ravel()

    return (nstart, xens_local)


# ===================================================================
#  Main
# ===================================================================
def main():
    print("=" * 64)
    print("  Linear Gaussian LETKF — multiprocessing-parallelised")
    print("=" * 64)

    # ---- Config ----
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--hscale", type=float, default=None)
    parser.add_argument("--covinflate1", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--ncores", type=int, default=None)
    parser.add_argument("--use_best", action="store_true")
    args = parser.parse_args()

    # Defaults
    basedir = os.path.dirname(os.path.abspath(__file__))
    defaults = dict(K=50, hscale=10, covinflate1=1.0, seed=42,
                    data=None, outdir=None, use_best=False, ncores=52)

    # Load YAML config if provided
    if args.config:
        import yaml
        cfg_path = args.config
        if not os.path.isabs(cfg_path):
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(basedir, cfg_path)
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        defaults.update({k: v for k, v in cfg.items() if v is not None})

    # CLI args override YAML
    for key in ['K', 'hscale', 'covinflate1', 'seed', 'data', 'outdir',
                'ncores']:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            defaults[key] = cli_val
    if args.use_best:
        defaults['use_best'] = True

    K = int(defaults['K'])
    hscale = float(defaults['hscale'])
    covinflate1 = float(defaults['covinflate1'])
    seed = int(defaults['seed'])
    data_file = defaults['data']
    outdir = defaults['outdir']
    use_best = defaults.get('use_best', False)
    num_workers = int(defaults['ncores'])

    if outdir is None:
        outdir = basedir
    if data_file is None:
        data_file = os.path.join(basedir, "linear_gaussian_data.npz")

    # Load sensitivity best params if requested
    if use_best:
        sens_file = os.path.join(basedir, "letkf_sensitivity_results.json")
        if os.path.exists(sens_file):
            with open(sens_file) as f:
                sens = json.load(f)
            best = sens.get('best', {})
            hscale = best.get('hscale', hscale)
            covinflate1 = best.get('covinflate1', covinflate1)
            K = int(best.get('K', K))
            print(f"[LETKF] Using best params from sensitivity: "
                  f"hscale={hscale}, covinflate1={covinflate1}, K={K}")

    np.random.seed(seed)

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
    ncells = d

    print(f"[LETKF] d={d}, T={T}, K={K}, hscale={hscale}, "
          f"covinflate1={covinflate1}")
    print(f"[LETKF] Grid: {Ngx}x{Ngy} = {ncells} cells")

    # ---- Load KF reference ----
    kf_file = os.path.join(basedir, "linear_gaussian_kf.npz")
    kf_mean = None
    if os.path.exists(kf_file):
        kf_mean = np.load(kf_file)['kf_mean']
        print(f"[LETKF] Loaded KF reference from {kf_file}")

    # ---- Work distribution (grid point chunks for workers) ----
    base_inc = ncells // num_workers
    remainder = ncells % num_workers
    chunks = []
    offset = 0
    for r in range(num_workers):
        sz = base_inc + (1 if r < remainder else 0)
        if sz > 0:
            chunks.append((offset, offset + sz - 1))
        offset += sz

    print(f"[LETKF] DA pool: {num_workers} workers, "
          f"{len(chunks)} chunks, "
          f"pts/chunk: {chunks[0][1]-chunks[0][0]+1}"
          f"–{chunks[-1][1]-chunks[-1][0]+1}")

    # ---- Shared-memory ensemble array ----
    shm_buf = mp.RawArray(ctypes.c_double, K * d)
    xens_np = np.frombuffer(shm_buf, dtype=np.float64).reshape(K, d)

    # Initialise ensemble
    init_ens = np.tile(Z0, (K, 1)) + 0.01 * np.random.randn(K, d)
    np.copyto(xens_np, init_ens)

    # ---- Storage ----
    letkf_mean = np.zeros((T + 1, d))
    letkf_mean[0] = Z0.copy()
    os.makedirs(outdir, exist_ok=True)

    obs_errvar = sigma_y ** 2

    # ---- Create DA pool ----
    da_pool = mp.Pool(
        num_workers,
        initializer=_init_da_worker,
        initargs=(shm_buf, (K, d), Ngy, Ngx))

    print(f"\n[LETKF] Starting {T} assimilation cycles ...")
    t_total_start = time.time()

    for k in range(T):
        t_step = time.time()

        # ---- Forecast (trivially cheap, serial) ----
        noise = sigma_z * np.random.randn(K, d)
        xens_np[:] = a_coeff * xens_np + noise

        # ---- Covariance inflation ----
        if covinflate1 != 1.0:
            xmean = xens_np.mean(axis=0)
            xens_np[:] = xmean + covinflate1 * (xens_np - xmean)

        t_fcst = time.time() - t_step

        # ---- Get observations ----
        mk = int(nobs_arr[k])
        obs_idx = obs_inds_arr[k, :mk].astype(int)
        y_k = y_obs_arr[k, :mk]

        if mk == 0:
            letkf_mean[k + 1] = xens_np.mean(axis=0)
            continue

        obs_errvar_vec = np.full(mk, obs_errvar)

        # ---- Parallel LETKF update ----
        t_da_start = time.time()

        worker_args = [
            (ns, ne, y_k, obs_idx, obs_errvar_vec, hscale)
            for ns, ne in chunks
        ]
        results = da_pool.map(_da_worker, worker_args)

        # Assemble results back into shared-memory ensemble
        for nstart, xens_local in results:
            nlocal = xens_local.shape[1]
            xens_np[:, nstart:nstart + nlocal] = xens_local

        t_da = time.time() - t_da_start

        # ---- RMSE ----
        z_a = xens_np.mean(axis=0)
        letkf_mean[k + 1] = z_a

        dt_wall = time.time() - t_step
        if (k + 1) % 10 == 0 or k == 0:
            rmse_str = ""
            if kf_mean is not None:
                rmse_k = np.sqrt(np.mean((z_a - kf_mean[k + 1]) ** 2))
                rmse_str = f"  RMSE={rmse_k:.6f}"
            print(f"  [{k+1:3d}/{T}]  nobs={mk:4d}  "
                  f"fcst={t_fcst:.2f}s  DA={t_da:.2f}s  "
                  f"wall={dt_wall:.2f}s{rmse_str}")

    # ---- Cleanup ----
    da_pool.close()
    da_pool.join()

    elapsed = time.time() - t_total_start
    print(f"\n[LETKF] Done in {elapsed:.1f}s")

    # Compute overall RMSE vs KF
    if kf_mean is not None:
        rmse_per_cycle = np.sqrt(
            np.mean((letkf_mean - kf_mean) ** 2, axis=1))
        rmse_overall = np.mean(rmse_per_cycle[1:])
        print(f"[LETKF] RMSE vs KF (per-cycle mean): {rmse_overall:.6f}")

    # Save
    outfile = os.path.join(outdir, "linear_gaussian_letkf_mpi.npz")
    np.savez(outfile, letkf_mean=letkf_mean, T=T, d=d,
             K=K, hscale=hscale, covinflate1=covinflate1,
             elapsed=elapsed, num_procs=num_workers)
    print(f"[LETKF] Saved to {outfile}")


if __name__ == "__main__":
    main()

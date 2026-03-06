"""
mlswe_letkf_nl_sensitivity.py
================================
Sensitivity analysis for the **nonlinear** LETKF twin experiment.

Sweeps over:
  - hcovlocal_scale  (localisation cutoff in km)
  - covinflate1      (RTPS inflation factor)

Runs a grid of experiments using ``run_mlswe_letkf_nl_twin.py`` with
``--hcovlocal_scale`` and ``--covinflate1`` CLI overrides, collects
mean RMSE values (against truth state), and saves results to JSON.

Usage
-----
    python3 mlswe_letkf_nl_sensitivity.py \\
        --nprocs 8 \\
        --config example_input_mlswe_letkf_nl_sensitivity.yml

Prerequisites
-------------
    Truth + synthetic obs must already exist in ``output_lsmcmc_nldata_V1/``:
        python run_mlswe_lsmcmc_nldata_V1_twin.py --truth-only
"""

import subprocess
import re
import sys
import json
import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser(
        description="NL-LETKF sensitivity analysis (arctan twin)")
    parser.add_argument('--ncores', type=int, default=52,
                        help='Number of CPU cores for mp.Pool')
    parser.add_argument('--config', type=str,
                        default='example_input_mlswe_letkf_nl_twin.yml',
                        help='YAML config file')
    parser.add_argument('--results', type=str,
                        default='mlswe_letkf_nl_sensitivity_results.json',
                        help='JSON file for incremental results')
    parser.add_argument('--timeout', type=int, default=1200,
                        help='Timeout per experiment in seconds')
    parser.add_argument('--nanals', type=int, default=25,
                        help='Ensemble size (default 25)')
    args = parser.parse_args()

    # ---- Parameter grid ----
    hscale_values_km = [60, 80, 100, 120, 150, 200, 300]
    alpha_values     = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]

    experiments = [(h, a) for h in hscale_values_km for a in alpha_values]
    total = len(experiments)

    # Load existing results (incremental)
    results = {}
    if os.path.exists(args.results):
        with open(args.results) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {args.results}")

    print("=" * 72)
    print("  NL-LETKF SENSITIVITY ANALYSIS (arctan twin)")
    print(f"  hcovlocal_scale (km): {hscale_values_km}")
    print(f"  covinflate1:          {alpha_values}")
    print(f"  Total experiments:    {total}")
    print(f"  CPU cores:            {args.ncores}")
    print(f"  Ensemble size:        {args.nanals}")
    print(f"  Config:               {args.config}")
    print(f"  Results file:         {args.results}")
    print("=" * 72)

    skipped = 0
    t_start_all = time.time()

    for idx, (hscale, alpha) in enumerate(experiments):
        key = f"h{hscale}_a{alpha}"

        # Skip if already done
        if key in results and results[key].get("vel") is not None:
            skipped += 1
            print(f"[{idx+1:3d}/{total}] h={hscale:4d}km  "
                  f"α={alpha:.1f}  — SKIP (cached)")
            continue

        print(f"\n{'='*60}")
        print(f"  [{idx+1:3d}/{total}]  hscale={hscale} km,  "
              f"covinflate1={alpha}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "run_mlswe_letkf_nl_twin.py", args.config,
            "--hcovlocal_scale", str(hscale),
            "--covinflate1", str(alpha),
            "--nanals", str(args.nanals),
            "--ncores", str(args.ncores),
        ]

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=args.timeout,
            )
            output = result.stdout + result.stderr
            elapsed = time.time() - t0

            # Parse mean RMSE from output
            vel_m = re.search(
                r'Mean RMSE_vel\s*=\s*([\d.eE+\-]+)', output)
            sst_m = re.search(
                r'Mean RMSE_sst\s*=\s*([\d.eE+\-]+)', output)
            ssh_m = re.search(
                r'Mean RMSE_ssh\s*=\s*([\d.eE+\-]+)', output)

            vel = float(vel_m.group(1)) if vel_m else None
            sst = float(sst_m.group(1)) if sst_m else None
            ssh = float(ssh_m.group(1)) if ssh_m else None

            results[key] = {
                "vel": vel, "sst": sst, "ssh": ssh,
                "time_s": round(elapsed, 1),
            }
            print(f"  -> RMSE_vel={vel}  RMSE_sst={sst}  "
                  f"RMSE_ssh={ssh}  ({elapsed:.0f}s)")

            if result.returncode != 0:
                print(f"  -> WARNING: exit code {result.returncode}")
                lines = output.strip().split('\n')
                for line in lines[-5:]:
                    print(f"     {line}")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  -> TIMEOUT ({elapsed:.0f}s)")
            results[key] = {
                "vel": None, "sst": None, "ssh": None,
                "time_s": round(elapsed, 1),
            }
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results[key] = {
                "vel": None, "sst": None, "ssh": None,
                "time_s": 0,
            }

        # Save incrementally
        with open(args.results, 'w') as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - t_start_all
    print(f"\n{'='*72}")
    print(f"  COMPLETED: {len(results)} results ({skipped} cached)")
    print(f"  Total wall time: {total_time/60:.1f} min")
    print(f"  Results saved to {args.results}")
    print(f"{'='*72}")

    # ---- Summary table ----
    _print_summary(results, hscale_values_km, alpha_values)


def _print_summary(results, hscale_values_km, alpha_values):
    """Print nicely-formatted RMSE tables + best parameters."""
    import numpy as np

    n_h = len(hscale_values_km)
    n_a = len(alpha_values)
    vel_arr = np.full((n_h, n_a), np.nan)
    sst_arr = np.full((n_h, n_a), np.nan)
    ssh_arr = np.full((n_h, n_a), np.nan)

    for i, h in enumerate(hscale_values_km):
        for j, a in enumerate(alpha_values):
            k = f"h{h}_a{a}"
            if k in results and results[k].get("vel") is not None:
                vel_arr[i, j] = results[k]["vel"]
                sst_arr[i, j] = results[k]["sst"]
                if results[k].get("ssh") is not None:
                    ssh_arr[i, j] = results[k]["ssh"]

    header = (f"{'hscale':>8s} | " +
              " | ".join(f"α={a:.1f} " for a in alpha_values))
    sep = "-" * len(header)

    for name, arr, fmt in [("Mean RMSE_vel (m/s)", vel_arr, ".4f"),
                           ("Mean RMSE_sst (K)",   sst_arr, ".3f"),
                           ("Mean RMSE_ssh (m)",   ssh_arr, ".3f")]:
        print(f"\n  {name}")
        print(header)
        print(sep)
        for i, h in enumerate(hscale_values_km):
            row = f"{h:7d}  | " + " | ".join(
                f"{arr[i,j]:6{fmt}}" if np.isfinite(arr[i, j])
                else "  NaN " for j in range(n_a))
            print(row)

    # Best combined (vel + sst, normalised)
    valid = np.isfinite(vel_arr) & np.isfinite(sst_arr)
    if valid.any():
        vel_norm = ((vel_arr - np.nanmin(vel_arr)) /
                    (np.nanmax(vel_arr) - np.nanmin(vel_arr) + 1e-12))
        sst_norm = ((sst_arr - np.nanmin(sst_arr)) /
                    (np.nanmax(sst_arr) - np.nanmin(sst_arr) + 1e-12))
        combined = vel_norm + sst_norm
        combined[~valid] = np.inf
        best = np.unravel_index(np.argmin(combined), combined.shape)
        print(f"\n  BEST COMBINED PARAMETERS:")
        print(f"    hcovlocal_scale = {hscale_values_km[best[0]]} km")
        print(f"    covinflate1     = {alpha_values[best[1]]}")
        print(f"    RMSE_vel        = {vel_arr[best]:.5f} m/s")
        print(f"    RMSE_sst        = {sst_arr[best]:.3f} K")
        if np.isfinite(ssh_arr[best]):
            print(f"    RMSE_ssh        = {ssh_arr[best]:.3f} m")

        bv = np.unravel_index(np.nanargmin(vel_arr), vel_arr.shape)
        bs = np.unravel_index(np.nanargmin(sst_arr), sst_arr.shape)
        print(f"\n  BEST vel: h={hscale_values_km[bv[0]]}km "
              f"α={alpha_values[bv[1]]}  RMSE_vel={vel_arr[bv]:.5f}")
        print(f"  BEST sst: h={hscale_values_km[bs[0]]}km "
              f"α={alpha_values[bs[1]]}  RMSE_sst={sst_arr[bs]:.3f}")
        if np.isfinite(ssh_arr).any():
            bh = np.unravel_index(np.nanargmin(ssh_arr), ssh_arr.shape)
            print(f"  BEST ssh: h={hscale_values_km[bh[0]]}km "
                  f"α={alpha_values[bh[1]]}  RMSE_ssh={ssh_arr[bh]:.3f}")


if __name__ == '__main__':
    main()

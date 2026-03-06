#!/usr/bin/env python
"""
run_all.py
==========
Master script to run all experiments for the linear Gaussian model.

Usage:
    python run_all.py                    # Run everything
    python run_all.py --steps 1,2,3      # Run only specific steps
    python run_all.py --skip_sensitivity # Skip LETKF sensitivity (use defaults)

Steps:
    1. Generate synthetic data
    2. Run Kalman Filter (reference)
    3. Run LSMCMC V1 (M independent runs)
    4. Run LSMCMC V2 (M independent runs)
    5. Run SMCMC (M independent runs)
    6. Run LETKF sensitivity analysis
    7. Run LETKF with best parameters
    8. Run EnKF
    9. Plot comparisons
"""
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run all linear Gaussian experiments")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated step numbers to run (default: all)")
    parser.add_argument("--skip_sensitivity", action="store_true",
                        help="Skip LETKF sensitivity, use default params")
    parser.add_argument("--M", type=int, default=50,
                        help="Number of independent SMCMC/LSMCMC runs")
    parser.add_argument("--M_single", action="store_true",
                        help="Run only M=1 for quick testing")
    args = parser.parse_args()

    basedir = os.path.dirname(os.path.abspath(__file__))

    if args.M_single:
        args.M = 1

    if args.steps:
        steps = set(int(s) for s in args.steps.split(','))
    else:
        steps = set(range(1, 10))
        if args.skip_sensitivity:
            steps.discard(6)

    # Step 1: Generate data
    if 1 in steps:
        print("\n" + "=" * 60)
        print("STEP 1: Generate synthetic data")
        print("=" * 60)
        from linear_forward_generate_data import main as gen_main
        gen_main(seed=42, outdir=basedir)

    # Step 2: Kalman Filter
    if 2 in steps:
        print("\n" + "=" * 60)
        print("STEP 2: Kalman Filter (reference)")
        print("=" * 60)
        from linear_forward_run_kf import main as kf_main
        kf_main(outdir=basedir)

    # Step 3: LSMCMC V1
    if 3 in steps:
        print("\n" + "=" * 60)
        print(f"STEP 3: LSMCMC V1 (M={args.M})")
        print("=" * 60)
        from linear_forward_run_lsmcmc_v1 import run_lsmcmc_v1
        import numpy as np

        data_file = os.path.join(basedir, "linear_gaussian_data.npz")
        dat = np.load(data_file)
        T = int(dat['T']); d = int(dat['d'])

        all_means = np.zeros((args.M, T + 1, d))
        all_elapsed = np.zeros(args.M)

        for isim in range(args.M):
            print(f"\n--- LSMCMC-V1 run {isim+1}/{args.M} ---")
            outfile = run_lsmcmc_v1(data_file=data_file, outdir=basedir,
                                     seed=1000 + isim, Gamma=900, Na=500, Nf=52)
            res = np.load(outfile)
            all_means[isim] = res['lsmcmc_mean']
            all_elapsed[isim] = float(res['elapsed'])

        avg_mean = np.mean(all_means, axis=0)
        outfile_avg = os.path.join(basedir, "linear_gaussian_lsmcmc_v1_avg.npz")
        np.savez(outfile_avg, lsmcmc_mean=avg_mean, all_means=all_means,
                 T=T, d=d, M=args.M, elapsed_mean=np.mean(all_elapsed),
                 elapsed_all=all_elapsed)
        print(f"LSMCMC V1 averaged: {outfile_avg}")

    # Step 4: LSMCMC V2
    if 4 in steps:
        print("\n" + "=" * 60)
        print(f"STEP 4: LSMCMC V2 (M={args.M})")
        print("=" * 60)
        from linear_forward_run_lsmcmc_v2 import run_lsmcmc_v2
        import numpy as np

        data_file = os.path.join(basedir, "linear_gaussian_data.npz")
        v2_outdir = os.path.join(basedir, "lsmcmc_v2_output")
        os.makedirs(v2_outdir, exist_ok=True)
        dat = np.load(data_file)
        T = int(dat['T']); d = int(dat['d'])

        all_means = np.zeros((args.M, T + 1, d))
        all_elapsed = np.zeros(args.M)

        for isim in range(args.M):
            print(f"\n--- LSMCMC-V2 run {isim+1}/{args.M} ---")
            sim_outdir = os.path.join(v2_outdir, f"sim_{isim:03d}")
            os.makedirs(sim_outdir, exist_ok=True)
            outfile, mean_arr, elapsed = run_lsmcmc_v2(
                data_file=data_file, outdir=sim_outdir,
                seed=2000 + isim, Gamma=900, Na=500,
                Nf=52, r_loc=10.0, ncores=1)
            all_means[isim] = mean_arr
            all_elapsed[isim] = elapsed

        avg_mean = np.mean(all_means, axis=0)
        outfile_avg = os.path.join(v2_outdir, "linear_gaussian_lsmcmc_v2_avg.npz")
        np.savez(outfile_avg, lsmcmc_mean=avg_mean, all_means=all_means,
                 T=T, d=d, M=args.M, elapsed_mean=np.mean(all_elapsed),
                 elapsed_all=all_elapsed)
        print(f"LSMCMC V2 averaged: {outfile_avg}")

    # Step 5: SMCMC
    if 5 in steps:
        print("\n" + "=" * 60)
        print(f"STEP 5: SMCMC (M={args.M})")
        print("=" * 60)
        from linear_forward_run_smcmc import run_smcmc
        import numpy as np

        data_file = os.path.join(basedir, "linear_gaussian_data.npz")
        dat = np.load(data_file)
        T = int(dat['T']); d = int(dat['d'])

        all_means = np.zeros((args.M, T + 1, d))
        all_elapsed = np.zeros(args.M)

        for isim in range(args.M):
            print(f"\n--- SMCMC run {isim+1}/{args.M} ---")
            outfile = run_smcmc(data_file=data_file, outdir=basedir,
                                seed=3000 + isim, Na=1000, Nf=52)
            res = np.load(outfile)
            all_means[isim] = res['smcmc_mean']
            all_elapsed[isim] = float(res['elapsed'])

        avg_mean = np.mean(all_means, axis=0)
        outfile_avg = os.path.join(basedir, "linear_gaussian_smcmc_avg.npz")
        np.savez(outfile_avg, smcmc_mean=avg_mean, all_means=all_means,
                 T=T, d=d, M=args.M, elapsed_mean=np.mean(all_elapsed),
                 elapsed_all=all_elapsed)
        print(f"SMCMC averaged: {outfile_avg}")

    # Step 6: LETKF sensitivity
    if 6 in steps:
        print("\n" + "=" * 60)
        print("STEP 6: LETKF sensitivity analysis")
        print("=" * 60)
        sys.argv = [
            'run_letkf_sensitivity.py',
            '--data', os.path.join(basedir, "linear_gaussian_data.npz"),
            '--kf', os.path.join(basedir, "linear_gaussian_kf.npz"),
            '--outdir', basedir,
            '--K', '50',
        ]
        from linear_forward_run_letkf_sensitivity import main as sens_main
        sens_main()

    # Step 7: LETKF with best params
    if 7 in steps:
        print("\n" + "=" * 60)
        print("STEP 7: LETKF (best parameters)")
        print("=" * 60)
        sys.argv = [
            'run_letkf.py',
            '--data', os.path.join(basedir, "linear_gaussian_data.npz"),
            '--outdir', basedir,
            '--use_best',
        ]
        from linear_forward_run_letkf import main as letkf_main
        letkf_main()

    # Step 8: EnKF
    if 8 in steps:
        print("\n" + "=" * 60)
        print("STEP 8: EnKF")
        print("=" * 60)
        from linear_forward_run_enkf import run_enkf
        run_enkf(data_file=os.path.join(basedir, "linear_gaussian_data.npz"),
                 outdir=basedir, K=50)

    # Step 9: Plots
    if 9 in steps:
        print("\n" + "=" * 60)
        print("STEP 9: Comparison plots")
        print("=" * 60)
        sys.argv = ['plot_comparison.py', '--basedir', basedir]
        from linear_forward_plot_comparison import main as plot_main
        plot_main()

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

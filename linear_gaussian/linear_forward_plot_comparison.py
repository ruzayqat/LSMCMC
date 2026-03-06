#!/usr/bin/env python
"""
plot_comparison.py
==================
Compare LSMCMC V1, V2, LETKF against the KF reference for the linear
Gaussian model.

Plots generated:
1. Coordinate time series (e.g. coord 50) — KF vs each filter
2. Snapshot of the full grid at a chosen assimilation time
3. Bar chart: % of |errors| < sigma_y/2 and computational time
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Publication-quality font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})

# Global mapping from internal keys to display labels
DISPLAY_NAMES = {
    'KF': 'KF',
    'LSMCMC_V1': 'LSMCMC V1',
    'LSMCMC_V2': 'LSMCMC V2',
    'SMCMC': 'SMCMC',
    'EnKF': 'EnKF',
    'LETKF': 'LETKF',
}

def _dn(key):
    """Return display name for a filter key."""
    return DISPLAY_NAMES.get(key, key)


def load_results(basedir):
    """Load all available result files."""
    results = {}

    # KF
    kf_file = os.path.join(basedir, "linear_gaussian_kf.npz")
    if os.path.exists(kf_file):
        kf = np.load(kf_file)
        results['KF'] = {'mean': kf['kf_mean'], 'elapsed': float(kf['elapsed'])}

    # Data (for truth and sigma_y)
    dat_file = os.path.join(basedir, "linear_gaussian_data.npz")
    if os.path.exists(dat_file):
        dat = np.load(dat_file)
        results['_data'] = {
            'Z_truth': dat['Z_truth'],
            'sigma_y': float(dat['sigma_y']),
            'T': int(dat['T']),
            'd': int(dat['d']),
            'Ngx': int(dat['Ngx']),
            'Ngy': int(dat['Ngy']),
            'obs_inds': dat['obs_inds'],
            'nobs': dat['nobs'],
        }

    # LSMCMC V1 (averaged) — prefer new subfolder output
    v1_file = os.path.join(basedir, "lsmcmc_v1_output",
                           "linear_gaussian_lsmcmc_v1_avg.npz")
    if not os.path.exists(v1_file):
        v1_file = os.path.join(basedir, "linear_gaussian_lsmcmc_v1_avg.npz")
    if not os.path.exists(v1_file):
        v1_file = os.path.join(basedir, "linear_gaussian_lsmcmc_v1.npz")
    if os.path.exists(v1_file):
        v1 = np.load(v1_file)
        # Wall-clock time ≈ max of parallel runs
        if 'elapsed_all' in v1:
            v1_elapsed = float(np.max(v1['elapsed_all']))
        else:
            v1_elapsed = float(v1.get('elapsed_mean', v1.get('elapsed', 0)))
        results['LSMCMC_V1'] = {
            'mean': v1['lsmcmc_mean'],
            'elapsed': v1_elapsed,
        }

    # LSMCMC V2 (averaged) — prefer new subfolder output
    v2_file = os.path.join(basedir, "lsmcmc_v2_output",
                           "linear_gaussian_lsmcmc_v2_avg.npz")
    if not os.path.exists(v2_file):
        v2_file = os.path.join(basedir, "linear_gaussian_lsmcmc_v2_avg.npz")
    if not os.path.exists(v2_file):
        v2_file = os.path.join(basedir, "linear_gaussian_lsmcmc_v2.npz")
    if os.path.exists(v2_file):
        v2 = np.load(v2_file)
        # Wall-clock time ≈ max of parallel runs
        if 'elapsed_all' in v2:
            v2_elapsed = float(np.max(v2['elapsed_all']))
        else:
            v2_elapsed = float(v2.get('elapsed_mean', v2.get('elapsed', 0)))
        results['LSMCMC_V2'] = {
            'mean': v2['lsmcmc_mean'],
            'elapsed': v2_elapsed,
        }

    # LETKF
    le_file = os.path.join(basedir, "linear_gaussian_letkf.npz")
    if os.path.exists(le_file):
        le = np.load(le_file)
        results['LETKF'] = {
            'mean': le['letkf_mean'],
            'elapsed': float(le['elapsed']),
        }

    # SMCMC (no localization)
    sm_file = os.path.join(basedir, "linear_gaussian_smcmc_avg.npz")
    if not os.path.exists(sm_file):
        sm_file = os.path.join(basedir, "linear_gaussian_smcmc.npz")
    if os.path.exists(sm_file):
        sm = np.load(sm_file)
        results['SMCMC'] = {
            'mean': sm['smcmc_mean'],
            'elapsed': float(sm.get('elapsed_mean', sm.get('elapsed', 0))),
        }

    # EnKF
    ek_file = os.path.join(basedir, "linear_gaussian_enkf.npz")
    if os.path.exists(ek_file):
        ek = np.load(ek_file)
        results['EnKF'] = {
            'mean': ek['enkf_mean'],
            'elapsed': float(ek['elapsed']),
        }

    return results


def plot_coord_timeseries(results, plotdir, coord=50):
    """Plot filter mean at a single coordinate vs KF."""
    if 'KF' not in results or '_data' not in results:
        return

    T = results['_data']['T']
    kf_mean = results['KF']['mean']
    sigma_y = results['_data']['sigma_y']
    times = np.arange(T + 1)

    filters = ['LSMCMC_V1', 'LSMCMC_V2', 'SMCMC', 'EnKF', 'LETKF']
    colors = {'LSMCMC_V1': 'C1', 'LSMCMC_V2': 'C2', 'SMCMC': 'C4',
              'EnKF': 'C5', 'LETKF': 'C3'}
    present = [f for f in filters if f in results]

    fig, axes = plt.subplots(len(present), 1, figsize=(12, 3 * len(present)),
                             sharex=True)
    if len(present) == 1:
        axes = [axes]

    for ax, fname in zip(axes, present):
        fmean = results[fname]['mean']
        ax.plot(times, kf_mean[:, coord], 'b-', lw=1.5, label='KF')
        ax.plot(times, fmean[:, coord], color=colors[fname], lw=1.0,
                alpha=0.8, label=_dn(fname))
        ax.set_ylabel(f'$Z_{{k}}^{{({coord})}}$')
        ax.legend(loc='upper right')
        ax.set_title(f'{_dn(fname)} vs KF at coordinate {coord}')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Assimilation cycle $k$')
    plt.tight_layout()
    figpath = os.path.join(plotdir, f"coord{coord}_KF_vs_filters.png")
    plt.savefig(figpath, dpi=150)
    plt.close()
    print(f"  Saved {figpath}")


def plot_coord_individual(results, plotdir, coord=50):
    """Separate plot for each filter (matching paper style)."""
    if 'KF' not in results or '_data' not in results:
        return

    T = results['_data']['T']
    kf_mean = results['KF']['mean']
    times = np.arange(T + 1)

    filters = ['LSMCMC_V1', 'LSMCMC_V2', 'SMCMC', 'EnKF', 'LETKF']
    colors = {'LSMCMC_V1': 'C1', 'LSMCMC_V2': 'C2', 'SMCMC': 'C4',
              'EnKF': 'C5', 'LETKF': 'C3'}

    for fname in filters:
        if fname not in results:
            continue
        fmean = results[fname]['mean']
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(times, kf_mean[:, coord], 'b-', lw=1.5, label='KF')
        ax.plot(times, fmean[:, coord], color=colors[fname], lw=1.0,
                alpha=0.8, label=_dn(fname))
        ax.set_xlabel('Assimilation cycle $k$')
        ax.set_ylabel(f'$Z_{{k}}^{{({coord})}}$')
        ax.legend(loc='upper right')
        ax.set_title(f'KF vs {_dn(fname)} — coordinate {coord}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figpath = os.path.join(plotdir,
                               f"coord{coord}_KF_vs_{fname}.png")
        plt.savefig(figpath, dpi=150)
        plt.close()
        print(f"  Saved {figpath}")


def plot_snapshot(results, plotdir, cycle=22):
    """Snapshot of the full grid at a given assimilation time."""
    if 'KF' not in results or '_data' not in results:
        return

    data = results['_data']
    Ngy, Ngx = data['Ngy'], data['Ngx']

    filters = ['KF', 'LSMCMC_V1', 'LSMCMC_V2', 'SMCMC', 'EnKF', 'LETKF']
    present = [f for f in filters if f in results]
    n = len(present)

    kf_grid = results['KF']['mean'][cycle].reshape(Ngy, Ngx)

    # Colour range from KF
    vmin, vmax = kf_grid.min(), kf_grid.max()

    # Filters that are not KF (for difference row)
    non_kf = [f for f in present if f != 'KF']

    # Compute a single global dmax across all non-KF differences
    global_dmax = 1e-10
    for fname in non_kf:
        diff = results[fname]['mean'][cycle].reshape(Ngy, Ngx) - kf_grid
        global_dmax = max(global_dmax, abs(diff.min()), abs(diff.max()))

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))

    for j, fname in enumerate(present):
        fgrid = results[fname]['mean'][cycle].reshape(Ngy, Ngx)

        ax1 = axes[0, j]
        im1 = ax1.imshow(fgrid, origin='lower', cmap='RdBu_r',
                         vmin=vmin, vmax=vmax)
        ax1.set_title(_dn(fname))
        ax1.set_xticks([]); ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = axes[1, j]
        if fname == 'KF':
            # Remove the KF−KF subplot (trivially zero)
            ax2.set_visible(False)
        else:
            diff = fgrid - kf_grid
            im2 = ax2.imshow(diff, origin='lower', cmap='bwr',
                             vmin=-global_dmax, vmax=global_dmax)
            ax2.set_title(f'{_dn(fname)} − KF')
            ax2.set_xticks([]); ax2.set_yticks([])
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(f'Snapshot at cycle {cycle}', y=1.02)
    plt.tight_layout()
    figpath = os.path.join(plotdir, f"snapshot_cycle{cycle}.png")
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {figpath}")


def plot_metrics(results, plotdir):
    """Bar chart of % errors < sigma_y/2 and computational time."""
    if 'KF' not in results or '_data' not in results:
        return

    data = results['_data']
    sigma_y = data['sigma_y']
    T, d = data['T'], data['d']
    kf_mean = results['KF']['mean']

    filters = ['LSMCMC_V1', 'LSMCMC_V2', 'SMCMC', 'EnKF', 'LETKF']
    present = [f for f in filters if f in results]
    if not present:
        return

    pcts = []
    times_list = []
    labels = []
    colors = {'LSMCMC_V1': 'C1', 'LSMCMC_V2': 'C2', 'SMCMC': 'C4',
              'EnKF': 'C5', 'LETKF': 'C3'}
    for fname in present:
        fmean = results[fname]['mean']
        abs_err = np.abs(fmean[1:] - kf_mean[1:])  # (T, d)
        pct = 100.0 * np.mean(abs_err < sigma_y / 2.0)
        pcts.append(pct)
        times_list.append(results[fname]['elapsed'])
        labels.append(_dn(fname))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(labels))
    bar_colors = [colors.get(f, 'C0') for f in present]

    ax1.bar(x, pcts, color=bar_colors, alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel('% of |errors| < $\\sigma_y/2$')
    ax1.set_title('Accuracy vs KF')
    ax1.set_ylim([0, 105])
    for i, v in enumerate(pcts):
        ax1.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)

    ax2.bar(x, times_list, color=bar_colors, alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Computational time')
    for i, v in enumerate(times_list):
        ax2.text(i, v + 1, f"{v:.1f}s", ha='center', fontsize=12)

    plt.tight_layout()
    figpath = os.path.join(plotdir, "metrics_comparison.png")
    plt.savefig(figpath, dpi=150)
    plt.close()
    print(f"  Saved {figpath}")


def plot_obs_swaths(results, plotdir, frames=[5, 10, 13]):
    """Plot accumulated observation swaths (matching paper Figure)."""
    if '_data' not in results:
        return

    data = results['_data']
    T = data['T']
    Ngy, Ngx = data['Ngy'], data['Ngx']
    obs_inds = data['obs_inds']
    nobs = data['nobs']

    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 4.5))
    if len(frames) == 1:
        axes = [axes]

    for ax, frame in zip(axes, frames):
        grid = np.zeros((Ngy, Ngx))

        # Current obs only (blue)
        mk = int(nobs[frame])
        idx = obs_inds[frame, :mk]
        valid = idx >= 0
        iy, ix = np.unravel_index(idx[valid], (Ngy, Ngx))
        grid[iy, ix] = 1

        # Custom colormap: white (unobserved), blue (current swath)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['white', 'royalblue'])
        ax.imshow(grid, origin='lower', cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f'Cycle {frame + 1}')
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')

    plt.suptitle('SWOT-like observation swaths', y=1.02)
    plt.tight_layout()
    figpath = os.path.join(plotdir, "obs_swaths.png")
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {figpath}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default=None)
    parser.add_argument("--coord", type=int, default=50)
    parser.add_argument("--snapshot_cycle", type=int, default=22)
    args = parser.parse_args()

    if args.basedir is None:
        args.basedir = os.path.dirname(os.path.abspath(__file__))

    plotdir = os.path.join(args.basedir, "plots")
    os.makedirs(plotdir, exist_ok=True)

    results = load_results(args.basedir)

    print(f"[Plot] Available results: {[k for k in results if not k.startswith('_')]}")
    print(f"[Plot] Output dir: {plotdir}")

    # Generate all plots
    plot_obs_swaths(results, plotdir)
    plot_coord_timeseries(results, plotdir, coord=args.coord)
    plot_coord_individual(results, plotdir, coord=args.coord)
    plot_snapshot(results, plotdir, cycle=args.snapshot_cycle)
    plot_metrics(results, plotdir)

    print("[Plot] Done.")


if __name__ == "__main__":
    main()

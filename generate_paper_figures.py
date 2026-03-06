#!/usr/bin/env python
"""
generate_paper_figures.py
=========================
Generate publication-quality figures for the LSMCMC paper.
All figures use large fonts, proper axis labels, and are saved as PDF for LaTeX.

Figures produced (paper_figures/):
  LG:
    lg_rmse_timeseries.pdf   - RMSE against KF
    lg_rmse_vs_truth.pdf     - RMSE against truth (all methods)
    lg_snapshot.pdf           - KF / V1 / LETKF at cycles 50, 100
    lg_coord50.pdf            - time series at coord 50
    lg_obs_swaths.pdf         - SWOT-like observation pattern
  MLSWE linear:
    mlswe_obs_pattern.pdf     - drifter + SWOT scatter
    ldata_compare_rmse.pdf    - 3-panel vel/SST/SSH comparison
    ldata_v1_fields.pdf       - SSH/U/V/HYCOM/SST field panels
  MLSWE NL twin:
    nltwin_compare_rmse.pdf   - 3-panel comparison (+ LETKF if available)
    nltwin_v1_fields.pdf      - field panels
  MLSWE NL real:
    nlreal_compare_rmse.pdf   - 3-panel comparison (+ LETKF 46 cycles)
    nlreal_v1_fields.pdf      - field panels
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset

# ── Publication-quality global settings ──────────────────────────────────
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_figures')
BASEDIR = os.path.dirname(os.path.abspath(__file__))
LON_RANGE = (-60, -20)
LAT_RANGE = (10, 45)


def _fill_nan_nearest(arr_2d):
    """Fill NaN in a 2-D array with nearest finite value."""
    mask = np.isnan(arr_2d)
    if not mask.any():
        return arr_2d
    if mask.all():
        arr_2d[:] = 0.0
        return arr_2d
    ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
    arr_2d[mask] = arr_2d[tuple(ind[:, mask])]
    return arr_2d


def savefig(fig, name):
    """Save figure as PDF to OUTDIR."""
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
#  MLSWE HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

def load_mlswe_analysis(nc_path):
    """Load MLSWE analysis output. Returns data dict."""
    with Dataset(nc_path, 'r') as nc:
        raw = np.asarray(nc.variables['lsmcmc_mean'][:])  # (time, 3, 4, ny, nx)
        rmse_vel = np.asarray(nc.variables['rmse_vel'][:])
        rmse_sst = np.asarray(nc.variables['rmse_sst'][:])
        rmse_ssh = np.asarray(nc.variables['rmse_ssh'][:]) if 'rmse_ssh' in nc.variables else None
        obs_times = np.asarray(nc.variables['obs_times'][:]) if 'obs_times' in nc.variables else None
        ny = int(nc.ny) if hasattr(nc, 'ny') else raw.shape[3]
        nx = int(nc.nx) if hasattr(nc, 'nx') else raw.shape[4]
        H_b = np.asarray(nc.variables['H_b'][:]) if 'H_b' in nc.variables else None
    return {
        'raw': raw,
        'rmse_vel': rmse_vel,
        'rmse_sst': rmse_sst,
        'rmse_ssh': rmse_ssh,
        'obs_times': obs_times,
        'ny': ny, 'nx': nx,
        'H_b': H_b,
    }


def load_mlswe_obs(nc_path):
    """Load MLSWE merged observations."""
    with Dataset(nc_path, 'r') as nc:
        yobs = np.asarray(nc.variables['yobs_all'][:])
        yind = np.asarray(nc.variables['yobs_ind_all'][:])
        yind0 = np.asarray(nc.variables['yobs_ind_level0_all'][:])
        times = np.asarray(nc.variables['obs_times'][:])
    return {'yobs': yobs, 'yind': yind, 'yind0': yind0, 'times': times}


def extract_ssh(data, cycle, H_b=None):
    """Extract SSH anomaly at a given cycle."""
    raw = data['raw']
    ny, nx = data['ny'], data['nx']
    h_total = raw[cycle, :, 0, :, :].sum(axis=0)
    if H_b is not None:
        ssh = h_total - H_b.reshape(ny, nx)
    else:
        ssh = h_total - 4000.0
    return ssh


def extract_sst(data, cycle):
    """Extract SST at a given cycle (Kelvin)."""
    return data['raw'][cycle, 0, 3, :, :]


def smooth_ocean(field, H_b, sigma=2.0):
    """Gaussian-smooth an ocean field, masking land (H_b < 200 m)."""
    ocean_mask = H_b >= 200.0 if H_b is not None else np.ones_like(field, dtype=bool)
    filled = field.copy()
    if ocean_mask.any():
        filled[~ocean_mask] = np.nanmean(field[ocean_mask])
    else:
        filled[~ocean_mask] = 0.0
    smoothed = gaussian_filter(filled, sigma=sigma)
    return np.ma.masked_where(~ocean_mask, smoothed)


def load_hycom_reanalysis(bc_path, model_lon, model_lat, model_time_sec):
    """Load HYCOM reanalysis and interpolate to model grid at a given time."""
    with Dataset(bc_path, 'r') as nc:
        bc_lat = np.asarray(nc.variables['lat'][:], dtype=np.float64)
        bc_lon = np.asarray(nc.variables['lon'][:], dtype=np.float64)
        bc_times = np.asarray(nc.variables['time'][:], dtype=np.float64)
        bc_ssh = np.asarray(nc.variables['ssh'][:], dtype=np.float64)
        bc_uo = np.asarray(nc.variables['uo'][:], dtype=np.float64)
        bc_vo = np.asarray(nc.variables['vo'][:], dtype=np.float64)
        bc_sst = np.asarray(nc.variables['sst'][:], dtype=np.float64)

    for arr3d in [bc_ssh, bc_uo, bc_vo, bc_sst]:
        for t in range(arr3d.shape[0]):
            _fill_nan_nearest(arr3d[t])
    if np.nanmean(bc_sst) < 100.0:
        bc_sst += 273.15

    t_idx = np.interp(model_time_sec, bc_times, np.arange(len(bc_times)))
    t_lo = int(np.floor(t_idx))
    t_hi = min(t_lo + 1, len(bc_times) - 1)
    alpha = t_idx - t_lo

    result = {}
    mg_lat, mg_lon = np.meshgrid(model_lat, model_lon, indexing='ij')
    for name, field3d in [('ssh', bc_ssh), ('u', bc_uo),
                          ('v', bc_vo), ('sst', bc_sst)]:
        snap = ((1 - alpha) * field3d[t_lo] + alpha * field3d[t_hi]
                if t_lo != t_hi else field3d[t_lo])
        interp = RegularGridInterpolator(
            (bc_lat, bc_lon), snap,
            method='linear', bounds_error=False, fill_value=np.nan)
        result[name] = interp((mg_lat, mg_lon))
    return result


def parse_letkf_nl_log(log_path):
    """Parse LETKF NL run log to extract per-cycle RMSE arrays."""
    if not os.path.exists(log_path):
        return None
    pattern = re.compile(
        r'\[(\d+)/\d+\].*vel=([\d.eE+-]+)\s+sst=([\d.]+)K\s+ssh=([\d.]+)m')
    vel_list, sst_list, ssh_list = [], [], []
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                vel_list.append(float(m.group(2)))
                sst_list.append(float(m.group(3)))
                ssh_list.append(float(m.group(4)))
    if not vel_list:
        return None
    return {
        'rmse_vel': np.array(vel_list),
        'rmse_sst': np.array(sst_list),
        'rmse_ssh': np.array(ssh_list),
    }


# ════════════════════════════════════════════════════════════════════════
#  LINEAR GAUSSIAN FIGURES
# ════════════════════════════════════════════════════════════════════════

def generate_lg_figures():
    """Generate all linear Gaussian experiment figures."""
    print("\n=== Linear Gaussian Figures ===")
    lgdir = os.path.join(BASEDIR, 'linear_gaussian')

    dat = np.load(os.path.join(lgdir, 'linear_gaussian_data.npz'))
    Z_truth = dat['Z_truth']
    T = int(dat['T'])
    d = int(dat['d'])
    Ngx = int(dat['Ngx'])
    Ngy = int(dat['Ngy'])
    obs_inds = dat['obs_inds']

    kf = np.load(os.path.join(lgdir, 'linear_gaussian_kf.npz'))
    kf_mean = kf['kf_mean']

    # ── Load M=1 filters ─────────────────────────────────────────────
    filters_m1 = {}
    for key, fname_pref in [('LSMCMC V1', 'lsmcmc_v1'), ('LSMCMC V2', 'lsmcmc_v2')]:
        f = os.path.join(lgdir, f'linear_gaussian_{fname_pref}.npz')
        if os.path.exists(f):
            data = np.load(f)
            filters_m1[key] = data['lsmcmc_mean']

    letkf_f = os.path.join(lgdir, 'linear_gaussian_letkf_K50.npz')
    letkf_mean = None
    if os.path.exists(letkf_f):
        letkf_data = np.load(letkf_f)
        letkf_mean = letkf_data['letkf_mean']

    # ── Load M=4 averaged filters ────────────────────────────────────
    filters_m4 = {}
    for key, fname_pref in [('LSMCMC V1', 'lsmcmc_v1'), ('LSMCMC V2', 'lsmcmc_v2')]:
        f = os.path.join(lgdir, f'linear_gaussian_{fname_pref}_avg.npz')
        if os.path.exists(f):
            data = np.load(f)
            filters_m4[key] = data['lsmcmc_mean']

    colors = {'LSMCMC V1': '#1f77b4', 'LSMCMC V2': '#ff7f0e',
              'LETKF': '#2ca02c'}

    # ── Fig 1: RMSE vs KF time series (M=1 + M=4 + LETKF) ──────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, fmean in filters_m1.items():
        rmse = np.sqrt(np.mean((fmean - kf_mean)**2, axis=1))
        ax.plot(range(T + 1), rmse, label=f'{key} ($M{{=}}1$)',
                color=colors[key], linewidth=2, linestyle='-')
    for key, fmean in filters_m4.items():
        rmse = np.sqrt(np.mean((fmean - kf_mean)**2, axis=1))
        ax.plot(range(T + 1), rmse, label=f'{key} ($M{{=}}4$)',
                color=colors[key], linewidth=2, linestyle='--')
    if letkf_mean is not None:
        rmse = np.sqrt(np.mean((letkf_mean - kf_mean)**2, axis=1))
        ax.plot(range(T + 1), rmse, label='LETKF ($K{=}50$)',
                color=colors['LETKF'], linewidth=2)
    ax.set_xlabel('Assimilation Cycle')
    ax.set_ylabel('RMSE vs Kalman Filter')
    ax.set_title('RMSE against Exact KF Solution')
    ax.legend(frameon=True, fancybox=True, shadow=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, 'lg_rmse_timeseries.pdf')

    # ── Fig 1b: RMSE vs Truth (all methods including KF) ─────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    rmse_kf = np.sqrt(np.mean((kf_mean - Z_truth)**2, axis=1))
    ax.plot(range(T + 1), rmse_kf, label='KF', color='gray', linewidth=2)
    for key, fmean in filters_m1.items():
        rmse = np.sqrt(np.mean((fmean - Z_truth)**2, axis=1))
        ax.plot(range(T + 1), rmse, label=f'{key} ($M{{=}}1$)',
                color=colors[key], linewidth=2, linestyle='-')
    for key, fmean in filters_m4.items():
        rmse = np.sqrt(np.mean((fmean - Z_truth)**2, axis=1))
        ax.plot(range(T + 1), rmse, label=f'{key} ($M{{=}}4$)',
                color=colors[key], linewidth=2, linestyle='--')
    if letkf_mean is not None:
        rmse = np.sqrt(np.mean((letkf_mean - Z_truth)**2, axis=1))
        ax.plot(range(T + 1), rmse, label='LETKF ($K{=}50$)',
                color=colors['LETKF'], linewidth=2)
    ax.set_xlabel('Assimilation Cycle')
    ax.set_ylabel('RMSE vs Truth')
    ax.set_title('RMSE against Truth')
    ax.legend(frameon=True, fancybox=True, shadow=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, 'lg_rmse_vs_truth.pdf')

    # ── Fig 2: Snapshot at cycles 50 and 100 (4×3 grid) ─────────────
    #   Row 0 (cycle 50): KF,       V1 M=1,   V2 M=1
    #   Row 1 (cycle 50): LETKF,    V1 M=4,   V2 M=4
    #   Row 2 (cycle 100): KF,      V1 M=1,   V2 M=1
    #   Row 3 (cycle 100): LETKF,   V1 M=4,   V2 M=4
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    cycle_mid, cycle_final = 50, T

    snapshot_grid = [
        # (row, col, data_array, label)
        # ── cycle 50 ──
        (0, 0, kf_mean,                          'KF'),
        (0, 1, filters_m1.get('LSMCMC V1'),      'V1 ($M{=}1$)'),
        (0, 2, filters_m1.get('LSMCMC V2'),      'V2 ($M{=}1$)'),
        (1, 0, letkf_mean,                       'LETKF ($K{=}50$)'),
        (1, 1, filters_m4.get('LSMCMC V1'),      'V1 ($M{=}4$)'),
        (1, 2, filters_m4.get('LSMCMC V2'),      'V2 ($M{=}4$)'),
        # ── cycle 100 ──
        (2, 0, kf_mean,                          'KF'),
        (2, 1, filters_m1.get('LSMCMC V1'),      'V1 ($M{=}1$)'),
        (2, 2, filters_m1.get('LSMCMC V2'),      'V2 ($M{=}1$)'),
        (3, 0, letkf_mean,                       'LETKF ($K{=}50$)'),
        (3, 1, filters_m4.get('LSMCMC V1'),      'V1 ($M{=}4$)'),
        (3, 2, filters_m4.get('LSMCMC V2'),      'V2 ($M{=}4$)'),
    ]

    for block_idx, cyc in enumerate([cycle_mid, cycle_final]):
        row_offset = block_idx * 2
        kf_grid = kf_mean[cyc].reshape(Ngy, Ngx)
        vmin, vmax = np.nanmin(kf_grid), np.nanmax(kf_grid)

        for r, c, arr, lbl in snapshot_grid:
            if r < row_offset or r >= row_offset + 2:
                continue
            if arr is not None:
                grid = arr[cyc].reshape(Ngy, Ngx)
            else:
                grid = np.zeros((Ngy, Ngx))
            im = axes[r, c].imshow(grid, origin='lower', vmin=vmin, vmax=vmax, cmap='RdBu_r')
            cyc_label = f'Cycle {cyc}'
            axes[r, c].set_title(f'{lbl} ({cyc_label})')
            plt.colorbar(im, ax=axes[r, c], shrink=0.8)

    for ax in axes.ravel():
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
    fig.tight_layout()
    savefig(fig, 'lg_snapshot.pdf')

    # ── Fig 3: Time series at the most-observed coordinate (M=4) ──
    # Find most-observed grid point
    obs_counts = np.zeros(d, dtype=int)
    for k in range(T):
        oi = obs_inds[k]
        valid = oi[oi >= 0]
        obs_counts[valid] += 1
    coord = int(np.argmax(obs_counts))
    n_obs_coord = obs_counts[coord]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(T + 1), kf_mean[:, coord], color='gray', label='KF', linewidth=2)
    for key, fmean in filters_m4.items():
        ax.plot(range(T + 1), fmean[:, coord], label=f'{key} ($M{{=}}4$)',
                color=colors.get(key, 'gray'), linewidth=1.5)
    if letkf_mean is not None:
        ax.plot(range(T + 1), letkf_mean[:, coord], label='LETKF ($K{=}50$)',
                color=colors['LETKF'], linewidth=1.5)
    ax.set_xlabel('Assimilation Cycle')
    ax.set_ylabel('State Value')
    ax.set_title(f'Time Series at Most-Observed Grid Point ({coord}, observed {n_obs_coord}/{T} cycles)')
    ax.legend(frameon=True, fancybox=True, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, 'lg_coord50.pdf')

    # ── Fig 4: Observation swaths (2×2 panel) ──────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, cyc in enumerate([0, 3, 7, 15]):
        r, c = divmod(idx, 2)
        if cyc < len(obs_inds):
            obs_i = obs_inds[cyc]
            grid = np.zeros(d)
            grid[obs_i] = 1.0
            axes[r, c].imshow(grid.reshape(Ngy, Ngx), origin='lower', cmap='Blues', vmin=0, vmax=1)
            axes[r, c].set_title(f'Cycle {cyc + 1} ({len(obs_i)} obs)', fontsize=18)
            axes[r, c].set_xlabel('$x$', fontsize=16)
            axes[r, c].set_ylabel('$y$', fontsize=16)
            axes[r, c].tick_params(labelsize=14)
    fig.suptitle('SWOT-like Swath Observation Pattern', fontsize=20, y=1.02)
    fig.tight_layout()
    savefig(fig, 'lg_obs_swaths.pdf')

    # ── Fig 5: LETKF sensitivity heatmap ─────────────────────────────
    sens_file = os.path.join(lgdir, 'letkf_sensitivity_results.json')
    if os.path.exists(sens_file):
        import json
        with open(sens_file) as f:
            sens_raw = json.load(f)
        sens_list = sens_raw['results']

        hscales = sorted(set(d['hscale'] for d in sens_list))
        alphas  = sorted(set(d['covinflate1'] for d in sens_list))
        nh, na = len(hscales), len(alphas)

        # Build RMSE grid  (hscale × alpha)
        rmse_grid = np.full((nh, na), np.nan)
        lookup = {(d['hscale'], d['covinflate1']): d['rmse'] for d in sens_list}
        for i, h in enumerate(hscales):
            for j, a in enumerate(alphas):
                if (h, a) in lookup:
                    rmse_grid[i, j] = lookup[(h, a)]

        fig, ax = plt.subplots(figsize=(8, 5.5))
        im = ax.imshow(rmse_grid, origin='lower', aspect='auto',
                        cmap='viridis_r',
                        vmin=np.nanmin(rmse_grid),
                        vmax=np.nanpercentile(rmse_grid, 95))
        ax.set_xticks(range(na))
        ax.set_xticklabels([f'{a:.2f}' for a in alphas], rotation=45, ha='right',
                           fontsize=10)
        ax.set_yticks(range(nh))
        ax.set_yticklabels([f'{h:g}' for h in hscales], fontsize=11)
        ax.set_xlabel(r'Inflation factor $\alpha$')
        ax.set_ylabel(r'Localization radius $h_{\mathrm{loc}}$')
        ax.set_title(r'LETKF RMSE vs KF ($K{=}50$)')
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label('RMSE vs KF')

        # Mark the best cell with a star
        best = min(sens_list, key=lambda d: d['rmse'])
        bi = hscales.index(best['hscale'])
        bj = alphas.index(best['covinflate1'])
        ax.plot(bj, bi, marker='*', markersize=18, color='white',
                markeredgecolor='black', markeredgewidth=1.0)
        ax.annotate(f'{best["rmse"]:.4f}',
                    xy=(bj, bi), xytext=(bj + 1.2, bi + 0.8),
                    fontsize=10, color='white', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

        fig.tight_layout()
        savefig(fig, 'lg_letkf_sensitivity.pdf')
    else:
        print("  [skip] No LETKF sensitivity data found.")

    print("  Linear Gaussian figures complete.")


# ════════════════════════════════════════════════════════════════════════
#  3-PANEL RMSE COMPARISON (vel / SST / SSH)
# ════════════════════════════════════════════════════════════════════════

def plot_compare_rmse_3panel(datasets, labels, colors_list, fname, title='RMSE Comparison',
                              sigma_vel=0.10, sigma_sst=0.40, sigma_ssh=0.25,
                              ssh_clip=None):
    """
    3-panel (vel / SST / SSH) RMSE comparison.
    Each entry in datasets is a dict with 'rmse_vel', 'rmse_sst', 'rmse_ssh'.
    Lines are truncated at the length of each dataset (handles LETKF early stopping).

    If ssh_clip is set, SSH traces above ssh_clip are omitted from the SSH panel
    and annotated with a text label showing their mean value.
    """
    has_ssh = any(d.get('rmse_ssh') is not None for d in datasets)
    nrows = 3 if has_ssh else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.5 * nrows), sharex=True)
    _ssh_skip_count = 0

    for d, lab, col in zip(datasets, labels, colors_list):
        n = len(d['rmse_vel'])
        t = np.arange(1, n + 1)
        vel = np.asarray(d['rmse_vel'], dtype=float)
        sst = np.asarray(d['rmse_sst'], dtype=float)
        ssh_arr = np.asarray(d['rmse_ssh'], dtype=float) if d.get('rmse_ssh') is not None else None
        # Truncate at first NaN/Inf
        valid = np.isfinite(vel) & np.isfinite(sst)
        if ssh_arr is not None:
            valid &= np.isfinite(ssh_arr)
        if not valid.all():
            last_valid = np.where(valid)[0][-1] + 1 if valid.any() else 0
            t = t[:last_valid]
            vel = vel[:last_valid]
            sst = sst[:last_valid]
            if ssh_arr is not None:
                ssh_arr = ssh_arr[:last_valid]

        axes[0].plot(t, vel, color=col, linewidth=2, label=lab)
        axes[1].plot(t, sst, color=col, linewidth=2, label=lab)
        if has_ssh and ssh_arr is not None:
            ssh_mean = float(np.nanmean(ssh_arr))
            if ssh_clip is not None and ssh_mean > ssh_clip:
                # Off-scale: annotate instead of plotting
                axes[2].text(0.98, 0.95 - 0.08 * _ssh_skip_count,
                             f'{lab}: {ssh_mean:.1f} m (off-scale)',
                             transform=axes[2].transAxes, ha='right', va='top',
                             fontsize=9, color=col,
                             bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                       ec=col, alpha=0.8))
                _ssh_skip_count += 1
            else:
                axes[2].plot(t, ssh_arr, color=col, linewidth=2, label=lab)

    axes[0].axhline(y=sigma_vel, color='gray', linestyle='--', linewidth=1.0, alpha=0.5,
                    label=f'$\\sigma_{{vel}}={sigma_vel}$')
    axes[1].axhline(y=sigma_sst, color='gray', linestyle='--', linewidth=1.0, alpha=0.5,
                    label=f'$\\sigma_{{SST}}={sigma_sst}$')
    if has_ssh:
        axes[2].axhline(y=sigma_ssh, color='gray', linestyle='--', linewidth=1.0, alpha=0.5,
                        label=f'$\\sigma_{{SSH}}={sigma_ssh}$')

    axes[0].set_ylabel('Velocity RMSE (m/s)')
    axes[0].set_title(title)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel('SST RMSE (K)')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    if has_ssh:
        axes[2].set_ylabel('SSH RMSE (m)')
        axes[2].legend(loc='lower right')
        axes[2].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Assimilation Cycle')
    fig.tight_layout()
    savefig(fig, fname)


# ════════════════════════════════════════════════════════════════════════
#  FIELD SNAPSHOT PLOTS (SSH / U / V / HYCOM / SST)
# ════════════════════════════════════════════════════════════════════════

def plot_field_panels(data, label, fname, hycom_init=None, hycom_final=None):
    """
    Multi-row field comparison: SSH, U, V, SST.
    Layout with HYCOM (4 rows x 3 cols):
      Row 0: Analysis SSH Init | Analysis SSH Final | HYCOM SSH Final
      Row 1: Analysis U Init   | Analysis U Final   | HYCOM U Final
      Row 2: Analysis V Init   | Analysis V Final   | HYCOM V Final
      Row 3: Analysis SST Init | Analysis SST Final | HYCOM SST Final
    (HYCOM initial is omitted because it is the same as the analysis initial.)
    SSH uses an averaged scale between analysis and HYCOM ranges.
    Each row shares a single colorbar on the right.
    Layout without HYCOM (4 rows x 2 cols):
      Row 0: SSH Init | SSH Final
      ...
    """
    ny, nx = data['ny'], data['nx']
    H_b = data['H_b']
    nassim = len(data['rmse_vel'])
    lon = np.linspace(LON_RANGE[0], LON_RANGE[1], nx)
    lat = np.linspace(LAT_RANGE[0], LAT_RANGE[1], ny)
    has_hycom = (hycom_init is not None and hycom_final is not None)

    ncols = 3 if has_hycom else 2
    nrows = 4

    # Use GridSpec with a dedicated narrow column for colorbars
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(5 * ncols + 0.8, 4.5 * nrows))
    width_ratios = [1] * ncols + [0.04]
    gs = GridSpec(nrows, ncols + 1, figure=fig, width_ratios=width_ratios,
                  wspace=0.25, hspace=0.30)
    axes = np.empty((nrows, ncols), dtype=object)
    cbar_axes = []
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c] = fig.add_subplot(gs[r, c])
        cbar_axes.append(fig.add_subplot(gs[r, ncols]))

    def _smooth(field):
        return smooth_ocean(field, H_b, sigma=2.0)

    def _plot(ax, field, title, cmap, vmin, vmax):
        ax.set_facecolor('0.88')
        im = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Lon ($^\\circ$E)')
        ax.set_ylabel('Lat ($^\\circ$N)')
        return im

    # ── Extract analysis fields ──
    ssh_init = extract_ssh(data, 0, H_b)
    ssh_final = extract_ssh(data, nassim, H_b)
    u_init = data['raw'][0, 0, 1, :, :]
    u_final = data['raw'][nassim, 0, 1, :, :]
    v_init = data['raw'][0, 0, 2, :, :]
    v_final = data['raw'][nassim, 0, 2, :, :]
    sst_init = extract_sst(data, 0) - 273.15
    sst_final = extract_sst(data, nassim) - 273.15

    ssh_init_s = _smooth(ssh_init)
    ssh_final_s = _smooth(ssh_final)
    u_init_s = _smooth(u_init)
    u_final_s = _smooth(u_final)
    v_init_s = _smooth(v_init)
    v_final_s = _smooth(v_final)
    sst_init_s = _smooth(sst_init)
    sst_final_s = _smooth(sst_final)

    # ── Compute HYCOM fields if available ──
    if has_hycom:
        hycom_ssh_final_s = _smooth(hycom_final['ssh'])
        hycom_u_final_s = _smooth(hycom_final['u'])
        hycom_v_final_s = _smooth(hycom_final['v'])
        hycom_sst_final_s = _smooth(hycom_final['sst'] - 273.15)

    # ── SSH scale: average of analysis range and HYCOM range ──
    vmax_ssh_ana = max(abs(np.nanmin(ssh_init_s)), abs(np.nanmax(ssh_init_s)),
                       abs(np.nanmin(ssh_final_s)), abs(np.nanmax(ssh_final_s)))
    if has_hycom:
        vmax_ssh_hycom = max(abs(np.nanmin(hycom_ssh_final_s)),
                            abs(np.nanmax(hycom_ssh_final_s)),
                            abs(np.nanmin(_smooth(hycom_init['ssh']))),
                            abs(np.nanmax(_smooth(hycom_init['ssh']))))
        vmax_ssh = (vmax_ssh_ana + vmax_ssh_hycom) / 2.0
    else:
        vmax_ssh = vmax_ssh_ana

    # ── U/V scales (include HYCOM if available) ──
    vmax_u = max(abs(np.nanmin(u_init_s)), abs(np.nanmax(u_init_s)),
                 abs(np.nanmin(u_final_s)), abs(np.nanmax(u_final_s)))
    vmax_v = max(abs(np.nanmin(v_init_s)), abs(np.nanmax(v_init_s)),
                 abs(np.nanmin(v_final_s)), abs(np.nanmax(v_final_s)))
    if has_hycom:
        vmax_u = max(vmax_u, abs(np.nanmin(hycom_u_final_s)), abs(np.nanmax(hycom_u_final_s)))
        vmax_v = max(vmax_v, abs(np.nanmin(hycom_v_final_s)), abs(np.nanmax(hycom_v_final_s)))

    # ── SST scale: 2nd-98th percentile over all panels ──
    all_sst_vals = np.concatenate([sst_init_s.compressed(), sst_final_s.compressed()])
    if has_hycom:
        all_sst_vals = np.concatenate([all_sst_vals, hycom_sst_final_s.compressed()])
    sst_vmin = np.percentile(all_sst_vals, 2)
    sst_vmax = np.percentile(all_sst_vals, 98)

    # ── Plot all panels (no per-panel colorbar) ──
    # Row 0: SSH
    im_ssh0 = _plot(axes[0, 0], ssh_init_s, f'{label} SSH (Initial)', 'RdBu_r', -vmax_ssh, vmax_ssh)
    im_ssh1 = _plot(axes[0, 1], ssh_final_s, f'{label} SSH (Final)', 'RdBu_r', -vmax_ssh, vmax_ssh)

    # Row 1: U velocity
    im_u0 = _plot(axes[1, 0], u_init_s, f'{label} U (Initial)', 'RdBu_r', -vmax_u, vmax_u)
    im_u1 = _plot(axes[1, 1], u_final_s, f'{label} U (Final)', 'RdBu_r', -vmax_u, vmax_u)

    # Row 2: V velocity
    im_v0 = _plot(axes[2, 0], v_init_s, f'{label} V (Initial)', 'RdBu_r', -vmax_v, vmax_v)
    im_v1 = _plot(axes[2, 1], v_final_s, f'{label} V (Final)', 'RdBu_r', -vmax_v, vmax_v)

    # Row 3: SST
    im_sst0 = _plot(axes[3, 0], sst_init_s, f'{label} SST (Initial)', 'RdYlBu_r', sst_vmin, sst_vmax)
    im_sst1 = _plot(axes[3, 1], sst_final_s, f'{label} SST (Final)', 'RdYlBu_r', sst_vmin, sst_vmax)

    if has_hycom:
        _plot(axes[0, 2], hycom_ssh_final_s, 'HYCOM SSH (Final)', 'RdBu_r', -vmax_ssh, vmax_ssh)
        _plot(axes[1, 2], hycom_u_final_s, 'HYCOM U (Final)', 'RdBu_r', -vmax_u, vmax_u)
        _plot(axes[2, 2], hycom_v_final_s, 'HYCOM V (Final)', 'RdBu_r', -vmax_v, vmax_v)
        _plot(axes[3, 2], hycom_sst_final_s, 'HYCOM SST (Final)', 'RdYlBu_r', sst_vmin, sst_vmax)

    # ── Add one shared colorbar per row in the dedicated cbar column ──
    row_ims = [im_ssh1, im_u1, im_v1, im_sst1]
    row_labels = ['m', 'm/s', 'm/s', '$^\\circ$C']
    for r, (im, cbl) in enumerate(zip(row_ims, row_labels)):
        fig.colorbar(im, cax=cbar_axes[r], label=cbl)

    savefig(fig, fname)


# ════════════════════════════════════════════════════════════════════════
#  OBSERVATION COVERAGE (scatter: drifters + SWOT)
# ════════════════════════════════════════════════════════════════════════

def plot_obs_scatter(obs_nc_path, fname, ny=70, nx=80):
    """
    Single-panel scatter plot showing drifter positions and SWOT swath positions,
    similar to the second subplot of mlswe_results_drifter_coverage.png.
    """
    lon = np.linspace(LON_RANGE[0], LON_RANGE[1], nx)
    lat = np.linspace(LAT_RANGE[0], LAT_RANGE[1], ny)
    ncells = ny * nx

    # Load bathymetry
    v1_nc = os.path.join(BASEDIR, 'output_lsmcmc_ldata_V1', 'mlswe_lsmcmc_out.nc')
    H_b = None
    if os.path.exists(v1_nc):
        with Dataset(v1_nc, 'r') as nc:
            H_b = np.asarray(nc.variables['H_b'][:]).reshape(ny, nx)

    # Load observations
    obs_data = load_mlswe_obs(obs_nc_path)
    yind = obs_data['yind']
    nassim = yind.shape[0]

    def get_drifter_lonlat(cycle_idx):
        ind1 = yind[cycle_idx]
        valid1 = (ind1 >= 0)
        u_inds = ind1[valid1]
        u_obs_mask = (u_inds >= ncells) & (u_inds < 2 * ncells)
        cell_flat_u = u_inds[u_obs_mask] - ncells
        iy_u = cell_flat_u // nx
        ix_u = cell_flat_u % nx
        return lon[ix_u], lat[iy_u]

    # Load SWOT files
    swot_dirs = [os.path.join(BASEDIR, d) for d in
                 ['data/swot_2024aug_new', 'data/swot_2024aug', 'data/swot']]
    swot_files = []
    for sd in swot_dirs:
        swot_files = sorted(glob.glob(os.path.join(sd, 'SWOT_*.nc')))
        if swot_files:
            break

    swot_lons_first, swot_lats_first = [], []
    swot_lons_last, swot_lats_last = [], []
    if swot_files:
        try:
            for fi, sf in enumerate(swot_files):
                with Dataset(sf, 'r') as nc:
                    lats_s = np.array(nc.variables['latitude'][:])
                    lons_raw = np.array(nc.variables['longitude'][:])
                lons_s = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
                valid = (np.isfinite(lons_s.ravel()) & np.isfinite(lats_s.ravel()) &
                         (lons_s.ravel() >= LON_RANGE[0]) & (lons_s.ravel() <= LON_RANGE[1]) &
                         (lats_s.ravel() >= LAT_RANGE[0]) & (lats_s.ravel() <= LAT_RANGE[1]))
                lons_v = lons_s.ravel()[valid]
                lats_v = lats_s.ravel()[valid]
                if fi < 3:
                    swot_lons_first.extend(lons_v)
                    swot_lats_first.extend(lats_v)
                if fi >= len(swot_files) - 3:
                    swot_lons_last.extend(lons_v)
                    swot_lats_last.extend(lats_v)
        except Exception as e:
            print(f"  Could not read SWOT files: {e}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    if H_b is not None:
        lon2d, lat2d = np.meshgrid(lon, lat)
        ax.contourf(lon2d, lat2d, H_b, levels=20, cmap='Blues', alpha=0.4)

    # SWOT swaths first (behind drifters)
    if len(swot_lons_first) > 0:
        ax.scatter(swot_lons_first, swot_lats_first, s=1, c='green', alpha=0.3, rasterized=True)
    if len(swot_lons_last) > 0:
        ax.scatter(swot_lons_last, swot_lats_last, s=1, c='orange', alpha=0.3, rasterized=True)

    # Drifters
    dlons1, dlats1 = get_drifter_lonlat(0)
    ax.scatter(dlons1, dlats1, s=18, c='red', edgecolors='white', linewidths=0.3, zorder=5)
    dlons2, dlats2 = np.array([]), np.array([])
    if nassim > 1:
        dlons2, dlats2 = get_drifter_lonlat(nassim - 1)
        ax.scatter(dlons2, dlats2, s=18, c='blue', edgecolors='white', linewidths=0.3,
                   alpha=0.7, zorder=5)

    # Custom legend with visible colored markers
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='white', markersize=8,
               label=f'Drifters Cycle 1 ({len(dlons1)})'),
    ]
    if nassim > 1 and len(dlons2) > 0:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markeredgecolor='white', markersize=8,
                   label=f'Drifters Cycle {nassim} ({len(dlons2)})'))
    if len(swot_lons_first) > 0:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                   markersize=10, label='SWOT first passes'))
    if len(swot_lons_last) > 0:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor='orange',
                   markersize=10, label='SWOT last passes'))

    ax.legend(handles=legend_elements, fontsize=12, loc='upper right',
              frameon=True, fancybox=True)
    ax.set_title('Observation Sources', fontsize=18)
    ax.set_xlabel('Longitude ($^\\circ$E)')
    ax.set_ylabel('Latitude ($^\\circ$N)')
    fig.tight_layout()
    savefig(fig, fname)


# ════════════════════════════════════════════════════════════════════════
#  MAIN GENERATION LOGIC
# ════════════════════════════════════════════════════════════════════════

def _find_bc_path():
    """Find the HYCOM boundary condition file."""
    for candidate in ['data/hycom_bc_2024aug.nc', 'data/hycom_bc.nc']:
        p = os.path.join(BASEDIR, candidate)
        if os.path.exists(p):
            return p
    return None


def _load_hycom_pair(bc_path, data):
    """Load HYCOM reanalysis at initial and final observation times."""
    ny, nx = data['ny'], data['nx']
    obs_times = data['obs_times']
    lon = np.linspace(LON_RANGE[0], LON_RANGE[1], nx)
    lat = np.linspace(LAT_RANGE[0], LAT_RANGE[1], ny)
    if obs_times is None or len(obs_times) < 2:
        return None, None
    try:
        hycom_init = load_hycom_reanalysis(bc_path, lon, lat, obs_times[0])
        hycom_final = load_hycom_reanalysis(bc_path, lon, lat, obs_times[-1])
        return hycom_init, hycom_final
    except Exception as e:
        print(f"  WARNING: Could not load HYCOM: {e}")
        return None, None


def generate_mlswe_ldata_figures():
    """Generate MLSWE linear data model figures."""
    print("\n=== MLSWE Linear Data Model Figures ===")

    v1_nc = os.path.join(BASEDIR, 'output_lsmcmc_ldata_V1', 'mlswe_lsmcmc_out.nc')
    v2_nc = os.path.join(BASEDIR, 'output_lsmcmc_ldata_V2', 'mlswe_lsmcmc_out.nc')
    letkf_nc_candidates = [
        os.path.join(BASEDIR, 'output_letkf_ldata_K25', 'mlswe_letkf_out.nc'),
        os.path.join(BASEDIR, 'output_letkf_ldata', 'mlswe_letkf_out.nc'),
        os.path.join(BASEDIR, 'output_letkf_best', 'mlswe_letkf_out.nc'),
    ]
    letkf_nc = None
    for c in letkf_nc_candidates:
        if os.path.exists(c):
            letkf_nc = c
            break
    v1_obs = os.path.join(BASEDIR, 'output_lsmcmc_ldata_V1', 'mlswe_merged_obs.nc')

    v1 = load_mlswe_analysis(v1_nc) if os.path.exists(v1_nc) else None
    v2 = load_mlswe_analysis(v2_nc) if os.path.exists(v2_nc) else None
    letkf = load_mlswe_analysis(letkf_nc) if letkf_nc and os.path.exists(letkf_nc) else None

    # 3-panel RMSE comparison
    datasets, labels, cols = [], [], []
    if v1:
        datasets.append(v1); labels.append('LSMCMC V1'); cols.append('#1f77b4')
    if v2:
        datasets.append(v2); labels.append('LSMCMC V2'); cols.append('#ff7f0e')
    if letkf:
        datasets.append(letkf); labels.append('LETKF'); cols.append('#2ca02c')
    if datasets:
        plot_compare_rmse_3panel(datasets, labels, cols, 'ldata_compare_rmse.pdf',
                                  title='Linear Data Model: RMSE Comparison')

    # Observation coverage (scatter)
    if os.path.exists(v1_obs):
        plot_obs_scatter(v1_obs, 'mlswe_obs_pattern.pdf')

    # Field panels (V1 + HYCOM)
    if v1:
        bc_path = _find_bc_path()
        hycom_init, hycom_final = None, None
        if bc_path:
            hycom_init, hycom_final = _load_hycom_pair(bc_path, v1)
        plot_field_panels(v1, 'LSMCMC V1', 'ldata_v1_fields.pdf',
                          hycom_init=hycom_init, hycom_final=hycom_final)

    print("  MLSWE linear data model figures complete.")


def generate_mlswe_nltwin_figures():
    """Generate MLSWE NL twin experiment figures."""
    print("\n=== MLSWE NL Twin Experiment Figures ===")

    v1_nc = os.path.join(BASEDIR, 'output_lsmcmc_nldata_twin_V1', 'mlswe_lsmcmc_out.nc')
    v1_hmc_nc = os.path.join(BASEDIR, 'output_lsmcmc_nldata_twin_V1_hmc', 'mlswe_lsmcmc_out.nc')
    v2_nc = os.path.join(BASEDIR, 'output_lsmcmc_nldata_twin_V2', 'mlswe_lsmcmc_out.nc')

    v1 = load_mlswe_analysis(v1_nc) if os.path.exists(v1_nc) else None
    v1_hmc = load_mlswe_analysis(v1_hmc_nc) if os.path.exists(v1_hmc_nc) else None
    v2 = load_mlswe_analysis(v2_nc) if os.path.exists(v2_nc) else None

    # Check for LETKF NL twin data
    letkf_twin_nc = os.path.join(BASEDIR, 'output_letkf_nl_twin', 'mlswe_letkf_out.nc')
    letkf_twin_log = os.path.join(BASEDIR, 'letkf_nltwin_run.log')
    letkf_twin = None
    if os.path.exists(letkf_twin_nc):
        letkf_twin = load_mlswe_analysis(letkf_twin_nc)
    elif os.path.exists(letkf_twin_log):
        letkf_twin = parse_letkf_nl_log(letkf_twin_log)
        if letkf_twin:
            print(f"  Loaded LETKF NL twin from log: {len(letkf_twin['rmse_vel'])} cycles")

    # 3-panel comparison RMSE (now includes V1 HMC)
    datasets, labels, cols = [], [], []
    if v1:
        datasets.append(v1); labels.append('LSMCMC V1 (pCN)'); cols.append('#1f77b4')
    if v1_hmc:
        datasets.append(v1_hmc); labels.append('LSMCMC V1 (HMC)'); cols.append('#9467bd')
    if v2:
        datasets.append(v2); labels.append('LSMCMC V2'); cols.append('#ff7f0e')
    if letkf_twin:
        datasets.append(letkf_twin); labels.append('LETKF'); cols.append('#2ca02c')
    if datasets:
        plot_compare_rmse_3panel(datasets, labels, cols, 'nltwin_compare_rmse.pdf',
                                  title='NL Twin: RMSE Comparison',
                                  sigma_ssh=0.50, ssh_clip=5.0)

    # Field panels (V1 HMC + HYCOM)
    field_data = v1_hmc if v1_hmc else v1
    field_label = 'LSMCMC V1 (HMC)' if v1_hmc else 'LSMCMC V1'
    if field_data:
        bc_path = _find_bc_path()
        hycom_init, hycom_final = None, None
        if bc_path:
            hycom_init, hycom_final = _load_hycom_pair(bc_path, field_data)
        plot_field_panels(field_data, field_label, 'nltwin_v1_fields.pdf',
                          hycom_init=hycom_init, hycom_final=hycom_final)

    print("  MLSWE NL twin figures complete.")


def generate_mlswe_nlreal_figures():
    """Generate MLSWE NL real data figures."""
    print("\n=== MLSWE NL Real Data Figures ===")

    v1_candidates = [
        os.path.join(BASEDIR, 'output_lsmcmc_nldata_real_V1', 'mlswe_lsmcmc_out.nc'),
        os.path.join(BASEDIR, 'output_lsmcmc_nldata_V1', 'mlswe_lsmcmc_out.nc'),
    ]
    v2_candidates = [
        os.path.join(BASEDIR, 'output_lsmcmc_nldata_real_V2', 'mlswe_lsmcmc_out.nc'),
        os.path.join(BASEDIR, 'output_lsmcmc_nldata_V2', 'mlswe_lsmcmc_out.nc'),
    ]

    v1, v2 = None, None
    for c in v1_candidates:
        if os.path.exists(c):
            v1 = load_mlswe_analysis(c)
            break
    for c in v2_candidates:
        if os.path.exists(c):
            v2 = load_mlswe_analysis(c)
            break

    # Parse LETKF NL real data from log file
    letkf_log = os.path.join(BASEDIR, 'letkf_nldata_run.log')
    letkf = parse_letkf_nl_log(letkf_log)
    if letkf:
        print(f"  Loaded LETKF NL real from log: {len(letkf['rmse_vel'])} cycles")

    # 3-panel comparison RMSE
    datasets, labels, cols = [], [], []
    if v1:
        datasets.append(v1); labels.append('LSMCMC V1'); cols.append('#1f77b4')
    if v2:
        datasets.append(v2); labels.append('LSMCMC V2'); cols.append('#ff7f0e')
    if letkf:
        datasets.append(letkf); labels.append('LETKF'); cols.append('#2ca02c')
    if datasets:
        plot_compare_rmse_3panel(datasets, labels, cols, 'nlreal_compare_rmse.pdf',
                                  title='NL Real Data: RMSE Comparison')

    # Field panels (V1 + HYCOM)
    if v1:
        bc_path = _find_bc_path()
        hycom_init, hycom_final = None, None
        if bc_path:
            hycom_init, hycom_final = _load_hycom_pair(bc_path, v1)
        plot_field_panels(v1, 'LSMCMC V1', 'nlreal_v1_fields.pdf',
                          hycom_init=hycom_init, hycom_final=hycom_final)

    print("  MLSWE NL real figures complete.")


def main():
    print("=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)

    generate_lg_figures()
    generate_mlswe_ldata_figures()
    generate_mlswe_nltwin_figures()
    generate_mlswe_nlreal_figures()

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTDIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()

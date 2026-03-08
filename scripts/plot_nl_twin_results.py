#!/usr/bin/env python
"""
plot_nl_twin_results.py
========================
Analyse and plot results from the NL-LSMCMC synthetic twin experiment.
Mimics the layout of plot_mlswe_results.py but compares analysis vs nature run
instead of analysis vs HYCOM / real observations.

Plots (same names as the linear version):
    1. _ssh.png              - SSH anomaly maps (initial + final)
    2. _u_velocity.png       - Eastward velocity maps (initial + final)
    3. _v_velocity.png       - Northward velocity maps (initial + final)
    4. _sst_maps.png         - SST maps (initial + final)
    4a-4d. _compare_*.png    - Analysis vs Nature Run comparison (2x3 panels)
    5. _drifter_coverage.png - Observation coverage heatmap + positions
    6. _rmse.png             - RMSE time series (vel, SSH, SST)
    6b. _rmse_hycom.png      - RMSE vs nature run (detailed)
    7. _timeseries.png       - Velocity time series at selected cells
    8. _sst_timeseries.png   - SST time series at selected cells
    9. _ssh_timeseries.png   - SSH time series at selected cells
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset


# ================================================================
#  Helpers
# ================================================================
def load_analysis(nc_path):
    """Load analysis (posterior mean) and RMSE from output NC."""
    with Dataset(nc_path, 'r') as nc:
        raw = np.asarray(nc.variables['lsmcmc_mean'][:])  # (time,3,4,ny,nx)
        rmse_vel = np.asarray(nc.variables['rmse_vel'][:])
        rmse_sst = np.asarray(nc.variables['rmse_sst'][:])
        rmse_ssh = np.asarray(nc.variables['rmse_ssh'][:])
        obs_times = np.asarray(nc.variables['obs_times'][:])
        H_b = np.asarray(nc.variables['H_b'][:])
    nt = raw.shape[0]
    ny, nx = raw.shape[3], raw.shape[4]
    ncells = ny * nx
    nlayers = raw.shape[1]
    nfields = nlayers * 4
    dimx = nfields * ncells
    flat = raw.reshape(nt, dimx)
    return flat, rmse_vel, rmse_sst, rmse_ssh, obs_times, nlayers, ny, nx, H_b


def load_truth(nc_path):
    """Load nature run trajectory."""
    with Dataset(nc_path, 'r') as nc:
        raw = np.asarray(nc.variables['truth'][:])  # (time,3,4,ny,nx)
    nt = raw.shape[0]
    dimx = raw.shape[1] * raw.shape[2] * raw.shape[3] * raw.shape[4]
    return raw.reshape(nt, dimx)


def load_obs(nc_path):
    """Load synthetic obs file."""
    with Dataset(nc_path, 'r') as nc:
        yobs = np.asarray(nc.variables['yobs_all'][:])
        yind = np.asarray(nc.variables['yobs_ind_all'][:])
        yind0 = np.asarray(nc.variables['yobs_ind_level0_all'][:])
        times = np.asarray(nc.variables['obs_times'][:])
        sig_y = float(nc.sig_y) if hasattr(nc, 'sig_y') else 0.1
        sig_y_arr = None
        if 'sig_y_all' in nc.variables:
            sig_y_arr = np.asarray(nc.variables['sig_y_all'][:])
    return yobs, yind, yind0, times, sig_y, sig_y_arr


# ================================================================
#  Main
# ================================================================
def main(outdir='./output_lsmcmc_nldata_V1', save_prefix='mlswe_results',
         config_file=None, method_label='NL-LSMCMC'):

    # ---- Locate files ----
    nc_path = os.path.join(outdir, 'mlswe_lsmcmc_out.nc')
    truth_path = os.path.join(outdir, 'truth_trajectory.nc')
    obs_path = os.path.join(outdir, 'synthetic_arctan_obs.nc')
    if not os.path.exists(nc_path):
        print(f"ERROR: {nc_path} not found"); sys.exit(1)
    if not os.path.exists(truth_path):
        print(f"ERROR: {truth_path} not found"); sys.exit(1)

    # ---- Config ----
    lon_min, lon_max = -60.0, -20.0
    lat_min, lat_max = 10.0, 45.0
    params_cfg = {}
    if config_file is not None and os.path.exists(config_file):
        import yaml
        with open(config_file) as f:
            params_cfg = yaml.safe_load(f)
        lon_min = params_cfg.get('lon_min', lon_min)
        lon_max = params_cfg.get('lon_max', lon_max)
        lat_min = params_cfg.get('lat_min', lat_min)
        lat_max = params_cfg.get('lat_max', lat_max)

    sig_y_vel = params_cfg.get('sig_y_uv', 0.10)
    sig_y_sst_cfg = params_cfg.get('sig_y_sst', 0.40)
    sig_y_ssh_cfg = params_cfg.get('sig_y_ssh', 0.50)

    # ---- Load data ----
    lsmcmc_mean, rmse_vel, rmse_sst, rmse_ssh, obs_times, \
        nlayers, ny, nx, H_b = load_analysis(nc_path)
    truth_flat = load_truth(truth_path)

    nassim = len(rmse_vel)
    ncells = ny * nx
    nfields = nlayers * 4
    dimx = nfields * ncells

    lon_grid = np.linspace(lon_min, lon_max, nx)
    lat_grid = np.linspace(lat_min, lat_max, ny)

    has_obs = os.path.exists(obs_path)
    if has_obs:
        yobs, yind, yind0, _, sig_y, sig_y_arr = load_obs(obs_path)
        print(f"Loaded synthetic obs: {obs_path}")
    else:
        yobs = yind = yind0 = sig_y_arr = None
        sig_y = 0.1

    print(f"Analysis: ({nassim+1}, {dimx}), Nature Run: {truth_flat.shape}")
    print(f"Grid: {ny}x{nx}, nassim={nassim}")

    # ---- Field extraction ----
    def get_fields(state_flat, step):
        vec = state_flat[step]
        h_total = vec[0*ncells:1*ncells].reshape(ny, nx)
        u0 = vec[1*ncells:2*ncells].reshape(ny, nx)
        v0 = vec[2*ncells:3*ncells].reshape(ny, nx)
        T0 = vec[3*ncells:4*ncells].reshape(ny, nx)
        return h_total, u0, v0, T0

    # ---- Ocean mask + helpers ----
    ocean_mask = H_b >= 200.0

    def smooth_ocean(field, sigma=2.0):
        filled = field.copy()
        filled[~ocean_mask] = np.nanmean(field[ocean_mask]) if ocean_mask.any() else 0.0
        smoothed = gaussian_filter(filled, sigma=sigma)
        return np.ma.masked_where(~ocean_mask, smoothed)

    def add_bathy_contour(ax):
        ax.contour(lon_grid, lat_grid, H_b, levels=[200], colors='0.4',
                   linewidths=0.8, linestyles='-')
        ax.contour(lon_grid, lat_grid, H_b, levels=[1000, 3000, 5000],
                   colors='0.6', linewidths=0.4, linestyles='--')

    def get_obs_lonlat(cycle_idx):
        """Get lon/lat of observations at a given cycle."""
        if not has_obs:
            return np.array([]), np.array([])
        ind = yind[cycle_idx]
        valid = (ind >= 0) & (ind < dimx) & np.isfinite(yobs[cycle_idx])
        inds = ind[valid]
        cell_flat = inds % ncells
        iy = cell_flat // nx
        ix = cell_flat % nx
        return lon_grid[ix], lat_grid[iy]

    # ---- RMSE stats ----
    mean_rmse_vel = np.nanmean(rmse_vel)
    mean_rmse_sst = np.nanmean(rmse_sst)
    mean_rmse_ssh = np.nanmean(rmse_ssh)
    print(f"\nRMSE vs Nature Run:")
    print(f"  Mean vel:  {mean_rmse_vel:.6f} m/s")
    print(f"  Mean SST:  {mean_rmse_sst:.4f} K")
    print(f"  Mean SSH:  {mean_rmse_ssh:.4f} m")

    # ---- Create plot dir ----
    plot_dir = os.path.join(outdir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    steps_plot = [0, nassim]
    labels_plot = ['Initial', f'Cycle {nassim}']

    # ================================================================
    #  Figure 1: SSH anomaly maps (initial + final)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for col, (step, label) in enumerate(zip(steps_plot, labels_plot)):
        h_tot, u, v, T = get_fields(lsmcmc_mean, step)
        ssh = h_tot - H_b
        ssh_smooth = smooth_ocean(ssh, sigma=3.0)
        ax = axes[col]
        vals = ssh_smooth.compressed()
        vmax = max(abs(np.percentile(vals, 2)),
                   abs(np.percentile(vals, 98)), 0.01)
        ax.set_facecolor('0.88')
        im = ax.pcolormesh(lon_grid, lat_grid, ssh_smooth, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        add_bathy_contour(ax)
        if step > 0:
            dlons, dlats = get_obs_lonlat(step - 1)
            ax.scatter(dlons, dlats, s=6, c='lime', edgecolors='k',
                       linewidths=0.3, zorder=5, label='Obs locations')
            ax.legend(fontsize=8, loc='lower right')
        ax.set_title(f'SSH anomaly -- {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (deg E)')
        ax.set_ylabel('Latitude (deg N)')
        plt.colorbar(im, ax=ax, label='SSH (m)', shrink=0.85)
    fig.suptitle('Sea Surface Height Anomaly (h_total - H_b) -- Analysis',
                 fontsize=14)
    p = os.path.join(plot_dir, f'{save_prefix}_ssh.png')
    fig.savefig(p, dpi=150); plt.close()
    print(f"\nSaved: {p}")

    # ================================================================
    #  Figure 2: Eastward velocity (u0)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for col, (step, label) in enumerate(zip(steps_plot, labels_plot)):
        h_tot, u, v, T = get_fields(lsmcmc_mean, step)
        u_smooth = smooth_ocean(u, sigma=2.0)
        ax = axes[col]
        ax.set_facecolor('0.88')
        vmax = max(np.percentile(np.abs(u_smooth.compressed()), 99), 0.01)
        im = ax.pcolormesh(lon_grid, lat_grid, u_smooth, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        add_bathy_contour(ax)
        if step > 0:
            dlons, dlats = get_obs_lonlat(step - 1)
            ax.scatter(dlons, dlats, s=6, c='lime', edgecolors='k',
                       linewidths=0.3, zorder=5, label='Obs locations')
            ax.legend(fontsize=8, loc='lower right')
        ax.set_title(f'Eastward Velocity (u0) -- {label}', fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Longitude (deg E)')
        ax.set_ylabel('Latitude (deg N)')
        plt.colorbar(im, ax=ax, label='u (m/s)', shrink=0.85)
    fig.suptitle('Eastward Velocity (u0) -- Analysis', fontsize=14)
    p = os.path.join(plot_dir, f'{save_prefix}_u_velocity.png')
    fig.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")

    # ================================================================
    #  Figure 3: Northward velocity (v0)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for col, (step, label) in enumerate(zip(steps_plot, labels_plot)):
        h_tot, u, v, T = get_fields(lsmcmc_mean, step)
        v_smooth = smooth_ocean(v, sigma=2.0)
        ax = axes[col]
        ax.set_facecolor('0.88')
        vmax = max(np.percentile(np.abs(v_smooth.compressed()), 99), 0.01)
        im = ax.pcolormesh(lon_grid, lat_grid, v_smooth, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        add_bathy_contour(ax)
        if step > 0:
            dlons, dlats = get_obs_lonlat(step - 1)
            ax.scatter(dlons, dlats, s=6, c='lime', edgecolors='k',
                       linewidths=0.3, zorder=5, label='Obs locations')
            ax.legend(fontsize=8, loc='lower right')
        ax.set_title(f'Northward Velocity (v0) -- {label}', fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Longitude (deg E)')
        ax.set_ylabel('Latitude (deg N)')
        plt.colorbar(im, ax=ax, label='v (m/s)', shrink=0.85)
    fig.suptitle('Northward Velocity (v0) -- Analysis', fontsize=14)
    p = os.path.join(plot_dir, f'{save_prefix}_v_velocity.png')
    fig.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")

    # ================================================================
    #  Figure 4: SST maps (shared color scale)
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sst_vals = []
    for step in steps_plot:
        _, _, _, T = get_fields(lsmcmc_mean, step)
        T_c = smooth_ocean(T - 273.15, sigma=2.0)
        sst_vals.extend(T_c.compressed())
    sst_vmin = np.percentile(sst_vals, 2)
    sst_vmax = np.percentile(sst_vals, 98)

    for col, (step, label) in enumerate(zip(steps_plot, labels_plot)):
        _, _, _, T = get_fields(lsmcmc_mean, step)
        T_c = smooth_ocean(T - 273.15, sigma=2.0)
        ax = axes[col]
        ax.set_facecolor('0.88')
        im = ax.pcolormesh(lon_grid, lat_grid, T_c, cmap='RdYlBu_r',
                           vmin=sst_vmin, vmax=sst_vmax, shading='auto')
        add_bathy_contour(ax)
        if step > 0:
            dlons, dlats = get_obs_lonlat(step - 1)
            ax.scatter(dlons, dlats, s=6, c='lime', edgecolors='k',
                       linewidths=0.3, zorder=5, label='Obs locations')
            ax.legend(fontsize=8, loc='lower right')
        ax.set_title(f'SST -- {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (deg E)')
        ax.set_ylabel('Latitude (deg N)')
        plt.colorbar(im, ax=ax, label='SST (deg C)', shrink=0.85)
    fig.suptitle('Sea Surface Temperature -- Analysis', fontsize=14)
    p = os.path.join(plot_dir, f'{save_prefix}_sst_maps.png')
    fig.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")

    # ================================================================
    #  Figures 4a-4d: Analysis vs Nature Run comparison (2x3 panels)
    # ================================================================
    def _plot_comparison(var_key, cmap, label_unit, title_prefix,
                         analysis_func, truth_func, sigma, save_name,
                         diverging=True):
        fig, axes = plt.subplots(2, 3, figsize=(21, 10),
                                 constrained_layout=True)
        step_labels = ['Initial (Cycle 0)', f'Final (Cycle {nassim})']
        steps = [0, nassim]

        for row, (step, rlabel) in enumerate(zip(steps, step_labels)):
            anal_field = analysis_func(step)
            anal_smooth = smooth_ocean(anal_field, sigma=sigma)
            truth_field = truth_func(step)
            truth_smooth = smooth_ocean(truth_field, sigma=sigma)

            anal_vals = anal_smooth.compressed()
            truth_vals = truth_smooth.compressed()
            all_vals = np.concatenate([anal_vals, truth_vals])
            if len(all_vals) == 0:
                continue
            if diverging:
                vmax = max(abs(np.percentile(all_vals, 2)),
                           abs(np.percentile(all_vals, 98)), 0.01)
                vmin = -vmax
            else:
                vmin = np.percentile(all_vals, 2)
                vmax = np.percentile(all_vals, 98)

            # Col 0: Analysis
            ax = axes[row, 0]
            ax.set_facecolor('0.88')
            im = ax.pcolormesh(lon_grid, lat_grid, anal_smooth,
                               cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            add_bathy_contour(ax)
            ax.set_title(f'Analysis -- {rlabel}', fontsize=11,
                         fontweight='bold')
            ax.set_xlabel('Longitude (deg E)')
            ax.set_ylabel('Latitude (deg N)')
            plt.colorbar(im, ax=ax, label=label_unit, shrink=0.85)

            # Col 1: Nature Run
            ax = axes[row, 1]
            ax.set_facecolor('0.88')
            im = ax.pcolormesh(lon_grid, lat_grid, truth_smooth,
                               cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            add_bathy_contour(ax)
            ax.set_title(f'Nature Run -- {rlabel}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Longitude (deg E)')
            ax.set_ylabel('Latitude (deg N)')
            plt.colorbar(im, ax=ax, label=label_unit, shrink=0.85)

            # Col 2: Difference (Analysis - Nature Run)
            diff = anal_smooth - truth_smooth
            diff_vals = diff.compressed()
            dmax = max(abs(np.percentile(diff_vals, 2)),
                       abs(np.percentile(diff_vals, 98)), 0.001) \
                if len(diff_vals) > 0 else 0.1
            ax = axes[row, 2]
            ax.set_facecolor('0.88')
            im = ax.pcolormesh(lon_grid, lat_grid, diff, cmap='RdBu_r',
                               vmin=-dmax, vmax=dmax, shading='auto')
            add_bathy_contour(ax)
            ax.set_title(f'Difference (Analysis - Nature Run) -- {rlabel}',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Longitude (deg E)')
            ax.set_ylabel('Latitude (deg N)')
            plt.colorbar(im, ax=ax, label=label_unit, shrink=0.85)

        fig.suptitle(f'{title_prefix}: Analysis vs Nature Run', fontsize=14)
        fpath = os.path.join(plot_dir, f'{save_prefix}_{save_name}.png')
        fig.savefig(fpath, dpi=150); plt.close()
        print(f"Saved: {fpath}")

    # SSH comparison
    _plot_comparison(
        'ssh', 'RdBu_r', 'SSH (m)', 'Sea Surface Height',
        lambda s: get_fields(lsmcmc_mean, s)[0] - H_b,
        lambda s: get_fields(truth_flat, s)[0] - H_b,
        3.0, 'compare_ssh', diverging=True)

    # U comparison
    _plot_comparison(
        'u', 'RdBu_r', 'u (m/s)', 'Eastward Velocity (u0)',
        lambda s: get_fields(lsmcmc_mean, s)[1],
        lambda s: get_fields(truth_flat, s)[1],
        2.0, 'compare_u', diverging=True)

    # V comparison
    _plot_comparison(
        'v', 'RdBu_r', 'v (m/s)', 'Northward Velocity (v0)',
        lambda s: get_fields(lsmcmc_mean, s)[2],
        lambda s: get_fields(truth_flat, s)[2],
        2.0, 'compare_v', diverging=True)

    # SST comparison
    _plot_comparison(
        'sst', 'RdYlBu_r', 'SST (deg C)', 'Sea Surface Temperature',
        lambda s: get_fields(lsmcmc_mean, s)[3] - 273.15,
        lambda s: get_fields(truth_flat, s)[3] - 273.15,
        2.0, 'compare_sst', diverging=False)

    # ================================================================
    #  Figure 5: Observation coverage
    # ================================================================
    if has_obs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 constrained_layout=True)

        # Panel 1: Obs coverage heatmap
        obs_coverage = np.zeros((ny, nx), dtype=int)
        for t in range(nassim):
            ind0 = yind0[t]
            valid = (ind0 >= 0) & (ind0 < ncells)
            for idx in ind0[valid]:
                iy, ix = divmod(int(idx), nx)
                obs_coverage[iy, ix] += 1

        ax = axes[0]
        im = ax.pcolormesh(lon_grid, lat_grid, obs_coverage, cmap='YlOrRd',
                           shading='auto')
        add_bathy_contour(ax)
        ax.set_title('Observation count per grid cell', fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('Longitude (deg E)')
        ax.set_ylabel('Latitude (deg N)')
        plt.colorbar(im, ax=ax, label='count')

        # Panel 2: Obs positions at first/last cycle (colored by type)
        ax2 = axes[1]
        if H_b is not None:
            ax2.contourf(lon_grid, lat_grid, H_b, levels=20, cmap='Blues',
                         alpha=0.4)

        # Separate SSH vs UV/SST obs for cycle 0
        ind0 = yind[0]
        valid0 = (ind0 >= 0) & np.isfinite(yobs[0])
        ind_v0 = ind0[valid0].astype(int)
        ssh_mask0 = ind_v0 < ncells
        uvsst_mask0 = ind_v0 >= ncells
        cell_ssh0 = ind_v0[ssh_mask0] % ncells
        cell_uvsst0 = ind_v0[uvsst_mask0] % ncells

        ax2.scatter(lon_grid[cell_uvsst0 % nx], lat_grid[cell_uvsst0 // nx],
                    s=12, c='red', edgecolors='white', linewidths=0.3,
                    label=f'Cycle 1 drifter ({uvsst_mask0.sum()})')
        ax2.scatter(lon_grid[cell_ssh0 % nx], lat_grid[cell_ssh0 // nx],
                    s=8, c='green', edgecolors='white', linewidths=0.3,
                    alpha=0.5, label=f'Cycle 1 SSH ({ssh_mask0.sum()})')

        if nassim > 1:
            indf = yind[nassim - 1]
            validf = (indf >= 0) & np.isfinite(yobs[nassim - 1])
            ind_vf = indf[validf].astype(int)
            ssh_maskf = ind_vf < ncells
            uvsst_maskf = ind_vf >= ncells
            cell_sshf = ind_vf[ssh_maskf] % ncells
            cell_uvsstf = ind_vf[uvsst_maskf] % ncells
            ax2.scatter(lon_grid[cell_uvsstf % nx],
                        lat_grid[cell_uvsstf // nx],
                        s=12, c='blue', edgecolors='white', linewidths=0.3,
                        alpha=0.6,
                        label=f'Cycle {nassim} drifter ({uvsst_maskf.sum()})')
            ax2.scatter(lon_grid[cell_sshf % nx],
                        lat_grid[cell_sshf // nx],
                        s=8, c='orange', edgecolors='white', linewidths=0.3,
                        alpha=0.5,
                        label=f'Cycle {nassim} SSH ({ssh_maskf.sum()})')

        ax2.set_title('Observation positions', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Longitude (deg E)')
        ax2.set_ylabel('Latitude (deg N)')
        ax2.legend(fontsize=8, loc='upper right')

        p = os.path.join(plot_dir, f'{save_prefix}_drifter_coverage.png')
        fig.savefig(p, dpi=150); plt.close()
        print(f"Saved: {p}")

    # ================================================================
    #  Figure 6: RMSE time series (3 panels: vel, SSH, SST)
    # ================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             constrained_layout=True)
    cycles = np.arange(1, nassim + 1)

    # Panel 1: Velocity RMSE
    ax = axes[0]
    ax.plot(cycles, rmse_vel, 'b-', lw=1.2, alpha=0.8,
            label=f'{method_label} vs Nature Run')
    ax.axhline(sig_y_vel, color='k', ls=':', lw=1.5,
               label=f'$\\sigma_y^{{vel}}$ = {sig_y_vel:.4f} m/s')
    ax.set_ylabel('UV RMSE (m/s)', fontsize=11)
    ax.set_title(f'Velocity RMSE  --  mean = {mean_rmse_vel:.5f} m/s',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 2: SSH RMSE
    ax = axes[1]
    ax.plot(cycles, rmse_ssh, 'g-', lw=1.2, alpha=0.8,
            label=f'{method_label} vs Nature Run')
    ax.axhline(sig_y_ssh_cfg, color='k', ls=':', lw=1.5,
               label=f'$\\sigma_{{ssh}}$ = {sig_y_ssh_cfg:.2f} m')
    ax.set_ylabel('SSH RMSE (m)', fontsize=11)
    ax.set_title(f'SSH RMSE  --  mean = {mean_rmse_ssh:.4f} m',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 3: SST RMSE
    ax = axes[2]
    ax.plot(cycles, rmse_sst, 'r-', lw=1.2, alpha=0.8,
            label=f'{method_label} vs Nature Run')
    ax.axhline(sig_y_sst_cfg, color='k', ls=':', lw=1.5,
               label=f'$\\sigma_y^{{sst}}$ = {sig_y_sst_cfg:.2f} K')
    ax.set_ylabel('SST RMSE (K)', fontsize=11)
    ax.set_xlabel('Assimilation Cycle', fontsize=11)
    ax.set_title(f'SST RMSE  --  mean = {mean_rmse_sst:.4f} K',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(f'{method_label}: RMSE vs Nature Run  '
                 f'(dotted = obs noise $\\sigma$)',
                 fontsize=14, fontweight='bold')
    p = os.path.join(plot_dir, f'{save_prefix}_rmse.png')
    fig.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")

    # ================================================================
    #  Figure 6b: RMSE vs Nature Run (detailed, with markers)
    # ================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             constrained_layout=True)

    ax = axes[0]
    ax.plot(cycles, rmse_vel, 'b-o', markersize=2, lw=1.2,
            label=f'Vel RMSE vs nature run (mean = {mean_rmse_vel:.4f} m/s)')
    ax.axhline(sig_y_vel, color='k', ls=':', lw=1.5,
               label=f'$\\sigma_y^{{vel}}$ = {sig_y_vel:.4f}')
    ax.set_ylabel('UV RMSE (m/s)', fontsize=11)
    ax.set_title(f'Velocity RMSE vs Nature Run  --  mean = {mean_rmse_vel:.5f} m/s',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(cycles, rmse_ssh, 'g-s', markersize=2, lw=1.2,
            label=f'SSH RMSE vs nature run (mean = {mean_rmse_ssh:.4f} m)')
    ax.axhline(sig_y_ssh_cfg, color='k', ls=':', lw=1.5,
               label=f'$\\sigma_{{ssh}}$ = {sig_y_ssh_cfg:.2f} m')
    ax.set_ylabel('SSH RMSE (m)', fontsize=11)
    ax.set_title(f'SSH RMSE vs Nature Run  --  mean = {mean_rmse_ssh:.4f} m',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    ax = axes[2]
    ax.plot(cycles, rmse_sst, 'r-D', markersize=2, lw=1.2,
            label=f'SST RMSE vs nature run (mean = {mean_rmse_sst:.3f} K)')
    ax.axhline(sig_y_sst_cfg, color='k', ls=':', lw=1.5,
               label=f'$\\sigma_y^{{sst}}$ = {sig_y_sst_cfg:.2f} K')
    ax.set_ylabel('SST RMSE (K)', fontsize=11)
    ax.set_xlabel('Assimilation Cycle', fontsize=11)
    ax.set_title(f'SST RMSE vs Nature Run  --  mean = {mean_rmse_sst:.4f} K',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

    fig.suptitle(f'{method_label}: Analysis RMSE vs Nature Run',
                 fontsize=14, fontweight='bold')
    p = os.path.join(plot_dir, f'{save_prefix}_rmse_hycom.png')
    fig.savefig(p, dpi=150); plt.close()
    print(f"Saved: {p}")

    # ================================================================
    #  Figure 7: Velocity time series at most-observed cells
    #            (analysis vs nature run)
    # ================================================================
    if has_obs:
        obs_count = np.zeros(ncells, dtype=int)
        for t in range(nassim):
            ind0t = yind0[t]
            valid = (ind0t >= 0) & (ind0t < ncells)
            for idx in ind0t[valid]:
                obs_count[int(idx)] += 1
        top_cells = np.argsort(obs_count)[-4:][::-1]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                                 constrained_layout=True)
        for panel, cell_flat in enumerate(top_cells):
            ax = axes.flat[panel]
            iy_c, ix_c = divmod(cell_flat, nx)
            lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
            u_idx = ncells + cell_flat
            v_idx = 2 * ncells + cell_flat

            anal_u = lsmcmc_mean[:, u_idx]
            anal_v = lsmcmc_mean[:, v_idx]
            truth_u = truth_flat[:, u_idx]
            truth_v = truth_flat[:, v_idx]

            ax.plot(np.arange(nassim + 1), truth_u, 'r--', alpha=0.6,
                    lw=1.5, label='Nature Run u')
            ax.plot(np.arange(nassim + 1), truth_v, 'b--', alpha=0.6,
                    lw=1.5, label='Nature Run v')
            ax.plot(np.arange(nassim + 1), anal_u, 'r-', alpha=0.8,
                    lw=1.5, label='Analysis u')
            ax.plot(np.arange(nassim + 1), anal_v, 'b-', alpha=0.8,
                    lw=1.5, label='Analysis v')
            ax.set_title(f'({lon_c:.1f} deg E, {lat_c:.1f} deg N) -- '
                         f'{obs_count[cell_flat]} obs',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Velocity (m/s)')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Analysis vs Nature Run: Velocity at Most-Observed Cells',
                     fontsize=13)
        p = os.path.join(plot_dir, f'{save_prefix}_timeseries.png')
        fig.savefig(p, dpi=150); plt.close()
        print(f"Saved: {p}")

    # ================================================================
    #  Figure 8: SST time series at selected cells
    # ================================================================
    if has_obs:
        sst_obs_count = np.zeros(ncells, dtype=int)
        for t in range(nassim):
            ind_t = yind[t]
            sst_mask_t = ((ind_t >= 3 * ncells) & (ind_t < 4 * ncells)
                          & np.isfinite(yobs[t]))
            for idx in ind_t[sst_mask_t]:
                cell = int(idx) - 3 * ncells
                if 0 <= cell < ncells:
                    sst_obs_count[cell] += 1
        top_sst = np.argsort(sst_obs_count)[-4:][::-1]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                                 constrained_layout=True)
        for panel in range(min(2, len(top_sst))):
            cell_flat = top_sst[panel]
            iy_c, ix_c = divmod(cell_flat, nx)
            lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
            t_idx = 3 * ncells + cell_flat

            anal_T = lsmcmc_mean[:, t_idx] - 273.15
            truth_T = truth_flat[:, t_idx] - 273.15

            ax = axes[panel]
            ax.plot(np.arange(nassim + 1), truth_T, 'k--', lw=1.5,
                    label='Nature Run SST')
            ax.plot(np.arange(nassim + 1), anal_T, 'r-', lw=1.5,
                    label='Analysis SST')
            ax.set_title(f'SST at ({lon_c:.1f} deg E, {lat_c:.1f} deg N) -- '
                         f'{sst_obs_count[cell_flat]} SST obs',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Temperature (deg C)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle('SST: Analysis vs Nature Run', fontsize=13)
        p = os.path.join(plot_dir, f'{save_prefix}_sst_timeseries.png')
        fig.savefig(p, dpi=150); plt.close()
        print(f"Saved: {p}")

    # ================================================================
    #  Figure 9: SSH time series at most SSH-observed cells
    # ================================================================
    if has_obs:
        ssh_obs_count = np.zeros(ncells, dtype=int)
        for t in range(nassim):
            ind_t = yind[t]
            ssh_mask_t = ((ind_t >= 0) & (ind_t < ncells)
                          & np.isfinite(yobs[t]))
            for idx in ind_t[ssh_mask_t]:
                ssh_obs_count[int(idx)] += 1

        top_ssh_cells = np.argsort(ssh_obs_count)[-4:][::-1]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                                 constrained_layout=True)
        for panel, cell_flat in enumerate(top_ssh_cells[:4]):
            ax = axes.flat[panel]
            iy_c, ix_c = divmod(cell_flat, nx)
            lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
            h_b_cell = H_b[iy_c, ix_c]

            anal_ssh = lsmcmc_mean[:, cell_flat].reshape(-1) - h_b_cell
            truth_ssh = truth_flat[:, cell_flat].reshape(-1) - h_b_cell

            # NOTE: SSH obs are arctan-transformed (arctan(h_total) ≈ π/2),
            #       not physical SSH, so we do NOT plot them here.

            ax.plot(np.arange(nassim + 1), truth_ssh, 'k--', lw=1.5,
                    label='Nature Run SSH')
            ax.plot(np.arange(nassim + 1), anal_ssh, 'b-', alpha=0.8,
                    lw=1.5, label='Analysis SSH')
            ax.set_title(f'({lon_c:.1f}°E, {lat_c:.1f}°N) — '
                         f'{ssh_obs_count[cell_flat]} SSH obs',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('SSH (m)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle('SSH: Analysis vs Nature Run & Observations at Most-Observed Cells',
                     fontsize=13)
        p = os.path.join(plot_dir, f'{save_prefix}_ssh_timeseries.png')
        fig.savefig(p, dpi=150); plt.close()
        print(f"Saved: {p}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY -- NL-LSMCMC Twin Experiment")
    print("=" * 60)
    print(f"  Model: 3-layer MLSWE, obs operator: arctan")
    print(f"  Grid: {ny} x {nx}  ({lon_min} to {lon_max} deg E, "
          f"{lat_min} to {lat_max} deg N)")
    print(f"  Assimilation cycles: {nassim}")
    print(f"  State dim: {dimx}")
    if has_obs:
        nobs_per = [np.sum((yind[t] >= 0) & np.isfinite(yobs[t]))
                    for t in range(nassim)]
        print(f"  Obs per cycle: {min(nobs_per)} - {max(nobs_per)}")
    print(f"  Vel  RMSE: mean={mean_rmse_vel:.6f}, "
          f"range=[{np.nanmin(rmse_vel):.6f}, {np.nanmax(rmse_vel):.6f}] m/s")
    print(f"  SST  RMSE: mean={mean_rmse_sst:.4f}, "
          f"range=[{np.nanmin(rmse_sst):.4f}, {np.nanmax(rmse_sst):.4f}] K")
    print(f"  SSH  RMSE: mean={mean_rmse_ssh:.4f}, "
          f"range=[{np.nanmin(rmse_ssh):.4f}, {np.nanmax(rmse_ssh):.4f}] m")
    print("=" * 60)
    plt.close('all')
    print(f"\nAll plots saved to: {plot_dir}/")


if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else './output_lsmcmc_nldata_V1'
    config = sys.argv[2] if len(sys.argv) > 2 else 'example_input_mlswe_nldata_V1_twin.yml'
    label = sys.argv[3] if len(sys.argv) > 3 else 'NL-LSMCMC'
    main(outdir=outdir, config_file=config, method_label=label)

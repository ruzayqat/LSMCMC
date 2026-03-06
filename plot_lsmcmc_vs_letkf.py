#!/usr/bin/env python
"""
plot_lsmcmc_vs_letkf.py
========================
Side-by-side comparison of MLSWE LSMCMC vs LETKF results.
Generates:
  1. SSH comparison (LSMCMC final, LETKF final, HYCOM reanalysis)
  2. Velocity comparison (u, v)
  3. SST comparison
  4. RMSE comparison time series
  5. Summary statistics table
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset


def _fill_nan_nearest(arr_2d):
    mask = np.isnan(arr_2d)
    if not mask.any():
        return arr_2d
    if mask.all():
        arr_2d[:] = 0.0
        return arr_2d
    ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
    arr_2d[mask] = arr_2d[tuple(ind[:, mask])]
    return arr_2d


def load_analysis(nc_path):
    with Dataset(nc_path, 'r') as nc:
        raw = np.asarray(nc.variables['lsmcmc_mean'][:])
        rmse_vel = np.asarray(nc.variables['rmse_vel'][:])
        rmse_sst = np.asarray(nc.variables['rmse_sst'][:])
        obs_times = np.asarray(nc.variables['obs_times'][:])
        nlayers = int(nc.nlayers)
        ny = int(nc.ny) if hasattr(nc, 'ny') else len(nc.dimensions['y'])
        nx = int(nc.nx) if hasattr(nc, 'nx') else len(nc.dimensions['x'])
    nt = raw.shape[0]
    dimx = nlayers * 4 * ny * nx
    flat = raw.reshape(nt, dimx)
    return flat, rmse_vel, rmse_sst, obs_times, nlayers, ny, nx


def load_obs(nc_path):
    with Dataset(nc_path, 'r') as nc:
        yobs = np.asarray(nc.variables['yobs_all'][:])
        yind = np.asarray(nc.variables['yobs_ind_all'][:])
        times = np.asarray(nc.variables['obs_times'][:])
        sig_y = float(nc.sig_y) if hasattr(nc, 'sig_y') else 0.01
    return yobs, yind, times, sig_y


def compute_rmse_vs_obs(lsmcmc_mean, yobs, yind, nassim, ncells):
    rmse_vel = np.full(nassim, np.nan)
    rmse_sst = np.full(nassim, np.nan)
    for t in range(nassim):
        y = yobs[t]; ind = yind[t]
        valid = (ind >= 0) & (ind < 4 * ncells) & np.isfinite(y)
        if valid.sum() == 0:
            continue
        ind_v = ind[valid].astype(int)
        y_v = y[valid]
        z_a = lsmcmc_mean[t + 1]
        res = z_a[ind_v] - y_v
        vel_m = (ind_v >= ncells) & (ind_v < 3 * ncells)
        sst_m = ind_v >= 3 * ncells
        if vel_m.sum() > 0:
            rmse_vel[t] = np.sqrt(np.mean(res[vel_m]**2))
        if sst_m.sum() > 0:
            rmse_sst[t] = np.sqrt(np.mean(res[sst_m]**2))
    return rmse_vel, rmse_sst


def load_hycom_reanalysis(bc_path, lon_grid, lat_grid, time_sec):
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

    t_idx = np.interp(time_sec, bc_times, np.arange(len(bc_times)))
    t_lo = int(np.floor(t_idx)); t_hi = min(t_lo + 1, len(bc_times) - 1)
    alpha = t_idx - t_lo
    mg_lat, mg_lon = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    result = {}
    for name, f3d in [('ssh', bc_ssh), ('u', bc_uo), ('v', bc_vo), ('sst', bc_sst)]:
        snap = (1 - alpha) * f3d[t_lo] + alpha * f3d[t_hi] if t_lo != t_hi else f3d[t_lo]
        interp = RegularGridInterpolator((bc_lat, bc_lon), snap,
                                          method='linear', bounds_error=False, fill_value=0.0)
        result[name] = interp((mg_lat, mg_lon))
    return result


def main():
    lsmcmc_dir = sys.argv[1] if len(sys.argv) > 1 else './output_lsmcmc'
    letkf_dir = sys.argv[2] if len(sys.argv) > 2 else './output_letkf'
    data_dir = './data'

    # --- Load both analyses ---
    lsmcmc_nc = os.path.join(lsmcmc_dir, 'mlswe_lsmcmc_out.nc')
    letkf_nc = os.path.join(letkf_dir, 'mlswe_letkf_out.nc')
    if not os.path.exists(letkf_nc):
        letkf_nc = os.path.join(letkf_dir, 'mlswe_lsmcmc_out.nc')

    lsm_mean, lsm_rv, lsm_rs, lsm_times, nlayers, ny, nx = load_analysis(lsmcmc_nc)
    let_mean, let_rv, let_rs, let_times, _, _, _ = load_analysis(letkf_nc)
    ncells = ny * nx
    nassim = lsm_mean.shape[0] - 1

    # --- Load obs ---
    obs_candidates = [
        os.path.join(lsmcmc_dir, 'mlswe_merged_obs.nc'),
        os.path.join(letkf_dir, 'mlswe_merged_obs.nc'),
        os.path.join(lsmcmc_dir, 'swe_drifter_obs.nc'),
        '../SWE_LSMCMC/output_hybrid_nf25_loc200/swe_drifter_obs.nc',
    ]
    obs_file = None
    for c in obs_candidates:
        if os.path.exists(c):
            obs_file = c; break
    has_obs = obs_file is not None
    if has_obs:
        yobs, yind, obs_times, sig_y = load_obs(obs_file)

    # --- Recompute RMSE from scratch ---
    if has_obs:
        lsm_rmse_vel, lsm_rmse_sst = compute_rmse_vs_obs(lsm_mean, yobs, yind, nassim, ncells)
        let_rmse_vel, let_rmse_sst = compute_rmse_vs_obs(let_mean, yobs, yind, nassim, ncells)
    else:
        lsm_rmse_vel = lsm_rv; lsm_rmse_sst = lsm_rs
        let_rmse_vel = let_rv; let_rmse_sst = let_rs

    # --- Grid ---
    lon_min, lon_max = -80.0, -20.0
    lat_min, lat_max = 10.0, 50.0
    lon_grid = np.linspace(lon_min, lon_max, nx)
    lat_grid = np.linspace(lat_min, lat_max, ny)

    # --- Bathymetry ---
    H_b = None
    for d in [data_dir, '.', './data']:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith('etopo_bathy_') and f.endswith('.npy'):
                    H_b = np.load(os.path.join(d, f))
                    H_b = np.maximum(np.abs(H_b), 100.0)
                    break
        if H_b is not None:
            break

    ocean_mask = H_b >= 200.0 if H_b is not None else np.ones((ny, nx), dtype=bool)

    def smooth_ocean(field, sigma=2.0):
        filled = field.copy()
        filled[~ocean_mask] = np.nanmean(field[ocean_mask]) if ocean_mask.any() else 0.0
        s = gaussian_filter(filled, sigma=sigma)
        return np.ma.masked_where(~ocean_mask, s)

    def add_bathy_contour(ax):
        if H_b is not None:
            ax.contour(lon_grid, lat_grid, H_b, levels=[200], colors='0.4', linewidths=0.8)
            ax.contour(lon_grid, lat_grid, H_b, levels=[1000, 3000, 5000],
                       colors='0.6', linewidths=0.4, linestyles='--')

    def get_fields(mean_arr, step):
        vec = mean_arr[step]
        h = vec[0*ncells:1*ncells].reshape(ny, nx)
        u = vec[1*ncells:2*ncells].reshape(ny, nx)
        v = vec[2*ncells:3*ncells].reshape(ny, nx)
        T = vec[3*ncells:4*ncells].reshape(ny, nx)
        return h, u, v, T

    # --- Output dir ---
    plot_dir = './comparison_plots'
    os.makedirs(plot_dir, exist_ok=True)

    # --- HYCOM reanalysis ---
    bc_path = os.path.join(data_dir, 'hycom_bc.nc')
    hycom_final = None
    if os.path.exists(bc_path) and lsm_times is not None:
        try:
            hycom_final = load_hycom_reanalysis(bc_path, lon_grid, lat_grid, lsm_times[nassim - 1])
        except Exception as e:
            print(f"Could not load HYCOM: {e}")

    # ================================================================
    #  Figure 1: SSH comparison (3 panels: LSMCMC, LETKF, HYCOM)
    # ================================================================
    ncols = 3 if hycom_final is not None else 2
    fig1, axes1 = plt.subplots(1, ncols, figsize=(7 * ncols, 5), constrained_layout=True)

    panels = [
        ('LSMCMC Analysis', get_fields(lsm_mean, nassim)),
        ('LETKF Analysis', get_fields(let_mean, nassim)),
    ]
    if hycom_final is not None:
        panels.append(('HYCOM Reanalysis', None))

    for col, (title, fields) in enumerate(panels):
        ax = axes1[col]
        ax.set_facecolor('0.88')
        if fields is not None:
            h, u, v, T = fields
            ssh = (h - H_b) if H_b is not None else (h - 4000.0)
        else:
            ssh = hycom_final['ssh']
        ssh_s = smooth_ocean(ssh, sigma=3.0)
        vals = ssh_s.compressed()
        vmax = max(abs(np.percentile(vals, 2)), abs(np.percentile(vals, 98)), 0.01)
        im = ax.pcolormesh(lon_grid, lat_grid, ssh_s, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        add_bathy_contour(ax)
        ax.set_title(f'SSH -- {title} (Cycle {nassim})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='SSH (m)', shrink=0.85)

    fig1.suptitle('Sea Surface Height Comparison (Final Cycle)', fontsize=14)
    fig1.savefig(os.path.join(plot_dir, 'compare_ssh.png'), dpi=150)
    print(f"Saved: {plot_dir}/compare_ssh.png")

    # ================================================================
    #  Figure 2: Velocity comparison (2x2: u LSMCMC/LETKF, v LSMCMC/LETKF)
    # ================================================================
    fig2, axes2 = plt.subplots(2, ncols, figsize=(7 * ncols, 10), constrained_layout=True)

    for row, (var_name, var_idx) in enumerate([('Eastward Velocity (u)', 1),
                                                ('Northward Velocity (v)', 2)]):
        datasets = [
            ('LSMCMC', get_fields(lsm_mean, nassim)[var_idx]),
            ('LETKF', get_fields(let_mean, nassim)[var_idx]),
        ]
        if hycom_final is not None:
            hkey = 'u' if var_idx == 1 else 'v'
            datasets.append(('HYCOM', hycom_final[hkey]))

        for col, (lbl, field) in enumerate(datasets):
            ax = axes2[row, col]
            ax.set_facecolor('0.88')
            f_s = smooth_ocean(field, sigma=2.0)
            vals = f_s.compressed()
            vmax = max(np.percentile(np.abs(vals), 99), 0.01)
            im = ax.pcolormesh(lon_grid, lat_grid, f_s, cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax, shading='auto')
            add_bathy_contour(ax)
            ax.set_title(f'{var_name} -- {lbl}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax, label='m/s', shrink=0.85)

    fig2.suptitle('Velocity Comparison (Final Cycle)', fontsize=14)
    fig2.savefig(os.path.join(plot_dir, 'compare_velocity.png'), dpi=150)
    print(f"Saved: {plot_dir}/compare_velocity.png")

    # ================================================================
    #  Figure 3: SST comparison
    # ================================================================
    fig3, axes3 = plt.subplots(1, ncols, figsize=(7 * ncols, 5), constrained_layout=True)

    sst_data = [
        ('LSMCMC', get_fields(lsm_mean, nassim)[3] - 273.15),
        ('LETKF', get_fields(let_mean, nassim)[3] - 273.15),
    ]
    if hycom_final is not None:
        sst_data.append(('HYCOM', hycom_final['sst'] - 273.15))

    for col, (lbl, field) in enumerate(sst_data):
        ax = axes3[col]
        ax.set_facecolor('0.88')
        f_s = smooth_ocean(field, sigma=2.0)
        im = ax.pcolormesh(lon_grid, lat_grid, f_s, cmap='RdYlBu_r', shading='auto')
        add_bathy_contour(ax)
        ax.set_title(f'SST -- {lbl}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='SST (°C)', shrink=0.85)

    fig3.suptitle('Sea Surface Temperature Comparison (Final Cycle)', fontsize=14)
    fig3.savefig(os.path.join(plot_dir, 'compare_sst.png'), dpi=150)
    print(f"Saved: {plot_dir}/compare_sst.png")

    # ================================================================
    #  Figure 4: RMSE comparison time series
    # ================================================================
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    cycles = np.arange(1, nassim + 1)

    # Velocity RMSE
    ax4a.plot(cycles, lsm_rmse_vel, 'b-o', ms=3, lw=1.5,
              label=f'LSMCMC (mean={np.nanmean(lsm_rmse_vel):.4f})')
    ax4a.plot(cycles, let_rmse_vel, 'r-s', ms=3, lw=1.5,
              label=f'LETKF (mean={np.nanmean(let_rmse_vel):.4f})')
    ax4a.set_xlabel('Assimilation Cycle', fontsize=12)
    ax4a.set_ylabel('Velocity RMSE (m/s)', fontsize=12)
    ax4a.set_title('Velocity RMSE: LSMCMC vs LETKF', fontsize=13, fontweight='bold')
    ax4a.legend(fontsize=10)
    ax4a.grid(True, alpha=0.3)
    # Set y-limit to show LSMCMC detail (LETKF may be much larger)
    lsm_vel_max = np.nanmax(lsm_rmse_vel) if np.any(np.isfinite(lsm_rmse_vel)) else 1.0
    let_vel_max = np.nanmax(let_rmse_vel) if np.any(np.isfinite(let_rmse_vel)) else 1.0
    ylim_vel = min(max(lsm_vel_max, let_vel_max) * 1.25, let_vel_max * 1.25)
    ax4a.set_ylim(bottom=0, top=ylim_vel)

    # SST RMSE
    ax4b.plot(cycles, lsm_rmse_sst, 'b-o', ms=3, lw=1.5,
              label=f'LSMCMC (mean={np.nanmean(lsm_rmse_sst):.3f} K)')
    ax4b.plot(cycles, let_rmse_sst, 'r-s', ms=3, lw=1.5,
              label=f'LETKF (mean={np.nanmean(let_rmse_sst):.3f} K)')
    ax4b.set_xlabel('Assimilation Cycle', fontsize=12)
    ax4b.set_ylabel('SST RMSE (K)', fontsize=12)
    ax4b.set_title('SST RMSE: LSMCMC vs LETKF', fontsize=13, fontweight='bold')
    ax4b.legend(fontsize=10)
    ax4b.grid(True, alpha=0.3)
    lsm_sst_max = np.nanmax(lsm_rmse_sst) if np.any(np.isfinite(lsm_rmse_sst)) else 1.0
    let_sst_max = np.nanmax(let_rmse_sst) if np.any(np.isfinite(let_rmse_sst)) else 1.0
    ylim_sst = min(max(lsm_sst_max, let_sst_max) * 1.25, let_sst_max * 1.25)
    ax4b.set_ylim(bottom=0, top=ylim_sst)

    fig4.suptitle('RMSE Comparison: LSMCMC vs LETKF (vs Observations)', fontsize=14)
    fig4.savefig(os.path.join(plot_dir, 'compare_rmse.png'), dpi=150)
    print(f"Saved: {plot_dir}/compare_rmse.png")

    # ================================================================
    #  Figure 5: RMSE comparison (zoomed to LSMCMC scale)
    # ================================================================
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    ax5a.plot(cycles, lsm_rmse_vel, 'b-o', ms=3, lw=1.5,
              label=f'LSMCMC (mean={np.nanmean(lsm_rmse_vel):.4f})')
    ax5a.set_xlabel('Assimilation Cycle', fontsize=12)
    ax5a.set_ylabel('Velocity RMSE (m/s)', fontsize=12)
    ax5a.set_title('Velocity RMSE: LSMCMC (zoomed)', fontsize=13, fontweight='bold')
    ax5a.legend(fontsize=10)
    ax5a.grid(True, alpha=0.3)
    ax5a.set_ylim(bottom=0, top=lsm_vel_max * 1.3)

    ax5b.plot(cycles, lsm_rmse_sst, 'b-o', ms=3, lw=1.5,
              label=f'LSMCMC (mean={np.nanmean(lsm_rmse_sst):.3f} K)')
    ax5b.set_xlabel('Assimilation Cycle', fontsize=12)
    ax5b.set_ylabel('SST RMSE (K)', fontsize=12)
    ax5b.set_title('SST RMSE: LSMCMC (zoomed)', fontsize=13, fontweight='bold')
    ax5b.legend(fontsize=10)
    ax5b.grid(True, alpha=0.3)
    ax5b.set_ylim(bottom=0, top=lsm_sst_max * 1.3)

    fig5.suptitle('RMSE: LSMCMC Detail View', fontsize=14)
    fig5.savefig(os.path.join(plot_dir, 'compare_rmse_zoomed.png'), dpi=150)
    print(f"Saved: {plot_dir}/compare_rmse_zoomed.png")

    # ================================================================
    #  Figure 6: SSH time series comparison
    # ================================================================
    if H_b is not None:
        fig6, axes6 = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        cycles_all = np.arange(nassim + 1)

        # Compute SSH statistics for both methods
        for method_idx, (method_mean, method_name, color) in enumerate([
            (lsm_mean, 'LSMCMC', 'blue'),
            (let_mean, 'LETKF', 'red'),
        ]):
            ssh_mean_ts = np.zeros(nassim + 1)
            ssh_min_ts = np.zeros(nassim + 1)
            ssh_max_ts = np.zeros(nassim + 1)
            ssh_std_ts = np.zeros(nassim + 1)
            for t in range(nassim + 1):
                h_total = get_fields(method_mean, t)[0]
                ssh_field = h_total - H_b
                ssh_ocean = ssh_field[ocean_mask]
                ssh_mean_ts[t] = ssh_ocean.mean()
                ssh_min_ts[t] = ssh_ocean.min()
                ssh_max_ts[t] = ssh_ocean.max()
                ssh_std_ts[t] = ssh_ocean.std()

            # Domain mean
            ax = axes6[0, 0]
            ax.plot(cycles_all, ssh_mean_ts, '-', color=color, lw=1.5,
                    label=method_name)

            # Domain min/max
            ax = axes6[0, 1]
            ax.fill_between(cycles_all, ssh_min_ts, ssh_max_ts,
                            alpha=0.15, color=color)
            ax.plot(cycles_all, ssh_min_ts, '-', color=color, lw=0.8, alpha=0.7)
            ax.plot(cycles_all, ssh_max_ts, '-', color=color, lw=0.8, alpha=0.7,
                    label=method_name)

        axes6[0, 0].axhline(0, color='gray', ls=':', lw=0.8)
        axes6[0, 0].set_xlabel('Cycle'); axes6[0, 0].set_ylabel('SSH (m)')
        axes6[0, 0].set_title('Domain-Mean SSH', fontsize=11, fontweight='bold')
        axes6[0, 0].legend(fontsize=9); axes6[0, 0].grid(True, alpha=0.3)

        axes6[0, 1].axhline(0, color='gray', ls=':', lw=0.8)
        axes6[0, 1].set_xlabel('Cycle'); axes6[0, 1].set_ylabel('SSH (m)')
        axes6[0, 1].set_title('SSH Min/Max Envelope', fontsize=11, fontweight='bold')
        axes6[0, 1].legend(fontsize=9); axes6[0, 1].grid(True, alpha=0.3)

        # Cell-level SSH comparison (pick 2 representative ocean cells)
        cell_choices = [ncells // 3, 2 * ncells // 3]
        for panel_idx, cell_flat in enumerate(cell_choices):
            iy_c, ix_c = divmod(cell_flat, nx)
            while not ocean_mask[iy_c, ix_c] and cell_flat < ncells - 1:
                cell_flat += 1
                iy_c, ix_c = divmod(cell_flat, nx)
            lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
            h_b_cell = H_b[iy_c, ix_c]
            ax = axes6[1, panel_idx]

            for method_mean, method_name, color in [
                (lsm_mean, 'LSMCMC', 'blue'),
                (let_mean, 'LETKF', 'red'),
            ]:
                anal_ssh = np.array([get_fields(method_mean, t)[0][iy_c, ix_c] - h_b_cell
                                     for t in range(nassim + 1)])
                ax.plot(cycles_all, anal_ssh, '-', color=color, lw=1.5,
                        label=method_name)

            ax.axhline(0, color='gray', ls=':', lw=0.8)
            ax.set_title(f'SSH at ({lon_c:.1f}E, {lat_c:.1f}N)  H_b={h_b_cell:.0f}m',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Cycle'); ax.set_ylabel('SSH (m)')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig6.suptitle('SSH Time Series: LSMCMC vs LETKF', fontsize=14)
        fig6.savefig(os.path.join(plot_dir, 'compare_ssh_timeseries.png'), dpi=150)
        print(f"Saved: {plot_dir}/compare_ssh_timeseries.png")

    # ================================================================
    #  Figure 7: Velocity + SST timeseries at selected cells
    # ================================================================
    if has_obs:
        # Find most-observed cells (based on u/v velocity observations)
        obs_count = np.zeros(ncells, dtype=int)
        for t in range(nassim):
            ind_t = yind[t]
            # u-velocity indices: ncells .. 2*ncells
            u_mask = (ind_t >= ncells) & (ind_t < 2*ncells)
            for idx in ind_t[u_mask]:
                obs_count[int(idx) - ncells] += 1
            # v-velocity indices: 2*ncells .. 3*ncells
            v_mask = (ind_t >= 2*ncells) & (ind_t < 3*ncells)
            for idx in ind_t[v_mask]:
                obs_count[int(idx) - 2*ncells] += 1
        top_cells = np.argsort(obs_count)[-4:][::-1]

        fig7, axes7 = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        for panel, cell_flat in enumerate(top_cells):
            ax = axes7.flat[panel]
            iy_c, ix_c = divmod(cell_flat, nx)
            lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
            u_idx = ncells + cell_flat
            v_idx = 2 * ncells + cell_flat

            # Observations
            obs_u = np.full(nassim, np.nan)
            obs_v = np.full(nassim, np.nan)
            for t in range(nassim):
                ind_t = yind[t]; y_t = yobs[t]
                mu = np.where((ind_t == u_idx) & np.isfinite(y_t))[0]
                if mu.size > 0: obs_u[t] = np.mean(y_t[mu])
                mv = np.where((ind_t == v_idx) & np.isfinite(y_t))[0]
                if mv.size > 0: obs_v[t] = np.mean(y_t[mv])

            ax.plot(np.arange(nassim), obs_u, 'kx', ms=7, label='Obs u')
            ax.plot(np.arange(nassim), obs_v, 'k+', ms=7, label='Obs v')

            # LSMCMC
            lsm_u = lsm_mean[:, u_idx]
            lsm_v = lsm_mean[:, v_idx]
            ax.plot(np.arange(nassim + 1), lsm_u, 'b-', lw=1.5, alpha=0.8,
                    label='LSMCMC u')
            ax.plot(np.arange(nassim + 1), lsm_v, 'b--', lw=1.5, alpha=0.8,
                    label='LSMCMC v')

            # LETKF
            let_u = let_mean[:, u_idx]
            let_v = let_mean[:, v_idx]
            ax.plot(np.arange(nassim + 1), let_u, 'r-', lw=1.5, alpha=0.8,
                    label='LETKF u')
            ax.plot(np.arange(nassim + 1), let_v, 'r--', lw=1.5, alpha=0.8,
                    label='LETKF v')

            ax.set_title(f'({lon_c:.1f}E, {lat_c:.1f}N) -- {obs_count[cell_flat]} obs',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Cycle'); ax.set_ylabel('Velocity (m/s)')
            ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)

        fig7.suptitle('Velocity Time Series: LSMCMC vs LETKF at Observed Cells',
                       fontsize=13)
        fig7.savefig(os.path.join(plot_dir, 'compare_velocity_timeseries.png'), dpi=150)
        print(f"Saved: {plot_dir}/compare_velocity_timeseries.png")

        # Figure 8: SST timeseries comparison
        sst_obs_count = np.zeros(ncells, dtype=int)
        for t in range(nassim):
            ind_t = yind[t]
            sst_mask_t = (ind_t >= 3*ncells) & (ind_t < 4*ncells) & np.isfinite(yobs[t])
            for idx in ind_t[sst_mask_t]:
                sst_obs_count[int(idx) - 3*ncells] += 1
        top_sst_cells = np.argsort(sst_obs_count)[-2:][::-1]

        fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for panel, cell_flat in enumerate(top_sst_cells):
            ax = axes8[panel]
            iy_c, ix_c = divmod(cell_flat, nx)
            lon_c, lat_c = lon_grid[ix_c], lat_grid[iy_c]
            T_idx = 3 * ncells + cell_flat

            obs_T = np.full(nassim, np.nan)
            for t in range(nassim):
                ind_t = yind[t]; y_t = yobs[t]
                mt = np.where((ind_t == T_idx) & np.isfinite(y_t))[0]
                if mt.size > 0: obs_T[t] = np.mean(y_t[mt])

            ax.plot(np.arange(nassim), obs_T, 'kx', ms=7, label='Obs SST')
            ax.plot(np.arange(nassim + 1), lsm_mean[:, T_idx], 'b-', lw=1.5,
                    alpha=0.8, label='LSMCMC')
            ax.plot(np.arange(nassim + 1), let_mean[:, T_idx], 'r-', lw=1.5,
                    alpha=0.8, label='LETKF')

            ax.set_title(f'SST at ({lon_c:.1f}E, {lat_c:.1f}N)  '
                         f'{sst_obs_count[cell_flat]} obs',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Cycle'); ax.set_ylabel('SST (K)')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        fig8.suptitle('SST Time Series: LSMCMC vs LETKF', fontsize=13)
        fig8.savefig(os.path.join(plot_dir, 'compare_sst_timeseries.png'), dpi=150)
        print(f"Saved: {plot_dir}/compare_sst_timeseries.png")

    # ================================================================
    #  Summary table
    # ================================================================
    print("\n" + "=" * 70)
    print("MLSWE COMPARISON SUMMARY: LSMCMC vs LETKF")
    print("=" * 70)
    print(f"{'Metric':<30s} {'LSMCMC':>15s} {'LETKF':>15s}")
    print("-" * 70)
    print(f"{'Assimilation cycles':<30s} {nassim:>15d} {nassim:>15d}")
    print(f"{'Grid (ny x nx)':<30s} {f'{ny}x{nx}':>15s} {f'{ny}x{nx}':>15s}")
    print(f"{'State dimension':<30s} {nlayers*4*ncells:>15d} {nlayers*4*ncells:>15d}")
    if has_obs:
        print(f"{'Mean Vel RMSE (m/s)':<30s} {np.nanmean(lsm_rmse_vel):>15.6f} {np.nanmean(let_rmse_vel):>15.6f}")
        print(f"{'Mean SST RMSE (K)':<30s} {np.nanmean(lsm_rmse_sst):>15.4f} {np.nanmean(let_rmse_sst):>15.4f}")
        if np.nanmean(let_rmse_vel) > 0:
            ratio_vel = np.nanmean(lsm_rmse_vel) / np.nanmean(let_rmse_vel)
            print(f"{'Vel RMSE ratio (LSM/LET)':<30s} {ratio_vel:>15.4f} {'':>15s}")
        if np.nanmean(let_rmse_sst) > 0:
            ratio_sst = np.nanmean(lsm_rmse_sst) / np.nanmean(let_rmse_sst)
            print(f"{'SST RMSE ratio (LSM/LET)':<30s} {ratio_sst:>15.4f} {'':>15s}")
    print("=" * 70)

    plt.close('all')
    print(f"\nAll comparison plots saved to: {plot_dir}/")


if __name__ == '__main__':
    main()

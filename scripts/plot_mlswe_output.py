#!/usr/bin/env python
"""
plot_mlswe_output.py
====================
Generate standard diagnostic plots from MLSWE LSMCMC or LETKF output.
Can be called standalone or imported as a function.

Usage:
    python plot_mlswe_output.py <output.nc> [output_dir]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def generate_output_plots(nc_path, outdir=None, method_label='LSMCMC',
                          lon_range=None, lat_range=None):
    """
    Generate diagnostic plots from an MLSWE output NetCDF file.

    Parameters
    ----------
    nc_path : str   Path to mlswe_lsmcmc_out.nc or mlswe_letkf_out.nc
    outdir  : str   Directory to save plots (default: same dir as nc_path)
    method_label : str   Label for titles ('LSMCMC' or 'LETKF')
    lon_range, lat_range : tuple  (min, max) for axis labels
    """
    if outdir is None:
        outdir = os.path.dirname(nc_path)
    os.makedirs(outdir, exist_ok=True)
    prefix = method_label.lower()

    with Dataset(nc_path, 'r') as nc:
        # lsmcmc_mean: (time, layer, field, y, x)  — field: h,u,v,T
        data = np.asarray(nc.variables['lsmcmc_mean'][:])
        rmse_vel = np.asarray(nc.variables['rmse_vel'][:])
        rmse_sst = np.asarray(nc.variables['rmse_sst'][:])
        rmse_ssh = np.asarray(nc.variables['rmse_ssh'][:]) if 'rmse_ssh' in nc.variables else None
        obs_times = np.asarray(nc.variables['obs_times'][:]) if 'obs_times' in nc.variables else None
        ny = int(nc.ny) if hasattr(nc, 'ny') else data.shape[3]
        nx = int(nc.nx) if hasattr(nc, 'nx') else data.shape[4]

    nassim = len(rmse_vel)
    nt = data.shape[0]  # nassim + 1

    # Time axis in hours
    if obs_times is not None and obs_times[0] > 0:
        t_hours = (obs_times - obs_times[0]) / 3600.0
    else:
        t_hours = np.arange(nt)

    # Lon/lat grid for labelling
    if lon_range is not None:
        lon = np.linspace(lon_range[0], lon_range[1], nx)
        lat = np.linspace(lat_range[0], lat_range[1], ny)
    else:
        lon = np.arange(nx)
        lat = np.arange(ny)

    # ================================================================
    # PLOT 1: RMSE time series
    # ================================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    cycles = np.arange(1, nassim + 1)

    ax = axes[0]
    ax.plot(cycles, rmse_vel, 'b-', linewidth=0.8, alpha=0.8)
    ax.axhline(np.nanmean(rmse_vel), color='b', linestyle='--', alpha=0.5,
               label=f'mean={np.nanmean(rmse_vel):.5f}')
    ax.set_ylabel('Velocity RMSE (m/s)', fontsize=11)
    ax.set_title(f'{method_label} — Velocity RMSE per Cycle', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(cycles, rmse_sst, 'r-', linewidth=0.8, alpha=0.8)
    ax.axhline(np.nanmean(rmse_sst), color='r', linestyle='--', alpha=0.5,
               label=f'mean={np.nanmean(rmse_sst):.4f} K')
    ax.set_ylabel('SST RMSE (K)', fontsize=11)
    ax.set_xlabel('Assimilation Cycle', fontsize=11)
    ax.set_title(f'{method_label} — SST RMSE per Cycle', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(outdir, f'{prefix}_rmse_timeseries.png')
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p1}")

    # ================================================================
    # PLOT 2: RMSE + SSH RMSE (if available) on one figure
    # ================================================================
    if rmse_ssh is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        ax = axes[0]
        ax.plot(cycles, rmse_vel, 'b-', linewidth=0.8)
        ax.axhline(np.nanmean(rmse_vel), color='b', ls='--', alpha=0.5,
                   label=f'mean={np.nanmean(rmse_vel):.5f}')
        ax.set_ylabel('Vel RMSE (m/s)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(cycles, rmse_sst, 'r-', linewidth=0.8)
        ax.axhline(np.nanmean(rmse_sst), color='r', ls='--', alpha=0.5,
                   label=f'mean={np.nanmean(rmse_sst):.4f} K')
        ax.set_ylabel('SST RMSE (K)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(cycles, rmse_ssh, 'g-', linewidth=0.8)
        ax.axhline(np.nanmean(rmse_ssh), color='g', ls='--', alpha=0.5,
                   label=f'mean={np.nanmean(rmse_ssh):.4f} m')
        ax.set_ylabel('SSH RMSE (m)')
        ax.set_xlabel('Cycle')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        fig.suptitle(f'{method_label} — RMSE Time Series', fontsize=13)
        plt.tight_layout()
        p1b = os.path.join(outdir, f'{prefix}_rmse_all.png')
        fig.savefig(p1b, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {p1b}")

    # ================================================================
    # PLOT 3: Final analysis fields (layer 0)
    # ================================================================
    # data shape: (time, layer, field, y, x)
    # field indices: 0=h, 1=u, 2=v, 3=T
    idx_final = nt - 1
    idx_init = 0

    h_final = data[idx_final, 0, 0, :, :]
    u_final = data[idx_final, 0, 1, :, :]
    v_final = data[idx_final, 0, 2, :, :]
    T_final = data[idx_final, 0, 3, :, :] - 273.15  # to Celsius
    speed_final = np.sqrt(u_final**2 + v_final**2)

    h_init = data[idx_init, 0, 0, :, :]
    T_init = data[idx_init, 0, 3, :, :] - 273.15

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # SSH init
    ax = axes[0, 0]
    ssh_init = h_init - np.nanmean(h_init)
    vmax = max(abs(np.nanmin(ssh_init)), abs(np.nanmax(ssh_init)), 0.5)
    im = ax.pcolormesh(lon, lat, ssh_init, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8, label='SSH anomaly (m)')
    ax.set_title('SSH Anomaly — Initial', fontsize=11)
    ax.set_ylabel('Latitude (°N)')

    # SSH final
    ax = axes[0, 1]
    ssh_final = h_final - np.nanmean(h_final)
    im = ax.pcolormesh(lon, lat, ssh_final, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8, label='SSH anomaly (m)')
    ax.set_title('SSH Anomaly — Final', fontsize=11)

    # Speed final
    ax = axes[0, 2]
    im = ax.pcolormesh(lon, lat, speed_final, cmap='viridis',
                       shading='nearest', vmin=0, vmax=min(1.0, np.nanpercentile(speed_final, 99)))
    plt.colorbar(im, ax=ax, shrink=0.8, label='Speed (m/s)')
    # Quiver (subsample)
    skip = max(1, ny // 15)
    ax.quiver(lon[::skip], lat[::skip], u_final[::skip, ::skip],
              v_final[::skip, ::skip], scale=15, alpha=0.7, width=0.003)
    ax.set_title('Surface Speed + Currents — Final', fontsize=11)

    # SST init
    ax = axes[1, 0]
    im = ax.pcolormesh(lon, lat, T_init, cmap='RdYlBu_r',
                       shading='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8, label='SST (°C)')
    ax.set_title('SST — Initial', fontsize=11)
    ax.set_ylabel('Latitude (°N)')
    ax.set_xlabel('Longitude (°E)')

    # SST final
    ax = axes[1, 1]
    im = ax.pcolormesh(lon, lat, T_final, cmap='RdYlBu_r',
                       shading='nearest',
                       vmin=np.nanmin(T_init), vmax=np.nanmax(T_init))
    plt.colorbar(im, ax=ax, shrink=0.8, label='SST (°C)')
    ax.set_title('SST — Final', fontsize=11)
    ax.set_xlabel('Longitude (°E)')

    # SST difference
    ax = axes[1, 2]
    dT = T_final - T_init
    vmax_dT = max(abs(np.nanmin(dT)), abs(np.nanmax(dT)), 0.5)
    im = ax.pcolormesh(lon, lat, dT, cmap='RdBu_r',
                       vmin=-vmax_dT, vmax=vmax_dT, shading='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8, label='ΔSST (°C)')
    ax.set_title('SST Change (Final − Initial)', fontsize=11)
    ax.set_xlabel('Longitude (°E)')

    for ax in axes.flat:
        ax.set_aspect('equal')

    fig.suptitle(f'{method_label} — Analysis Fields (Layer 0, Surface)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    p3 = os.path.join(outdir, f'{prefix}_fields.png')
    fig.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p3}")

    # ================================================================
    # PLOT 4: Hovmoller diagram (time vs longitude, lat-averaged SST)
    # ================================================================
    # Average SST over latitude at each time step
    sst_all = data[:, 0, 3, :, :] - 273.15  # (nt, ny, nx)
    sst_hovmoller = np.nanmean(sst_all, axis=1)  # (nt, nx)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(lon, t_hours[:nt], sst_hovmoller,
                       cmap='RdYlBu_r', shading='nearest')
    plt.colorbar(im, ax=ax, label='SST (°C)')
    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Time (hours)', fontsize=11)
    ax.set_title(f'{method_label} — Hovmoller: Latitude-Averaged SST', fontsize=12)
    plt.tight_layout()
    p4 = os.path.join(outdir, f'{prefix}_hovmoller_sst.png')
    fig.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p4}")

    print(f"[{method_label}] All plots saved to {outdir}/")
    return [p1, p3, p4]


# ================================================================
#  Standalone entry point
# ================================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_mlswe_output.py <output.nc> [output_dir] [LSMCMC|LETKF]")
        sys.exit(1)

    nc_file = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    label = sys.argv[3] if len(sys.argv) > 3 else (
        'LETKF' if 'letkf' in nc_file.lower() else 'LSMCMC')

    # Try to read config for lon/lat ranges
    import yaml
    config_candidates = [
        os.path.join(os.path.dirname(nc_file), '..', 'example_input_mlswe_test100.yml'),
        os.path.join(os.path.dirname(nc_file), '..', 'example_input_mlswe_ldata_V1.yml'),
    ]
    lon_range = lat_range = None
    for cfg in config_candidates:
        if os.path.exists(cfg):
            with open(cfg) as f:
                p = yaml.safe_load(f)
            lon_range = (p['lon_min'], p['lon_max'])
            lat_range = (p['lat_min'], p['lat_max'])
            break

    generate_output_plots(nc_file, outdir=out, method_label=label,
                          lon_range=lon_range, lat_range=lat_range)

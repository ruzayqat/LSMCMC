#!/usr/bin/env python
"""
generate_nlgamma_figures.py
============================
Generate all figures for the arctan + Cauchy non-Gaussian twin experiment.

Figures produced (paper_figures/):
    nlgamma_compare_rmse.pdf  - 3-panel V1 vs V2 (M=1) RMSE comparison
    nlgamma_letkf_blowup.pdf - LETKF RMSE blowup (separate figure)
    nlgamma_v1_fields.pdf     - V1 analysis field panels (SSH/U/V/SST + HYCOM)
    nlgamma_v2_fields.pdf     - V2 M=1 field panels  (SSH/U/V/SST + HYCOM)
    nlgamma_mcmc_diag_v1.pdf  - V1 MCMC diagnostics (block-partition reduced domain)
    nlgamma_mcmc_diag_v2.pdf  - V2 MCMC diagnostics (halo-localized block)
"""
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

BASEDIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR  = os.path.join(BASEDIR, 'paper_figures')
os.makedirs(OUTDIR, exist_ok=True)

LON_RANGE = (-60, -20)
LAT_RANGE = (10, 45)

# ── Publication-quality settings ──
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

def savefig(fig, name):
    """Save figure to OUTDIR in the given format, plus PNG for web display."""
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    # Also save PNG for notebook/web display
    if name.endswith('.pdf'):
        png_path = path.replace('.pdf', '.png')
        fig.savefig(png_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {path} (+ PNG)")
    else:
        print(f"  Saved: {path}")
    plt.close(fig)


# ────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────

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


def smooth_ocean(field, H_b, sigma=2.0):
    ocean_mask = H_b >= 200.0
    filled = field.copy()
    if ocean_mask.any():
        filled[~ocean_mask] = np.nanmean(field[ocean_mask])
    smoothed = gaussian_filter(filled, sigma=sigma)
    return np.ma.masked_where(~ocean_mask, smoothed)


def load_analysis(nc_path):
    with Dataset(nc_path, 'r') as nc:
        raw = np.asarray(nc.variables['lsmcmc_mean'][:])
        rmse_vel = np.asarray(nc.variables['rmse_vel'][:])
        rmse_sst = np.asarray(nc.variables['rmse_sst'][:])
        rmse_ssh = np.asarray(nc.variables['rmse_ssh'][:])
        H_b = np.asarray(nc.variables['H_b'][:])
        obs_times = (np.asarray(nc.variables['obs_times'][:])
                     if 'obs_times' in nc.variables else None)
        ny = raw.shape[3]
        nx = raw.shape[4]
    return {
        'raw': raw, 'rmse_vel': rmse_vel, 'rmse_sst': rmse_sst,
        'rmse_ssh': rmse_ssh, 'H_b': H_b, 'obs_times': obs_times,
        'ny': ny, 'nx': nx,
    }


def load_truth(nc_path):
    with Dataset(nc_path, 'r') as nc:
        truth = np.asarray(nc.variables['truth'][:])
        H_b = np.asarray(nc.variables['H_b'][:])
    return truth, H_b


def compute_rmse_from_raw(analysis_raw, truth_raw, nlayers=3, ny=70, nx=80):
    nt = analysis_raw.shape[0] - 1
    rmse_vel = np.zeros(nt)
    rmse_sst = np.zeros(nt)
    rmse_ssh = np.zeros(nt)
    for t in range(1, nt + 1):
        vel_err2, nvel = 0.0, 0
        for k in range(nlayers):
            u_err = analysis_raw[t, k, 1] - truth_raw[t, k, 1]
            v_err = analysis_raw[t, k, 2] - truth_raw[t, k, 2]
            vel_err2 += np.sum(u_err**2) + np.sum(v_err**2)
            nvel += 2 * ny * nx
        rmse_vel[t-1] = np.sqrt(vel_err2 / nvel)
        sst_err = analysis_raw[t, 0, 3] - truth_raw[t, 0, 3]
        rmse_sst[t-1] = np.sqrt(np.mean(sst_err**2))
        ssh_ana = np.sum(analysis_raw[t, :, 0], axis=0)
        ssh_tru = np.sum(truth_raw[t, :, 0], axis=0)
        rmse_ssh[t-1] = np.sqrt(np.mean((ssh_ana - ssh_tru)**2))
    return rmse_vel, rmse_sst, rmse_ssh


def parse_letkf_log(log_path):
    pattern = re.compile(
        r'\[(\d+)/240\]\s+nobs=\s*\d+\s+vel=([\d.eE+-]+)\s+sst=([\d.eE+-]+)\s+ssh=([\d.eE+-]+)')
    cycles, vel, sst, ssh = [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                cycles.append(int(m.group(1)))
                vel.append(float(m.group(2)))
                sst.append(float(m.group(3)))
                ssh.append(float(m.group(4)))
    return np.array(cycles), np.array(vel), np.array(sst), np.array(ssh)


def extract_ssh(raw, cycle, H_b):
    return np.sum(raw[cycle, :, 0, :, :], axis=0) - H_b


def extract_sst(raw, cycle):
    return raw[cycle, 0, 3, :, :]


# ── HYCOM loading ──

def load_hycom_reanalysis(bc_path, model_lon, model_lat, model_time_sec):
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


def _load_hycom_pair(obs_times, ny, nx):
    for cand in ['data/hycom_bc_2024aug.nc', 'data/hycom_bc.nc']:
        p = os.path.join(BASEDIR, cand)
        if os.path.exists(p):
            bc_path = p
            break
    else:
        print("  WARNING: HYCOM BC file not found")
        return None, None
    lon = np.linspace(LON_RANGE[0], LON_RANGE[1], nx)
    lat = np.linspace(LAT_RANGE[0], LAT_RANGE[1], ny)
    try:
        hycom_init = load_hycom_reanalysis(bc_path, lon, lat, obs_times[0])
        hycom_final = load_hycom_reanalysis(bc_path, lon, lat, obs_times[-1])
        return hycom_init, hycom_final
    except Exception as e:
        print(f"  WARNING: Could not load HYCOM: {e}")
        return None, None


# ────────────────────────────────────────────────────
# Field panels: SSH / U / V / SST  (4 rows × 3 cols)
# ────────────────────────────────────────────────────

def plot_field_panels(data, label, fname,
                       hycom_init=None, hycom_final=None):
    """
    4-row field comparison matching nltwin_v1_fields.pdf layout.
    With HYCOM: 4 rows × 3 cols (Init | Final | HYCOM Final).
    Without HYCOM: 4 rows × 2 cols (Init | Final).
    Rows: SSH anomaly, U velocity, V velocity, SST.
    Each row share a single colorbar.
    """
    ny, nx = data['ny'], data['nx']
    H_b = data['H_b']
    nassim = len(data['rmse_vel'])
    lon = np.linspace(LON_RANGE[0], LON_RANGE[1], nx)
    lat = np.linspace(LAT_RANGE[0], LAT_RANGE[1], ny)
    has_hycom = (hycom_init is not None and hycom_final is not None)

    ncols = 3 if has_hycom else 2
    nrows = 4

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
        im = ax.pcolormesh(lon, lat, field, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Lon ($^\\circ$E)')
        ax.set_ylabel('Lat ($^\\circ$N)')
        return im

    # ── Analysis fields ──
    ssh_init  = extract_ssh(data['raw'], 0, H_b)
    ssh_final = extract_ssh(data['raw'], nassim, H_b)
    u_init  = data['raw'][0, 0, 1, :, :]
    u_final = data['raw'][nassim, 0, 1, :, :]
    v_init  = data['raw'][0, 0, 2, :, :]
    v_final = data['raw'][nassim, 0, 2, :, :]
    sst_init  = extract_sst(data['raw'], 0) - 273.15
    sst_final = extract_sst(data['raw'], nassim) - 273.15

    ssh_init_s  = _smooth(ssh_init)
    ssh_final_s = _smooth(ssh_final)
    u_init_s  = _smooth(u_init)
    u_final_s = _smooth(u_final)
    v_init_s  = _smooth(v_init)
    v_final_s = _smooth(v_final)
    sst_init_s  = _smooth(sst_init)
    sst_final_s = _smooth(sst_final)

    # ── HYCOM fields ──
    if has_hycom:
        hycom_ssh_final_s = _smooth(hycom_final['ssh'])
        hycom_u_final_s   = _smooth(hycom_final['u'])
        hycom_v_final_s   = _smooth(hycom_final['v'])
        hycom_sst_final_s = _smooth(hycom_final['sst'] - 273.15)

    # ── Colour scales ──
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

    vmax_u = max(abs(np.nanmin(u_init_s)), abs(np.nanmax(u_init_s)),
                 abs(np.nanmin(u_final_s)), abs(np.nanmax(u_final_s)))
    vmax_v = max(abs(np.nanmin(v_init_s)), abs(np.nanmax(v_init_s)),
                 abs(np.nanmin(v_final_s)), abs(np.nanmax(v_final_s)))
    if has_hycom:
        vmax_u = max(vmax_u, abs(np.nanmin(hycom_u_final_s)),
                     abs(np.nanmax(hycom_u_final_s)))
        vmax_v = max(vmax_v, abs(np.nanmin(hycom_v_final_s)),
                     abs(np.nanmax(hycom_v_final_s)))

    all_sst_vals = np.concatenate([sst_init_s.compressed(),
                                   sst_final_s.compressed()])
    if has_hycom:
        all_sst_vals = np.concatenate([all_sst_vals,
                                       hycom_sst_final_s.compressed()])
    sst_vmin = np.percentile(all_sst_vals, 2)
    sst_vmax = np.percentile(all_sst_vals, 98)

    # ── Row 0: SSH ──
    im0 = _plot(axes[0, 0], ssh_init_s,
                f'{label} SSH (Initial)', 'RdBu_r', -vmax_ssh, vmax_ssh)
    im1 = _plot(axes[0, 1], ssh_final_s,
                f'{label} SSH (Final)', 'RdBu_r', -vmax_ssh, vmax_ssh)
    # ── Row 1: U ──
    _plot(axes[1, 0], u_init_s,
          f'{label} U (Initial)', 'RdBu_r', -vmax_u, vmax_u)
    im_u = _plot(axes[1, 1], u_final_s,
                 f'{label} U (Final)', 'RdBu_r', -vmax_u, vmax_u)
    # ── Row 2: V ──
    _plot(axes[2, 0], v_init_s,
          f'{label} V (Initial)', 'RdBu_r', -vmax_v, vmax_v)
    im_v = _plot(axes[2, 1], v_final_s,
                 f'{label} V (Final)', 'RdBu_r', -vmax_v, vmax_v)
    # ── Row 3: SST ──
    _plot(axes[3, 0], sst_init_s,
          f'{label} SST (Initial)', 'RdYlBu_r', sst_vmin, sst_vmax)
    im_sst = _plot(axes[3, 1], sst_final_s,
                   f'{label} SST (Final)', 'RdYlBu_r', sst_vmin, sst_vmax)

    if has_hycom:
        _plot(axes[0, 2], hycom_ssh_final_s,
              'HYCOM SSH (Final)', 'RdBu_r', -vmax_ssh, vmax_ssh)
        _plot(axes[1, 2], hycom_u_final_s,
              'HYCOM U (Final)', 'RdBu_r', -vmax_u, vmax_u)
        _plot(axes[2, 2], hycom_v_final_s,
              'HYCOM V (Final)', 'RdBu_r', -vmax_v, vmax_v)
        _plot(axes[3, 2], hycom_sst_final_s,
              'HYCOM SST (Final)', 'RdYlBu_r', sst_vmin, sst_vmax)

    # ── Shared colorbars ──
    row_ims   = [im1, im_u, im_v, im_sst]
    row_labels = ['m', 'm/s', 'm/s', '$^\\circ$C']
    for r, (im, cbl) in enumerate(zip(row_ims, row_labels)):
        fig.colorbar(im, cax=cbar_axes[r], label=cbl)

    savefig(fig, fname)


# ────────────────────────────────────────────────────
# RMSE comparison (V1 vs V2-M4-averaged)
# ────────────────────────────────────────────────────

def plot_compare_rmse(v1_rmse, v2_rmse, fname='nlgamma_compare_rmse.pdf',
                      v1_hmc_rmse=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), sharex=True)
    t = np.arange(1, 241)
    sigma_vel, sigma_sst, sigma_ssh = 0.10, 0.40, 0.50
    for ax, key, ylabel, sigma, title in zip(
            axes,
            ['vel', 'sst', 'ssh'],
            ['Velocity RMSE (m/s)', 'SST RMSE (K)', 'SSH RMSE (m)'],
            [sigma_vel, sigma_sst, sigma_ssh],
            ['Velocity', 'SST', 'SSH']):
        ax.plot(t, v1_rmse[key], color='#1f77b4', linewidth=1.8,
                label='LSMCMC V1 (pCN)')
        if v1_hmc_rmse is not None:
            ax.plot(t, v1_hmc_rmse[key], color='#9467bd', linewidth=1.8,
                    label='LSMCMC V1 (HMC)')
        ax.plot(t, v2_rmse[key], color='#ff7f0e', linewidth=1.8,
                label='LSMCMC V2 ($M{=}1$)')
        ax.axhline(y=sigma, color='gray', linestyle='--', linewidth=1.0,
                   alpha=0.5,
                   label=f'$\\sigma_{{{title.lower()}}}={sigma}$')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[0].set_title('arctan + Cauchy Twin: RMSE Comparison')
    axes[-1].set_xlabel('Assimilation Cycle')
    fig.tight_layout()
    savefig(fig, fname)


# ────────────────────────────────────────────────────
# LETKF blowup
# ────────────────────────────────────────────────────

def plot_letkf_blowup(cycles, vel, sst, ssh, fname='nlgamma_letkf_blowup.pdf'):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), sharex=True)
    for ax, data, ylabel in zip(
            axes, [vel, sst, ssh],
            ['Velocity RMSE (m/s)', 'SST RMSE (K)', 'SSH RMSE (m)']):
        finite = np.isfinite(data)
        ax.semilogy(cycles[finite], data[finite], 's-', color='#2ca02c',
                     linewidth=2.5, markersize=8, label='LETKF', zorder=3)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(0, 40)
    axes[0].set_title('LETKF Divergence: arctan + Cauchy Twin', fontsize=13)
    axes[-1].set_xlabel('Assimilation Cycle', fontsize=12)
    for ax in axes:
        ax.axvline(x=35, color='red', linestyle=':', linewidth=2, alpha=0.8)
    axes[0].annotate('LinAlgError\n(cycle $\\approx 35$)',
                     xy=(35, vel[np.isfinite(vel)][-1]),
                     xytext=(28, 1e12), fontsize=10, color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', fc='white',
                               ec='red', alpha=0.9))
    fig.tight_layout()
    savefig(fig, fname)


# ────────────────────────────────────────────────────
# MCMC diagnostics
# ────────────────────────────────────────────────────

def _run_diagnostic_chain(y_obs, y_ind, sig_y_vec, obs_state_inds,
                          fc_mean_local, truth_local, sig_x_loc,
                          params, ncells, label, rng=None):
    """
    Run one pCN MCMC diagnostic chain on the given obs subset.

    Returns dict with chain, loglik_trace, acc_trace, beta_trace,
    burn_in, acc_rate, beta_final, obs_state_inds, truth_local, ncells.
    """
    from scipy import sparse as sp

    n_obs_states = len(obs_state_inds)
    nobs = len(y_obs)
    sig_y_sq = sig_y_vec ** 2

    state_map = {int(si): i for i, si in enumerate(obs_state_inds)}
    local_obs_col = np.array([state_map[int(yi)] for yi in y_ind])
    H_loc = sp.csr_matrix(
        (np.ones(nobs), (np.arange(nobs), local_obs_col)),
        shape=(nobs, n_obs_states))

    nz_mask = sig_x_loc > 0
    n_nz = int(nz_mask.sum())
    inv_sig_x_nz = 1.0 / sig_x_loc[nz_mask]
    sig_x_nz = sig_x_loc[nz_mask]

    Nf = params.get('nforecast', 25)
    if rng is None:
        rng = np.random.default_rng(42)
    fc_ens = np.zeros((n_obs_states, Nf))
    for j in range(Nf):
        fc_ens[:, j] = fc_mean_local + sig_x_loc * rng.standard_normal(n_obs_states)

    nu = float(params.get('student_t_nu', 2.0))
    half_nup1 = 0.5 * (nu + 1.0)

    def _log_lik(z):
        Hz = H_loc @ z
        pred = np.arctan(Hz)
        eps = y_obs - pred
        return np.sum(-half_nup1 * np.log(1.0 + eps**2 / (nu * sig_y_sq)))

    def _log_trans_all(z):
        diff = z[nz_mask, None] - fc_ens[nz_mask, :]
        return -0.5 * np.sum((diff * inv_sig_x_nz[:, None]) ** 2, axis=0)

    burn_in   = int(params.get('burn_in', 500))
    n_samples = int(params.get('mcmc_N', 2000))
    # β initial: scale with dimension — 2.38/√d is optimal for Gaussian RWM
    beta_init = min(0.95, max(0.02, 2.38 / np.sqrt(n_obs_states)))
    beta = beta_init
    # Target acceptance for pCN diagnostic
    target_acc = 0.35
    total = burn_in + n_samples

    chain = np.zeros((total, n_obs_states))
    loglik_trace = np.zeros(total)
    acc_trace = np.zeros(total, dtype=bool)
    beta_trace = np.zeros(total)

    i_curr = rng.integers(Nf)
    z_curr = fc_ens[:, i_curr].copy()
    log_lik = _log_lik(z_curr)
    n_accept = 0
    _log_beta = np.log(beta)
    # Aggressive early adaptation: high γ₀, low κ
    _rm_gamma0 = 1.0
    _rm_kappa  = 0.55

    for s in range(total):
        lw_raw = _log_trans_all(z_curr)
        lw = lw_raw - lw_raw.max()
        weights = np.exp(lw)
        wsum = weights.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            weights[:] = 1.0 / Nf
        else:
            weights /= wsum
        i_curr = rng.choice(Nf, p=weights)
        m_i = fc_ens[:, i_curr]

        rho = np.sqrt(1.0 - beta ** 2)
        z_prop = z_curr.copy()
        z_prop[nz_mask] = (
            m_i[nz_mask]
            + rho * (z_curr[nz_mask] - m_i[nz_mask])
            + beta * sig_x_nz * rng.standard_normal(n_nz))
        log_lik_prop = _log_lik(z_prop)
        log_alpha = log_lik_prop - log_lik
        if np.log(rng.uniform()) < log_alpha:
            z_curr = z_prop
            log_lik = log_lik_prop
            n_accept += 1
            acc_trace[s] = True

        if s > 0:
            gamma_s = _rm_gamma0 / (1.0 + s) ** _rm_kappa
            _log_beta += gamma_s * (float(acc_trace[s]) - target_acc)
            _log_beta = np.clip(_log_beta, np.log(1e-4), np.log(0.999))
            beta = np.exp(_log_beta)

        chain[s] = z_curr
        loglik_trace[s] = log_lik
        beta_trace[s] = beta

    acc_rate = n_accept / total
    print(f"    [{label}] {n_obs_states} state dims, {nobs} vel obs, Nf={Nf}, "
          f"β_init={beta_init:.3f}, β_final={beta:.4f}, "
          f"acceptance = {acc_rate:.3f}  ({total} samples)")

    return dict(chain=chain, loglik_trace=loglik_trace,
                acc_trace=acc_trace, beta_trace=beta_trace,
                burn_in=burn_in, n_samples=n_samples,
                acc_rate=acc_rate, beta_final=beta,
                obs_state_inds=obs_state_inds,
                truth_local=truth_local, ncells=ncells,
                nobs=nobs, n_obs_states=n_obs_states, label=label)


def _run_diagnostic_chain_hmc(y_obs, y_ind, sig_y_vec, obs_state_inds,
                              fc_mean_local, truth_local, sig_x_loc,
                              params, ncells, label, rng=None):
    """
    Run an HMC MCMC diagnostic chain on the given obs subset.

    Uses leapfrog integration with mass matrix = diag(1/sig_x^2)
    (i.e. we work in the *whitened* coordinate q = (z - m)/sigma).

    Returns dict compatible with _plot_diagnostic_panel.
    """
    from scipy import sparse as sp

    n_obs_states = len(obs_state_inds)
    nobs = len(y_obs)
    sig_y_sq = sig_y_vec ** 2

    state_map = {int(si): i for i, si in enumerate(obs_state_inds)}
    local_obs_col = np.array([state_map[int(yi)] for yi in y_ind])
    H_loc = sp.csr_matrix(
        (np.ones(nobs), (np.arange(nobs), local_obs_col)),
        shape=(nobs, n_obs_states))

    nz_mask = sig_x_loc > 0
    n_nz = int(nz_mask.sum())
    inv_sig_x = np.zeros(n_obs_states)
    inv_sig_x[nz_mask] = 1.0 / sig_x_loc[nz_mask]
    inv_sig_x_sq = inv_sig_x ** 2

    Nf = params.get('nforecast', 25)
    if rng is None:
        rng = np.random.default_rng(42)
    fc_ens = np.zeros((n_obs_states, Nf))
    for j in range(Nf):
        fc_ens[:, j] = fc_mean_local + sig_x_loc * rng.standard_normal(n_obs_states)

    nu = float(params.get('student_t_nu', 2.0))
    half_nup1 = 0.5 * (nu + 1.0)

    # --- log-likelihood and gradient ---
    def _log_lik(z):
        Hz = H_loc @ z
        pred = np.arctan(Hz)
        eps = y_obs - pred
        return np.sum(-half_nup1 * np.log(1.0 + eps**2 / (nu * sig_y_sq)))

    def _grad_log_lik(z):
        """Gradient of log-likelihood w.r.t. z."""
        Hz = np.asarray(H_loc @ z).ravel()
        pred = np.arctan(Hz)
        eps = y_obs - pred
        dpred_dHz = 1.0 / (1.0 + Hz**2)   # d(arctan)/d(Hz)
        # d/d(Hz) of  -half_nup1 * log(1 + eps^2/(nu*s^2))
        #   = -half_nup1 * 2*eps / (nu*s^2 + eps^2) * (-dpred_dHz)
        #   = half_nup1 * 2*eps*dpred_dHz / (nu*s^2 + eps^2)
        denom = nu * sig_y_sq + eps**2
        dloglik_dHz = half_nup1 * 2.0 * eps * dpred_dHz / denom
        return np.asarray(H_loc.T @ dloglik_dHz).ravel()

    # --- log-prior (Gaussian mixture from ensemble) and gradient ---
    def _log_prior(z):
        """log p(z) = log( (1/Nf) sum_k N(z; mu_k, Sigma) )"""
        diff = z[nz_mask, None] - fc_ens[nz_mask, :]     # (n_nz, Nf)
        log_k = -0.5 * np.sum((diff * inv_sig_x[nz_mask, None])**2, axis=0)
        max_lk = log_k.max()
        return max_lk + np.log(np.sum(np.exp(log_k - max_lk))) - np.log(Nf)

    def _grad_log_prior(z):
        """Gradient of log p(z): weighted mixture of (mu_k - z)/sig^2."""
        diff = z[nz_mask, None] - fc_ens[nz_mask, :]     # (n_nz, Nf)
        log_k = -0.5 * np.sum((diff * inv_sig_x[nz_mask, None])**2, axis=0)
        max_lk = log_k.max()
        wt = np.exp(log_k - max_lk)
        wt /= wt.sum()
        # d/dz_j of log-prior = sum_k wt_k * (-(z_j - mu_kj) / sig_j^2)
        grad = np.zeros(n_obs_states)
        grad[nz_mask] = -np.sum(
            diff * inv_sig_x[nz_mask, None]**2 * wt[None, :], axis=1)
        return grad

    def _log_posterior(z):
        return _log_lik(z) + _log_prior(z)

    def _grad_log_posterior(z):
        return _grad_log_lik(z) + _grad_log_prior(z)

    # --- HMC parameters ---
    burn_in   = int(params.get('burn_in', 500))
    n_samples = int(params.get('mcmc_N', 2000))
    total = burn_in + n_samples

    # Mass matrix M = diag(1/sig_x^2) → kinetic = 0.5 p^T M^{-1} p
    # = 0.5 sum_j (p_j * sig_x_j)^2
    # p ~ N(0, M)  → p = inv_sig_x * normal
    # Leapfrog step size and number of steps
    L_steps = int(params.get('hmc_L', 20))
    eps_init = float(params.get('hmc_eps', 0.01))
    eps_hmc = eps_init

    # Dual-averaging adaptation (NUTS-style)
    target_acc = 0.65   # HMC optimal ~ 0.65
    log_eps = np.log(eps_hmc)
    mu_da = np.log(10.0 * eps_init)
    log_eps_bar = 0.0
    H_da = 0.0
    gamma_da = 0.05
    t0_da = 10.0
    kappa_da = 0.75

    chain = np.zeros((total, n_obs_states))
    loglik_trace = np.zeros(total)
    acc_trace = np.zeros(total, dtype=bool)
    beta_trace = np.zeros(total)   # store eps_hmc trajectory

    # Initialize from a random ensemble member
    i_curr = rng.integers(Nf)
    z_curr = fc_ens[:, i_curr].copy()
    log_post_curr = _log_posterior(z_curr)
    n_accept = 0

    for s in range(total):
        # Draw momentum: p ~ N(0, M) where M = diag(inv_sig_x^2)
        p_curr = inv_sig_x * rng.standard_normal(n_obs_states)
        kinetic_curr = 0.5 * np.sum((p_curr * sig_x_loc)**2)

        # Leapfrog
        z_prop = z_curr.copy()
        p_prop = p_curr.copy()
        grad = _grad_log_posterior(z_prop)

        # Half step for momentum
        p_prop += 0.5 * eps_hmc * grad

        for _ in range(L_steps - 1):
            # Full step for position: z += eps * M^{-1} p = eps * sig_x^2 * p
            z_prop += eps_hmc * (sig_x_loc**2) * p_prop
            grad = _grad_log_posterior(z_prop)
            p_prop += eps_hmc * grad

        # Final full position step
        z_prop += eps_hmc * (sig_x_loc**2) * p_prop
        grad = _grad_log_posterior(z_prop)
        # Half step for momentum
        p_prop += 0.5 * eps_hmc * grad

        kinetic_prop = 0.5 * np.sum((p_prop * sig_x_loc)**2)
        log_post_prop = _log_posterior(z_prop)

        # MH accept/reject
        log_alpha = (log_post_prop - log_post_curr
                     + kinetic_curr - kinetic_prop)
        alpha_clipped = min(1.0, np.exp(min(log_alpha, 0.0)))

        if np.log(rng.uniform()) < log_alpha:
            z_curr = z_prop
            log_post_curr = log_post_prop
            n_accept += 1
            acc_trace[s] = True

        # Dual averaging adaptation (during burn-in)
        if s < burn_in:
            m_da = s + 1
            H_da = (1.0 - 1.0/(m_da + t0_da)) * H_da + \
                   (1.0/(m_da + t0_da)) * (target_acc - alpha_clipped)
            log_eps = mu_da - np.sqrt(m_da) / gamma_da * H_da
            log_eps = np.clip(log_eps, np.log(1e-6), np.log(1.0))
            log_eps_bar = m_da**(-kappa_da) * log_eps + \
                          (1.0 - m_da**(-kappa_da)) * log_eps_bar
            eps_hmc = np.exp(log_eps)
        else:
            eps_hmc = np.exp(log_eps_bar)  # use averaged step size

        chain[s] = z_curr
        loglik_trace[s] = _log_lik(z_curr)
        beta_trace[s] = eps_hmc

    acc_rate = n_accept / total
    print(f"    [{label}] {n_obs_states} state dims, {nobs} vel obs, Nf={Nf}, "
          f"L={L_steps}, ε_init={eps_init:.4f}, ε_final={eps_hmc:.6f}, "
          f"acceptance = {acc_rate:.3f}  ({total} samples)")

    return dict(chain=chain, loglik_trace=loglik_trace,
                acc_trace=acc_trace, beta_trace=beta_trace,
                burn_in=burn_in, n_samples=n_samples,
                acc_rate=acc_rate, beta_final=eps_hmc,
                obs_state_inds=obs_state_inds,
                truth_local=truth_local, ncells=ncells,
                nobs=nobs, n_obs_states=n_obs_states, label=label)


def _plot_diagnostic_panel(result, fname, field_type=1):
    """Plot 4-col diagnostic figure for one chain result.

    Parameters
    ----------
    field_type : int
        Which layer-0 field to track: 0=h, 1=u, 2=v, 3=T.
    """
    chain = result['chain']
    burn_in = result['burn_in']
    obs_state_inds = result['obs_state_inds']
    truth_local = result['truth_local']
    ncells = result['ncells']
    acc_rate = result['acc_rate']
    beta = result['beta_final']
    nobs = result['nobs']
    n_obs_states = result['n_obs_states']
    label = result['label']
    _ft_labels = {0: 'h', 1: 'u', 2: 'v', 3: 'T'}

    # Find the most-observed cell for the requested field_type
    most_obs_cell = result.get('most_obs_cell', None)
    picks, labels, cell_indices = [], [], []
    if most_obs_cell is not None:
        # Use the pre-computed most-observed cell
        target_global = most_obs_cell + field_type * ncells
        for idx, si in enumerate(obs_state_inds):
            if int(si) == target_global:
                picks.append(idx)
                labels.append(_ft_labels.get(field_type, '?'))
                cell_indices.append(most_obs_cell)
                break
    if len(picks) == 0:
        # Fallback: first state component matching field_type
        for idx, si in enumerate(obs_state_inds):
            ft = (int(si) % (4 * ncells)) // ncells
            if ft == field_type:
                cell_idx = int(si) % ncells
                picks.append(idx)
                labels.append(_ft_labels.get(field_type, '?'))
                cell_indices.append(cell_idx)
                break
    n_pick = len(picks)
    if n_pick == 0:
        print(f"  WARNING: [{label}] no velocity state components found")
        return

    fig, axes = plt.subplots(n_pick, 4, figsize=(20, 4.5 * n_pick))
    if n_pick == 1:
        axes = axes[np.newaxis, :]

    for row, (idx, lab, ci) in enumerate(zip(picks, labels, cell_indices)):
        ch = chain[:, idx]
        post = ch[burn_in:]
        truth_val = truth_local[idx]

        ax = axes[row, 0]
        ax.plot(ch, linewidth=0.3, color='#1f77b4', alpha=0.7)
        ax.axvline(x=burn_in, color='red', ls='--', lw=1, label='burn-in')
        ax.axhline(y=truth_val, color='green', lw=1.5, alpha=0.8,
                   label='nature run')
        ax.set_ylabel(f'${lab}_{{{ci}}}$', fontsize=19)
        ax.tick_params(axis='both', labelsize=14)
        if row == 0:
            ax.set_title('Trace Plot', fontsize=21)
            ax.legend(fontsize=16)
        if row == n_pick - 1:
            ax.set_xlabel('MCMC iteration', fontsize=18)

        ax = axes[row, 1]
        max_lag = min(300, len(post) // 2)
        if np.std(post) > 0:
            from numpy.fft import fft, ifft
            n = len(post)
            x = post - post.mean()
            fft_x = fft(x, n=2*n)
            acf_full = np.real(ifft(fft_x * np.conj(fft_x)))[:n]
            acf_full /= acf_full[0]
            ax.bar(np.arange(min(max_lag+1, len(acf_full))),
                   acf_full[:max_lag+1],
                   color='#1f77b4', alpha=0.7, width=1.0)
        ax.axhline(y=0, color='black', lw=0.5)
        ax.set_ylim(-0.2, 1.05)
        ax.tick_params(axis='both', labelsize=14)
        if row == 0:
            ax.set_title('Autocorrelation', fontsize=21)
        if row == n_pick - 1:
            ax.set_xlabel('Lag', fontsize=18)

        ax = axes[row, 2]
        run_mean = np.cumsum(post) / np.arange(1, len(post) + 1)
        ax.plot(run_mean, color='#ff7f0e', lw=1.5)
        ax.axhline(y=truth_val, color='green', lw=1.5, alpha=0.8,
                    label='nature run')
        # Widen y-range for readability
        all_vals = np.concatenate([run_mean, [truth_val]])
        y_lo, y_hi = all_vals.min(), all_vals.max()
        y_pad = 0.6 * max(y_hi - y_lo, 1e-6)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.tick_params(axis='both', labelsize=14)
        if row == 0:
            ax.set_title('Running Mean', fontsize=21)
            ax.legend(fontsize=16)
        if row == n_pick - 1:
            ax.set_xlabel('Post burn-in iteration', fontsize=18)

        ax = axes[row, 3]
        ax.hist(post, bins=50, density=True, color='#1f77b4', alpha=0.7)
        ax.axvline(x=truth_val, color='green', lw=2, label='nature run')
        ax.axvline(x=np.mean(post), color='#ff7f0e', lw=2, ls='--',
                   label='post. mean')
        ax.tick_params(axis='both', labelsize=14)
        if row == 0:
            ax.set_title('Posterior', fontsize=21)
            ax.legend(fontsize=16)
        if row == n_pick - 1:
            ax.set_xlabel('Value', fontsize=18)

    kernel_str = 'HMC' if 'HMC' in label else 'pCN'
    if kernel_str == 'HMC':
        param_str = (f'$\\varepsilon$ = {beta:.4f}')
    else:
        param_str = (r'$\beta_{\mathrm{final}}$' + f' = {beta:.3f}')
    fig.suptitle(
        f'MCMC Diagnostics ({label}): arctan + Cauchy {kernel_str}  '
        f'({nobs} vel obs, {n_obs_states} states, '
        f'acc. = {acc_rate:.1%}, '
        f'{param_str})',
        fontsize=22, y=1.02)
    fig.tight_layout()
    savefig(fig, fname)


def generate_mcmc_diagnostics(v1_data, truth_raw):
    """
    Produce TWO diagnostic figures that mirror the actual V1 / V2 filter:
      1) **V1** — block-partition localization identical to
         ``get_observed_blocks_cells``: ALL cells in observed partition
         blocks × 4 layer-0 fields (h0, u0, v0, T0).
      2) **V2** — halo-based localization identical to
         ``_precompute_block_localization``: one representative block,
         all cells within *r_loc* of the block centroid × 4 layer-0
         fields, with Gaspari–Cohn noise inflation.
    """
    import yaml
    from loc_smcmc_swe_exact_from_Gauss import (
        partition_domain, build_H_loc_from_global)
    print("  Running MCMC diagnostic chains ...")

    # ── Load V1 config ──
    cfg_v1 = os.path.join(BASEDIR, 'nlgamma_ldata', 'example_input_nlgamma_twin_v1.yml')
    if not os.path.exists(cfg_v1):
        print(f"  WARNING: {cfg_v1} not found, skipping MCMC diagnostics")
        return
    with open(cfg_v1) as f:
        params_v1 = yaml.safe_load(f)

    # ── Load V2 config ──
    cfg_v2 = os.path.join(BASEDIR, 'nlgamma_ldata',
                          'example_input_nlgamma_twin_v2.yml')
    if os.path.exists(cfg_v2):
        with open(cfg_v2) as f:
            params_v2 = yaml.safe_load(f)
    else:
        params_v2 = dict(params_v1,
                         num_subdomains=700, r_loc=1.5, burn_in=500)

    nx_grid = params_v1['dgx']
    ny_grid = params_v1['dgy']
    ncells = nx_grid * ny_grid
    nlayers = 3
    nfields_l0 = 4                       # h, u, v, T
    dimx = nlayers * nfields_l0 * ncells

    # ── Load observations ──
    obs_nc = os.path.join(BASEDIR, 'output_nlgamma_twin_V1',
                          'synthetic_nlgamma_obs.nc')
    if not os.path.exists(obs_nc):
        print("  WARNING: obs file not found, skipping MCMC diagnostics")
        return
    with Dataset(obs_nc, 'r') as nc:
        yobs_all = np.asarray(nc.variables['yobs_all'][:])
        yind_all = np.asarray(nc.variables['yobs_ind_all'][:])
        sig_y_all = (np.asarray(nc.variables['sig_y_all'][:])
                     if 'sig_y_all' in nc.variables else None)

    cycle = min(119, yobs_all.shape[0] - 1)
    y_cycle = yobs_all[cycle]
    yi_cycle = yind_all[cycle]
    valid = yi_cycle >= 0
    y_all = y_cycle[valid]
    ind_all = yi_cycle[valid].astype(int)
    sig_y_raw = (sig_y_all[cycle][valid] if sig_y_all is not None
                 else np.ones(len(y_all)) * 0.1)

    # Velocity-only mask (field index 1=u, 2=v within layer 0)
    field_type_per_obs = (ind_all % (nfields_l0 * ncells)) // ncells
    vel_mask = (field_type_per_obs == 1) | (field_type_per_obs == 2)

    y_obs_vel = y_all[vel_mask]
    y_ind_vel = ind_all[vel_mask]
    sig_y_vel = sig_y_raw[vel_mask]

    # ── Build sig_x for full domain ──
    sig_x_uv  = params_v1.get('sig_x_uv', 0.15)
    sig_x_ssh = params_v1.get('sig_x_ssh', 0.50)
    sig_x_sst = params_v1.get('sig_x_sst', 1.0)
    sig_x_full = np.zeros(dimx)
    for k in range(nlayers):
        base = k * nfields_l0 * ncells
        sig_x_full[base:base + ncells] = sig_x_ssh         # h
        sig_x_full[base + ncells:base + 2*ncells] = sig_x_uv   # u
        sig_x_full[base + 2*ncells:base + 3*ncells] = sig_x_uv # v
        sig_x_full[base + 3*ncells:base + 4*ncells] = sig_x_sst  # T

    # ═══════════════════════════════════════════════════════════
    # (A)  V1 — per-field chains on block-partition reduced domain
    #
    # The posterior factorizes across fields because:
    #   - prior is independent per field
    #   - each obs picks a single u or v component via H
    # So g(y|z)·f(z) = g_u(y_u|u)·f_u(u) · g_v(y_v|v)·f_v(v) · f_h(h) · f_T(T)
    # We run separate chains for u and v (observed); h and T have no obs.
    # ═══════════════════════════════════════════════════════════
    print("  --- V1 diagnostic (per-field chains on block-partition) ---")
    Gamma_v1 = int(params_v1.get('num_subdomains', 50))
    _, part_labels_v1, _, _, _, _, _ = partition_domain(
        ny_grid, nx_grid, N=Gamma_v1)

    # Find cells in observed blocks
    obs_cell_v1 = y_ind_vel % ncells
    obs_r, obs_c = np.unravel_index(obs_cell_v1, (ny_grid, nx_grid))
    obs_blocks = np.unique(part_labels_v1[obs_r, obs_c])
    block_mask = np.isin(part_labels_v1, obs_blocks)
    ij_v1 = np.argwhere(block_mask)
    flat0_v1 = np.ravel_multi_index(
        (ij_v1[:, 0], ij_v1[:, 1]), (ny_grid, nx_grid))
    n_block_cells = len(flat0_v1)
    print(f"    Γ={Gamma_v1}, {len(obs_blocks)} observed blocks, "
          f"{n_block_cells} cells")

    # Split obs into u-only and v-only
    obs_field_type = (y_ind_vel % (nfields_l0 * ncells)) // ncells
    u_obs_mask = obs_field_type == 1
    v_obs_mask = obs_field_type == 2

    # Per-field state indices (just the cells for that field)
    u_state_inds = np.sort(flat0_v1 + 1 * ncells)   # u0 block
    v_state_inds = np.sort(flat0_v1 + 2 * ncells)   # v0 block

    params_v1_diag = dict(params_v1, mcmc_N=2000, burn_in=500,
                          nforecast=25,
                          hmc_L=10, hmc_eps=0.005)

    # Find the most-observed cell for u-velocity
    u_obs_cells = y_ind_vel[u_obs_mask] % ncells
    u_cell_counts = np.bincount(u_obs_cells.astype(int), minlength=ncells)
    most_obs_u_cell = int(np.argmax(u_cell_counts))
    most_obs_u_count = int(u_cell_counts[most_obs_u_cell])
    r_mo, c_mo = np.unravel_index(most_obs_u_cell, (ny_grid, nx_grid))
    print(f"    Most-observed u-cell: {most_obs_u_cell} "
          f"(row={r_mo}, col={c_mo}), {most_obs_u_count} obs")

    # Only run u-field chain (v-row removed)
    for fld_name, fld_type, state_inds, fld_obs_mask in [
            ('u', 1, u_state_inds, u_obs_mask)]:
        y_fld = y_obs_vel[fld_obs_mask]
        yi_fld = y_ind_vel[fld_obs_mask]
        sy_fld = sig_y_vel[fld_obs_mask]
        # Keep only obs that map into state_inds
        col_set = set(state_inds.tolist())
        kept = np.array([int(g) in col_set for g in yi_fld])
        y_fld = y_fld[kept]
        yi_fld = yi_fld[kept]
        sy_fld = sy_fld[kept]
        fc_fld = v1_data['raw'][cycle].ravel()[state_inds].copy()
        truth_fld = truth_raw[cycle + 1].ravel()[state_inds]
        sx_fld = sig_x_full[state_inds]
        print(f"    {fld_name}-field: {len(state_inds)} state dims, "
              f"{len(y_fld)} obs")
        res = _run_diagnostic_chain_hmc(
            y_fld, yi_fld, sy_fld, state_inds,
            fc_fld, truth_fld, sx_fld, params_v1_diag, ncells,
            label=f'V1 HMC: {fld_name}-field',
            rng=np.random.default_rng(42))
        res['most_obs_cell'] = most_obs_u_cell
        _plot_diagnostic_panel(
            res, f'nlgamma_mcmc_diag_v1_{fld_name}.pdf',
            field_type=fld_type)

    # ═══════════════════════════════════════════════════════════
    # (B)  V2 — halo-based localization (matches filter)
    # ═══════════════════════════════════════════════════════════
    print("  --- V2 diagnostic (halo-localized block) ---")
    Gamma_v2 = int(params_v2.get('num_subdomains', 700))
    r_loc = float(params_v2.get('r_loc', 1.5))
    _, part_labels_v2, _, _, _, _, _ = partition_domain(
        ny_grid, nx_grid, N=Gamma_v2)

    grid_rows, grid_cols = np.meshgrid(
        np.arange(ny_grid), np.arange(nx_grid), indexing='ij')
    obs_cell_v2 = y_ind_vel % ncells
    obs_row_v2, obs_col_v2 = np.unravel_index(
        obs_cell_v2, (ny_grid, nx_grid))
    unique_blocks_v2 = np.unique(part_labels_v2)

    # Do a Gaspari-Cohn helper (same as filter's _gaspari_cohn)
    def _gaspari_cohn(d, c):
        r = np.abs(d) / c
        gc = np.zeros_like(d, dtype=float)
        m1 = r <= 1.0
        m2 = (r > 1.0) & (r <= 2.0)
        r1 = r[m1]
        gc[m1] = 1.0 - (5./3.)*r1**2 + (5./8.)*r1**3 \
                 + 0.5*r1**4 - 0.25*r1**5
        r2 = r[m2]
        gc[m2] = 4.0 - 5.*r2 + (5./3.)*r2**2 + (5./8.)*r2**3 \
                 - 0.5*r2**4 + (1./12.)*r2**5 - 2./(3.*r2)
        return np.clip(gc, 0.0, 1.0)

    # Find a representative block with observations
    best_block = None
    best_nobs = 0
    for block_id in unique_blocks_v2:
        bm = (part_labels_v2 == block_id)
        bij = np.argwhere(bm)
        cy, cx = bij[:, 0].mean(), bij[:, 1].mean()
        dy_o = np.abs(obs_row_v2 - cy)
        dx_o = np.abs(obs_col_v2 - cx)
        dist_o = np.sqrt(dy_o**2 + dx_o**2)
        n_near = int(np.sum(dist_o < r_loc))
        if n_near > best_nobs:
            best_nobs = n_near
            best_block = block_id

    if best_block is None or best_nobs == 0:
        print("  WARNING: no V2 block has observations, skipping V2 diag")
    else:
        bm = (part_labels_v2 == best_block)
        bij = np.argwhere(bm)
        cy, cx = bij[:, 0].mean(), bij[:, 1].mean()

        # Nearby observations
        dy_o = np.abs(obs_row_v2 - cy)
        dx_o = np.abs(obs_col_v2 - cx)
        dist_obs_v2 = np.sqrt(dy_o**2 + dx_o**2)
        nearby = dist_obs_v2 < r_loc
        nearby_idx = np.where(nearby)[0]
        obs_global_v2 = y_ind_vel[nearby_idx]

        # Halo cells (grid cells within r_loc of centroid)
        dy_g = np.abs(grid_rows - cy)
        dx_g = np.abs(grid_cols - cx)
        halo_mask = np.sqrt(dy_g**2 + dx_g**2) < r_loc
        halo_ij = np.argwhere(halo_mask)
        halo_flat_0 = np.ravel_multi_index(
            (halo_ij[:, 0], halo_ij[:, 1]), (ny_grid, nx_grid))
        block_flat_0 = np.ravel_multi_index(
            (bij[:, 0], bij[:, 1]), (ny_grid, nx_grid))
        halo_flat_0 = np.unique(np.concatenate(
            [halo_flat_0, block_flat_0]))

        # Expand to 4 layer-0 fields
        halo_cells = np.sort(np.concatenate(
            [halo_flat_0 + f * ncells for f in range(nfields_l0)]))
        block_cells = np.sort(np.concatenate(
            [block_flat_0 + f * ncells for f in range(nfields_l0)]))

        H_loc_v2 = build_H_loc_from_global(
            obs_global_v2, halo_cells, drop_unmapped=True)

        # Gaspari-Cohn noise inflation
        dist_local_v2 = dist_obs_v2[nearby_idx][:H_loc_v2.shape[0]]
        gc_w = _gaspari_cohn(dist_local_v2, r_loc)
        gc_w = np.maximum(gc_w, 1e-6)

        # Build y_obs and sig_y for this block
        mapped_obs = nearby_idx[:H_loc_v2.shape[0]]
        y_obs_v2 = y_obs_vel[mapped_obs]
        sig_y_v2_base = sig_y_vel[mapped_obs]
        gc_noise_inflate = params_v2.get('gc_noise_inflate', True)
        if gc_noise_inflate:
            sig_y_v2 = sig_y_v2_base / np.sqrt(gc_w)
        else:
            sig_y_v2 = sig_y_v2_base.copy()
        y_ind_v2_mapped = y_ind_vel[mapped_obs]

        print(f"    Γ={Gamma_v2}, r_loc={r_loc}, block {best_block}, "
              f"{len(halo_flat_0)} halo cells → {len(halo_cells)} "
              f"halo states, {len(block_cells)} block states, "
              f"{H_loc_v2.shape[0]} obs")

        fc_mean_v2 = v1_data['raw'][cycle].ravel()[halo_cells].copy()
        truth_v2 = truth_raw[cycle + 1].ravel()[halo_cells]
        sig_x_v2 = sig_x_full[halo_cells]

        # Find most-observed cell within V2 halo
        v2_obs_cells = y_ind_v2_mapped % ncells
        v2_cell_counts = np.bincount(v2_obs_cells.astype(int),
                                      minlength=ncells)
        most_obs_v2_cell = int(np.argmax(v2_cell_counts))
        print(f"    V2 most-observed cell: {most_obs_v2_cell}")

        res_v2 = _run_diagnostic_chain(
            y_obs_v2, y_ind_v2_mapped, sig_y_v2, halo_cells,
            fc_mean_v2, truth_v2, sig_x_v2, params_v2, ncells,
            label='V2: halo block', rng=np.random.default_rng(99))
        res_v2['most_obs_cell'] = most_obs_v2_cell
        _plot_diagnostic_panel(res_v2, 'nlgamma_mcmc_diag_v2.pdf')


# ────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────

def main():
    print("=== Generating NL-Gamma (arctan + Cauchy) Figures ===\n")

    # ── Load data ──
    v1_nc = os.path.join(BASEDIR, 'output_nlgamma_twin_V1',
                         'mlswe_lsmcmc_out.nc')
    truth_nc = os.path.join(BASEDIR, 'output_nlgamma_twin_V1',
                            'truth_trajectory.nc')
    letkf_log = os.path.join(BASEDIR, 'nlgamma_letkf.log')

    v1 = load_analysis(v1_nc)
    truth_raw, H_b = load_truth(truth_nc)
    ny, nx = v1['ny'], v1['nx']
    print(f"  V1: vel={np.mean(v1['rmse_vel']):.4f}  "
          f"sst={np.mean(v1['rmse_sst']):.4f}  "
          f"ssh={np.mean(v1['rmse_ssh']):.4f}")

    # V2 M=1 single run
    v2_nc = os.path.join(BASEDIR, 'output_nlgamma_twin_V2',
                         'mlswe_lsmcmc_out.nc')
    with Dataset(v2_nc) as nc:
        v2_raw = np.asarray(nc['lsmcmc_mean'][:])
    v2_vel, v2_sst, v2_ssh = compute_rmse_from_raw(v2_raw, truth_raw)
    print(f"  V2 M=1: vel={np.mean(v2_vel):.4f}  "
          f"sst={np.mean(v2_sst):.4f}  "
          f"ssh={np.mean(v2_ssh):.4f}")

    # HYCOM
    hycom_init, hycom_final = _load_hycom_pair(v1['obs_times'], ny, nx)

    # ── Load V1 HMC (if available) ──
    v1_hmc_nc = os.path.join(BASEDIR, 'output_nlgamma_twin_V1_hmc',
                             'mlswe_lsmcmc_out.nc')
    v1_hmc_rmse = None
    if os.path.exists(v1_hmc_nc):
        v1_hmc = load_analysis(v1_hmc_nc)
        v1_hmc_rmse = {'vel': v1_hmc['rmse_vel'], 'sst': v1_hmc['rmse_sst'],
                       'ssh': v1_hmc['rmse_ssh']}
        print(f"  V1 HMC: vel={np.mean(v1_hmc['rmse_vel']):.4f}  "
              f"sst={np.mean(v1_hmc['rmse_sst']):.4f}  "
              f"ssh={np.mean(v1_hmc['rmse_ssh']):.4f}")
    else:
        print(f"  WARNING: V1 HMC output not found at {v1_hmc_nc}")

    # ── Figure 1: RMSE comparison ──
    v1_rmse  = {'vel': v1['rmse_vel'], 'sst': v1['rmse_sst'],
                'ssh': v1['rmse_ssh']}
    v2_rmse = {'vel': v2_vel, 'sst': v2_sst, 'ssh': v2_ssh}
    plot_compare_rmse(v1_rmse, v2_rmse, v1_hmc_rmse=v1_hmc_rmse)

    # ── Figure 2: LETKF blowup ──
    if os.path.exists(letkf_log):
        cycles, vel, sst, ssh = parse_letkf_log(letkf_log)
        if len(cycles) > 0:
            print(f"  LETKF: {len(cycles)} cycles before crash")
            plot_letkf_blowup(cycles, vel, sst, ssh)
    else:
        print("  WARNING: LETKF log not found, skipping blowup figure")

    # ── Figure 3: V1 field panels (like nltwin_v1_fields.pdf) ──
    plot_field_panels(v1, 'LSMCMC V1', 'nlgamma_v1_fields.pdf',
                       hycom_init=hycom_init, hycom_final=hycom_final)

    # ── Figure 4: V2 M=1 field panels ──
    v2_data = {
        'raw': v2_raw,
        'rmse_vel': v2_vel, 'rmse_sst': v2_sst, 'rmse_ssh': v2_ssh,
        'H_b': H_b, 'obs_times': v1['obs_times'], 'ny': ny, 'nx': nx,
    }
    plot_field_panels(v2_data, 'LSMCMC V2 ($M{=}1$)', 'nlgamma_v2_fields.pdf',
                       hycom_init=hycom_init, hycom_final=hycom_final)

    # ── Figure 5: MCMC diagnostics ──
    generate_mcmc_diagnostics(v1, truth_raw)

    print("\nDone!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
mcmc_diagnostics.py
===================
Standalone script to diagnose MCMC convergence for the NL-LSMCMC V2
block-localized filter.

Runs one DA cycle (first cycle) and records the **full MCMC chain**
(including burn-in) for a few representative blocks.  Produces:

  1. Traceplots of selected state components (h, u, v, T)
  2. Autocorrelation plots
  3. Running-mean plots
  4. Multiple-chain R-hat and ESS estimates (via chunked single chain)
  5. Log-likelihood trace
  6. Comparison of posterior mean vs nature run

Usage:
    python3 mcmc_diagnostics.py [config.yml]
"""

import os, sys, yaml, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse as sp
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(__file__))

from mlswe.model import MLSWE, coriolis_array
import run_mlswe_lsmcmc_ldata_V1 as run_mlswe_lsmcmc
from run_mlswe_lsmcmc_nldata_V1_twin import (
    generate_truth_and_obs, _load_sst_ssh_refs, obs_operator_arctan,
    _save_truth,
)
from mlswe.lsmcmc_nl_V2 import (
    NL_SMCMC_MLSWE_Filter_V2,
    partition_domain, lonlat_to_dxdy, build_H_loc_from_global,
)

# =====================================================================
#  Diagnostic MCMC worker — returns FULL chain + log-lik trace
# =====================================================================
def _diag_mcmc_worker(fc_halo, y_local, H_loc, sigma_y_loc,
                       sig_x_halo, block_in_halo,
                       n_samples, burn_in, step_size,
                       adapt, adapt_interval, target_acc, thin,
                       obs_op_name, seed,
                       kernel='gibbs_mh', pcn_beta=0.3,
                       hmc_leapfrog_steps=10):
    """
    Like _block_mcmc_worker but returns full chain including burn-in,
    log-likelihood trace, acceptance trace, and step-size trace.
    Supports kernels: gibbs_mh, pcn, mala, hmc.
    """
    rng = np.random.default_rng(seed)

    if obs_op_name == 'arctan':
        def obs_operator(z, H, _):
            return np.arctan(H @ z)
    else:
        obs_operator = None

    Nf = fc_halo.shape[1]
    dim = fc_halo.shape[0]

    nz_mask = sig_x_halo > 0
    n_nz = int(nz_mask.sum())
    if n_nz == 0:
        return None

    inv_sig_x_nz = 1.0 / sig_x_halo[nz_mask]
    sig_x_nz = sig_x_halo[nz_mask]
    sig_x_sq_nz = sig_x_nz ** 2
    prop_scale = step_size * sig_x_nz
    beta = float(pcn_beta)
    hmc_L = int(hmc_leapfrog_steps)

    def _log_lik(z):
        if obs_operator is not None:
            y_pred = obs_operator(z, H_loc, None)
        else:
            y_pred = H_loc @ z
        residual = y_local - y_pred
        return -0.5 * np.sum((residual / sigma_y_loc) ** 2)

    def _log_trans(z, m_i):
        diff = z[nz_mask] - m_i[nz_mask]
        return -0.5 * np.sum((diff * inv_sig_x_nz) ** 2)

    def _log_trans_all(z):
        diff = z[nz_mask, None] - fc_halo[nz_mask, :]
        return -0.5 * np.sum((diff * inv_sig_x_nz[:, None]) ** 2,
                             axis=0)

    def _grad_log_target(z, m_i):
        Hz = H_loc @ z
        if obs_op_name == 'arctan':
            pred = np.arctan(Hz)
            dpred = 1.0 / (1.0 + Hz ** 2)
        else:
            pred = Hz
            dpred = np.ones_like(Hz)
        residual = y_local - pred
        grad = np.asarray(
            H_loc.T @ (dpred * residual / sigma_y_loc ** 2)
        ).ravel()
        grad[nz_mask] -= (
            z[nz_mask] - m_i[nz_mask]) / sig_x_sq_nz
        return grad

    # Init
    i_curr = rng.integers(Nf)
    z_curr = fc_halo[:, i_curr].copy()
    log_lik = _log_lik(z_curr)
    log_trans = _log_trans(z_curr, fc_halo[:, i_curr])

    total_iters = burn_in + n_samples * thin

    n_block_cells = len(block_in_halo)
    chain = np.zeros((total_iters, n_block_cells))
    loglik_trace = np.zeros(total_iters)
    accept_trace = np.zeros(total_iters, dtype=bool)
    step_trace = np.zeros(total_iters)
    i_trace = np.zeros(total_iters, dtype=int)

    n_accept = 0
    _log_beta = np.log(max(beta, 1e-4))  # Robbins-Monro state

    for s in range(total_iters):
        # Gibbs step for i (common)
        lw_raw = _log_trans_all(z_curr)
        lw = lw_raw - lw_raw.max()
        weights = np.exp(lw)
        wsum = weights.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            weights[:] = 1.0 / Nf
        else:
            weights /= wsum
        i_curr = rng.choice(Nf, p=weights)
        log_trans = lw_raw[i_curr]
        m_i = fc_halo[:, i_curr]

        accepted = False

        if kernel == 'gibbs_mh':
            z_prop = z_curr.copy()
            z_prop[nz_mask] += prop_scale * rng.standard_normal(n_nz)
            log_lik_prop = _log_lik(z_prop)
            log_trans_prop = _log_trans(z_prop, m_i)
            log_alpha = ((log_lik_prop + log_trans_prop)
                         - (log_lik + log_trans))
            if np.log(rng.random()) < log_alpha:
                z_curr = z_prop
                log_lik = log_lik_prop
                log_trans = log_trans_prop
                n_accept += 1
                accepted = True

        elif kernel == 'pcn':
            rho = np.sqrt(1.0 - beta ** 2)
            z_prop = z_curr.copy()
            z_prop[nz_mask] = (
                m_i[nz_mask]
                + rho * (z_curr[nz_mask] - m_i[nz_mask])
                + beta * sig_x_nz * rng.standard_normal(n_nz))
            log_lik_prop = _log_lik(z_prop)
            log_alpha = log_lik_prop - log_lik
            if np.log(rng.random()) < log_alpha:
                z_curr = z_prop
                log_lik = log_lik_prop
                log_trans = _log_trans(z_curr, m_i)
                n_accept += 1
                accepted = True

        elif kernel == 'mala':
            grad = _grad_log_target(z_curr, m_i)
            tau = step_size
            drift_nz = (tau ** 2 / 2) * sig_x_sq_nz * grad[nz_mask]
            z_prop = z_curr.copy()
            z_prop[nz_mask] += (
                drift_nz
                + tau * sig_x_nz * rng.standard_normal(n_nz))
            if not np.all(np.isfinite(z_prop)):
                pass
            else:
                log_lik_prop = _log_lik(z_prop)
                log_trans_prop = _log_trans(z_prop, m_i)
                grad_prop = _grad_log_target(z_prop, m_i)
                drift_rev_nz = (
                    (tau ** 2 / 2) * sig_x_sq_nz
                    * grad_prop[nz_mask])
                diff_fwd = (z_prop[nz_mask] - z_curr[nz_mask]
                            - drift_nz) / (tau * sig_x_nz)
                log_q_fwd = -0.5 * np.sum(diff_fwd ** 2)
                diff_rev = (z_curr[nz_mask] - z_prop[nz_mask]
                            - drift_rev_nz) / (tau * sig_x_nz)
                log_q_rev = -0.5 * np.sum(diff_rev ** 2)
                log_alpha = (
                    (log_lik_prop + log_trans_prop + log_q_rev)
                    - (log_lik + log_trans + log_q_fwd))
                if np.log(rng.random()) < log_alpha:
                    z_curr = z_prop
                    log_lik = log_lik_prop
                    log_trans = log_trans_prop
                    n_accept += 1
                    accepted = True

        elif kernel == 'hmc':
            p_nz = rng.standard_normal(n_nz) * inv_sig_x_nz
            KE_old = 0.5 * np.sum(p_nz ** 2 * sig_x_sq_nz)
            q = z_curr.copy()
            eps = step_size
            grad_nz = _grad_log_target(q, m_i)[nz_mask]
            p_nz += (eps / 2) * grad_nz
            diverged = False
            for ll in range(hmc_L):
                q[nz_mask] += eps * sig_x_sq_nz * p_nz
                if not np.all(np.isfinite(q[nz_mask])):
                    diverged = True
                    break
                grad_nz = _grad_log_target(q, m_i)[nz_mask]
                if ll < hmc_L - 1:
                    p_nz += eps * grad_nz
                else:
                    p_nz += (eps / 2) * grad_nz
            if diverged or not np.all(np.isfinite(p_nz)):
                pass
            else:
                KE_new = 0.5 * np.sum(p_nz ** 2 * sig_x_sq_nz)
                log_lik_prop = _log_lik(q)
                log_trans_prop = _log_trans(q, m_i)
                H_old = -log_lik - log_trans + KE_old
                H_new = -log_lik_prop - log_trans_prop + KE_new
                log_alpha = -(H_new - H_old)
                if np.log(rng.random()) < log_alpha:
                    z_curr = q
                    log_lik = log_lik_prop
                    log_trans = log_trans_prop
                    n_accept += 1
                    accepted = True

        # Adapt: Robbins-Monro for pCN, block for others
        if adapt and s > 0:
            if kernel == 'pcn':
                _rm_g = 0.5 / (1.0 + s) ** 0.6
                _log_beta += _rm_g * (float(accepted) - target_acc)
                _log_beta = max(min(_log_beta, np.log(0.999)), np.log(1e-4))
                beta = np.exp(_log_beta)
            elif s < burn_in and s % adapt_interval == 0:
                acc_now = n_accept / (s + 1)
                if acc_now < target_acc * 0.6:
                    step_size *= 0.8
                elif acc_now > target_acc * 1.6:
                    step_size *= 1.2
                step_size = np.clip(step_size, 1e-6, 10.0)
                if kernel == 'gibbs_mh':
                    prop_scale = step_size * sig_x_nz

        # Record
        chain[s, :] = z_curr[block_in_halo]
        loglik_trace[s] = log_lik
        accept_trace[s] = accepted
        step_trace[s] = step_size if kernel != 'pcn' else beta
        i_trace[s] = i_curr

    acc_rate = n_accept / total_iters

    return {
        'chain': chain,               # (total_iters, n_block_cells)
        'loglik_trace': loglik_trace,  # (total_iters,)
        'accept_trace': accept_trace,  # (total_iters,) bool
        'step_trace': step_trace,      # (total_iters,)
        'i_trace': i_trace,            # (total_iters,)
        'acc_rate': acc_rate,
        'burn_in': burn_in,
        'n_samples': n_samples,
        'block_in_halo': block_in_halo,
        'n_obs': len(y_local),
        'dim_halo': dim,
    }


# =====================================================================
#  Convergence diagnostics
# =====================================================================
def compute_ess(chain_1d):
    """Effective sample size via autocorrelation (Geyer's initial
    positive sequence estimator, simplified)."""
    n = len(chain_1d)
    if n < 4:
        return n
    x = chain_1d - chain_1d.mean()
    var = np.var(x)
    if var < 1e-30:
        return n

    # FFT-based autocorrelation
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n]
    acf /= acf[0]

    # Sum autocorrelations until they go negative
    tau = 1.0
    for k in range(1, n // 2):
        if acf[k] < 0:
            break
        tau += 2 * acf[k]

    return n / tau


def split_rhat(chain_1d, n_splits=4):
    """Split-R-hat: split single chain into n_splits pieces and
    compute the Gelman-Rubin R-hat."""
    n = len(chain_1d)
    chunk = n // n_splits
    if chunk < 10:
        return np.nan

    chains = [chain_1d[i*chunk:(i+1)*chunk] for i in range(n_splits)]
    chain_means = np.array([c.mean() for c in chains])
    chain_vars = np.array([c.var(ddof=1) for c in chains])

    grand_mean = chain_means.mean()
    B = chunk * np.var(chain_means, ddof=1)   # between-chain var
    W = np.mean(chain_vars)                    # within-chain var

    if W < 1e-30:
        return 1.0

    var_hat = (1 - 1/chunk) * W + B / chunk
    rhat = np.sqrt(var_hat / W)
    return rhat


def autocorr(chain_1d, max_lag=200):
    """Compute autocorrelation function up to max_lag."""
    n = len(chain_1d)
    max_lag = min(max_lag, n - 1)
    x = chain_1d - chain_1d.mean()
    var = np.var(x)
    if var < 1e-30:
        return np.ones(max_lag + 1)
    fft_x = np.fft.fft(x, n=2*n)
    acf_full = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n]
    acf_full /= acf_full[0]
    return acf_full[:max_lag + 1]


# =====================================================================
#  Plotting functions
# =====================================================================
def plot_traceplots(diag, truth_block, outdir, block_label=''):
    """Traceplot of selected state components."""
    chain = diag['chain']
    burn_in = diag['burn_in']
    n_iters, n_cells = chain.shape

    # Pick up to 8 representative components
    # Assume block cells = [h, u, v, T] for layer0
    nc_block = n_cells // 4   # cells per field in block
    if nc_block == 0:
        nc_block = 1

    # Pick the first spatial cell for each field
    field_names = ['h (SSH)', 'u', 'v', 'T (SST)']
    indices = []
    for f in range(min(4, n_cells)):
        idx = f * nc_block
        if idx < n_cells:
            indices.append(idx)

    n_panels = len(indices) + 1  # +1 for log-lik
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3*n_panels),
                             sharex=True)

    # Log-likelihood trace
    ax = axes[0]
    ax.plot(diag['loglik_trace'], linewidth=0.5, alpha=0.7, color='k')
    ax.axvline(burn_in, color='r', linestyle='--', linewidth=1,
               label=f'burn-in = {burn_in}')
    ax.set_ylabel('log-likelihood')
    ax.set_title(f'MCMC Diagnostics — Block {block_label}  '
                 f'(halo_dim={diag["dim_halo"]}, n_obs={diag["n_obs"]}, '
                 f'acc_rate={diag["acc_rate"]:.3f})')
    ax.legend(loc='upper right')

    for i, idx in enumerate(indices):
        ax = axes[i + 1]
        ax.plot(chain[:, idx], linewidth=0.3, alpha=0.7)
        ax.axvline(burn_in, color='r', linestyle='--', linewidth=1)

        # Post-burn-in stats
        post = chain[burn_in:, idx]
        ess = compute_ess(post)
        rhat = split_rhat(post)
        pmean = post.mean()

        truth_val = truth_block[idx] if idx < len(truth_block) else np.nan
        ax.axhline(truth_val, color='g', linestyle='-', linewidth=1.5,
                   alpha=0.8, label=f'truth = {truth_val:.4f}')
        ax.axhline(pmean, color='orange', linestyle='--', linewidth=1,
                   alpha=0.8, label=f'post mean = {pmean:.4f}')

        fname = field_names[i] if i < len(field_names) else f'dim {idx}'
        ax.set_ylabel(fname)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'{fname}: ESS={ess:.0f}, R-hat={rhat:.3f}',
                     fontsize=10)

    axes[-1].set_xlabel('MCMC iteration')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'traceplot_{block_label}.png'),
                dpi=150)
    plt.close()
    print(f'  Saved traceplot_{block_label}.png')


def plot_autocorrelation(diag, outdir, block_label=''):
    """Autocorrelation plots for selected components."""
    chain = diag['chain']
    burn_in = diag['burn_in']
    n_iters, n_cells = chain.shape

    nc_block = max(n_cells // 4, 1)
    field_names = ['h (SSH)', 'u', 'v', 'T (SST)']
    indices = [f * nc_block for f in range(min(4, n_cells))
               if f * nc_block < n_cells]

    fig, axes = plt.subplots(len(indices), 1,
                             figsize=(10, 3*len(indices)),
                             sharex=True)
    if len(indices) == 1:
        axes = [axes]

    max_lag = min(500, (n_iters - burn_in) // 2)
    for i, idx in enumerate(indices):
        post = chain[burn_in:, idx]
        acf = autocorr(post, max_lag)

        ax = axes[i]
        ax.bar(range(len(acf)), acf, width=1.0, alpha=0.7)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axhline(1.96 / np.sqrt(len(post)), color='r',
                   linestyle='--', linewidth=0.8, label='95% CI')
        ax.axhline(-1.96 / np.sqrt(len(post)), color='r',
                   linestyle='--', linewidth=0.8)

        fname = field_names[i] if i < len(field_names) else f'dim {idx}'
        ax.set_ylabel(f'ACF ({fname})')
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Lag')
    plt.suptitle(f'Autocorrelation — Block {block_label}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'autocorr_{block_label}.png'),
                dpi=150)
    plt.close()
    print(f'  Saved autocorr_{block_label}.png')


def plot_running_mean(diag, truth_block, outdir, block_label=''):
    """Running mean to see convergence."""
    chain = diag['chain']
    burn_in = diag['burn_in']
    n_iters, n_cells = chain.shape

    nc_block = max(n_cells // 4, 1)
    field_names = ['h (SSH)', 'u', 'v', 'T (SST)']
    indices = [f * nc_block for f in range(min(4, n_cells))
               if f * nc_block < n_cells]

    fig, axes = plt.subplots(len(indices), 1,
                             figsize=(10, 3*len(indices)),
                             sharex=True)
    if len(indices) == 1:
        axes = [axes]

    post_start = burn_in
    for i, idx in enumerate(indices):
        post = chain[post_start:, idx]
        cumsum = np.cumsum(post)
        running_mean = cumsum / np.arange(1, len(post) + 1)

        ax = axes[i]
        ax.plot(running_mean, linewidth=0.8)

        truth_val = truth_block[idx] if idx < len(truth_block) else np.nan
        ax.axhline(truth_val, color='g', linestyle='-', linewidth=1.5,
                   alpha=0.8, label=f'truth = {truth_val:.4f}')

        fname = field_names[i] if i < len(field_names) else f'dim {idx}'
        ax.set_ylabel(f'Running mean ({fname})')
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Post burn-in iteration')
    plt.suptitle(f'Running Mean — Block {block_label}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'running_mean_{block_label}.png'),
                dpi=150)
    plt.close()
    print(f'  Saved running_mean_{block_label}.png')


def plot_step_size_and_acceptance(diag, outdir, block_label=''):
    """Step size adaptation and cumulative acceptance trace."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(diag['step_trace'], linewidth=0.8)
    ax1.axvline(diag['burn_in'], color='r', linestyle='--', linewidth=1)
    ax1.set_ylabel('Step size')
    ax1.set_title(f'Step Size Adaptation — Block {block_label}')

    # Cumulative acceptance rate
    cum_acc = np.cumsum(diag['accept_trace']) / np.arange(
        1, len(diag['accept_trace']) + 1)
    ax2.plot(cum_acc, linewidth=0.8)
    ax2.axvline(diag['burn_in'], color='r', linestyle='--', linewidth=1)
    ax2.axhline(0.234, color='g', linestyle='--', alpha=0.5,
                label='target 0.234')
    ax2.set_ylabel('Cumulative acceptance rate')
    ax2.set_xlabel('MCMC iteration')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'step_accept_{block_label}.png'),
                dpi=150)
    plt.close()
    print(f'  Saved step_accept_{block_label}.png')


def print_summary_table(diagnostics, truth_blocks):
    """Print a table of convergence diagnostics for all blocks."""
    print("\n" + "=" * 85)
    print(f"{'Block':>6} {'nObs':>5} {'hDim':>6} {'AccR':>6} "
          f"{'ESS_h':>7} {'ESS_u':>7} {'ESS_v':>7} {'ESS_T':>7} "
          f"{'Rhat_h':>7} {'Rhat_u':>7}")
    print("-" * 85)

    for blk_label, diag in diagnostics.items():
        chain = diag['chain']
        burn_in = diag['burn_in']
        n_cells = chain.shape[1]
        nc_block = max(n_cells // 4, 1)

        indices = [f * nc_block for f in range(4)
                   if f * nc_block < n_cells]

        ess_vals = []
        rhat_vals = []
        for idx in indices:
            post = chain[burn_in:, idx]
            ess_vals.append(compute_ess(post))
            rhat_vals.append(split_rhat(post))
        while len(ess_vals) < 4:
            ess_vals.append(np.nan)
        while len(rhat_vals) < 4:
            rhat_vals.append(np.nan)

        print(f"{blk_label:>6} {diag['n_obs']:>5} {diag['dim_halo']:>6} "
              f"{diag['acc_rate']:>6.3f} "
              f"{ess_vals[0]:>7.0f} {ess_vals[1]:>7.0f} "
              f"{ess_vals[2]:>7.0f} {ess_vals[3]:>7.0f} "
              f"{rhat_vals[0]:>7.3f} {rhat_vals[1]:>7.3f}")

    print("=" * 85)


# =====================================================================
#  Also run V1 MCMC on same data for side-by-side comparison
# =====================================================================
def run_v1_mcmc_diagnostic(fc_local, y_loc, H_loc, sig_y_vec,
                            sig_x_loc, n_samples, burn_in,
                            step_size, adapt, adapt_interval,
                            target_acc, thin, obs_op_name, seed,
                            block_cells_in_loc,
                            kernel='gibbs_mh', pcn_beta=0.3,
                            hmc_leapfrog_steps=10):
    """Run V1-style MCMC (all obs, all observed-block cells) and
    return diagnostics chain.
    Supports kernels: gibbs_mh, pcn, mala, hmc.
    """

    rng = np.random.default_rng(seed)

    if obs_op_name == 'arctan':
        def obs_operator(z, H, _):
            return np.arctan(H @ z)
    else:
        obs_operator = None

    Nf = fc_local.shape[1]
    dim = fc_local.shape[0]

    nz_mask = sig_x_loc > 0
    n_nz = int(nz_mask.sum())
    if n_nz == 0:
        return None

    inv_sig_x_nz = 1.0 / sig_x_loc[nz_mask]
    sig_x_nz = sig_x_loc[nz_mask]
    sig_x_sq_nz = sig_x_nz ** 2
    prop_scale = step_size * sig_x_nz
    beta = float(pcn_beta)
    hmc_L = int(hmc_leapfrog_steps)

    def _log_lik(z):
        if obs_operator is not None:
            y_pred = obs_operator(z, H_loc, None)
        else:
            y_pred = H_loc @ z
        residual = y_loc - y_pred
        return -0.5 * np.sum((residual / sig_y_vec) ** 2)

    def _log_trans(z, m_i):
        diff = z[nz_mask] - m_i[nz_mask]
        return -0.5 * np.sum((diff * inv_sig_x_nz) ** 2)

    def _log_trans_all(z):
        diff = z[nz_mask, None] - fc_local[nz_mask, :]
        return -0.5 * np.sum((diff * inv_sig_x_nz[:, None]) ** 2,
                             axis=0)

    def _grad_log_target(z, m_i):
        Hz = H_loc @ z
        if obs_op_name == 'arctan':
            pred = np.arctan(Hz)
            dpred = 1.0 / (1.0 + Hz ** 2)
        else:
            pred = Hz
            dpred = np.ones_like(Hz)
        residual = y_loc - pred
        grad = np.asarray(
            H_loc.T @ (dpred * residual / sig_y_vec ** 2)
        ).ravel()
        grad[nz_mask] -= (
            z[nz_mask] - m_i[nz_mask]) / sig_x_sq_nz
        return grad

    i_curr = rng.integers(Nf)
    z_curr = fc_local[:, i_curr].copy()
    log_lik = _log_lik(z_curr)
    log_trans = _log_trans(z_curr, fc_local[:, i_curr])

    total_iters = burn_in + n_samples * thin
    n_track = len(block_cells_in_loc)
    chain = np.zeros((total_iters, n_track))
    loglik_trace = np.zeros(total_iters)
    accept_trace = np.zeros(total_iters, dtype=bool)
    step_trace = np.zeros(total_iters)

    n_accept = 0
    _log_beta = np.log(max(beta, 1e-4))  # Robbins-Monro state
    for s in range(total_iters):
        # Gibbs step (common)
        lw_raw = _log_trans_all(z_curr)
        lw = lw_raw - lw_raw.max()
        weights = np.exp(lw)
        wsum = weights.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            weights[:] = 1.0 / Nf
        else:
            weights /= wsum
        i_curr = rng.choice(Nf, p=weights)
        log_trans = lw_raw[i_curr]
        m_i = fc_local[:, i_curr]

        accepted = False

        if kernel == 'gibbs_mh':
            z_prop = z_curr.copy()
            z_prop[nz_mask] += prop_scale * rng.standard_normal(n_nz)
            log_lik_prop = _log_lik(z_prop)
            log_trans_prop = _log_trans(z_prop, m_i)
            log_alpha = ((log_lik_prop + log_trans_prop)
                         - (log_lik + log_trans))
            if np.log(rng.random()) < log_alpha:
                z_curr = z_prop
                log_lik = log_lik_prop
                log_trans = log_trans_prop
                n_accept += 1
                accepted = True

        elif kernel == 'pcn':
            rho = np.sqrt(1.0 - beta ** 2)
            z_prop = z_curr.copy()
            z_prop[nz_mask] = (
                m_i[nz_mask]
                + rho * (z_curr[nz_mask] - m_i[nz_mask])
                + beta * sig_x_nz * rng.standard_normal(n_nz))
            log_lik_prop = _log_lik(z_prop)
            log_alpha = log_lik_prop - log_lik
            if np.log(rng.random()) < log_alpha:
                z_curr = z_prop
                log_lik = log_lik_prop
                log_trans = _log_trans(z_curr, m_i)
                n_accept += 1
                accepted = True

        elif kernel == 'mala':
            grad = _grad_log_target(z_curr, m_i)
            tau = step_size
            drift_nz = (tau ** 2 / 2) * sig_x_sq_nz * grad[nz_mask]
            z_prop = z_curr.copy()
            z_prop[nz_mask] += (
                drift_nz
                + tau * sig_x_nz * rng.standard_normal(n_nz))
            if not np.all(np.isfinite(z_prop)):
                pass
            else:
                log_lik_prop = _log_lik(z_prop)
                log_trans_prop = _log_trans(z_prop, m_i)
                grad_prop = _grad_log_target(z_prop, m_i)
                drift_rev_nz = (
                    (tau ** 2 / 2) * sig_x_sq_nz
                    * grad_prop[nz_mask])
                diff_fwd = (z_prop[nz_mask] - z_curr[nz_mask]
                            - drift_nz) / (tau * sig_x_nz)
                log_q_fwd = -0.5 * np.sum(diff_fwd ** 2)
                diff_rev = (z_curr[nz_mask] - z_prop[nz_mask]
                            - drift_rev_nz) / (tau * sig_x_nz)
                log_q_rev = -0.5 * np.sum(diff_rev ** 2)
                log_alpha = (
                    (log_lik_prop + log_trans_prop + log_q_rev)
                    - (log_lik + log_trans + log_q_fwd))
                if np.log(rng.random()) < log_alpha:
                    z_curr = z_prop
                    log_lik = log_lik_prop
                    log_trans = log_trans_prop
                    n_accept += 1
                    accepted = True

        elif kernel == 'hmc':
            p_nz = rng.standard_normal(n_nz) * inv_sig_x_nz
            KE_old = 0.5 * np.sum(p_nz ** 2 * sig_x_sq_nz)
            q = z_curr.copy()
            eps = step_size
            grad_nz = _grad_log_target(q, m_i)[nz_mask]
            p_nz += (eps / 2) * grad_nz
            diverged = False
            for ll in range(hmc_L):
                q[nz_mask] += eps * sig_x_sq_nz * p_nz
                if not np.all(np.isfinite(q[nz_mask])):
                    diverged = True
                    break
                grad_nz = _grad_log_target(q, m_i)[nz_mask]
                if ll < hmc_L - 1:
                    p_nz += eps * grad_nz
                else:
                    p_nz += (eps / 2) * grad_nz
            if diverged or not np.all(np.isfinite(p_nz)):
                pass
            else:
                KE_new = 0.5 * np.sum(p_nz ** 2 * sig_x_sq_nz)
                log_lik_prop = _log_lik(q)
                log_trans_prop = _log_trans(q, m_i)
                H_old = -log_lik - log_trans + KE_old
                H_new = -log_lik_prop - log_trans_prop + KE_new
                log_alpha = -(H_new - H_old)
                if np.log(rng.random()) < log_alpha:
                    z_curr = q
                    log_lik = log_lik_prop
                    log_trans = log_trans_prop
                    n_accept += 1
                    accepted = True

        # Adapt: Robbins-Monro for pCN, block for others
        if adapt and s > 0:
            if kernel == 'pcn':
                _rm_g = 0.5 / (1.0 + s) ** 0.6
                _log_beta += _rm_g * (float(accepted) - target_acc)
                _log_beta = max(min(_log_beta, np.log(0.999)), np.log(1e-4))
                beta = np.exp(_log_beta)
            elif s < burn_in and s % adapt_interval == 0:
                acc_now = n_accept / (s + 1)
                if acc_now < target_acc * 0.6:
                    step_size *= 0.8
                elif acc_now > target_acc * 1.6:
                    step_size *= 1.2
                step_size = np.clip(step_size, 1e-6, 10.0)
                if kernel == 'gibbs_mh':
                    prop_scale = step_size * sig_x_nz

        chain[s, :] = z_curr[block_cells_in_loc]
        loglik_trace[s] = log_lik
        accept_trace[s] = accepted
        step_trace[s] = step_size if kernel != 'pcn' else beta

    return {
        'chain': chain,
        'loglik_trace': loglik_trace,
        'accept_trace': accept_trace,
        'step_trace': step_trace,
        'acc_rate': n_accept / total_iters,
        'burn_in': burn_in,
        'n_samples': n_samples,
        'block_in_halo': block_cells_in_loc,
        'n_obs': len(y_loc),
        'dim_halo': dim,
    }


# =====================================================================
#  Main
# =====================================================================
def main():
    config_file = (sys.argv[1] if len(sys.argv) > 1
                   else 'example_input_mlswe_nldata_V2_twin.yml')
    with open(config_file) as f:
        params = yaml.safe_load(f)

    print("=" * 60)
    print("  MCMC Convergence Diagnostics")
    print("=" * 60)

    outdir = params.get('lsmcmc_dir', './output_lsmcmc_nldata_V2')
    diag_dir = os.path.join(outdir, 'mcmc_diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    nx = params['dgx']
    ny = params['dgy']
    nc = nx * ny
    lon = np.linspace(params['lon_min'], params['lon_max'], nx)
    lat = np.linspace(params['lat_min'], params['lat_max'], ny)

    # Bathymetry
    H_b = run_mlswe_lsmcmc.load_bathymetry(params, ny, nx, lon, lat)

    # Boundary conditions
    bc_file = params.get('bc_file', './data/hycom_bc.nc')
    from mlswe.boundary_handler import MLBoundaryHandler
    bc_handler = MLBoundaryHandler(
        nc_path=bc_file,
        model_lon=lon, model_lat=lat, H_b=H_b,
        H_mean=params.get('H_mean', 4000.0),
        H_rest=params.get('H_rest', [100.0, 400.0, 3500.0]),
        T_rest=params.get('T_rest', [298.15, 283.15, 275.15]),
        alpha_h=params.get('alpha_h', [0.6, 0.3, 0.1]),
        beta_vel=params.get('beta_vel', [1.0, 1.0, 1.0]),
        n_ghost=params.get('bc_n_ghost', 2),
        sponge_width=params.get('sponge_width', 8),
        sponge_timescale=params.get('sponge_timescale', 3600.0),
    )

    obs_file = params.get('obs_file',
                          './data/obs_2024aug/swe_drifter_obs.nc')
    with Dataset(obs_file, 'r') as nc_ds:
        obs_times = np.asarray(nc_ds.variables['obs_times'][:])
    tstart = obs_times[0]

    _load_sst_ssh_refs(params, bc_file, lon, lat, ny, nx, obs_times)

    lat_centre = 0.5 * (params['lat_min'] + params['lat_max'])
    dx_m = np.deg2rad(abs(lon[1] - lon[0])) * 6.371e6 * np.cos(
        np.deg2rad(lat_centre))
    dy_m = np.deg2rad(abs(lat[1] - lat[0])) * 6.371e6

    h0, u0, v0, T0 = run_mlswe_lsmcmc.init_from_bc_handler(
        bc_handler, H_b, tstart,
        H_rest=params.get('H_rest', [100.0, 400.0, 3500.0]),
        T_rest=params.get('T_rest', [298.15, 283.15, 275.15]),
        beta_vel=params.get('beta_vel', [1.0, 0.3, 0.05]),
        dx=dx_m, dy=dy_m,
        geostrophic_blend=params.get('geostrophic_blend', 0.5),
    )
    params['ic_h0'] = h0
    params['ic_u0'] = u0
    params['ic_v0'] = v0
    params['ic_T0'] = T0

    if bc_handler is not None:
        bc_handler.release_full_fields()

    # ---- Load truth ----
    truth_file = os.path.join(outdir, 'truth_trajectory.nc')
    if not os.path.exists(truth_file):
        print("Generating truth trajectory...")
        truth_states, synth_obs_file = generate_truth_and_obs(
            params, H_b, bc_handler, tstart,
            obs_file, outdir, rng_seed=42)
        _save_truth(outdir, truth_states, H_b, ny, nx)
    else:
        print(f"Loading truth from {truth_file}")
        with Dataset(truth_file, 'r') as ds:
            truth_arr = np.array(ds.variables['truth'][:])
        truth_states = truth_arr.reshape(
            truth_arr.shape[0], -1).astype(np.float64)

    # ---- Load synthetic obs ----
    synth_obs_file = os.path.join(outdir, 'synthetic_arctan_obs.nc')
    with Dataset(synth_obs_file, 'r') as ds:
        yobs = np.array(ds.variables['yobs_all'][:])
        yind = np.array(ds.variables['yobs_ind_all'][:])
        sig_y_all = np.array(ds.variables['sig_y_all'][:])

    # ---- Set up filter (just for partition & precompute) ----
    f_2d = coriolis_array(params['lat_min'], params['lat_max'], ny, nx)
    model_kw = dict(
        rho=[float(r) for r in params.get('rho', [1023, 1026, 1028])],
        dx=dx_m, dy=dy_m,
        dt=float(params['dt']),
        f0=f_2d,
        g=params.get('g', 9.81),
        H_b=H_b,
        H_mean=float(params.get('H_mean', 4000.0)),
        H_rest=[float(x) for x in params.get('H_rest',
                                               [100, 400, 3500])],
        bottom_drag=params.get('bottom_drag', 1e-6),
        diff_coeff=params.get('diff_coeff', 500.0),
        diff_order=params.get('diff_order', 1),
        tracer_diff=params.get('tracer_diff', 100.0),
        bc_handler=bc_handler,
        tstart=tstart,
        precision=params.get('precision', 'double'),
        sst_nudging_rate=params.get('sst_nudging_rate', 0.0),
        sst_nudging_ref=params.get('sst_nudging_ref', None),
        sst_nudging_ref_times=params.get('sst_nudging_ref_times', None),
        ssh_relax_rate=params.get('ssh_relax_rate', 0.0),
        ssh_relax_ref=params.get('ssh_relax_ref', None),
        ssh_relax_ref_times=params.get('ssh_relax_ref_times', None),
        sst_flux_type=params.get('sst_flux_type', None),
        sst_alpha=float(params.get('sst_alpha', 15.0)),
        sst_h_mix=float(params.get('sst_h_mix', 50.0)),
        sst_T_air=params.get('sst_T_air', None),
        sst_T_air_times=params.get('sst_T_air_times', None),
        ssh_relax_interior_floor=float(
            params.get('ssh_relax_interior_floor', 0.1)),
        shallow_drag_depth=float(
            params.get('shallow_drag_depth', 500.0)),
        shallow_drag_coeff=float(
            params.get('shallow_drag_coeff', 5.0e-4)),
    )

    # Create ensemble (Nf identical members)
    Nf = params['nforecast']
    forecast = np.zeros((12 * nc, Nf))
    for j in range(Nf):
        mdl = MLSWE(
            [hk.copy() for hk in h0],
            [uk.copy() for uk in u0],
            [vk.copy() for vk in v0],
            T0=[Tk.copy() for Tk in T0],
            **model_kw)
        mdl.timesteps = 1
        forecast[:, j] = mdl.state_flat.copy()

    # ---- Advance ensemble one cycle (with noise) ----
    t_freq = int(params.get('assim_timesteps',
                             params.get('t_freq', 48)))
    sig_x_uv = params.get('sig_x_uv', params.get('sig_x', 0.15))
    sig_x_sst = params.get('sig_x_sst', 1.0)
    sig_x_ssh = params.get('sig_x_ssh', sig_x_uv)
    assimilate_fields = str(params.get('assimilate_fields', 'uv_sst'))
    assim_uv = 'uv' in assimilate_fields
    assim_ssh = 'ssh' in assimilate_fields
    assim_sst = 'sst' in assimilate_fields

    sig_per_field = []
    for k in range(3):
        if k == 0:
            sig_h0 = sig_x_ssh if assim_ssh else 0.0
            sig_u0 = sig_x_uv if assim_uv else 0.0
            sig_t0 = sig_x_sst if assim_sst else 0.0
            sig_per_field.extend([sig_h0, sig_u0, sig_u0, sig_t0])
        else:
            sig_per_field.extend([0.0, 0.0, 0.0, 0.0])
    sig_x_vec = np.repeat(sig_per_field, nc)

    print("Advancing ensemble one cycle...")
    np.random.seed(42)
    models = []
    for j in range(Nf):
        mdl_j = MLSWE(
            [hk.copy() for hk in h0],
            [uk.copy() for uk in u0],
            [vk.copy() for vk in v0],
            T0=[Tk.copy() for Tk in T0],
            **model_kw)
        mdl_j.timesteps = 1
        for _ in range(t_freq):
            mdl_j._timestep()
        mdl_j.state_flat[:] += np.random.normal(scale=sig_x_vec)
        forecast[:, j] = mdl_j.state_flat.copy()
        models.append(mdl_j)

    print(f"  Ensemble spread: h_std={np.std(forecast[:nc], axis=1).mean():.4f}, "
          f"u_std={np.std(forecast[nc:2*nc], axis=1).mean():.4f}")

    # ---- Get first cycle observations ----
    cycle = 0
    y = yobs[cycle]
    ind = yind[cycle]
    sig_y_cycle = sig_y_all[cycle]
    valid = (ind >= 0) & np.isfinite(y)
    y_valid = y[valid]
    ind_valid = ind[valid].astype(int)
    sig_y_valid = sig_y_cycle[valid]

    obs_ind_ml = ind_valid  # layer0 identity
    obs_cell_indices = ind_valid % nc

    print(f"  Cycle 0: {len(y_valid)} obs, obs range [{y_valid.min():.3f}, {y_valid.max():.3f}]")

    # ---- V2 block localization setup ----
    num_subdomains = params.get('num_subdomains', 80)
    block_list, labels, nblocks, nby, nbx, bh, bw = \
        partition_domain(ny, nx, num_subdomains)

    r_loc = float(params.get('r_loc', 8.0))
    gc_noise_inflate = bool(params.get('gc_noise_inflate', True))

    # Precompute per-block info (replicating filter._precompute_block_localization)
    obs_row, obs_col = np.unravel_index(obs_cell_indices, (ny, nx))
    unique_blocks = np.unique(labels)
    grid_rows, grid_cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    nf_loc = 4  # fields per layer

    def gaspari_cohn(dist, c):
        r = np.abs(dist) / c
        rho = np.zeros_like(r)
        m1 = r <= 1.0
        m2 = (~m1) & (r <= 2.0)
        r1 = r[m1]
        rho[m1] = (1.0 - (5./3.)*r1**2 + (5./8.)*r1**3
                   + 0.5*r1**4 - 0.25*r1**5)
        r2 = r[m2]
        rho[m2] = (4.0 - 5.0*r2 + (5./3.)*r2**2
                   + (5./8.)*r2**3 - 0.5*r2**4
                   + (1./12.)*r2**5 - 2./(3.*r2))
        return rho

    block_infos = []
    for block_id in unique_blocks:
        block_mask = (labels == block_id)
        block_ij = np.argwhere(block_mask)
        cy = block_ij[:, 0].mean()
        cx = block_ij[:, 1].mean()

        dy_o = np.abs(obs_row - cy)
        dx_o = np.abs(obs_col - cx)
        dist_obs = np.sqrt(dy_o**2 + dx_o**2)

        nearby = dist_obs < r_loc
        if not np.any(nearby):
            block_infos.append(None)
            continue

        nearby_idx = np.where(nearby)[0]
        obs_global = obs_ind_ml[nearby_idx]

        dy_g = np.abs(grid_rows - cy)
        dx_g = np.abs(grid_cols - cx)
        halo_mask = np.sqrt(dy_g**2 + dx_g**2) < r_loc
        halo_ij = np.argwhere(halo_mask)
        halo_flat_0 = np.ravel_multi_index(
            (halo_ij[:, 0], halo_ij[:, 1]), (ny, nx))
        block_flat_0 = np.ravel_multi_index(
            (block_ij[:, 0], block_ij[:, 1]), (ny, nx))
        halo_flat_0 = np.unique(np.concatenate(
            [halo_flat_0, block_flat_0]))

        halo_cells = np.sort(np.concatenate(
            [halo_flat_0 + f * nc for f in range(nf_loc)]))
        block_cells = np.sort(np.concatenate(
            [block_flat_0 + f * nc for f in range(nf_loc)]))
        block_in_halo = np.searchsorted(halo_cells, block_cells)

        H_loc = build_H_loc_from_global(
            obs_global, halo_cells, drop_unmapped=True)

        dist_local = dist_obs[nearby_idx][:H_loc.shape[0]]
        gc_w = gaspari_cohn(dist_local, r_loc)
        gc_w = np.maximum(gc_w, 1e-6)

        sigma_y_base = sig_y_valid[nearby_idx][:H_loc.shape[0]]
        if gc_noise_inflate:
            sigma_y_local = sigma_y_base / np.sqrt(gc_w)
        else:
            sigma_y_local = sigma_y_base.copy()

        block_infos.append({
            'block_id': block_id,
            'nearby_idx': nearby_idx,
            'halo_cells': halo_cells,
            'block_cells': block_cells,
            'block_in_halo': block_in_halo,
            'H_loc': H_loc,
            'sigma_y_local': sigma_y_local,
            'n_obs': H_loc.shape[0],
        })

    active_infos = [b for b in block_infos if b is not None]
    print(f"  {len(active_infos)}/{len(unique_blocks)} active blocks")

    # ---- Also set up V1 localization (all observed blocks) ----
    # V1 uses all obs with block-partition (no halo, no GC)
    from mlswe.lsmcmc_nl_V1 import partition_domain as v1_partition
    v1_nsub = params.get('num_subdomains', 50)
    # Use V1 config's num_subdomains
    try:
        with open('example_input_mlswe_nldata_V1_twin.yml') as fv1:
            v1_params = yaml.safe_load(fv1)
        v1_nsub = v1_params.get('num_subdomains', 50)
    except:
        v1_nsub = 50
    v1_bl, v1_labels, v1_nblocks, *_ = v1_partition(ny, nx, v1_nsub)

    # V1 observed blocks
    obs_r, obs_c = np.unravel_index(obs_cell_indices, (ny, nx))
    obs_blocks_v1 = np.unique(v1_labels[obs_r, obs_c])
    v1_loc_mask = np.isin(v1_labels, obs_blocks_v1)
    v1_block_flat_0 = np.where(v1_loc_mask.ravel())[0]
    v1_loc_cells = np.sort(np.concatenate(
        [v1_block_flat_0 + f * nc for f in range(nf_loc)]))
    v1_H_loc = build_H_loc_from_global(obs_ind_ml, v1_loc_cells)
    # V1 uses raw sig_y
    v1_sig_y_vec = sig_y_valid
    v1_sig_x_loc = sig_x_vec[v1_loc_cells]
    v1_fc_local = forecast[v1_loc_cells, :]

    print(f"  V1 localization: {len(v1_loc_cells)} state dims, "
          f"{v1_H_loc.shape[0]} obs, {len(obs_blocks_v1)} blocks")

    # ---- Select representative blocks for diagnosis ----
    # Pick blocks with different obs counts: low, medium, high
    obs_counts = [b['n_obs'] for b in active_infos]
    sorted_idx = np.argsort(obs_counts)

    # Pick ~5 blocks spanning the range
    n_diag = min(5, len(active_infos))
    pick_idx = np.linspace(0, len(active_infos)-1, n_diag, dtype=int)
    diag_blocks = [active_infos[sorted_idx[i]] for i in pick_idx]

    n_samples = params['mcmc_N']
    burn_in = params.get('burn_in', n_samples)
    step_size = float(params.get('mcmc_step_size', 0.05))
    adapt = bool(params.get('mcmc_adapt', True))
    adapt_interval = int(params.get('mcmc_adapt_interval', 25))
    target_acc = float(params.get('mcmc_target_acc', 0.234))
    thin = max(1, int(params.get('mcmc_thin', 1)))
    pcn_beta = float(params.get('pcn_beta', 0.3))
    hmc_leapfrog_steps = int(params.get('hmc_leapfrog_steps', 10))

    # Kernels to diagnose — run all 4
    kernels_to_test = ['pcn']

    print(f"\nMCMC config: N={n_samples}, burn_in={burn_in}, "
          f"step_size={step_size}, adapt_interval={adapt_interval}")
    print(f"  pcn_beta={pcn_beta}, hmc_leapfrog_steps={hmc_leapfrog_steps}")
    print(f"Running diagnostics on {n_diag} blocks "
          f"(obs range: {min(obs_counts)}–{max(obs_counts)})...")
    print(f"Kernels: {kernels_to_test}\n")

    # Truth for cycle 1 (after forecast)
    truth_cycle1 = truth_states[1]

    # V1 block setup (computed once, used for all kernels)
    first_block_cells = diag_blocks[0]['block_cells']
    v1_block_in_loc = np.searchsorted(v1_loc_cells, first_block_cells)
    valid_v1 = v1_block_in_loc < len(v1_loc_cells)
    v1_block_in_loc = v1_block_in_loc[valid_v1]

    # ---- Run diagnostic MCMC for each kernel ----
    all_diagnostics = {}
    for kern in kernels_to_test:
        print(f"\n{'='*50}")
        print(f"  Kernel: {kern}")
        print(f"{'='*50}")

        kern_dir = os.path.join(diag_dir, kern)
        os.makedirs(kern_dir, exist_ok=True)

        diagnostics_kern = {}

        # V2 blocks
        for binfo in diag_blocks:
            blk_id = binfo['block_id']
            fc_halo = forecast[binfo['halo_cells'], :]
            y_local = y_valid[binfo['nearby_idx']][:binfo['H_loc'].shape[0]]
            sig_x_halo = sig_x_vec[binfo['halo_cells']]

            print(f"  V2 Block {blk_id}: {binfo['n_obs']} obs, "
                  f"halo_dim={len(binfo['halo_cells'])}, "
                  f"block_dim={len(binfo['block_cells'])}")

            t0 = time.time()
            diag = _diag_mcmc_worker(
                fc_halo, y_local, binfo['H_loc'],
                binfo['sigma_y_local'], sig_x_halo,
                binfo['block_in_halo'],
                n_samples, burn_in, step_size,
                adapt, adapt_interval, target_acc, thin,
                'arctan', seed=42,
                kernel=kern, pcn_beta=pcn_beta,
                hmc_leapfrog_steps=hmc_leapfrog_steps)
            elapsed = time.time() - t0

            if diag is None:
                print(f"    -> Skipped (no active dimensions)")
                continue

            print(f"    -> acc_rate={diag['acc_rate']:.3f}, "
                  f"time={elapsed:.1f}s")

            label = f'v2_blk{blk_id}_{kern}'
            diagnostics_kern[label] = diag

            truth_block = truth_cycle1[binfo['block_cells']]

            plot_traceplots(diag, truth_block, kern_dir,
                            block_label=f'v2_blk{blk_id}')
            plot_autocorrelation(diag, kern_dir,
                                 block_label=f'v2_blk{blk_id}')
            plot_running_mean(diag, truth_block, kern_dir,
                              block_label=f'v2_blk{blk_id}')
            plot_step_size_and_acceptance(diag, kern_dir,
                                          block_label=f'v2_blk{blk_id}')

        # V1 comparison
        if len(v1_block_in_loc) > 0:
            print(f"\n  V1 MCMC ({kern}): {v1_H_loc.shape[0]} obs, "
                  f"loc_dim={len(v1_loc_cells)} ...")
            try:
                v1_N = v1_params.get('mcmc_N', 2000)
                v1_burn = v1_params.get('burn_in', 1000)
                v1_step = float(v1_params.get('mcmc_step_size', 0.5))
                v1_adapt_int = int(
                    v1_params.get('mcmc_adapt_interval', 50))
            except Exception:
                v1_N = 2000
                v1_burn = 1000
                v1_step = 0.5
                v1_adapt_int = 50

            t0 = time.time()
            v1_diag = run_v1_mcmc_diagnostic(
                v1_fc_local, y_valid, v1_H_loc, v1_sig_y_vec,
                v1_sig_x_loc, v1_N, v1_burn, v1_step,
                adapt, v1_adapt_int, target_acc, thin,
                'arctan', seed=42,
                block_cells_in_loc=v1_block_in_loc,
                kernel=kern, pcn_beta=pcn_beta,
                hmc_leapfrog_steps=hmc_leapfrog_steps)
            v1_elapsed = time.time() - t0
            print(f"    -> V1 acc_rate={v1_diag['acc_rate']:.3f}, "
                  f"time={v1_elapsed:.1f}s")

            diagnostics_kern[f'v1_global_{kern}'] = v1_diag

            truth_v1_block = truth_cycle1[first_block_cells[valid_v1]]
            plot_traceplots(v1_diag, truth_v1_block, kern_dir,
                            block_label='v1_global')
            plot_autocorrelation(v1_diag, kern_dir,
                                 block_label='v1_global')
            plot_running_mean(v1_diag, truth_v1_block, kern_dir,
                              block_label='v1_global')
            plot_step_size_and_acceptance(v1_diag, kern_dir,
                                          block_label='v1_global')

        all_diagnostics[kern] = diagnostics_kern

    # ---- Summary table per kernel ----
    for kern, diags in all_diagnostics.items():
        print(f"\n{'='*50}")
        print(f"  Summary: kernel={kern}")
        print(f"{'='*50}")
        print_summary_table(diags, None)

    print(f"\nAll plots saved to: {diag_dir}/")
    print("Done.")


if __name__ == '__main__':
    main()

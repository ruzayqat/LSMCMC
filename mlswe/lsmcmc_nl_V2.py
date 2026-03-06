#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lsmcmc_nl_V2.py — Nonlinear LSMCMC (V2) with halo-based localization
======================================================================

Samples from the SMCMC target distribution

    π̂(z_t, i) ∝ g(z_t, y_t) · f(Z_{t-1}^{(i)}, z_t) · p(i)

using per-block MCMC (Gibbs-within-MH or joint RW-MH), rather than
the exact-from-Gaussian sampler in ``lsmcmc_V2.py``.

V2 strategy: for each partition block, expand a halo of radius
``r_loc`` grid points, collect nearby observations, apply
Gaspari-Cohn tapering to observation noise, and run MCMC on the
**small local state** (halo cells only).  After MCMC, retain only
the block cells (discard the halo).

Because each block's MCMC operates on a much smaller state
(e.g. ~800 vs ~20 000 for V1), this approach is dramatically
cheaper.  Blocks are processed in parallel via ThreadPoolExecutor.

Two MCMC kernels are implemented:
  1. **gibbs_mh** — exact Gibbs step for i, then RW-MH for z
  2. **joint_mh** — propose (z*, i*) jointly

Adaptive step-size tuning during burn-in targets ~23.4%
acceptance rate.
"""

import os
import sys
import time
import warnings
import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
from netCDF4 import Dataset

# --------------- imports from the project ---------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from loc_smcmc_swe_exact_from_Gauss import (
        partition_domain,
        get_divisors,
        gaussian_block_means,
    )
except ImportError:
    raise ImportError(
        "Cannot import partition_domain / gaussian_block_means. "
        "Ensure loc_smcmc_swe_exact_from_Gauss.py is on PYTHONPATH.")

from mlswe.model import MLSWE, coriolis_array, lonlat_to_dxdy


# ===================================================================
#  Utility functions (same as lsmcmc_V2.py)
# ===================================================================
def build_H_loc_from_global(obs_ind_global, local_state_indices,
                             drop_unmapped=False):
    """Map global observation indices to a local state vector.

    Returns a sparse CSR matrix H_loc of shape (d_y_eff, d_p) that
    maps the local state vector to the local observations.
    """
    idx_map = {g: p for p, g in enumerate(local_state_indices)}
    d_p = len(local_state_indices)
    rows, cols = [], []
    out_row = 0
    for g in obs_ind_global:
        g = int(g)
        if g in idx_map:
            rows.append(out_row)
            cols.append(idx_map[g])
            out_row += 1
        elif not drop_unmapped:
            out_row += 1
    d_y_eff = out_row
    data = np.ones(len(rows), dtype=float)
    if len(rows) == 0:
        return sp.csr_matrix((d_y_eff, d_p))
    return sp.coo_matrix(
        (data, (rows, cols)), shape=(d_y_eff, d_p)).tocsr()


def rescaled_block_means(samples, M, mu):
    """Reduce N samples to M via grouped means, rescaled to preserve
    variance."""
    means, groups = gaussian_block_means(samples, M)
    Mi = np.array([len(g) for g in groups])
    scale = np.sqrt(Mi)[None, :]
    mu = mu.reshape(-1, 1)
    return mu + scale * (means - mu)


# ===================================================================
#  Multiprocessing workers  (identical to lsmcmc.py)
# ===================================================================
_mp_mlswe_kwargs = None
_mp_worker_model = None


def _mp_init_worker():
    global _mp_worker_model
    _mp_worker_model = MLSWE(**_mp_mlswe_kwargs)


def _mp_advance(args):
    state_flat, t_val, nsteps = args
    mdl = _mp_worker_model
    mdl.state_flat = state_flat
    mdl.t = t_val
    for _ in range(nsteps):
        mdl._timestep()
        if not np.all(np.isfinite(mdl.state_flat)):
            return state_flat.copy(), t_val, True
    return mdl.state_flat.copy(), float(mdl.t), False


# ===================================================================
#  Standalone block MCMC worker  (runs in mp.Pool — bypasses GIL)
# ===================================================================
def _block_mcmc_worker(args):
    """Run MCMC on a single block's halo state.

    This is a standalone (picklable) function so it can run in a
    separate process via multiprocessing.Pool, avoiding the GIL
    bottleneck that makes ThreadPoolExecutor useless for
    compute-bound MCMC loops.

    Parameters (packed in tuple)
    ----------------------------
    block_idx, fc_halo, y_local, H_loc_tuple, sigma_y_loc,
    sig_x_halo, block_cells, block_in_halo, prior_sprd,
    n_samples, burn_in, step_size, adapt, adapt_interval,
    target_acc, thin, kernel, obs_op_name, rtps_alpha_val, seed
    """
    (block_idx, fc_halo, y_local, H_loc_tuple, sigma_y_loc,
     sig_x_halo, block_cells, block_in_halo, prior_sprd,
     n_samples, burn_in, step_size, adapt, adapt_interval,
     target_acc, thin, kernel, obs_op_name,
     rtps_alpha_val, Nf_out, shared_perm_seed,
     pcn_beta, hmc_leapfrog_steps, seed) = args

    from scipy import sparse as sp

    rng = np.random.default_rng(seed)

    # Reconstruct sparse H_loc
    data, indices, indptr, shape = H_loc_tuple
    H_loc = sp.csr_matrix((data, indices, indptr), shape=shape)

    # Obs operator
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
        mu = np.mean(fc_halo, axis=1)
        samples_block = np.tile(mu, (n_samples, 1)).T
        # Extract block cells
        samples_block = samples_block[block_in_halo, :]
        block_mean = np.mean(samples_block, axis=1)
        anal_block = rescaled_block_means(
            samples_block, Nf_out, block_mean)
        return {
            'block_cells': block_cells,
            'anal_block': anal_block,
            'mu_block': block_mean,
            'acc_rate': 0.0,
        }

    inv_sig_x_nz = 1.0 / sig_x_halo[nz_mask]
    sig_x_nz = sig_x_halo[nz_mask]
    sig_x_sq_nz = sig_x_nz ** 2
    prop_scale = step_size * sig_x_nz
    beta = float(pcn_beta)
    hmc_L = int(hmc_leapfrog_steps)

    # Log-likelihood
    def _log_lik(z):
        if obs_operator is not None:
            y_pred = obs_operator(z, H_loc, None)
        else:
            y_pred = H_loc @ z
        residual = y_local - y_pred
        return -0.5 * np.sum((residual / sigma_y_loc) ** 2)

    # Log-transition
    def _log_trans(z, m_i):
        diff = z[nz_mask] - m_i[nz_mask]
        return -0.5 * np.sum((diff * inv_sig_x_nz) ** 2)

    def _log_trans_all(z):
        diff = z[nz_mask, None] - fc_halo[nz_mask, :]
        return -0.5 * np.sum(
            (diff * inv_sig_x_nz[:, None]) ** 2, axis=0)

    # Gradient of log target (for MALA / HMC)
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

    # Initialise
    i_curr = rng.integers(Nf)
    z_curr = fc_halo[:, i_curr].copy()
    log_lik = _log_lik(z_curr)
    log_trans = _log_trans(z_curr, fc_halo[:, i_curr])

    total_iters = burn_in + n_samples * thin
    samples = np.zeros((dim, n_samples))
    n_accept = 0
    s_out = 0
    _log_beta = np.log(max(beta, 1e-4))  # Robbins-Monro state (pCN)
    _log_step = np.log(max(step_size, 1e-6))  # Robbins-Monro state (HMC/MALA)

    for s in range(total_iters):
        # --- Gibbs step for i (vectorized, common) ---
        log_weights_raw = _log_trans_all(z_curr)
        lw = log_weights_raw - log_weights_raw.max()
        weights = np.exp(lw)
        wsum = weights.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            weights[:] = 1.0 / Nf
        else:
            weights /= wsum
        i_curr = rng.choice(Nf, p=weights)
        log_trans = log_weights_raw[i_curr]
        m_i = fc_halo[:, i_curr]

        # --- Proposal step for z (kernel-dependent) ---
        _n_acc_before = n_accept
        if kernel == 'gibbs_mh':
            z_prop = z_curr.copy()
            z_prop[nz_mask] += prop_scale * rng.standard_normal(
                n_nz)
            log_lik_prop = _log_lik(z_prop)
            log_trans_prop = _log_trans(z_prop, m_i)
            log_alpha = ((log_lik_prop + log_trans_prop)
                         - (log_lik + log_trans))
            if np.log(rng.random()) < log_alpha:
                z_curr = z_prop
                log_lik = log_lik_prop
                log_trans = log_trans_prop
                n_accept += 1

        elif kernel == 'pcn':
            rho = np.sqrt(1.0 - beta ** 2)
            z_prop = z_curr.copy()
            z_prop[nz_mask] = (
                m_i[nz_mask]
                + rho * (z_curr[nz_mask] - m_i[nz_mask])
                + beta * sig_x_nz * rng.standard_normal(n_nz))
            log_lik_prop = _log_lik(z_prop)
            # pCN: prior terms cancel — acceptance = lik ratio
            log_alpha = log_lik_prop - log_lik
            if np.log(rng.random()) < log_alpha:
                z_curr = z_prop
                log_lik = log_lik_prop
                log_trans = _log_trans(z_curr, m_i)
                n_accept += 1

        elif kernel == 'mala':
            grad = _grad_log_target(z_curr, m_i)
            tau = step_size
            drift_nz = (tau ** 2 / 2) * sig_x_sq_nz * grad[nz_mask]
            z_prop = z_curr.copy()
            z_prop[nz_mask] += (
                drift_nz
                + tau * sig_x_nz * rng.standard_normal(n_nz))
            if not np.all(np.isfinite(z_prop)):
                pass  # reject (leave z_curr unchanged)
            else:
                log_lik_prop = _log_lik(z_prop)
                log_trans_prop = _log_trans(z_prop, m_i)
                grad_prop = _grad_log_target(z_prop, m_i)
                drift_rev_nz = (
                    (tau ** 2 / 2) * sig_x_sq_nz
                    * grad_prop[nz_mask])
                # Forward q(z'|z)
                diff_fwd = (z_prop[nz_mask] - z_curr[nz_mask]
                            - drift_nz) / (tau * sig_x_nz)
                log_q_fwd = -0.5 * np.sum(diff_fwd ** 2)
                # Reverse q(z|z')
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

        elif kernel == 'hmc':
            # Momentum: p ~ N(0, M), M = diag(1/σ_x²)
            p_nz = rng.standard_normal(n_nz) * inv_sig_x_nz
            KE_old = 0.5 * np.sum(p_nz ** 2 * sig_x_sq_nz)
            q = z_curr.copy()
            eps = step_size
            # Leapfrog
            grad_nz = _grad_log_target(q, m_i)[nz_mask]
            p_nz += (eps / 2) * grad_nz  # half step (+ because grad of log π)
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
                pass  # reject
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

        # --- Adapt: Robbins-Monro for all kernels ---
        if adapt and s > 0:
            _accepted_this = (n_accept > _n_acc_before)
            if kernel == 'pcn':
                _rm_g = 0.5 / (1.0 + s) ** 0.6
                _log_beta += _rm_g * (float(_accepted_this) - target_acc)
                _log_beta = max(min(_log_beta, np.log(0.999)), np.log(1e-4))
                beta = np.exp(_log_beta)
            else:
                # Robbins-Monro for HMC / MALA / gibbs_mh
                _rm_g = 0.5 / (1.0 + s) ** 0.6
                _log_step += _rm_g * (float(_accepted_this) - target_acc)
                _log_step = max(min(_log_step, np.log(10.0)), np.log(1e-6))
                step_size = np.exp(_log_step)
                if kernel == 'gibbs_mh':
                    prop_scale = step_size * sig_x_nz

        if s >= burn_in and (s - burn_in) % thin == 0:
            samples[:, s_out] = z_curr
            s_out += 1
            if s_out >= n_samples:
                break

    acc_rate = n_accept / max(total_iters, 1)

    if not np.all(np.isfinite(samples)):
        return None

    # Retain only block cells
    samples_block = samples[block_in_halo, :]

    # Reduce n_samples → Nf via rescaled block means (preserves variance)
    block_mean = np.mean(samples_block, axis=1)
    # Use shared permutation seed for cross-block coherence
    np.random.seed(shared_perm_seed)
    anal_block = rescaled_block_means(
        samples_block, Nf_out, block_mean)
    mu_block = block_mean

    # RTPS inflation (only if enabled)
    if rtps_alpha_val > 0:
        amean = np.mean(
            anal_block, axis=1, keepdims=True)
        asprd = np.std(anal_block, axis=1)
        safe = np.maximum(asprd, 1e-30)
        infl = 1.0 + rtps_alpha_val * (
            prior_sprd - asprd) / safe
        infl = np.maximum(infl, 1.0)
        anal_block = amean + infl[:, None] * (
            anal_block - amean)

    return {
        'block_cells': block_cells,
        'anal_block': anal_block,
        'mu_block': mu_block,
        'acc_rate': acc_rate,
    }


# ===================================================================
#  Filter class
# ===================================================================
class NL_SMCMC_MLSWE_Filter_V2:
    """
    Nonlinear LSMCMC filter for the 3-layer primitive-equations model,
    using per-block MCMC with halo localization and Gaspari-Cohn
    tapering.

    Samples from
        π̂(z, i) ∝ g(z, y) · f(m_i, z) · p(i)
    independently for each partition block, where the local state
    is the halo around the block (radius ``r_loc`` grid points).

    Parameters
    ----------
    isim : int
        Simulation index.
    params : dict
        Configuration dictionary.
    obs_operator : callable or None
        Nonlinear observation operator.  Signature:
            y_pred = obs_operator(z_local, H_loc, obs_ind_local)
        If *None*, the linear operator ``H_loc @ z`` is used.
    """

    def __init__(self, isim, params, obs_operator=None):
        self.isim = isim
        self.params = params
        self.obs_operator = obs_operator

        # ---- Grid ----
        self.nx = params['dgx']
        self.ny = params['dgy']
        self.ncells = self.ny * self.nx
        self.nlayers = 3
        self.fields_per_layer = 4
        self.nfields = self.nlayers * self.fields_per_layer
        self.dimx = self.nfields * self.ncells

        # ---- Time stepping ----
        self.dt = params['dt']
        self.T = params['T']
        self.assim_timesteps = params.get(
            'assim_timesteps', params.get('t_freq', 48))
        self.nassim = self.T // self.assim_timesteps

        # ---- Ensemble / MCMC parameters ----
        self.nforecast = params['nforecast']
        self.mcmc_N = params['mcmc_N']
        self.burn_in = params.get('burn_in', self.mcmc_N)
        self.mcmc_iters = self.mcmc_N + self.burn_in

        # MCMC-specific tuning
        self.mcmc_step_size = float(
            params.get('mcmc_step_size', 0.5))
        self.mcmc_kernel = str(
            params.get('mcmc_kernel', 'gibbs_mh'))
        self.mcmc_adapt = bool(params.get('mcmc_adapt', True))
        self.mcmc_adapt_interval = int(
            params.get('mcmc_adapt_interval', 50))
        self.mcmc_target_acc = float(
            params.get('mcmc_target_acc', 0.234))
        self.mcmc_thin = max(1, int(params.get('mcmc_thin', 1)))
        self.pcn_beta = float(params.get('pcn_beta', 0.3))
        self.hmc_leapfrog_steps = int(
            params.get('hmc_leapfrog_steps', 10))

        # ---- Storage ----
        self.lsmcmc_mean = np.zeros((self.nassim + 1, self.dimx))
        self.RMSE = np.empty(self.nassim)

        # ---- Physical parameters ----
        self.H_mean = params.get('H_mean', 4000.0)
        self.H_rest = params.get('H_rest', [100.0, 400.0, 3500.0])
        self.rho = params.get('rho', [1023.0, 1026.0, 1028.0])
        self.T_rest = params.get('T_rest', [298.0, 283.0, 275.0])

        # ---- Noise ----
        self.sig_x_uv = params.get(
            'sig_x_uv', params.get('sig_x', 0.15))
        self.sig_x_sst = params.get('sig_x_sst', 1.0)
        self.sig_x_ssh = params.get('sig_x_ssh', self.sig_x_uv)
        self.sig_x = self.sig_x_uv

        sig_huv = self.sig_x_uv
        sig_t = self.sig_x_sst
        sig_h = self.sig_x_ssh

        assimilate_fields = str(
            params.get('assimilate_fields', 'uv_sst'))
        use_swot_ssh = bool(params.get('use_swot_ssh', False))
        assim_uv = 'uv' in assimilate_fields
        assim_ssh = 'ssh' in assimilate_fields
        assim_sst = 'sst' in assimilate_fields

        sig_per_field = []
        for k in range(self.nlayers):
            if k == 0:
                sig_h0 = sig_h   if assim_ssh else 0.0
                sig_u0 = sig_huv if assim_uv  else 0.0
                sig_t0 = sig_t   if assim_sst else 0.0
                layer_sig_k = [sig_h0, sig_u0, sig_u0, sig_t0]
            else:
                layer_sig_k = [0.0, 0.0, 0.0, 0.0]
            sig_per_field.extend(layer_sig_k)
        self.sig_x_vec = np.repeat(sig_per_field, self.ncells)

        print(f"[NL-LSMCMC-v2] assimilate_fields="
              f"'{assimilate_fields}' -> uv={assim_uv}, "
              f"ssh={assim_ssh}, sst={assim_sst}")
        print(f"[NL-LSMCMC-v2] Noise: h0={sig_h0:.4f}, "
              f"u0,v0={sig_u0:.4f}, T0={sig_t0:.4f}, "
              f"layers 1,2=0")

        # ---- Partition ----
        self.num_subdomains = params.get('num_subdomains', 480)
        block_list, labels, nblocks, nby, nbx, bh, bw = \
            partition_domain(self.ny, self.nx, self.num_subdomains)
        self.partition_labels = labels
        self.block_list = block_list
        self.n_blocks = nblocks

        # ---- Grid coordinates ----
        self.lon_min = params['lon_min']
        self.lon_max = params['lon_max']
        self.lat_min = params['lat_min']
        self.lat_max = params['lat_max']
        self.lon = np.linspace(self.lon_min, self.lon_max, self.nx)
        self.lat = np.linspace(self.lat_min, self.lat_max, self.ny)

        lat_center = 0.5 * (self.lat_min + self.lat_max)
        dlon = (self.lon_max - self.lon_min) / (self.nx - 1)
        dlat = (self.lat_max - self.lat_min) / (self.ny - 1)
        self.dx, self.dy = lonlat_to_dxdy(lat_center, dlon, dlat)
        self.f_2d = coriolis_array(
            self.lat_min, self.lat_max, self.ny, self.nx)

        # ---- Localization parameters ----
        self.r_loc = float(params.get('r_loc', 15.0))
        self.rtps_alpha = float(params.get('rtps_alpha', 0.5))
        self.gc_noise_inflate = bool(
            params.get('gc_noise_inflate', True))
        self.reset_interval = int(
            params.get('reset_interval', 0))
        self.n_block_workers = int(
            params.get('n_block_workers', 16))

        # ---- Multiprocessing ----
        self.ncores = params.get('ncores', 1)
        self._use_mp = self.ncores > 1
        self._mp_pool = None
        self._block_pool = None  # persistent pool for block MCMC

        # ---- Internal localization cache ----
        self._block_loc_info = []

        print(f"[NL-LSMCMC-v2] Grid: {self.ny}×{self.nx}, "
              f"dimx: {self.dimx}, dt: {self.dt}s, "
              f"nassim: {self.nassim}, r_loc: {self.r_loc}, "
              f"ncores: {self.ncores}")
        print(f"[NL-LSMCMC-v2] MCMC: kernel={self.mcmc_kernel}, "
              f"N={self.mcmc_N}, burn_in={self.burn_in}, "
              f"step_size={self.mcmc_step_size}, "
              f"adapt={self.mcmc_adapt}, thin={self.mcmc_thin}")

    # ------------------------------------------------------------------
    #  Observation index mapping — surface only
    # ------------------------------------------------------------------
    def _obs_ind_to_layer0(self, obs_ind_sv):
        return obs_ind_sv

    def _obs_ind_to_cell(self, obs_ind_sv):
        return obs_ind_sv % self.ncells

    # ------------------------------------------------------------------
    #  Build model instances
    # ------------------------------------------------------------------
    def _make_model_kwargs(self, H_b, bc_handler, tstart):
        return dict(
            rho=self.rho,
            dx=self.dx, dy=self.dy, dt=self.dt,
            f0=self.f_2d,
            g=self.params.get('g', 9.81),
            H_b=H_b,
            H_mean=self.H_mean,
            H_rest=self.H_rest,
            bottom_drag=self.params.get('bottom_drag', 1e-6),
            diff_coeff=self.params.get('diff_coeff', 500.0),
            diff_order=self.params.get('diff_order', 1),
            tracer_diff=self.params.get('tracer_diff', 100.0),
            bc_handler=bc_handler,
            tstart=tstart,
            precision=self.params.get('precision', 'double'),
            sst_nudging_rate=self.params.get(
                'sst_nudging_rate', 0.0),
            sst_nudging_ref=self.params.get(
                'sst_nudging_ref', None),
            sst_nudging_ref_times=self.params.get(
                'sst_nudging_ref_times', None),
            ssh_relax_rate=self.params.get('ssh_relax_rate', 0.0),
            ssh_relax_ref=self.params.get('ssh_relax_ref', None),
            ssh_relax_ref_times=self.params.get(
                'ssh_relax_ref_times', None),
            sst_flux_type=self.params.get('sst_flux_type', None),
            sst_alpha=float(self.params.get('sst_alpha', 15.0)),
            sst_h_mix=float(self.params.get('sst_h_mix', 50.0)),
            sst_T_air=self.params.get('sst_T_air', None),
            sst_T_air_times=self.params.get(
                'sst_T_air_times', None),
            ssh_relax_interior_floor=float(
                self.params.get('ssh_relax_interior_floor', 0.1)),
        )

    def _make_init_state(self, H_b, bc_handler, tstart):
        from mlswe.boundary_handler import MLBoundaryHandler

        if 'ic_h0' in self.params and self.params['ic_h0'] is not None:
            h0_list = [np.array(hk, dtype=np.float64)
                       for hk in self.params['ic_h0']]
            u0_list = [np.array(uk, dtype=np.float64)
                       for uk in self.params['ic_u0']]
            v0_list = [np.array(vk, dtype=np.float64)
                       for vk in self.params['ic_v0']]
            T0_list = [np.array(Tk, dtype=np.float64)
                       for Tk in self.params['ic_T0']]
            print("[NL-LSMCMC-v2] Using full-domain HYCOM IC")
            if bc_handler is not None:
                state = {}
                for k in range(self.nlayers):
                    state[f'h{k}'] = h0_list[k]
                    state[f'u{k}'] = u0_list[k]
                    state[f'v{k}'] = v0_list[k]
                    state[f'T{k}'] = T0_list[k]
                state = bc_handler(state, tstart)
                for k in range(self.nlayers):
                    h0_list[k] = state[f'h{k}']
                    u0_list[k] = state[f'u{k}']
                    v0_list[k] = state[f'v{k}']
                    T0_list[k] = state[f'T{k}']
            return h0_list, u0_list, v0_list, T0_list

        # Fallback
        print("[NL-LSMCMC-v2] WARNING: No HYCOM IC, rest state")
        h0_list, u0_list, v0_list, T0_list = [], [], [], []
        H_rest_total = sum(self.H_rest)
        for k in range(self.nlayers):
            h_k = np.full((self.ny, self.nx), self.H_rest[k],
                          dtype=np.float64)
            if H_b is not None:
                if k < self.nlayers - 1:
                    ratio = np.where(H_b < H_rest_total,
                                     H_b / H_rest_total, 1.0)
                    h_k = np.maximum(self.H_rest[k] * ratio, 5.0)
                else:
                    h_above = sum(h0_list)
                    h_k = np.maximum(H_b - h_above, 10.0)
            h0_list.append(h_k)
            u0_list.append(np.zeros((self.ny, self.nx),
                                    dtype=np.float64))
            v0_list.append(np.zeros((self.ny, self.nx),
                                    dtype=np.float64))
            T0_list.append(np.full((self.ny, self.nx),
                                   self.T_rest[k], dtype=np.float64))
        if bc_handler is not None:
            state = {}
            for k in range(self.nlayers):
                state[f'h{k}'] = h0_list[k]
                state[f'u{k}'] = u0_list[k]
                state[f'v{k}'] = v0_list[k]
                state[f'T{k}'] = T0_list[k]
            state = bc_handler(state, tstart)
            for k in range(self.nlayers):
                h0_list[k] = state[f'h{k}']
                u0_list[k] = state[f'u{k}']
                v0_list[k] = state[f'v{k}']
                T0_list[k] = state[f'T{k}']
        return h0_list, u0_list, v0_list, T0_list

    # ------------------------------------------------------------------
    #  Advance ensemble
    # ------------------------------------------------------------------
    def _advance_ensemble(self, member_times, nsteps, add_noise=True):
        Nf = self.forecast.shape[1]
        blown = []

        if self._use_mp and self._mp_pool is not None:
            args = [(self.forecast[:, j].copy(),
                     float(member_times[j]), nsteps)
                    for j in range(Nf)]
            results = self._mp_pool.map(_mp_advance, args)
            for j, (state_flat_new, t_new, blew_up) in enumerate(results):
                if blew_up or not np.all(
                        np.isfinite(state_flat_new)):
                    blown.append(j)
                    continue
                if add_noise:
                    state_flat_new += np.random.normal(
                        scale=self.sig_x_vec)
                member_times[j] = t_new
                self.forecast[:, j] = state_flat_new
        else:
            mdl = self._serial_model
            for j in range(Nf):
                mdl.state_flat = self.forecast[:, j].copy()
                mdl.t = float(member_times[j])
                old_flat = self.forecast[:, j].copy()
                for _ in range(nsteps):
                    mdl._timestep()
                    if not np.all(np.isfinite(mdl.state_flat)):
                        blown.append(j)
                        break
                else:
                    flat = mdl.state_flat.copy()
                    if add_noise:
                        flat += np.random.normal(
                            scale=self.sig_x_vec)
                    member_times[j] = float(mdl.t)
                    self.forecast[:, j] = flat

        if blown:
            healthy = [j for j in range(Nf) if j not in blown]
            if healthy:
                ens_mean = np.mean(
                    self.forecast[:, healthy], axis=1)
                for j in blown:
                    perturbed = ens_mean + np.random.normal(
                        scale=0.05 * self.sig_x_vec)
                    self.forecast[:, j] = perturbed
            warnings.warn(
                f"Reset {len(blown)}/{Nf} blown-up members")

    # ------------------------------------------------------------------
    #  Gaspari-Cohn
    # ------------------------------------------------------------------
    @staticmethod
    def _gaspari_cohn(dist, c):
        r = np.abs(dist) / c
        rho = np.zeros_like(r)
        m1 = r <= 1.0
        m2 = (~m1) & (r <= 2.0)
        r1 = r[m1]
        rho[m1] = (1.0 - (5.0/3.0)*r1**2 + (5.0/8.0)*r1**3
                   + 0.5*r1**4 - 0.25*r1**5)
        r2 = r[m2]
        rho[m2] = (4.0 - 5.0*r2 + (5.0/3.0)*r2**2
                   + (5.0/8.0)*r2**3 - 0.5*r2**4
                   + (1.0/12.0)*r2**5 - 2.0/(3.0*r2))
        rho = np.clip(rho, 0.0, 1.0)
        return rho

    # ------------------------------------------------------------------
    #  Per-block halo localization (same as lsmcmc_V2.py)
    # ------------------------------------------------------------------
    def _precompute_block_localization(self, obs_cell_indices,
                                       obs_ind_ml, sig_y_base):
        """Precompute per-block localization structures."""
        r_loc = self.r_loc
        ny, nx = self.ny, self.nx
        ncells = self.ncells
        nf_loc = self.fields_per_layer

        obs_row, obs_col = np.unravel_index(
            obs_cell_indices, (ny, nx))

        unique_blocks = np.unique(self.partition_labels)

        grid_rows, grid_cols = np.meshgrid(
            np.arange(ny), np.arange(nx), indexing='ij')

        self._block_loc_info = []

        for block_id in unique_blocks:
            block_mask = (self.partition_labels == block_id)
            block_ij = np.argwhere(block_mask)

            cy = block_ij[:, 0].mean()
            cx = block_ij[:, 1].mean()

            # Nearby observations
            dy_o = np.abs(obs_row - cy)
            dx_o = np.abs(obs_col - cx)
            dist_obs = np.sqrt(dy_o**2 + dx_o**2)

            nearby = dist_obs < r_loc
            if not np.any(nearby):
                self._block_loc_info.append(None)
                continue

            nearby_idx = np.where(nearby)[0]
            obs_global = obs_ind_ml[nearby_idx]

            # Halo cells
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
                [halo_flat_0 + f * ncells
                 for f in range(nf_loc)]))

            block_cells = np.sort(np.concatenate(
                [block_flat_0 + f * ncells
                 for f in range(nf_loc)]))

            block_in_halo = np.searchsorted(
                halo_cells, block_cells)

            H_loc = build_H_loc_from_global(
                obs_global, halo_cells, drop_unmapped=True)

            dist_local = dist_obs[nearby_idx][:H_loc.shape[0]]
            gc_w = self._gaspari_cohn(dist_local, r_loc)
            gc_w = np.maximum(gc_w, 1e-6)

            if (isinstance(sig_y_base, np.ndarray)
                    and sig_y_base.size > 1):
                sigma_y_local_base = \
                    sig_y_base[nearby_idx][:H_loc.shape[0]]
            else:
                sigma_y_local_base = np.full(
                    H_loc.shape[0],
                    float(np.atleast_1d(sig_y_base)[0]))

            if self.gc_noise_inflate:
                sigma_y_local = sigma_y_local_base / np.sqrt(gc_w)
            else:
                sigma_y_local = sigma_y_local_base.copy()

            self._block_loc_info.append({
                'nearby_idx': nearby_idx,
                'halo_cells': halo_cells,
                'block_cells': block_cells,
                'block_in_halo': block_in_halo,
                'H_loc': H_loc,
                'sigma_y_local': sigma_y_local,
            })

        n_active = sum(
            1 for b in self._block_loc_info if b is not None)
        n_total = len(unique_blocks)
        if n_active > 0:
            avg_nobs = np.mean([
                b['H_loc'].shape[0]
                for b in self._block_loc_info if b is not None])
            avg_halo = np.mean([
                len(b['halo_cells'])
                for b in self._block_loc_info if b is not None])
        else:
            avg_nobs = avg_halo = 0.0
        print(f'  Localization: {n_active}/{n_total} active, '
              f'r_loc={r_loc:.1f}, avg obs={avg_nobs:.1f}, '
              f'avg halo={avg_halo:.0f}')

    # ==================================================================
    #  MCMC target-density (per-block, local state)
    # ==================================================================
    @staticmethod
    def _log_likelihood_local(z_local, y, H_loc, sig_y_vec,
                               obs_operator=None):
        """log g(z, y) = -0.5 ||(y - h(z)) / σ_y||²"""
        if obs_operator is not None:
            y_pred = obs_operator(z_local, H_loc, None)
        else:
            if sp.issparse(H_loc):
                y_pred = H_loc.dot(z_local)
            else:
                y_pred = H_loc @ z_local
        residual = y - y_pred
        return -0.5 * np.sum((residual / sig_y_vec) ** 2)

    @staticmethod
    def _log_transition_local(z_local, m_i, inv_sig_x_nz, nz_mask):
        """log f(m_i, z) = -0.5 ||(z - m_i) / σ_x||²"""
        diff = z_local[nz_mask] - m_i[nz_mask]
        return -0.5 * np.sum((diff * inv_sig_x_nz) ** 2)

    @staticmethod
    def _log_transition_all_local(z_local, fc_halo, inv_sig_x_nz,
                                  nz_mask):
        """Vectorized: log f(m_i, z) for ALL i simultaneously.

        Returns (Nf,) array.
        """
        diff = z_local[nz_mask, None] - fc_halo[nz_mask, :]  # (n_nz, Nf)
        return -0.5 * np.sum((diff * inv_sig_x_nz[:, None]) ** 2, axis=0)

    # ==================================================================
    #  Per-block MCMC: Gibbs-within-MH
    # ==================================================================
    def _mcmc_block_gibbs_mh(self, fc_halo, y_local, H_loc,
                              sig_y_vec, sig_x_halo,
                              n_samples, burn_in, rng,
                              obs_operator=None):
        """Gibbs-within-MH on a single block's halo state.

        Parameters
        ----------
        fc_halo : (d_halo, Nf) — forecast at halo cells
        y_local : (d_y_loc,) — local observations
        H_loc   : sparse (d_y_loc, d_halo)
        sig_y_vec : (d_y_loc,) — GC-tapered obs noise
        sig_x_halo : (d_halo,) — per-component process noise
        n_samples, burn_in : int
        rng : np.random.Generator

        Returns
        -------
        samples : (d_halo, n_samples)
        acc_rate : float
        """
        Nf = fc_halo.shape[1]
        dim = fc_halo.shape[0]

        nz_mask = sig_x_halo > 0
        n_nz = nz_mask.sum()
        if n_nz == 0:
            return np.tile(np.mean(fc_halo, axis=1),
                           (n_samples, 1)).T, 0.0

        # Initialise
        inv_sig_x_nz = 1.0 / sig_x_halo[nz_mask]  # precomputed
        prop_scale = self.mcmc_step_size * sig_x_halo[nz_mask]

        i_curr = rng.integers(Nf)
        z_curr = fc_halo[:, i_curr].copy()
        log_lik = self._log_likelihood_local(
            z_curr, y_local, H_loc, sig_y_vec, obs_operator)
        log_trans = self._log_transition_local(
            z_curr, fc_halo[:, i_curr], inv_sig_x_nz, nz_mask)

        step_size = self.mcmc_step_size
        adapt_interval = self.mcmc_adapt_interval
        target_acc = self.mcmc_target_acc
        thin = self.mcmc_thin

        total_iters = burn_in + n_samples * thin
        samples = np.zeros((dim, n_samples))
        n_accept = 0
        s_out = 0
        _log_s = np.log(max(step_size, 1e-6))

        for s in range(total_iters):
            _n_acc_before = n_accept
            # --- Gibbs step for i (vectorized) ---
            log_weights_raw = self._log_transition_all_local(
                z_curr, fc_halo, inv_sig_x_nz, nz_mask)
            lw = log_weights_raw - log_weights_raw.max()
            weights = np.exp(lw)
            wsum = weights.sum()
            if wsum <= 0 or not np.isfinite(wsum):
                weights[:] = 1.0 / Nf
            else:
                weights /= wsum
            i_curr = rng.choice(Nf, p=weights)
            # Reuse cached value — no redundant call
            log_trans = log_weights_raw[i_curr]

            # --- MH step for z given i ---
            z_prop = z_curr.copy()
            z_prop[nz_mask] += prop_scale * rng.standard_normal(n_nz)

            log_lik_prop = self._log_likelihood_local(
                z_prop, y_local, H_loc, sig_y_vec, obs_operator)
            log_trans_prop = self._log_transition_local(
                z_prop, fc_halo[:, i_curr], inv_sig_x_nz, nz_mask)

            log_alpha = ((log_lik_prop + log_trans_prop)
                         - (log_lik + log_trans))

            if np.log(rng.random()) < log_alpha:
                z_curr = z_prop
                log_lik = log_lik_prop
                log_trans = log_trans_prop
                n_accept += 1

            # Adapt — Robbins-Monro
            if self.mcmc_adapt and s > 0:
                _accepted_this = (n_accept > _n_acc_before)
                _rm_g = 0.5 / (1.0 + s) ** 0.6
                _log_s += _rm_g * (float(_accepted_this) - target_acc)
                _log_s = max(min(_log_s, np.log(10.0)), np.log(1e-4))
                step_size = np.exp(_log_s)
                prop_scale = step_size * sig_x_halo[nz_mask]

            if s >= burn_in and (s - burn_in) % thin == 0:
                samples[:, s_out] = z_curr
                s_out += 1
                if s_out >= n_samples:
                    break

        acc_rate = n_accept / max(total_iters, 1)
        return samples, acc_rate

    # ==================================================================
    #  Per-block MCMC: Joint RW-MH
    # ==================================================================
    def _mcmc_block_joint_mh(self, fc_halo, y_local, H_loc,
                              sig_y_vec, sig_x_halo,
                              n_samples, burn_in, rng,
                              obs_operator=None):
        """Joint RW-MH on a single block's halo state."""
        Nf = fc_halo.shape[1]
        dim = fc_halo.shape[0]

        nz_mask = sig_x_halo > 0
        n_nz = nz_mask.sum()
        if n_nz == 0:
            return np.tile(np.mean(fc_halo, axis=1),
                           (n_samples, 1)).T, 0.0

        i_curr = rng.integers(Nf)
        z_curr = fc_halo[:, i_curr].copy()
        inv_sig_x_nz = 1.0 / sig_x_halo[nz_mask]
        prop_scale = self.mcmc_step_size * sig_x_halo[nz_mask]

        log_lik = self._log_likelihood_local(
            z_curr, y_local, H_loc, sig_y_vec, obs_operator)
        log_trans = self._log_transition_local(
            z_curr, fc_halo[:, i_curr], inv_sig_x_nz, nz_mask)

        step_size = self.mcmc_step_size
        adapt_interval = self.mcmc_adapt_interval
        target_acc = self.mcmc_target_acc
        thin = self.mcmc_thin

        total_iters = burn_in + n_samples * thin
        samples = np.zeros((dim, n_samples))
        n_accept = 0
        s_out = 0
        _log_s = np.log(max(step_size, 1e-4))

        for s in range(total_iters):
            _n_acc_before = n_accept
            i_prop = rng.integers(Nf)
            z_prop = z_curr.copy()
            z_prop[nz_mask] += prop_scale * rng.standard_normal(n_nz)

            log_lik_prop = self._log_likelihood_local(
                z_prop, y_local, H_loc, sig_y_vec, obs_operator)
            log_trans_prop = self._log_transition_local(
                z_prop, fc_halo[:, i_prop], inv_sig_x_nz, nz_mask)

            log_alpha = ((log_lik_prop + log_trans_prop)
                         - (log_lik + log_trans))

            if np.log(rng.random()) < log_alpha:
                z_curr = z_prop
                i_curr = i_prop
                log_lik = log_lik_prop
                log_trans = log_trans_prop
                n_accept += 1

            # Adapt — Robbins-Monro
            if self.mcmc_adapt and s > 0:
                _accepted_this = (n_accept > _n_acc_before)
                _rm_g = 0.5 / (1.0 + s) ** 0.6
                _log_s += _rm_g * (float(_accepted_this) - target_acc)
                _log_s = max(min(_log_s, np.log(10.0)), np.log(1e-4))
                step_size = np.exp(_log_s)
                prop_scale = step_size * sig_x_halo[nz_mask]

            if s >= burn_in and (s - burn_in) % thin == 0:
                samples[:, s_out] = z_curr
                s_out += 1
                if s_out >= n_samples:
                    break

        acc_rate = n_accept / max(total_iters, 1)
        return samples, acc_rate

    # ==================================================================
    #  Block-by-block localized assimilation with MCMC
    # ==================================================================
    def _assimilate_localized(self, cycle, y_valid, obs_ind_ml,
                              sig_y_cycle, forecast_mean, H_b):
        """Per-block localized assimilation using MCMC.

        For each block independently:
          1. Gather halo forecast and local observations.
          2. Run MCMC (Gibbs-within-MH or joint RW-MH) on the halo
             state to sample from π̂(z, i).
          3. Retain only the block cells from the MCMC output.
          4. Reduce samples → Nf via rescaled_block_means.
          5. Apply RTPS inflation.

        Blocks are processed in parallel via mp.Pool (true process
        parallelism, bypassing the GIL).

        Returns
        -------
        acc_list : list of float — per-block acceptance rates
        n_analyzed : int
        """
        Nf = self.nforecast
        N_a = self.mcmc_N

        self.lsmcmc_mean[cycle + 1] = forecast_mean.copy()

        fc_prior = self.forecast.copy()

        master_rng = np.random.default_rng()
        n_blocks = len(self._block_loc_info)
        block_seeds = master_rng.integers(0, 2**63, size=n_blocks)

        sig_x_vec = self.sig_x_vec
        rtps_alpha_val = self.rtps_alpha
        mcmc_kernel = self.mcmc_kernel
        burn_in = self.burn_in

        # Determine obs operator name for pickling
        obs_op_name = None
        if self.obs_operator is not None:
            obs_op_name = 'arctan'

        # ---- Build args for each block ----
        # Shared permutation seed ensures all blocks group MCMC
        # samples identically → spatially coherent ensemble members
        shared_perm_seed = int(np.random.randint(0, 2**31))
        block_args = []
        for block_idx in range(n_blocks):
            info = self._block_loc_info[block_idx]
            if info is None:
                block_args.append(None)
                continue

            H_loc = info['H_loc']
            d_y_loc = H_loc.shape[0]
            if d_y_loc == 0:
                block_args.append(None)
                continue

            nearby_idx = info['nearby_idx']
            halo_cells = info['halo_cells']
            block_cells = info['block_cells']
            block_in_halo = info['block_in_halo']
            sigma_y_loc = info['sigma_y_local']

            y_local = y_valid[nearby_idx][:d_y_loc]
            fc_halo = fc_prior[halo_cells, :]
            prior_sprd = np.std(
                fc_prior[block_cells, :], axis=1)
            sig_x_halo = sig_x_vec[halo_cells]

            # Serialize sparse H_loc for pickling
            H_loc_tuple = (H_loc.data.copy(),
                           H_loc.indices.copy(),
                           H_loc.indptr.copy(),
                           H_loc.shape)

            block_args.append((
                block_idx, fc_halo, y_local, H_loc_tuple,
                sigma_y_loc, sig_x_halo, block_cells,
                block_in_halo, prior_sprd,
                N_a, burn_in, self.mcmc_step_size,
                self.mcmc_adapt, self.mcmc_adapt_interval,
                self.mcmc_target_acc, self.mcmc_thin,
                mcmc_kernel, obs_op_name,
                rtps_alpha_val, Nf,
                int(shared_perm_seed),
                float(self.pcn_beta),
                int(self.hmc_leapfrog_steps),
                int(block_seeds[block_idx]),
            ))

        # Filter out None entries
        valid_args = [a for a in block_args if a is not None]

        # ---- Process blocks in parallel via mp.Pool ----
        if len(valid_args) > 0:
            n_workers = min(len(valid_args), self.n_block_workers)
            if n_workers > 1 and self._block_pool is not None:
                results_valid = self._block_pool.map(
                    _block_mcmc_worker, valid_args)
            elif n_workers > 1:
                with mp.Pool(n_workers) as pool:
                    results_valid = pool.map(
                        _block_mcmc_worker, valid_args)
            else:
                results_valid = [
                    _block_mcmc_worker(a) for a in valid_args]
        else:
            results_valid = []

        # ---- Apply results ----
        acc_list = []
        n_analyzed = 0
        for result in results_valid:
            if result is None:
                continue
            bc = result['block_cells']
            self.forecast[bc, :] = result['anal_block']
            self.lsmcmc_mean[cycle + 1, bc] = result['mu_block']
            acc_list.append(result['acc_rate'])
            n_analyzed += 1

        return acc_list, n_analyzed

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    def run(self, H_b, bc_handler, obs_file, tstart):
        """Run the full NL-LSMCMC V2 filter."""
        yobs, yind, obs_times, sig_y = self._load_obs(obs_file)

        h0, u0, v0, T0 = self._make_init_state(
            H_b, bc_handler, tstart)
        model_kw = self._make_model_kwargs(H_b, bc_handler, tstart)

        init_model = MLSWE(h0, u0, v0, T0=T0, **model_kw)
        init_model.timesteps = 1
        self.lsmcmc_mean[0] = init_model.state_flat.copy()

        # Set up MP pool
        if self._use_mp:
            global _mp_mlswe_kwargs
            _mp_mlswe_kwargs = dict(
                rho=self.rho,
                dx=self.dx, dy=self.dy, dt=self.dt,
                f0=self.f_2d,
                g=self.params.get('g', 9.81),
                H_b=H_b,
                H_mean=self.H_mean,
                H_rest=self.H_rest,
                bottom_drag=self.params.get('bottom_drag', 1e-6),
                diff_coeff=self.params.get('diff_coeff', 500.0),
                diff_order=self.params.get('diff_order', 1),
                tracer_diff=self.params.get('tracer_diff', 100.0),
                bc_handler=bc_handler,
                tstart=tstart,
                precision=self.params.get('precision', 'double'),
                sst_nudging_rate=self.params.get(
                    'sst_nudging_rate', 0.0),
                sst_nudging_ref=self.params.get(
                    'sst_nudging_ref', None),
                sst_nudging_ref_times=self.params.get(
                    'sst_nudging_ref_times', None),
                ssh_relax_rate=self.params.get(
                    'ssh_relax_rate', 0.0),
                ssh_relax_ref=self.params.get(
                    'ssh_relax_ref', None),
                ssh_relax_ref_times=self.params.get(
                    'ssh_relax_ref_times', None),
                sst_flux_type=self.params.get(
                    'sst_flux_type', None),
                sst_alpha=float(
                    self.params.get('sst_alpha', 15.0)),
                sst_h_mix=float(
                    self.params.get('sst_h_mix', 50.0)),
                sst_T_air=self.params.get('sst_T_air', None),
                sst_T_air_times=self.params.get(
                    'sst_T_air_times', None),
                ssh_relax_interior_floor=float(
                    self.params.get(
                        'ssh_relax_interior_floor', 0.1)),
                shallow_drag_depth=float(
                    self.params.get('shallow_drag_depth', 500.0)),
                shallow_drag_coeff=float(
                    self.params.get('shallow_drag_coeff', 5.0e-4)),
            )
            _mp_mlswe_kwargs['h0'] = [hk.copy() for hk in h0]
            _mp_mlswe_kwargs['u0'] = [uk.copy() for uk in u0]
            _mp_mlswe_kwargs['v0'] = [vk.copy() for vk in v0]
            _mp_mlswe_kwargs['T0'] = [Tk.copy() for Tk in T0]
            n_workers = min(self.ncores, self.nforecast)
            print(f"[NL-LSMCMC-v2] MP pool: {n_workers} workers")
            self._mp_pool = mp.Pool(
                n_workers, initializer=_mp_init_worker)

            # Persistent pool for block MCMC (no initializer needed)
            n_block_w = min(self.ncores, self.n_block_workers)
            print(f"[NL-LSMCMC-v2] Block MCMC pool: "
                  f"{n_block_w} workers")
            self._block_pool = mp.Pool(n_block_w)
        else:
            print(f"[NL-LSMCMC-v2] Serial mode (ncores=1)")

        # Initialise ensemble from the single init_model (no extra MLSWE instances)
        Nf = self.nforecast
        self.forecast = np.zeros((self.dimx, Nf))
        ic_flat = init_model.state_flat.copy()
        member_times = np.full(Nf, float(init_model.t))
        for j in range(Nf):
            self.forecast[:, j] = ic_flat
        self._serial_model = init_model

        print(f"[NL-LSMCMC-v2] Ensemble: {Nf} members (flat arrays, no model copies)")
        _spread = np.std(self.forecast, axis=1)
        assert np.allclose(_spread, 0.0), \
            "Members not identical at init!"

        nc = self.ncells
        t_freq = self.assim_timesteps
        rmse_vel_all = np.full(self.nassim, np.nan)
        rmse_sst_all = np.full(self.nassim, np.nan)
        rmse_ssh_all = np.full(self.nassim, np.nan)

        # Seed RNG for reproducible ensemble noise
        np.random.seed(self.params.get('random_seed', 42))

        for cycle in range(self.nassim):
            t0_wall = time.time()

            self._advance_ensemble(
                member_times, t_freq, add_noise=(self.sig_x > 0))
            forecast_mean = np.mean(self.forecast, axis=1)

            if (cycle + 1) % 50 == 0 or cycle == 0:
                h_std = np.std(self.forecast[0:nc, :], axis=1)
                u_std = np.std(self.forecast[nc:2*nc, :], axis=1)
                print(f"  [spread] h_std: mean={h_std.mean():.3f} "
                      f"max={h_std.max():.3f}  "
                      f"u_std: mean={u_std.mean():.3f} "
                      f"max={u_std.max():.3f}")

            # Get observations
            if cycle >= len(yobs):
                self.lsmcmc_mean[cycle + 1] = forecast_mean
                continue
            y = yobs[cycle]
            ind = yind[cycle]
            valid = (ind >= 0) & np.isfinite(y)
            if valid.sum() == 0:
                self.lsmcmc_mean[cycle + 1] = forecast_mean
                continue

            y_valid = y[valid]
            ind_valid = ind[valid].astype(int)

            if isinstance(sig_y, np.ndarray) and sig_y.ndim == 2:
                sig_y_cycle = sig_y[cycle][valid]
            elif isinstance(sig_y, np.ndarray) and sig_y.ndim == 1:
                sig_y_cycle = (sig_y[valid] if sig_y.size > 1
                               else float(sig_y))
            else:
                sig_y_cycle = float(sig_y)

            obs_ind_ml = self._obs_ind_to_layer0(ind_valid)
            obs_cell = self._obs_ind_to_cell(ind_valid)

            # Precompute block localization
            self._precompute_block_localization(
                obs_cell, obs_ind_ml, sig_y_cycle)

            # Block-by-block MCMC assimilation
            acc_list, n_analyzed = self._assimilate_localized(
                cycle, y_valid, obs_ind_ml, sig_y_cycle,
                forecast_mean, H_b)

            # Post-analysis SSH relaxation
            ssh_relax = float(
                self.params.get('ssh_relax_rate', 0.0))
            if ssh_relax > 0:
                nc_ = self.ncells
                h_total_anal = self.lsmcmc_mean[
                    cycle + 1, :nc_].copy()
                H_b_flat_ = H_b.ravel()
                eta_anal = h_total_anal - H_b_flat_
                t_now = float(member_times[0])
                ssh_ref_3d = self.params.get(
                    'ssh_relax_ref', None)
                ssh_ref_times = self.params.get(
                    'ssh_relax_ref_times', None)
                if (ssh_ref_3d is not None
                        and ssh_ref_times is not None):
                    from mlswe.model import MLSWE as _MLSWE
                    _tmp = _MLSWE.__new__(_MLSWE)
                    _tmp._ssh_ref_3d = ssh_ref_3d
                    _tmp._ssh_ref_times = ssh_ref_times
                    _tmp._ssh_ref_static = None
                    eta_ref = _tmp._get_ssh_ref(t_now)
                    if eta_ref is None:
                        eta_ref = np.zeros_like(eta_anal)
                    else:
                        eta_ref = eta_ref.ravel()
                else:
                    eta_ref = np.zeros_like(eta_anal)
                anal_relax_frac = float(
                    self.params.get(
                        'ssh_analysis_relax_frac', 0.5))
                eta_correction = -anal_relax_frac * (
                    eta_anal - eta_ref)
                self.lsmcmc_mean[cycle + 1, :nc_] = (
                    h_total_anal + eta_correction)
                for j_ens in range(Nf):
                    h_j = self.forecast[:nc_, j_ens]
                    eta_j = h_j - H_b_flat_
                    self.forecast[:nc_, j_ens] = (
                        h_j - anal_relax_frac
                        * (eta_j - eta_ref))

            # Periodic reset & rejuvenation
            if (self.reset_interval > 0
                    and (cycle + 1) % self.reset_interval == 0):
                ens_mean = np.mean(
                    self.forecast, axis=1, keepdims=True)
                self.forecast = np.tile(ens_mean, (1, Nf))
                noise = (np.random.normal(
                    size=self.forecast.shape)
                    * self.sig_x_vec[:, None])
                self.forecast += noise
                self.forecast += (
                    ens_mean - np.mean(
                        self.forecast, axis=1, keepdims=True))
                print(f'  !! Reset at cycle {cycle+1}')

            # (No model sync needed — state tracked in self.forecast)

            # RMSE
            z_a = self.lsmcmc_mean[cycle + 1]
            if (hasattr(self, '_truth_state')
                    and self._truth_state is not None):
                z_t = self._truth_state[cycle + 1]
                _at_obs = (hasattr(self, '_rmse_obs_only')
                           and self._rmse_obs_only)
                if _at_obs:
                    # Real data: RMSE at observation locations vs HYCOM
                    diff_obs = z_a[obs_ind_ml] - z_t[obs_ind_ml]
                    vel_mask = ((obs_ind_ml >= nc)
                                & (obs_ind_ml < 3*nc))
                    sst_mask = ((obs_ind_ml >= 3*nc)
                                & (obs_ind_ml < 4*nc))
                    ssh_mask = ((obs_ind_ml >= 0)
                                & (obs_ind_ml < nc))
                    if vel_mask.sum() > 0:
                        rmse_vel_all[cycle] = np.sqrt(
                            np.mean(diff_obs[vel_mask]**2))
                    if sst_mask.sum() > 0:
                        rmse_sst_all[cycle] = np.sqrt(
                            np.mean(diff_obs[sst_mask]**2))
                    if ssh_mask.sum() > 0:
                        rmse_ssh_all[cycle] = np.sqrt(
                            np.mean(diff_obs[ssh_mask]**2))
                else:
                    # Twin experiment: RMSE vs truth over ALL grid cells
                    diff = z_a - z_t
                    vel_diff = diff[nc:3*nc]
                    rmse_vel_all[cycle] = np.sqrt(np.mean(vel_diff**2))
                    sst_diff = diff[3*nc:4*nc]
                    rmse_sst_all[cycle] = np.sqrt(np.mean(sst_diff**2))
                    ssh_diff = diff[0:nc]
                    rmse_ssh_all[cycle] = np.sqrt(np.mean(ssh_diff**2))
            else:
                H_z = z_a[obs_ind_ml]
                residuals = H_z - y_valid
                vel_mask = ((obs_ind_ml >= nc)
                            & (obs_ind_ml < 3*nc))
                sst_mask = ((obs_ind_ml >= 3*nc)
                            & (obs_ind_ml < 4*nc))
                ssh_mask = ((obs_ind_ml >= 0)
                            & (obs_ind_ml < nc))
                if vel_mask.sum() > 0:
                    rmse_vel_all[cycle] = np.sqrt(
                        np.mean(residuals[vel_mask]**2))
                if sst_mask.sum() > 0:
                    rmse_sst_all[cycle] = np.sqrt(
                        np.mean(residuals[sst_mask]**2))
                if ssh_mask.sum() > 0:
                    rmse_ssh_all[cycle] = np.sqrt(
                        np.mean(residuals[ssh_mask]**2))

            elapsed = time.time() - t0_wall
            if (cycle + 1) % 10 == 0 or cycle == 0:
                _h_total = z_a[0:nc]
                _ssh = _h_total - H_b.ravel()
                _fmean_ssh = forecast_mean[0:nc] - H_b.ravel()
                acc_arr = (np.array(acc_list)
                           if acc_list else np.array([0.0]))
                print(
                    f"  Cycle {cycle+1}/{self.nassim}  "
                    f"vel_RMSE={rmse_vel_all[cycle]:.6f}  "
                    f"sst_RMSE={rmse_sst_all[cycle]:.4f}  "
                    f"ssh_RMSE={rmse_ssh_all[cycle]:.4f}  "
                    f"SSH=[{_ssh.min():.1f},{_ssh.max():.1f}]  "
                    f"fcst_SSH="
                    f"[{_fmean_ssh.min():.1f},"
                    f"{_fmean_ssh.max():.1f}]  "
                    f"acc={np.mean(acc_arr):.3f} "
                    f"[{acc_arr.min():.3f},"
                    f"{acc_arr.max():.3f}]  "
                    f"blks={n_analyzed}  "
                    f"({elapsed:.1f}s)")

        # Cleanup
        if self._mp_pool is not None:
            self._mp_pool.close()
            self._mp_pool.join()
            self._mp_pool = None
            print("[NL-LSMCMC-v2] MP pool closed.")
        if self._block_pool is not None:
            self._block_pool.close()
            self._block_pool.join()
            self._block_pool = None
            print("[NL-LSMCMC-v2] Block MCMC pool closed.")

        self.rmse_vel = rmse_vel_all
        self.rmse_sst = rmse_sst_all
        self.rmse_ssh = rmse_ssh_all
        return self.lsmcmc_mean

    # ------------------------------------------------------------------
    def _load_obs(self, obs_file):
        nc = Dataset(obs_file, 'r')
        yobs = np.asarray(nc.variables['yobs_all'][:])
        yind = np.asarray(nc.variables['yobs_ind_all'][:])
        times = np.asarray(nc.variables['obs_times'][:])
        if 'sig_y_all' in nc.variables:
            sig_y = np.asarray(nc.variables['sig_y_all'][:])
        elif hasattr(nc, 'sig_y'):
            sig_y = float(nc.sig_y)
        else:
            sig_y = float(self.params.get('sig_y', 0.01))
        nc.close()
        return yobs, yind, times, sig_y

    # ------------------------------------------------------------------
    def save_results(self, outdir, obs_times=None, H_b=None):
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, 'mlswe_lsmcmc_out.nc')
        ds = Dataset(outfile, 'w', format='NETCDF4')

        ds.createDimension('time', self.nassim + 1)
        ds.createDimension('layer', self.nlayers)
        ds.createDimension('field', self.fields_per_layer)
        ds.createDimension('y', self.ny)
        ds.createDimension('x', self.nx)
        ds.createDimension('cycle', self.nassim)

        v = ds.createVariable(
            'lsmcmc_mean', 'f4',
            ('time', 'layer', 'field', 'y', 'x'), zlib=True)
        reshaped = self.lsmcmc_mean.reshape(
            self.nassim + 1, self.nlayers,
            self.fields_per_layer, self.ny, self.nx)
        v[:] = reshaped.astype(np.float32)

        vr = ds.createVariable('rmse_vel', 'f4', ('cycle',))
        vr[:] = self.rmse_vel.astype(np.float32)
        vs = ds.createVariable('rmse_sst', 'f4', ('cycle',))
        vs[:] = self.rmse_sst.astype(np.float32)
        vh = ds.createVariable('rmse_ssh', 'f4', ('cycle',))
        vh[:] = self.rmse_ssh.astype(np.float32)

        if H_b is not None:
            vb = ds.createVariable('H_b', 'f4', ('y', 'x'),
                                   zlib=True)
            vb[:] = H_b.astype(np.float32)

        if obs_times is not None:
            ot = obs_times[:self.nassim + 1]
            if len(ot) < self.nassim + 1:
                if len(obs_times) >= 2:
                    dt_obs = obs_times[1] - obs_times[0]
                    ot = np.concatenate(
                        [[obs_times[0] - dt_obs], ot])
                else:
                    ot = np.concatenate(
                        [[obs_times[0]], ot])
                ot = ot[:self.nassim + 1]
            vt = ds.createVariable(
                'obs_times', 'f8', ('time',))
            vt[:] = ot

        ds.nlayers = self.nlayers
        ds.fields_per_layer = self.fields_per_layer
        ds.ny = self.ny
        ds.nx = self.nx
        ds.close()
        print(f"[NL-LSMCMC-v2] Saved results to {outfile}")

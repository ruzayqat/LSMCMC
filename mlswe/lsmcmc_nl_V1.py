#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lsmcmc_nl.py — Nonlinear LSMCMC (V1) with block-partition localization
=======================================================================

Samples from the SMCMC target distribution

    π̂(z_t, i) ∝ g(z_t, y_t) · f(Z_{t-1}^{(i)}, z_t) · p(i)

using MCMC (Gibbs-within-MH or joint RW-MH), rather than the
exact-from-Gaussian sampler used in ``lsmcmc.py``.

This allows nonlinear observation operators  h : z → y_pred ,
where the likelihood is  g(z, y) = N(y; h(z), Σ_y).

When h is linear (H @ z), the code defaults to the matrix operator
but the MCMC framework works for any callable.

V1 strategy: the MCMC state is ALL cells in observed partition blocks
(same block-partition localization as the original ``lsmcmc.py``).
The local state dimension can be large (up to ~20 000), so MCMC may
be costly.  See ``lsmcmc_nl_V2.py`` (V2) for the cheaper per-block
halo approach.

Two MCMC kernels are implemented:
  1. **gibbs_mh** — Gibbs-within-MH: exact Gibbs step for ensemble
     index *i* (categorical), then RW-MH step for the continuous
     state *z*.
  2. **joint_mh** — Joint RW-MH: propose (z*, i*) jointly using a
     symmetric random walk on z and uniform proposal on i.

Adaptive step-size tuning during burn-in targets an acceptance rate
of ~0.234 (optimal for high-dimensional RW-MH).
"""

import os
import sys
import time
import warnings
import numpy as np
import multiprocessing as mp
from netCDF4 import Dataset

# --------------- imports from the project ---------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from loc_smcmc_swe_exact_from_Gauss import (
        partition_domain,
        get_divisors,
        build_H_loc_from_global,
        gaussian_block_means,
    )
except ImportError:
    from mlswe.lsmcmc_V2 import build_H_loc_from_global
    try:
        from loc_smcmc_swe_exact_from_Gauss import (
            partition_domain, get_divisors, gaussian_block_means)
    except ImportError:
        raise ImportError(
            "Cannot import partition_domain / gaussian_block_means. "
            "Ensure loc_smcmc_swe_exact_from_Gauss.py is on PYTHONPATH.")

from mlswe.model import MLSWE, coriolis_array, lonlat_to_dxdy


# ===================================================================
#  Utility: rescaled block means (variance-preserving)
# ===================================================================
def rescaled_block_means(samples, M, mu):
    """Reduce N samples to M via grouped means, rescaled to preserve
    variance."""
    means, groups = gaussian_block_means(samples, M)
    Mi = np.array([len(g) for g in groups])
    scale = np.sqrt(Mi)[None, :]
    mu = mu.reshape(-1, 1)
    return mu + scale * (means - mu), groups


# ===================================================================
#  Multiprocessing workers  (same as lsmcmc.py)
# ===================================================================
_mp_model_template = None   # pre-built MLSWE; fast-copied per worker
_mp_worker_model = None


def _mp_init_worker():
    """Create per-worker model via shallow copy + state array copies."""
    import copy
    global _mp_worker_model
    mdl = _mp_model_template
    new_mdl = copy.copy(mdl)
    new_mdl.h = [hk.copy() for hk in mdl.h]
    new_mdl.u = [uk.copy() for uk in mdl.u]
    new_mdl.v = [vk.copy() for vk in mdl.v]
    if mdl.use_tracer:
        new_mdl.T = [Tk.copy() if Tk is not None else None for Tk in mdl.T]
    _mp_worker_model = new_mdl


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
#  Parallel MCMC worker (runs one independent chain per process)
# ===================================================================
def _mcmc_chain_worker(args):
    """
    Run a single MCMC chain (Gibbs-within-MH or Joint-MH).

    Called via multiprocessing.Pool.map() to run independent chains
    concurrently.  Each chain uses its own random seed for
    reproducibility and independence.

    Parameters (packed in tuple)
    ----------
    fc_local   : (d, Nf) forecast ensemble (local)
    y          : (d_y,) observations
    H_loc_data : (H_loc.data, H_loc.indices, H_loc.indptr, H_loc.shape)
                 CSR components for the obs operator
    sig_y_vec  : (d_y,) observation noise per obs
    sig_x_loc  : (d,) process noise per state component
    n_samples  : int — number of post-burn-in samples for this chain
    burn_in    : int
    step_size  : float
    adapt      : bool
    adapt_interval : int
    target_acc : float
    thin       : int
    kernel     : str — 'gibbs_mh' or 'joint_mh'
    obs_op_name : str or None — 'arctan' or None (linear)
    seed       : int — RNG seed for this chain

    Returns
    -------
    samples    : (d, n_samples)
    n_accept   : int
    final_step : float
    """
    from scipy import sparse as sp

    (fc_local, y, H_loc_tuple, sig_y_vec, sig_x_loc,
     n_samples, burn_in, step_size, adapt, adapt_interval,
     target_acc, thin, kernel, obs_op_name,
     pcn_beta, hmc_leapfrog_steps, seed) = args

    rng = np.random.RandomState(seed)

    # Reconstruct sparse H_loc
    data, indices, indptr, shape = H_loc_tuple
    H_loc = sp.csr_matrix((data, indices, indptr), shape=shape)

    # Obs operator
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
        samples = np.tile(np.mean(fc_local, axis=1),
                          (n_samples, 1)).T
        return samples, 0, step_size

    inv_sig_x_nz = 1.0 / sig_x_loc[nz_mask]
    sig_x_nz = sig_x_loc[nz_mask]
    sig_x_sq_nz = sig_x_nz ** 2
    prop_scale = step_size * sig_x_nz
    beta = float(pcn_beta)
    hmc_L = int(hmc_leapfrog_steps)

    # Log-likelihood helper
    def _log_lik(z):
        if obs_operator is not None:
            y_pred = obs_operator(z, H_loc, None)
        else:
            y_pred = H_loc @ z
        residual = y - y_pred
        return -0.5 * np.sum((residual / sig_y_vec) ** 2)

    # Log-transition helpers
    def _log_trans(z, m_i):
        diff = z[nz_mask] - m_i[nz_mask]
        return -0.5 * np.sum((diff * inv_sig_x_nz) ** 2)

    def _log_trans_all(z):
        diff = z[nz_mask, None] - fc_local[nz_mask, :]
        return -0.5 * np.sum((diff * inv_sig_x_nz[:, None]) ** 2,
                             axis=0)

    # Gradient of log target (for MALA / HMC)
    def _grad_log_target(z, m_i):
        Hz = H_loc @ z
        if obs_op_name == 'arctan':
            pred = np.arctan(Hz)
            dpred = 1.0 / (1.0 + Hz ** 2)
        else:
            pred = Hz
            dpred = np.ones_like(Hz)
        residual = y - pred
        grad = np.asarray(
            H_loc.T @ (dpred * residual / sig_y_vec ** 2)
        ).ravel()
        grad[nz_mask] -= (
            z[nz_mask] - m_i[nz_mask]) / sig_x_sq_nz
        return grad

    # Initialise
    i_curr = rng.randint(Nf)
    z_curr = fc_local[:, i_curr].copy()
    log_lik = _log_lik(z_curr)
    log_trans = _log_trans(z_curr, fc_local[:, i_curr])

    total_iters = burn_in + n_samples * thin
    samples = np.zeros((dim, n_samples))
    n_accept = 0
    s_out = 0
    _log_beta = np.log(max(beta, 1e-4))  # Robbins-Monro state (pCN)
    _log_step = np.log(max(step_size, 1e-6))  # Robbins-Monro state (HMC/MALA)

    for s in range(total_iters):
        # --- Gibbs step for i (common to all kernels) ---
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

        # --- Proposal step for z (kernel-dependent) ---
        _n_acc_before = n_accept
        if kernel == 'gibbs_mh':
            z_prop = z_curr.copy()
            z_prop[nz_mask] += prop_scale * rng.randn(n_nz)
            log_lik_prop = _log_lik(z_prop)
            log_trans_prop = _log_trans(z_prop, m_i)
            log_alpha = ((log_lik_prop + log_trans_prop)
                         - (log_lik + log_trans))
            if np.log(rng.rand()) < log_alpha:
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
                + beta * sig_x_nz * rng.randn(n_nz))
            log_lik_prop = _log_lik(z_prop)
            log_alpha = log_lik_prop - log_lik
            if np.log(rng.rand()) < log_alpha:
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
                + tau * sig_x_nz * rng.randn(n_nz))
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
                if np.log(rng.rand()) < log_alpha:
                    z_curr = z_prop
                    log_lik = log_lik_prop
                    log_trans = log_trans_prop
                    n_accept += 1

        elif kernel == 'hmc':
            p_nz = rng.randn(n_nz) * inv_sig_x_nz
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
                if np.log(rng.rand()) < log_alpha:
                    z_curr = q
                    log_lik = log_lik_prop
                    log_trans = log_trans_prop
                    n_accept += 1

        # Adaptive step size — Robbins-Monro for all kernels
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

        # Store
        if s >= burn_in and (s - burn_in) % thin == 0:
            samples[:, s_out] = z_curr
            s_out += 1
            if s_out >= n_samples:
                break

    return samples, n_accept, step_size


# ===================================================================
#  Filter class
# ===================================================================
class NL_SMCMC_MLSWE_Filter:
    """
    Nonlinear LSMCMC filter for the 3-layer primitive-equations model,
    using MCMC (Gibbs-within-MH or joint RW-MH) to sample from

        π̂(z, i) ∝ g(z, y) · f(m_i, z) · p(i)

    with block-partition localization (same as ``lsmcmc.py``).

    The observation operator ``h`` may be nonlinear.  When not
    provided, the default linear operator H_loc @ z is used.

    Parameters
    ----------
    isim : int
        Simulation index (for multi-run scripts).
    params : dict
        Configuration dictionary (from YAML).
    obs_operator : callable or None
        Nonlinear observation operator.  Signature:
            y_pred = obs_operator(z_local, H_loc, obs_ind_local)
        If *None*, the linear operator ``H_loc @ z_local`` is used.
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
        self.fields_per_layer = 4   # h, u, v, T
        self.nfields = self.nlayers * self.fields_per_layer  # 12
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
        self.mcmc_step_size = float(params.get('mcmc_step_size', 0.5))
        self.mcmc_kernel = str(params.get('mcmc_kernel', 'gibbs_mh'))
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

        self.verbose = params.get('verbose', True)

        if self.verbose:
            print(f"[NL-LSMCMC-v1] assimilate_fields='{assimilate_fields}' "
                  f"-> uv={assim_uv}, ssh={assim_ssh}, sst={assim_sst}")
            print(f"[NL-LSMCMC-v1] Noise: h0={sig_h0:.4f}, "
                  f"u0,v0={sig_u0:.4f}, T0={sig_t0:.4f}, layers 1,2=0")

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

        # ---- Multiprocessing ----
        self.ncores = params.get('ncores', 1)
        self._use_mp = self.ncores > 1
        self._mp_pool = None
        self._mcmc_pool = None
        # Number of parallel MCMC chains (default: min(ncores, 8))
        self.mcmc_chains = int(
            params.get('mcmc_chains', min(self.ncores, 8)))
        if self.mcmc_chains < 2:
            self.mcmc_chains = 1      # fall back to serial

        if self.verbose:
            print(f"[NL-LSMCMC-v1] Grid: {self.ny}×{self.nx}, "
                  f"dimx: {self.dimx}, dt: {self.dt}s, "
                  f"nassim: {self.nassim}, ncores: {self.ncores}")
            print(f"[NL-LSMCMC-v1] MCMC: kernel={self.mcmc_kernel}, "
                  f"N={self.mcmc_N}, burn_in={self.burn_in}, "
                  f"step_size={self.mcmc_step_size}, "
                  f"adapt={self.mcmc_adapt}, thin={self.mcmc_thin}, "
                  f"chains={self.mcmc_chains}")

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
            sst_nudging_rate=self.params.get('sst_nudging_rate', 0.0),
            sst_nudging_ref=self.params.get('sst_nudging_ref', None),
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
            sst_T_air_times=self.params.get('sst_T_air_times', None),
            ssh_relax_interior_floor=float(
                self.params.get('ssh_relax_interior_floor', 0.1)),
            shallow_drag_depth=float(
                self.params.get('shallow_drag_depth', 500.0)),
            shallow_drag_coeff=float(
                self.params.get('shallow_drag_coeff', 5.0e-4)),
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
            if self.verbose:
                print("[NL-LSMCMC-v1] Using full-domain HYCOM IC")
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

        # Fallback: rest state
        if self.verbose:
            print("[NL-LSMCMC-v1] WARNING: No HYCOM IC, using rest state")
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
    #  Advance ensemble  (model-free: uses only flat arrays)
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
                if blew_up or not np.all(np.isfinite(state_flat_new)):
                    blown.append(j)
                    continue
                if add_noise:
                    state_flat_new += np.random.normal(
                        scale=self.sig_x_vec)
                member_times[j] = t_new
                self.forecast[:, j] = state_flat_new
        else:
            # Serial fallback: reuse the single model kept from init
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
                        flat += np.random.normal(scale=self.sig_x_vec)
                    member_times[j] = float(mdl.t)
                    self.forecast[:, j] = flat

        if blown:
            healthy = [j for j in range(Nf) if j not in blown]
            if healthy:
                ens_mean = np.mean(self.forecast[:, healthy], axis=1)
                for j in blown:
                    perturbed = ens_mean + np.random.normal(
                        scale=0.05 * self.sig_x_vec)
                    self.forecast[:, j] = perturbed
            warnings.warn(
                f"Reset {len(blown)}/{Nf} blown-up ensemble members")

    # ------------------------------------------------------------------
    #  Block-partition localization (same as lsmcmc.py)
    # ------------------------------------------------------------------
    def get_observed_blocks_cells(self, obs_cell_indices):
        if obs_cell_indices.size == 0:
            return np.array([], dtype=int)
        yobs_r, xobs_c = np.unravel_index(
            obs_cell_indices, (self.ny, self.nx))
        obs_blocks = np.unique(
            self.partition_labels[yobs_r, xobs_c])
        mask = np.isin(self.partition_labels, obs_blocks)
        ij = np.argwhere(mask)
        if ij.size == 0:
            return np.array([], dtype=int)
        flat0 = np.ravel_multi_index(
            (ij[:, 0], ij[:, 1]), (self.ny, self.nx))
        nfields_l0 = self.fields_per_layer
        all_idx = np.concatenate(
            [flat0 + f * self.ncells for f in range(nfields_l0)])
        all_idx.sort()
        return all_idx

    # ==================================================================
    #  MCMC target-density evaluation
    # ==================================================================
    def _log_likelihood(self, z_local, y, H_loc, sig_y_vec,
                        obs_operator=None):
        """
        log g(z, y) = -0.5 || (y - h(z)) / σ_y ||²

        Parameters
        ----------
        z_local : (d_local,) array — local state vector
        y       : (d_y,) array — observations
        H_loc   : sparse matrix (d_y, d_local)
        sig_y_vec : (d_y,) array — per-obs noise std
        obs_operator : callable or None
        """
        if obs_operator is not None:
            y_pred = obs_operator(z_local, H_loc, None)
        else:
            y_pred = H_loc @ z_local
        residual = y - y_pred
        return -0.5 * np.sum((residual / sig_y_vec) ** 2)

    def _log_transition(self, z_local, m_i, inv_sig_x_nz, nz_mask):
        """
        log f(m_i, z) = -0.5 || (z - m_i) / σ_x ||²

        Parameters
        ----------
        z_local     : (d_local,) array — candidate state
        m_i         : (d_local,) array — forecast of ensemble member i
        inv_sig_x_nz : (n_nz,) array — precomputed 1/σ_x for nonzero entries
        nz_mask     : (d_local,) bool array — mask for σ_x > 0
        """
        diff = z_local[nz_mask] - m_i[nz_mask]
        return -0.5 * np.sum((diff * inv_sig_x_nz) ** 2)

    def _log_transition_all(self, z_local, fc_local, inv_sig_x_nz,
                            nz_mask):
        """
        Vectorized: compute log f(m_i, z) for ALL i simultaneously.

        Returns (Nf,) array of log-transition densities.
        """
        diff = z_local[nz_mask, None] - fc_local[nz_mask, :]  # (n_nz, Nf)
        return -0.5 * np.sum((diff * inv_sig_x_nz[:, None]) ** 2, axis=0)

    # ==================================================================
    #  MCMC kernel 1: Gibbs-within-MH
    # ==================================================================
    def _mcmc_gibbs_within_mh(self, fc_local, y, H_loc, sig_y_vec,
                               sig_x_loc, n_samples, burn_in,
                               obs_operator=None):
        """
        Gibbs-within-MH sampler for (z, i).

        1. Gibbs step for i:
             π(i | z) ∝ f(m_i, z)  →  exact categorical draw
        2. MH step for z given i:
             z* = z + τ · σ_x · ε,  ε ~ N(0, I)
             accept with MH ratio

        Parameters
        ----------
        fc_local : (d_local, Nf)  — forecast ensemble (local)
        y        : (d_y,)
        H_loc    : sparse (d_y, d_local)
        sig_y_vec : (d_y,)
        sig_x_loc : (d_local,)
        n_samples : int
        burn_in   : int
        obs_operator : callable or None

        Returns
        -------
        samples : (d_local, n_samples)  — post–burn-in
        acc_rate : float
        final_step_size : float
        """
        Nf = fc_local.shape[1]
        dim = fc_local.shape[0]

        # Ensure σ_x > 0 mask for proposal
        nz_mask = sig_x_loc > 0
        n_nz = nz_mask.sum()
        if n_nz == 0:
            # No noise components → cannot do MCMC; return forecast mean
            samples = np.tile(np.mean(fc_local, axis=1),
                              (n_samples, 1)).T
            return samples, 0.0, self.mcmc_step_size

        # Precompute inverse noise for the nonzero components
        inv_sig_x_nz = 1.0 / sig_x_loc[nz_mask]   # (n_nz,)
        step_size = self.mcmc_step_size
        adapt_interval = self.mcmc_adapt_interval
        target_acc = self.mcmc_target_acc
        prop_scale = step_size * sig_x_loc[nz_mask]  # reused each iter

        # Initialise from a random ensemble member
        i_curr = np.random.randint(Nf)
        z_curr = fc_local[:, i_curr].copy()
        log_lik = self._log_likelihood(
            z_curr, y, H_loc, sig_y_vec, obs_operator)
        log_trans = self._log_transition(
            z_curr, fc_local[:, i_curr], inv_sig_x_nz, nz_mask)

        total_iters = burn_in + n_samples * self.mcmc_thin
        samples = np.zeros((dim, n_samples))
        n_accept = 0
        s_out = 0  # sample counter

        for s in range(total_iters):
            # --- Gibbs step for i (vectorized over all Nf) ---
            log_weights_raw = self._log_transition_all(
                z_curr, fc_local, inv_sig_x_nz, nz_mask)
            lw = log_weights_raw - log_weights_raw.max()
            weights = np.exp(lw)
            wsum = weights.sum()
            if wsum <= 0 or not np.isfinite(wsum):
                weights[:] = 1.0 / Nf
            else:
                weights /= wsum
            i_curr = np.random.choice(Nf, p=weights)
            # Reuse already-computed value — no redundant call
            log_trans = log_weights_raw[i_curr]

            # --- MH step for z given i ---
            z_prop = z_curr.copy()
            z_prop[nz_mask] += prop_scale * np.random.randn(n_nz)

            log_lik_prop = self._log_likelihood(
                z_prop, y, H_loc, sig_y_vec, obs_operator)
            log_trans_prop = self._log_transition(
                z_prop, fc_local[:, i_curr], inv_sig_x_nz, nz_mask)

            log_alpha = ((log_lik_prop + log_trans_prop)
                         - (log_lik + log_trans))

            if np.log(np.random.rand()) < log_alpha:
                z_curr = z_prop
                log_lik = log_lik_prop
                log_trans = log_trans_prop
                n_accept += 1

            # --- Adaptive step size during burn-in ---
            if (self.mcmc_adapt and s < burn_in
                    and s > 0 and s % adapt_interval == 0):
                acc_rate_now = n_accept / (s + 1)
                if acc_rate_now < target_acc * 0.6:
                    step_size *= 0.8
                    prop_scale = step_size * sig_x_loc[nz_mask]
                elif acc_rate_now > target_acc * 1.6:
                    step_size *= 1.2
                    prop_scale = step_size * sig_x_loc[nz_mask]
                step_size = np.clip(step_size, 1e-4, 10.0)

            # --- Store sample (after burn-in, with thinning) ---
            if s >= burn_in and (s - burn_in) % self.mcmc_thin == 0:
                samples[:, s_out] = z_curr
                s_out += 1
                if s_out >= n_samples:
                    break

        acc_rate = n_accept / max(total_iters, 1)
        return samples, acc_rate, step_size

    # ==================================================================
    #  MCMC kernel 2: Joint RW-MH
    # ==================================================================
    def _mcmc_joint_mh(self, fc_local, y, H_loc, sig_y_vec,
                        sig_x_loc, n_samples, burn_in,
                        obs_operator=None):
        """
        Joint RW-MH sampler for (z, i).

        Propose z* = z + τ·σ_x·ε and i* ~ Uniform{1,...,Nf} jointly.
        Accept/reject with MH ratio (proposal is symmetric).

        Returns
        -------
        samples, acc_rate, final_step_size
        """
        Nf = fc_local.shape[1]
        dim = fc_local.shape[0]

        nz_mask = sig_x_loc > 0
        n_nz = nz_mask.sum()
        if n_nz == 0:
            samples = np.tile(np.mean(fc_local, axis=1),
                              (n_samples, 1)).T
            return samples, 0.0, self.mcmc_step_size

        inv_sig_x_nz = 1.0 / sig_x_loc[nz_mask]
        step_size = self.mcmc_step_size
        adapt_interval = self.mcmc_adapt_interval
        target_acc = self.mcmc_target_acc
        prop_scale = step_size * sig_x_loc[nz_mask]

        i_curr = np.random.randint(Nf)
        z_curr = fc_local[:, i_curr].copy()
        log_lik = self._log_likelihood(
            z_curr, y, H_loc, sig_y_vec, obs_operator)
        log_trans = self._log_transition(
            z_curr, fc_local[:, i_curr], inv_sig_x_nz, nz_mask)

        total_iters = burn_in + n_samples * self.mcmc_thin
        samples = np.zeros((dim, n_samples))
        n_accept = 0
        s_out = 0

        for s in range(total_iters):
            # Joint proposal
            i_prop = np.random.randint(Nf)
            z_prop = z_curr.copy()
            z_prop[nz_mask] += prop_scale * np.random.randn(n_nz)

            log_lik_prop = self._log_likelihood(
                z_prop, y, H_loc, sig_y_vec, obs_operator)
            log_trans_prop = self._log_transition(
                z_prop, fc_local[:, i_prop], inv_sig_x_nz, nz_mask)

            log_alpha = ((log_lik_prop + log_trans_prop)
                         - (log_lik + log_trans))

            if np.log(np.random.rand()) < log_alpha:
                z_curr = z_prop
                i_curr = i_prop
                log_lik = log_lik_prop
                log_trans = log_trans_prop
                n_accept += 1

            # Adapt during burn-in
            if (self.mcmc_adapt and s < burn_in
                    and s > 0 and s % adapt_interval == 0):
                acc_rate_now = n_accept / (s + 1)
                if acc_rate_now < target_acc * 0.6:
                    step_size *= 0.8
                    prop_scale = step_size * sig_x_loc[nz_mask]
                elif acc_rate_now > target_acc * 1.6:
                    step_size *= 1.2
                    prop_scale = step_size * sig_x_loc[nz_mask]
                step_size = np.clip(step_size, 1e-4, 10.0)

            if s >= burn_in and (s - burn_in) % self.mcmc_thin == 0:
                samples[:, s_out] = z_curr
                s_out += 1
                if s_out >= n_samples:
                    break

        acc_rate = n_accept / max(total_iters, 1)
        return samples, acc_rate, step_size

    # ==================================================================
    #  Dispatch MCMC kernel
    # ==================================================================
    def _run_mcmc(self, fc_local, y, H_loc, sig_y_vec, sig_x_loc,
                  obs_operator=None):
        """Run the configured MCMC kernel and return (samples, acc_rate,
        step_size).

        When ncores > 1 AND a multiprocessing pool is available, runs
        multiple independent chains in parallel and merges the samples.
        Each chain gets (mcmc_N / n_chains) samples with its own burn-in.
        """
        n_chains = self.mcmc_chains if self._use_mp else 1

        if n_chains > 1:
            return self._run_mcmc_parallel(
                fc_local, y, H_loc, sig_y_vec, sig_x_loc,
                obs_operator, n_chains)

        # Single-chain fallback
        if self.mcmc_kernel == 'gibbs_mh':
            return self._mcmc_gibbs_within_mh(
                fc_local, y, H_loc, sig_y_vec, sig_x_loc,
                self.mcmc_N, self.burn_in, obs_operator)
        elif self.mcmc_kernel == 'joint_mh':
            return self._mcmc_joint_mh(
                fc_local, y, H_loc, sig_y_vec, sig_x_loc,
                self.mcmc_N, self.burn_in, obs_operator)
        else:
            raise ValueError(
                f"Unknown MCMC kernel: '{self.mcmc_kernel}'. "
                f"Choose 'gibbs_mh' or 'joint_mh'.")

    def _run_mcmc_parallel(self, fc_local, y, H_loc, sig_y_vec,
                           sig_x_loc, obs_operator, n_chains):
        """
        Run multiple independent MCMC chains in parallel, then merge.

        Each chain produces (mcmc_N // n_chains) post-burn-in samples.
        All chains run their own burn-in independently.  Results are
        concatenated to produce the final sample matrix.
        """
        from scipy import sparse as sp

        total_samples = self.mcmc_N
        per_chain = max(total_samples // n_chains, 10)

        # Pack H_loc as CSR components (picklable)
        if sp.issparse(H_loc):
            H_csr = H_loc.tocsr()
            H_tuple = (H_csr.data.copy(), H_csr.indices.copy(),
                       H_csr.indptr.copy(), H_csr.shape)
        else:
            H_csr = sp.csr_matrix(H_loc)
            H_tuple = (H_csr.data.copy(), H_csr.indices.copy(),
                       H_csr.indptr.copy(), H_csr.shape)

        # Detect obs operator name
        obs_op_name = None
        if obs_operator is not None:
            obs_op_name = 'arctan'

        base_seed = np.random.randint(0, 2**31)
        args_list = [
            (fc_local.copy(), y.copy(), H_tuple,
             sig_y_vec.copy(), sig_x_loc.copy(),
             per_chain, self.burn_in, self.mcmc_step_size,
             self.mcmc_adapt, self.mcmc_adapt_interval,
             self.mcmc_target_acc, self.mcmc_thin,
             self.mcmc_kernel, obs_op_name,
             float(self.pcn_beta),
             int(self.hmc_leapfrog_steps),
             base_seed + c)
            for c in range(n_chains)
        ]

        # Reuse persistent MCMC pool (created in run())
        results = self._mcmc_pool.map(_mcmc_chain_worker, args_list)

        # Merge samples from all chains
        all_samples = np.concatenate(
            [r[0] for r in results], axis=1)
        total_accept = sum(r[1] for r in results)
        total_iters_all = sum(
            self.burn_in + per_chain * self.mcmc_thin
            for _ in range(n_chains))
        acc_rate = total_accept / max(total_iters_all, 1)
        avg_step = np.mean([r[2] for r in results])

        return all_samples, acc_rate, avg_step

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    def run(self, H_b, bc_handler, obs_file, tstart):
        """Run the full NL-LSMCMC filter."""
        # Load observations
        yobs, yind, obs_times, sig_y = self._load_obs(obs_file)

        # Initial state
        h0, u0, v0, T0 = self._make_init_state(
            H_b, bc_handler, tstart)
        model_kw = self._make_model_kwargs(H_b, bc_handler, tstart)

        init_model = MLSWE(h0, u0, v0, T0=T0, **model_kw)
        init_model.timesteps = 1
        self.lsmcmc_mean[0] = init_model.state_flat.copy()

        # Set up multiprocessing pool
        if self._use_mp:
            global _mp_model_template
            # Reuse the already-built model as template; workers deepcopy it
            # (skips CFL check, gradient computation, Coriolis, array alloc)
            _mp_model_template = init_model

            n_workers = min(self.ncores, self.nforecast)
            if self.verbose:
                print(f"[NL-LSMCMC-v1] Starting forecast MP pool "
                      f"with {n_workers} workers")
            self._mp_pool = mp.Pool(
                n_workers, initializer=_mp_init_worker)

            # Persistent MCMC pool (no initializer needed)
            if self.mcmc_chains > 1:
                if self.verbose:
                    print(f"[NL-LSMCMC-v1] Starting MCMC pool "
                          f"with {self.mcmc_chains} chains")
                self._mcmc_pool = mp.Pool(self.mcmc_chains)
        else:
            if self.verbose:
                print(f"[NL-LSMCMC-v1] Serial mode (ncores=1)")

        # Initialise ensemble from the single init_model (no extra MLSWE instances)
        Nf = self.nforecast
        self.forecast = np.zeros((self.dimx, Nf))
        ic_flat = init_model.state_flat.copy()
        member_times = np.full(Nf, float(init_model.t))
        for j in range(Nf):
            self.forecast[:, j] = ic_flat
        # Keep one model for serial fallback (no extra memory)
        self._serial_model = init_model
        if self.verbose:
            print(f"[NL-LSMCMC-v1] Ensemble: {Nf} members (flat arrays, no model copies)")

        nc = self.ncells
        t_freq = self.assim_timesteps
        rmse_vel_all = np.full(self.nassim, np.nan)
        rmse_sst_all = np.full(self.nassim, np.nan)
        rmse_ssh_all = np.full(self.nassim, np.nan)

        # Seed RNG for reproducible ensemble noise
        np.random.seed(42)

        # ---- Main assimilation loop ----
        for cycle in range(self.nassim):
            t0_wall = time.time()

            # Advance ensemble
            self._advance_ensemble(
                member_times, t_freq, add_noise=(self.sig_x > 0))
            forecast_mean = np.mean(self.forecast, axis=1)

            # Ensemble spread diagnostic
            if self.verbose and ((cycle + 1) % 50 == 0 or cycle == 0):
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

            # Per-obs noise
            if isinstance(sig_y, np.ndarray) and sig_y.ndim == 2:
                sig_y_cycle = sig_y[cycle][valid]
            elif isinstance(sig_y, np.ndarray) and sig_y.ndim == 1:
                sig_y_cycle = (sig_y[valid] if sig_y.size > 1
                               else float(sig_y))
            else:
                sig_y_cycle = float(sig_y)

            obs_ind_ml = self._obs_ind_to_layer0(ind_valid)
            obs_ind_cell = self._obs_ind_to_cell(ind_valid)

            # Block-partition localization
            loc_cells = self.get_observed_blocks_cells(obs_ind_cell)
            if loc_cells.size == 0:
                self.lsmcmc_mean[cycle + 1] = forecast_mean
                continue

            # Local observation operator
            Ho = build_H_loc_from_global(obs_ind_ml, loc_cells)

            # Per-obs σ_y (keep only mapped obs)
            if (isinstance(sig_y_cycle, np.ndarray)
                    and sig_y_cycle.size == y_valid.size):
                col_set = set(loc_cells.tolist())
                kept = np.array(
                    [int(g) in col_set for g in obs_ind_ml])
                if kept.any():
                    sig_y_vec = sig_y_cycle[kept]
                    y_loc = y_valid[kept]
                else:
                    sig_y_vec = sig_y_cycle
                    y_loc = y_valid
            else:
                s = (float(sig_y_cycle)
                     if np.isscalar(sig_y_cycle)
                     else float(sig_y_cycle.item()))
                sig_y_vec = np.full(len(y_valid), s)
                y_loc = y_valid

            sig_x_loc = self.sig_x_vec[loc_cells]

            # ============ MCMC sampling ============
            try:
                fc_local = self.forecast[loc_cells, :]  # (d, Nf)
                samples, acc_rate, final_step = self._run_mcmc(
                    fc_local, y_loc, Ho, sig_y_vec, sig_x_loc,
                    self.obs_operator)

                # Reduce MCMC samples → Nf analysis members
                mu = np.mean(samples, axis=1)
                anal_ens_loc, _ = gaussian_block_means(
                    samples, self.nforecast)
            except Exception as e:
                warnings.warn(
                    f"Cycle {cycle}: MCMC failed: {e}")
                self.lsmcmc_mean[cycle + 1] = forecast_mean
                continue

            # Update state
            self.lsmcmc_mean[cycle + 1] = forecast_mean.copy()
            self.lsmcmc_mean[cycle + 1, loc_cells] = mu
            self.forecast[loc_cells, :] = anal_ens_loc

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
                        h_j - anal_relax_frac * (eta_j - eta_ref))

            # (No model sync needed — state tracked in self.forecast)

            # RMSE — against truth if available, else state-space
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
                    # velocity: u0 and v0
                    vel_diff = diff[nc:3*nc]
                    rmse_vel_all[cycle] = np.sqrt(np.mean(vel_diff**2))
                    # SST: T0
                    sst_diff = diff[3*nc:4*nc]
                    rmse_sst_all[cycle] = np.sqrt(np.mean(sst_diff**2))
                    # SSH: h_total - H_b
                    ssh_diff = diff[0:nc]
                    rmse_ssh_all[cycle] = np.sqrt(np.mean(ssh_diff**2))
            else:
                H_z = z_a[obs_ind_ml]          # state at obs locations
                residuals = H_z - y_valid      # state-space innovation
                vel_mask = (obs_ind_ml >= nc) & (obs_ind_ml < 3*nc)
                sst_mask = ((obs_ind_ml >= 3*nc)
                            & (obs_ind_ml < 4*nc))
                ssh_mask = (obs_ind_ml >= 0) & (obs_ind_ml < nc)
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
                # MCMC sample diagnostics
                _mu_loc = mu  # MCMC posterior mean on loc_cells
                _h_cells = [c for c in loc_cells if c < nc]
                _v_cells = [c for c in loc_cells
                            if nc <= c < 3*nc]
                _t_cells = [c for c in loc_cells
                            if 3*nc <= c < 4*nc]
                diag_parts = []
                if _h_cells:
                    _h_vals = z_a[_h_cells]
                    diag_parts.append(
                        f"h=[{_h_vals.min():.1f},{_h_vals.max():.1f}]")
                if _v_cells:
                    _v_vals = z_a[_v_cells]
                    diag_parts.append(
                        f"uv=[{_v_vals.min():.3f},{_v_vals.max():.3f}]")
                if _t_cells:
                    _t_vals = z_a[_t_cells]
                    diag_parts.append(
                        f"T=[{_t_vals.min():.2f},{_t_vals.max():.2f}]")
                diag_str = "  ".join(diag_parts)
                if self.verbose:
                    print(
                        f"  Cycle {cycle+1}/{self.nassim}  "
                        f"vel_RMSE={rmse_vel_all[cycle]:.6f}  "
                        f"sst_RMSE={rmse_sst_all[cycle]:.4f}  "
                        f"ssh_RMSE={rmse_ssh_all[cycle]:.4f}  "
                        f"SSH=[{_ssh.min():.1f},{_ssh.max():.1f}]  "
                        f"acc_rate={acc_rate:.3f}  "
                        f"step={final_step:.4f}  "
                        f"({elapsed:.1f}s)")
                    if diag_str:
                        print(f"    [state] {diag_str}")

        # Cleanup
        if self._mcmc_pool is not None:
            self._mcmc_pool.close()
            self._mcmc_pool.join()
            self._mcmc_pool = None
            if self.verbose:
                print("[NL-LSMCMC-v1] MCMC pool closed.")
        if self._mp_pool is not None:
            self._mp_pool.close()
            self._mp_pool.join()
            self._mp_pool = None
            if self.verbose:
                print("[NL-LSMCMC-v1] Forecast MP pool closed.")

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
            self.nassim + 1, self.nlayers, self.fields_per_layer,
            self.ny, self.nx)
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
                    ot = np.concatenate([[obs_times[0]], ot])
                ot = ot[:self.nassim + 1]
            vt = ds.createVariable('obs_times', 'f8', ('time',))
            vt[:] = ot

        ds.nlayers = self.nlayers
        ds.fields_per_layer = self.fields_per_layer
        ds.ny = self.ny
        ds.nx = self.nx
        ds.close()
        print(f"[NL-LSMCMC-v1] Saved results to {outfile}")

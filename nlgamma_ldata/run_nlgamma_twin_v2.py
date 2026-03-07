#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_nlgamma_twin_v2.py
=======================
NL twin experiment — V2: Per-block halo-localized MCMC with
the **correct Student-t likelihood**.

Both V1 and V2 use the correct Student-t likelihood.  The difference is
purely algorithmic:
    * **V1**: P parallel MCMC chains over the **full domain**, each
      producing N_a/P samples (same burn-in for all chains).
    * **V2**: Loop over all partition blocks in parallel, run an
      independent MCMC chain **per block** with halo localization.

Student-t noise model
---------------------
    ε ~ σ · t(ν)

    p(ε) = Γ((ν+1)/2) / (Γ(ν/2) √(νπ) σ) * (1 + ε²/(νσ²))^{-(ν+1)/2}

    log p(ε) = const - ((ν+1)/2) * log(1 + ε²/(νσ²))

    where  ε = y − h(Hz)  and  h(x) = sign(x)x².

The V2 filter uses ``NL_SMCMC_MLSWE_Filter_V2`` (halo-localized MCMC)
and monkey-patches ``_block_mcmc_worker`` to inject the cubic
observation operator and Student-t log-likelihood.

Usage
-----
    python3 -u nlgamma_ldata/run_nlgamma_twin_v2.py [config.yml]

Default config: ``nlgamma_ldata/example_input_nlgamma_twin_v2.yml``
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# ---- Reuse everything from the V1 twin runner ----
import nlgamma_ldata.run_nlgamma_twin as _v1_twin

# ---- V2 filter from the main codebase ----
from mlswe.lsmcmc_nl_V2 import NL_SMCMC_MLSWE_Filter_V2
import mlswe.lsmcmc_nl_V2 as _nl_loc_module

# Obs operator (same as V1)
obs_operator_pow32 = _v1_twin.obs_operator_pow32


# =====================================================================
#  Student-t-likelihood block MCMC worker
# =====================================================================
def _block_mcmc_worker_studentt(args):
    """
    MCMC on a single block — identical to the standard worker EXCEPT
    _log_lik uses the Student-t density instead of Gaussian.

    The Student-t degrees of freedom (ν) are passed via the
    ``obs_op_name`` field as ``'cubic_studentt:<nu>'``.
    """
    from scipy import sparse as sp
    from mlswe.lsmcmc_nl_V2 import rescaled_block_means

    (block_idx, fc_halo, y_local, H_loc_tuple, sigma_y_loc,
     sig_x_halo, block_cells, block_in_halo, prior_sprd,
     n_samples, burn_in, step_size, adapt, adapt_interval,
     target_acc, thin, kernel, obs_op_name,
     rtps_alpha_val, Nf_out, shared_perm_seed,
     pcn_beta, hmc_leapfrog_steps, seed) = args

    rng = np.random.default_rng(seed)

    # Reconstruct sparse H_loc
    data, indices, indptr, shape = H_loc_tuple
    H_loc = sp.csr_matrix((data, indices, indptr), shape=shape)

    # Parse Student-t nu from obs_op_name  ('cubic_studentt:2.0')
    nu = 2.0
    if obs_op_name is not None and 'studentt:' in obs_op_name:
        nu = float(obs_op_name.split(':')[1])

    sig_y_sq = sigma_y_loc ** 2
    half_nup1 = 0.5 * (nu + 1.0)

    # Obs operator: h(x) = arctan(x)
    def _h(z):
        Hz = H_loc @ z
        return np.arctan(Hz)

    Nf = fc_halo.shape[1]
    dim = fc_halo.shape[0]

    nz_mask = sig_x_halo > 0
    n_nz = int(nz_mask.sum())
    if n_nz == 0:
        mu = np.mean(fc_halo, axis=1)
        samples_block = np.tile(mu, (n_samples, 1)).T
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

    # ---- Student-t log-likelihood ----
    def _log_lik(z):
        y_pred = _h(z)
        eps = y_local - y_pred                    # residual ε
        ll = np.sum(-half_nup1 * np.log(1.0 + eps ** 2 / (nu * sig_y_sq)))
        return ll

    # ---- Log-transition (Gaussian prior, unchanged) ----
    def _log_trans(z, m_i):
        diff = z[nz_mask] - m_i[nz_mask]
        return -0.5 * np.sum((diff * inv_sig_x_nz) ** 2)

    def _log_trans_all(z):
        diff = z[nz_mask, None] - fc_halo[nz_mask, :]
        return -0.5 * np.sum(
            (diff * inv_sig_x_nz[:, None]) ** 2, axis=0)

    # ---- Gradient of log target (for MALA / HMC) ----
    def _grad_log_target(z, m_i):
        Hz_raw = H_loc @ z
        pred = np.arctan(Hz_raw)
        dpred = 1.0 / (1.0 + Hz_raw ** 2)  # d/dx [arctan(x)] = 1/(1+x²)
        eps = y_local - pred
        dll_deps = -(nu + 1.0) * eps / (nu * sig_y_sq + eps ** 2)
        dll_dpred = -dll_deps
        grad = np.asarray(H_loc.T @ (dpred * dll_dpred)).ravel()
        grad[nz_mask] -= (z[nz_mask] - m_i[nz_mask]) / sig_x_sq_nz
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
    _log_beta = np.log(max(beta, 1e-4))  # Robbins-Monro state

    for s in range(total_iters):
        # --- Gibbs step for i ---
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

        # --- Proposal step for z ---
        _n_acc_before = n_accept
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

        # --- Adapt: Robbins-Monro for pCN, block for others ---
        if adapt and s > 0:
            _accepted_this = (n_accept > _n_acc_before)
            if kernel == 'pcn':
                _rm_g = 0.5 / (1.0 + s) ** 0.6
                _log_beta += _rm_g * (float(_accepted_this) - target_acc)
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

    block_mean = np.mean(samples_block, axis=1)
    np.random.seed(shared_perm_seed)
    anal_block = rescaled_block_means(
        samples_block, Nf_out, block_mean)

    # RTPS inflation
    if rtps_alpha_val > 0:
        amean = np.mean(anal_block, axis=1, keepdims=True)
        asprd = np.std(anal_block, axis=1)
        safe = np.maximum(asprd, 1e-30)
        infl = 1.0 + rtps_alpha_val * (prior_sprd - asprd) / safe
        infl = np.maximum(infl, 1.0)
        anal_block = amean + infl[:, None] * (anal_block - amean)

    return {
        'block_cells': block_cells,
        'anal_block': anal_block,
        'mu_block': block_mean,
        'acc_rate': acc_rate,
    }


# =====================================================================
#  V2 filter subclass
# =====================================================================
class _TwinFilterV2(NL_SMCMC_MLSWE_Filter_V2):
    """V2 filter with h(x)=sign(x)x² obs-operator and Student-t likelihood."""

    _truth_state = None

    def __init__(self, isim, params):
        # Encode student_t_nu into obs_op_name so it reaches the worker
        nu = float(params.get('student_t_nu', 2.0))
        params = dict(params)
        params['obs_operator_name'] = f'cubic_studentt:{nu}'
        super().__init__(isim, params,
                         obs_operator=obs_operator_pow32)


def main():
    config_file = (sys.argv[1] if len(sys.argv) > 1
                   else os.path.join(_HERE, 'example_input_nlgamma_twin_v2.yml'))

    # Monkey-patch the V1 twin module to use our V2 filter
    _v1_twin._TwinFilter = _TwinFilterV2
    _v1_twin._OrigFilter = NL_SMCMC_MLSWE_Filter_V2

    # Monkey-patch the block worker in lsmcmc_nl_V2 to use Student-t lik
    _nl_loc_module._block_mcmc_worker = _block_mcmc_worker_studentt

    if len(sys.argv) <= 1:
        sys.argv.append(config_file)
    else:
        sys.argv[1] = config_file

    _v1_twin.main()


if __name__ == '__main__':
    main()

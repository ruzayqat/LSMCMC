#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_nlgamma_twin.py
====================
Synthetic twin experiment with:
    * **Nonlinear observation operator:**  h(x) = sign(x) x²
    * **Student-t observation noise:**  ε ~ t(ν, σ)  with ν = 2
      (heavy tails, infinite variance)

The forward (process) model remains Gaussian.

Workflow
--------
1. Run a "nature run" model forward (no noise, no DA).
2. At each cycle, generate synthetic observations:
       y = h(H · z_truth) + ε,     ε ~ σ · t(ν)
   where h(x) = sign(x)x² is a nonlinear operator
   and the noise follows a Student-t distribution.
3. Run the NL-LSMCMC V1 filter (P parallel MCMC chains over the full
   domain, each producing N_a/P samples, same burn-in for all) with
   the **correct Student-t likelihood** inside the MCMC.
4. Compute RMSE against the nature run state.

The V2 variant (``run_nlgamma_twin_v2.py``) uses the same Student-t
likelihood but with per-block halo-localized MCMC instead of
full-domain parallel chains.

Usage
-----
    python3 -u nlgamma_ldata/run_nlgamma_twin.py [config.yml]

Default config: ``nlgamma_ldata/input_nlgamma_twin.yml``
"""
import os
import sys
import time
import yaml
import warnings
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SWE_DIR = os.path.join(_ROOT, '..', 'SWE_LSMCMC')
if os.path.isdir(_SWE_DIR):
    sys.path.insert(0, os.path.abspath(_SWE_DIR))

from netCDF4 import Dataset
from mlswe.model import MLSWE, coriolis_array
from mlswe.lsmcmc_nl_V1 import NL_SMCMC_MLSWE_Filter
import mlswe.lsmcmc_nl_V1 as _nl_module
import run_mlswe_lsmcmc_ldata_V1          # reuse data helpers

# Module-level Student-t degrees of freedom (set in main, read by workers)
_STUDENT_T_NU = 2.0


# =====================================================================
#  Nonlinear observation operator:  h(x) = arctan(x)
# =====================================================================
def obs_operator_pow32(z_local, H_loc, obs_ind_local):
    """y_pred = arctan(Hz)."""
    Hz = H_loc @ z_local
    return np.arctan(Hz)


# =====================================================================
#  Parallel MCMC chain worker with arctan + Student-t log-likelihood
# =====================================================================
def _mcmc_chain_worker_studentt(args):
    """
    MCMC chain worker for V1 (P parallel chains over full domain).

    Same structure as ``mlswe.lsmcmc_nl_V1._mcmc_chain_worker`` but with:
        * obs operator:  h(x) = sign(x) x²
        * Student-t log-likelihood instead of Gaussian

    Uses the module-level ``_STUDENT_T_NU`` for the degrees of freedom.
    """
    from scipy import sparse as sp
    from scipy.special import gammaln

    (fc_local, y, H_loc_tuple, sig_y_vec, sig_x_loc,
     n_samples, burn_in, step_size, adapt, adapt_interval,
     target_acc, thin, kernel, obs_op_name,
     pcn_beta, hmc_leapfrog_steps, seed) = args

    rng = np.random.RandomState(seed)

    # Reconstruct sparse H_loc
    data, indices, indptr, shape = H_loc_tuple
    H_loc = sp.csr_matrix((data, indices, indptr), shape=shape)

    # Student-t parameters
    nu = _STUDENT_T_NU
    sig_y_sq = sig_y_vec ** 2
    # Pre-compute constant:  log C = logΓ((ν+1)/2) - logΓ(ν/2)
    #                                - 0.5*log(νπ) - log(σ)
    # We only need the ε-dependent part for MH ratios:
    #   ll(ε) = -((ν+1)/2) * log(1 + ε²/(νσ²))
    half_nup1 = 0.5 * (nu + 1.0)

    # Obs operator: h(x) = arctan(x)
    def _h(z):
        Hz = H_loc @ z
        return np.arctan(Hz)

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

    # ---- Student-t log-likelihood ----
    def _log_lik(z):
        y_pred = _h(z)
        eps = y - y_pred
        ll = np.sum(-half_nup1 * np.log(1.0 + eps ** 2 / (nu * sig_y_sq)))
        return ll

    # ---- Log-transition (Gaussian prior) ----
    def _log_trans(z, m_i):
        diff = z[nz_mask] - m_i[nz_mask]
        return -0.5 * np.sum((diff * inv_sig_x_nz) ** 2)

    def _log_trans_all(z):
        diff = z[nz_mask, None] - fc_local[nz_mask, :]
        return -0.5 * np.sum(
            (diff * inv_sig_x_nz[:, None]) ** 2, axis=0)

    # ---- Gradient of log target (for MALA / HMC) ----
    def _grad_log_target(z, m_i):
        Hz_raw = H_loc @ z
        pred = np.arctan(Hz_raw)
        # d/dx [arctan(x)] = 1/(1+x²)
        dpred = 1.0 / (1.0 + Hz_raw ** 2)

        eps = y - pred
        # d log_lik / d eps = -(nu+1) * eps / (nu*sig_y_sq + eps^2)
        dll_deps = -(nu + 1.0) * eps / (nu * sig_y_sq + eps ** 2)
        # d/d_pred = -d/d_eps
        dll_dpred = -dll_deps

        grad = np.asarray(
            H_loc.T @ (dpred * dll_dpred)
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
    _log_beta = np.log(max(beta, 1e-4))  # Robbins-Monro state

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
            # Mass matrix M = diag(1/sig_x^2), M^{-1} = diag(sig_x^2)
            # p ~ N(0, M);  KE = 0.5 p^T M^{-1} p
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
    return samples, n_accept, step_size


# =====================================================================
#  Load SST / SSH reference fields
# =====================================================================
def _load_sst_ssh_refs(params, bc_file, lon, lat, ny, nx, obs_times):
    """Populate params with sst_nudging_ref, ssh_relax_ref, etc."""
    import glob
    from datetime import datetime
    data_dir = params.get('data_dir', './data')

    sst_rate = params.get('sst_nudging_rate', 0.0)
    if sst_rate > 0:
        grid_tag = f'{ny}x{nx}'
        sst_files = sorted(glob.glob(os.path.join(
            data_dir, f'hycom_sst_ref_*_{grid_tag}_3d.npy')))
        sst_time_files = sorted(glob.glob(os.path.join(
            data_dir, f'hycom_sst_ref_*_{grid_tag}_times.npy')))
        if not sst_files:
            sst_files = sorted(glob.glob(os.path.join(
                data_dir, 'hycom_sst_ref_*_3d.npy')))
            sst_time_files = sorted(glob.glob(os.path.join(
                data_dir, 'hycom_sst_ref_*_times.npy')))
        if sst_files:
            sst_ref = np.load(sst_files[-1])
            sst_ref_times_raw = np.load(sst_time_files[-1])
            t0_str = params.get('obs_time_start', '2024-08-01T00:00:00')
            t0_dt = datetime.strptime(t0_str[:19], '%Y-%m-%dT%H:%M:%S')
            epoch = (t0_dt - datetime(1970, 1, 1)).total_seconds()
            sst_ref_times = sst_ref_times_raw + epoch
            params['sst_nudging_ref'] = sst_ref
            params['sst_nudging_ref_times'] = sst_ref_times
            print(f"[nlg-twin] SST nudging: lam={sst_rate:.6f} s^-1, "
                  f"ref shape={sst_ref.shape}")
        else:
            print("[nlg-twin] WARNING: SST nudging enabled but no ref found")

    ssh_rate = float(params.get('ssh_relax_rate', 0.0))
    if ssh_rate > 0:
        try:
            from scipy.interpolate import RegularGridInterpolator
            with Dataset(bc_file, 'r') as nc:
                bc_lat = np.asarray(nc.variables['lat'][:])
                bc_lon = np.asarray(nc.variables['lon'][:])
                bc_times = np.asarray(nc.variables['time'][:])
                bc_ssh = np.asarray(nc.variables['ssh'][:])
            bc_ssh[np.isnan(bc_ssh)] = 0.0
            mg_lat, mg_lon = np.meshgrid(lat, lon, indexing='ij')
            ssh_ref_3d = np.zeros((len(bc_times), ny, nx))
            for ti in range(len(bc_times)):
                interp = RegularGridInterpolator(
                    (bc_lat, bc_lon), bc_ssh[ti],
                    method='linear', bounds_error=False,
                    fill_value=0.0)
                ssh_ref_3d[ti] = interp((mg_lat, mg_lon))
            params['ssh_relax_ref'] = ssh_ref_3d
            params['ssh_relax_ref_times'] = bc_times
            print(f"[nlg-twin] SSH relaxation: rate={ssh_rate:.2e}, "
                  f"ref shape={ssh_ref_3d.shape}")
        except Exception as e:
            print(f"[nlg-twin] WARNING: SSH ref load failed: {e}")

    if params.get('sst_flux_type') is not None:
        if 'sst_nudging_ref' in params:
            params['sst_T_air'] = params['sst_nudging_ref']
            params['sst_T_air_times'] = params['sst_nudging_ref_times']


# =====================================================================
#  Wrapper that injects the obs operator AND nature run state
# =====================================================================
_OrigFilter = NL_SMCMC_MLSWE_Filter


class _TwinFilter(_OrigFilter):
    """NL filter with h(x)=sign(x)x² obs-operator and Student-t likelihood.
    V1: P parallel MCMC chains over the full domain.
    """

    _truth_state = None

    def __init__(self, isim, params):
        super().__init__(isim, params, obs_operator=obs_operator_pow32)


# =====================================================================
#  Generate nature run trajectory + synthetic observations
# =====================================================================
def generate_truth_and_obs(params, H_b, bc_handler, tstart,
                           real_obs_file, outdir, rng_seed=42):
    """
    Run a single "nature run" model forward and produce synthetic obs
    with the nonlinear obs operator + Student-t noise.

    Student-t noise parameters are read from params:
        student_t_nu (ν)  — degrees of freedom  (default 2.0)

    The noise is: ε = σ · t(ν), so E[ε] = 0 for ν>1.
    For ν ≤ 2, variance is infinite (heavy tails).

    Returns
    -------
    truth_states : (nassim+1, dimx)
    synth_obs_file : str
    """
    nassim = int(params['nassim'])
    t_freq = int(params.get('t_freq',
                             params.get('assim_timesteps', 48)))

    # Student-t noise parameters
    student_t_nu = float(params.get('student_t_nu', 2.0))

    # --- Load real obs for locations ---
    nc_real = Dataset(real_obs_file, 'r')
    real_yind = np.asarray(nc_real.variables['yobs_ind_all'][:])
    obs_times = np.asarray(nc_real.variables['obs_times'][:])
    sig_y_scalar = float(nc_real.sig_y) if hasattr(nc_real, 'sig_y') else 0.1
    nc_real.close()

    n_cycles = min(nassim, real_yind.shape[0])
    max_nobs_drifter = real_yind.shape[1]

    nx = int(params['dgx'])
    ny = int(params['dgy'])
    nc = nx * ny

    # SSH obs locations from linear merged file
    linear_merged = params.get(
        'linear_merged_obs',
        './output_lsmcmc_ldata_V1/mlswe_merged_obs.nc')
    nc_lin = Dataset(linear_merged, 'r')
    lin_yind = np.asarray(nc_lin.variables['yobs_ind_all'][:])
    nc_lin.close()

    n_lin_cycles = min(n_cycles, lin_yind.shape[0])
    ssh_ind_per_cycle = []
    for c in range(n_lin_cycles):
        inds_c = lin_yind[c]
        valid_c = inds_c[(inds_c >= 0) & (inds_c < nc)]
        ssh_ind_per_cycle.append(valid_c.astype(int))
    while len(ssh_ind_per_cycle) < n_cycles:
        ssh_ind_per_cycle.append(ssh_ind_per_cycle[-1].copy())

    max_ssh_obs = max(len(s) for s in ssh_ind_per_cycle)
    max_nobs = max_nobs_drifter + max_ssh_obs
    print(f"[nlg-twin] SSH obs from: {linear_merged}")
    print(f"[nlg-twin] max_nobs={max_nobs} "
          f"(drifter max={max_nobs_drifter}, ssh max={max_ssh_obs})")

    # ICs
    if 'ic_h0' in params and params['ic_h0'] is not None:
        h0 = [np.array(hk, dtype=np.float64) for hk in params['ic_h0']]
        u0 = [np.array(uk, dtype=np.float64) for uk in params['ic_u0']]
        v0 = [np.array(vk, dtype=np.float64) for vk in params['ic_v0']]
        T0 = [np.array(Tk, dtype=np.float64) for Tk in params['ic_T0']]
    else:
        raise RuntimeError("ICs must be set before calling "
                           "generate_truth_and_obs")

    lon = np.linspace(params['lon_min'], params['lon_max'], nx)
    lat = np.linspace(params['lat_min'], params['lat_max'], ny)
    lat_centre = 0.5 * (params['lat_min'] + params['lat_max'])
    dx_m = np.deg2rad(abs(lon[1] - lon[0])) * 6.371e6 * np.cos(
        np.deg2rad(lat_centre))
    dy_m = np.deg2rad(abs(lat[1] - lat[0])) * 6.371e6
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

    truth_mdl = MLSWE(
        [hk.copy() for hk in h0],
        [uk.copy() for uk in u0],
        [vk.copy() for vk in v0],
        T0=[Tk.copy() for Tk in T0],
        **model_kw)
    truth_mdl.timesteps = 1

    dimx = truth_mdl.state_flat.size
    truth_states = np.zeros((n_cycles + 1, dimx))
    truth_states[0] = truth_mdl.state_flat.copy()

    # --- Forward-run nature run & build synthetic obs ---
    rng = np.random.default_rng(rng_seed)
    synth_yobs = np.full((n_cycles, max_nobs), np.nan)
    synth_yind = np.full((n_cycles, max_nobs), -1, dtype=np.int32)
    synth_sigy = np.full((n_cycles, max_nobs), np.nan)

    sig_y_uv  = float(params.get('sig_y_uv',  0.10))
    sig_y_sst = float(params.get('sig_y_sst', 0.40))
    sig_y_ssh = float(params.get('sig_y_ssh', 0.50))

    print(f"[nlg-twin] Obs operator: h(x) = arctan(x)")
    print(f"[nlg-twin] Obs noise: Student-t(ν={student_t_nu}, σ=sig_y)")
    print(f"[nlg-twin] sig_y: uv={sig_y_uv}, sst={sig_y_sst}, "
          f"ssh={sig_y_ssh}")
    print(f"[nlg-twin] Generating nature run ({n_cycles} cycles, "
          f"t_freq={t_freq}) ...")
    t0_gen = time.time()

    for cycle in range(n_cycles):
        for _ in range(t_freq):
            truth_mdl._timestep()
            if not np.all(np.isfinite(truth_mdl.state_flat)):
                warnings.warn(f"[nlg-twin] Nature run blew up at cycle "
                              f"{cycle+1}")
                truth_mdl.state_flat = truth_states[cycle].copy()
                break
        truth_states[cycle + 1] = truth_mdl.state_flat.copy()

        # --- Drifter obs (UV + SST) ---
        ind_raw = real_yind[cycle]
        valid = ind_raw >= 0
        drifter_ind = (ind_raw[valid].astype(int) if valid.any()
                       else np.array([], dtype=int))

        # --- SSH obs from linear merged file ---
        ssh_ind = ssh_ind_per_cycle[cycle]

        obs_ind = np.concatenate([drifter_ind, ssh_ind])
        nv = len(obs_ind)
        if nv == 0:
            continue

        z_truth = truth_states[cycle + 1]
        z_at_obs = z_truth[obs_ind]

        # Apply nonlinear operator: h(x) = arctan(x)
        y_clean = np.arctan(z_at_obs)

        # Per-obs noise σ (by variable type)
        sig_obs = np.empty(nv)
        for i_obs in range(nv):
            idx = obs_ind[i_obs]
            if idx < nc:          # SSH
                sig_obs[i_obs] = sig_y_ssh
            elif idx < 3 * nc:    # u or v
                sig_obs[i_obs] = sig_y_uv
            elif idx < 4 * nc:    # SST
                sig_obs[i_obs] = sig_y_sst
            else:
                sig_obs[i_obs] = sig_y_scalar

        # Student-t noise: ε = σ · t(ν)
        noise = sig_obs * rng.standard_t(df=student_t_nu, size=nv)

        y_noisy = y_clean + noise

        synth_yobs[cycle, :nv] = y_noisy
        synth_yind[cycle, :nv] = obs_ind
        synth_sigy[cycle, :nv] = sig_obs

        if (cycle + 1) % 50 == 0 or cycle == 0:
            n_out = np.sum(np.abs(noise) > 2 * sig_obs)
            _ssh_truth = z_truth[:nc] - H_b.ravel()
            print(f"  [nlg-twin] cycle {cycle+1}/{n_cycles}  "
                  f"nobs={nv} (|ε|>2σ: {n_out}/{nv})  "
                  f"y_clean=[{y_clean.min():.4f},{y_clean.max():.4f}]  "
                  f"SSH_nature_run=[{_ssh_truth.min():.2f},"
                  f"{_ssh_truth.max():.2f}]")

    elapsed_gen = time.time() - t0_gen
    print(f"[nlg-twin] Nature run generated in {elapsed_gen:.1f}s")

    # --- Write synthetic obs NetCDF ---
    os.makedirs(outdir, exist_ok=True)
    synth_obs_file = os.path.join(outdir, 'synthetic_nlgamma_obs.nc')
    ds = Dataset(synth_obs_file, 'w', format='NETCDF4')
    ds.createDimension('n_cycles', n_cycles)
    ds.createDimension('max_nobs', max_nobs)

    vy = ds.createVariable('yobs_all', 'f8', ('n_cycles', 'max_nobs'))
    vy[:] = synth_yobs
    vi = ds.createVariable('yobs_ind_all', 'i4', ('n_cycles', 'max_nobs'))
    vi[:] = synth_yind
    vi0 = ds.createVariable('yobs_ind_level0_all', 'i4',
                            ('n_cycles', 'max_nobs'))
    vi0[:] = synth_yind
    vs = ds.createVariable('sig_y_all', 'f8', ('n_cycles', 'max_nobs'))
    vs[:] = synth_sigy
    vt = ds.createVariable('obs_times', 'f8', ('n_cycles',))
    vt[:] = obs_times[:n_cycles]
    ds.sig_y = float(sig_y_scalar)
    ds.obs_operator = 'cubic'
    ds.student_t_nu = student_t_nu
    ds.noise_type = 'student_t'
    ds.close()
    print(f"[nlg-twin] Wrote synthetic obs: {synth_obs_file}  "
          f"({n_cycles} cycles, max_nobs={max_nobs})")

    return truth_states, synth_obs_file


# =====================================================================
#  Save nature run trajectory
# =====================================================================
def _save_truth(outdir, truth_states, H_b, ny, nx):
    """Save nature run trajectory to NetCDF."""
    outfile = os.path.join(outdir, 'truth_trajectory.nc')
    ds = Dataset(outfile, 'w', format='NETCDF4')
    nt, dimx = truth_states.shape
    nlayers = 3
    fields_per_layer = 4
    ds.createDimension('time', nt)
    ds.createDimension('layer', nlayers)
    ds.createDimension('field', fields_per_layer)
    ds.createDimension('y', ny)
    ds.createDimension('x', nx)
    v = ds.createVariable('truth', 'f4',
                          ('time', 'layer', 'field', 'y', 'x'),
                          zlib=True)
    reshaped = truth_states.reshape(nt, nlayers, fields_per_layer, ny, nx)
    v[:] = reshaped.astype(np.float32)
    if H_b is not None:
        vb = ds.createVariable('H_b', 'f4', ('y', 'x'), zlib=True)
        vb[:] = H_b.astype(np.float32)
    ds.close()
    print(f"[nlg-twin] Saved nature run: {outfile}")


# =====================================================================
#  Worker for M-run parallel execution
# =====================================================================
def _twin_worker(args):
    """Run one independent filter simulation for M-run averaging."""
    isim, seed = args
    params = dict(_g_twin_params)
    params['ncores'] = _g_twin_ncores_per_worker
    params['random_seed'] = seed
    params['verbose'] = False
    np.random.seed(seed)

    filt = _TwinFilter(isim, params)
    filt._truth_state = _g_twin_truth_states
    t0 = time.time()
    filt.run(_g_twin_H_b, _g_twin_bc_handler,
             _g_twin_obs_file, _g_twin_tstart)
    elapsed = time.time() - t0

    print(f"  [done] run {isim+1}  seed={seed}  elapsed={elapsed:.1f}s  "
          f"vel={np.nanmean(filt.rmse_vel):.6f}  "
          f"sst={np.nanmean(filt.rmse_sst):.4f}  "
          f"ssh={np.nanmean(filt.rmse_ssh):.4f}", flush=True)

    return (isim, filt.lsmcmc_mean.copy(),
            filt.rmse_vel.copy(), filt.rmse_sst.copy(),
            filt.rmse_ssh.copy(), elapsed)


# =====================================================================
#  Main
# =====================================================================
def main():
    config_file = (sys.argv[1] if len(sys.argv) > 1
                   else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'input_nlgamma_twin.yml'))
    with open(config_file) as f:
        params = yaml.safe_load(f)

    print("=" * 64)
    print("  NL Twin Experiment — cubic obs + Student-t noise")
    print("  V1: correct Student-t likelihood, P parallel MCMC chains")
    print("=" * 64)

    nx = params['dgx']
    ny = params['dgy']
    lon = np.linspace(params['lon_min'], params['lon_max'], nx)
    lat = np.linspace(params['lat_min'], params['lat_max'], ny)

    H_b = run_mlswe_lsmcmc_ldata_V1.load_bathymetry(params, ny, nx, lon, lat)

    bc_file = params.get('bc_file', './data/hycom_bc_2024aug.nc')
    if not os.path.exists(bc_file):
        bc_file = os.path.join(_ROOT, 'data', 'hycom_bc_2024aug.nc')
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
    print(f"Real obs file (for locations): {obs_file}")

    with Dataset(obs_file, 'r') as nc_obj:
        obs_times = np.asarray(nc_obj.variables['obs_times'][:])
    tstart = obs_times[0]

    _load_sst_ssh_refs(params, bc_file, lon, lat, ny, nx, obs_times)

    lat_centre = 0.5 * (params['lat_min'] + params['lat_max'])
    dx_m = np.deg2rad(abs(lon[1] - lon[0])) * 6.371e6 * np.cos(
        np.deg2rad(lat_centre))
    dy_m = np.deg2rad(abs(lat[1] - lat[0])) * 6.371e6
    h0, u0, v0, T0 = run_mlswe_lsmcmc_ldata_V1.init_from_bc_handler(
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

    outdir = params.get('lsmcmc_dir', './output_nlgamma_twin_V1')
    os.makedirs(outdir, exist_ok=True)

    # ---- Generate nature run + synthetic obs (or load existing) ----
    force_regen = '--force-regen' in sys.argv
    truth_file = os.path.join(outdir, 'truth_trajectory.nc')
    synth_obs_file = os.path.join(outdir, 'synthetic_nlgamma_obs.nc')

    if (not force_regen
            and os.path.isfile(truth_file)
            and os.path.isfile(synth_obs_file)):
        print(f"[nlg-twin] Reusing existing nature run: {truth_file}")
        ds_t = Dataset(truth_file, 'r')
        truth_arr = np.asarray(ds_t.variables['truth'][:],
                               dtype=np.float64)
        ds_t.close()
        truth_states = truth_arr.reshape(truth_arr.shape[0], -1)
        print(f"[nlg-twin] Reusing existing synthetic obs: "
              f"{synth_obs_file}")
        print(f"[nlg-twin]   truth_states shape = {truth_states.shape}")
    else:
        if force_regen:
            print("[nlg-twin] --force-regen: regenerating nature run + obs")
        truth_states, synth_obs_file = generate_truth_and_obs(
            params, H_b, bc_handler, tstart,
            obs_file, outdir, rng_seed=42)
        _save_truth(outdir, truth_states, H_b, ny, nx)

    # ---- Check for --truth-only mode ----
    truth_only = '--truth-only' in sys.argv
    if truth_only:
        print(f"\n{'='*64}")
        print(f"  Nature-run-only complete.")
        print(f"  Nature run: {outdir}/truth_trajectory.nc")
        print(f"  Synthetic obs: {synth_obs_file}")
        print(f"{'='*64}")
        return

    # ---- Monkey-patch the chain worker with Student-t likelihood ----
    global _STUDENT_T_NU
    _STUDENT_T_NU = float(params.get('student_t_nu', 2.0))
    _nl_module._mcmc_chain_worker = _mcmc_chain_worker_studentt

    _TwinFilter._truth_state = truth_states
    run_mlswe_lsmcmc_ldata_V1.Loc_SMCMC_MLSWE_Filter = _TwinFilter

    params['obs_file'] = synth_obs_file
    params['use_swot_ssh'] = False
    params['obs_operator_name'] = 'cubic_studentt'

    print(f"\n{'='*64}")
    print(f"  Running NL-LSMCMC V1 on synthetic cubic+Student-t obs")
    print(f"  (correct Student-t likelihood, P parallel chains)")
    print(f"{'='*64}")

    nassim = int(params['nassim'])
    M = int(params.get('M', 1))
    ncores = int(params.get('ncores', 1))
    workers = int(params.get('workers', ncores))

    if M <= 1:
        # ==== Single run (backward-compatible) ====
        filt = _TwinFilter(0, params)
        filt._truth_state = truth_states

        t_wall = time.time()
        filt.run(H_b, bc_handler, synth_obs_file, tstart)
        elapsed = time.time() - t_wall

        filt.save_results(outdir, obs_times=obs_times, H_b=H_b)

        print(f"\n{'='*64}")
        print(f"  NL twin (cubic+t) V1 complete in {elapsed:.1f}s "
              f"({elapsed/60:.1f} min)")
        print(f"  Mean vel RMSE:  {np.nanmean(filt.rmse_vel):.6f} m/s")
        print(f"  Mean SST RMSE:  {np.nanmean(filt.rmse_sst):.4f} K")
        print(f"  Mean SSH RMSE:  {np.nanmean(filt.rmse_ssh):.4f} m")
        print(f"  Output: {outdir}/")
        print(f"{'='*64}")

    else:
        # ==== M independent runs, Welford-averaged ====
        global _g_twin_params, _g_twin_H_b, _g_twin_bc_handler
        global _g_twin_obs_file, _g_twin_tstart, _g_twin_truth_states
        global _g_twin_ncores_per_worker
        _g_twin_params = params
        _g_twin_H_b = H_b
        _g_twin_bc_handler = bc_handler
        _g_twin_obs_file = synth_obs_file
        _g_twin_tstart = tstart
        _g_twin_truth_states = truth_states

        n_workers = min(workers, M)
        ncores_per_worker = max(1, ncores // n_workers)
        _g_twin_ncores_per_worker = ncores_per_worker
        dimx = 12 * ny * nx

        print(f"\n[M-run] Launching M={M} independent runs on "
              f"{n_workers} workers")
        print(f"[M-run] ncores_per_worker={ncores_per_worker}  "
              f"(total={ncores})")

        avg_mean = np.zeros((nassim + 1, dimx), dtype=np.float64)
        avg_rmse_vel = np.zeros(nassim, dtype=np.float64)
        avg_rmse_sst = np.zeros(nassim, dtype=np.float64)
        avg_rmse_ssh = np.zeros(nassim, dtype=np.float64)
        all_elapsed = []
        count = 0

        worker_args = [(i, 1000 + i) for i in range(M)]

        t_wall = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_twin_worker, wa) for wa in worker_args]
            for fut in futures:
                isim, mean_arr, rv, rs, rh, elapsed_i = fut.result()
                count += 1
                avg_mean += (mean_arr - avg_mean) / count
                avg_rmse_vel += (rv - avg_rmse_vel) / count
                avg_rmse_sst += (rs - avg_rmse_sst) / count
                avg_rmse_ssh += (rh - avg_rmse_ssh) / count
                all_elapsed.append(elapsed_i)

        total_elapsed = time.time() - t_wall

        # Save averaged results
        filt = _TwinFilter.__new__(_TwinFilter)
        filt.nassim = nassim
        filt.nlayers = 3
        filt.fields_per_layer = 4
        filt.ny = ny
        filt.nx = nx
        filt.ncells = ny * nx
        filt.lsmcmc_mean = avg_mean
        filt.rmse_vel = avg_rmse_vel
        filt.rmse_sst = avg_rmse_sst
        filt.rmse_ssh = avg_rmse_ssh
        filt.save_results(outdir, obs_times=obs_times, H_b=H_b)

        print(f"\n{'='*64}")
        print(f"  NL twin (cubic+t) complete: M={M} runs averaged")
        print(f"  Wall time:      {total_elapsed:.1f}s "
              f"({total_elapsed/60:.1f} min)")
        print(f"  Mean run time:  {np.mean(all_elapsed):.1f}s")
        print(f"  Mean vel RMSE:  {np.nanmean(avg_rmse_vel):.6f} m/s")
        print(f"  Mean SST RMSE:  {np.nanmean(avg_rmse_sst):.4f} K")
        print(f"  Mean SSH RMSE:  {np.nanmean(avg_rmse_ssh):.4f} m")
        print(f"  Output: {outdir}/")
        print(f"{'='*64}")


if __name__ == '__main__':
    main()

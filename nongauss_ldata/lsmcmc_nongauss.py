"""
lsmcmc_nongauss.py  –  LSMCMC V2 filter with non-Gaussian observation noise
=============================================================================

Extends ``Loc_SMCMC_MLSWE_Filter_V2`` to handle **ε-contaminated Gaussian**
(mixture-of-Gaussians) observation noise while keeping exact direct sampling
from the posterior (no MCMC needed).

Mathematical background
-----------------------
The standard LSMCMC V2 assumes:

    Observation noise:   V_t ~ N(0, R),  R = diag(σ²_y)
    Process noise:       W_t ~ N(0, Σ_x)
    Obs model:           Y_t = H Z_t + V_t           (linear)

so the posterior is a mixture of N_f Gaussians and can be sampled
directly (exact Gaussian-mixture posterior).

Here we replace the observation noise with a **J-component Gaussian
mixture** (non-Gaussian, heavy-tailed):

    V_t ~ Σ_{j=1}^{J}  α_j  N(0, R_j),

    R_j = diag(s_j² · σ²_y)

where  α_j  are mixture weights (Σ α_j = 1)  and  s_j  are scale
factors applied to the base observation noise σ_y.  The default
configuration is an ε-contamination model:

    α = [1-ε,  ε],    s = [1,  c],    c >> 1

This gives a core of precise observations plus a fraction ε of
outliers with inflated noise (scale c × σ_y).

**Key result:**  The posterior is still a Gaussian mixture — now with
J × N_f  components — and can be sampled directly:

    π̂(z | y) ∝ g(y | z) · π(z)

    π(z) = (1/N_f) Σ_i  N(z; m_i, Σ_x)        [forecast mixture]

    g(y|z) = Σ_j  α_j  N(y; Hz, R_j)           [mixture likelihood]

    => π̂(z|y) = Σ_{j,i}  w_{j,i}  N(z; μ_{j,i}, P_j)

where

    S_j     = R_j + H Σ_x H^T                  [innovation cov, diagonal]
    w_{j,i} ∝ α_j · N(y; H m_i, S_j)           [mixture weights]
    μ_{j,i} = m_i + Σ_x H^T S_j^{-1}(y - Hm_i) [posterior mean]
    P_j     = Σ_x − Σ_x H^T S_j^{-1} H Σ_x    [posterior cov, diagonal]

Sampling proceeds identically to the Gaussian case, except we draw
component indices from all J × N_f weights instead of just N_f.
"""
import os
import sys
import numpy as np
import scipy.sparse as sp

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mlswe.lsmcmc_V2 import (
    Loc_SMCMC_MLSWE_Filter_V2,
    rescaled_block_means,
)


class NonGaussianFilter(Loc_SMCMC_MLSWE_Filter_V2):
    """
    Localized LSMCMC V2 with ε-contaminated Gaussian observation noise.

    The observation-noise distribution is a J-component Gaussian mixture
    (non-Gaussian / heavy-tailed).  The posterior remains a Gaussian
    mixture with J × N_f components and is sampled directly — no MCMC.

    New YAML parameters
    -------------------
    obs_noise_weights : list of float
        Mixture weights  α_j  (must sum to 1).  Default ``[0.8, 0.2]``.
    obs_noise_scales : list of float
        Scale factors  s_j  applied to base σ_y.  Default ``[1.0, 5.0]``.
    """

    def __init__(self, isim, params):
        super().__init__(isim, params)

        # ---- Non-Gaussian obs-noise mixture parameters ----
        weights = np.asarray(
            params.get('obs_noise_weights', [0.8, 0.2]), dtype=np.float64)
        scales = np.asarray(
            params.get('obs_noise_scales', [1.0, 5.0]), dtype=np.float64)

        assert len(weights) == len(scales), (
            "obs_noise_weights and obs_noise_scales must have same length")
        weights /= weights.sum()

        self.obs_mix_weights = weights      # (J,)
        self.obs_mix_scales = scales        # (J,)
        self.J_mix = len(weights)

        print(f"[NonGauss] Observation noise: {self.J_mix}-component "
              f"Gaussian mixture")
        print(f"[NonGauss]   weights α = {self.obs_mix_weights.tolist()}")
        print(f"[NonGauss]   scales  s = {self.obs_mix_scales.tolist()}")

    # ------------------------------------------------------------------
    #  Override the per-block assimilation to use mixture likelihood
    # ------------------------------------------------------------------
    def _assimilate_localized(self, cycle, y_valid, obs_ind_ml,
                              sig_y_cycle, forecast_mean, H_b):
        """
        Block-by-block localized assimilation with non-Gaussian
        (mixture-of-Gaussians) observation noise.

        For each partition block independently:
          For each likelihood component j = 1..J:
            1. Build  S_j = diag(s_j² σ²_y(obs) + σ²_x(obs))
            2. Compute log-weights  log w_{j,i} for all forecast
               members i = 1..N_f
            3. Compute posterior mean  μ_{j,i}  and variance  P_j
          Merge all J × N_f components:
            4. Normalise weights across all J × N_f
            5. Draw N_a indices → sample from corresponding Gaussian
            6. Reduce N_a → N_f via rescaled_block_means
            7. Apply RTPS inflation

        Returns
        -------
        ess_list : list of float
        n_analyzed : int
        """
        Nf = self.nforecast
        N_a = self.mcmc_N
        J = self.J_mix
        mix_alpha = self.obs_mix_weights     # (J,)
        mix_scale = self.obs_mix_scales      # (J,)

        self.lsmcmc_mean[cycle + 1] = forecast_mean.copy()

        fc_prior = self.forecast.copy()

        master_rng = np.random.default_rng()
        n_blocks = len(self._block_loc_info)
        block_seeds = master_rng.integers(0, 2**63, size=n_blocks)

        sig_x_vec = self.sig_x_vec
        rtps_alpha_val = self.rtps_alpha

        def process_block(block_idx):
            info = self._block_loc_info[block_idx]
            if info is None:
                return None

            nearby_idx  = info['nearby_idx']
            halo_cells  = info['halo_cells']
            block_cells = info['block_cells']
            block_in_halo = info['block_in_halo']
            H_loc       = info['H_loc']
            sigma_y_loc = info['sigma_y_local']   # (d_y_loc,) base σ_y

            d_halo  = len(halo_cells)
            d_block = len(block_cells)
            d_y_loc = H_loc.shape[0]

            if d_y_loc == 0:
                return None

            y_local = y_valid[nearby_idx][:d_y_loc]

            fc_halo  = fc_prior[halo_cells, :]       # (d_halo, Nf)
            fc_block = fc_halo[block_in_halo, :]      # (d_block, Nf)
            prior_sprd = np.std(fc_prior[block_cells, :], axis=1)

            is_sp = sp.issparse(H_loc)
            rng = np.random.default_rng(block_seeds[block_idx])

            # ---- Prior noise at halo/block cells (diagonal) ----
            sx2_halo = sig_x_vec[halo_cells] ** 2     # (d_halo,)

            csr = H_loc.tocsr() if is_sp else None
            if csr is not None:
                obs_halo_col = csr.indices[csr.indptr[:-1]][:d_y_loc]
            else:
                obs_halo_col = np.argmax(H_loc, axis=1)

            sx2_at_obs = sx2_halo[obs_halo_col]       # (d_y_loc,)

            # H_b = columns of H_loc at block positions in halo
            if is_sp:
                H_b_mat = H_loc[:, block_in_halo]
                if sp.issparse(H_b_mat):
                    H_b_mat = H_b_mat.toarray()
            else:
                H_b_mat = H_loc[:, block_in_halo]

            sx2_block = sx2_halo[block_in_halo]        # (d_block,)

            # Innovation  y - H m_i  (shared across all j)
            if is_sp:
                Hm = H_loc.dot(fc_halo)               # (d_y, Nf)
            else:
                Hm = H_loc @ fc_halo
            innov_all = y_local.reshape(-1, 1) - Hm    # (d_y, Nf)

            # ---- Loop over J likelihood components ----
            all_logw = np.empty((J, Nf))
            all_post_means = np.empty((J, d_block, Nf))
            all_P_std = np.empty((J, d_block))

            for j in range(J):
                s_j = mix_scale[j]
                a_j = mix_alpha[j]

                # R_j = (s_j * sigma_y_loc)^2
                R_diag_j = (s_j * sigma_y_loc) ** 2    # (d_y_loc,)

                # S_j = σ²_x(obs) + R_j   (diagonal)
                S_diag_j = sx2_at_obs + R_diag_j
                S_diag_j = np.maximum(S_diag_j, 1e-30)
                Sinv_diag_j = 1.0 / S_diag_j           # (d_y_loc,)

                # log w_{j,i} = log α_j
                #             - 0.5 Σ log(S_j_kk)
                #             - 0.5 Σ (y_k - [Hm_i]_k)^2 / S_j_kk
                logw_j = (np.log(a_j + 1e-300)
                          - 0.5 * np.sum(np.log(S_diag_j))
                          - 0.5 * np.sum(
                              innov_all**2 * Sinv_diag_j[:, None],
                              axis=0))
                all_logw[j] = logw_j                    # (Nf,)

                # Posterior mean:  μ_{j,i} = m_i + Σ_x H^T S_j^{-1}(y-Hm_i)
                K_innov_j = (sx2_block[:, None]
                             * ((H_b_mat * Sinv_diag_j[:, None]).T
                                @ innov_all))
                all_post_means[j] = fc_block + K_innov_j  # (d_block, Nf)

                # Posterior variance (diagonal):
                #   P_j[c] = σ²_x[c] − σ⁴_x[c] Σ_k H²_kc / S_j_kk
                Hb2_Sinv_j = np.sum(
                    H_b_mat**2 * Sinv_diag_j[:, None], axis=0)
                P_diag_j = sx2_block - sx2_block**2 * Hb2_Sinv_j
                P_diag_j = np.maximum(P_diag_j, 0.0)
                all_P_std[j] = np.sqrt(P_diag_j)       # (d_block,)

            # ---- Normalise weights across all J × Nf components ----
            flat_logw = all_logw.ravel()                # (J*Nf,)
            flat_logw -= flat_logw.max()
            flat_w = np.exp(flat_logw)
            flat_w /= flat_w.sum()
            ess = 1.0 / np.sum(flat_w ** 2)

            # ---- Sample N_a from the J × Nf Gaussian mixture ----
            flat_idxs = rng.choice(J * Nf, size=N_a, p=flat_w)
            j_idxs = flat_idxs // Nf                    # which lik. comp
            i_idxs = flat_idxs % Nf                     # which forecast mbr

            noise = rng.standard_normal((d_block, N_a))
            samples = (all_post_means[j_idxs, :, i_idxs].T
                       + all_P_std[j_idxs].T * noise)   # (d_block, N_a)

            if not np.all(np.isfinite(samples)):
                return None

            # ---- Reduce N_a → Nf via rescaled block means ----
            block_mean = np.mean(samples, axis=1)
            anal_block = rescaled_block_means(
                samples, Nf, block_mean)

            # ---- RTPS inflation ----
            if rtps_alpha_val > 0:
                amean = np.mean(anal_block, axis=1, keepdims=True)
                asprd = np.std(anal_block, axis=1)
                safe = np.maximum(asprd, 1e-30)
                infl = 1.0 + rtps_alpha_val * (
                    prior_sprd - asprd) / safe
                infl = np.maximum(infl, 1.0)
                anal_block = amean + infl[:, None] * (
                    anal_block - amean)

            return {
                'block_cells': block_cells,
                'anal_block':  anal_block,
                'mu_block':    block_mean,
                'ess':         ess,
            }

        # ---- Process blocks (parallel or serial) ----
        from concurrent.futures import ThreadPoolExecutor
        n_workers = min(n_blocks, self.n_block_workers)
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(
                    process_block, range(n_blocks)))
        else:
            results = [process_block(i) for i in range(n_blocks)]

        # ---- Apply results ----
        ess_list = []
        n_analyzed = 0
        for result in results:
            if result is None:
                continue
            bc = result['block_cells']
            self.forecast[bc, :] = result['anal_block']
            self.lsmcmc_mean[cycle + 1, bc] = result['mu_block']
            ess_list.append(result['ess'])
            n_analyzed += 1

        return ess_list, n_analyzed

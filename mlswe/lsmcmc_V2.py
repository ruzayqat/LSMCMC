"""
mlswe.lsmcmc_V2  –  Localized SMCMC filter with per-block halo localization
=============================================================================

New variant of the LSMCMC filter for the 3-layer MLSWE model that uses
per-block halo-based localization with Gaspari-Cohn tapering, following
the approach in ``loc_smcmc_sqg_exact_from_Gauss.py``.

Key differences from ``mlswe.lsmcmc_V1``:
    1. Per-block halo localization with distance-based observation selection
    2. Gaspari-Cohn tapering of observation noise to smoothly damp distant obs
    3. All ensemble members initialized identically (required by SMCMC theory)
    4. Block-by-block independent assimilation (parallelized via threads)
    5. RTPS inflation and periodic ensemble reset / rejuvenation
    6. Models synced to analysis state after each assimilation cycle

State vector ordering (same as ``mlswe.lsmcmc_V1``):
    [ h₀, u₀, v₀, T₀,   h₁, u₁, v₁, T₁,   h₂, u₂, v₂, T₂ ]
    each block (ny × nx).   dimx = 12 × ny × nx.

Surface-only observations (drifter u, v, SST, SWOT SSH) map to layer-0
fields.
"""
import os
import sys
import time
import warnings
import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from netCDF4 import Dataset

# Import from the local copy of loc_smcmc_swe_exact_from_Gauss.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from loc_smcmc_swe_exact_from_Gauss import (
    partition_domain,
    get_divisors,
)

from mlswe.model import MLSWE, coriolis_array, lonlat_to_dxdy


# ===================================================================
#  Utility functions (self-contained versions from SQG reference)
# ===================================================================

def build_H_loc_from_global(obs_indices, sv_ind_Q, drop_unmapped=True):
    """
    Build localised observation matrix mapping local state → observations.

    Parameters
    ----------
    obs_indices : 1-D int array
        Global state-vector indices of observations.
    sv_ind_Q : 1-D int array
        Global state-vector indices of the local (halo) state.
    drop_unmapped : bool
        If True, drop obs rows whose global index is not in sv_ind_Q.

    Returns
    -------
    H_loc : scipy.sparse CSR matrix  (d_y_eff, d_halo)
    """
    obs = np.asarray(obs_indices, dtype=int)
    sv = np.asarray(sv_ind_Q, dtype=int)
    d_y = obs.size
    d_p = sv.size
    col_map = {g: i for i, g in enumerate(sv)}
    rows, cols, mapped = [], [], []
    for r, g in enumerate(obs):
        if g in col_map:
            rows.append(len(mapped) if drop_unmapped else r)
            cols.append(col_map[g])
            mapped.append(r)
    d_y_eff = len(mapped) if drop_unmapped else d_y
    data = np.ones(len(rows), dtype=float)
    if len(rows) == 0:
        return sp.csr_matrix((d_y_eff, d_p))
    return sp.coo_matrix((data, (rows, cols)),
                         shape=(d_y_eff, d_p)).tocsr()


def gaussian_block_means(samples, M):
    """Average N iid sample-columns into M groups → (d, M)."""
    d, n = samples.shape
    perm = np.random.permutation(n)
    base, rem = divmod(n, M)
    means = np.zeros((d, M))
    groups = []
    start = 0
    for i in range(M):
        size = base + (1 if i < rem else 0)
        idx = perm[start:start + size]
        groups.append(idx)
        means[:, i] = samples[:, start:start + size].mean(axis=1)
        start += size
    return means, groups


def rescaled_block_means(samples, M, mu):
    """Reduce N samples to M via grouped means, rescaled to preserve variance."""
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
            return state_flat.copy(), t_val, True   # blew up
    return mdl.state_flat.copy(), float(mdl.t), False


# ===================================================================
#  Filter class
# ===================================================================
class Loc_SMCMC_MLSWE_Filter_V2:
    """
    Localized SMCMC filter for the 3-layer primitive-equations model
    with per-block halo localization and Gaspari-Cohn tapering.

    Follows the localization approach of
    ``loc_smcmc_sqg_exact_from_Gauss.py`` (SQG project), adapted to
    the MLSWE state-vector layout (12 fields × ncells).

    Ensemble members are initialised **identically** at t = 0 as
    required by SMCMC theory.  Diversity is created by the posterior
    sampling at the first assimilation cycle.
    """

    def __init__(self, isim, params):
        self.isim = isim
        self.params = params

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

        # ---- MCMC parameters ----
        self.nforecast = params['nforecast']
        self.mcmc_N = params['mcmc_N']
        self.burn_in = params.get('burn_in', self.mcmc_N)
        self.mcmc_iters = self.mcmc_N + self.burn_in

        # ---- Storage ----
        self.lsmcmc_mean = np.zeros((self.nassim + 1, self.dimx))
        self.RMSE = np.empty(self.nassim)

        # ---- Physical parameters ----
        self.H_mean = params.get('H_mean', 4000.0)
        self.H_rest = params.get('H_rest', [100.0, 400.0, 3500.0])
        self.rho = params.get('rho', [1023.0, 1026.0, 1028.0])
        self.T_rest = params.get('T_rest', [298.0, 283.0, 275.0])

        # ---- Noise ----
        self.sig_x_uv = params.get('sig_x_uv', params.get('sig_x', 0.15))
        self.sig_x_sst = params.get('sig_x_sst', 1.0)
        self.sig_x_ssh = params.get('sig_x_ssh', self.sig_x_uv)
        self.sig_x = self.sig_x_uv   # backward-compat alias

        sig_huv = self.sig_x_uv
        sig_t = self.sig_x_sst
        sig_h = self.sig_x_ssh

        assimilate_fields = str(params.get('assimilate_fields', 'uv_sst'))
        use_swot_ssh = bool(params.get('use_swot_ssh', False))
        assim_uv = 'uv' in assimilate_fields
        assim_ssh = use_swot_ssh
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

        print(f"[LSMCMC-v2] assimilate_fields='{assimilate_fields}' "
              f"-> uv={assim_uv}, ssh={assim_ssh}, sst={assim_sst}")
        print(f"[LSMCMC-v2] Noise: h0={sig_h0:.4f}, u0,v0={sig_u0:.4f}, "
              f"T0={sig_t0:.4f}, layers 1,2=0 (unobserved)")

        # ---- Partition (for initial block structure) ----
        self.num_subdomains = params.get('num_subdomains', 480)
        block_list, labels, nblocks, nby, nbx, bh, bw = partition_domain(
            self.ny, self.nx, self.num_subdomains)
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
        self.f_2d = coriolis_array(self.lat_min, self.lat_max, self.ny, self.nx)

        # ---- Localization parameters (NEW — from SQG approach) ----
        self.r_loc = float(params.get('r_loc', 15.0))
        self.rtps_alpha = float(params.get('rtps_alpha', 0.5))
        self.reset_interval = int(params.get('reset_interval', 0))
        self.n_block_workers = int(params.get('n_block_workers', os.cpu_count() or 1))

        # ---- Multiprocessing (for model advance) ----
        self.ncores = params.get('ncores', 1)
        self._use_mp = self.ncores > 1
        self._mp_pool = None

        # ---- Internal localization cache (filled each cycle) ----
        self._block_loc_info = []

        print(f"[LSMCMC-v2] Grid: {self.ny}×{self.nx}, "
              f"layers: {self.nlayers}, "
              f"dimx: {self.dimx}, "
              f"dt: {self.dt}s, "
              f"nassim: {self.nassim}, "
              f"r_loc: {self.r_loc}, "
              f"rtps_alpha: {self.rtps_alpha}, "
              f"ncores: {self.ncores}")

    # ------------------------------------------------------------------
    #  Observation index mapping — surface only (same as lsmcmc.py)
    # ------------------------------------------------------------------
    def _obs_ind_to_layer0(self, obs_ind_sv):
        """Map single-layer obs indices to multi-layer state vector.
        Identity for surface observations."""
        return obs_ind_sv

    def _obs_ind_to_cell(self, obs_ind_sv):
        """Convert any obs_ind to cell index [0, ncells)."""
        return obs_ind_sv % self.ncells

    # ------------------------------------------------------------------
    #  Build model instances (same as lsmcmc.py)
    # ------------------------------------------------------------------
    def _make_model_kwargs(self, H_b, bc_handler, tstart):
        """Return kwargs dict for MLSWE constructor."""
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
        )

    def _make_init_state(self, H_b, bc_handler, tstart):
        """Create initial layer states (same as lsmcmc.py)."""
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
            print("[LSMCMC-v2] Using full-domain HYCOM initial conditions")
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
        print("[LSMCMC-v2] WARNING: No HYCOM IC provided, using rest state")
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
            u0_list.append(np.zeros((self.ny, self.nx), dtype=np.float64))
            v0_list.append(np.zeros((self.ny, self.nx), dtype=np.float64))
            T0_list.append(np.full((self.ny, self.nx), self.T_rest[k],
                                   dtype=np.float64))
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
    #  Advance ensemble (same as lsmcmc.py)
    # ------------------------------------------------------------------
    def _advance_ensemble(self, models, nsteps, add_noise=True):
        """Advance all ensemble members by *nsteps* timesteps."""
        Nf = len(models)
        blown = []

        if self._use_mp and self._mp_pool is not None:
            args = [(mdl.state_flat.copy(), float(mdl.t), nsteps)
                    for mdl in models]
            results = self._mp_pool.map(_mp_advance, args)
            for j, res in enumerate(results):
                state_flat_new, t_new, blew_up = res
                if blew_up or not np.all(np.isfinite(state_flat_new)):
                    blown.append(j)
                    continue
                if add_noise:
                    state_flat_new += np.random.normal(
                        scale=self.sig_x_vec)
                models[j].state_flat = state_flat_new
                models[j].t = t_new
                self.forecast[:, j] = models[j].state_flat.copy()
        else:
            for j in range(Nf):
                mdl = models[j]
                old_flat = mdl.state_flat.copy()
                for _ in range(nsteps):
                    mdl._timestep()
                    if not np.all(np.isfinite(mdl.state_flat)):
                        blown.append(j)
                        mdl.state_flat = old_flat
                        break
                else:
                    flat = mdl.state_flat.copy()
                    if add_noise:
                        flat += np.random.normal(scale=self.sig_x_vec)
                        mdl.state_flat = flat
                    self.forecast[:, j] = flat

        if blown:
            healthy = [j for j in range(Nf) if j not in blown]
            if healthy:
                ens_mean = np.mean(self.forecast[:, healthy], axis=1)
                for j in blown:
                    perturbed = ens_mean + np.random.normal(
                        scale=0.05 * self.sig_x_vec)
                    models[j].state_flat = perturbed
                    self.forecast[:, j] = perturbed
            warnings.warn(
                f"Reset {len(blown)}/{Nf} blown-up ensemble members")

    # ------------------------------------------------------------------
    #  Gaspari-Cohn compactly-supported correlation function
    # ------------------------------------------------------------------
    @staticmethod
    def _gaspari_cohn(dist, c):
        """Gaspari-Cohn 5th-order polynomial.

        Parameters
        ----------
        dist : array  — distances
        c    : float  — half-width (support = 2c)

        Returns
        -------
        rho : array in [0, 1]
        """
        r = np.abs(dist) / c
        rho = np.zeros_like(r)
        m1 = r <= 1.0
        m2 = (~m1) & (r <= 2.0)
        r1 = r[m1]
        rho[m1] = (1.0 - (5.0/3.0)*r1**2 + (5.0/8.0)*r1**3
                   + 0.5*r1**4 - 0.25*r1**5)
        r2 = r[m2]
        rho[m2] = (4.0 - 5.0*r2 + (5.0/3.0)*r2**2 + (5.0/8.0)*r2**3
                   - 0.5*r2**4 + (1.0/12.0)*r2**5 - 2.0/(3.0*r2))
        rho = np.clip(rho, 0.0, 1.0)
        return rho

    # ------------------------------------------------------------------
    #  Per-block halo localization (from SQG approach)
    # ------------------------------------------------------------------
    def _precompute_block_localization(self, obs_cell_indices,
                                       obs_ind_ml, sig_y_base):
        """
        Precompute per-block localization structures for the current
        observation network.

        For each partition block we find:
          - nearby observations (within ``r_loc`` grid points of block
            centre),
          - halo cells (all grid cells within ``r_loc``, expanded across
            layer-0 fields h₀, u₀, v₀, T₀),
          - the local observation operator H mapping halo → local obs,
          - Gaspari-Cohn–tapered per-obs σ_y,
          - mapping indices from block cells inside the halo vector.

        Parameters
        ----------
        obs_cell_indices : 1-D int array
            Cell-level index [0, ncells) for every observation.
        obs_ind_ml : 1-D int array
            Global state-vector indices for every observation (layer-0).
        sig_y_base : float or 1-D array
            Base observation-noise std (scalar or per-obs).
        """
        r_loc = self.r_loc
        ny, nx = self.ny, self.nx
        ncells = self.ncells
        nf_loc = self.fields_per_layer   # 4 — only layer 0 in halo

        # Spatial positions of each observation
        obs_row, obs_col = np.unravel_index(obs_cell_indices, (ny, nx))

        unique_blocks = np.unique(self.partition_labels)

        # Pre-build coordinate grids (reused for every block)
        grid_rows, grid_cols = np.meshgrid(
            np.arange(ny), np.arange(nx), indexing='ij')

        self._block_loc_info = []

        for block_id in unique_blocks:
            block_mask = (self.partition_labels == block_id)
            block_ij = np.argwhere(block_mask)

            cy = block_ij[:, 0].mean()
            cx = block_ij[:, 1].mean()

            # ---- nearby observations (non-periodic distance) ----
            dy_o = np.abs(obs_row - cy)
            dx_o = np.abs(obs_col - cx)
            dist_obs = np.sqrt(dy_o**2 + dx_o**2)

            nearby = dist_obs < r_loc
            if not np.any(nearby):
                self._block_loc_info.append(None)
                continue

            nearby_idx = np.where(nearby)[0]

            # Global obs indices (already layer-0 state indices)
            obs_global = obs_ind_ml[nearby_idx]

            # ---- halo: all cells within r_loc of block centre ----
            dy_g = np.abs(grid_rows - cy)
            dx_g = np.abs(grid_cols - cx)
            halo_mask = np.sqrt(dy_g**2 + dx_g**2) < r_loc

            halo_ij = np.argwhere(halo_mask)
            halo_flat_0 = np.ravel_multi_index(
                (halo_ij[:, 0], halo_ij[:, 1]), (ny, nx))

            # ALSO include block spatial cells to guarantee block ⊂ halo
            block_flat_0 = np.ravel_multi_index(
                (block_ij[:, 0], block_ij[:, 1]), (ny, nx))
            halo_flat_0 = np.unique(np.concatenate(
                [halo_flat_0, block_flat_0]))

            # Expand to layer-0 fields: h₀, u₀, v₀, T₀
            halo_cells = np.sort(np.concatenate(
                [halo_flat_0 + f * ncells for f in range(nf_loc)]))

            # Block cells across layer-0 fields
            block_cells = np.sort(np.concatenate(
                [block_flat_0 + f * ncells for f in range(nf_loc)]))

            # Position of every block cell inside halo_cells (sorted)
            block_in_halo = np.searchsorted(halo_cells, block_cells)

            # Local H mapping halo → nearby obs
            H_loc = build_H_loc_from_global(
                obs_global, halo_cells, drop_unmapped=True)

            # Gaspari-Cohn–tapered per-obs σ_y
            dist_local = dist_obs[nearby_idx][:H_loc.shape[0]]
            gc_w = self._gaspari_cohn(dist_local, r_loc)
            gc_w = np.maximum(gc_w, 1e-6)

            if isinstance(sig_y_base, np.ndarray) and sig_y_base.size > 1:
                sigma_y_local_base = sig_y_base[nearby_idx][:H_loc.shape[0]]
            else:
                sigma_y_local_base = np.full(
                    H_loc.shape[0], float(np.atleast_1d(sig_y_base)[0]))

            sigma_y_local = sigma_y_local_base / np.sqrt(gc_w)

            self._block_loc_info.append({
                'nearby_idx': nearby_idx,
                'halo_cells': halo_cells,
                'block_cells': block_cells,
                'block_in_halo': block_in_halo,
                'H_loc': H_loc,
                'sigma_y_local': sigma_y_local,
            })

        n_active = sum(1 for b in self._block_loc_info if b is not None)
        n_total = len(unique_blocks)
        if n_active > 0:
            avg_nobs = np.mean([b['H_loc'].shape[0]
                                for b in self._block_loc_info
                                if b is not None])
            avg_halo = np.mean([len(b['halo_cells'])
                                for b in self._block_loc_info
                                if b is not None])
        else:
            avg_nobs = avg_halo = 0.0
        print(f'  Localization: {n_active}/{n_total} blocks active, '
              f'r_loc={r_loc:.1f}, avg local obs={avg_nobs:.1f}, '
              f'avg halo size={avg_halo:.0f}')

    # ------------------------------------------------------------------
    #  Block-by-block localized assimilation (from SQG approach)
    # ------------------------------------------------------------------
    def _assimilate_localized(self, cycle, y_valid, obs_ind_ml,
                              sig_y_cycle, forecast_mean, H_b):
        """
        Block-by-block localized assimilation using exact sampling from
        the Gaussian-mixture posterior.

        For each partition block independently:
            1. Build diagonal S = diag(σ²_x(obs)) + R   (exploiting
               selection-operator structure of H)
            2. Compute mixture weights  w_i ∝ N(y; H mᵢ, S)
            3. Draw N_a component indices from Categorical(w)
            4. Sample from the diagonal posterior Gaussian of each
               selected component
            5. Reduce N_a → Nf via ``rescaled_block_means``
            6. Apply RTPS inflation

        Handles the diagonal prior covariance Σ_x = diag(σ²_x_k)
        arising from different noise amplitudes for h, u, v, T.

        Returns
        -------
        ess_list : list of float
            Effective sample size per analysed block.
        n_analyzed : int
            Number of blocks that were analysed.
        """
        Nf = self.nforecast
        N_a = self.mcmc_N

        self.lsmcmc_mean[cycle + 1] = forecast_mean.copy()

        # Freeze the prior so all blocks see the same forecast
        fc_prior = self.forecast.copy()

        # Per-block RNG seeds for thread safety
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
            sigma_y_loc = info['sigma_y_local']

            d_halo  = len(halo_cells)
            d_block = len(block_cells)
            d_y_loc = H_loc.shape[0]

            if d_y_loc == 0:
                return None

            # Local observation values
            y_local = y_valid[nearby_idx][:d_y_loc]

            # Forecast at halo / block cells
            fc_halo  = fc_prior[halo_cells, :]      # (d_halo, Nf)
            fc_block = fc_halo[block_in_halo, :]     # (d_block, Nf)
            prior_sprd = np.std(fc_prior[block_cells, :], axis=1)

            is_sp = sp.issparse(H_loc)
            rng = np.random.default_rng(block_seeds[block_idx])

            # ---- Observation-error variance (GC-tapered) ----
            R_diag = sigma_y_loc ** 2                # (d_y_loc,)

            # ---- Prior noise at halo cells (diagonal) ----
            sx2_halo = sig_x_vec[halo_cells] ** 2    # (d_halo,)

            # Find which halo column each obs row maps to.
            # H_loc is CSR with exactly one nonzero per row.
            csr = H_loc.tocsr() if is_sp else None
            if csr is not None:
                obs_halo_col = csr.indices[csr.indptr[:-1]][:d_y_loc]
            else:
                obs_halo_col = np.argmax(H_loc, axis=1)

            sx2_at_obs = sx2_halo[obs_halo_col]      # (d_y_loc,)

            # ---- Innovation covariance (diagonal): S = σ²_x(obs) + R ----
            S_diag = sx2_at_obs + R_diag
            S_diag = np.maximum(S_diag, 1e-30)
            Sinv_diag = 1.0 / S_diag                 # (d_y_loc,)

            # ---- Mixture weights ----
            if is_sp:
                Hm = H_loc.dot(fc_halo)              # (d_y, Nf)
            else:
                Hm = H_loc @ fc_halo
            innov_all = y_local.reshape(-1, 1) - Hm  # (d_y, Nf)

            logw = -0.5 * np.sum(
                innov_all**2 * Sinv_diag[:, None], axis=0)
            logw -= logw.max()
            w = np.exp(logw)
            w /= w.sum()
            ess = 1.0 / np.sum(w ** 2)

            # ---- Posterior for *block* cells ----
            # H_b = columns of H_loc at block positions in halo
            if is_sp:
                H_b = H_loc[:, block_in_halo]
                if sp.issparse(H_b):
                    H_b = H_b.toarray()
            else:
                H_b = H_loc[:, block_in_halo]

            sx2_block = sx2_halo[block_in_halo]       # (d_block,)

            # K_block @ innovation  (diagonal Σ_x version)
            #   K[c,j] = σ²_x[c] · H_b[j,c] / S[j]
            K_innov = (sx2_block[:, None]
                       * ((H_b * Sinv_diag[:, None]).T @ innov_all))

            post_means = fc_block + K_innov           # (d_block, Nf)

            # Posterior variance (diagonal):
            #   P[c] = σ²_x[c] − σ²_x[c]² · Σ_j H_b[j,c]² / S[j]
            Hb2_Sinv = np.sum(
                H_b**2 * Sinv_diag[:, None], axis=0)  # (d_block,)
            P_diag = sx2_block - sx2_block**2 * Hb2_Sinv
            P_diag = np.maximum(P_diag, 0.0)
            P_std = np.sqrt(P_diag)

            # ---- Sample N_a from the Gaussian mixture ----
            idxs = rng.choice(Nf, size=N_a, p=w)
            noise = rng.standard_normal((d_block, N_a))
            samples = post_means[:, idxs] + P_std[:, None] * noise

            if not np.all(np.isfinite(samples)):
                return None

            # ---- Reduce N_a → Nf via rescaled block means ----
            block_mean = np.mean(samples, axis=1)
            anal_block = rescaled_block_means(
                samples, Nf, block_mean)               # (d_block, Nf)
            mu_block = block_mean

            # ---- RTPS inflation ----
            if rtps_alpha_val > 0:
                amean = np.mean(anal_block, axis=1, keepdims=True)
                asprd = np.std(anal_block, axis=1)
                safe  = np.maximum(asprd, 1e-30)
                infl  = 1.0 + rtps_alpha_val * (
                    prior_sprd - asprd) / safe
                infl  = np.maximum(infl, 1.0)
                anal_block = amean + infl[:, None] * (
                    anal_block - amean)

            return {
                'block_cells': block_cells,
                'anal_block':  anal_block,
                'mu_block':    mu_block,
                'ess':         ess,
            }

        # ---- Process blocks (parallel or serial) ----
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

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    def run(self, H_b, bc_handler, obs_file, tstart):
        """Run the full localized LSMCMC filter."""
        # --- Load observations ---
        yobs, yind, obs_times, sig_y = self._load_obs(obs_file)

        # --- Initial state ---
        h0, u0, v0, T0 = self._make_init_state(H_b, bc_handler, tstart)
        model_kw = self._make_model_kwargs(H_b, bc_handler, tstart)

        init_model = MLSWE(h0, u0, v0, T0=T0, **model_kw)
        init_model.timesteps = 1
        self.lsmcmc_mean[0] = init_model.state_flat.copy()

        # --- Set up multiprocessing pool ---
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
            )
            _mp_mlswe_kwargs['h0'] = [hk.copy() for hk in h0]
            _mp_mlswe_kwargs['u0'] = [uk.copy() for uk in u0]
            _mp_mlswe_kwargs['v0'] = [vk.copy() for vk in v0]
            _mp_mlswe_kwargs['T0'] = [Tk.copy() for Tk in T0]
            n_workers = min(self.ncores, self.nforecast)
            print(f"[LSMCMC-v2] Starting multiprocessing pool "
                  f"with {n_workers} workers for {self.nforecast} "
                  f"forecasts")
            self._mp_pool = mp.Pool(n_workers,
                                    initializer=_mp_init_worker)
        else:
            print(f"[LSMCMC-v2] Serial mode (ncores=1)")

        # --- Create ensemble — ALL IDENTICAL (SMCMC requirement) ---
        Nf = self.nforecast
        self.forecast = np.zeros((self.dimx, Nf))
        models = []
        for j in range(Nf):
            mdl = MLSWE(
                [hk.copy() for hk in h0],
                [uk.copy() for uk in u0],
                [vk.copy() for vk in v0],
                T0=[Tk.copy() for Tk in T0],
                **model_kw)
            mdl.timesteps = 1
            # *** NO perturbation — all members identical ***
            models.append(mdl)
            self.forecast[:, j] = mdl.state_flat.copy()

        print(f"[LSMCMC-v2] Ensemble init: {Nf} IDENTICAL members "
              f"from HYCOM IC (SMCMC theory)")
        # Verify all members are truly identical
        _spread = np.std(self.forecast, axis=1)
        assert np.allclose(_spread, 0.0), \
            "Ensemble members are NOT identical at init!"
        _nc = self.ncells
        _h0_mean = self.forecast[:_nc, 0].mean()
        _u0_mean = self.forecast[_nc:2*_nc, 0].mean()
        _v0_mean = self.forecast[2*_nc:3*_nc, 0].mean()
        _T0_mean = self.forecast[3*_nc:4*_nc, 0].mean()
        _ssh0 = self.forecast[:_nc, 0] - H_b.ravel()
        print(f"  IC check: h_mean={_h0_mean:.1f}m, "
              f"u_mean={_u0_mean:.4f}m/s, v_mean={_v0_mean:.4f}m/s, "
              f"T_mean={_T0_mean:.2f}K, "
              f"SSH=[{_ssh0.min():.3f},{_ssh0.max():.3f}]m, "
              f"spread={_spread.max():.2e}")

        nc = self.ncells

        # ---- Main assimilation loop ----
        t_freq = self.assim_timesteps
        rmse_vel_all = np.full(self.nassim, np.nan)
        rmse_sst_all = np.full(self.nassim, np.nan)
        rmse_ssh_all = np.full(self.nassim, np.nan)

        for cycle in range(self.nassim):
            t0_wall = time.time()

            # Advance ensemble
            self._advance_ensemble(
                models, t_freq, add_noise=(self.sig_x > 0))

            # Forecast mean
            forecast_mean = np.mean(self.forecast, axis=1)

            # Diagnostic: ensemble spread
            if (cycle + 1) % 50 == 0 or cycle == 0:
                h_std = np.std(self.forecast[0:nc, :], axis=1)
                u_std = np.std(self.forecast[nc:2*nc, :], axis=1)
                print(f"  [spread] h_std: mean={h_std.mean():.3f} "
                      f"max={h_std.max():.3f}  "
                      f"u_std: mean={u_std.mean():.3f} "
                      f"max={u_std.max():.3f}")

            # ---- Get observations for this cycle ----
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

            # Per-obs noise for this cycle
            if isinstance(sig_y, np.ndarray) and sig_y.ndim == 2:
                sig_y_cycle = sig_y[cycle][valid]
            elif isinstance(sig_y, np.ndarray) and sig_y.ndim == 1:
                sig_y_cycle = (sig_y[valid] if sig_y.size > 1
                               else float(sig_y))
            else:
                sig_y_cycle = float(sig_y)

            # Map to multi-layer state vector
            obs_ind_ml = self._obs_ind_to_layer0(ind_valid)
            obs_cell   = self._obs_ind_to_cell(ind_valid)

            # ---- Precompute block localization for this cycle ----
            self._precompute_block_localization(
                obs_cell, obs_ind_ml, sig_y_cycle)

            # ---- Block-by-block assimilation ----
            ess_list, n_analyzed = self._assimilate_localized(
                cycle, y_valid, obs_ind_ml, sig_y_cycle,
                forecast_mean, H_b)

            # ---- Post-analysis SSH relaxation ----
            ssh_relax = float(self.params.get('ssh_relax_rate', 0.0))
            if ssh_relax > 0:
                nc_ = self.ncells
                h_total_anal = self.lsmcmc_mean[cycle + 1, :nc_].copy()
                H_b_flat_ = H_b.ravel()
                eta_anal = h_total_anal - H_b_flat_
                t_now = float(models[0].t) if models else tstart
                ssh_ref_3d = self.params.get('ssh_relax_ref', None)
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
                    self.params.get('ssh_analysis_relax_frac', 0.5))
                eta_correction = -anal_relax_frac * (eta_anal - eta_ref)
                h_total_corrected = h_total_anal + eta_correction
                self.lsmcmc_mean[cycle + 1, :nc_] = h_total_corrected
                for j_ens in range(Nf):
                    h_j = self.forecast[:nc_, j_ens]
                    eta_j = h_j - H_b_flat_
                    self.forecast[:nc_, j_ens] = (
                        h_j - anal_relax_frac * (eta_j - eta_ref))

            # ---- Periodic reset & rejuvenation ----
            if (self.reset_interval > 0
                    and (cycle + 1) % self.reset_interval == 0):
                ens_mean = np.mean(
                    self.forecast, axis=1, keepdims=True)
                self.forecast = np.tile(ens_mean, (1, Nf))
                noise = (np.random.normal(size=self.forecast.shape)
                         * self.sig_x_vec[:, None])
                self.forecast += noise
                # Re-centre to preserve the mean exactly
                self.forecast += (
                    ens_mean
                    - np.mean(self.forecast, axis=1, keepdims=True))
                print(f'  !! Reset & Rejuvenation at cycle {cycle+1}')

            # ---- Sync models with analysis state ----
            for j in range(Nf):
                models[j].state_flat = self.forecast[:, j].copy()

            # ---- RMSE ----
            z_a = self.lsmcmc_mean[cycle + 1]
            H_z = z_a[obs_ind_ml]
            residuals = H_z - y_valid
            vel_mask = (obs_ind_ml >= nc) & (obs_ind_ml < 3*nc)
            sst_mask = (obs_ind_ml >= 3*nc) & (obs_ind_ml < 4*nc)
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
                _fmean_h = forecast_mean[0:nc]
                _fmean_ssh = _fmean_h - H_b.ravel()
                ess_arr = (np.array(ess_list) if ess_list
                           else np.array([0.0]))
                print(
                    f"  Cycle {cycle+1}/{self.nassim}  "
                    f"vel_RMSE={rmse_vel_all[cycle]:.6f}  "
                    f"sst_RMSE={rmse_sst_all[cycle]:.4f}  "
                    f"ssh_RMSE={rmse_ssh_all[cycle]:.4f}  "
                    f"SSH=[{_ssh.min():.1f},{_ssh.max():.1f}]  "
                    f"fcst_SSH="
                    f"[{_fmean_ssh.min():.1f},{_fmean_ssh.max():.1f}]  "
                    f"ESS={np.mean(ess_arr):.1f}/{Nf} "
                    f"[{ess_arr.min():.1f},{ess_arr.max():.1f}]  "
                    f"blks={n_analyzed}  "
                    f"({elapsed:.1f}s)")

        # ---- Cleanup ----
        if self._mp_pool is not None:
            self._mp_pool.close()
            self._mp_pool.join()
            self._mp_pool = None
            print("[LSMCMC-v2] Multiprocessing pool closed.")

        self.rmse_vel = rmse_vel_all
        self.rmse_sst = rmse_sst_all
        self.rmse_ssh = rmse_ssh_all
        return self.lsmcmc_mean

    # ------------------------------------------------------------------
    #  Load observations (same as lsmcmc.py)
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
    #  Save results (same as lsmcmc.py)
    # ------------------------------------------------------------------
    def save_results(self, outdir, obs_times=None, H_b=None):
        """Save analysis mean, RMSE, and bathymetry to NetCDF."""
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
            vb = ds.createVariable('H_b', 'f4', ('y', 'x'), zlib=True)
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
        print(f"[LSMCMC-v2] Saved results to {outfile}")

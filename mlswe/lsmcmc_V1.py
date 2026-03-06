"""
mlswe.lsmcmc  –  Local SMCMC filter adapted for the 3-layer MLSWE model
=========================================================================

Thin wrapper that re-uses the exact-from-Gaussian machinery from the
single-layer SWE_LSMCMC project, adapting only the state-vector layout
and model instantiation for 3 layers.

The state vector ordering is:
    [ h₀, u₀, v₀, T₀,   h₁, u₁, v₁, T₁,   h₂, u₂, v₂, T₂ ]
    each block (ny × nx).   dimx = 12 × ny × nx.

Surface-only observations (drifter u, v, SST) map to layer-0 fields.
"""
import os
import sys
import time
import warnings
import numpy as np
import multiprocessing as mp
from netCDF4 import Dataset

# Add the SWE_LSMCMC parent to sys.path so we can import its machinery
_SWE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         '..', 'SWE_LSMCMC')
if os.path.isdir(_SWE_DIR):
    _SWE_DIR = os.path.abspath(_SWE_DIR)
    if _SWE_DIR not in sys.path:
        sys.path.insert(0, _SWE_DIR)

from loc_smcmc_swe_exact_from_Gauss import (
    partition_domain,
    get_divisors,
)

from mlswe.model import MLSWE, coriolis_array, lonlat_to_dxdy


# ===================================================================
#  Multiprocessing workers
# ===================================================================
_mp_model_template = None   # pre-built MLSWE; fast-copied per worker
_mp_worker_model = None


def _mp_init_worker():
    """Create per-worker model via shallow copy + state array copies.

    Only mutable arrays (h, u, v, T) are duplicated; everything else
    (H_b, f, dHb_dx, bc_handler, …) is shared read-only (COW after fork).
    This is ~100× faster than copy.deepcopy and avoids duplicating the
    large bc_handler object tree.
    """
    import copy
    global _mp_worker_model
    mdl = _mp_model_template
    new_mdl = copy.copy(mdl)                         # shallow copy
    new_mdl.h = [hk.copy() for hk in mdl.h]          # mutable state
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
        # Early exit if blown up
        if not np.all(np.isfinite(mdl.state_flat)):
            return state_flat.copy(), t_val, True   # flag: blew up
    return mdl.state_flat.copy(), float(mdl.t), False


# ===================================================================
#  Filter class
# ===================================================================
class Loc_SMCMC_MLSWE_Filter:
    """
    Local SMCMC filter for the 3-layer primitive equations model.

    Mirrors the single-layer ``Loc_SMCMC_SWE_Filter`` but with
    state dimension = 12 × ny × nx  (4 fields × 3 layers).
    """

    def __init__(self, isim, params):
        self.isim = isim
        self.params = params

        # Grid
        self.nx = params['dgx']
        self.ny = params['dgy']
        self.ncells = self.ny * self.nx
        self.nlayers = 3
        self.fields_per_layer = 4   # h, u, v, T
        self.nfields = self.nlayers * self.fields_per_layer  # 12
        self.dimx = self.nfields * self.ncells

        # Time stepping
        self.dt = params['dt']
        self.T = params['T']
        self.assim_timesteps = params.get('assim_timesteps', params.get('t_freq', 48))
        self.nassim = self.T // self.assim_timesteps

        # MCMC parameters
        self.nforecast = params['nforecast']
        self.mcmc_N = params['mcmc_N']
        self.burn_in = params.get('burn_in', self.mcmc_N)
        self.mcmc_iters = self.mcmc_N + self.burn_in

        # Storage
        self.lsmcmc_mean = np.zeros((self.nassim + 1, self.dimx))
        self.RMSE = np.empty(self.nassim)

        # Model physical parameters (needed before noise computation)
        self.H_mean = params.get('H_mean', 4000.0)
        self.H_rest = params.get('H_rest', [100.0, 400.0, 3500.0])
        self.rho = params.get('rho', [1023.0, 1026.0, 1028.0])
        self.T_rest = params.get('T_rest', [298.0, 283.0, 275.0])

        # Noise
        self.sig_x_uv = params.get('sig_x_uv', params.get('sig_x', 0.15))
        self.sig_x_sst = params.get('sig_x_sst', 1.0)
        self.sig_x_ssh = params.get('sig_x_ssh', self.sig_x_uv)  # separate h noise
        # Keep sig_x as alias for backward compat (used in _advance_ensemble)
        self.sig_x = self.sig_x_uv

        # Build per-component noise vector
        # Order: [h_total,u0,v0,T0, 0,u1,v1,T1, 0,u2,v2,T2]
        #
        # Principle: only add noise to fields that are being assimilated.
        # assimilate_fields options: 'uv', 'uv_ssh', 'uv_sst', 'uv_ssh_sst',
        #                           'ssh', 'ssh_sst'
        sig_huv = self.sig_x_uv
        sig_t = self.sig_x_sst
        sig_h = self.sig_x_ssh

        assimilate_fields = str(params.get('assimilate_fields', 'uv_sst'))
        use_swot_ssh = bool(params.get('use_swot_ssh', False))
        assim_uv  = 'uv' in assimilate_fields
        assim_ssh = use_swot_ssh          # SSH observed → h must have noise
        assim_sst = 'sst' in assimilate_fields

        sig_per_field = []
        for k in range(self.nlayers):
            if k == 0:
                sig_h0 = sig_h   if assim_ssh else 0.0
                sig_u0 = sig_huv if assim_uv  else 0.0
                sig_t0 = sig_t   if assim_sst else 0.0
                layer_sig_k = [sig_h0, sig_u0, sig_u0, sig_t0]
            else:
                # layers 1,2 are never observed → zero noise
                layer_sig_k = [0.0, 0.0, 0.0, 0.0]
            for fi, s in enumerate(layer_sig_k):
                sig_per_field.append(s)   # no decay needed (unobserved layers get 0)
        self.sig_x_vec = np.repeat(sig_per_field, self.ncells)

        # Verbosity control (set verbose=False for M-run workers)
        self.verbose = params.get('verbose', True)

        if self.verbose:
            print(f"[LSMCMC] assimilate_fields='{assimilate_fields}' "
                  f"-> uv={assim_uv}, ssh={assim_ssh}, sst={assim_sst}")
            print(f"[LSMCMC] Noise: h0={sig_h0:.4f}, u0,v0={sig_u0:.4f}, "
                  f"T0={sig_t0:.4f}, layers 1,2=0 (unobserved)")

        # Localisation
        self.num_subdomains = params.get('num_subdomains', 480)
        block_list, labels, nblocks, nby, nbx, bh, bw = partition_domain(
            self.ny, self.nx, self.num_subdomains)
        self.partition_labels = labels
        self.block_list = block_list
        self.n_blocks = nblocks

        # Grid coordinates
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

        # Multiprocessing
        self.ncores = params.get('ncores', 1)
        self._use_mp = self.ncores > 1
        self._mp_pool = None

        if self.verbose:
            print(f"[MLSWE-LSMCMC] Grid: {self.ny}×{self.nx}, "
                  f"layers: {self.nlayers}, "
                  f"dimx: {self.dimx}, "
                  f"dt: {self.dt}s, "
                  f"nassim: {self.nassim}, "
                  f"ncores: {self.ncores}")

    # ------------------------------------------------------------------
    #  Observation index mapping — surface only
    # ------------------------------------------------------------------
    def _obs_ind_to_layer0(self, obs_ind_sv):
        """
        Convert single-layer SWE obs indices (0-based into [h,u,v,T] × ncells)
        to multi-layer obs indices (into layer-0 of the 12-field state vector).

        Single-layer layout:  [h, u, v, T]  ← obs_ind ranges [0, 4*ncells)
        Multi-layer layout:   [h₀,u₀,v₀,T₀, h₁,u₁,v₁,T₁, h₂,u₂,v₂,T₂]

        Layer 0 occupies the first 4*ncells entries, so mapping is identity
        for surface observations.
        """
        # obs_ind_sv already references fields 0..3 (h,u,v,T) which are
        # the first 4 fields = layer 0 in our layout.
        return obs_ind_sv  # identity for surface observations

    def _obs_ind_to_cell(self, obs_ind_sv):
        """Convert any obs_ind to cell index [0, ncells)."""
        return obs_ind_sv % self.ncells

    # ------------------------------------------------------------------
    #  Build model instances
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
            sst_nudging_ref_times=self.params.get('sst_nudging_ref_times', None),
            ssh_relax_rate=self.params.get('ssh_relax_rate', 0.0),
            ssh_relax_ref=self.params.get('ssh_relax_ref', None),
            ssh_relax_ref_times=self.params.get('ssh_relax_ref_times', None),
            sst_flux_type=self.params.get('sst_flux_type', None),
            sst_alpha=float(self.params.get('sst_alpha', 15.0)),
            sst_h_mix=float(self.params.get('sst_h_mix', 50.0)),
            sst_T_air=self.params.get('sst_T_air', None),
            sst_T_air_times=self.params.get('sst_T_air_times', None),
            ssh_relax_interior_floor=float(self.params.get('ssh_relax_interior_floor', 0.1)),
            shallow_drag_depth=float(self.params.get('shallow_drag_depth', 500.0)),
            shallow_drag_coeff=float(self.params.get('shallow_drag_coeff', 5.0e-4)),
        )

    def _make_init_state(self, H_b, bc_handler, tstart):
        """Create initial layer states.

        If HYCOM IC was passed via params (ic_h0, ic_u0, ic_v0, ic_T0),
        use those (full-domain HYCOM interpolation).
        Otherwise, fall back to rest-state + boundary-only values.
        """
        from mlswe.boundary_handler import MLBoundaryHandler

        # ---- Preferred path: full-domain HYCOM IC ----
        if 'ic_h0' in self.params and self.params['ic_h0'] is not None:
            h0_list = [np.array(hk, dtype=np.float64) for hk in self.params['ic_h0']]
            u0_list = [np.array(uk, dtype=np.float64) for uk in self.params['ic_u0']]
            v0_list = [np.array(vk, dtype=np.float64) for vk in self.params['ic_v0']]
            T0_list = [np.array(Tk, dtype=np.float64) for Tk in self.params['ic_T0']]
            if self.verbose:
                print("[MLSWE-LSMCMC] Using full-domain HYCOM initial conditions")

            # Still apply BC to enforce boundary consistency
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

        # ---- Fallback: rest-state + boundary-only ----
        if self.verbose:
            print("[MLSWE-LSMCMC] WARNING: No HYCOM IC provided, using rest state "
                  "(interior u=v=0, T=T_rest). This is usually WRONG.")
        # Initialize so that Σ hₖ = H_b (SSH = 0 initially)
        h0_list = []
        u0_list = []
        v0_list = []
        T0_list = []

        H_rest_total = sum(self.H_rest)

        for k in range(self.nlayers):
            h_k = np.full((self.ny, self.nx), self.H_rest[k], dtype=np.float64)
            if H_b is not None:
                if k < self.nlayers - 1:
                    # Upper layers: scale down for shallow bathymetry
                    ratio = np.where(H_b < H_rest_total,
                                     H_b / H_rest_total, 1.0)
                    h_k = np.maximum(self.H_rest[k] * ratio, 5.0)
                else:
                    # Bottom layer: fill remaining depth
                    h_above = sum(h0_list)
                    h_k = np.maximum(H_b - h_above, 10.0)
            h0_list.append(h_k)
            u0_list.append(np.zeros((self.ny, self.nx), dtype=np.float64))
            v0_list.append(np.zeros((self.ny, self.nx), dtype=np.float64))
            T0_list.append(np.full((self.ny, self.nx), self.T_rest[k],
                                    dtype=np.float64))

        # If BC handler exists, apply once to set boundary values
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
    def _advance_ensemble(self, models, nsteps, add_noise=True):
        """Advance all ensemble members by nsteps timesteps.

        Uses multiprocessing.Pool when self._use_mp is True. Each worker
        has its own MLSWE model initialised via _mp_init_worker.
        """
        Nf = len(models)
        blown = []

        if self._use_mp and self._mp_pool is not None:
            # ---- Parallel path ----
            args = [(mdl.state_flat.copy(), float(mdl.t), nsteps)
                    for mdl in models]
            results = self._mp_pool.map(_mp_advance, args)
            for j, res in enumerate(results):
                state_flat_new, t_new, blew_up = res
                if blew_up or not np.all(np.isfinite(state_flat_new)):
                    blown.append(j)
                    continue
                if add_noise:
                    state_flat_new += np.random.normal(scale=self.sig_x_vec)
                models[j].state_flat = state_flat_new
                models[j].t = t_new
                self.forecast[:, j] = models[j].state_flat.copy()
        else:
            # ---- Serial path ----
            for j in range(Nf):
                mdl = models[j]
                old_flat = mdl.state_flat.copy()
                for _ in range(nsteps):
                    mdl._timestep()
                    if not np.all(np.isfinite(mdl.state_flat)):
                        blown.append(j)
                        mdl.state_flat = old_flat  # revert
                        break
                else:
                    flat = mdl.state_flat.copy()
                    if add_noise:
                        flat += np.random.normal(scale=self.sig_x_vec)
                        mdl.state_flat = flat
                    self.forecast[:, j] = flat

        # Reset blown-up members to ensemble mean of healthy members
        if blown:
            healthy = [j for j in range(Nf) if j not in blown]
            if healthy:
                ens_mean = np.mean(self.forecast[:, healthy], axis=1)
                for j in blown:
                    perturbed = ens_mean + np.random.normal(
                        scale=0.05 * self.sig_x_vec)
                    models[j].state_flat = perturbed
                    self.forecast[:, j] = perturbed
            warnings.warn(f"Reset {len(blown)}/{Nf} blown-up ensemble members")

    # ------------------------------------------------------------------
    #  Localisation: find all cells in observed blocks (matches SWE_LSMCMC)
    # ------------------------------------------------------------------
    def get_observed_blocks_cells(self, obs_cell_indices):
        """
        Return flat indices into the full state vector of all grid cells
        that belong to a partition block containing >= 1 observation.

        Only includes **layer 0** fields (h0, u0, v0, T0), since observations
        are surface-only and deep layers have no observational constraint.
        This keeps the local analysis dimension at 4×cells (same as SWE),
        preventing spurious updates to unobserved deep-layer variables.

        Parameters
        ----------
        obs_cell_indices : 1-D int array
            Cell-level indices [0, ncells) of observed points.
        """
        if obs_cell_indices.size == 0:
            return np.array([], dtype=int)
        yobs_r, xobs_c = np.unravel_index(obs_cell_indices, (self.ny, self.nx))
        obs_blocks = np.unique(self.partition_labels[yobs_r, xobs_c])
        mask = np.isin(self.partition_labels, obs_blocks)
        ij = np.argwhere(mask)
        if ij.size == 0:
            return np.array([], dtype=int)
        flat0 = np.ravel_multi_index((ij[:, 0], ij[:, 1]), (self.ny, self.nx))
        # Only layer 0 fields: h0, u0, v0, T0 (4 fields)
        nfields_l0 = self.fields_per_layer   # 4
        all_idx = np.concatenate([flat0 + f * self.ncells
                                   for f in range(nfields_l0)])
        all_idx.sort()
        return all_idx

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    def run(self, H_b, bc_handler, obs_file, tstart):
        """Run the full LSMCMC filter."""
        # Load observations (from SWE_LSMCMC obs NetCDF)
        yobs, yind, obs_times, sig_y = self._load_obs(obs_file)

        # Initial state
        h0, u0, v0, T0 = self._make_init_state(H_b, bc_handler, tstart)
        model_kw = self._make_model_kwargs(H_b, bc_handler, tstart)

        # Create initial model & set initial state in lsmcmc_mean
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
                print(f"[MLSWE-LSMCMC] Starting multiprocessing pool "
                      f"with {n_workers} workers for {self.nforecast} forecasts")
            self._mp_pool = mp.Pool(n_workers, initializer=_mp_init_worker)
        else:
            if self.verbose:
                print(f"[MLSWE-LSMCMC] Serial mode (ncores=1)")

        # Create ensemble of models
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
            # All members identical (SMCMC theory: spread comes from noise)
            models.append(mdl)
            self.forecast[:, j] = mdl.state_flat.copy()
        if self.verbose:
            print(f"[MLSWE-LSMCMC] Ensemble init: {Nf} IDENTICAL members (SMCMC theory)")

        # Import posterior-sampling from SWE_LSMCMC
        try:
            from loc_smcmc_swe_exact_from_Gauss import (
                sample_posterior_mixture_sparse,
                gaussian_block_means,
                build_H_loc_from_global,
            )
        except ImportError:
            print("[MLSWE-LSMCMC] ERROR: Cannot import from SWE_LSMCMC. "
                  "Ensure ../SWE_LSMCMC is in PYTHONPATH.")
            raise

        nc = self.ncells

        # ---- Main assimilation loop ----
        t_freq = self.assim_timesteps
        rmse_vel_all = np.full(self.nassim, np.nan)
        rmse_sst_all = np.full(self.nassim, np.nan)
        rmse_ssh_all = np.full(self.nassim, np.nan)

        for cycle in range(self.nassim):
            t0_wall = time.time()

            # Advance ensemble
            self._advance_ensemble(models, t_freq, add_noise=(self.sig_x > 0))

            # Forecast mean
            forecast_mean = np.mean(self.forecast, axis=1)

            # Diagnostic: ensemble spread of h_total
            if self.verbose and ((cycle + 1) % 50 == 0 or cycle == 0):
                _nc = self.ncells
                h_std = np.std(self.forecast[0:_nc, :], axis=1)
                u_std = np.std(self.forecast[_nc:2*_nc, :], axis=1)
                print(f"  [spread] h_std: mean={h_std.mean():.3f} max={h_std.max():.3f}"
                      f"  u_std: mean={u_std.mean():.3f} max={u_std.max():.3f}")

            # Get observations for this cycle
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
                sig_y_cycle = sig_y[valid] if sig_y.size > 1 else float(sig_y)
            else:
                sig_y_cycle = float(sig_y)

            # Map single-layer obs_ind to multi-layer state vector
            obs_ind_ml = self._obs_ind_to_layer0(ind_valid)
            obs_ind_cell = self._obs_ind_to_cell(ind_valid)

            # Get all cells in blocks containing observations
            loc_cells = self.get_observed_blocks_cells(obs_ind_cell)
            if loc_cells.size == 0:
                self.lsmcmc_mean[cycle + 1] = forecast_mean
                continue

            # Build local H matrix
            Ho = build_H_loc_from_global(obs_ind_ml, loc_cells)

            # Per-obs sig_y: filter to mapped obs only
            if isinstance(sig_y_cycle, np.ndarray) and sig_y_cycle.size == y_valid.size:
                # Keep only obs that map into loc_cells
                col_set = set(loc_cells.tolist())
                kept = np.array([int(g) in col_set for g in obs_ind_ml])
                if kept.any():
                    sig_y_vec = sig_y_cycle[kept]
                    y_loc = y_valid[kept]
                else:
                    sig_y_vec = sig_y_cycle
                    y_loc = y_valid
            else:
                s = float(sig_y_cycle) if np.isscalar(sig_y_cycle) else float(sig_y_cycle.item())
                sig_y_vec = np.full(len(y_valid), s)
                y_loc = y_valid

            # Local state & noise
            sig_x_loc = self.sig_x_vec[loc_cells]

            # Sample posterior
            try:
                anal_ens_loc, mu = sample_posterior_mixture_sparse(
                    self.forecast[loc_cells].T,
                    Ho,
                    sig_x_loc,
                    sig_y_vec,
                    y_loc,
                    n_samples=self.mcmc_N,
                )
                anal_ens_loc, _ = gaussian_block_means(anal_ens_loc, self.nforecast)
            except Exception as e:
                warnings.warn(f"Cycle {cycle}: posterior sampling failed: {e}")
                self.lsmcmc_mean[cycle + 1] = forecast_mean
                continue

            # Update
            self.lsmcmc_mean[cycle + 1] = forecast_mean.copy()
            self.lsmcmc_mean[cycle + 1, loc_cells] = mu
            self.forecast[loc_cells, :] = anal_ens_loc

            # Post-analysis SSH relaxation: pull h_total back toward
            # H_b + eta_ref to prevent analysis-induced SSH drift.
            ssh_relax = float(self.params.get('ssh_relax_rate', 0.0))
            if ssh_relax > 0:
                nc_ = self.ncells
                fpl = self.fields_per_layer
                h_total_anal = self.lsmcmc_mean[cycle + 1, :nc_].copy()
                H_b_flat_ = H_b.ravel()
                eta_anal = h_total_anal - H_b_flat_
                # Get SSH ref at current model time
                t_now = float(models[0].t) if models else tstart
                ssh_ref_3d = self.params.get('ssh_relax_ref', None)
                ssh_ref_times = self.params.get('ssh_relax_ref_times', None)
                if ssh_ref_3d is not None and ssh_ref_times is not None:
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
                # Apply relaxation: fraction of the deviation to remove
                # Use a fixed fraction per analysis step (e.g. 0.5 = remove 50%)
                anal_relax_frac = float(self.params.get('ssh_analysis_relax_frac', 0.5))
                eta_correction = -anal_relax_frac * (eta_anal - eta_ref)
                h_total_corrected = h_total_anal + eta_correction
                self.lsmcmc_mean[cycle + 1, :nc_] = h_total_corrected
                # Also correct the forecast ensemble h-field
                for j_ens in range(self.forecast.shape[1]):
                    h_j = self.forecast[:nc_, j_ens]
                    eta_j = h_j - H_b_flat_
                    self.forecast[:nc_, j_ens] = h_j - anal_relax_frac * (eta_j - eta_ref)

            # Sync models with analysis for sequential DA
            Nf_ = self.forecast.shape[1]
            for j in range(Nf_):
                models[j].state_flat = \
                    self.forecast[:, j].copy()

            # RMSE
            z_a = self.lsmcmc_mean[cycle + 1]
            H_z = z_a[obs_ind_ml]
            residuals = H_z - y_valid
            nc = self.ncells
            vel_mask = ((obs_ind_ml >= nc) & (obs_ind_ml < 3*nc))
            sst_mask = (obs_ind_ml >= 3*nc) & (obs_ind_ml < 4*nc)
            ssh_mask = (obs_ind_ml >= 0) & (obs_ind_ml < nc)
            if vel_mask.sum() > 0:
                rmse_vel_all[cycle] = np.sqrt(np.mean(residuals[vel_mask]**2))
            if sst_mask.sum() > 0:
                rmse_sst_all[cycle] = np.sqrt(np.mean(residuals[sst_mask]**2))
            if ssh_mask.sum() > 0:
                rmse_ssh_all[cycle] = np.sqrt(np.mean(residuals[ssh_mask]**2))

            elapsed = time.time() - t0_wall
            if self.verbose and ((cycle + 1) % 10 == 0 or cycle == 0):
                # SSH diagnostic: h0 slot = h_total
                _nc = self.ncells
                _h_total = z_a[0:_nc]
                _ssh = _h_total - H_b.ravel()
                # Also show model-forecast SSH for comparison
                _fmean_h = forecast_mean[0:_nc]
                _fmean_ssh = _fmean_h - H_b.ravel()
                print(f"  Cycle {cycle+1}/{self.nassim}  "
                      f"vel_RMSE={rmse_vel_all[cycle]:.6f}  "
                      f"sst_RMSE={rmse_sst_all[cycle]:.4f}  "
                      f"ssh_RMSE={rmse_ssh_all[cycle]:.4f}  "
                      f"SSH=[{_ssh.min():.1f},{_ssh.max():.1f}]  "
                      f"fcst_SSH=[{_fmean_ssh.min():.1f},{_fmean_ssh.max():.1f}]  "
                      f"({elapsed:.1f}s)")

        # Clean up pool
        if self._mp_pool is not None:
            self._mp_pool.close()
            self._mp_pool.join()
            self._mp_pool = None
            if self.verbose:
                print("[MLSWE-LSMCMC] Multiprocessing pool closed.")

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
        # Per-obs noise
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
        """Save analysis mean, RMSE, and bathymetry to NetCDF."""
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, 'mlswe_lsmcmc_out.nc')
        nc = Dataset(outfile, 'w', format='NETCDF4')

        nc.createDimension('time', self.nassim + 1)
        nc.createDimension('layer', self.nlayers)
        nc.createDimension('field', self.fields_per_layer)
        nc.createDimension('y', self.ny)
        nc.createDimension('x', self.nx)
        nc.createDimension('cycle', self.nassim)

        # Reshape lsmcmc_mean to (time, layer, field, y, x)
        v = nc.createVariable('lsmcmc_mean', 'f4',
                              ('time', 'layer', 'field', 'y', 'x'), zlib=True)
        reshaped = self.lsmcmc_mean.reshape(
            self.nassim + 1, self.nlayers, self.fields_per_layer,
            self.ny, self.nx)
        v[:] = reshaped.astype(np.float32)

        vr = nc.createVariable('rmse_vel', 'f4', ('cycle',))
        vr[:] = self.rmse_vel.astype(np.float32)
        vs = nc.createVariable('rmse_sst', 'f4', ('cycle',))
        vs[:] = self.rmse_sst.astype(np.float32)
        vh = nc.createVariable('rmse_ssh', 'f4', ('cycle',))
        vh[:] = self.rmse_ssh.astype(np.float32)

        # Save bathymetry so plotter always uses the correct H_b
        if H_b is not None:
            vb = nc.createVariable('H_b', 'f4', ('y', 'x'), zlib=True)
            vb[:] = H_b.astype(np.float32)

        if obs_times is not None:
            # obs_times may have nassim entries; we need nassim+1 (incl. initial)
            ot = obs_times[:self.nassim + 1]
            if len(ot) < self.nassim + 1:
                # Prepend t0 = first_obs - dt_obs if we have at least 2
                if len(obs_times) >= 2:
                    dt_obs = obs_times[1] - obs_times[0]
                    ot = np.concatenate([[obs_times[0] - dt_obs], ot])
                else:
                    ot = np.concatenate([[obs_times[0]], ot])
                ot = ot[:self.nassim + 1]
            vt = nc.createVariable('obs_times', 'f8', ('time',))
            vt[:] = ot

        nc.nlayers = self.nlayers
        nc.fields_per_layer = self.fields_per_layer
        nc.ny = self.ny
        nc.nx = self.nx
        nc.close()
        print(f"[MLSWE-LSMCMC] Saved results to {outfile}")

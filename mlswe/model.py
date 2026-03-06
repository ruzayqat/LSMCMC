"""
mlswe.model  –  3-layer primitive-equations (multi-layer shallow water) solver
==============================================================================

Solves the Boussinesq, hydrostatic, multi-layer shallow-water equations on
an equidistant A-grid with 4th-order Runge–Kutta time integration.

Layers are numbered  k = 0 (surface),  1 (thermocline),  2 (deep / bottom).
Densities  ρ₀ < ρ₁ < ρ₂  are constant within each layer.

**Per-layer prognostic variables:** hₖ (thickness), uₖ, vₖ, Tₖ

**State vector ordering (flat):**
    [ h₀, u₀, v₀, T₀,  h₁, u₁, v₁, T₁,  h₂, u₂, v₂, T₂ ]
    each of shape (ny, nx)  →  dimx = n_layers × 4 × ny × nx

Pressure-gradient force in layer k (Boussinesq):

    F_k = −g ∇η  +  Σ_{j<k}  g'_{j,k} ∇hⱼ

where
    η = Σₖ hₖ − H_b   (sea surface height)
    g'_{j,k} = g (ρₖ − ρⱼ) / ρ_ref

This gives baroclinic compensation: when warm (light) surface water piles
up, the interface tilts and the pressure gradient in deeper layers is
reduced, keeping deep-water currents weak — matching HYCOM-like behaviour.

References
----------
* Cushman-Roisin & Beckers (2011), Ch. 11 — Multi-layer models
* Hallberg (1997), "Stable split time stepping schemes for large-scale
  ocean modelling", J. Comput. Phys.
"""
from __future__ import print_function
import numpy as np
import warnings

# ---------------------------------------------------------------------------
#  Physical constants
# ---------------------------------------------------------------------------
EARTH_RADIUS = 6.371e6       # m
OMEGA        = 7.2921e-5     # rad/s
GRAVITY      = 9.81          # m/s²
RHO_REF      = 1025.0        # reference density (kg/m³)


# ---------------------------------------------------------------------------
#  Bathymetry smoothing (Beckmann & Haidvogel, 1993)
# ---------------------------------------------------------------------------
def smooth_bathymetry(H_b, r_max=0.2, max_iter=200, verbose=True):
    """
    Iteratively smooth bathymetry to satisfy the Beckmann-Haidvogel (1993)
    slope-parameter criterion

        r = |H(i) - H(j)| / (H(i) + H(j))  <  r_max

    for every pair of adjacent grid cells (i, j).  This prevents
    pressure-gradient errors and solution blow-up near steep topography
    (islands, continental shelves) in multi-layer ocean models.

    The algorithm applies local Laplacian smoothing only to cells whose
    neighbours violate the criterion, preserving deep-ocean bathymetry.

    Parameters
    ----------
    H_b : ndarray (ny, nx)
        Bathymetry (positive, ocean depth) [m].
    r_max : float
        Maximum allowed slope parameter.  Typical values:
        0.2 for sigma-coordinate models, 0.3-0.5 for layer models.
    max_iter : int
        Maximum smoothing iterations.
    verbose : bool
        Print convergence info.

    Returns
    -------
    H_smooth : ndarray (ny, nx)
        Smoothed bathymetry.
    """
    H = H_b.copy().astype(np.float64)
    ny, nx = H.shape

    for it in range(max_iter):
        # Slope parameter in x-direction: (ny, nx-1)
        denom_x = H[:, 1:] + H[:, :-1]
        np.maximum(denom_x, 1.0, out=denom_x)
        rx = np.abs(H[:, 1:] - H[:, :-1]) / denom_x

        # Slope parameter in y-direction: (ny-1, nx)
        denom_y = H[1:, :] + H[:-1, :]
        np.maximum(denom_y, 1.0, out=denom_y)
        ry = np.abs(H[1:, :] - H[:-1, :]) / denom_y

        max_r = max(float(rx.max()), float(ry.max()))
        if max_r <= r_max:
            if verbose:
                print(f"[smooth_bathy] Converged after {it} iterations, "
                      f"max_r = {max_r:.4f}")
            break

        # Mark cells involved in a violation
        mask = np.zeros((ny, nx), dtype=bool)
        viol_x = rx > r_max
        viol_y = ry > r_max
        mask[:, :-1] |= viol_x
        mask[:, 1:]  |= viol_x
        mask[:-1, :] |= viol_y
        mask[1:, :]  |= viol_y

        # 5-point neighbourhood average
        H_sum = H.copy()
        cnt = np.ones_like(H)
        H_sum[:, 1:]  += H[:, :-1]; cnt[:, 1:]  += 1
        H_sum[:, :-1] += H[:, 1:];  cnt[:, :-1] += 1
        H_sum[1:, :]  += H[:-1, :]; cnt[1:, :]  += 1
        H_sum[:-1, :] += H[1:, :];  cnt[:-1, :] += 1
        H_avg = H_sum / cnt

        # Blend only at violating cells (alpha = 0.5)
        H = np.where(mask, 0.5 * H + 0.5 * H_avg, H)
    else:
        if verbose:
            print(f"[smooth_bathy] Did NOT converge after {max_iter} iters, "
                  f"max_r = {max_r:.4f}")

    return H


class MLSWE:
    r"""
    3-layer primitive-equations shallow-water model.

    Parameters
    ----------
    h0 : list of 3 ndarrays (ny, nx)
        Initial layer thicknesses [m], top-to-bottom.
    u0, v0 : list of 3 ndarrays (ny, nx)
        Initial velocities [m/s].
    T0 : list of 3 ndarrays (ny, nx) or None
        Initial temperatures [K].  None → no tracer.
    rho : array-like (3,)
        Layer densities [kg/m³], light to dense.
    dx, dy : float
        Grid spacing [m].
    dt : float
        Time step [s].
    f0 : float or ndarray (ny,1) or (ny,nx)
        Coriolis parameter [1/s].
    g : float
        Gravitational acceleration [m/s²].
    H_b : ndarray (ny, nx) or None
        Bathymetry (ocean floor depth, positive downward) [m].
    H_mean : float
        Mean depth [m], used when H_b is None.
    H_rest : array-like (3,)
        Rest-state layer thicknesses [m], top to bottom.
        Σ H_rest ≈ H_mean (or mean of H_b).
    bottom_drag : float
        Linear bottom friction [1/s] applied to bottom layer only.
    diff_coeff : float
        Laplacian diffusion for momentum [m²/s].
    diff_order : int
        1 → Laplacian, 2 → bi-harmonic.
    tracer_diff : float
        Diffusion coefficient for temperature [m²/s].
    bc_handler : callable or None
        ``bc_handler(state_dict, t) → state_dict``
    tstart : float
        Initial model time [s].
    precision : str
        'single' or 'double'.
    """

    N_LAYERS = 3
    FIELDS_PER_LAYER = 4   # h, u, v, T

    def __init__(
        self,
        h0, u0, v0,
        T0=None,
        rho=(1023.0, 1026.0, 1028.0),
        dx=10.0e3,
        dy=10.0e3,
        dt=60.0,
        f0=1.0e-4,
        g=GRAVITY,
        H_b=None,
        H_mean=4000.0,
        H_rest=(100.0, 400.0, 3500.0),
        bottom_drag=1.0e-6,
        diff_coeff=500.0,
        diff_order=1,
        tracer_diff=100.0,
        bc_handler=None,
        tstart=0.0,
        precision="double",
        sst_nudging_rate=0.0,
        sst_nudging_ref=None,
        sst_nudging_ref_times=None,
        # SSH relaxation: dh_total/dt += -ssh_relax_rate * (h_total - H_b)
        ssh_relax_rate=0.0,
        ssh_relax_ref=None,
        ssh_relax_ref_times=None,
        # Surface heat flux (Newtonian cooling): Q = -alpha*(T - T_air)
        sst_flux_type=None,
        sst_alpha=15.0,
        sst_h_mix=50.0,
        sst_T_air=None,
        sst_T_air_times=None,
        # Barotropic velocity projection (force all layers to same velocity)
        # Set to False to allow baroclinic velocity structure (e.g., for LETKF)
        apply_barotropic_projection=True,
        # Interior floor for boundary-local SSH relaxation mask
        ssh_relax_interior_floor=0.1,
        # Enhanced drag near shallow topography (islands)
        shallow_drag_depth=500.0,
        shallow_drag_coeff=5.0e-4,
    ):
        dtype = np.float64 if precision == "double" else np.float32
        self.dtype = dtype
        NL = self.N_LAYERS

        self.ny, self.nx = h0[0].shape
        self.dx = dtype(dx)
        self.dy = dtype(dy)
        self.dt = dtype(dt)
        self.g  = dtype(g)
        self.H_mean = dtype(H_mean)
        self.bottom_drag = dtype(bottom_drag)
        self.diff_coeff = dtype(diff_coeff)
        self.diff_order = int(diff_order)
        self.tracer_diff = dtype(tracer_diff)
        self.t = dtype(tstart)
        self.bc_handler = bc_handler
        self.apply_barotropic_projection = bool(apply_barotropic_projection)
        self._ssh_relax_floor = float(ssh_relax_interior_floor)

        # Layer densities & reduced-gravity matrix
        self.rho = np.array(rho, dtype=dtype)
        assert len(self.rho) == NL
        assert np.all(np.diff(self.rho) > 0), "Densities must increase top→bottom"
        self.rho_ref = dtype(RHO_REF)

        # g'_{j,k} = g (ρ_k - ρ_j) / ρ_ref   for j < k
        self.gprime = np.zeros((NL, NL), dtype=dtype)
        for k in range(NL):
            for j in range(k):
                self.gprime[j, k] = self.g * (self.rho[k] - self.rho[j]) / self.rho_ref

        # Rest-state thicknesses
        self.H_rest = np.array(H_rest, dtype=dtype)
        assert len(self.H_rest) == NL
        self.H_rest_total = float(self.H_rest.sum())

        # Coriolis
        f_arr = np.asarray(f0, dtype=dtype)
        if f_arr.ndim == 0:
            self.f = np.full((self.ny, self.nx), f_arr, dtype=dtype)
        elif f_arr.ndim == 1:
            self.f = np.broadcast_to(f_arr[:, None],
                                     (self.ny, self.nx)).copy().astype(dtype)
        else:
            self.f = f_arr.copy().astype(dtype)

        # Bathymetry
        if H_b is not None:
            self.H_b = np.array(H_b, dtype=dtype, copy=True)
            self.dHb_dx = self._ddx(self.H_b, self.dx)
            self.dHb_dy = self._ddy(self.H_b, self.dy)
        else:
            self.H_b = None
            self.dHb_dx = None
            self.dHb_dy = None

        # Enhanced drag near shallow topography (islands, shelves)
        # Ramps linearly from 0 at H_b = shallow_drag_depth to
        # shallow_drag_coeff at H_b = H_min.  This damps spurious
        # velocities generated by steep bathymetry gradients near
        # islands without affecting deep-ocean dynamics.
        self.shallow_drag_coeff = dtype(shallow_drag_coeff)
        self.shallow_drag_depth = dtype(shallow_drag_depth)
        if self.H_b is not None and self.shallow_drag_coeff > 0:
            depth_ratio = np.clip(
                1.0 - self.H_b / float(self.shallow_drag_depth), 0.0, 1.0)
            self.shallow_drag_mask = (
                self.shallow_drag_coeff * depth_ratio).astype(dtype)
        else:
            self.shallow_drag_mask = np.zeros(
                (self.ny, self.nx), dtype=dtype)

        # State arrays: lists of 3 (ny, nx) arrays
        self.h = [np.array(h0[k], dtype=dtype, copy=True) for k in range(NL)]
        self.u = [np.array(u0[k], dtype=dtype, copy=True) for k in range(NL)]
        self.v = [np.array(v0[k], dtype=dtype, copy=True) for k in range(NL)]

        self.use_tracer = T0 is not None
        if self.use_tracer:
            self.T = [np.array(T0[k], dtype=dtype, copy=True) for k in range(NL)]
        else:
            self.T = [None] * NL

        # CFL check (barotropic mode dominates)
        H_max = float(np.max(self.H_b)) if self.H_b is not None else float(self.H_mean)
        c_max = np.sqrt(float(self.g) * H_max)
        cfl_dt = 0.5 * min(float(self.dx), float(self.dy)) / c_max
        if float(self.dt) > cfl_dt:
            warnings.warn(
                f"CFL violation: dt={float(self.dt):.1f}s > limit "
                f"{cfl_dt:.1f}s (c_max={c_max:.1f} m/s). "
                f"Reducing to {0.9*cfl_dt:.1f}s.",
                RuntimeWarning, stacklevel=2)
            self.dt = dtype(0.9 * cfl_dt)

        # SST nudging (Newtonian relaxation) for layer-0 temperature
        self.sst_nudging_rate = dtype(sst_nudging_rate)
        if sst_nudging_ref is not None and np.ndim(sst_nudging_ref) == 3:
            self._sst_ref_3d = np.asarray(sst_nudging_ref, dtype=dtype)
            self._sst_ref_times = np.asarray(sst_nudging_ref_times, dtype=np.float64)
        elif sst_nudging_ref is not None:
            self._sst_ref_3d = None
            self._sst_ref_static = np.asarray(sst_nudging_ref, dtype=dtype)
            self._sst_ref_times = None
        else:
            self._sst_ref_3d = None
            self._sst_ref_static = None
            self._sst_ref_times = None

        # SSH relaxation (Newtonian toward reference or zero)
        self.ssh_relax_rate = dtype(ssh_relax_rate)
        if ssh_relax_ref is not None and np.ndim(ssh_relax_ref) == 3:
            self._ssh_ref_3d = np.asarray(ssh_relax_ref, dtype=dtype)
            self._ssh_ref_times = np.asarray(ssh_relax_ref_times, dtype=np.float64)
        elif ssh_relax_ref is not None:
            self._ssh_ref_3d = None
            self._ssh_ref_static = np.asarray(ssh_relax_ref, dtype=dtype)
            self._ssh_ref_times = None
        else:
            self._ssh_ref_3d = None
            self._ssh_ref_static = None
            self._ssh_ref_times = None

        # Surface heat flux (Newtonian cooling)
        RHO_SW = 1025.0
        CP_SW = 3990.0
        self.sst_flux_type = sst_flux_type
        self.sst_alpha = dtype(sst_alpha)
        self.sst_h_mix = dtype(sst_h_mix)
        if sst_flux_type == 'newtonian':
            self._flux_lambda = dtype(sst_alpha / (RHO_SW * CP_SW * sst_h_mix))
        else:
            self._flux_lambda = dtype(0.0)
        if sst_T_air is not None and np.ndim(sst_T_air) == 3:
            self._T_air_3d = np.asarray(sst_T_air, dtype=dtype)
            self._T_air_times = np.asarray(sst_T_air_times, dtype=np.float64)
        elif sst_T_air is not None:
            self._T_air_3d = None
            self._T_air_static = np.asarray(sst_T_air, dtype=dtype)
            self._T_air_times = None
        else:
            self._T_air_3d = None
            self._T_air_static = None
            self._T_air_times = None

        self.timesteps = 1

    # ------------------------------------------------------------------
    #  State as flat vector  [h0,u0,v0,T0, h1,u1,v1,T1, h2,u2,v2,T2]
    # ------------------------------------------------------------------
    @property
    def nfields(self):
        """Total number of 2-D fields in the state."""
        fpl = self.FIELDS_PER_LAYER if self.use_tracer else (self.FIELDS_PER_LAYER - 1)
        return self.N_LAYERS * fpl

    @property
    def state_flat(self):
        """Return flat state vector.

        To make the DA problem equivalent to SWE, the h0 slot stores
        h_total (= Σhₖ) instead of the layer-0 thickness.  Slots h1, h2
        are set to zero (unused).  The setter reverses this mapping.
        """
        n = self.ny * self.nx
        NL = self.N_LAYERS
        h_total = self.h[0] + self.h[1] + self.h[2]
        parts = []
        for k in range(NL):
            if k == 0:
                parts.append(h_total.ravel())           # h0 slot ← h_total
            else:
                parts.append(np.zeros(n, dtype=self.dtype))  # h1,h2 slots ← 0
            parts.extend([self.u[k].ravel(), self.v[k].ravel()])
            if self.use_tracer:
                parts.append(self.T[k].ravel())
        return np.concatenate(parts)

    @state_flat.setter
    def state_flat(self, vec):
        """Load state from flat vector.

        h0 slot is interpreted as h_total.  Individual layer thicknesses
        are derived: hₖ = (H_rest_k / H_rest_total) × max(h_total, 10).
        h1, h2 slots are ignored.
        """
        n = self.ny * self.nx
        NL = self.N_LAYERS
        fpl = self.FIELDS_PER_LAYER if self.use_tracer else (self.FIELDS_PER_LAYER - 1)

        # Read h_total from h0 slot
        h_total = vec[0:n].reshape(self.ny, self.nx).astype(self.dtype)
        np.maximum(h_total, 10.0, out=h_total)  # match SWE h≥10
        for k in range(NL):
            self.h[k] = (self.H_rest[k] / self.H_rest_total) * h_total

        # Read u, v, T for each layer (skip h slots for layers 1,2)
        for k in range(NL):
            off = k * fpl * n
            self.u[k] = vec[off+n:off+2*n].reshape(self.ny, self.nx).astype(self.dtype)
            self.v[k] = vec[off+2*n:off+3*n].reshape(self.ny, self.nx).astype(self.dtype)
            if self.use_tracer:
                self.T[k] = vec[off+3*n:off+4*n].reshape(self.ny, self.nx).astype(self.dtype)

    @property
    def dimx(self):
        return self.nfields * self.ny * self.nx

    # ------------------------------------------------------------------
    #  Sea surface height
    # ------------------------------------------------------------------
    def ssh(self):
        """Compute SSH = Σ hₖ − H_b  (or − H_mean for flat bottom)."""
        h_total = sum(self.h)
        if self.H_b is not None:
            return h_total - self.H_b
        else:
            return h_total - self.H_mean

    # ------------------------------------------------------------------
    #  SST nudging reference
    # ------------------------------------------------------------------
    def _get_sst_ref(self, t):
        """Return T_ref(t) for SST nudging (time-interpolated if 3-D)."""
        if self._sst_ref_3d is not None:
            times = self._sst_ref_times
            if t <= times[0]:
                return self._sst_ref_3d[0]
            if t >= times[-1]:
                return self._sst_ref_3d[-1]
            idx = int(np.searchsorted(times, t)) - 1
            idx = min(idx, len(times) - 2)
            dt_snap = float(times[idx + 1] - times[idx])
            if dt_snap == 0:
                return self._sst_ref_3d[idx]
            w = (t - times[idx]) / dt_snap
            return (1.0 - w) * self._sst_ref_3d[idx] + w * self._sst_ref_3d[idx + 1]
        if self._sst_ref_static is not None:
            return self._sst_ref_static
        return None

    def _get_ssh_ref(self, t):
        """Return SSH reference(t) for SSH relaxation (time-interpolated if 3-D)."""
        if hasattr(self, '_ssh_ref_3d') and self._ssh_ref_3d is not None:
            times = self._ssh_ref_times
            if t <= times[0]:
                return self._ssh_ref_3d[0]
            if t >= times[-1]:
                return self._ssh_ref_3d[-1]
            idx = int(np.searchsorted(times, t)) - 1
            idx = min(idx, len(times) - 2)
            dt_snap = float(times[idx + 1] - times[idx])
            if dt_snap == 0:
                return self._ssh_ref_3d[idx]
            w = (t - times[idx]) / dt_snap
            return (1.0 - w) * self._ssh_ref_3d[idx] + w * self._ssh_ref_3d[idx + 1]
        if hasattr(self, '_ssh_ref_static') and self._ssh_ref_static is not None:
            return self._ssh_ref_static
        return None

    def _get_ssh_relax_mask(self):
        """
        Return the SSH relaxation weight mask (ny, nx).

        Based on the sponge mask from the BC handler, but with a small
        non-zero floor in the interior to prevent unbounded SSH drift.

        Structure:
          - Ghost cells: weight = 1.0 (full relaxation)
          - Sponge zone:  exponential decay (strong → weak)
          - Interior:     weight = ssh_relax_interior_floor (mild relaxation)
        """
        if self.bc_handler is not None and hasattr(self.bc_handler, 'sponge_mask'):
            floor = getattr(self, '_ssh_relax_floor', 0.1)
            mask = np.maximum(self.bc_handler.sponge_mask, floor)
            return mask
        return None

    def _get_T_air(self, t):
        """Return T_air(t) for Newtonian heat flux (time-interpolated if 3-D)."""
        if hasattr(self, '_T_air_3d') and self._T_air_3d is not None:
            times = self._T_air_times
            if t <= times[0]:
                return self._T_air_3d[0]
            if t >= times[-1]:
                return self._T_air_3d[-1]
            idx = int(np.searchsorted(times, t)) - 1
            idx = min(idx, len(times) - 2)
            dt_snap = float(times[idx + 1] - times[idx])
            if dt_snap == 0:
                return self._T_air_3d[idx]
            w = (t - times[idx]) / dt_snap
            return (1.0 - w) * self._T_air_3d[idx] + w * self._T_air_3d[idx + 1]
        if hasattr(self, '_T_air_static') and self._T_air_static is not None:
            return self._T_air_static
        return None

    # ------------------------------------------------------------------
    #  Spatial derivatives (2nd order centred, A-grid)
    # ------------------------------------------------------------------
    @staticmethod
    def _ddx(q, dx):
        dq = np.empty_like(q)
        dq[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2.0 * dx)
        dq[:, 0]  = (q[:, 1]  - q[:, 0])  / dx
        dq[:, -1] = (q[:, -1] - q[:, -2]) / dx
        return dq

    @staticmethod
    def _ddy(q, dy):
        dq = np.empty_like(q)
        dq[1:-1, :] = (q[2:, :] - q[:-2, :]) / (2.0 * dy)
        dq[0, :]  = (q[1, :]  - q[0, :])  / dy
        dq[-1, :] = (q[-1, :] - q[-2, :]) / dy
        return dq

    @staticmethod
    def _laplacian(q, dx, dy):
        lap = np.zeros_like(q)
        lap[1:-1, 1:-1] = (
            (q[1:-1, 2:] - 2.0*q[1:-1, 1:-1] + q[1:-1, :-2]) / (dx*dx) +
            (q[2:, 1:-1] - 2.0*q[1:-1, 1:-1] + q[:-2, 1:-1]) / (dy*dy)
        )
        return lap

    def _diffusion(self, q):
        if self.diff_coeff == 0.0:
            return np.zeros_like(q)
        d = self._laplacian(q, self.dx, self.dy)
        for _ in range(1, self.diff_order):
            d = self._laplacian(d, self.dx, self.dy)
        sign = 1.0 if (self.diff_order % 2 == 1) else -1.0
        return sign * self.diff_coeff * d

    # ------------------------------------------------------------------
    #  RHS (tendencies) for all layers
    # ------------------------------------------------------------------
    def rhs(self, h, u, v, T=None):
        """
        Compute tendencies (dh/dt, du/dt, dv/dt, dT/dt) for each layer.

        The pressure gradient in layer k (Boussinesq, hydrostatic):

            PGF_k = −g ∇η  +  Σ_{j<k} g'_{j,k} ∇hⱼ

        where η = (Σ hₖ) − H_b  is the sea surface height.

        Parameters
        ----------
        h, u, v : lists of 3 arrays (ny, nx)
        T : list of 3 arrays or Nones

        Returns
        -------
        dh, du, dv, dT : lists of 3 arrays (ny, nx)
        """
        NL = self.N_LAYERS
        ddx, ddy = self._ddx, self._ddy
        dx, dy, g = self.dx, self.dy, self.g

        # SSH and its gradient (barotropic pressure)
        h_total = h[0] + h[1] + h[2]
        if self.H_b is not None:
            eta = h_total - self.H_b
        else:
            eta = h_total - self.H_mean
        deta_dx = ddx(eta, dx)
        deta_dy = ddy(eta, dy)

        du_dt = [None] * NL
        dv_dt = [None] * NL
        dT_dt = [None] * NL

        # --- Barotropic continuity (single equation, like SWE) ---
        # ∂h_total/∂t = −∇·(Σ hₖ uₖ) + ν∇²h_total
        flux_x = sum(h[k] * u[k] for k in range(NL))
        flux_y = sum(h[k] * v[k] for k in range(NL))
        dh_total_dt = (-ddx(flux_x, dx)
                       - ddy(flux_y, dy)
                       + self._diffusion(h_total))

        # SSH relaxation: dh_total/dt += -lambda_ssh * (eta - eta_ref)
        # When a sponge mask is available from the BC handler, the relaxation
        # rate is modulated spatially: strong near boundaries, zero in interior.
        # This prevents the global relaxation from fighting DA corrections.
        if self.ssh_relax_rate > 0:
            ssh_ref = self._get_ssh_ref(self.t)
            if ssh_ref is not None:
                ssh_anom = eta - ssh_ref
            else:
                ssh_anom = eta

            # Modulate by sponge mask if available
            relax_mask = self._get_ssh_relax_mask()
            if relax_mask is not None:
                dh_total_dt = dh_total_dt - self.ssh_relax_rate * relax_mask * ssh_anom
            else:
                dh_total_dt = dh_total_dt - self.ssh_relax_rate * ssh_anom

        # Distribute to layers proportionally (preserves layer structure)
        dh_dt = [None] * NL
        for k in range(NL):
            dh_dt[k] = (self.H_rest[k] / self.H_rest_total) * dh_total_dt

        for k in range(NL):
            # --- Pressure gradient for layer k ---
            # Pure barotropic PGF (same as SWE): -g ∂η/∂x
            # NOTE: Baroclinic PGF is removed because with proportional
            # layering (h_k = frac_k * h_total) it introduces a spurious
            # ∇H_b forcing term that drives persistent SSH drift.
            # The velocity averaging already removes any baroclinic shear,
            # so the reduced-gravity PGF has no physical effect anyway.
            pgf_x = -g * deta_dx
            pgf_y = -g * deta_dy

            # --- Momentum: duₖ/dt ---
            # Bottom drag applied to ALL layers (matches SWE behaviour
            # where drag acts on the depth-averaged velocity)
            # Includes enhanced drag near shallow topography (islands)
            total_drag = self.bottom_drag + self.shallow_drag_mask
            drag_u = -total_drag * u[k]
            drag_v = -total_drag * v[k]

            du_dt[k] = (-u[k] * ddx(u[k], dx)
                        - v[k] * ddy(u[k], dy)
                        + self.f * v[k]
                        + pgf_x
                        + drag_u
                        + self._diffusion(u[k]))

            dv_dt[k] = (-u[k] * ddx(v[k], dx)
                        - v[k] * ddy(v[k], dy)
                        - self.f * u[k]
                        + pgf_y
                        + drag_v
                        + self._diffusion(v[k]))

            # --- Tracer (temperature) ---
            if T is not None and T[k] is not None:
                dT_dt[k] = (-u[k] * ddx(T[k], dx)
                            - v[k] * ddy(T[k], dy)
                            + self.tracer_diff * self._laplacian(T[k], dx, dy))
                # SST nudging (layer 0 only): dT/dt += -λ (T - T_ref)
                if k == 0 and self.sst_nudging_rate > 0:
                    T_ref = self._get_sst_ref(self.t)
                    if T_ref is not None:
                        dT_dt[k] = dT_dt[k] - self.sst_nudging_rate * (T[k] - T_ref)
                # Surface heat flux (layer 0 only): dT/dt += -α/(ρ·cp·h_mix) * (T - T_air)
                if k == 0 and self._flux_lambda > 0:
                    T_air = self._get_T_air(self.t)
                    if T_air is not None:
                        dT_dt[k] = dT_dt[k] - self._flux_lambda * (T[k] - T_air)
            else:
                dT_dt[k] = np.zeros((self.ny, self.nx), dtype=self.dtype)

        return dh_dt, du_dt, dv_dt, dT_dt

    # ------------------------------------------------------------------
    #  Boundary conditions
    # ------------------------------------------------------------------
    def _apply_bc(self, h, u, v, t, T=None):
        """Apply external boundary conditions or zero-gradient Neumann."""
        NL = self.N_LAYERS
        if self.bc_handler is not None:
            state = {}
            for k in range(NL):
                state[f'h{k}'] = h[k]
                state[f'u{k}'] = u[k]
                state[f'v{k}'] = v[k]
                if T is not None and T[k] is not None:
                    state[f'T{k}'] = T[k]
            state = self.bc_handler(state, t)
            for k in range(NL):
                h[k] = state[f'h{k}']
                u[k] = state[f'u{k}']
                v[k] = state[f'v{k}']
                if T is not None and T[k] is not None:
                    T[k] = state.get(f'T{k}', T[k])
        else:
            # Zero-gradient (Neumann) boundaries
            for k in range(NL):
                for q in [h[k], u[k], v[k]]:
                    q[0, :]  = q[1, :]
                    q[-1, :] = q[-2, :]
                    q[:, 0]  = q[:, 1]
                    q[:, -1] = q[:, -2]
                if T is not None and T[k] is not None:
                    T[k][0, :]  = T[k][1, :]
                    T[k][-1, :] = T[k][-2, :]
                    T[k][:, 0]  = T[k][:, 1]
                    T[k][:, -1] = T[k][:, -2]
        return h, u, v, T

    # ------------------------------------------------------------------
    #  Single RK4 time step
    # ------------------------------------------------------------------
    def _timestep(self):
        dt = self.dt
        NL = self.N_LAYERS
        has_T = self.use_tracer

        # Save initial state
        h0 = [hk.copy() for hk in self.h]
        u0 = [uk.copy() for uk in self.u]
        v0 = [vk.copy() for vk in self.v]
        T0 = [Tk.copy() if Tk is not None else None for Tk in self.T]

        def _copy_list(lst):
            return [a.copy() if a is not None else None for a in lst]

        def _add_lists(a, b, coeff):
            return [a[k] + coeff * b[k] if a[k] is not None else None
                    for k in range(NL)]

        # ---- k1 ----
        dh1, du1, dv1, dT1 = self.rhs(h0, u0, v0, T0 if has_T else None)

        def _project_h(h_list):
            """Project to proportional layer structure at RK4 intermediates."""
            ht = h_list[0] + h_list[1] + h_list[2]
            np.maximum(ht, 10.0, out=ht)
            for kk in range(NL):
                h_list[kk] = (self.H_rest[kk] / self.H_rest_total) * ht
            return h_list

        # ---- k2 ----
        h2 = _project_h(_add_lists(h0, dh1, 0.5 * dt))
        u2 = _add_lists(u0, du1, 0.5 * dt)
        v2 = _add_lists(v0, dv1, 0.5 * dt)
        T2 = _add_lists(T0, dT1, 0.5 * dt) if has_T else T0
        h2, u2, v2, T2 = self._apply_bc(h2, u2, v2, self.t + 0.5*dt, T2)
        dh2, du2, dv2, dT2 = self.rhs(h2, u2, v2, T2 if has_T else None)

        # ---- k3 ----
        h3 = _project_h(_add_lists(h0, dh2, 0.5 * dt))
        u3 = _add_lists(u0, du2, 0.5 * dt)
        v3 = _add_lists(v0, dv2, 0.5 * dt)
        T3 = _add_lists(T0, dT2, 0.5 * dt) if has_T else T0
        h3, u3, v3, T3 = self._apply_bc(h3, u3, v3, self.t + 0.5*dt, T3)
        dh3, du3, dv3, dT3 = self.rhs(h3, u3, v3, T3 if has_T else None)

        # ---- k4 ----
        h4 = _project_h(_add_lists(h0, dh3, dt))
        u4 = _add_lists(u0, du3, dt)
        v4 = _add_lists(v0, dv3, dt)
        T4 = _add_lists(T0, dT3, dt) if has_T else T0
        h4, u4, v4, T4 = self._apply_bc(h4, u4, v4, self.t + dt, T4)
        dh4, du4, dv4, dT4 = self.rhs(h4, u4, v4, T4 if has_T else None)

        # ---- Combine ----
        dt6 = dt / 6.0
        for k in range(NL):
            self.h[k] = h0[k] + dt6 * (dh1[k] + 2*dh2[k] + 2*dh3[k] + dh4[k])
            self.u[k] = u0[k] + dt6 * (du1[k] + 2*du2[k] + 2*du3[k] + du4[k])
            self.v[k] = v0[k] + dt6 * (dv1[k] + 2*dv2[k] + 2*dv3[k] + dv4[k])
            if has_T and T0[k] is not None:
                self.T[k] = T0[k] + dt6 * (dT1[k] + 2*dT2[k] + 2*dT3[k] + dT4[k])

        # Final BC
        self.h, self.u, self.v, self.T = self._apply_bc(
            self.h, self.u, self.v, self.t + dt, self.T)

        # Project h to proportional layer structure (ensures mass consistency)
        h_total = self.h[0] + self.h[1] + self.h[2]
        np.maximum(h_total, 10.0, out=h_total)  # match SWE h≥10
        for k in range(NL):
            self.h[k] = (self.H_rest[k] / self.H_rest_total) * h_total

        # Barotropic velocity projection: force all layers to have the
        # same (depth-averaged) velocity.  This prevents inter-layer
        # velocity divergence from accumulating into SSH drift, while
        # still allowing per-layer temperature advection.
        # u_bar = Σ(h_k u_k) / h_total  (mass-weighted average)
        if self.apply_barotropic_projection:
            u_bar = sum(self.h[k] * self.u[k] for k in range(NL)) / np.maximum(h_total, 1.0)
            v_bar = sum(self.h[k] * self.v[k] for k in range(NL)) / np.maximum(h_total, 1.0)
            for k in range(NL):
                self.u[k] = u_bar.copy()
                self.v[k] = v_bar.copy()

        # Velocity clamp (same as SWE: ±20 m/s)
        for k in range(NL):
            np.clip(self.u[k], -20.0, 20.0, out=self.u[k])
            np.clip(self.v[k], -20.0, 20.0, out=self.v[k])
            if has_T and self.T[k] is not None:
                np.clip(self.T[k], 260.0, 310.0, out=self.T[k])

        # NaN recovery (rare)
        for k in range(NL):
            if not np.isfinite(self.h[k]).all():
                self.h[k][~np.isfinite(self.h[k])] = self.H_rest[k]
            if not np.isfinite(self.u[k]).all():
                self.u[k][~np.isfinite(self.u[k])] = 0.0
            if not np.isfinite(self.v[k]).all():
                self.v[k][~np.isfinite(self.v[k])] = 0.0
            if has_T and self.T[k] is not None and not np.isfinite(self.T[k]).all():
                self.T[k][~np.isfinite(self.T[k])] = 280.0

        self.t += dt

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------
    def advance(self):
        """Advance by ``self.timesteps`` steps."""
        for _ in range(self.timesteps):
            self._timestep()
        return [hk.copy() for hk in self.h]

    def cfl(self):
        """Current CFL number (barotropic mode)."""
        if self.H_b is not None:
            H_max = float(np.max(sum(self.h)))
        else:
            H_max = float(np.sum(self.H_rest))
        c = np.sqrt(float(self.g) * H_max)
        umax = max(float(np.max(np.abs(uk))) for uk in self.u) + c
        vmax = max(float(np.max(np.abs(vk))) for vk in self.v) + c
        return float(self.dt * max(umax / self.dx, vmax / self.dy))

    # ------------------------------------------------------------------
    #  Extract surface fields (for observation operator / plotting)
    # ------------------------------------------------------------------
    def surface_state(self):
        """Return surface SSH, u₀, v₀, T₀ as a dict."""
        return {
            'ssh': self.ssh(),
            'u': self.u[0].copy(),
            'v': self.v[0].copy(),
            'T': self.T[0].copy() if self.T[0] is not None else None,
        }


# ---------------------------------------------------------------------------
#  Helpers (shared with scripts)
# ---------------------------------------------------------------------------

def coriolis_array(lat_min, lat_max, ny, nx, dtype=np.float64):
    """Return f(y) = 2 Ω sin(lat) broadcast to (ny, nx)."""
    lats = np.linspace(lat_min, lat_max, ny)
    f_1d = 2.0 * OMEGA * np.sin(np.deg2rad(lats))
    return np.broadcast_to(f_1d[:, None], (ny, nx)).copy().astype(dtype)


def lonlat_to_dxdy(lat_center, dlon_deg, dlat_deg):
    """Convert degrees to approximate metres at a given latitude."""
    dx = np.deg2rad(dlon_deg) * EARTH_RADIUS * np.cos(np.deg2rad(lat_center))
    dy = np.deg2rad(dlat_deg) * EARTH_RADIUS
    return float(dx), float(dy)

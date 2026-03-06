"""
mlswe.boundary_handler  –  Multi-layer boundary condition handler
=================================================================

Reads boundary data from HYCOM (surface SSH, uo, vo, SST) and distributes
it across the 3 model layers using simplified assumptions:

Layer distribution strategy
----------------------------
* **SSH → Layer thicknesses:**
    hₖ = H_rest_k + αₖ · ssh_hycom
  where αₖ distributes the SSH perturbation.  The surface layer gets the
  largest share (α₀ = 0.6, α₁ = 0.3, α₂ = 0.1), reflecting that
  baroclinic SSH is primarily a surface phenomenon.

* **Velocities:**
    Layer 0 (surface):  u₀ = uo_hycom,  v₀ = vo_hycom  (full surface current)
    Layer 1 (thermo):   u₁ = β₁ · uo_hycom   (β₁ = 0.3, reduced)
    Layer 2 (deep):     u₂ = β₂ · uo_hycom   (β₂ = 0.05, nearly quiescent)

* **Temperature:**
    Layer 0: T₀ = SST_hycom
    Layer 1: T₁ = T_rest₁  (fixed thermocline temperature)
    Layer 2: T₂ = T_rest₂  (fixed deep temperature)

These are reasonable simplifications when only surface data is available.
"""
from __future__ import print_function
import os
import re
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None


class MLBoundaryHandler:
    """
    Boundary-condition handler for the 3-layer MLSWE model.

    Parameters
    ----------
    nc_path : str
        Path to HYCOM boundary-condition NetCDF (ssh, uo, vo, sst).
    model_lon, model_lat : 1-D arrays
        Model grid coordinates.
    H_b : ndarray (ny, nx) or None
        Bathymetry.
    H_rest : array-like (3,)
        Rest-state layer thicknesses [m].
    T_rest : array-like (3,)
        Rest-state layer temperatures [K].
    alpha_h : array-like (3,)
        SSH → thickness distribution weights (must sum to 1).
    beta_vel : array-like (3,)
        Velocity factors per layer (1.0 = full surface current).
    n_ghost : int
        Number of boundary ghost cells (hard-clamped).
    sponge_width : int
        Width of sponge/relaxation zone BEYOND ghost cells (cells).
        Total affected zone = n_ghost + sponge_width.
    sponge_timescale : float
        Relaxation e-folding timescale at inner edge of sponge [s].
        At ghost cells the clamping is instantaneous (weight=1).
        At the inner edge of the sponge, weight = dt/sponge_timescale.
    time_offset : float
        Added to model time before BC lookup.
    """

    def __init__(
        self,
        nc_path,
        model_lon,
        model_lat,
        H_b=None,
        H_mean=4000.0,
        H_rest=(100.0, 400.0, 3500.0),
        T_rest=(298.0, 283.0, 275.0),
        alpha_h=(0.6, 0.3, 0.1),
        beta_vel=(1.0, 1.0, 1.0),
        n_ghost=2,
        sponge_width=8,
        sponge_timescale=3600.0,
        time_offset=0.0,
        verbose=True,
    ):
        self.H_b = np.asarray(H_b, dtype=np.float64) if H_b is not None else None
        self.H_mean = float(H_mean)
        self.H_rest = np.array(H_rest, dtype=np.float64)
        self.T_rest = np.array(T_rest, dtype=np.float64)
        self.alpha_h = np.array(alpha_h, dtype=np.float64)
        self.beta_vel = np.array(beta_vel, dtype=np.float64)
        self.n_ghost = int(n_ghost)
        self.sponge_width = int(sponge_width)
        self.sponge_timescale = float(sponge_timescale)
        self.time_offset = float(time_offset)
        self.verbose = verbose
        self.model_lon = np.asarray(model_lon, dtype=np.float64)
        self.model_lat = np.asarray(model_lat, dtype=np.float64)

        # Load NetCDF
        self._load_from_nc(nc_path)

        # Fill NaN in source data
        for arr in (self.ssh, self.uo, self.vo):
            _fill_nan_nearest(arr)
        if self.bc_sst is not None:
            _fill_nan_nearest(self.bc_sst)

        # Build sponge weight mask (ny, nx) — 0 interior, 1 at boundary
        self._build_sponge_mask()

        # Pre-interpolate to boundary AND sponge points
        self._precompute()

    def _build_sponge_mask(self):
        """
        Build a 2-D sponge weight mask (ny, nx).

        Structure (from boundary edge inward):
          - Ghost cells (0 .. n_ghost-1):  weight = 1.0 (hard clamp)
          - Sponge zone (n_ghost .. n_ghost+sponge_width-1):
                weight decays exponentially from 1→0 over sponge_width cells
          - Interior:  weight = 0.0 (free dynamics)

        The mask gives the *nudging coefficient* per cell:
          field_new = (1 - w) * field_model + w * field_hycom
        where w = sponge_mask[j, i].
        """
        ny = self.model_lat.size
        nx = self.model_lon.size
        ng = self.n_ghost
        sw = self.sponge_width

        # Distance from nearest boundary edge (in cells)
        # d=0 at outermost row/col, d=1 at next, etc.
        dist_j = np.minimum(np.arange(ny), np.arange(ny)[::-1])
        dist_i = np.minimum(np.arange(nx), np.arange(nx)[::-1])
        dist_2d = np.minimum(dist_j[:, None], dist_i[None, :]).astype(np.float64)

        mask = np.zeros((ny, nx), dtype=np.float64)

        # Ghost cells: hard clamp (w = 1)
        mask[dist_2d < ng] = 1.0

        # Sponge zone: exponential decay from w≈0.95 (just outside ghost) to w≈0
        if sw > 0:
            sponge_region = (dist_2d >= ng) & (dist_2d < ng + sw)
            # Normalised position: 0 at ghost edge, 1 at sponge inner edge
            # Shift by 0.5 cell so that d_norm starts at 0.5/sw (not exactly 1.0)
            d_norm = (dist_2d - ng + 0.5) / float(sw)
            # Exponential decay: exp(-3 * d_norm) gives ~5% at inner edge
            sponge_weight = np.exp(-3.0 * d_norm)
            mask[sponge_region] = sponge_weight[sponge_region]

        self.sponge_mask = mask

        # Track which cells are in sponge zone (not ghost) for pre-interpolation
        self.sponge_zone = (dist_2d >= ng) & (dist_2d < ng + sw)

        n_ghost_cells = int((dist_2d < ng).sum())
        n_sponge_cells = int(self.sponge_zone.sum())
        if self.verbose:
            print(f"[MLBoundaryHandler] Sponge mask: {n_ghost_cells} ghost cells "
                  f"(w=1.0) + {n_sponge_cells} sponge cells "
                  f"(w decays exp to ~0), total zone = {ng + sw} cells")

    def _load_from_nc(self, nc_path):
        if Dataset is None:
            raise ImportError("netCDF4 required")
        if not os.path.exists(nc_path):
            raise FileNotFoundError(f"BC file not found: {nc_path}")

        nc = Dataset(nc_path, 'r')
        self.bc_lon = np.asarray(nc.variables['lon'][:], dtype=np.float64)
        self.bc_lat = np.asarray(nc.variables['lat'][:], dtype=np.float64)

        # Parse time
        t_var = nc.variables['time']
        try:
            m = re.match(r'(\w+)\s+since\s+(.+)', t_var.units.strip())
            if m:
                unit = m.group(1).lower().rstrip('s')
                ref_str = m.group(2).strip()
                from datetime import datetime
                for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                            '%Y-%m-%d %H:%M', '%Y-%m-%d'):
                    try:
                        ref_dt = datetime.strptime(ref_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(ref_str)
                mult = {'second': 1.0, 'minute': 60.0,
                        'hour': 3600.0, 'day': 86400.0}[unit]
                epoch = datetime(1970, 1, 1)
                off = (ref_dt - epoch).total_seconds()
                raw_t = np.asarray(t_var[:], dtype=np.float64)
                self.bc_times = raw_t * mult + off
            else:
                raise ValueError('non-CF units')
        except Exception:
            self.bc_times = np.asarray(t_var[:], dtype=np.float64)

        self.ssh = np.asarray(nc.variables['ssh'][:], dtype=np.float64)
        self.uo  = np.asarray(nc.variables['uo'][:],  dtype=np.float64)
        self.vo  = np.asarray(nc.variables['vo'][:],  dtype=np.float64)

        sst_name = 'sst'
        if sst_name in nc.variables:
            self.bc_sst = np.asarray(nc.variables[sst_name][:], dtype=np.float64)
            if np.nanmean(self.bc_sst) < 100.0:
                self.bc_sst += 273.15
        else:
            self.bc_sst = None

        nc.close()

    def _precompute(self):
        ny = self.model_lat.size
        nx = self.model_lon.size
        ng = self.n_ghost
        sw = self.sponge_width
        total_zone = ng + sw

        # Collect ALL cells in ghost + sponge zone
        bdy = set()
        for g in range(total_zone):
            for j in range(ny):
                bdy.add((j, g))
                bdy.add((j, nx-1-g))
            for i in range(nx):
                bdy.add((g, i))
                bdy.add((ny-1-g, i))

        bdy = sorted(bdy)
        self.bdy_iy = np.array([b[0] for b in bdy], dtype=int)
        self.bdy_ix = np.array([b[1] for b in bdy], dtype=int)
        self.bdy_lon = self.model_lon[self.bdy_ix]
        self.bdy_lat = self.model_lat[self.bdy_iy]

        # Pre-extract sponge weights for these points
        self.bdy_weights = self.sponge_mask[self.bdy_iy, self.bdy_ix]

        self._pre_interp_to_bdy()

    def _pre_interp_to_bdy(self):
        pts = np.column_stack([self.bdy_lat, self.bdy_lon])
        ntime = self.ssh.shape[0]
        nbdy = len(self.bdy_iy)

        def interp_all_times(field_3d):
            out = np.zeros((ntime, nbdy), dtype=np.float64)
            for it in range(ntime):
                interp = RegularGridInterpolator(
                    (self.bc_lat, self.bc_lon), field_3d[it],
                    method='linear', bounds_error=False, fill_value=None)
                out[it] = interp(pts)
            return out

        if self.verbose:
            print("[MLBoundaryHandler] Pre-interpolating BC fields to "
                  f"boundary ({nbdy} points, {ntime} time steps) ...")
        self._ssh_bdy = interp_all_times(self.ssh)
        self._uo_bdy  = interp_all_times(self.uo)
        self._vo_bdy  = interp_all_times(self.vo)
        if self.bc_sst is not None:
            self._sst_bdy = interp_all_times(self.bc_sst)
        else:
            self._sst_bdy = None

        # Keep original fields for full-domain interpolation (e.g., initialization)
        # They are released after init via release_full_fields() to save ~3.8 GB/rank.
        if self.verbose:
            print("[MLBoundaryHandler] Pre-interpolation done.")

    def release_full_fields(self):
        """Free the full HYCOM grid arrays (no longer needed after init).

        Each array is (ntime, nlat_hycom, nlon_hycom) float64 ≈ 979 MB.
        Four arrays total ≈ 3.83 GB per rank — all dead weight after
        initial-state creation and boundary pre-interpolation.
        """
        freed = 0
        for attr in ('ssh', 'uo', 'vo', 'bc_sst'):
            arr = getattr(self, attr, None)
            if arr is not None:
                freed += arr.nbytes
                delattr(self, attr)
        import gc
        gc.collect()
        if self.verbose:
            print(f"[MLBoundaryHandler] Released full HYCOM fields: "
                  f"{freed / 1e9:.2f} GB freed")

    def _interp_time_bdy(self, field_bdy, t):
        t_query = t + self.time_offset
        idx = np.searchsorted(self.bc_times, t_query) - 1
        idx = max(0, min(idx, len(self.bc_times) - 2))
        t0 = self.bc_times[idx]
        t1 = self.bc_times[idx + 1]
        dt = t1 - t0
        alpha = (t_query - t0) / dt if dt > 0 else 0.0
        alpha = np.clip(alpha, 0.0, 1.0)
        return (1.0 - alpha) * field_bdy[idx] + alpha * field_bdy[idx + 1]

    def get_full_field(self, field_name, t):
        """
        Interpolate HYCOM field to full model domain at time t.

        Parameters
        ----------
        field_name : str
            One of: 'ssh', 'uo', 'vo', 'sst'
        t : float
            Model time (seconds since epoch)

        Returns
        -------
        field : ndarray (ny, nx)
            Interpolated field on model grid
        """
        # Select source field
        if field_name == 'ssh':
            src = getattr(self, 'ssh', None)
        elif field_name == 'uo':
            src = getattr(self, 'uo', None)
        elif field_name == 'vo':
            src = getattr(self, 'vo', None)
        elif field_name == 'sst':
            src = getattr(self, 'bc_sst', None)
        else:
            raise ValueError(f"Unknown field: {field_name}")
        if src is None:
            raise RuntimeError(
                f"Full HYCOM field '{field_name}' not available — "
                f"release_full_fields() was already called. "
                f"Call get_full_field() before releasing.")

        # Time interpolation
        t_query = t + self.time_offset
        idx = np.searchsorted(self.bc_times, t_query) - 1
        idx = max(0, min(idx, len(self.bc_times) - 2))
        t0 = self.bc_times[idx]
        t1 = self.bc_times[idx + 1]
        dt = t1 - t0
        alpha = (t_query - t0) / dt if dt > 0 else 0.0
        alpha = np.clip(alpha, 0.0, 1.0)
        field_t = (1.0 - alpha) * src[idx] + alpha * src[idx + 1]

        # Spatial interpolation to model grid
        ny = self.model_lat.size
        nx = self.model_lon.size
        mg_lat, mg_lon = np.meshgrid(self.model_lat, self.model_lon, indexing='ij')
        interp = RegularGridInterpolator(
            (self.bc_lat, self.bc_lon), field_t,
            method='linear', bounds_error=False, fill_value=None)
        pts = np.column_stack([mg_lat.ravel(), mg_lon.ravel()])
        field_out = interp(pts).reshape(ny, nx)
        return field_out

    def __call__(self, state, t):
        """
        Apply multi-layer boundary conditions with sponge relaxation.

        Ghost cells (w=1.0): hard-clamped to HYCOM (Dirichlet).
        Sponge zone (0<w<1): nudged toward HYCOM: f = (1-w)*f_model + w*f_hycom.
        Interior (w=0): untouched.

        state : dict with keys h0,u0,v0,T0, h1,u1,v1,T1, h2,u2,v2,T2
        t     : model time (seconds, epoch)
        """
        ssh_bdy = self._interp_time_bdy(self._ssh_bdy, t)
        uo_bdy  = self._interp_time_bdy(self._uo_bdy, t)
        vo_bdy  = self._interp_time_bdy(self._vo_bdy, t)

        iy, ix = self.bdy_iy, self.bdy_ix
        w = self.bdy_weights  # sponge weight per point

        # Compute local H_rest scaling for shallow boundary cells
        H_rest_total = float(self.H_rest.sum())
        if self.H_b is not None:
            H_b_bdy = self.H_b[iy, ix]
            scale = np.minimum(H_b_bdy / H_rest_total, 1.0)
        else:
            H_b_bdy = np.full(len(iy), float(self.H_mean))
            scale = np.ones(len(iy))

        # Layer thicknesses: proportional to H_rest (matches model dynamics)
        h_total_bdy = np.maximum(H_b_bdy + ssh_bdy, 10.0)

        for k in range(3):
            # Target HYCOM values
            h_target = (self.H_rest[k] / H_rest_total) * h_total_bdy
            u_target = self.beta_vel[k] * uo_bdy
            v_target = self.beta_vel[k] * vo_bdy

            # Sponge relaxation: blend model toward HYCOM
            h_model = state[f'h{k}'][iy, ix]
            u_model = state[f'u{k}'][iy, ix]
            v_model = state[f'v{k}'][iy, ix]

            state[f'h{k}'][iy, ix] = (1.0 - w) * h_model + w * h_target
            state[f'u{k}'][iy, ix] = (1.0 - w) * u_model + w * u_target
            state[f'v{k}'][iy, ix] = (1.0 - w) * v_model + w * v_target

            # Temperature
            if f'T{k}' in state:
                T_model = state[f'T{k}'][iy, ix]
                if k == 0 and self._sst_bdy is not None:
                    sst_bdy = self._interp_time_bdy(self._sst_bdy, t)
                    T_target = sst_bdy
                else:
                    T_target = self.T_rest[k]
                state[f'T{k}'][iy, ix] = (1.0 - w) * T_model + w * T_target

        return state


# ---------------------------------------------------------------------------
def _fill_nan_nearest(arr_3d):
    """In-place fill NaN in each 2-D slice with nearest finite value."""
    for t in range(arr_3d.shape[0]):
        s = arr_3d[t]
        mask = np.isnan(s)
        if not mask.any():
            continue
        if mask.all():
            s[:] = 0.0
            continue
        ind = distance_transform_edt(mask, return_distances=False,
                                      return_indices=True)
        s[mask] = s[tuple(ind[:, mask])]


# ---------------------------------------------------------------------------
#  Geostrophic initialisation
# ---------------------------------------------------------------------------
OMEGA = 7.2921e-5   # Earth rotation rate [rad/s]
GRAVITY = 9.81      # gravitational acceleration [m/s²]


def geostrophic_velocities(ssh, lat_grid, lon_grid, dx, dy, g=GRAVITY):
    """
    Compute geostrophically balanced velocities from SSH field.

    Geostrophic balance:
        u_g = -(g/f) * dSSH/dy
        v_g =  (g/f) * dSSH/dx

    Parameters
    ----------
    ssh : ndarray (ny, nx)
        Sea surface height [m].
    lat_grid : 1-D array (ny,)
        Latitude of grid centres [degrees].
    lon_grid : 1-D array (nx,)
        Longitude of grid centres [degrees].
    dx, dy : float
        Grid spacing [m].
    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    u_geo, v_geo : ndarray (ny, nx)
        Geostrophic velocity components [m/s].
    """
    ny, nx = ssh.shape

    # Coriolis parameter f = 2Ω sin(lat)
    lat_rad = np.deg2rad(lat_grid)
    f = 2.0 * OMEGA * np.sin(lat_rad)
    # Avoid division by zero near equator: clamp |f| >= 1e-5 s⁻¹ (|lat| ≳ 2°)
    f = np.where(np.abs(f) < 1e-5, np.sign(f + 1e-30) * 1e-5, f)
    f_2d = f[:, np.newaxis] * np.ones((1, nx))  # (ny, nx)

    # SSH gradients (central differences, one-sided at boundaries)
    deta_dy = np.zeros_like(ssh)
    deta_dx = np.zeros_like(ssh)

    # ∂η/∂y (j direction)
    deta_dy[1:-1, :] = (ssh[2:, :] - ssh[:-2, :]) / (2.0 * dy)
    deta_dy[0, :] = (ssh[1, :] - ssh[0, :]) / dy
    deta_dy[-1, :] = (ssh[-1, :] - ssh[-2, :]) / dy

    # ∂η/∂x (i direction)
    deta_dx[:, 1:-1] = (ssh[:, 2:] - ssh[:, :-2]) / (2.0 * dx)
    deta_dx[:, 0] = (ssh[:, 1] - ssh[:, 0]) / dx
    deta_dx[:, -1] = (ssh[:, -1] - ssh[:, -2]) / dx

    # Geostrophic velocities
    u_geo = -(g / f_2d) * deta_dy
    v_geo = (g / f_2d) * deta_dx

    # Clamp extreme values (bathymetric features can create large gradients)
    np.clip(u_geo, -5.0, 5.0, out=u_geo)
    np.clip(v_geo, -5.0, 5.0, out=v_geo)

    return u_geo, v_geo

import numpy as np


def generate_swath_observations(
    x, y, Lx, Ly, frame, nassim,
    swath_width=50_000.0,
    gap_width=20_000.0,
    fixed_nobs=None,
    cycles_to_cross=20,   # NEW: how many cycles to go left->right
    oscillate=False,      # NEW: if True use back-and-forth (triangular) motion
    jitter=0.0,           # NEW: small random jitter in meters to break ties (0.0 disables)
    angle_even=70,
    angle_odd=110
):
    """
    Generate two swaths of observations mimicking SWOT mission.

    New args:
      cycles_to_cross: number of cycles to move the swath center from left to right
      oscillate: if True perform back-and-forth motion (triangular wave) across domain
      jitter: add small random jitter (meters) to the center_x to avoid repeated identical selections
    """

    nx, ny = x.shape[1], x.shape[0]
    dx = Lx / nx
    dy = Ly / ny

    angle_deg = angle_even if frame % 2 == 0 else angle_odd
    angle_rad = np.deg2rad(angle_deg)
    normal = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

    # Determine center_x using cycles_to_cross and optionally oscillate
    if cycles_to_cross <= 0:
        cycles_to_cross = max(1, nassim)

    # linear speed per cycle to cross domain in cycles_to_cross cycles
    step = Lx / float(cycles_to_cross)

    if oscillate:
        # triangular wave: goes 0 -> Lx in cycles_to_cross, then Lx -> 0 in next cycles_to_cross, repeat
        period = 2 * cycles_to_cross
        pos = frame % period
        if pos <= cycles_to_cross:
            frac = pos / float(cycles_to_cross)   # 0..1 left->right
        else:
            frac = (period - pos) / float(cycles_to_cross)  # 1..0 right->left
        center_x = frac * Lx
    else:
        # wrap-around linear motion: increases by 'step' each cycle with modulo Lx
        center_x = (frame * step) % Lx

    # apply small jitter if requested
    if jitter and jitter > 0.0:
        center_x = (center_x + np.random.default_rng(frame).uniform(-jitter, jitter)) % Lx

    center = np.array([center_x, Ly / 2.0])

    # Flatten grid into (N, 2) shape
    pos = np.stack([x.ravel(), y.ravel()], axis=1)

    # Consider positions shifted by -Lx, 0, +Lx to simulate periodic wrapping (keeps continuity)
    shifts = [-Lx, 0.0, Lx]
    xobs_all = []
    yobs_all = []

    offset = (gap_width + swath_width) / 2.0
    half_w = swath_width / 2.0

    for shift in shifts:
        pos_shifted = pos.copy()
        pos_shifted[:, 0] += shift
        vecs = pos_shifted - center
        # distance along the normal direction
        dists = vecs @ normal

        in_swath = ((np.abs(dists + offset) <= half_w) |
                    (np.abs(dists - offset) <= half_w))

        selected = pos_shifted[in_swath]
        if selected.size > 0:
            # Wrap x back into base domain
            selected[:, 0] = np.mod(selected[:, 0], Lx)
            xobs_all.append(selected[:, 0])
            yobs_all.append(selected[:, 1])

    if len(xobs_all) == 0:
        # fallback: no points found (very narrow swath). Return empty arrays
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float), 0

    # Combine and deduplicate
    xobs = np.concatenate(xobs_all)
    yobs = np.concatenate(yobs_all)

    # Map to nearest grid indices (this is where small shifts can be lost if step < dx)
    ix = np.round(xobs / dx).astype(int)
    iy = np.round(yobs / dy).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    inds = np.ravel_multi_index((iy, ix), (ny, nx))
    obs_inds = np.unique(inds)

    # If user wants fixed_nobs, either subsample or add nearest points
    if fixed_nobs is None:
        fixed_nobs = len(obs_inds)
    elif len(obs_inds) > fixed_nobs:
        obs_inds = np.random.choice(obs_inds, fixed_nobs, replace=False)
    elif len(obs_inds) < fixed_nobs:
        missing = fixed_nobs - len(obs_inds)
        all_indices = np.arange(nx * ny)
        outside = np.setdiff1d(all_indices, obs_inds, assume_unique=True)

        # Use Euclidean distance to the current swath center to pick extra points
        outside_x = x.ravel()[outside]
        outside_y = y.ravel()[outside]
        outside_pos = np.vstack((outside_x, outside_y)).T

        # compute distance
        dist_to_center = np.linalg.norm(outside_pos - center, axis=1)
        closest_indices = outside[np.argsort(dist_to_center)[:missing]]
        obs_inds = np.concatenate((obs_inds, closest_indices))

    xob = x.ravel()[obs_inds]
    yob = y.ravel()[obs_inds]

    return obs_inds, xob, yob, fixed_nobs
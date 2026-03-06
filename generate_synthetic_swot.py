#!/usr/bin/env python3
"""
Generate SWOT-like synthetic SSH observations for cycles WITHOUT real SWOT.

Mimics real SWOT orbit characteristics derived from the actual data:
  - Two tracks (A and B) separated by ~25° longitude
  - Each track pair appears every ~12 cycles (alternating 11 and 13)
  - Both tracks drift westward by ~1.5° per appearance
  - When a track exits the western boundary, it wraps to the east

Swath geometry (realistic):
  - Width: ~3 grid cells (~1.5° ≈ 120 km, matching real KaRIn swath)
  - Slightly tilted (not perfectly N-S), mimicking the descending/ascending
    orbit inclination (~10-15° from vertical)
  - ~200 observations per cycle (comparable to real SWOT coverage on this grid)

The combined real + synthetic data is saved to a new file.
"""
import os
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator

# ---- Config ----
NX, NY = 80, 70
LON_MIN, LON_MAX = -60.0, -20.0
LAT_MIN, LAT_MAX = 10.0, 45.0
lon_grid = np.linspace(LON_MIN, LON_MAX, NX)
lat_grid = np.linspace(LAT_MIN, LAT_MAX, NY)
LON_RANGE = LON_MAX - LON_MIN  # 45°

NASSIM = 240
T_START = datetime(2024, 8, 1, 0, 0, 0)
DT, T_FREQ = 75.0, 48
obs_dt = timedelta(seconds=T_FREQ * DT)

SIG_SSH = 0.5
DATA_DIR = './data'

# SWOT swath parameters
SWATH_WIDTH_CELLS = 3       # ~1.5° ≈ 120 km (real KaRIn width)
TRACK_SEPARATION = 25.0     # degrees between track A and B centres
REPEAT_PERIOD = [11, 13]    # alternating cycle gaps
LON_DRIFT_PER_APPEAR = -1.5 # degrees westward per appearance
# Tilt: longitude shift per latitude step (degrees lon per lat cell)
# ~12° from vertical → tan(12°) ≈ 0.21 → 0.21 * 0.5° ≈ 0.1° lon per lat cell
SWATH_TILT = 0.10           # degrees lon per latitude row

# ---- Load bathymetry ----
H_b = np.load(os.path.join(DATA_DIR,
    'etopo_bathy_-60.0_-20.0_10.0_45.0_70x80.npy')).astype(np.float64)
H_b = np.maximum(np.abs(H_b), 100.0)

# ---- Load pre-binned real SWOT ----
binned_file = './data/swot_2024aug_new/swot_ssh_binned_80x70.nc'
nc = Dataset(binned_file, 'r')
ssha_all = nc.variables['ssha'][:]
cell_all = nc.variables['cell_index'][:]
nobs_arr = nc.variables['n_obs'][:]
nc.close()

real_cells = []
real_ssha = []
for c in range(NASSIM):
    n = int(nobs_arr[c])
    if n > 0:
        real_cells.append(cell_all[c, :n].astype(int))
        real_ssha.append(ssha_all[c, :n].astype(np.float64))
    else:
        real_cells.append(np.array([], dtype=int))
        real_ssha.append(np.array([], dtype=np.float64))

has_real = np.array([len(a) > 0 for a in real_cells])
print(f"Real SWOT: {has_real.sum()}/240 cycles")

# ---- Load HYCOM SSH reference ----
bc_file = os.path.join(DATA_DIR, 'hycom_bc_2024aug.nc')
nc = Dataset(bc_file, 'r')
bc_ssh = nc.variables['ssh'][:]
bc_times = nc.variables['time'][:]
bc_lat = nc.variables['lat'][:]
bc_lon = nc.variables['lon'][:]
nc.close()

ssh_ref_3d = np.zeros((len(bc_times), NY, NX), dtype=np.float64)
for t in range(len(bc_times)):
    interp = RegularGridInterpolator(
        (bc_lat, bc_lon), bc_ssh[t], method='linear',
        bounds_error=False, fill_value=0.0)
    pts = np.array(np.meshgrid(lat_grid, lon_grid, indexing='ij')).reshape(2, -1).T
    ssh_ref_3d[t] = interp(pts).reshape(NY, NX)

tstart_epoch = (T_START - datetime(1970, 1, 1)).total_seconds()
obs_epoch = np.array([tstart_epoch + (i + 1) * T_FREQ * DT for i in range(NASSIM)])


# ---- Swath schedule ----
def generate_swath_schedule(nassim, lon_grid, track_a_start_lon=-23.5,
                            track_sep=25.0, first_cycle=7,
                            repeat_periods=(11, 13), drift_per=-1.5):
    """Dict mapping cycle -> list of centre longitudes."""
    lon_min, lon_max = lon_grid[0], lon_grid[-1]
    lon_range = lon_max - lon_min
    schedule = {}
    current_lon_a = track_a_start_lon
    current_cycle = first_cycle
    period_idx = 0
    track_a_events = []
    while current_cycle < nassim:
        track_a_events.append((current_cycle, current_lon_a))
        gap = repeat_periods[period_idx % len(repeat_periods)]
        current_cycle += gap
        current_lon_a += drift_per
        if current_lon_a < lon_min:
            current_lon_a += lon_range
        period_idx += 1
    for cyc_a, lon_a in track_a_events:
        cyc_b = cyc_a + 2
        lon_b = lon_a - track_sep
        if lon_b < lon_min:
            lon_b += lon_range
        schedule.setdefault(cyc_a, []).append(lon_a)
        if cyc_b < nassim:
            schedule.setdefault(cyc_b, []).append(lon_b)
    return schedule


def make_swath_cells_tilted(center_lon, lon_grid, nx, ny,
                            swath_width=3, tilt=0.10):
    """Generate cell indices for a thin, tilted SWOT-like swath.

    Parameters
    ----------
    center_lon : float
        Longitude of swath centre at the bottom (iy=0) of the domain.
    swath_width : int
        Width of swath in grid cells (default 3 ~ 1.5 deg ~ 120 km).
    tilt : float
        Longitude shift per latitude row (degrees), giving the swath
        a slight diagonal.  Positive = swath leans eastward with
        increasing latitude.
    """
    half_w = swath_width // 2
    cells = []
    for iy in range(ny):
        # Centre longitude shifts with latitude (tilt)
        lon_c = center_lon + iy * tilt
        ix_center = int(np.argmin(np.abs(lon_grid - lon_c)))
        for dix in range(-half_w, half_w + 1):
            ix = ix_center + dix
            if 0 <= ix < nx:
                cells.append(iy * nx + ix)
    return np.array(cells, dtype=int)


# Generate schedule
schedule = generate_swath_schedule(NASSIM, lon_grid)
sched_cycles = sorted(schedule.keys())
print(f"Synthetic schedule: {len(schedule)} cycles with planned swaths")

# ---- Fill cycles that lack real SWOT ----
rng = np.random.default_rng(42)
synth_cells = [np.array([], dtype=int) for _ in range(NASSIM)]
synth_ssha = [np.array([], dtype=np.float64) for _ in range(NASSIM)]
is_synthetic = np.zeros(NASSIM, dtype=bool)

n_synth = 0
for c in range(NASSIM):
    if has_real[c]:
        continue

    if c in schedule:
        swath_lons = schedule[c]
    else:
        before = [s for s in sched_cycles if s <= c]
        after  = [s for s in sched_cycles if s > c]
        if before and after:
            cb, ca = before[-1], after[0]
            frac = (c - cb) / (ca - cb)
            interp_lon = schedule[cb][0] + frac * (schedule[ca][0] - schedule[cb][0])
            swath_lons = [interp_lon]
        elif before:
            swath_lons = [schedule[before[-1]][0] + LON_DRIFT_PER_APPEAR * 0.1]
        else:
            swath_lons = [schedule[after[0]][0]]

    all_cells = []
    for sl in swath_lons:
        cells = make_swath_cells_tilted(sl, lon_grid, NX, NY,
                                        swath_width=SWATH_WIDTH_CELLS,
                                        tilt=SWATH_TILT)
        all_cells.append(cells)
    all_cells = np.unique(np.concatenate(all_cells))

    # HYCOM SSH at nearest time snapshot
    idx = int(np.searchsorted(bc_times, obs_epoch[c], side='right')) - 1
    idx = max(0, min(idx, len(bc_times) - 1))
    ssh_field = ssh_ref_3d[idx].ravel()

    ssha_vals = ssh_field[all_cells] + rng.normal(0.0, SIG_SSH, size=len(all_cells))

    synth_cells[c] = all_cells
    synth_ssha[c] = ssha_vals
    is_synthetic[c] = True
    n_synth += 1

print(f"Synthetic swaths generated for {n_synth} cycles")
print(f"Real SWOT: {has_real.sum()} cycles, Synthetic: {n_synth} cycles")

# ---- Combine real + synthetic ----
combined_cells = []
combined_ssha = []
combined_source = []  # 0=real, 1=synthetic

for c in range(NASSIM):
    if has_real[c]:
        combined_cells.append(real_cells[c])
        combined_ssha.append(real_ssha[c])
        combined_source.append(np.zeros(len(real_cells[c]), dtype=np.int8))
    elif is_synthetic[c]:
        combined_cells.append(synth_cells[c])
        combined_ssha.append(synth_ssha[c])
        combined_source.append(np.ones(len(synth_cells[c]), dtype=np.int8))
    else:
        combined_cells.append(np.array([], dtype=int))
        combined_ssha.append(np.array([], dtype=np.float64))
        combined_source.append(np.array([], dtype=np.int8))

max_nobs = max(len(a) for a in combined_cells)
print(f"\nMax obs/cycle: {max_nobs}")
print(f"Coverage: {sum(1 for a in combined_cells if len(a) > 0)}/240 cycles")

# ---- Write NetCDF ----
out_file = './data/swot_2024aug_new/swot_ssh_combined_80x70.nc'
ssha_arr = np.full((NASSIM, max_nobs), np.nan, dtype=np.float32)
cell_arr = np.full((NASSIM, max_nobs), -1, dtype=np.int32)
src_arr  = np.full((NASSIM, max_nobs), -1, dtype=np.int8)
nobs_out = np.zeros(NASSIM, dtype=np.int32)

for c in range(NASSIM):
    n = len(combined_cells[c])
    nobs_out[c] = n
    if n > 0:
        ssha_arr[c, :n] = combined_ssha[c]
        cell_arr[c, :n] = combined_cells[c]
        src_arr[c, :n]  = combined_source[c]

with Dataset(out_file, 'w', format='NETCDF4') as nc:
    nc.createDimension('cycle', NASSIM)
    nc.createDimension('max_obs', max_nobs)

    v = nc.createVariable('ssha', 'f4', ('cycle', 'max_obs'), zlib=True)
    v[:] = ssha_arr
    v.long_name = 'SSH anomaly (real SWOT binned or HYCOM synthetic)'
    v.units = 'm'

    v2 = nc.createVariable('cell_index', 'i4', ('cycle', 'max_obs'), zlib=True)
    v2[:] = cell_arr
    v2.long_name = 'flat grid cell index (iy*NX + ix)'

    v3 = nc.createVariable('source', 'i1', ('cycle', 'max_obs'), zlib=True)
    v3[:] = src_arr
    v3.long_name = '0=real SWOT, 1=synthetic HYCOM'

    v4 = nc.createVariable('n_obs', 'i4', ('cycle',), zlib=True)
    v4[:] = nobs_out

    nc.nx = NX
    nc.ny = NY
    nc.nassim = NASSIM
    nc.sig_ssh = SIG_SSH
    nc.swath_width_cells = SWATH_WIDTH_CELLS
    nc.swath_tilt_deg_per_row = SWATH_TILT
    nc.description = ('Combined real SWOT + synthetic HYCOM SSH. '
                       'Synthetic swaths: thin (~3 cells), tilted, '
                       '~200 obs/cycle.')

print(f"\nSaved: {out_file} ({os.path.getsize(out_file)/1024:.0f} KB)")

# ---- Stats ----
for label, mask in [('Real', ~is_synthetic & has_real),
                     ('Synthetic', is_synthetic)]:
    cycs = np.where(mask)[0]
    counts = [len(combined_cells[c]) for c in cycs]
    if counts:
        print(f"  {label}: {len(cycs)} cycles, "
              f"obs/cycle: {np.mean(counts):.0f} mean, "
              f"{np.median(counts):.0f} median, "
              f"[{min(counts)}, {max(counts)}] range")

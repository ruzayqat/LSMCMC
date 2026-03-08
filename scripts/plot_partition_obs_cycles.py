#!/usr/bin/env python
"""
Plot domain partition with observation locations at the first and last cycle.
Two subplots side by side.
"""
import os, sys, numpy as np, yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SWE_LSMCMC'))
from loc_smcmc_swe_exact_from_Gauss import partition_domain

# ---- Config ----
config_file = sys.argv[1] if len(sys.argv) > 1 else 'example_input_mlswe_test100.yml'
with open(config_file) as f:
    params = yaml.safe_load(f)

nx = params['dgx']; ny = params['dgy']
lon = np.linspace(params['lon_min'], params['lon_max'], nx)
lat = np.linspace(params['lat_min'], params['lat_max'], ny)
nassim = params.get('nassim', 100)
ncells = nx * ny
num_subdomains = params.get('num_subdomains', 50)

# ---- Partition ----
partitions, labels, N, nby, nbx, bh, bw = partition_domain(ny, nx, num_subdomains)

# ---- Load drifter obs ----
obs_file = params.get('obs_file', './data/obs_2024aug/swe_drifter_obs.nc')
with Dataset(obs_file, 'r') as nc:
    ind_all = np.asarray(nc.variables['yobs_ind_all'][:])

# ---- Load SWOT SSH obs ----
data_dir = params.get('data_dir', './data')
swot_dir = params.get('swot_dir', os.path.join(data_dir, 'swot_2024aug_new'))
swot_file = None
for tag in [f'{ny}x{nx}', f'{nx}x{ny}']:
    for prefix in ['swot_ssh_combined', 'swot_ssh_binned']:
        c = os.path.join(swot_dir, f'{prefix}_{tag}.nc')
        if os.path.exists(c):
            swot_file = c; break
    if swot_file: break

swot_cell_bin = swot_nobs_bin = None
if swot_file:
    with Dataset(swot_file, 'r') as nc:
        swot_cell_bin = nc.variables['cell_index'][:]
        swot_nobs_bin = nc.variables['n_obs'][:]

dlon = (lon[-1] - lon[0]) / (nx - 1)
dlat = (lat[-1] - lat[0]) / (ny - 1)

def get_obs_at_cycle(cycle_idx):
    """Extract UV, SST, SSH cell indices for one cycle."""
    valid = ind_all[cycle_idx] >= 0
    inds = ind_all[cycle_idx, valid].astype(int)
    uv_cells, sst_cells = [], []
    for idx in inds:
        field = idx // ncells
        cell = idx % ncells
        if field == 1 or field == 2:
            uv_cells.append(cell)
        elif field == 3:
            sst_cells.append(cell)
    ssh_cells = []
    if swot_cell_bin is not None and cycle_idx < swot_cell_bin.shape[0]:
        n = int(swot_nobs_bin[cycle_idx])
        if n > 0:
            ssh_cells = swot_cell_bin[cycle_idx, :n].astype(int).tolist()
    return np.unique(uv_cells), np.unique(sst_cells), np.unique(ssh_cells)

def cells_to_lonlat(cells):
    cells = np.asarray(cells, dtype=int)
    if len(cells) == 0:
        return np.array([]), np.array([])
    return lon[cells % nx], lat[cells // nx]

# ---- Plot ----
cycle_first = 0
cycle_last = min(nassim, ind_all.shape[0]) - 1

fig, axes = plt.subplots(1, 2, figsize=(22, 9))

np.random.seed(42)
cmap_colors = plt.cm.Pastel2(np.linspace(0, 1, num_subdomains))
order = np.arange(num_subdomains); np.random.shuffle(order)
shuffled_colors = cmap_colors[order]

for ax_i, (ax, cyc) in enumerate(zip(axes, [cycle_first, cycle_last])):
    # Draw partition blocks
    for label, ((y0, y1), (x0, x1)) in enumerate(partitions):
        rl = lon[x0] - dlon/2; rb = lat[y0] - dlat/2
        rw = (x1 - x0) * dlon; rh = (y1 - y0) * dlat
        ax.add_patch(plt.Rectangle((rl, rb), rw, rh,
                                    facecolor=shuffled_colors[label],
                                    edgecolor='gray', linewidth=0.4, alpha=0.5))
        ax.add_patch(plt.Rectangle((rl, rb), rw, rh,
                                    facecolor='none', edgecolor='k',
                                    linewidth=0.7, alpha=0.6))

    uv_c, sst_c, ssh_c = get_obs_at_cycle(cyc)
    uv_lon, uv_lat = cells_to_lonlat(uv_c)
    sst_lon, sst_lat = cells_to_lonlat(sst_c)
    ssh_lon, ssh_lat = cells_to_lonlat(ssh_c)

    s = 18
    if len(ssh_lon):
        ax.scatter(ssh_lon, ssh_lat, s=s, c='dodgerblue', marker='o', alpha=0.7,
                   edgecolors='none', label=f'SSH/SWOT ({len(ssh_c)})', zorder=3)
    if len(uv_lon):
        ax.scatter(uv_lon, uv_lat, s=s+2, c='red', marker='o', alpha=0.8,
                   edgecolors='none', label=f'UV/drifter ({len(uv_c)})', zorder=4)
    if len(sst_lon):
        ax.scatter(sst_lon, sst_lat, s=s+6, c='orange', marker='D', alpha=0.8,
                   edgecolors='none', label=f'SST/drifter ({len(sst_c)})', zorder=5)

    # Count total obs (with duplicates for u+v)
    valid = ind_all[cyc] >= 0
    n_total = int(valid.sum())

    ax.set_xlim(params['lon_min'] - 1, params['lon_max'] + 1)
    ax.set_ylim(params['lat_min'] - 1, params['lat_max'] + 1)
    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)
    ax.set_title(f'Cycle {cyc}  ({n_total} total obs: '
                 f'{len(uv_c)} UV cells, {len(sst_c)} SST cells, '
                 f'{len(ssh_c)} SSH cells)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.3)

fig.suptitle(f'Domain Partition ({num_subdomains} subdomains, {nby}×{nbx}) — '
             f'Observation Locations per Cycle\n'
             f'Grid: {ny}×{nx},  '
             f'lon=[{params["lon_min"]},{params["lon_max"]}],  '
             f'lat=[{params["lat_min"]},{params["lat_max"]}]',
             fontsize=14, y=1.01)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'domain_partition_obs_cycles.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()

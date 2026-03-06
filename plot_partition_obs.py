#!/usr/bin/env python
"""
Plot domain partitioned into subdomains with all observation locations.
Shows SST (drifter), UV (drifter), and SSH (SWOT) observations as small circles.
"""
import os
import sys
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SWE_LSMCMC'))
from loc_smcmc_swe_exact_from_Gauss import partition_domain

# ---- Load config ----
config_file = sys.argv[1] if len(sys.argv) > 1 else 'example_input_mlswe_test100.yml'
with open(config_file) as f:
    params = yaml.safe_load(f)

nx = params['dgx']
ny = params['dgy']
lon = np.linspace(params['lon_min'], params['lon_max'], nx)
lat = np.linspace(params['lat_min'], params['lat_max'], ny)
num_subdomains = params.get('num_subdomains', 50)

# ---- Partition domain ----
partitions, labels, N, nby, nbx, bh, bw = partition_domain(ny, nx, num_subdomains)
print(f"Partition: {N} subdomains ({nby}y × {nbx}x), block size {bh}×{bw}")

# ---- Load drifter obs ----
obs_file = params.get('obs_file', './data/obs_2024aug/swe_drifter_obs.nc')
with Dataset(obs_file, 'r') as nc:
    yobs_all = np.asarray(nc.variables['yobs_all'][:])
    ind_all  = np.asarray(nc.variables['yobs_ind_all'][:])
    ind0_all = np.asarray(nc.variables['yobs_ind_level0_all'][:])

nassim = min(yobs_all.shape[0], params.get('nassim', 100))
ncells = nx * ny

# Collect all unique drifter UV and SST observation cell indices
# ind_all has observation indices into state vector
# For 3-layer MLSWE: fields per layer = [h, u, v, T], so state is
#   layer0: h(0..ncells-1), u(ncells..2*ncells-1), v(2*ncells..3*ncells-1), T(3*ncells..4*ncells-1)
# ind0_all stores the level-0 (grid cell) indices
# drifter obs are: u, v (velocity) and T (SST)
# From the obs file, we can figure out which are velocity vs SST by looking at
# the state-vector index ranges

uv_cells = set()
sst_cells = set()

for i in range(nassim):
    valid = ind_all[i] >= 0
    inds = ind_all[i, valid].astype(int)
    valid0 = ind0_all[i] >= 0
    inds0  = ind0_all[i, valid0].astype(int)

    for idx in inds:
        field = idx // ncells
        cell = idx % ncells
        if field == 1 or field == 2:  # u or v
            uv_cells.add(cell)
        elif field == 3:              # T (SST)
            sst_cells.add(cell)

# Convert cell indices to (row, col) then to (lon, lat)
def cells_to_lonlat(cells):
    cells = np.array(list(cells), dtype=int)
    if len(cells) == 0:
        return np.array([]), np.array([])
    rows = cells // nx
    cols = cells % nx
    return lon[cols], lat[rows]

uv_lon, uv_lat = cells_to_lonlat(uv_cells)
sst_lon, sst_lat = cells_to_lonlat(sst_cells)

print(f"Drifter UV cells: {len(uv_cells)}, SST cells: {len(sst_cells)}")

# ---- Load SWOT SSH obs ----
data_dir = params.get('data_dir', './data')
swot_dir = params.get('swot_dir', os.path.join(data_dir, 'swot_2024aug_new'))
combined = None
binned = None
for tag in [f'{ny}x{nx}', f'{nx}x{ny}']:
    c = os.path.join(swot_dir, f'swot_ssh_combined_{tag}.nc')
    b = os.path.join(swot_dir, f'swot_ssh_binned_{tag}.nc')
    if os.path.exists(c):
        combined = c
    if os.path.exists(b):
        binned = b
swot_file = combined if combined else binned

ssh_cells = set()
if swot_file:
    with Dataset(swot_file, 'r') as nc:
        cell_bin = nc.variables['cell_index'][:]
        nobs_bin = nc.variables['n_obs'][:]
    for c in range(min(nassim, cell_bin.shape[0])):
        n = int(nobs_bin[c])
        if n > 0:
            for cell in cell_bin[c, :n].astype(int):
                ssh_cells.add(cell)
    print(f"SWOT SSH cells: {len(ssh_cells)} (from {swot_file})")
else:
    print("No SWOT SSH file found")

ssh_lon, ssh_lat = cells_to_lonlat(ssh_cells)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(14, 9))

# Draw partition as colored blocks with light pastel colors
np.random.seed(42)
cmap_colors = plt.cm.Pastel2(np.linspace(0, 1, num_subdomains))
# Shuffle to avoid adjacent blocks having similar colors
order = np.arange(num_subdomains)
np.random.shuffle(order)
shuffled_colors = cmap_colors[order]

# Map labels to shuffled colors
dlon = (lon[-1] - lon[0]) / (nx - 1)
dlat = (lat[-1] - lat[0]) / (ny - 1)

for label, ((y0, y1), (x0, x1)) in enumerate(partitions):
    rect_lon = lon[x0] - dlon/2
    rect_lat = lat[y0] - dlat/2
    rect_w = (x1 - x0) * dlon
    rect_h = (y1 - y0) * dlat
    color = shuffled_colors[label]
    ax.add_patch(plt.Rectangle((rect_lon, rect_lat), rect_w, rect_h,
                                facecolor=color, edgecolor='gray',
                                linewidth=0.5, alpha=0.6))

# Draw subdomain boundaries more prominently
for label, ((y0, y1), (x0, x1)) in enumerate(partitions):
    rect_lon = lon[x0] - dlon/2
    rect_lat = lat[y0] - dlat/2
    rect_w = (x1 - x0) * dlon
    rect_h = (y1 - y0) * dlat
    ax.add_patch(plt.Rectangle((rect_lon, rect_lat), rect_w, rect_h,
                                facecolor='none', edgecolor='k',
                                linewidth=0.8, alpha=0.7))

# Plot observations
s = 12  # marker size
ax.scatter(ssh_lon, ssh_lat, s=s, c='dodgerblue', marker='o', alpha=0.7,
           edgecolors='none', label=f'SSH / SWOT ({len(ssh_cells)} cells)', zorder=3)
ax.scatter(uv_lon, uv_lat, s=s, c='red', marker='o', alpha=0.7,
           edgecolors='none', label=f'UV / drifter ({len(uv_cells)} cells)', zorder=4)
ax.scatter(sst_lon, sst_lat, s=s+4, c='orange', marker='D', alpha=0.7,
           edgecolors='none', label=f'SST / drifter ({len(sst_cells)} cells)', zorder=5)

ax.set_xlim(params['lon_min'] - 1, params['lon_max'] + 1)
ax.set_ylim(params['lat_min'] - 1, params['lat_max'] + 1)
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title(f'Domain Partition ({num_subdomains} subdomains, {nby}×{nbx}) '
             f'with Observation Locations\n'
             f'Grid: {ny}×{nx}, lon=[{params["lon_min"]},{params["lon_max"]}], '
             f'lat=[{params["lat_min"]},{params["lat_max"]}]',
             fontsize=13)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linewidth=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'domain_partition_obs.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()

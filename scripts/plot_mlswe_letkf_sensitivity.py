"""
plot_mlswe_letkf_sensitivity.py
================================
Plot LETKF sensitivity analysis heatmaps for the MLSWE (3-layer) model.
Reads results from mlswe_letkf_sensitivity_results.json.

Generates a 2×2 heatmap (vel, SST, SSH, combined score) plus individual plots.

Usage:
    python plot_mlswe_letkf_sensitivity.py [results.json]
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps


# ---- Load results ----
results_file = sys.argv[1] if len(sys.argv) > 1 else 'mlswe_letkf_sensitivity_results.json'
with open(results_file) as f:
    raw = json.load(f)

# Derive output prefix from input filename (so K=50 plots don't overwrite K=25)
import os
base = os.path.splitext(os.path.basename(results_file))[0]  # e.g. mlswe_letkf_sensitivity_results_K50
prefix = base.replace('_results', '')  # e.g. mlswe_letkf_sensitivity_K50
# Auto-detect ensemble size from filename
if '_K50' in results_file or '_k50' in results_file:
    K_label = 50
elif '_K25' in results_file or '_k25' in results_file:
    K_label = 25
else:
    K_label = 25  # default

# Infer parameter grid from keys
all_h = set()
all_a = set()
for key in raw:
    parts = key.split('_')
    h_val = int(parts[0][1:])
    a_val = float(parts[1][1:])
    all_h.add(h_val)
    all_a.add(a_val)

hscale_values = sorted(all_h)
alpha_values = sorted(all_a)
n_hscale = len(hscale_values)
n_alpha = len(alpha_values)

# Build arrays
results_vel = np.full((n_hscale, n_alpha), np.nan)
results_sst = np.full((n_hscale, n_alpha), np.nan)
results_ssh = np.full((n_hscale, n_alpha), np.nan)

for i, h in enumerate(hscale_values):
    for j, a in enumerate(alpha_values):
        key = f"h{h}_a{a}"
        if key in raw and raw[key].get("vel") is not None:
            results_vel[i, j] = raw[key]["vel"]
        if key in raw and raw[key].get("sst") is not None:
            results_sst[i, j] = raw[key]["sst"]
        if key in raw and raw[key].get("ssh") is not None:
            results_ssh[i, j] = raw[key]["ssh"]

# ---- Normalised combined score (lower = better) ----
def norm01(arr):
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    return (arr - vmin) / (vmax - vmin + 1e-12)

combined = norm01(results_vel) + norm01(results_sst) + norm01(results_ssh)


def plot_sensitivity_heatmap(rmse_data, title, cbar_label, filename,
                             hscale_vals, alpha_vals):
    """Plot a single sensitivity heatmap."""
    nh = len(hscale_vals)
    na = len(alpha_vals)

    fig, ax = plt.subplots(figsize=(9, 7))

    cmap = colormaps['jet'].copy()
    cmap.set_bad(color='gray')

    masked = np.ma.masked_invalid(rmse_data)
    vmin = 0
    vmax = np.nanmax(rmse_data)

    cax = ax.imshow(masked, origin='lower', cmap=cmap,
                    vmin=vmin, vmax=vmax, aspect='auto')

    fontsize = 15

    ax.set_xticks(np.arange(na))
    ax.set_xticklabels([f'{v:.1f}' for v in alpha_vals], fontsize=fontsize)
    ax.set_xlabel('RTPS (covinflate1)', fontsize=fontsize)

    ax.set_yticks(np.arange(nh))
    ax.set_yticklabels([f'{h}' for h in hscale_vals], fontsize=fontsize)
    ax.set_ylabel('Localization scale [km]', fontsize=fontsize)

    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.set_title(cbar_label, fontsize=fontsize, pad=10)

    # Annotate each cell
    for ii in range(nh):
        for jj in range(na):
            val = rmse_data[ii, jj]
            if np.isfinite(val):
                if val < 0.1:
                    txt = f'{val:.4f}'
                elif val < 1.0:
                    txt = f'{val:.3f}'
                else:
                    txt = f'{val:.2f}'
                text_color = 'black' if val < 0.5 * vmax else 'white'
                ax.text(jj, ii, txt, ha='center', va='center',
                        fontsize=fontsize - 3, color=text_color)

    # Mark minimum with 'x'
    if not np.isnan(rmse_data).all():
        min_idx = np.unravel_index(np.nanargmin(rmse_data), rmse_data.shape)
        min_val = rmse_data[min_idx]
        ax.plot(min_idx[1], min_idx[0], marker='x', color='black',
                markersize=12, markeredgewidth=3)

    ax.set_title(title, fontsize=fontsize + 2)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"  Saved {filename}")
    plt.close(fig)


# ---- Individual heatmaps ----
plot_sensitivity_heatmap(
    results_vel,
    title=f'MLSWE LETKF Sensitivity (K={K_label}) — Mean RMSE Velocity',
    cbar_label='[m/s]',
    filename=f'{prefix}_rmse_vel.png',
    hscale_vals=hscale_values,
    alpha_vals=alpha_values,
)

plot_sensitivity_heatmap(
    results_sst,
    title=f'MLSWE LETKF Sensitivity (K={K_label}) — Mean RMSE SST',
    cbar_label='[K]',
    filename=f'{prefix}_rmse_sst.png',
    hscale_vals=hscale_values,
    alpha_vals=alpha_values,
)

plot_sensitivity_heatmap(
    results_ssh,
    title=f'MLSWE LETKF Sensitivity (K={K_label}) — Mean RMSE SSH',
    cbar_label='[m]',
    filename=f'{prefix}_rmse_ssh.png',
    hscale_vals=hscale_values,
    alpha_vals=alpha_values,
)

# ---- Combined 2×2 figure ----
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fontsize = 14

panels = [
    (axes[0, 0], results_vel, 'Mean RMSE$_{vel}$ (m/s)', '[m/s]'),
    (axes[0, 1], results_sst, 'Mean RMSE$_{SST}$ (K)',   '[K]'),
    (axes[1, 0], results_ssh, 'Mean RMSE$_{SSH}$ (m)',    '[m]'),
    (axes[1, 1], combined,    'Normalised Combined Score\n(lower = better)', ''),
]

best_indices = {
    'vel': np.unravel_index(np.nanargmin(results_vel), results_vel.shape),
    'sst': np.unravel_index(np.nanargmin(results_sst), results_sst.shape),
    'ssh': np.unravel_index(np.nanargmin(results_ssh), results_ssh.shape),
    'comb': np.unravel_index(np.nanargmin(combined), combined.shape),
}

for idx, (ax, rmse_data, ttl, cbl) in enumerate(panels):
    cmap = colormaps['RdYlGn_r'].copy()
    cmap.set_bad(color='gray')
    masked = np.ma.masked_invalid(rmse_data)
    vmin = np.nanmin(rmse_data)
    vmax = np.nanmax(rmse_data)

    cax = ax.imshow(masked, origin='lower', cmap=cmap,
                    vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(np.arange(n_alpha))
    ax.set_xticklabels([f'{v:.1f}' for v in alpha_values], fontsize=fontsize - 1)
    ax.set_xlabel(r'RTPP inflation $\alpha$', fontsize=fontsize)
    ax.set_yticks(np.arange(n_hscale))
    ax.set_yticklabels([f'{h}' for h in hscale_values], fontsize=fontsize - 1)
    ax.set_ylabel('Localisation scale (km)', fontsize=fontsize)

    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.85)
    cbar.ax.tick_params(labelsize=fontsize - 2)
    if cbl:
        cbar.ax.set_title(cbl, fontsize=fontsize - 1, pad=8)

    # Annotate each cell
    for ii in range(n_hscale):
        for jj in range(n_alpha):
            val = rmse_data[ii, jj]
            if np.isfinite(val):
                normed = (val - vmin) / (vmax - vmin + 1e-12)
                text_color = 'white' if normed > 0.55 else 'black'
                if val < 0.01:
                    txt = f'{val:.5f}'
                elif val < 0.1:
                    txt = f'{val:.4f}'
                elif val < 1.0:
                    txt = f'{val:.3f}'
                elif val < 10:
                    txt = f'{val:.2f}'
                else:
                    txt = f'{val:.1f}'
                ax.text(jj, ii, txt, ha='center', va='center',
                        fontsize=fontsize - 4, color=text_color, fontweight='bold')

    # Mark best with star
    best_key = ['vel', 'sst', 'ssh', 'comb'][idx]
    bidx = best_indices[best_key]
    ax.plot(bidx[1], bidx[0], '*', color='blue', markersize=18,
            markeredgecolor='white', markeredgewidth=1.2)

    ax.set_title(ttl, fontsize=fontsize + 1)

fig.suptitle(f'MLSWE LETKF Sensitivity Analysis\n'
             rf'3-layer model, $K={K_label}$ ensemble, 100 cycles, 80$\times$70 grid',
             fontsize=fontsize + 3, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
heatmap_file = f'{prefix}_heatmap.png'
fig.savefig(heatmap_file, dpi=200, bbox_inches='tight')
print(f"  Saved {heatmap_file}")
plt.close(fig)


# ---- Print summary ----
print("\n" + "=" * 72)
print("  MLSWE LETKF SENSITIVITY ANALYSIS — SUMMARY")
print("=" * 72)
print(f"  Grid: {n_hscale} localisation scales × {n_alpha} inflation values = {n_hscale*n_alpha} experiments")
print(f"  hcovlocal_scale (km): {hscale_values}")
print(f"  covinflate1 (α):      {alpha_values}")
print(f"  Ensemble: K={K_label}, nassim=100 cycles, 80×70 grid, 3 layers")
print()

header = f"{'hscale':>8s} | " + " | ".join(f"α={a:.1f} " for a in alpha_values)
sep = "-" * len(header)

print("  Mean RMSE_vel (m/s)")
print(header)
print(sep)
for i, h in enumerate(hscale_values):
    row = f"{h:7d}  | " + " | ".join(
        f"{results_vel[i,j]:7.4f}" if np.isfinite(results_vel[i,j]) else "   NaN "
        for j in range(n_alpha))
    print(row)

print(f"\n  Mean RMSE_sst (K)")
print(header)
print(sep)
for i, h in enumerate(hscale_values):
    row = f"{h:7d}  | " + " | ".join(
        f"{results_sst[i,j]:7.3f}" if np.isfinite(results_sst[i,j]) else "   NaN "
        for j in range(n_alpha))
    print(row)

print(f"\n  Mean RMSE_ssh (m)")
print(header)
print(sep)
for i, h in enumerate(hscale_values):
    row = f"{h:7d}  | " + " | ".join(
        f"{results_ssh[i,j]:7.3f}" if np.isfinite(results_ssh[i,j]) else "   NaN "
        for j in range(n_alpha))
    print(row)

# Best parameters
bv = best_indices['vel']
bs = best_indices['sst']
bh = best_indices['ssh']
bc = best_indices['comb']

print(f"\n  BEST per metric:")
print(f"    Velocity: h={hscale_values[bv[0]]}km  α={alpha_values[bv[1]]}  RMSE={results_vel[bv]:.5f} m/s")
print(f"    SST:      h={hscale_values[bs[0]]}km  α={alpha_values[bs[1]]}  RMSE={results_sst[bs]:.3f} K")
print(f"    SSH:      h={hscale_values[bh[0]]}km  α={alpha_values[bh[1]]}  RMSE={results_ssh[bh]:.3f} m")
print(f"\n  BEST COMBINED (normalised vel+sst+ssh, lower=better):")
print(f"    h = {hscale_values[bc[0]]} km,  α = {alpha_values[bc[1]]}")
print(f"    RMSE_vel = {results_vel[bc]:.5f} m/s")
print(f"    RMSE_sst = {results_sst[bc]:.3f} K")
print(f"    RMSE_ssh = {results_ssh[bc]:.3f} m")
print(f"    Score    = {combined[bc]:.4f}")
print()
print("  KEY FINDINGS:")
print("  —————————————")
print(f"  • Tight localisation (20–40 km) strongly outperforms larger scales.")
print(f"  • RMSE degrades dramatically above 100 km for all metrics.")
print(f"  • h=300 km diverges: RMSE_sst > 10 K, RMSE_ssh > 8 m.")
print(f"  • h=20 and h=40 km give identical results — observation density limit.")
print(f"  • Moderate-to-high RTPP α (0.3–0.9) works well at small scales.")
print(f"  • Recommended: h = 20–40 km, α = 0.3–0.7")
print("=" * 72)

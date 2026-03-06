#!/usr/bin/env python3
"""
Generate a 2x2 illustration comparing V1 (joint) and V2 (halo-based per-block)
localization for linear vs nonlinear data models.

  (0,0)  V1 linear   -- direct sampling on the shaded domain
  (0,1)  V2 linear   -- direct sampling on each block in parallel
  (1,0)  V1 nonlinear -- parallel MCMC chains on the shaded domain
  (1,1)  V2 nonlinear -- MCMC chain on each block with N_a iterations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

np.random.seed(42)

# -- Grid and block parameters -------------------------------------------------
Nx, Ny = 100, 100
bx, by = 10, 8          # 80 blocks
block_w = Nx / bx
block_h = Ny / by

# -- Generate random observation locations --------------------------------------
n_obs = 120
obs_x = np.random.uniform(0, Nx, n_obs)
obs_y = np.random.uniform(0, Ny, n_obs)

empty_blocks = {(0, 0), (3, 2), (7, 5), (9, 7), (1, 6),
                (5, 0), (8, 3), (4, 7), (6, 1), (2, 4)}

obs_bi = (obs_x / block_w).astype(int).clip(0, bx - 1)
obs_bj = (obs_y / block_h).astype(int).clip(0, by - 1)

keep = np.array([(obs_bi[k], obs_bj[k]) not in empty_blocks
                 for k in range(n_obs)])
obs_x, obs_y = obs_x[keep], obs_y[keep]
obs_bi, obs_bj = obs_bi[keep], obs_bj[keep]
observed_blocks = set(zip(obs_bi.tolist(), obs_bj.tolist()))

# -- Helpers --------------------------------------------------------------------
def draw_blocks(ax, fill_func=None):
    for i in range(bx):
        for j in range(by):
            x0, y0 = i * block_w, j * block_h
            fc = fill_func(i, j) if fill_func else 'none'
            rect = mpatches.FancyBboxPatch(
                (x0, y0), block_w, block_h,
                boxstyle="square,pad=0",
                facecolor=fc if fc else 'none',
                edgecolor='#555555', linewidth=0.6,
                alpha=0.35 if fc and fc != 'none' else 1.0,
                zorder=1)
            ax.add_patch(rect)


def setup_ax(ax):
    ax.set_xlim(-1, Nx + 1)
    ax.set_ylim(-1, Ny + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.tick_params(labelsize=13)


# -- V1 fill function ----------------------------------------------------------
def v1_fill(i, j):
    return '#4CAF50' if (i, j) in observed_blocks else '#BDBDBD'


# -- V2 selected blocks -- far apart -------------------------------------------
block1 = (2, 2)      # lower-left area
block2 = (3, 6)      # upper area, guaranteed observed
r_loc = 22

c1x = block1[0] * block_w + block_w / 2
c1y = block1[1] * block_h + block_h / 2
c2x = block2[0] * block_w + block_w / 2
c2y = block2[1] * block_h + block_h / 2


def draw_v2_panel(ax):
    """Draw V2 per-block halo localization panel."""
    draw_blocks(ax, fill_func=lambda i, j: 'none')
    ax.scatter(obs_x, obs_y, s=20, c='blue', zorder=5)

    # Highlight two selected blocks
    for (bi, bj), color in [(block1, '#E53935'), (block2, '#FF9800')]:
        x0, y0 = bi * block_w, bj * block_h
        rect = mpatches.FancyBboxPatch(
            (x0, y0), block_w, block_h,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor='k', linewidth=1.8,
            alpha=0.45, zorder=3)
        ax.add_patch(rect)

    # r_loc circles
    for cx, cy, col in [(c1x, c1y, '#E53935'), (c2x, c2y, '#FF9800')]:
        circle = plt.Circle((cx, cy), r_loc, fill=False, edgecolor=col,
                            linewidth=2.0, linestyle='--', zorder=4)
        ax.add_patch(circle)
        ax.plot(cx, cy, 'x', color=col, markersize=9,
                markeredgewidth=2.2, zorder=6)

    # Highlight obs inside circles
    dist1 = np.sqrt((obs_x - c1x)**2 + (obs_y - c1y)**2)
    dist2 = np.sqrt((obs_x - c2x)**2 + (obs_y - c2y)**2)
    in1, in2 = dist1 <= r_loc, dist2 <= r_loc
    ax.scatter(obs_x[in1 & ~in2], obs_y[in1 & ~in2], s=28,
               edgecolors='#E53935', facecolors='none', linewidths=1.4, zorder=6)
    ax.scatter(obs_x[in2 & ~in1], obs_y[in2 & ~in1], s=28,
               edgecolors='#FF9800', facecolors='none', linewidths=1.4, zorder=6)
    ax.scatter(obs_x[in1 & in2], obs_y[in1 & in2], s=28,
               edgecolors='purple', facecolors='none', linewidths=1.4, zorder=6)

    # r_loc annotation
    angle = np.pi / 4
    ax.annotate('', xy=(c1x, c1y),
                xytext=(c1x + r_loc * np.cos(angle),
                        c1y + r_loc * np.sin(angle)),
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.4),
                zorder=7)
    ax.text(c1x + r_loc * 0.45 * np.cos(angle) + 1,
            c1y + r_loc * 0.45 * np.sin(angle) + 2,
            '$r_{\\mathrm{loc}}$', fontsize=16, color='#E53935',
            fontweight='bold', zorder=7)


def draw_v1_panel(ax):
    """Draw V1 joint localization panel."""
    draw_blocks(ax, fill_func=v1_fill)
    ax.scatter(obs_x, obs_y, s=20, c='blue', zorder=5)


# -- V1 legend elements ---------------------------------------------------------
v1_legend = [
    mpatches.Patch(facecolor='#4CAF50', edgecolor='#555555', alpha=0.35,
                   label='Observed blocks'),
    mpatches.Patch(facecolor='#BDBDBD', edgecolor='#555555', alpha=0.35,
                   label='Unobserved blocks'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
           markersize=7, label='Observations'),
]

# -- V2 legend elements ---------------------------------------------------------
v2_legend = [
    mpatches.Patch(facecolor='#E53935', edgecolor='k', alpha=0.45,
                   label='Block $G_i$'),
    mpatches.Patch(facecolor='#FF9800', edgecolor='k', alpha=0.45,
                   label='Block $G_j$'),
    Line2D([0], [0], color='#E53935', linestyle='--', linewidth=2,
           label='$r_{\\mathrm{loc}}$ circle'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
           markersize=7, label='Observations'),
]

# =============================================================================
#  Build 1x2 figure (V1 and V2)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)

# Titles
titles = {
    0: '(a) V1',
    1: '(b) V2',
}

# Descriptive annotations
annotations = {
    0: ('LGOM: Direct sampling on the shaded domain.\n'
        'NLOM or NGOM: Parallel MCMC chains on the shaded domain\n'
        r'with $N_{\mathrm{burn}}+\lceil N_a/P \rceil$ iterations'),
    1: ('LGOM: Direct sampling on each observed block (in parallel).\n'
        'NLOM or NGOM: MCMC chain on each observed block (in parallel)\n'
        r'with $N_{\mathrm{burn}}+N_a$ iterations'),
}

for col in range(2):
    ax = axes[col]
    if col == 0:
        draw_v1_panel(ax)
        ax.legend(handles=v1_legend, loc='upper right',
                  fontsize=13, framealpha=0.9)
    else:
        draw_v2_panel(ax)
        ax.legend(handles=v2_legend, loc='upper right',
                  fontsize=12, framealpha=0.9)

    setup_ax(ax)
    ax.set_title(titles[col], fontsize=16, fontweight='bold', pad=10)

    # Place annotation inside the plot at bottom-left
    ax.text(0.03, 0.03, annotations[col],
            transform=ax.transAxes, fontsize=13,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='wheat',
                      edgecolor='gray', alpha=0.88),
            zorder=10)

# -- Save ----------------------------------------------------------------------
fig.savefig('figures/localization_v1_v2_illustration.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig('figures/localization_v1_v2_illustration.png',
            bbox_inches='tight', dpi=300)
plt.close()
print("Saved: figures/localization_v1_v2_illustration.pdf")
print("Saved: figures/localization_v1_v2_illustration.png")

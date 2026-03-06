# MLSWE\_LSMCMC — Multi-Layer Shallow-Water Equations with Data Assimilation

Companion code for:

> **Two Localization Strategies for Sequential MCMC Data Assimilation
> with Applications to Nonlinear Non-Gaussian Geophysical Models** \
> Hamza Ruzayqat, Hristo G. Chipilski, Omar Knio


Three-layer isopycnal shallow-water equations solver with two data
assimilation methods — **LSMCMC** (Local Sequential Monte Carlo with MCMC
moves) and **LETKF** (Local Ensemble Transform Kalman Filter) — applied
to the North Atlantic.  A companion Jupyter notebook
([LSMCMC.ipynb](LSMCMC.ipynb)) provides a
step-by-step running guide and reproduces all figures in the paper.

---

## Table of Contents

1. [Governing Equations](#governing-equations)
2. [Numerical Methods](#numerical-methods)
3. [Boundary Conditions](#boundary-conditions)
4. [Observations](#observations)
5. [Data Assimilation — LSMCMC (Linear)](#data-assimilation--lsmcmc)
6. [LSMCMC Localization Variants](#lsmcmc-localization-variants)
7. [Nonlinear LSMCMC](#nonlinear-lsmcmc)
8. [Data Assimilation — LETKF](#data-assimilation--letkf)
9. [SSH Control Strategy](#ssh-control-strategy)
10. [Sensitivity Analysis](#sensitivity-analysis)
11. [Implementation Notes: Bathymetry and Island Cells](#implementation-notes-bathymetry-and-island-cells)
12. [Experiments](#experiments)
13. [Data Preparation](#data-preparation)
14. [Project Structure](#project-structure)
15. [Configuration Reference](#configuration-reference)
16. [Usage Examples](#usage-examples)
17. [Generating Plots](#generating-plots)
18. [Dependencies](#dependencies)
19. [References](#references)
20. [Citation](#citation)

---



---

## Governing Equations

### Layer Configuration

The ocean is represented by three stacked isopycnal layers indexed
$k = 0$ (surface), $1$ (thermocline), $2$ (deep/bottom), each with
constant reference density $\rho_k$.

| Layer | Description  | $\rho_k$ (kg m⁻³) | $H_{\text{rest},k}$ (m) | $T_{\text{rest},k}$ (K) |
|:-----:|:-------------|:-------------------:|:------------------------:|:------------------------:|
| 0     | Surface      | 1023                | 100                      | 298.15 (≈ 25 °C)        |
| 1     | Thermocline  | 1026                | 400                      | 283.15 (≈ 10 °C)        |
| 2     | Deep/bottom  | 1028                | 3500                     | 275.15 (≈ 2 °C)         |

Total rest-state depth: $\sum_k H_{\text{rest},k} = 4000$ m.

### Reduced Gravities

$$
g'_{j,k} = g\,\frac{\rho_k - \rho_j}{\rho_{\text{ref}}},
\qquad \rho_{\text{ref}} = 1025 \text{ kg m}^{-3}
$$

| Pair   | Value (m s⁻²) |
|:------:|:-------------:|
| $g'_{01}$ | 0.0287     |
| $g'_{02}$ | 0.0479     |
| $g'_{12}$ | 0.0191     |

### State Vector

$$
\mathbf{x} = \bigl[\,
  \underbrace{h_0,\, u_0,\, v_0,\, T_0}_{\text{layer 0}},\;
  \underbrace{h_1,\, u_1,\, v_1,\, T_1}_{\text{layer 1}},\;
  \underbrace{h_2,\, u_2,\, v_2,\, T_2}_{\text{layer 2}}
\,\bigr]
$$

Each field lives on an $(n_y \times n_x)$ grid, giving
$\dim(\mathbf{x}) = 12 \times n_y \times n_x$.
With the default grid $(70 \times 80)$ this is **67 200**.

> **Implementation note.** In the flat state vector the $h_0$ slot stores
> $h_{\text{total}} = \sum_k h_k$ (total water-column thickness) while
> $h_1,\, h_2$ are zero placeholders. Individual layer thicknesses are
> recovered by proportional splitting before each RHS evaluation.

### Sea-Surface Height

$$
\eta = \sum_{k=0}^{2} h_k - H_b
$$

where $H_b(x,y)$ is the bathymetric depth (positive downward, from ETOPO).

### Continuity (Barotropic)

A single barotropic continuity equation governs the total thickness:

$$
\frac{\partial h_{\text{total}}}{\partial t}
  = -\nabla \cdot \!\left(\sum_{k=0}^{2} h_k\,\mathbf{u}_k\right) +
    \nu\,\nabla^2 h_{\text{total}} -
    \lambda_{\text{ssh}}\bigl(\eta - \eta_{\text{ref}}(t)\bigr)
$$

The last term is **SSH relaxation** toward a HYCOM reference
(see [SSH Control Strategy](#ssh-control-strategy)).

The tendency is distributed to individual layers proportionally:

$$
\frac{\partial h_k}{\partial t}
  = \frac{H_{\text{rest},k}}{H_{\text{rest,total}}}
    \;\frac{\partial h_{\text{total}}}{\partial t}
$$

### Momentum (per layer $k$)

$$
\frac{\partial u_k}{\partial t}
  = -u_k\frac{\partial u_k}{\partial x} -
    v_k\frac{\partial u_k}{\partial y} +
    f\,v_k -
    g\frac{\partial\eta}{\partial x} -
    r_b\,u_k +
    \nu\,\nabla^2 u_k
$$

$$
\frac{\partial v_k}{\partial t}
  = -u_k\frac{\partial v_k}{\partial x} -
    v_k\frac{\partial v_k}{\partial y} -
    f\,u_k -
    g\frac{\partial\eta}{\partial y} -
    r_b\,v_k +
    \nu\,\nabla^2 v_k
$$

where:

| Symbol | Description | Default |
|--------|-------------|---------|
| $f$    | Coriolis parameter (latitude-dependent or constant) | $f_0 = 10^{-4}$ s⁻¹ |
| $g$    | Gravitational acceleration | 9.81 m s⁻² |
| $r_b$  | Linear bottom drag coefficient (all layers) | $10^{-6}$ s⁻¹ |
| $\nu$  | Momentum diffusion coefficient | 500 m² s⁻¹ |

> **Note.** Baroclinic pressure-gradient terms
> $\sum_{j<k} g'_{j,k}\,\nabla h_j$ are **not applied** because, with
> the proportional layering scheme, they introduce a spurious
> $\nabla H_b$ forcing. Only the barotropic PGF $-g\nabla\eta$ is used.

### Temperature (Passive Tracer, per layer $k$)

$$
\frac{\partial T_k}{\partial t}
  = -u_k\frac{\partial T_k}{\partial x} -
    v_k\frac{\partial T_k}{\partial y} +
    \kappa_T\,\nabla^2 T_k +
    \mathcal{F}_k
$$

where $\kappa_T$ is the tracer diffusion coefficient (default 100 m² s⁻¹)
and $\mathcal{F}_k$ collects forcing terms that act only on the surface
layer ($k=0$):

**SST nudging** (Newtonian relaxation toward HYCOM SST):

$$
\mathcal{F}_{\text{nudge}} = -\lambda_{\text{SST}}\bigl(T_0 - T_{\text{ref}}(t)\bigr),
\qquad \lambda_{\text{SST}} = 2.78 \times 10^{-4}\text{ s}^{-1}\;(\tau \approx 1\text{ h})
$$

**Surface heat flux** (Newtonian bulk formula):

$$
\mathcal{F}_{\text{flux}} = -\frac{\alpha}{\rho_{sw}\,c_p\,h_{\text{mix}}}
  \bigl(T_0 - T_{\text{air}}(t)\bigr)
$$

| Symbol | Value | Description |
|--------|-------|-------------|
| $\alpha$ | 100 W m⁻² K⁻¹ | Bulk transfer coefficient |
| $\rho_{sw}$ | 1025 kg m⁻³ | Seawater density |
| $c_p$ | 3990 J kg⁻¹ K⁻¹ | Specific heat capacity |
| $h_{\text{mix}}$ | 0.1 m | Mixed-layer depth for flux |

---

## Numerical Methods

### Spatial Discretization

- **Grid type:** Equidistant **Arakawa A-grid** (all variables collocated).
- **Domain:** North Atlantic, longitude $[-60°,\, -20°]$ (80 pts),
  latitude $[10°,\, 45°]$ (70 pts).
- **Grid spacing:** $\Delta x = \Delta\lambda\,a\cos\phi / (180/\pi)$,
  $\Delta y = \Delta\phi\,a / (180/\pi)$, evaluated at the domain centre.

**First derivatives** — 2nd-order centred (interior), 1st-order one-sided
(boundaries):

$$
\left.\frac{\partial q}{\partial x}\right|_{i,j}
  \approx \frac{q_{i,j+1} - q_{i,j-1}}{2\,\Delta x}
$$

**Laplacian** — standard 5-point stencil:

$$
\nabla^2 q \approx
  \frac{q_{i,j+1} - 2q_{i,j} + q_{i,j-1}}{\Delta x^2} +
  \frac{q_{i+1,j} - 2q_{i,j} + q_{i-1,j}}{\Delta y^2}
$$

Bi-harmonic diffusion (`diff_order=2`) is also supported by applying the
Laplacian twice with sign flip.

### Time Integration

Classical **4th-order Runge–Kutta** (RK4):

$$
\mathbf{x}^{n+1} = \mathbf{x}^n +
  \frac{\Delta t}{6}\bigl(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\bigr)
$$

Boundary conditions are re-applied at each sub-stage.

### Post-Step Projections

After each RK4 step:

1. **Proportional layer projection:**
   $h_k = (H_{\text{rest},k}/H_{\text{rest,total}})\,\max(h_{\text{total}},\, 10)$
2. **Barotropic velocity projection:** all layers set to the mass-weighted
   mean velocity:
   $\bar{u} = \sum_k h_k u_k \,/\, \max(h_{\text{total}},\, 1)$,
   then $u_k \leftarrow \bar{u}$ for all $k$.
3. **Stability clamps:** $|u_k|,\,|v_k| \leq 20$ m s⁻¹,
   $T_k \in [260,\, 310]$ K, $h_k \geq 5$ m.
4. **NaN recovery:** replace NaN values with rest-state.

### CFL Condition

The time step is checked against:

$$
\Delta t_{\text{CFL}} = 0.5\;\frac{\min(\Delta x,\,\Delta y)}{\sqrt{g\,H_{\max}}}
$$

Auto-reduced to $0.9\,\Delta t_{\text{CFL}}$ if violated. Default
$\Delta t = 75$ s.

---

## Boundary Conditions

HYCOM reanalysis surface fields (SSH, $u_o$, $v_o$, SST) from
`data/hycom_bc.nc` are distributed to the 3-layer boundary ghost cells
($n_{\text{ghost}} = 2$).

### Thickness (SSH → $h_k$)

$$
h_k^{\text{bdy}}
  = \frac{H_{\text{rest},k}}{H_{\text{rest,total}}}
    \;\max\!\bigl(H_b^{\text{bdy}} + \eta_{\text{HYCOM}},\; 10\bigr)
$$

### Velocity

$$
u_k^{\text{bdy}} = \beta_k\,u_{\text{HYCOM}},
\qquad
v_k^{\text{bdy}} = \beta_k\,v_{\text{HYCOM}}
$$

Default $\boldsymbol{\beta} = (1.0,\, 1.0,\, 1.0)$ (depth-uniform).

### Temperature

$$
T_0^{\text{bdy}} = \text{SST}_{\text{HYCOM}},
\qquad
T_{1,2}^{\text{bdy}} = T_{\text{rest},k}
$$

### Implementation

All HYCOM fields are pre-interpolated from the BC grid to model boundary
points at construction time. At runtime, linear temporal interpolation
between bracketing snapshots is applied. NaN (land) values are filled
using nearest-neighbour via the Euclidean distance transform.

---

## Observations

### Surface Drifters

OSMC drifter data, August 1–12 2024.
Observations are **surface-only** and update layer-0 fields:

| Observable | Field index | Noise $\sigma_y$ |
|:----------:|:-----------:|:-----------------:|
| $u_0$      | 1           | 0.10 m s⁻¹       |
| $v_0$      | 2           | 0.10 m s⁻¹       |
| SST ($T_0$)| 3           | 0.4 K             |

### SWOT SSH

Real SWOT satellite along-track SSH observations from PO.DAAC
(Level-2 KaRIn Low Rate products).

#### SWOT Data Processing Pipeline

**Step 1 — Absolute Dynamic Topography (ADT).**
Raw SWOT L2 LR NetCDF files contain several SSH-related variables that
must be combined to obtain ADT comparable to model $\eta$:

$$
\texttt{ADT} = \texttt{ssha\textunderscore karin} +
             \texttt{height\textunderscore cor\textunderscore xover} +
             \underbrace{(\texttt{MSS}_{\text{CNES/CLS}} - \texttt{geoid})}_{\text{MDT}}
$$

| Variable | Description |
|----------|-------------|
| `ssha_karin` | Sea surface height anomaly from KaRIn (2D: nlines × npixels) |
| `height_cor_xover` | Crossover calibration correction (removes $\pm 3$ m orbit bias) |
| `mean_sea_surface_cnescls` | CNES/CLS Mean Sea Surface (MSS) |
| `geoid` | Geoid height |

The crossover correction (`height_cor_xover`) is critical — without it,
adjacent ascending/descending passes can differ by several metres due
to residual orbit errors. The quality flag `height_cor_xover_qual` is
checked: only points with quality $\leq 1$ (good or suspect) are retained.

**Step 2 — Quality Filtering.**

```
valid = isfinite(ssha_karin) & isfinite(lon) & isfinite(lat)
      & isfinite(height_cor_xover) & (xover_qual <= 1)
      & isfinite(MSS) & isfinite(geoid)
      & (ssha_karin_qual == 0)
```

**Step 3 — Binning onto the Model Grid.**
Valid SWOT pixels are binned onto the $80 \times 70$ model grid
($0.5°$ cells). For each valid pixel, the nearest model grid cell is
found via `np.searchsorted`; only pixels within half a grid-cell
distance are retained. Each pixel is assigned to the nearest
assimilation cycle (1-hour intervals, $\pm 1$ hour tolerance). Within
each (cycle, cell) bin, all ADT values are averaged to produce a single
observation. Output: `swot_ssh_binned_80x70.nc`.

**Step 4 — Gap Filling with Synthetic SWOT.**
For assimilation cycles without real SWOT coverage (the satellite passes
a given region only every $\sim$11–13 cycles), synthetic SWOT-like
observations are generated from the HYCOM SSH reference field. Two
simulated tracks separated by $\sim 25°$ longitude sweep the domain,
each with a $\sim 1.5°$ swath width ($\approx 120$ km, matching the
real KaRIn swath). Additive Gaussian noise with $\sigma = 0.5$ m is
applied. The real and synthetic observations are merged into
`swot_ssh_combined_80x70.nc` with a source flag (0 = real, 1 = synthetic).

**Step 5 — Assimilation.**
Binned SSH values are mapped to field index 0
($h_{\text{total}} = H_b + \eta + \varepsilon$) with observation noise
$\sigma_{\text{ssh}} = 0.10$ m (production) / $0.50$ m (default).
Approximately 15–25% of grid cells are observed per cycle.

Drifter and SSH observations are merged into a single observation vector
per assimilation cycle.

---

## Data Assimilation — LSMCMC

### Overview

**Local Sequential Monte Carlo with MCMC moves.** The domain is
partitioned into rectangular subdomains. Within each subdomain containing
observations, the posterior is sampled using an exact-from-Gaussian
proposal with mixture importance weighting.

### Algorithm

For each assimilation cycle $n = 1, \ldots, N_{\text{assim}}$:

**Step 1 — Forecast.** Advance all $N_f$ ensemble members by `t_freq`
timesteps. Add process noise:

$$
\mathbf{x}_j^f \;\leftarrow\; \mathbf{x}_j^f + \boldsymbol{\varepsilon}_j,
\qquad \boldsymbol{\varepsilon}_j \sim \mathcal{N}(\mathbf{0},\,\text{diag}(\boldsymbol{\sigma}_\mathbf{x}^2))
$$

| Field | Noise std $\sigma_{\mathbf{x}}$ |
|-------|:-------------------------------:|
| $h_0$ (total thickness) | `sig_x_ssh` = 0.2 (real data) / 0.5 (twin) |
| $u_0,\, v_0$            | `sig_x_uv` = 0.15    |
| $T_0$ (SST)             | `sig_x_sst` = 1.0 |
| All layer 1, 2 fields   | 0 (unobserved)    |

**Step 2 — Localisation.** Partition domain into $N_{\text{sub}}$ = 50
rectangular blocks. Only blocks with $\geq 1$ observation are updated.
Only layer-0 fields ($h_0, u_0, v_0, T_0$) are included in the local
state vector.

**Step 3 — Posterior Sampling (Exact-from-Gaussian).**

The prior is a Gaussian mixture:

$$
\pi_{\text{prior}}(\mathbf{x})
  = \frac{1}{N_f}\sum_{i=1}^{N_f}
    \mathcal{N}\!\bigl(\mathbf{x};\;\mathbf{m}_i,\;\boldsymbol{\Sigma}_\mathbf{x}\bigr)
$$

The likelihood is:

$$
g(\mathbf{x},\,\mathbf{y})
  = \mathcal{N}\!\bigl(\mathbf{y};\;\mathbf{H}\mathbf{x},\;\boldsymbol{\Sigma}_\mathbf{y}\bigr)
$$

Each mixture component has a Gaussian posterior with mean:

$$
\boldsymbol{\mu}_i
  = \mathbf{m}_i +
    \boldsymbol{\Sigma}_\mathbf{x}\,\mathbf{H}^T\,\mathbf{S}^{-1}
      (\mathbf{y} - \mathbf{H}\mathbf{m}_i)
$$

where $\mathbf{S} = \boldsymbol{\Sigma}\_\mathbf{y} + \mathbf{H}\boldsymbol{\Sigma}\_\mathbf{x}\mathbf{H}^T$.

**Step 4 — Importance Weights.**

$$
\log w_i = -\tfrac{1}{2}\bigl\|\mathbf{L}^{-1}(\mathbf{y} - \mathbf{H}\mathbf{m}_i)\bigr\|^2,
\qquad \mathbf{S} = \mathbf{L}\mathbf{L}^T
$$

Normalised: $w_i \leftarrow w_i / \sum_j w_j$.

**Step 5 — Sampling.** Draw $N_{\text{MCMC}}$ samples: select component
$i$ with probability $w_i$, then draw:

$$
\mathbf{x}^*
  = \boldsymbol{\mu}_i +
    \mathbf{z}_0 -
    \boldsymbol{\Sigma}_\mathbf{x}\,\mathbf{H}^T\,\mathbf{S}^{-1}
      (\mathbf{H}\mathbf{z}_0 + \mathbf{L}\boldsymbol{\eta})
$$

where $\mathbf{z}\_0 = \text{diag}(\boldsymbol{\sigma}\_\mathbf{x})\,\boldsymbol{\varepsilon}$,
$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$,
$\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$.

**Step 6 — Gaussian Block Means.** Randomly partition the $N_{\text{MCMC}}$
samples into $N_f$ groups and average each group → $N_f$ analysis
ensemble members (low-variance resampling).

**Step 7 — Post-Analysis SSH Relaxation.** See
[SSH Control Strategy](#ssh-control-strategy).

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nforecast` | 25 | Ensemble size $N_f$ |
| `mcmc_N` | 500 | MCMC samples per subdomain |
| `burn_in` | 500 | MCMC burn-in |
| `num_subdomains` | 50 | Number of localization blocks |

### Ensemble Initialization

All ensemble members are initialized identically to the HYCOM initial
condition (no random perturbation). This is required by SMCMC theory:
the process noise injected at each forecast step generates the necessary
ensemble spread, and perturbing the initial state can degrade filter
performance by introducing artificial diversity that does not reflect
the true prior uncertainty.

---

## LSMCMC Localization Variants

Two localization strategies are implemented for LSMCMC. Both use the
same exact-from-Gaussian posterior sampling (linear case) or MCMC
sampling (nonlinear case), but differ in how the domain is partitioned,
how observations are associated with state variables, and **how the
likelihood $g$ and transition kernel $f$ are restricted to local
subsets of the state**.

At each assimilation cycle, the LSMCMC target distribution is:

$$
\hat{\pi}(z, i) \propto
  \underbrace{g(z, \mathbf{y})}_{\text{likelihood}} \;\cdot\;
  \underbrace{f(\mathbf{m}_i, z)}_{\text{transition}} \;\cdot\;
  p(i)
$$

Localization restricts both $g$ and $f$ to operate on **local subsets**
of the full state vector $z$, reducing the dimensionality of the
sampling problem from $d = 67{,}200$ to a manageable local dimension
$d_{\text{loc}}$.

### Variant 1: Block-Partition Localization (Original)

**File:** `mlswe/lsmcmc_V1.py` (linear), `mlswe/lsmcmc_nl_V1.py` (nonlinear)

The domain is divided into $N_{\text{sub}}$ rectangular blocks using
`partition_domain()`. At each assimilation cycle, blocks containing at
least one observation are identified. The local state vector for each
active block consists of **all grid cells** within those blocks (layer-0
fields only). All observations falling within the union of active blocks
are used in a single global posterior-sampling step.

This approach treats the observed region as one connected local problem:

$$
\mathbf{x}_{\text{local}} = \bigl\lbrace\mathbf{x}[c] \;:\; c \in \bigcup_{b \in \mathcal{B}_{\text{obs}}} \text{cells}(b)\bigr\rbrace
$$

where $\mathcal{B}_{\text{obs}}$ is the set of blocks containing
observations. The observation operator $\mathbf{H}$ maps the local state
to the full observation vector.

**How $g$ and $f$ are affected in V1:**

- **Likelihood $g$:** The observation operator $\mathbf{H}$ is restricted
  to the observed blocks. In the linear case,

  $$g(z_{\text{loc}}, \mathbf{y}) = \mathcal{N}(\mathbf{y};\; \mathbf{H}_{\text{loc}}\, z_{\text{loc}},\; \boldsymbol{\Sigma}_y)$$

  In the nonlinear case,

  $$g(z_{\text{loc}}, \mathbf{y}) = \mathcal{N}(\mathbf{y};\; \arctan(\mathbf{H}_{\text{loc}}\, z_{\text{loc}}),\; \boldsymbol{\Sigma}_y)$$

  All observations in the union of observed blocks contribute with equal
  weight — no spatial tapering is applied.

- **Transition kernel $f$:** The prior

  $$f(\mathbf{m}_i, z) = \mathcal{N}(z;\; \mathbf{m}_i,\; \boldsymbol{\Sigma}_x)$$

  is restricted to the same local indices: only the diagonal entries of
  $\boldsymbol{\Sigma}_x$ corresponding to the observed-block cells are used.
  Unobserved blocks retain their forecast mean unchanged.

**Advantages:**
- Simple implementation; single posterior-sampling call per cycle.
- All observations contribute to a single consistent update.
- Naturally handles spatially dense observation networks.

**Disadvantages:**
- The local state dimension can be large when many blocks are active,
  increasing the cost of Cholesky factorization $\mathcal{O}(d^3)$.
- No spatial tapering — observations at the edge of an active block
  have the same influence as those at the centre.

### Variant 2: Halo-Based Localization with Gaspari–Cohn Tapering

**File:** `mlswe/lsmcmc_V2.py` (linear), `mlswe/lsmcmc_nl_V2.py` (nonlinear)

Inspired by the SQG localization approach, this variant processes each block
independently with a **halo** of radius $r_{\text{loc}}$ grid cells
around it. For each block $b$:

1. Expand block $b$ by $r_{\text{loc}}$ cells in all directions to form
   the **halo region** $\mathcal{H}_b$.
2. Select all observations falling within $\mathcal{H}_b$.
3. Apply **Gaspari–Cohn tapering** to the observation noise: inflate
   $\sigma_y^2$ by $1/w_{\text{GC}}(r)$ where $r$ is the distance from
   the observation to the block centre and $w_{\text{GC}}$ is the
   compactly supported correlation function. This down-weights distant
   observations smoothly.
4. Solve the local posterior for the halo state, then **retain only the
   cells belonging to block $b$** (discard the halo margin).

**How $g$ and $f$ are affected in V2:**

- **Likelihood $g$:** For each block $b$, the observation noise covariance
  is inflated by the inverse Gaspari–Cohn weight:

$$
\sigma_{y,\text{loc}}^2(r) = \frac{\sigma_y^2}{\max\bigl(w_{\text{GC}}(r),\; \epsilon\bigr)}
$$

  This means observations near the block centre contribute at full
  strength (weight $\approx 1$), while observations near the halo
  boundary are down-weighted (weight $\to 0$) — effectively making the
  likelihood **spatially tapered**. In the nonlinear case, the same
  tapering applies to the arctan likelihood.

- **Transition kernel $f$:** The prior

  $$f(\mathbf{m}_i, z_{\text{halo}}) = \mathcal{N}(z_{\text{halo}};\; \mathbf{m}_{i,\text{halo}},\; \boldsymbol{\Sigma}_{x,\text{halo}})$$

  uses only the halo subset of the forecast. After sampling, only the
  **core block cells** are retained; halo margin cells are discarded.
  This means the transition kernel operates on a larger state than what
  is kept, providing smooth boundary conditions for the local problem.

After all blocks are processed independently (parallelized via
`ThreadPoolExecutor`), the per-block analysis fields are stitched back
into the full state vector. RTPS inflation is applied globally afterward.

**Advantages:**
- Each block is a small independent problem — well-suited for parallel
  execution.
- Gaspari–Cohn tapering provides smooth spatial decay of observation
  influence, reducing edge effects.
- Scales well to large domains and dense observation networks.

**Disadvantages:**
- Block boundaries can introduce discontinuities in the analysis
  (mitigated by the overlapping halo).
- Requires careful tuning of $r_{\text{loc}}$ and block count.
- Observations influence multiple overlapping halos, but each halo solves
  independently — no global consistency guarantee.

### Localization Illustration

The figure below illustrates both localization strategies. The left column shows a
simple domain partitioned into blocks; the right column shows how V1 (block union)
and V2 (halo + tapering) handle the same observation pattern.

![Localization V1 vs V2](figures/localization_v1_v2_illustration.png)

### Additional Features in Variant 2

- **RTPS inflation:** After per-block analysis, the ensemble spread is
  relaxed toward the prior spread with coefficient $\alpha_{\text{RTPS}}$:

$$
{x'}^a_{j} \;\leftarrow\; {x'}^a_{j} \cdot \left(1 + \alpha_{\text{RTPS}}\;\frac{\sigma^b - \sigma^a}{\max(\sigma^a,\; 10^{-10})}\right)
$$

- **Periodic ensemble reset:** Every $N_{\text{reset}}$ cycles, the
  ensemble is re-collapsed to the mean (all members set to the current
  analysis mean). This prevents ensemble collapse from accumulating over
  long integrations.

### Comparison: V1 vs V2 Results — Linear Case (240 Cycles)

Both variants were run with identical settings: 25 ensemble members,
identical initialization from HYCOM, 240 assimilation cycles (1-hour
intervals), OSMC drifter + SWOT SSH observations, linear observation
operator ($\mathbf{H}\mathbf{x}$).

| | V1 (Block-Partition) | V2 (Halo, $r_{\text{loc}}=20$, 80 blocks) |
|--|:---:|:---:|
| Mean vel RMSE (m/s) | **0.0098** | 0.0260 |
| Mean SST RMSE (K) | **0.146** | 0.175 |
| Mean SSH RMSE (m) | **0.537** | 0.575 |
| Wall time per cycle | ~1.4 s | ~1.7 s |

#### Key Observations (Linear Case)

1. **V1 outperforms V2** on all metrics by $\sim 2.5\times$ in velocity.
   The block-partition approach benefits from solving a single, globally
   consistent posterior over all observed blocks, avoiding the
   information loss inherent in treating blocks independently.

2. **V2 required careful tuning** to avoid SSH blow-up near complex
   bathymetry. With the initial settings ($r_{\text{loc}} = 15$,
   50 blocks), SSH diverged near the Canary Islands. Increasing to
   $r_{\text{loc}} = 20$ and 80 blocks resolved the issue.

3. **V1's superior performance** in the linear case stems from the fact
   that, on this $70 \times 80$ grid with ~560 observations per cycle,
   the observed blocks span most of the domain — making the "local"
   problem nearly global. V2's per-block independence is more
   advantageous on larger grids where the global problem would be
   prohibitively expensive.

4. Both variants use **exact-from-Gaussian sampling** in the linear case,
   which produces perfectly distributed posterior samples with no MCMC
   mixing issues.

---

## Nonlinear LSMCMC

### Overview

The nonlinear LSMCMC extends the linear filter to handle a **nonlinear
observation operator**. A synthetic twin experiment uses the arctan
observation operator

$$
\mathbf{y} = \arctan(\mathbf{H}\mathbf{x}) + \boldsymbol{\varepsilon},
\qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\,\boldsymbol{\Sigma}_\mathbf{y})
$$

which prevents the exact-from-Gaussian closed-form posterior used in the
linear case. Instead, the posterior must be sampled via MCMC.

### Target Distribution

$$
\hat{\pi}(z, i) \propto
  \underbrace{g(z, \mathbf{y})}_{\text{likelihood}}
  \;\cdot\;
  \underbrace{f(\mathbf{m}_i, z)}_{\text{transition}}
  \;\cdot\;
  \underbrace{p(i)}_{\text{uniform}}
$$

where:

- **Likelihood:** $\log g = -\tfrac{1}{2}\|\boldsymbol{\Sigma}_\mathbf{y}^{-1/2}(\mathbf{y} - \arctan(\mathbf{H}z))\|^2$
- **Transition:** $\log f = -\tfrac{1}{2}\|{\boldsymbol{\Sigma}\_\mathbf{x}}^{-1/2}(z - \mathbf{m}\_i)\|^2$
- **Index prior:** $p(i) = 1/N_f$ (uniform)

### MCMC Sampling

At each MCMC iteration, a **Gibbs step** resamples the ensemble index $i$
(selecting which forecast member $\mathbf{m}_i$ to centre the transition
kernel on), followed by a **proposal step** for the state $z$ using one
of four available kernels.

### MCMC Kernels

Four MCMC kernels are implemented, selectable via the YAML parameter
`mcmc_kernel`:

#### 1. Gibbs-within-MH (`gibbs_mh`)

Standard random-walk Metropolis–Hastings with isotropic Gaussian proposal
scaled by the prior standard deviation:

$$
z' = z + \epsilon \cdot \boldsymbol{\sigma}_\mathbf{x} \odot \boldsymbol{\xi},
\qquad \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Acceptance ratio:

$$
\alpha = \frac{g(z', \mathbf{y})\,f(\mathbf{m}_i, z')}
              {g(z, \mathbf{y})\,f(\mathbf{m}_i, z)}
$$

Optimal acceptance rate $\sim 23.4\%$ in high dimensions. Scales as
$\mathcal{O}(d^{-1/2})$ — acceptance degrades severely with state
dimension $d$.

#### 2. Preconditioned Crank–Nicolson (`pcn`)

A **prior-preserving** proposal that is invariant under the transition
kernel:

$$
z'_{nz} = \mathbf{m}_{i,nz} +
         \sqrt{1 - \beta^2}\,(z_{nz} - \mathbf{m}_{i,nz}) +
         \beta\,\boldsymbol{\sigma}_{\mathbf{x},nz} \odot \boldsymbol{\xi}
$$

where $\beta \in (0,1)$ is the step-size parameter (`pcn_beta`).

The key property of pCN is that the **prior terms cancel** in the
acceptance ratio, leaving only the likelihood ratio:

$$
\alpha = \frac{g(z', \mathbf{y})}{g(z, \mathbf{y})}
$$

This makes pCN **dimension-robust**: the acceptance rate is independent
of the state dimension $d$, unlike RW-MH where it decays as
$d^{-1/2}$. Optimal acceptance rate $\sim 40\%$.

#### 3. Metropolis-Adjusted Langevin Algorithm (`mala`)

Gradient-informed proposal using the score function of the target:

$$
z' = z + \frac{\tau^2}{2}\boldsymbol{\sigma}_\mathbf{x}^2 \odot
     \nabla_z \log\pi(z, i) +
   \tau\boldsymbol{\sigma}_\mathbf{x} \odot \boldsymbol{\xi}
$$

The proposal is **asymmetric** $(q(z'|z) \neq q(z|z'))$, requiring a
Hastings correction with the reverse proposal density. Scales as
$\mathcal{O}(d^{-1/3})$.

#### 4. Hamiltonian Monte Carlo (`hmc`)

Simulates Hamiltonian dynamics with momentum $\mathbf{p}$ and mass
matrix $\mathbf{M} = \text{diag}(1/\boldsymbol{\sigma}_\mathbf{x}^2)$:

$$
\dot{q} = \mathbf{M}^{-1}\mathbf{p}
  = \boldsymbol{\sigma}_\mathbf{x}^2 \odot \mathbf{p},
\qquad
\dot{p} = \nabla_z \log\pi(z, i)
$$

Integrated with the leapfrog scheme for $L$ steps of size $\epsilon$
(`hmc_leapfrog_steps`). Acceptance via the Hamiltonian:
$\alpha = \exp(-(H_{\text{new}} - H_{\text{old}}))$. Scales as
$\mathcal{O}(d^{-1/4})$, the best theoretical scaling.

#### Gradient of the Log-Target

All gradient-based kernels (MALA, HMC) use:

$$
\nabla_z \log\pi
  = \mathbf{H}^T\!\left[
    \frac{1}{1 + (\mathbf{H}z)^2} \odot
    \frac{\mathbf{y} - \arctan(\mathbf{H}z)}{\boldsymbol{\sigma}_\mathbf{y}^2}
    \right] -
  \frac{z - \mathbf{m}_i}{\boldsymbol{\sigma}_\mathbf{x}^2}
$$

### Adaptive Step-Size Tuning

During burn-in, the step-size parameter is adjusted every
`mcmc_adapt_interval` iterations to match a target acceptance rate.
For pCN, the parameter $\beta$ is scaled by $0.8/1.2$
(clipped to $[10^{-4},\, 0.999]$). For the other kernels,
`step_size` is scaled similarly (clipped to $[10^{-6},\, 10]$).

### Nonlinear Localization Variants

The same two localization strategies (V1 block-partition, V2 per-block
halo) are available for the nonlinear case:

| | V1 NL (`lsmcmc_nl_V1.py`) | V2 NL (`lsmcmc_nl_V2.py`) |
|--|--|--|
| Local state dim | ~21,000 (all observed-block cells) | ~600–800 (per-block halo) |
| Observations used | All (~528 per cycle) | Nearby only (~2–50 per block) |
| MCMC chains | 8 parallel independent chains | 1 chain per block, blocks in parallel |
| Parallelism | `multiprocessing.Pool` over chains | `mp.Pool` over blocks |

### MCMC Convergence Diagnostics

The diagnostic script `mcmc_diagnostics.py` runs one DA cycle and
records full MCMC chains (including burn-in) for representative blocks.
It computes Effective Sample Size (ESS), split $\hat{R}$,
autocorrelation, and generates traceplots.

**Results with `gibbs_mh` kernel** (RW-MH) — catastrophic non-convergence:

| Block | Dim | ESS (out of 5000) | $\hat{R}$ |
|-------|:---:|:-----------------:|:---------:|
| V2 blocks | 600–784 | 3–11 | 1.2–2.2 |
| V1 global | 21,056 | 3–4 | ~2.0 |

Root cause: random-walk MH in 600–21,000 dimensions with acceptance
rate decaying as $d^{-1/2}$.

**Results with `pcn` kernel** — converged:

| Block | nObs | Dim | AccRate | ESS | $\hat{R}$ |
|-------|:----:|:---:|:-------:|:---:|:---------:|
| V2 blk14 | 2 | 784 | 0.424 | 1070–1158 | 1.000–1.001 |
| V2 blk74 | 9 | 608 | 0.195 | 205–304 | 1.006–1.008 |
| V2 blk20 | 12 | 784 | 0.196 | 159–171 | 1.008–1.023 |
| V2 blk38 | 24 | 784 | 0.295 | 44–99 | 1.021–1.055 |
| V2 blk39 | 50 | 684 | 0.187 | 126–202 | 1.010–1.016 |
| V1 global | 528 | 21,056 | 0.305 | 4–7 | 2.0–2.8 |

pCN achieves **100$\times$ higher ESS** than gibbs_mh for V2 blocks,
with $\hat{R} < 1.06$ (well-converged). However, V1's 21,000-dimensional
joint space remains too large even for pCN.

### Comparison: V1 vs V2 — Nonlinear Twin Experiment (240 Cycles)

| | V1 NL (gibbs_mh, 8 chains) | V2 NL (gibbs_mh) | V2 NL (pCN) |
|--|:---:|:---:|:---:|
| $N_f$ | 25 | 25 | 25 |
| mcmc_N / burn_in | 2000 / 1000 | 2000 / 1000 | 5000 / 3000 |
| Mean vel RMSE (m/s) | **0.058** | 0.168 | 0.149 |
| Mean SST RMSE (K) | **0.385** | 1.117 | 0.991 |
| Mean SSH RMSE (m) | **0.280** | 0.971 | 0.897 |
| Wall time (min) | 33.0 | ~30 | 28.9 |
| MCMC ESS per block | 3–4 | 3–11 | 44–1158 |

#### Key Observations (Nonlinear Case)

1. **V1 NL outperforms V2 NL** by $\sim 3\times$ in velocity RMSE,
   despite both having poor MCMC convergence with `gibbs_mh`. V1
   benefits from the `gaussian_block_means` averaging step where MCMC
   samples are grouped and averaged into $N_f$ analysis members — this
   averaging provides regularization that masks the poor mixing.

2. **pCN improves V2 convergence dramatically** (ESS from 3–11 to
   44–1158) but the RMSE improvement is modest (0.168 → 0.149 m/s).
   This suggests the bottleneck is not just MCMC mixing but the
   per-block independence assumption: each block's posterior is solved
   with only 2–50 local observations, limiting how much the state can
   be constrained.

3. **V1 NL with `gibbs_mh` works despite poor ESS** because the 8
   parallel chains provide some diversity, and the block-mean averaging
   acts as a variance-reducing estimator. With 528 observations
   informing 21,000 states simultaneously, even poor samples carry more
   global information than well-mixed per-block samples with 2–50
   observations each.

4. **pCN is essential for V2 NL** — the `gibbs_mh` kernel produces
   essentially uncorrelated samples (ESS $\approx 3$), making the
   posterior mean estimate unreliable. pCN's dimension-independent
   acceptance rate is critical for the ~700-dimensional per-block
   problems.

5. **V1 NL cannot use pCN effectively** because its 21,000-dimensional
   state space overwhelms even pCN (ESS = 4–7). The block-partition
   localization lumps too many cells into a single MCMC problem.

---

## Data Assimilation — LETKF

### Overview

**Local Ensemble Transform Kalman Filter** (Hunt et al., 2007) with
Gaspari–Cohn localisation and RTPS (Relaxation To Prior Spread) inflation.
Surface observations update all 12 fields at each grid point through
ensemble covariances.

### Algorithm

For each assimilation cycle $n$:

**Step 1 — Ensemble Forecast.** Advance $K$ members by `t_freq`
timesteps. MPI-parallelised: members distributed across ranks in chunks.

**Step 2 — Stability Clamp.** Per layer: $h_k \geq 5$ m,
$|u_k|,|v_k| \leq 20$ m s⁻¹, $T_k \in [250,\,320]$ K.

**Step 3 — LETKF Update (per grid point $n$).** For grid point $n$
with $p$ locally-weighted observations:

Ensemble perturbation matrix in observation space:

$$
\mathbf{Y}^b = \mathbf{H}\,\mathbf{X}' \in \mathbb{R}^{p \times K}
$$

Inverse analysis error covariance in ensemble space:

$$
\widetilde{\mathbf{P}}_a^{-1}
  = (K-1)\,\mathbf{I} +
    (\mathbf{Y}^b)^T\,\mathbf{R}_{\text{loc}}^{-1}\,\mathbf{Y}^b
$$

where $\mathbf{R}_{\text{loc}} = \text{diag}\!\bigl(\sigma_y^2 / w_{\text{loc}}\bigr)$
incorporates Gaspari–Cohn localization weights $w_{\text{loc}}$.

Eigendecomposition:
$\widetilde{\mathbf{P}}_a^{-1} = \mathbf{E}\boldsymbol{\Lambda}\mathbf{E}^T$,
so $\widetilde{\mathbf{P}}_a = \mathbf{E}\boldsymbol{\Lambda}^{-1}\mathbf{E}^T$.

Mean analysis weight:

$$
\bar{\mathbf{w}}_a
  = \widetilde{\mathbf{P}}_a\,(\mathbf{Y}^b)^T\,\mathbf{R}_{\text{loc}}^{-1}
    \bigl(\mathbf{y} - \mathbf{H}\bar{\mathbf{x}}\bigr)
$$

Perturbation transform:

$$
\mathbf{W}_a = \sqrt{K-1}\;\widetilde{\mathbf{P}}_a^{1/2}
$$

Analysis ensemble:

$$
x_a^{(k)}[n]
  = \bar{x}[n] + \sum_{j=1}^{K} x'^{(j)}[n]\;\bigl(\bar{\mathbf{w}}_a + \mathbf{W}_a\bigr)_{j,k}
$$

The weight matrix $\mathbf{W}$ computed from surface observations is
applied to **all 12 fields** at grid point $n$, allowing information to
propagate from the surface to deep layers through ensemble covariances.

**Step 4 — RTPS Inflation.**

$$
\alpha_{f,n}
  = 1 + \rho\;\frac{\sigma_f^b - \sigma_f^a}
                    {\max(\sigma_f^a,\; 10^{-10})}
$$

$$
{x'}^a_{k,f,n} \;\leftarrow\; \alpha_{f,n}\;{x'}^a_{k,f,n}
$$

where $\rho$ = `covinflate1` and $\sigma^b$, $\sigma^a$ are background
and analysis ensemble spreads.

**Step 5 — Post-Analysis SSH Relaxation.** For each ensemble member $k$,
the SSH anomaly is corrected by distributing it equally across all 3 layers:

$$
h_{k,\ell} \;\leftarrow\;
  h_{k,\ell} - \frac{\alpha_{\text{relax}}}{N_{\text{layers}}}
  \bigl(\eta_k - \eta_{\text{ref}}(t)\bigr),
  \qquad \ell = 0,1,2
$$

### Gaspari–Cohn Localization Function

Compactly supported 5th-order piecewise polynomial with cutoff $L$:

$$
\text{GC}(r) = \begin{cases}
  1 - \frac{5}{3}z^2 + \frac{5}{8}z^3 + \frac{1}{2}z^4 - \frac{1}{4}z^5
    & 0 \leq r/L \leq 0.5 \\
  4 - 5z^2 + \frac{5}{3}z^3 + \frac{5}{8}z^4 - \frac{1}{2}z^5 - \frac{2}{3z}
    & 0.5 < r/L < 1 \\
  0 & r/L \geq 1
\end{cases}
$$

where $z = 2r/L$ and $r$ is the great-circle (haversine) distance
between the grid point and the observation.

### MPI Parallelization

- **Forecast phase:** ensemble members distributed across ranks in chunks
  of $\lceil K/P \rceil$. Gathered via `MPI_Allgather`.
- **Analysis phase:** grid points divided into contiguous blocks across
  ranks. Each rank performs the local LETKF update for its block.
  Results collected on the master via point-to-point `send/recv`, then
  `MPI_Bcast` to all ranks.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nanals` | 25 | Ensemble size $K$ |
| `hcovlocal_scale` | 200 km | GC localization cutoff |
| `covinflate1` | 0.95 | RTPS inflation coefficient $\rho$ |

---

## SSH Control Strategy

Without intervention, the 3-layer model's SSH grows unboundedly because
velocity perturbations propagate through the barotropic continuity
equation $(\partial h/\partial t = -\nabla\cdot(h\mathbf{u}))$ into
thickness anomalies. Three mechanisms control this:

### 1. Forecast-Time SSH Relaxation

Added directly to the continuity equation RHS in `model.py`:

$$
\frac{\partial h_{\text{total}}}{\partial t}
  \mathrel{+}= -\lambda_{\text{ssh}}\bigl(\eta - \eta_{\text{ref}}(t)\bigr)
$$

Default $\lambda_{\text{ssh}} = 9.3 \times 10^{-5}$ s⁻¹ ($\tau \approx 3$ h).

### 2. Analysis-Time SSH Correction

After each DA analysis step, the total column thickness is pulled toward
the reference:

$$
h_{\text{total}}^{\text{corrected}}
  = h_{\text{total}}^a -
    \alpha_{\text{relax}}\bigl(\eta^a - \eta_{\text{ref}}(t)\bigr)
$$

Default $\alpha_{\text{relax}} = 0.10$ in production (removes 10 % of
the SSH anomaly each cycle); $0.0$ in the default config.

### 3. Surface Heat Flux

Newtonian cooling prevents SST (and hence SSH through steric effects)
from drifting:

$$
\frac{\partial T_0}{\partial t}
  \mathrel{+}= -\frac{\alpha}{\rho_{sw}\,c_p\,h_{\text{mix}}}
  (T_0 - T_{\text{air}})
$$

### Process Noise

The thickness noise is set to `sig_x_ssh = 0.2` (real data) /
`0.5` (twin experiments).  Together with velocity perturbations acting through
the continuity equation, this maintains adequate ensemble spread in SSH.

---

## Sensitivity Analysis

The `mlswe_letkf_sensitivity.py` script performs a grid search over
LETKF tuning parameters — localisation scale and RTPS inflation — for
different ensemble sizes. The `--nanals` CLI override allows testing
the effect of ensemble size independently from the YAML config.

Two complete sensitivity studies were conducted on a reduced domain
($80 \times 70$ grid, $\Delta x \approx 50$ km, $\Delta y \approx 56$ km,
100 assimilation cycles, 52 MPI ranks):

### Experiment 1: $K = 25$ Ensemble Members

| Parameter | Search values |
|-----------|---------------|
| `hcovlocal_scale` (km) | 20, 40, 60, 80, 100, 200, 300 |
| `covinflate1` (RTPS $\rho$) | 0.1, 0.3, 0.5, 0.7, 0.8, 0.9 |

#### Results — Mean RMSE (velocity, m/s)

| h (km) | α=0.1 | α=0.3 | α=0.5 | α=0.7 | α=0.8 | α=0.9 |
|-------:|------:|------:|------:|------:|------:|------:|
| 20 | 0.0081 | **0.0074** | 0.0080 | **0.0074** | 0.0080 | 0.0078 |
| 40 | 0.0081 | 0.0074 | 0.0080 | 0.0074 | 0.0080 | 0.0078 |
| 60 | 0.0093 | 0.0088 | 0.0087 | 0.0087 | 0.0085 | 0.0087 |
| 80 | 0.0100 | 0.0112 | 0.0104 | 0.0095 | 0.0096 | 0.0095 |
| 100 | 0.0101 | 0.0094 | 0.0088 | 0.0089 | 0.0088 | 0.0088 |
| 200 | 0.0291 | 0.0264 | 0.0305 | 0.0224 | 0.0231 | 0.0210 |
| 300 | 0.1317 | 0.1595 | 0.1906 | 0.2464 | 0.2383 | 0.3122 |

#### Results — Mean RMSE (SST, K)

| h (km) | α=0.1 | α=0.3 | α=0.5 | α=0.7 | α=0.8 | α=0.9 |
|-------:|------:|------:|------:|------:|------:|------:|
| 20 | 0.257 | **0.226** | 0.256 | 0.241 | 0.245 | 0.238 |
| 40 | 0.257 | 0.226 | 0.256 | 0.241 | 0.245 | 0.238 |
| 60 | 0.303 | 0.284 | 0.285 | 0.277 | 0.270 | 0.270 |
| 80 | 0.355 | 0.391 | 0.370 | 0.356 | 0.336 | 0.347 |
| 100 | 0.452 | 0.445 | 0.394 | 0.392 | 0.371 | 0.366 |
| 200 | 3.046 | 2.625 | 2.525 | 2.185 | 2.267 | 2.005 |
| 300 | 10.289 | 11.418 | 10.757 | 11.164 | 10.918 | 10.734 |

#### Results — Mean RMSE (SSH, m)

| h (km) | α=0.1 | α=0.3 | α=0.5 | α=0.7 | α=0.8 | α=0.9 |
|-------:|------:|------:|------:|------:|------:|------:|
| 20 | 0.879 | 0.853 | 0.805 | 0.804 | 0.799 | **0.795** |
| 40 | 0.879 | 0.853 | 0.805 | 0.804 | 0.799 | 0.795 |
| 60 | 0.961 | 0.952 | 0.929 | 0.888 | 0.895 | 0.871 |
| 80 | 1.200 | 1.192 | 1.160 | 1.152 | 1.094 | 1.088 |
| 100 | 1.417 | 1.344 | 1.284 | 1.198 | 1.194 | 1.172 |
| 200 | 3.433 | 3.014 | 2.819 | 2.637 | 2.524 | 2.278 |
| 300 | 10.277 | 9.749 | 9.111 | 8.867 | 8.551 | 8.517 |

#### Best Parameters ($K = 25$)

| Metric | Best $h$ (km) | Best $\alpha$ | RMSE |
|--------|:-------------:|:-------------:|-----:|
| Velocity | 20 | 0.7 | 0.00737 m/s |
| SST | 20 | 0.3 | 0.226 K |
| SSH | 20 | 0.9 | 0.795 m |
| **Combined** | **20** | **0.7** | vel=0.0074, sst=0.241, ssh=0.804 |

#### Discussion: Sub-Grid Localisation

With $K = 25$ the optimal localisation scale is 20–40 km, which is
**smaller than the grid spacing** ($\Delta x \approx 50$ km,
$\Delta y \approx 56$ km). This means the filter is performing
essentially **point-wise updates** — observations influence only the grid
cell in which they lie.

This behaviour is expected for small-ensemble, high-dimensional DA:
- The state dimension is $12 \times 80 \times 70 = 67{,}200$ but the
  sample covariance from $K = 25$ members has rank 24.
- Estimated cross-correlations between distant grid points are dominated
  by **spurious sampling noise**, so aggressive localisation is needed to
  prevent filter divergence.
- The identical results at $h = 20$ and $h = 40$ km confirm that both
  scales are already sub-grid, hitting the floor imposed by the
  observation and grid density.

### Experiment 2: $K = 50$ Ensemble Members

To test whether a larger ensemble supports physically meaningful
(super-grid) localisation scales, the experiment was repeated with
$K = 50$ and localisation scales restricted to $h \geq \Delta x$:

| Parameter | Search values |
|-----------|---------------|
| `hcovlocal_scale` (km) | 60, 80, 100, 120, 150, 200, 300 |
| `covinflate1` (RTPS $\rho$) | 0.1, 0.3, 0.5, 0.7, 0.8, 0.9 |

#### Results — Mean RMSE (velocity, m/s)

| h (km) | α=0.1 | α=0.3 | α=0.5 | α=0.7 | α=0.8 | α=0.9 |
|-------:|------:|------:|------:|------:|------:|------:|
| 60 | 0.0095 | 0.0095 | 0.0094 | 0.0095 | **0.0092** | 0.0093 |
| 80 | 0.0092 | 0.0088 | 0.0086 | 0.0086 | 0.0087 | 0.0088 |
| 100 | **0.0083** | 0.0087 | 0.0085 | 0.0088 | 0.0085 | 0.0087 |
| 120 | 0.0104 | 0.0104 | 0.0102 | 0.0093 | 0.0090 | 0.0089 |
| 150 | 0.0099 | 0.0095 | 0.0094 | 0.0097 | 0.0095 | 0.0096 |
| 200 | 0.0132 | 0.0117 | 0.0117 | 0.0114 | 0.0111 | 0.0108 |
| 300 | 0.0283 | 0.0222 | 0.0189 | 0.0174 | 0.0179 | 0.0138 |

#### Results — Mean RMSE (SST, K)

| h (km) | α=0.1 | α=0.3 | α=0.5 | α=0.7 | α=0.8 | α=0.9 |
|-------:|------:|------:|------:|------:|------:|------:|
| 60 | 0.225 | 0.222 | 0.216 | 0.212 | 0.212 | **0.211** |
| 80 | 0.248 | 0.229 | 0.226 | 0.228 | 0.235 | 0.227 |
| 100 | 0.260 | 0.252 | 0.236 | 0.242 | 0.240 | 0.241 |
| 120 | 0.313 | 0.290 | 0.294 | 0.273 | 0.253 | 0.263 |
| 150 | 0.412 | 0.403 | 0.369 | 0.375 | 0.373 | 0.351 |
| 200 | 0.792 | 0.656 | 0.618 | 0.570 | 0.593 | 0.619 |
| 300 | 3.064 | 2.765 | 2.300 | 1.969 | 1.792 | 1.540 |

#### Results — Mean RMSE (SSH, m)

| h (km) | α=0.1 | α=0.3 | α=0.5 | α=0.7 | α=0.8 | α=0.9 |
|-------:|------:|------:|------:|------:|------:|------:|
| 60 | 0.927 | 0.922 | 0.910 | 0.907 | 0.903 | **0.890** |
| 80 | 1.188 | 1.189 | 1.143 | 1.133 | 1.135 | 1.129 |
| 100 | 1.193 | 1.157 | 1.138 | 1.128 | 1.104 | 1.074 |
| 120 | 1.314 | 1.262 | 1.232 | 1.193 | 1.204 | 1.151 |
| 150 | 1.729 | 1.641 | 1.519 | 1.492 | 1.439 | 1.430 |
| 200 | 2.170 | 1.979 | 1.840 | 1.722 | 1.667 | 1.658 |
| 300 | 4.260 | 3.643 | 3.231 | 2.975 | 2.749 | 2.682 |

#### Best Parameters ($K = 50$)

| Metric | Best $h$ (km) | Best $\alpha$ | RMSE |
|--------|:-------------:|:-------------:|-----:|
| Velocity | 100 | 0.1 | 0.00832 m/s |
| SST | 60 | 0.9 | 0.211 K |
| SSH | 60 | 0.9 | 0.890 m |
| **Combined** | **100** | **0.8** | vel=0.0085, sst=0.240, ssh=1.104 |

### Comparison: $K = 25$ vs $K = 50$

| | $K = 25$ (best) | $K = 50$ at $h=60$ | $K = 50$ at $h=100$ |
|--|:---:|:---:|:---:|
| Optimal $h$ | 20 km (sub-grid) | 60 km (≈ $\Delta x$) | 100 km (2$\Delta x$) |
| $\text{RMSE}_{\text{vel}}$ | **0.0074** | 0.0092 | 0.0083 |
| $\text{RMSE}_{\text{SST}}$ | 0.226 | **0.211** | 0.240 |
| $\text{RMSE}_{\text{SSH}}$ | **0.795** | 0.890 | 1.074 |

#### Key Findings

1. **Localisation scale shifts upward with ensemble size.** With
   $K = 25$ the optimum is sub-grid ($h = 20$ km); with $K = 50$ the
   optimum moves to $h = 60$–$100$ km, which is physically meaningful
   (1–2$\Delta x$, comparable to the first baroclinic Rossby radius in
   the subtropical North Atlantic, $\sim 30$–$60$ km).

2. **$K = 50$ improves SST** at the super-grid scales: $\text{RMSE}_{\text{SST}}$
   drops to 0.211 K at $h = 60$ km (vs 0.270 K for $K = 25$ at $h = 60$ km,
   a 22% improvement). The larger ensemble enables beneficial cross-cell
   covariances without introducing spurious correlations.

3. **$K = 25$ retains the best velocity and SSH** overall because
   sub-grid localisation acts as a very strong regulariser, effectively
   converting the LETKF into a pointwise optimal interpolation scheme.
   The $K = 50$ filter at $h = 100$ km achieves competitive velocity
   RMSE (0.0083 vs 0.0074) at a physically consistent scale.

4. **RMSE degrades sharply above $h = 150$ km** for both ensemble sizes.
   At $h = 300$ km the filter diverges for $K = 25$ ($\text{RMSE}_{\text{SST}}$
   $> 10$ K) and severely degrades for $K = 50$ ($\text{RMSE}_{\text{SST}}$
   $\approx 1.5$–$3$ K). This is well above the Rossby deformation
   radius, confirming that localisation must respect the physics of
   mesoscale decorrelation.

5. **RTPS inflation sensitivity** is moderate at small scales ($h \leq 100$ km)
   and becomes important at large scales where high $\alpha$ (0.7–0.9)
   partially compensates for over-smoothing by maintaining ensemble spread.

6. **Recommended operational parameters:**
   - For speed: $K = 25$, $h = 20$–$40$ km, $\alpha = 0.3$–$0.7$
   - For physical consistency: $K = 50$, $h = 60$–$100$ km, $\alpha = 0.7$–$0.9$

### Running

```bash
# K=25 sensitivity (fast, ~3 hours with 52 cores)
python mlswe_letkf_sensitivity.py --nprocs 52 --nanals 25 \
    --config example_input_mlswe_test100.yml \
    --results mlswe_letkf_sensitivity_results.json

# K=50 sensitivity (~6 hours with 52 cores)
python mlswe_letkf_sensitivity.py --nprocs 52 --nanals 50 \
    --config example_input_mlswe_test100.yml \
    --results mlswe_letkf_sensitivity_results_K50.json
```

### Plotting

```bash
python plot_mlswe_letkf_sensitivity.py mlswe_letkf_sensitivity_results.json
python plot_mlswe_letkf_sensitivity.py mlswe_letkf_sensitivity_results_K50.json
```

---

## Implementation Notes: Bathymetry and Island Cells

### The Shallow-Cell Problem

The North Atlantic domain $[-60°,\, -20°] \times [10°,\, 45°]$ contains
several mid-ocean islands and seamounts (e.g.\ the Azores near 39° N, 31° W).
The ETOPO bathymetry file enforces a minimum depth of $H_{\min} = 100$ m,
which means island-adjacent cells can have $H_b = 100$ m surrounded by
open-ocean cells at $H_b \geq 1600$ m. This creates **sharp bathymetry
cliffs** spanning $\Delta H \sim 1500$ m across a single grid spacing
($\Delta x \approx 50$ km).

These shallow cells are problematic for ensemble methods because:

1. **CFL violations.** The barotropic phase speed
   $c = \sqrt{gH}$ is much lower in shallow cells ($c \approx 31$ m/s
   at 100 m) than neighbors ($c \approx 125$ m/s at 1600 m). Large
   pressure gradients across the cliff drive velocities that can exceed
   the CFL limit during the RK4 advance.
2. **Process noise amplification.** With $\sigma_h = 0.15$ m of process
   noise applied every cycle to $h_{\text{total}}$, the SSH anomaly at a
   100 m cell is proportionally much larger than at a 4000 m cell. Over
   50 ensemble members, some will develop unphysical SSH anomalies.
3. **NaN cascade.** Once a single cell goes NaN during the RK4 forecast,
   the stencil propagates it to all four neighbors, creating a growing
   cross-shaped NaN pattern.

### Diagnosis

In the initial LETKF implementation, cell $(y{=}58,\, x{=}57)$ at
approximately $39.4°$ N, $31.1°$ W (near the Azores) exhibited
intermittent NaN at 38 out of 240 cycles. All 50 ensemble members
blew up simultaneously because the model's internal NaN recovery
(which replaces NaN $h \to H_{\text{rest}}$, $u,v \to 0$) fires
inside `advance()` but cannot prevent the next forecast from
re-exploding at the same unstable cell.

The LSMCMC method does **not** suffer from this issue because its
per-subdomain resampling effectively rejects particles that diverge,
keeping only physically reasonable states.

### Fix (Bug #4)

Two safeguards were added to `run_mlswe_letkf_mpi.py`:

1. **Bathymetry-aware stability clamp.** Instead of a flat
   $h_{\text{total}} \geq 5$ m, the post-forecast clamp now enforces

   $$h_{\text{total}} \geq \max\!\bigl(5,\; H_b - 20\bigr)$$

   This limits the SSH anomaly to $\eta \geq -20$ m, preventing the
   ensemble from producing absurdly negative SSH at shallow cells
   (e.g.\ $h_{\text{total}} = 19$ m at $H_b = 100$ m $\Rightarrow$
   $\eta = -81$ m, which was the root cause).

2. **Post-analysis NaN recovery.** After the LETKF update, RTPS
   inflation, and SSH relaxation, any ensemble member containing
   non-finite values is replaced element-wise with the `nanmean` of
   the finite members (or the background mean if all members are NaN
   at that cell). The bathymetry-aware clamp is then re-applied.

### Recommendations

When applying this code to **any ocean domain**, inspect the bathymetry
for cells where $H_b$ is near $H_{\min}$ and surrounded by much deeper
water. Such cells typically arise from:

- **Islands** (Azores, Canaries, Cape Verde, etc.)
- **Seamounts** and mid-ocean ridges
- **Continental shelf breaks** (abrupt depth transitions)

Practitioners should:

- Verify the minimum bathymetry (`H_min` in `load_bathymetry()`) is
  physically reasonable for the grid resolution.
- Monitor the NaN recovery log messages during cycling; frequent
  triggering (>10% of cycles) suggests the bathymetry gradient is too
  steep for the current grid spacing and time step.
- Consider increasing `H_min` or smoothing the bathymetry near
  problematic cells if NaN recovery fires too often.

---

## Experiments

This project implements four categories of data assimilation experiments,
progressing from simple linear synthetic tests to nonlinear assimilation
of real oceanographic data. Each experiment uses LSMCMC with both V1
(block-partition) and V2 (halo + Gaspari–Cohn) localization.

### 1. Linear Gaussian Synthetic Data

**Directory:** `linear_gaussian/`

A simplified test case on a $120 \times 120$ periodic grid
(state dimension $d = 14{,}400$) with:
- Synthetic "SWOT-like" swath observations (periodic diagonal passes)
- Linear observation operator: $\mathbf{y} = \mathbf{H}\mathbf{z} + \boldsymbol{\varepsilon}$
- Known Gaussian prior and likelihood → exact posterior available for
  verification

This experiment validates the LSMCMC machinery (V1 and V2) and compares
against LETKF. It runs quickly (minutes) and is useful for debugging.

**Config:** `linear_gaussian/input_linear_letkf.yml`

**Runner:**
```bash
cd linear_gaussian
python3 -u linear_forward_run_lsmcmc_v1.py input_linear_letkf.yml
```

**Sample results:**

| | |
|:--:|:--:|
| ![LG Observation Swaths](figures/lg_obs_swaths.png) | ![LG RMSE Timeseries](figures/lg_rmse_timeseries.png) |
| Synthetic swath observation pattern | RMSE timeseries showing filter convergence |
| ![LG Snapshot](figures/lg_snapshot.png) | |
| Analysis snapshot at a single cycle | |

### 2a. MLSWE — Linear Data Model with Real Data

**Configs:** `example_input_mlswe_ldata_V1.yml`, `example_input_mlswe_ldata_V2.yml`

The full 3-layer MLSWE model ($70 \times 80$ grid, $d = 67{,}200$) is
run with a **linear observation operator**:

$$
\mathbf{y} = \mathbf{H}\mathbf{z} + \boldsymbol{\varepsilon}
$$

Observations are real oceanographic data:
- **Drifter velocities** from the Global Drifter Program (GDP)
- **SWOT SSH** from the Surface Water and Ocean Topography satellite,
  binned onto the model grid via `prebin_swot_ssh.py` and gap-filled
  with synthetic observations via `generate_synthetic_swot.py`

RMSE is computed at observation locations against HYCOM reanalysis.

**Runners:**
```bash
# V1 (block-partition)
nohup python3 -u run_mlswe_lsmcmc_ldata_V1.py example_input_mlswe_ldata_V1.yml \
    > log_ldata_v1.txt 2>&1 &

# V2 (halo + Gaspari–Cohn)
nohup python3 -u run_mlswe_lsmcmc_ldata_V2.py example_input_mlswe_ldata_V2.yml \
    > log_ldata_v2.txt 2>&1 &
```

**Sample results:**

| | |
|:--:|:--:|
| ![Linear V1 RMSE](figures/linear_v1_rmse.png) | ![Linear V2 RMSE](figures/linear_v2_rmse.png) |
| V1 RMSE timeseries vs HYCOM | V2 RMSE timeseries vs HYCOM |
| ![Linear V1 SSH Comparison](figures/linear_v1_compare_ssh.png) | |
| SSH field: Forecast vs Analysis vs HYCOM | |

### 2b-i. MLSWE — Nonlinear Data Model with Real Data

**Configs:** `example_input_mlswe_nlrealdata_V1.yml`, `example_input_mlswe_nlrealdata_V2.yml`

The same 3-layer MLSWE model but with a **nonlinear observation
operator**:

$$
\mathbf{y} = \arctan(\mathbf{H}\mathbf{z}) + \boldsymbol{\varepsilon}
$$

The $\arctan$ transformation makes the posterior non-Gaussian, so
exact-from-Gaussian sampling is replaced by MCMC:
- **V1** uses a Gibbs-MH kernel (`gibbs_mh`)
- **V2** uses a preconditioned Crank–Nicolson (pCN) kernel that is
  robust in high dimensions — the prior term cancels in the
  Metropolis–Hastings acceptance ratio

Observations are the same real drifter + SWOT data. RMSE is computed
at observation locations against HYCOM reanalysis.

**Runners:**
```bash
# V1 (block-partition, gibbs_mh kernel)
nohup python3 -u run_mlswe_lsmcmc_nlrealdata_V1.py \
    example_input_mlswe_nlrealdata_V1.yml > log_nlreal_v1.txt 2>&1 &

# V2 (halo + GC, pCN kernel)
nohup python3 -u run_mlswe_lsmcmc_nlrealdata_V2.py \
    example_input_mlswe_nlrealdata_V2.yml > log_nlreal_v2.txt 2>&1 &
```

**Sample results:**

| | |
|:--:|:--:|
| ![NL Real V1 RMSE](figures/nl_real_v1_rmse.png) | ![NL Real V2 RMSE](figures/nl_real_v2_rmse.png) |
| V1 RMSE timeseries vs HYCOM | V2 RMSE timeseries vs HYCOM |
| ![NL Real V2 SSH](figures/nl_real_v2_ssh_ts.png) | ![NL Real V2 Velocity](figures/nl_real_v2_vel_ts.png) |
| V2 SSH timeseries at selected grid points | V2 velocity/SST timeseries vs HYCOM |

### 2b-ii. MLSWE — Nonlinear Data Model with Synthetic Twin Data

**Configs:** `example_input_mlswe_nldata_V1_twin.yml`, `example_input_mlswe_nldata_V2_twin.yml`

A controlled experiment where:
1. A "truth" run of the MLSWE model generates the ground truth state.
2. Synthetic observations are created by applying $\arctan(\mathbf{H}\mathbf{z}_{\text{true}}) + \boldsymbol{\varepsilon}$.
3. The LSMCMC filter assimilates these observations starting from a
   perturbed initial condition.

Since the true state is known everywhere, RMSE is computed over
**all grid cells** (not just observation locations). This provides
a clean test of the filter's ability to recover the full state from
nonlinear observations.

**Runners:**
```bash
# V1 (block-partition)
nohup python3 -u run_mlswe_lsmcmc_nldata_V1_twin.py \
    example_input_mlswe_nldata_V1_twin.yml > log_twin_v1.txt 2>&1 &

# V2 (halo + GC, pCN kernel)
nohup python3 -u run_mlswe_lsmcmc_nldata_V2_twin.py \
    example_input_mlswe_nldata_V2_twin.yml > log_twin_v2.txt 2>&1 &
```

**Sample results:**

| | |
|:--:|:--:|
| ![Twin V1 RMSE](figures/nl_twin_v1_rmse.png) | ![Twin V2 RMSE](figures/nl_twin_v2_rmse.png) |
| V1 RMSE timeseries vs truth | V2 RMSE timeseries vs truth |
| ![Twin V1 SSH](figures/nl_twin_v1_ssh_ts.png) | ![Twin V2 SSH](figures/nl_twin_v2_ssh_ts.png) |
| V1 SSH timeseries: truth vs analysis | V2 SSH timeseries: truth vs analysis |
| ![Twin V1 Velocity](figures/nl_twin_v1_vel_ts.png) | |
| V1 velocity timeseries: truth vs analysis | |

### 2c. MLSWE — Cauchy (Non-Gaussian) Noise Twin Experiments

**Directory:** `nlgamma_ldata/`

**Configs:** `nlgamma_ldata/input_nlgamma_twin.yml` (V1), `nlgamma_ldata/input_nlgamma_twin_v2.yml` (V2)

A twin experiment on the full 3-layer MLSWE grid with:
- **Linear observation operator** $\mathbf{y} = \mathbf{H}\mathbf{z} + \boldsymbol{\varepsilon}$
- **Cauchy (heavy-tailed) observation noise** instead of Gaussian, making
  the likelihood non-log-concave and requiring MCMC sampling even though
  the observation operator is linear
- Multiple shape parameter values ($m = 1, 2, 3, 4$) are tested via
  dedicated configs (`input_nlgamma_twin_v2_m1.yml`, …, `input_nlgamma_twin_v2_m4.yml`)
- LETKF baseline: `nlgamma_ldata/run_nlgamma_twin_letkf.py`

**Runners:**
```bash
cd nlgamma_ldata

# V1 (block-partition, pCN kernel)
nohup python3 -u run_nlgamma_twin.py input_nlgamma_twin.yml \
    > log_nlgamma_v1.txt 2>&1 &

# V2 (halo + GC, pCN kernel)
nohup python3 -u run_nlgamma_twin_v2.py input_nlgamma_twin_v2.yml \
    > log_nlgamma_v2.txt 2>&1 &

# LETKF baseline
nohup python3 -u run_nlgamma_twin_letkf.py input_nlgamma_twin_letkf.yml \
    > log_nlgamma_letkf.txt 2>&1 &
```

### Experiment Summary Table

| Experiment | Obs Operator | Noise | Data | Sampling | Config prefix | Runner prefix |
|:-----------|:-------------|:------|:-----|:---------|:-------------|:-------------|
| Linear Gaussian | $\mathbf{H}z$ | Gaussian | Synthetic swaths | Exact Gaussian | `input_linear_*` | `linear_gaussian/run_*` |
| MLSWE Linear | $\mathbf{H}z$ | Gaussian | Real drifter+SWOT | Exact Gaussian | `example_input_mlswe_ldata_*` | `run_mlswe_lsmcmc_ldata_*` |
| MLSWE NL Real | $\arctan(\mathbf{H}z)$ | Gaussian | Real drifter+SWOT | MCMC (pCN) | `example_input_mlswe_nlrealdata_*` | `run_mlswe_lsmcmc_nlrealdata_*` |
| MLSWE NL Twin | $\arctan(\mathbf{H}z)$ | Gaussian | Synthetic from truth | MCMC (pCN) | `example_input_mlswe_nldata_*_twin` | `run_mlswe_lsmcmc_nldata_*_twin` |
| MLSWE Cauchy Twin | $\mathbf{H}z$ | Cauchy | Synthetic from truth | MCMC (pCN) | `input_nlgamma_twin*` | `nlgamma_ldata/run_nlgamma_twin*` |

---

## Data Preparation

All MLSWE experiments (linear, nonlinear, Cauchy) require external
oceanographic data.  A single master script downloads everything:

```bash
python3 download_all_data.py
```

This runs six steps:

| Step | What | Source | Output |
|------|------|--------|--------|
| 1 | HYCOM boundary conditions | HYCOM GOFS 3.1 OPeNDAP | `data/hycom_bc_2024aug.nc` |
| 2 | ETOPO bathymetry | NOAA ETOPO OPeNDAP | `data/etopo_bathy_*.npy` |
| 3 | OSMC drifter positions | OSMC ERDDAP | `data/osmc_drifters_*.csv` |
| 4 | SWOT L2 SSH | NASA PO.DAAC via `earthaccess` | `data/swot_2024aug_new/*.nc` |
| 5 | SSH relaxation reference | Computed from HYCOM BC | `data/hycom_ssh_ref_*.npy` |
| 6 | SST nudging reference | Computed from HYCOM BC | `data/hycom_sst_ref_*.npy` |

After downloading, process observations:

```bash
python3 generate_drifter_obs.py          # drifter CSV → obs NetCDF
python3 prebin_swot_ssh.py               # SWOT L2 → model-grid SSH
python3 generate_synthetic_swot.py       # gap-fill missing SWOT cycles
```

The **Linear Gaussian** experiment (under `linear_gaussian/`) is
fully self-contained — run `linear_gaussian/linear_forward_generate_data.py`
to create its own synthetic data; no external downloads are needed.

See [`LSMCMC.ipynb`](LSMCMC.ipynb) for
detailed per-experiment data instructions.

---

## Project Structure

```
MLSWE_LSMCMC/
├── mlswe/                              # Model package
│   ├── __init__.py
│   ├── model.py                        # 3-layer MLSWE solver (RK4, A-grid)
│   ├── boundary_handler.py             # Multi-layer HYCOM BC handler
│   ├── lsmcmc_V1.py                    # LSMCMC filter — linear V1 (block-partition)
│   ├── lsmcmc_V2.py                    # LSMCMC filter — linear V2 (halo + GC)
│   ├── lsmcmc_nl_V1.py                 # LSMCMC filter — nonlinear V1 (block-partition, MCMC)
│   ├── lsmcmc_nl_V2.py                 # LSMCMC filter — nonlinear V2 (halo + GC, MCMC)
│   └── letkf.py                        # LETKF local update utilities
├── linear_gaussian/                    # Linear Gaussian synthetic experiment
│   ├── run_full_comparison.py          # Run V1, V2, and LETKF together
│   ├── linear_forward_generate_data.py # Generate synthetic observations
│   ├── linear_forward_run_letkf_mpi.py # LETKF runner (multiprocessing)
│   ├── linear_forward_run_lsmcmc_v1.py # LSMCMC V1 runner
│   ├── linear_forward_run_lsmcmc_v2.py # LSMCMC V2 runner
│   ├── linear_forward_run_kf.py        # Exact Kalman filter
│   ├── linear_forward_run_letkf_sensitivity.py
│   ├── generate_swath_observations.py  # SWOT-like dual-swath obs generator
│   └── input_linear_*.yml              # YAML configs for LG experiment
├── nlgamma_ldata/                      # Cauchy (non-Gaussian) noise twin experiment
│   ├── run_nlgamma_twin.py             # V1 runner
│   ├── run_nlgamma_twin_v2.py          # V2 runner
│   ├── run_nlgamma_twin_letkf.py       # LETKF baseline runner
│   └── input_nlgamma_twin*.yml         # YAML configs (m=1,2,3,4 variants)
├── nongauss_ldata/                     # Arctan observation operator twin experiment
│   ├── run_nongauss_twin.py            # V1 runner
│   ├── run_nongauss_twin_v2.py         # V2 runner
│   ├── run_nongauss_twin_letkf.py      # LETKF baseline runner
│   └── input_nongauss*.yml             # YAML configs
├── run_mlswe_lsmcmc_ldata_V1.py        # Linear LSMCMC V1 runner
├── run_mlswe_lsmcmc_ldata_V2.py        # Linear LSMCMC V2 runner
├── run_mlswe_lsmcmc_nldata_V1.py       # NL LSMCMC V1 runner (synthetic obs)
├── run_mlswe_lsmcmc_nldata_V2.py       # NL LSMCMC V2 runner (synthetic obs)
├── run_mlswe_lsmcmc_nldata_V1_twin.py  # NL LSMCMC V1 twin experiment runner
├── run_mlswe_lsmcmc_nldata_V2_twin.py  # NL LSMCMC V2 twin experiment runner
├── run_mlswe_lsmcmc_nlrealdata_V1.py   # NL LSMCMC V1 real-data runner
├── run_mlswe_lsmcmc_nlrealdata_V2.py   # NL LSMCMC V2 real-data runner
├── run_mlswe_ldata_letkf_mpi.py        # LETKF runner (multiprocessing, linear obs)
├── run_mlswe_letkf_nl_twin.py          # LETKF runner (NL twin experiment)
├── run_mlswe_letkf_nldata.py           # LETKF runner (NL real-data)
├── mcmc_diagnostics.py                 # MCMC convergence diagnostics (ESS, R-hat, traceplots)
├── prebin_swot_ssh.py                  # SWOT L2 → model-grid SSH binning (ADT computation)
├── generate_synthetic_swot.py          # Gap-fill missing SWOT cycles with synthetic obs
├── generate_drifter_obs.py             # Drifter observation processing
├── download_all_data.py                # Master data download script (6 steps)
├── download_hycom_bc.py                # HYCOM GOFS 3.1 download via OPeNDAP
├── drifter_data.py                     # GDP / OSMC drifter loading utilities
├── swot_ssh_data.py                    # SWOT L2 SSH download via earthaccess
├── mlswe_letkf_sensitivity.py          # LETKF parameter sensitivity sweep
├── plot_mlswe_results.py               # Figures for a single DA method
├── plot_lsmcmc_vs_letkf.py             # LSMCMC vs LETKF comparison figures
├── plot_mlswe_letkf_sensitivity.py     # Sensitivity heatmap plots
├── plot_nl_twin_results.py             # Twin experiment comparison plots
├── generate_nlgamma_figures.py         # Cauchy noise experiment figures
├── generate_localization_illustration.py  # V1 vs V2 localization diagram
├── generate_paper_figures.py           # All paper figures
├── generate_all_figures.py            # Runs all 3 figure scripts in one command
├── LSMCMC.ipynb                        # Jupyter notebook: running guide + figure reproduction
├── run_v2_after_v1.sh                  # Helper: run V2 after V1 completes
├── run_nlgamma_sequential.sh           # Helper: run Cauchy experiments sequentially
├── example_input_mlswe_*.yml           # YAML configs for MLSWE experiments
├── paper/                              # Manuscript (JAMES format)
│   ├── LSMCMC_filter.tex               # Main LaTeX source (standalone)
│   ├── agujournal2019 template/        # JAMES-formatted version
│   │   ├── LSMCMC_filter_JAMES.tex
│   │   ├── references.bib
│   │   └── figures/
│   └── figures/                        # Shared figure PDFs
├── data/                               # Input data
│   ├── etopo_bathy_*.npy               # Bathymetry
│   ├── gdp_hourly_*.csv                # Drifter observations
│   ├── hycom_bc.nc                     # HYCOM boundary conditions
│   ├── hycom_sst_ref_*_3d.npy          # SST nudging reference (time × ny × nx)
│   ├── hycom_sst_ref_*_times.npy       # SST reference time axis
│   ├── hycom_ssh_ref_*.npy             # SSH relaxation reference
│   ├── swot_ssh_binned_80x70.nc        # Real SWOT SSH (binned to model grid)
│   ├── swot_ssh_combined_80x70.nc      # Real + synthetic SWOT SSH
│   └── ncep_t2m_*.npy                  # NCEP 2-m air temperature
├── output_lsmcmc_ldata_V1/             # Linear LSMCMC V1 output (240 cycles)
├── output_lsmcmc_ldata_V2/             # Linear LSMCMC V2 output (240 cycles)
├── output_lsmcmc_nldata_twin_V1/       # NL LSMCMC V1 twin output
├── output_lsmcmc_nldata_twin_V2/       # NL LSMCMC V2 twin output
├── output_lsmcmc_nldata_real_V1/       # NL LSMCMC V1 real-data output
├── output_lsmcmc_nldata_real_V2/       # NL LSMCMC V2 real-data output
├── output_nlgamma_twin_V1*/            # Cauchy V1 twin outputs
├── output_nlgamma_twin_V2*/            # Cauchy V2 twin outputs (m=1,2,3,4)
├── output_letkf/                       # LETKF output NetCDF
├── comparison_plots/                   # Generated comparison figures
├── requirements.txt                    # Python dependencies (pip)
├── environment.yml                     # Conda environment (also used by Binder)
├── LICENSE                             # MIT License
├── CITATION.cff                        # Machine-readable citation metadata
└── README.md
```

---

## Configuration Reference

All parameters are set in the YAML config file (default
`example_input_mlswe_ldata_V1.yml`).

### Time & Grid

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 75 | Time step (s) |
| `t_freq` / `assim_timesteps` | 48 | Steps between assimilation (48 × 75 s = 1 h) |
| `T` | 11520 | Total time steps (240 × 48) |
| `nassim` | 240 | Number of assimilation cycles |
| `dgx` | 80 | Grid points in x |
| `dgy` | 70 | Grid points in y |
| `lon_min`, `lon_max` | −60, −20 | Longitude range (°E) |
| `lat_min`, `lat_max` | 10, 45 | Latitude range (°N) |

### Layer Properties

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho` | [1023, 1026, 1028] | Layer densities (kg m⁻³) |
| `H_rest` | [100, 400, 3500] | Rest-state thicknesses (m) |
| `T_rest` | [298.15, 283.15, 275.15] | Rest-state temperatures (K) |
| `alpha_h` | [0.6, 0.3, 0.1] | SSH → thickness BC distribution |
| `beta_vel` | [1.0, 1.0, 1.0] | Velocity depth factors |
| `noise_decay` | [1.0, 0.01, 0.01] | DA noise decay per layer |

### Physics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g` | 9.81 | Gravity (m s⁻²) |
| `bottom_drag` | 1e-6 | Linear drag (s⁻¹) |
| `diff_coeff` | 500 | Momentum diffusion (m² s⁻¹) |
| `diff_order` | 1 | 1 = Laplacian, 2 = bi-harmonic |
| `tracer_diff` | 100 | Temperature diffusion (m² s⁻¹) |
| `sst_nudging_rate` | 2.78e-4 | SST nudging $\lambda$ (s⁻¹), $\tau \approx 1$ h |
| `ssh_relax_rate` | 9.3e-5 | SSH relaxation $\lambda$ (s⁻¹), $\tau \approx 3$ h |
| `ssh_analysis_relax_frac` | 0.0 | Post-DA SSH correction fraction (disabled; 0.10 in older runs) |
| `sst_flux_type` | `'newtonian'` | Surface heat flux type |
| `sst_alpha` | 100 | Bulk transfer coefficient (W m⁻² K⁻¹) |
| `sst_h_mix` | 0.1 | Mixed-layer depth for flux (m) |

### Process Noise

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sig_x_uv` | 0.15 | Velocity ($u, v$) noise std (m s⁻¹) |
| `sig_x_ssh` | 0.2 | Thickness noise std (m) (real data); 0.5 (twin experiments) |
| `sig_x_sst` | 1.0 | SST noise std (K) |

### Observation Noise

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sig_y_uv` | 0.10 | Drifter velocity noise (m s⁻¹) |
| `sig_y_sst` | 0.4 | SST observation noise (K) |
| `use_swot_ssh` | True | Enable SWOT SSH observations |
| `sig_y_ssh` | 0.25 | SWOT SSH noise (m) (real data); 0.50 (twin experiments) |
| `ssh_obs_fraction` | 0.15 | Fraction of cells observed per cycle |

### LSMCMC (V1 — Block-Partition)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nforecast` | 25 | Ensemble size |
| `mcmc_N` | 500 | MCMC samples per subdomain |
| `burn_in` | 500 | MCMC burn-in iterations |
| `num_subdomains` | 50 | Number of localization blocks |
| `ncores` | 50 | Multiprocessing workers |

### LSMCMC (V2 — Halo + GC Tapering)

All V1 parameters apply, plus:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_loc` | 2.5 | Halo localization radius (grid cells); 1.5 for NL experiments |
| `rtps_alpha` | 0.5 | RTPS relaxation-to-prior-spread coefficient |
| `reset_interval` | 100 | Periodic ensemble reset interval (0 = disabled) |
| `n_block_workers` | 25 | Workers for parallel block processing (varies by server) |

### Nonlinear LSMCMC (MCMC Parameters)

These parameters apply to both V1 NL and V2 NL:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mcmc_kernel` | `'pcn'` | MCMC kernel: `pcn`, `gibbs_mh`, `mala`, `hmc` |
| `mcmc_step_size` | 0.5 (V1) / 0.05 (V2) | Step size for gibbs_mh / mala / hmc |
| `mcmc_N` | 2000 | Post-burn-in MCMC samples |
| `burn_in` | 500 | MCMC burn-in iterations |
| `mcmc_adapt` | true | Enable adaptive step-size tuning |
| `mcmc_adapt_interval` | 50 | Iterations between adaptations |
| `mcmc_target_acc` | 0.35 | Target acceptance rate for pCN (0.234 for gibbs_mh) |
| `mcmc_thin` | 1 | Thinning factor |
| `pcn_beta` | 0.3 | pCN step-size parameter $\beta \in (0,1)$ |
| `hmc_leapfrog_steps` | 10 | HMC leapfrog integration steps |
| `mcmc_chains` | min(ncores, 8) | Parallel MCMC chains (V1 only) |

### LETKF

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nanals` | 50 | Ensemble size (production); 25 (default) |
| `hcovlocal_scale` | 60 | GC localization half-width (km) |
| `covinflate1` | 0.90 | RTPS inflation coefficient (production); 0.95 (default) |
| `covinflate2` | −1 | Negative → multiplicative RTPS |

---

## Usage Examples

All experiments follow the same pattern: a **runner script** reads a
**YAML config file** and executes the data assimilation loop. For long
runs, use `nohup` with unbuffered output (`python3 -u`) so that progress
is logged in real time.

### Quick Start: Linear LSMCMC (real data)

```bash
# V1 — Block-partition localization
python3 -u run_mlswe_lsmcmc_ldata_V1.py example_input_mlswe_ldata_V1.yml

# V2 — Halo + Gaspari–Cohn localization
python3 -u run_mlswe_lsmcmc_ldata_V2.py example_input_mlswe_ldata_V2.yml
```

### Nonlinear LSMCMC — Real Data

```bash
# V1 (block-partition, gibbs_mh kernel)
nohup python3 -u run_mlswe_lsmcmc_nlrealdata_V1.py \
    example_input_mlswe_nlrealdata_V1.yml > log_nlreal_v1.txt 2>&1 &

# V2 (halo + GC, pCN kernel — recommended)
nohup python3 -u run_mlswe_lsmcmc_nlrealdata_V2.py \
    example_input_mlswe_nlrealdata_V2.yml > log_nlreal_v2.txt 2>&1 &
```

### Nonlinear LSMCMC — Twin Experiments

```bash
# V1 twin (block-partition)
nohup python3 -u run_mlswe_lsmcmc_nldata_V1_twin.py \
    example_input_mlswe_nldata_V1_twin.yml > log_twin_v1.txt 2>&1 &

# V2 twin (halo + GC, pCN kernel)
nohup python3 -u run_mlswe_lsmcmc_nldata_V2_twin.py \
    example_input_mlswe_nldata_V2_twin.yml > log_twin_v2.txt 2>&1 &
```

### Cauchy (Non-Gaussian) Noise Twin Experiments

```bash
cd nlgamma_ldata

# V1
nohup python3 -u run_nlgamma_twin.py input_nlgamma_twin.yml \
    > log_nlgamma_v1.txt 2>&1 &

# V2 (default shape m=1; use input_nlgamma_twin_v2_m2.yml etc. for other values)
nohup python3 -u run_nlgamma_twin_v2.py input_nlgamma_twin_v2.yml \
    > log_nlgamma_v2.txt 2>&1 &

# LETKF baseline
nohup python3 -u run_nlgamma_twin_letkf.py input_nlgamma_twin_letkf.yml \
    > log_nlgamma_letkf.txt 2>&1 &
```

### Running V2 After V1 (Sequential)

If V2 should start after V1 finishes (e.g., to reuse resources), use the
helper script:

```bash
nohup bash run_v2_after_v1.sh > log_sequential.txt 2>&1 &
```

### MCMC Convergence Diagnostics

After a nonlinear run completes, check convergence with:

```bash
python3 -u mcmc_diagnostics.py example_input_mlswe_nldata_V2_twin.yml
```

This produces traceplots, effective sample size (ESS), and split-$\hat{R}$
diagnostics for the MCMC chains.

### SWOT SSH Pre-Processing

Before running experiments with real SSH data, the SWOT observations must
be pre-processed:

```bash
# Step 1: Bin raw SWOT L2 files onto model grid (computes ADT)
python3 prebin_swot_ssh.py

# Step 2: Fill gaps with synthetic SWOT observations
python3 generate_synthetic_swot.py
```

### LETKF (multiprocessing-parallel)

```bash
python3 -u run_mlswe_ldata_letkf_mpi.py example_input_mlswe_letkf_best.yml
```

> **Note.** Despite the legacy `_mpi` suffix in the filename, LETKF now
> uses Python `multiprocessing` (not MPI).  Parallelism is configured via
> `ncores` in the YAML file; the script creates two `mp.Pool`s — one for
> ensemble forecasts and one for the local DA updates.

CLI overrides for parameter tuning:

```bash
python3 -u run_mlswe_ldata_letkf_mpi.py example_input_mlswe_letkf_best.yml \
    --hcovlocal_scale 100 --covinflate1 0.5 --nanals 50
```

### LETKF Sensitivity Analysis

```bash
python3 mlswe_letkf_sensitivity.py --nprocs 50 --nanals 25 \
    --config example_input_mlswe_test100.yml
python3 plot_mlswe_letkf_sensitivity.py
```

---

## Generating Plots

Diagnostic plots are generated by dedicated plotting scripts. Each
experiment type has its own plotter.

### Real-Data Experiments (Linear or Nonlinear)

```bash
# Usage: python3 plot_mlswe_results.py <output_dir> <config_file> <label>

# Linear V1
python3 plot_mlswe_results.py ./output_lsmcmc_ldata_V1 \
    example_input_mlswe_ldata_V1.yml "Linear V1"

# Linear V2
python3 plot_mlswe_results.py ./output_lsmcmc_ldata_V2 \
    example_input_mlswe_ldata_V2.yml "Linear V2"

# Nonlinear Real V1
python3 plot_mlswe_results.py ./output_lsmcmc_nldata_real_V1 \
    example_input_mlswe_nlrealdata_V1.yml "NL Real V1"

# Nonlinear Real V2
python3 plot_mlswe_results.py ./output_lsmcmc_nldata_real_V2 \
    example_input_mlswe_nlrealdata_V2.yml "NL Real V2"
```

For real-data experiments, the plotter loads HYCOM reanalysis as the
reference ("truth") and computes RMSE at observation locations only.
Velocity and SST timeseries show HYCOM reference curves.

### Twin Experiments

```bash
# Usage: python3 plot_nl_twin_results.py <output_dir> <file_prefix> <config_file> <label>

# NL Twin V1
python3 plot_nl_twin_results.py ./output_lsmcmc_nldata_twin_V1 \
    nldata_V1_twin example_input_mlswe_nldata_V1_twin.yml "NL Twin V1"

# NL Twin V2
python3 plot_nl_twin_results.py ./output_lsmcmc_nldata_twin_V2 \
    nldata_V2_twin example_input_mlswe_nldata_V2_twin.yml "NL Twin V2"
```

For twin experiments, the plotter loads the saved "truth" state and
computes RMSE over all grid cells. SSH timeseries show the true SSH
(dashed) vs the analysis SSH (solid).

### Comparison Plots

```bash
# LSMCMC V1 vs V2 vs LETKF side-by-side
python3 plot_lsmcmc_vs_letkf.py
```

### Localization Illustration

```bash
python3 generate_localization_illustration.py
```

Generates `figures/localization_v1_v2_illustration.png` showing how V1
and V2 partition the domain and assign observations.

### Cauchy (Non-Gaussian) Noise Figures

```bash
python3 generate_nlgamma_figures.py
```

### All Paper Figures (Single Command)

Generate every paper figure at once:

```bash
python3 generate_all_figures.py
```

This runs `generate_paper_figures.py`, `generate_nlgamma_figures.py`, and
`regen_timeseries_1x2.py` in sequence.  All output is saved to
`paper_figures/`.

Alternatively, the Jupyter notebook `LSMCMC.ipynb` reproduces
all 19 figures interactively and displays them inline:

```bash
jupyter notebook LSMCMC.ipynb
```

### Output Plot Files

Each run generates the following diagnostic plots in its output directory:

| File | Description |
|:-----|:------------|
| `rmse_ssh_timeseries.png` | SSH RMSE over time |
| `rmse_vel_timeseries.png` | Velocity RMSE over time |
| `rmse_sst_timeseries.png` | SST RMSE over time |
| `ssh_timeseries_*.png` | SSH at selected grid points |
| `vel_timeseries_*.png` | Velocity at selected grid points |
| `sst_timeseries_*.png` | SST at selected grid points |
| `ssh_snapshot_*.png` | SSH field snapshots |
| `vel_snapshot_*.png` | Velocity field snapshots |
| `obs_count_*.png` | Observation density per cycle |

---

## Dependencies

| Package    | Purpose |
|------------|---------|
| NumPy      | Array operations |
| SciPy      | Interpolation, spatial algorithms |
| netCDF4    | I/O for observations and output |
| PyYAML     | Configuration parsing |
| matplotlib | Plotting |

All parallelism uses Python's built-in `multiprocessing` module
(`mp.Pool` with shared-memory arrays); no external MPI library is
required.

Install the core dependencies with:

```bash
pip install numpy scipy netCDF4 pyyaml matplotlib
```

---

## References

- Cushman-Roisin, B. & Beckers, J.-M. (2011). *Introduction to
  Geophysical Fluid Dynamics*, Ch. 11. Academic Press.
- Hallberg, R. (1997). Stable split time stepping schemes for large-scale
  ocean modelling. *J. Comput. Phys.*, 135(1), 54–65.
- Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). Efficient data
  assimilation for spatiotemporal chaos: A local ensemble transform
  Kalman filter. *Physica D*, 230(1–2), 112–126.
- Gaspari, G. & Cohn, S. E. (1999). Construction of correlation functions
  in two and three dimensions. *Q. J. R. Meteorol. Soc.*, 125, 723–757.
- Whiteley, N., Kantas, N., & Sherlock, C. (2021). Local Sequential Monte
  Carlo Methods. *Stat. Sci.*, 36(2), 270–286.
- Cotter, S. L., Roberts, G. O., Stuart, A. M., & White, D. (2013). MCMC
  methods for functions: modifying old algorithms to make them faster.
  *Statist. Sci.*, 28(3), 424–446.
- Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In *Handbook of
  Markov Chain Monte Carlo* (Ch. 5). Chapman & Hall/CRC.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Ruzayqat2026LSMCMC,
  author  = {Ruzayqat, Hamza and Chipilski, Hristo G. and Knio, Omar},
  title   = {Two Localization Strategies for Sequential {MCMC} Data
             Assimilation with Applications to Nonlinear Non-{Gaussian}
             Geophysical Models},
  year    = {2026}
}
```

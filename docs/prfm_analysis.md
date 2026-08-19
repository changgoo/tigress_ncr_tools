# PRFM pressure, weight, and feedback-yield analysis

## Scope

`plot-suite-prfm` tests the pressure-regulated, feedback-modulated (PRFM)
description of the 32-model TIGRESS-NCR suite. It reduces raw vertical-profile
(`zprof`) outputs and history files into:

- the two-phase midplane thermal, turbulent, and magnetic stresses;
- the external-gravity and gas-self-gravity vertical weights;
- pressure-to-weight ratios;
- component and total feedback yields; and
- suite-level pressure-weight-SFR diagnostic figures.

The default analysis interval is $t=200$--600. The implementation is
`src/tigress_ncr_tools/plot_suite_prfm.py`.

## PRFM relations being tested

Vertical dynamical equilibrium follows from the horizontally averaged vertical
momentum equation,

$$
\frac{d}{dz}
\left(P_{\rm th}+P_{\rm turb}+\Pi_B\right)
= -\rho\frac{\partial\Phi}{\partial z}.
$$

Integrating from the midplane to a vertical boundary $z_b$ gives

$$
P_{\rm tot}(0)-P_{\rm tot}(z_b)
= \int_0^{z_b}\rho\frac{\partial\Phi}{\partial z}\,dz
\equiv \mathcal{W}.
$$

The present diagnostic compares the measured midplane pressure directly with
the integrated weight,

$$
P_{\rm tot,2p}\approx\mathcal{W},
\qquad
\frac{P_{\rm tot,2p}}{\mathcal{W}}\approx 1.
$$

It does not subtract the stress at the top of the domain, so this comparison
implicitly treats the boundary stress as negligible.

Feedback modulation is expressed using a component yield,

$$
P_c = \Upsilon_c\Sigma_{\rm SFR},
$$

where $c$ denotes thermal, turbulent, turbulent-field magnetic, mean-field
magnetic, or their sum. Combining pressure regulation and feedback modulation
gives the schematic PRFM prediction

$$
\Sigma_{\rm SFR}
\approx \frac{\mathcal{W}}{\Upsilon_{\rm tot}}.
$$

The code measures every term directly rather than inserting an analytic
approximation for the vertical weight.

## Inputs and alignment requirements

For each selected dump, the reducer reads:

| Source | Required quantities |
|---|---|
| `phase7`, `phase11`, `phase12`, `phase13` z-profiles | `z`, `A`, `Ek3`, `P`, `PB1-3`, `dPB1-3` |
| `whole` z-profile | `z`, `dWext`, `dWsg` |
| primary history file | `time`, `sfr10`, `sfr40` |
| `athinput*`, `<domain1>` | `x1min`, `x1max`, `x2min`, `x2max` |

All four phase profiles and the whole-gas profile must have the same stored
time and exactly the same $z$ grid. The weight grid must be uniformly
spaced. The horizontal box area is

$$
A_{\rm box}
=(x_{1,\max}-x_{1,\min})(x_{2,\max}-x_{2,\min}).
$$

The parser reads the raw text profiles directly and does not generate NetCDF
or other sidecar files.

## Two-phase gas and midplane operator

The two-phase gas is the sum

$$
Q_{\rm 2p}(z)
=Q_7(z)+Q_{11}(z)+Q_{12}(z)+Q_{13}(z)
$$

for every required phase-profile quantity $Q$. In particular,

$$
A_{\rm 2p}(z)=\sum_{p\in\{7,11,12,13\}} A_p(z).
$$

Let the half-width of the midplane slab be $h$, with $h=10\ {\rm pc}$ by
default. For a grid quantity $Q_j=Q(z_j)$, define

$$
\mathcal{M}_h[Q]
=\frac{1}{N_h}\sum_{j:\,|z_j|\le h}Q_j.
$$

The pressure reduction uses a ratio of slab-averaged extensive quantities,

$$
\frac{\mathcal{M}_h[Q_{\rm 2p}]}
     {\mathcal{M}_h[A_{\rm 2p}]},
$$

rather than averaging the conditional ratio $Q_{\rm 2p}(z)/A_{\rm 2p}(z)$
separately at each height. The reported two-phase midplane area fraction is

$$
f_{A,\rm 2p}
=\frac{\mathcal{M}_h[A_{\rm 2p}]}{A_{\rm box}}.
$$

## Midplane pressure components

The z-profile fields are horizontal extensive sums. After two-phase summation
and midplane reduction, the vertical turbulent stress is

$$
P_{\rm turb}^{\rm raw}
=\frac{2\mathcal{M}_h[E_{k,3,\rm 2p}]}
       {\mathcal{M}_h[A_{\rm 2p}]}.
$$

The factor of two converts the stored vertical kinetic-energy density
$E_{k,3}=\rho v_z^2/2$ to the Reynolds stress $\rho v_z^2$.

The thermal pressure is

$$
P_{\rm th}^{\rm raw}
=\frac{\mathcal{M}_h[P_{\rm 2p}]}
       {\mathcal{M}_h[A_{\rm 2p}]}.
$$

For a magnetic field, the vertical Maxwell support is the difference between
horizontal magnetic pressure and vertical magnetic tension,

$$
\Pi_B
=\frac{B_x^2+B_y^2-B_z^2}{8\pi}.
$$

The fluctuating-field contribution recorded by the z-profiles is therefore

$$
\Pi_{\delta B}^{\rm raw} =
\frac{
\mathcal{M}_h[
 dPB_{1,\rm 2p}+dPB_{2,\rm 2p}-dPB_{3,\rm 2p}
]}
{\mathcal{M}_h[A_{\rm 2p}]}.
$$

First form the total magnetic Maxwell stress,

$$
\Pi_{B,\rm all}^{\rm raw} =
\frac{
\mathcal{M}_h[
 PB_{1,\rm 2p}+PB_{2,\rm 2p}-PB_{3,\rm 2p}
]}
{\mathcal{M}_h[A_{\rm 2p}]},
$$

then define the ordered or mean-field residual as

$$
\Pi_{\overline B}^{\rm raw}
=\Pi_{B,\rm all}^{\rm raw}-\Pi_{\delta B}^{\rm raw}.
$$

Thus, “mean-field” in the output specifically means total magnetic stress
minus the stored fluctuating-field stress. The total pressure used throughout
this analysis is

$$
P_{\rm tot,2p}
=P_{\rm turb}+P_{\rm th}
 +\Pi_{\delta B}+\Pi_{\overline B}.
$$

Radiation pressure, cosmic-ray pressure, and hot-gas pressure are not included
in this definition of `pressure_total`.

## Vertical weight

Weight is constructed from the whole-gas `dWext` and `dWsg` profiles, not
from the two-phase profiles. This is intentional: the diagnostic compares
two-phase midplane support with the gravitational weight of the complete gas
column.

For $a\in\{\mathrm{ext},\mathrm{sg}\}$, let $dW_{a,j}$ be the signed,
horizontally summed force-density profile and let $\Delta z$ be the uniform
grid spacing. The implemented inward integral is

$$
W_{a,j}^{\rm raw} =
\begin{cases}
-\Delta z\displaystyle\sum_{\substack{k\le j\\z_k<0}}dW_{a,k},
& z_j<0,\\[8pt]
 \Delta z\displaystyle\sum_{\substack{k\ge j\\z_k\ge0}}dW_{a,k},
& z_j\ge0.
\end{cases}
$$

In words, the lower-half profile is integrated upward from the lower boundary,
and the upper-half profile is integrated downward from the upper boundary.
The midplane weight per full horizontal area is

$$
\mathcal{W}_a^{\rm raw}
=\frac{\mathcal{M}_h[W_a^{\rm raw}]}{A_{\rm box}}.
$$

The two stored components and their sum are

$$
\mathcal{W}_{\rm ext},\qquad
\mathcal{W}_{\rm sg},\qquad
\mathcal{W}
=\mathcal{W}_{\rm ext}+\mathcal{W}_{\rm sg}.
$$

The instantaneous pressure-balance diagnostic is

$$
R_{\rm PW}(t)
=\frac{P_{\rm tot,2p}(t)}{\mathcal{W}(t)}.
$$

## Pressure and yield units

Raw z-profile pressure-like quantities are converted to $P/k_B$, in
${\rm K\,cm^{-3}}$, using

$$
C_P
=\frac{\mu_{\rm H}m_{\rm H}(1\ {\rm km\,s^{-1}})^2}{k_B},
$$

with

$$
\mu_{\rm H}=1.4271,\qquad
m_{\rm H}=1.008\,m_u.
$$

Numerically,

$$
C_P\simeq173.014.
$$

Every pressure and area-normalized weight above is multiplied by $C_P$.
The corresponding CSV columns therefore store $P/k_B$ or
$\mathcal{W}/k_B$, not pressure in cgs energy-density units.

For a pressure component stored as $P_c/k_B$, the feedback yield is

$$
\Upsilon_c(t)
=C_Y
\frac{P_c(t)/k_B}{\Sigma_{\rm SFR,40}(t)},
$$

where

$$
C_Y
=\frac{k_B(1\ {\rm kpc})^2(1\ {\rm yr})}
       {M_\odot(1\ {\rm km\,s^{-1}})}
\simeq2.08626\times10^{-4}.
$$

With $P_c/k_B$ in ${\rm K\,cm^{-3}}$ and
$\Sigma_{\rm SFR}$ in
$M_\odot\,{\rm kpc^{-2}\,yr^{-1}}$, this gives $\Upsilon_c$ in
${\rm km\,s^{-1}}$. The total yield obeys

$$
\Upsilon_{\rm tot}
=\Upsilon_{\rm turb}+\Upsilon_{\rm th}
 +\Upsilon_{\delta B}+\Upsilon_{\overline B}
=C_Y\frac{P_{\rm tot,2p}/k_B}{\Sigma_{\rm SFR,40}}.
$$

A snapshot with nonfinite or nonpositive `sfr40` receives a missing
(`NaN`) yield rather than an infinite value.

## SFR matching, model colors, and temporal statistics

At every z-profile time $t_j$, `sfr10` and `sfr40` are linearly
interpolated from the primary history:

$$
\Sigma_{\rm SFR,\tau}(t_j)
=\mathrm{interp}
\left[t_j;\{t_{\rm hst},\Sigma_{\rm SFR,\tau}\}\right],
\qquad \tau\in\{10,40\}.
$$

The pressure-SFR relations and all yields use `sfr40`. Model color and ordering
use the continuously time-weighted history average

$$
\left\langle\Sigma_{\rm SFR,10}\right\rangle_{t_1-t_2}
=\frac{1}{t_2-t_1}
 \int_{t_1}^{t_2}\Sigma_{\rm SFR,10}(t)\,dt,
$$

evaluated by trapezoidal integration with interpolated values at the exact
time bounds. The default colormap is logarithmically normalized `plasma`,
trimmed to the base-map interval 0.06--0.90.

The per-model summary uses an unweighted mean over the selected z-profile
snapshots,

$$
\langle X\rangle_{\rm snap}
=\frac{1}{N}\sum_{j=1}^{N}X(t_j),
$$

plus the 16th, 50th, and 84th percentiles. In particular,

$$
\left\langle R_{\rm PW}\right\rangle_{\rm snap}
=\left\langle\frac{P_{\rm tot,2p}}{\mathcal{W}}\right\rangle_{\rm snap}
$$

is not generally equal to
$$
\frac{\langle P_{\rm tot,2p}\rangle}
     {\langle\mathcal{W}\rangle}.
$$

Likewise, yields are formed snapshot-by-snapshot
before their temporal statistics are calculated.

## Figures

`prfm_pressure_weight_relations.png` contains:

1. $\langle P_{\rm tot,2p}\rangle$ versus
   $\langle\mathcal{W}\rangle$, with the one-to-one line;
2. $\langle P_{\rm tot,2p}\rangle$ versus
   $\langle\Sigma_{\rm SFR,40}\rangle$; and
3. $\langle\mathcal{W}\rangle$ versus
   $\langle\Sigma_{\rm SFR,40}\rangle$.

`prfm_pressure_components_yields.png` has four pressure panels and four yield
panels for

$$
P_{\rm turb},\quad P_{\rm th},\quad
\Pi_{\delta B},\quad\Pi_{\overline B}.
$$

Each point is a model’s snapshot mean. Horizontal and vertical line segments
show the independent 16th--84th percentile intervals; they do not represent a
joint confidence region. Axes are logarithmic. A nonpositive model mean is
not drawn, and a percentile segment whose lower endpoint is nonpositive is
omitted. Instantaneous magnetic Maxwell stresses may legitimately be
negative because vertical tension can exceed horizontal magnetic pressure.

## Output files and columns

The default output folder is `SUITE/prfm_diagnostics/`.

### `prfm_time_series.csv`

There is one row per selected model and z-profile dump. Columns are:

- identifiers: `model`, `dump_id`, `time`;
- phase coverage: `area_two_phase_fraction`;
- histories: `sfr10`, `sfr40`;
- pressures: `pressure_turbulent`, `pressure_thermal`,
  `pressure_magnetic_turbulent`, `pressure_magnetic_mean`,
  `pressure_total`;
- weights: `weight_external`, `weight_self_gravity`, `weight_total`;
- balance: `pressure_weight_ratio`; and
- yields: `yield_turbulent`, `yield_thermal`,
  `yield_magnetic_turbulent`, `yield_magnetic_mean`, `yield_total`.

Pressure and weight columns are in ${\rm K\,cm^{-3}}$, equivalent to
pressure divided by $k_B$. Yield columns are in ${\rm km\,s^{-1}}$.

### `prfm_model_summary.csv`

This table contains `model`, `samples`, `time_min`, `time_max`, and
`mean_sfr10_color`. For every summarized physical field $X$, it contains

- `X_mean`;
- `X_p16`;
- `X_p50`; and
- `X_p84`.

Rows retain the descending `mean_sfr10_color` model order.

### Other products

`model_sfr_colors.csv` records the rank, model name, time-weighted mean
`sfr10`, plotted hexadecimal color, and averaging bounds. The two PNG files
are the diagnostic
figures described above.

## Running the analysis

The default full-suite command is

```bash
plot-suite-prfm /tigress/changgoo/anvil/TIGRESS-NCR-suite
```

Useful options include:

```bash
# Change the analysis interval.
plot-suite-prfm SUITE --start 250 --stop 550

# Read every fourth selected z-profile.
plot-suite-prfm SUITE --stride 4 --overwrite

# Change the midplane slab to |z| <= 20 pc.
plot-suite-prfm SUITE --midplane-half-width 20 --overwrite

# Write products somewhere else.
plot-suite-prfm SUITE --output-dir /path/to/prfm_output
```

If `prfm_time_series.csv` already exists, it is reused unless `--overwrite`
is given. Use `--overwrite` whenever the source profiles, time interval,
stride, midplane width, or reduction code changes. Reusing a cache does not
revalidate those settings.

## Interpretation and limitations

- Agreement near $P_{\rm tot,2p}=\mathcal{W}$ tests vertical dynamical
  equilibrium for the adopted support terms.
- A nearly constant $\Upsilon_c$ across SFR measures a linear feedback
  pressure response for component $c$.
- Two-phase pressure and whole-gas weight deliberately use different phase
  selections.
- `pressure_total` excludes radiation, cosmic-ray, and hot-phase pressure.
- The comparison uses midplane pressure rather than
  $P_{\rm mid}-P_{\rm boundary}$.
- Snapshot summary means are not cadence-weighted, although the output cadence
  is expected to be nearly uniform. Model-color `sfr10` is time-weighted.
- Percentile bars describe temporal variability of each axis independently.
- The analysis measures simulation terms directly; it does not fit a PRFM
  equilibrium model or infer causality from the plotted correlations.

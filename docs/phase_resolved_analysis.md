# Six-phase evolution and velocity diagnostics from z-profiles

`plot-suite-phases` measures the time evolution, vertical structure, velocity
dispersion, and magnetic characteristic speeds of six ISM phase groups using
only horizontally reduced `zprof` files. No full VTK snapshots are required.
The phase definitions and Athena profile indices follow Table 3 of
[Kim et al. (2023, ApJ 946, 3)](https://doi.org/10.3847/1538-4357/acbd3a)
([arXiv:2211.13293](https://arxiv.org/abs/2211.13293)).

## 1. Six displayed phase groups

| displayed group | component profiles | physical selection |
|---|---:|---|
| CNM+CMM | `phase7 + phase11` | cold neutral plus molecular/cold molecular gas |
| UNM | `phase12` | unstable neutral gas, \(500<T<6000\) K |
| WNM | `phase13` | warm neutral gas, \(6000<T<3.5\times10^4\) K |
| WIM | `phase9 + phase10` | photoionized and collisionally ionized warm gas |
| WHIM | `phase5` | \(3.5\times10^4<T<5\times10^5\) K |
| HIM | `phase6` | \(T>5\times10^5\) K |

The underlying network distinguishes CMM (`phase7`) from CNM (`phase11`) and
splits WIM into WPIM (`phase9`, 6000--15000 K) and WCIM (`phase10`,
15000--35000 K). They are combined only at reduction time. `phase8` is the
unstable ionized medium (UIM). It is not silently folded into one of the six
requested groups: the difference between the six-group sum and `whole.zprof`
is stored explicitly as the UIM closure residual. `phase14`, the aggregate
WHIM+HIM profile, is not used because its two components are available.

## 2. Mass and volume fractions

At height cell \(z_j\), `d` is the horizontally integrated mass density and
`A` is the horizontally integrated selected area. For phase \(p\),

\[
M_p=\sum_j d_{p,j}\,\Delta z,
\qquad
V_p=\sum_j A_{p,j}\,\Delta z.
\]

Whole-box fractions divide these by the corresponding `whole.zprof` sums.
The whole-volume sum is checked against \(L_xL_yL_z\) for every snapshot.
Selected six-phase sums are not renormalized; their difference from unity is
reported in `phase_closure_time_series.csv` as the UIM residual.

The common instantaneous gas scale height is

\[
H_{\rm gas}(t)=
\left[\frac{\sum_j d_{{\rm whole},j}z_j^2\Delta z}
{\sum_j d_{{\rm whole},j}\Delta z}\right]^{1/2}.
\]

Fractions labeled `hgas` use the slab \(|z|\leq H_{\rm gas}(t)\). Boundary
cells receive their exact geometric overlap length rather than being selected
by cell center. A common total-gas aperture makes the phase fractions directly
comparable at each time.

## 3. Scale heights

Each phase has separate mass- and volume-weighted RMS heights over the full
box:

\[
H_{M,p}^2=\frac{\sum_j d_{p,j}z_j^2\Delta z}
{\sum_j d_{p,j}\Delta z},
\qquad
H_{V,p}^2=\frac{\sum_j A_{p,j}z_j^2\Delta z}
{\sum_j A_{p,j}\Delta z}.
\]

These are structural diagnostics. They are distinct from the common
\(H_{\rm gas}\) used to define the inner aperture.

## 4. Velocity dispersions

For either the whole box or the overlap-weighted inner slab, component means
and dispersions are

\[
\bar v_i=\frac{\sum_j M_{i,j}w_j}{\sum_j d_jw_j},
\qquad
\sigma_i^2=\frac{2\sum_j E_{k,i,j}w_j}{\sum_j d_jw_j}-\bar v_i^2,
\]

where \(w_j\) is \(\Delta z\) or the slab overlap. The radial and vertical
components use `M1/Ek1` and `M3/Ek3`. The azimuthal component uses
`dM2/dEk2`, so the background shearing flow is removed. The reported 3D
value is

\[
\sigma_{\rm 3D}=\sqrt{\sigma_1^2+\sigma_2^2+\sigma_3^2}.
\]

Subtracting the phase mean makes this a dispersion rather than an RMS speed
that includes coherent phase flow. Component means and dispersions, as well
as the 3D result, remain in the time-series CSV.

## 5. Effective vertical support speed

The PRFM vertical dynamical-equilibrium diagnostic uses the total vertical
stress

\[
P_{{\rm tot},z}=P_{{\rm turb},z}+P_{\rm th}+\Pi_B,
\qquad
\Pi_B=PB_1+PB_2-PB_3.
\]

Here `P_turb,z = 2*Ek3`, `P_th = P`, and `Pi_B` is the total Maxwell support,
including both ordered and fluctuating fields. For phase \(p\) and either the
whole-box or inner-slab weights \(w_j\), the effective vertical support speed is

\[
\sigma_{{\rm eff},z,p}=
\left[
\frac{\sum_j(2E_{k,3,p,j}+P_{p,j}+PB_{1,p,j}+PB_{2,p,j}-PB_{3,p,j})w_j}
{\sum_jd_{p,j}w_j}
\right]^{1/2}.
\]

This is the pressure-to-column-mass quantity used in PRFM vertical dynamical
equilibrium. It is not the vertical turbulent dispersion: the kinetic term is
the raw vertical Reynolds stress, without subtracting coherent vertical flow,
and thermal and magnetic support are also included. Neutral and ionized values
are reconstructed by adding the component-phase stress and mass integrals;
the Whole value is measured directly from `whole.zprof`.

## 6. Mean and perturbed Alfvén speeds

The simulation defines `dB<i>` relative to the whole-horizontal mean field at
each height. The reducer therefore obtains
\(\overline B_i(z)=B_{i,{\rm whole}}(z)/A_{\rm whole}(z)\) from
`whole.zprof`. The mean-field energy assigned to phase \(p\) is the energy of
that same background field in the phase-selected volume; the perturbed energy
is the phase profile's stored `dPB<i>`:

\[
v_{A,{\rm mean},i}^2=
\frac{\sum_j A_{p,j}\overline B_i^2(z_j)w_j}{\sum_jd_{p,j}w_j},
\qquad
v_{A,{\rm pert},i}^2=
\frac{2\sum_jdPB_{i,p,j}w_j}{\sum_jd_{p,j}w_j}.
\]

Using `PB-dPB` for a phase would be incorrect: the perturbation is referenced
to the whole-horizontal mean, so the phase-restricted cross term need not
vanish and the difference can even be negative. The estimator above matches
the exact `dump_zprof.c` definition and is positive by construction. The 3D
values are component quadrature sums. These are phase magnetic-energy-to-mass
diagnostics; they are not an average of the local ratio \(B/\sqrt\rho\).
An empty phase/region is stored as `NaN`.

## 7. Running and outputs

```bash
plot-suite-phases /tigress/changgoo/anvil/TIGRESS-NCR-suite --workers 8

# Re-read every z-profile after changing definitions or source outputs.
plot-suite-phases /tigress/changgoo/anvil/TIGRESS-NCR-suite \
  --workers 8 --overwrite
```

The default time range is 0--600 Myr, while model summaries use 400--600 Myr.
Evolution plots retain the full time range, but their y-axis limits are based
only on finite values at 200--600 Myr. This keeps startup anomalies from
compressing the scientifically relevant evolution without hiding when those
early curves leave the displayed range.
`--stride` can produce a lower-cadence exploratory reduction without changing
the source data. The default directory `SUITE/phase_evolution_zprof/` contains:

- `phase_time_series.csv`: one row per model, time, and displayed or aggregate
  phase (`neutral`, `ionized`, and `whole` are included);
- `phase_closure_time_series.csv`: selected-six sums and UIM residuals;
- `phase_model_summary.csv`: mean, standard deviation, median, 16th/84th
  percentiles, and counts over the summary interval;
- `phase_correlation_model_summary.csv`: temporal summaries for the six
  displayed phases plus neutral, ionized, and true whole-gas rows used by the
  correlation matrices;
- `phase_two_phase_fraction_summary.csv`: the same temporal statistics for
  reduced neutral (CNM+CMM+UNM+WNM) and ionized (WIM+WHIM+HIM) fractions,
  without renormalizing away the UIM residual;
- `phase_parameter_correlations.csv`: model-by-model Spearman coefficients
  for the standard displayed summaries against the four direct inputs
  \(\Sigma_*,H_*,\Omega,q\), the derived \(\kappa,\rho_*\), and mean
  \(\Sigma_{\rm SFR,10}\) over 200--600 Myr;
- `model_sfr_colors.csv`: the exact model ranking and color mapping;
- `phase_fraction_model_summary.png`: temporal medians and 16th--84th
  percentile ranges for mass/volume fractions in the box and inner slab;
- `phase_structure_speed_model_summary.png`: the corresponding phase scale
  heights, 3D velocity dispersions, effective vertical support speeds, and
  mean/perturbed Alfvén speeds;
- `phase_mass_fraction_box_parameter_relations.png`,
  `phase_volume_fraction_box_parameter_relations.png`,
  `phase_mass_fraction_hgas_parameter_relations.png`, and
  `phase_volume_fraction_hgas_parameter_relations.png`: direct relations for
  every displayed phase against all four input and two derived parameters,
  plus mean \(\Sigma_{\rm SFR,10}\);
- `phase_neutral_fraction_parameter_relations.png` and
  `phase_ionized_fraction_parameter_relations.png`: separate direct-relation
  figures for the two reduced phase groups;
- `phase_mass_scale_height_pc_parameter_relations.png` and
  `phase_volume_scale_height_pc_parameter_relations.png`: full scale-height
  scatter corresponding to the two scale-height matrix panels;
- `phase_sigma_3d_box_parameter_relations.png`,
  `phase_sigma_3d_hgas_parameter_relations.png`,
  `phase_sigma_eff_z_box_parameter_relations.png`,
  `phase_sigma_eff_z_hgas_parameter_relations.png`, and the four corresponding
  `phase_alfven_{mean,perturbed}_3d_{box,hgas}_parameter_relations.png`
  products: full 32-model scatter with temporal 16th--84th percentile bars
  corresponding to every dynamics matrix panel;
- `phase_fractions_parameter_correlations.png`,
  `phase_scale_heights_parameter_correlations.png`, and
  `phase_dynamics_parameter_correlations.png`: annotated phase-by-parameter
  Spearman heatmaps with nine phase rows, seven predictors, and a high-contrast
  diverging scale;
- whole-box and within-\(H_{\rm gas}\) fraction-evolution figures;
- phase mass/volume scale-height evolution;
- whole-box and inner-slab 3D velocity-dispersion evolution;
- whole-box and inner-slab effective-vertical-support-speed evolution;
- whole-box and inner-slab mean/perturbed Alfvén-speed evolution.

Thus every fraction, scale-height, velocity-dispersion, effective-support, and
Alfvén diagnostic has a time-evolution view, a direct parameter-scatter view,
and a correlation matrix view. Related time series remain grouped into compact
evolution figures.
The ensemble-summary points are the 400--600 Myr temporal medians for each
model and phase; their bars are the corresponding temporal 16th--84th
percentiles. Black connecting symbols show the median across models only as a
visual guide. Fraction correlations are shown in both direct parameter-relation
panels and coefficient matrices. In the direct panels, point color encodes
mean SFR, bars show temporal 16th--84th percentiles, and each panel reports its
Spearman coefficient. The reduced neutral and ionized values are summed at
every time before temporal statistics are calculated. They can sum to less than unity
because UIM remains an explicit, unassigned residual.

All parameter-relation x limits are computed from the finite predictor values
alone, with a 5% margin in linear or logarithmic plotting space. Temporal
y-error bars therefore cannot expand shared logarithmic x axes into
unphysical low-parameter ranges.

The remaining heatmaps correlate the same per-model temporal medians and do
not treat the correlated direct/derived predictors as independent causal
experiments. In particular, \(\Sigma_*\)--\(\rho_*\) and
\(\Omega\)--\(\kappa\) trends should be interpreted together.

Neutral and ionized scale heights and speeds are reconstructed from the
component masses, volumes, first moments, and second moments at each time;
they are not averages of the six-phase diagnostics. The Whole row is measured
directly from `whole.zprof`, so it includes UIM. The matrices include mean
\(\Sigma_{\rm SFR,10}\) as the seventh x-axis predictor. The Whole volume RMS
height is fixed by the common box geometry, and Whole mass/volume fractions
are unity by definition. Their rank correlations are undefined and appear as
`--` rather than as spurious zero correlations.

The long-form CSV is the primary analysis product. It preserves the three
velocity components and both magnetic components even when the standard
figures show only their 3D combinations.

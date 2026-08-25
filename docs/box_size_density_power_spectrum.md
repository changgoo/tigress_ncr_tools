# Column-density power spectra for the box-size suite

The box-size suite contains three logical models, each split across an early
and late run:

| Logical model | Segments | Transverse size |
| --- | --- | --- |
| R8_8pc_NCR_Lxy1024 | early, late | 1024 pc |
| R8_8pc_NCR_Lxy2048 | early, late | 2048 pc |
| R8_8pc_NCR_Lxy4096 | early, late | 4096 pc |

The plot-box-size-density-spectrum command applies the shear-aware estimator
documented in [density_power_spectrum.md](density_power_spectrum.md) to their
face-on theta0 column-density maps. It joins projections by Athena output
number, allows a boundary duplicate to be replaced by the later segment, and
selects each requested snapshot by the time stored in the projection header.

The branches do not have their final box sizes over their full histories. The
4096-pc branch expands from 1024 to 4096 pc at about 83 Myr, and the 2048-pc
branch expands from 1024 to 2048 pc between 200 and 201 Myr. The common
fixed-domain comparison interval is therefore 201--600 Myr. The command checks
the first selected map against the box size encoded by the logical model name
and rejects a pre-expansion interval instead of silently assigning the wrong
physical Fourier scale.

## Run the analysis

The production invocation is:

    module load anaconda3/2024.10
    PYTHONPATH=src python -m tigress_ncr_tools.plot_box_size_density_spectrum \
      /projects/c/changgoo/nasa_athena/TIGRESS-NCR \
      --workers 3

An installed editable checkout can use the shorter entry point:

    plot-box-size-density-spectrum \
      /projects/c/changgoo/nasa_athena/TIGRESS-NCR \
      --workers 3

The defaults analyze every integer target time from 201 through 600 Myr, use
no window or padding, form 40 logarithmic annuli, and summarize 201--600 Myr.
Use --overwrite-2d after changing the map-level estimator. Use --overwrite to
rebuild the collective reduction and figures while retaining compatible
per-model 2D caches.

## Common physical-wavenumber grid

The three boxes do not share a fundamental mode. The collective radial edges
therefore span

\[
k_{\min}=\min_i\left(\frac{2\pi}{L_i}\right)
        =\frac{2\pi}{4096\ {\rm pc}}
\]

through the smallest directional Nyquist wavenumber shared by all models.
A smaller box has no modes in some low-\(k\) annuli; those bins have
mode_count zero and power NaN. This retains the large-scale information from
the larger boxes while making all overlapping bins directly comparable in
physical \(k\). The comparison figure uses physical \(k\) on the lower axis
and \(\lambda=2\pi/k\) on the upper axis. A single \(kL/(2\pi)\) axis would be
ambiguous because \(L\) differs among curves.

The integral scale uses every finite positive annulus available to a model.
It is therefore a box-size convergence diagnostic rather than a measurement
over an artificially truncated common wavelength interval. The spectral slope
and quadrupole summaries retain the fixed 64--256 pc band defined in
[density_power_spectrum.md](density_power_spectrum.md).

## Interpretation and possible alternative diagnostics

The angle-averaged spectra are visually well converged over their common
wavelength range; increasing the box primarily adds longer-wavelength modes.
The scalar diagnostics need additional care because they compress the spectrum
or its angular structure in ways that can retain explicit box-size dependence.

### Spectral slope and curvature

The implemented power-law support is identical for every model and snapshot:
the fit always uses the same 10 bin centers from 68.6 through 238.9 pc.
There are no missing fit bins. The slopes measured from the 201--600 Myr mean
spectra are

| Box size [pc] | \(\alpha\), 64--256 pc |
| ---: | ---: |
| 1024 | 2.332 |
| 2048 | 2.331 |
| 4096 | 2.458 |

Thus the 1024- and 2048-pc results agree closely. The larger-box offset is not
caused by inconsistent fitting support. Instead, the mean spectra have
measurable curvature:

| Wavelength band [pc] | 1024 pc | 2048 pc | 4096 pc |
| --- | ---: | ---: | ---: |
| 128--256 | 2.216 | 2.071 | 2.127 |
| 64--128 | 2.598 | 2.664 | 2.873 |
| 96--256 | 2.203 | 2.149 | 2.230 |

The steepening is strongest near 64 pc, which is only eight pixels at the
common 8-pc resolution. A resolution study is needed before interpreting that
part of the curve as an inertial-range slope. Possible alternatives are to
show a running local slope \(\alpha(k)=-d\ln P/d\ln k\), fit a curved or broken
spectrum, or define a resolution-motivated lower wavelength cutoff before
examining box-size convergence. The 96--256 pc result is recorded as a useful
sensitivity check, not as a replacement selected after seeing the answer.

A direct shape comparison is less derivative-sensitive than a fitted slope.
For a reference spectrum \(P_{\rm ref}\), define

\[
D_{\rm shape} =
\left\langle
  \left[
    \Delta\ln P-\left\langle\Delta\ln P\right\rangle_B
  \right]^2
\right\rangle_B^{1/2},
\qquad
\Delta\ln P=\ln P-\ln P_{\rm ref}.
\]

Subtracting the band mean removes an overall amplitude offset. Relative to the
4096-pc mean spectrum over 64--256 pc, \(D_{\rm shape}=0.071\) for the
1024-pc model and 0.054 for the 2048-pc model, corresponding to only about
5--7 percent RMS shape differences.

### Fixed-band spectral moments

The unrestricted integral scale includes every wavelength available in each
box and is therefore expected to grow when a larger box adds low-\(k\) power.
A band-limited alternative uses identical physical bounds \(B\) for every
model:

\[
\sigma_B^2 = \int_B E(k)\,dk,
\qquad
L_B =
\frac{\displaystyle\int_B E(k)\frac{2\pi}{k}\,dk}
     {\displaystyle\int_B E(k)\,dk}.
\]

Here \(\sigma_B^2\) is the overdensity variance contributed by the band and
\(L_B\) is its energy-weighted wavelength. Instantaneous measurements followed
by a 201--600 Myr temporal median give

| Band [pc] | Quantity | 1024 pc | 2048 pc | 4096 pc |
| --- | --- | ---: | ---: | ---: |
| 64--256 | \(L_B\) [pc] | 144.7 | 144.4 | 146.7 |
| 64--256 | \(\sigma_B^2\) | 0.219 | 0.219 | 0.230 |
| 64--512 | \(L_B\) [pc] | 231.9 | 229.5 | 229.2 |
| 64--512 | \(\sigma_B^2\) | 0.350 | 0.362 | 0.370 |

These fixed-band moments are substantially better converged than the
unrestricted \(L_{\rm in}\). Other bounded choices, such as the median
wavelength of cumulative band variance or the peak of
\(k^2P_\delta/(2\pi)\), are also possible. If the peak lies at the lowest
available wavenumber, it should be reported as unresolved rather than as a
box-independent scale. The dependence of \(L_B\) on an explicitly varied upper
wavelength cutoff is itself a useful convergence curve.

### Quadrupole bias and spatial coherence

Normalizing \(Q_2\) by total band power removes amplitude dependence but does
not remove the positive magnitude bias from a finite number of weighted
modes. For an angle-randomized null with independent conjugate-mode pairs,

\[
\left\langle |Q_{2,\rm null}|^2 \right\rangle
= \frac{\sum_j P_j^2}{\left(\sum_j P_j\right)^2}
= \frac{1}{N_{\rm eff}},
\]

which motivates

\[
A_{2,\rm deb} =
\sqrt{\max\left(A_2^2-\left\langle|Q_{2,\rm null}|^2\right\rangle,0\right)}.
\]

The production caches give the following median diagnostic values:

| Box size [pc] | Raw \(A_2\) | Null floor | \(A_{2,\rm deb}\) |
| ---: | ---: | ---: | ---: |
| 1024 | 0.265 | 0.109 | 0.240 |
| 2048 | 0.160 | 0.054 | 0.151 |
| 4096 | 0.124 | 0.028 | 0.121 |

Finite-mode bias explains part, but not all, of the box-size trend. A global
complex quadrupole also mixes local anisotropy strength with coherence of the
preferred direction across the domain. Independently oriented regions cancel
more efficiently in a larger box even if their local morphology is unchanged.

A more local alternative is to remap first, divide every model into
identically sized physical patches, apply a nonperiodic window, and calculate
\(Q_{2,p}\) for each patch. Two complementary summaries are

\[
A_{2,\rm local} = {\rm median}_p |Q_{2,p}|,
\qquad
C_{\rm orient} =
\frac{\left|\sum_p W_pQ_{2,p}\right|}
     {\sum_p W_p|Q_{2,p}|}.
\]

They separate local anisotropy strength from large-scale orientation
coherence. Each patch statistic should receive its own exact-lattice null
correction. Repeatedly subsampling the larger boxes to match the smaller
box's effective mode count is another useful test that isolates estimator
bias, although it does not remove genuine spatial decoherence.

### Relation to parameter trends in the fixed-box suite

These box-size sensitivities do not by themselves invalidate trends within
the main TIGRESS-NCR parameter suite. Those models share the same box size,
pixel scale, Fourier lattice, radial edges, fit interval, and quadrupole mode
support. Consequently, systematic changes with SFR, rotation, shear, or
stellar gravity cannot be attributed to changing fundamental modes or to the
box-size dependence quantified above. Coherent trends within that controlled
suite can therefore represent real responses to the varied physical
parameters.

The caveat is about absolute interpretation: unrestricted integral scales,
single slopes fitted across a curved spectrum, and global quadrupole
magnitudes retain estimator-specific sensitivities. Fixed-band moments,
local-slope or shape diagnostics, null-debiased local anisotropy, and temporal
block bootstrap intervals provide useful cross-checks. The existing
16th--84th temporal percentiles describe physical variability; they are not
uncertainties on the temporal median because adjacent snapshots are
correlated.

## Products

By default, files are written under
*SUITE/density_power_spectrum_box_size_theta0/*:

- *density_power_2d/MODEL.npz*: one full shear-remapped 2D periodogram series
  per logical model. The early and late source paths are recorded.
- *box_size_density_power_spectra.npz*: the common-grid radial spectra,
  quadrupoles, time diagnostics, box sizes, mean SFRs, and provenance.
- *box_size_density_power_spectrum_time_mean.png*: 201--600 Myr mean
  dimensional spectra and temporal-median scale-dependent quadrupole
  amplitudes.
- *density_power_2d_time_mean/MODEL.npz*: physical-coordinate time-mean 2D
  spectra. These remain separate because their Fourier arrays have different
  shapes.
- *box_size_density_power_2d_time_mean.png*: a shared-scale comparison of the
  central physical Fourier modes.
- *box_size_density_power_spectrum_diagnostics.csv* and *.png*: integral
  scale, 64--256 pc slope, and band quadrupole amplitude/angle versus box
  size. Points are temporal medians (axial circular means for angle), and bars
  span the 16th--84th percentiles.

The archive stores the joined segment directories, exact source projection
for every target time, effective shear parameters from both segments, actual
header times, pixel and box sizes, and all normalization definitions.

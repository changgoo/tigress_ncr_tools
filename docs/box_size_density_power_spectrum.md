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

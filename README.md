# TIGRESS-NCR tools

Lightweight, self-contained readers and command-line helpers for TIGRESS-NCR
simulation output. The readers do not require `pyathena`.

## Installation

```bash
python -m pip install -e "$HOME/tigress_ncr_tools"
```

The editable installation also makes the `pathena` reader package available
to Athena's `python/summary_plot.py` and related analysis scripts.

## Suite commands

```bash
check-suite /anvil/scratch/x-ckim5/TIGRESS-NCR
plot-suite-hst /anvil/scratch/x-ckim5/TIGRESS-NCR

# SFR-colored whole-domain kinetic, thermal, and Alfvén speed evolution.
plot-suite-hst-evolution /tigress/changgoo/anvil/TIGRESS-NCR-suite

# Shear-aware theta0 spectra of delta=Sigma/<Sigma>-1.
plot-suite-density-spectrum /tigress/changgoo/anvil/TIGRESS-NCR-suite --movie

# The same proj2d PDF/PS analysis for H I column and emission measure.
plot-suite-density-pdf /tigress/changgoo/anvil/TIGRESS-NCR-suite --quantity hi --workers 8
plot-suite-density-spectrum /tigress/changgoo/anvil/TIGRESS-NCR-suite --quantity em --workers 8

# Six-phase fractions, heights, velocity dispersions, and split Alfvén speeds.
plot-suite-phases /tigress/changgoo/anvil/TIGRESS-NCR-suite --workers 8

# Rank 32 models by <SFR10> over t=200--600 and render 4x8 theta0 movies.
plot-suite-evolution /tigress/changgoo/anvil/TIGRESS-NCR-suite --movie
plot-suite-evolution /tigress/changgoo/anvil/TIGRESS-NCR-suite \
  --map hydrogen-phases --movie

# SFR-ranked 4x8 XZ projections and movie at 1-Myr cadence.
plot-suite-xz /tigress/changgoo/anvil/TIGRESS-NCR-suite --movie --overwrite

# PBS Professional (for example, NASA Athena) is detected automatically.
check-suite /nobackup/$USER/TIGRESS-NCR
plot-suite-hst /nobackup/$USER/TIGRESS-NCR

# Pack the Lxy=4096, 2048, and 1024 pc late-run maps into one canvas.
# The default suite is /tigress/changgoo/nasa_athena/TIGRESS-NCR.
plot-suite-projections 30

# Theta=0 PDFs and shear-corrected power spectra for all late runs.
surface-density-stats /tigress/changgoo/nasa_athena/TIGRESS-NCR
```

`check-suite` combines Slurm or PBS accounting, current-attempt error logs,
history progress, and output age. Run directories are discovered from their
generated batch scripts or simulation outputs rather than a fixed model-name
pattern. Use `--scheduler` or `--model-glob` to override auto-detection.
`plot-suite-hst` writes `hst_summary.png` and
`hst_sfr_grid.png` in the suite directory unless `--output-dir` is supplied.
`plot-suite-hst-evolution` reads each model's primary and `whole.hst`
histories and writes `velocity_dispersions.png` plus an exact
`model_sfr_colors.csv` key beneath `SUITE/hst_evolution/`. The seven panels are
`sqrt(2*x1KE/mass)`, `sqrt(2*x2dke/mass)` (background shear removed),
`sqrt(2*x3KE/mass)`, `sqrt(P/mass)`, and `sqrt(2*xNME/mass)` for all three
magnetic components. Model color is a logarithmically normalized `plasma`
map of mean `sfr10` over `t=200--600`. Additional velocity-dispersion figures
use `viridis` for orbital frequency, `cividis` for stellar midplane density,
and `magma` for shear parameter. Use `--cmap`, `--color-scale`, `--yscale`,
`--start`, and `--stop` to change the SFR presentation or time interval.

The history-evolution output directory also contains
`velocity_dispersion_summary.csv`, with 200--600 Myr means, standard
deviations, medians, 16th--84th percentiles, and sample counts. The 7-by-3
`velocity_dispersions_correlations.png` plots the three kinetic dispersions,
thermal speed, and three Alfvén components against epicyclic frequency,
stellar midplane density, and mean SFR. The 4-by-3
`velocity_dispersions_derived_correlations.png` presents the corresponding 3D
kinetic speed, 3D Alfvén speed, turbulent Mach number, and
magnetic-pressure-corrected Mach number. Both use `plasma` mean-SFR colors.

`plot-suite-evolution` writes the ranked 4x8 total-gas surface-density frames,
`model_order.csv`, and (with `--movie`) an MP4 beneath
`SUITE/surface_density_evolution_theta0/`. Use `--map hydrogen-phases` for a
fixed-stretch pseudocolor movie beneath `SUITE/hydrogen_phase_evolution_theta0/`:
red is molecular hydrogen (`2H2`), green is atomic hydrogen (`HI`), and blue
is ionized hydrogen (`HII`). All channels count hydrogen nuclei and share one
physical surface-density stretch across every panel and time. The default
phase settings are `--phase-scale 25 --asinh-q 10 --hi-green-scale 0.65`.
The default evolution is outputs 0--600 inclusive, ordered from the highest
time-averaged `sfr10` at top left to the lowest at bottom right. `--start`,
`--stop`, and `--stride` select a subset, while `--sfr-start` and `--sfr-stop`
change the ranking interval.

`plot-suite-xz` reads the `pdf2d/x1-x3` projection integrated along the
azimuthal direction, converts `nH` to gas surface density using the projected
cell area, and preserves the physical 1:4 X-to-Z aspect ratio in every panel.
It uses the same SFR ranking and writes figures, `model_order.csv`, and
panel-level source provenance in `xz_sources.csv` beneath
`SUITE/surface_density_evolution_x1-x3/`. The default cadence is 1 Myr from
`t=0` through 600. Native files with zero data or invalid geometry produce a
blank panel, recorded as `blank-corrupt-pdf2d`; no temporal substitution is
performed. Use `--movie` to encode all available frames, `--stride 100` for
seven overview figures, or `--corrupt-policy vtk` to reconstruct a corrupt
panel when an exact-time archived VTK snapshot exists.


`plot-suite-projections` reads only `*_late/proj2d/thetaANGLE` and aligns the
runs by stored physical time, independent of snapshot number. Restart-overlap
times are deduplicated before matching. By default it writes
all combined frames inside the suite, for example
`/tigress/changgoo/nasa_athena/TIGRESS-NCR/projection_theta30/`. Use
`--start`, `--stop`, and `--stride` to render a subset:

```bash
plot-suite-projections 30 --start 0 --stop 1
plot-suite-projections theta45 --stride 10
plot-suite-projections 30 --movie --fps-in 15 --fps-out 15
```

With `--movie`, the command uses the same ffmpeg workflow and defaults as
`Athena-TIGRESS-NCR/python/summary_movie.py`. Movies default to
`SUITE/movies/projection_thetaANGLE.mp4`. Codec auto-detection prefers
`libx264` and falls back to `mpeg4`; `--movie-path`, `--codec`, `--crf`,
`--qscale`, and `--bitrate` provide the same controls.

## Suite density power spectra

`plot-suite-density-spectrum` analyzes the complete 32-model theta0 sequence using
`delta = Sigma/<Sigma> - 1`. It first remaps each map to periodic shearing
coordinates, then assigns each Fourier mode its instantaneous physical
wavenumber with `kx = kx0 + q*Omega*t_remap*ky`. Runtime `problem/qshear` and
`problem/Omega` batch-script overrides take precedence over the athinput
template values. The regenerated `R8_8pc_NCR_row0000` projection sequence is
included in the archive and every derived product.

The transform is now a two-stage pipeline. Each model first stores its complete
601-snapshot 2D periodogram series under
`proj2d/theta0/density_power_2d/density_power_2d.npz`; the collective 1D
spectra and all radial diagnostics are then reduced from those caches. Use
`--workers N` for model-level parallel generation and `--overwrite-2d` only
when the projection maps or transform settings change.

The command also deposits the 200--600 Myr spectra onto a common physical
`(kx,ky)` grid and writes `density_power_2d_time_mean.npz` plus the SFR-ranked
4-by-8 `density_power_2d_time_mean.png`. The figure uses a common logarithmic
`plasma` scale and defaults to `|kx*Lx/(2*pi)|,|ky*Ly/(2*pi)| <= 16`; change
the view with `--power2d-mode-limit`.

The mathematical definition, discrete normalization, shear-coordinate
derivation, and archive-field inventory are documented in
[`docs/density_power_spectrum.md`](docs/density_power_spectrum.md).

The default output directory is `SUITE/density_power_spectrum_theta0/`. It
contains the complete `P_delta(t,k)` archive, a `t=200--600` mean-spectrum
comparison, an integral-scale/spectral-slope CSV, four two-panel SFR relation
figures, a 4-by-3 correlation figure, the exact model/SFR/color key, and—with
`--movie`—the 601-frame spectrum evolution and MP4. The slope fits
`P_delta proportional to k^-alpha`
over the fixed band `64 pc < 2*pi/k < 256 pc`. Scatter colors use `plasma`,
`viridis`, `cividis`, and `magma` for SFR, orbital frequency, stellar midplane
density, and shear.
Spectrum figures use `kL/(2*pi)` on the lower axis and the corresponding
physical wavelength `lambda=2*pi/k` in pc on the upper axis.

`L_in`, `alpha`, and the 64--256 pc quadrupole are measured separately for
every 1-Myr spectrum. The angle-averaged figure shows mean power and median
`A2=abs(Q2)` rather than the former dimensionless-power panel. Relation
figures show the 200--600 Myr temporal median with 16th--84th percentile bars.
The 4-by-3 figure compares `L_in`, `alpha`, `A2`, and axial angle `phi2` with
epicyclic frequency `kappa=sqrt(2*(2-q))*Omega`, stellar midplane density, and
mean SFR. The
archive also stores the temporal mean and standard deviation, the full time
series, and the older diagnostics measured from the mean spectrum.

```bash
plot-suite-density-spectrum /tigress/changgoo/anvil/TIGRESS-NCR-suite --movie

# Recompute the archive and every movie frame after source projections change.
plot-suite-density-spectrum /tigress/changgoo/anvil/TIGRESS-NCR-suite \
  --overwrite --movie
```

## Suite density PDFs

`plot-suite-density-pdf` measures area-weighted one-point PDFs of
`delta=Sigma/<Sigma>-1` and `s=ln(Sigma/<Sigma>)` for every clean 1-Myr theta0
map. It stores instantaneous pixel standard deviations and summarizes them
over 200--600 Myr with medians and 16th--84th percentile bars. A 2-by-3 figure
correlates both widths with epicyclic frequency, stellar midplane density, and
mean SFR. A companion 2-by-7 figure compares the same widths with the three
mass-weighted kinetic dispersions, thermal speed, and three Alfvén speeds from
the whole-domain histories, using 200--600 Myr medians and percentile bars.
A separate 2-by-4 panel compares both PDF widths with the 3D turbulent and
Alfvén speeds, the turbulent Mach number, and its magnetic-pressure-corrected
form.

The output directory `SUITE/density_pdf_theta0/` also contains all-model median
PDFs and a 4-by-8 grid comparing each median `s`-PDF with its moment-matched
Gaussian, corresponding to a log-normal column-density model. Definitions and
archive fields are documented in
[`docs/density_pdf.md`](docs/density_pdf.md).
The two-panel median-PDF figure automatically removes empty far-tail ranges
below a PDF-density envelope of `1e-4`.

```bash
plot-suite-density-pdf /tigress/changgoo/anvil/TIGRESS-NCR-suite --workers 4
```

Add `--quantity hi` or `--quantity em` to run the identical `proj2d` PDF
estimator for atomic-hydrogen column or emission measure. The power-spectrum
command accepts the same option and keeps separate per-species 2D caches and
suite products. Field definitions, tracer-specific PDF ranges, commands, and
interpretation are in
[`docs/projected_species_analysis.md`](docs/projected_species_analysis.md).

`plot-hi-fraction-mask MODEL VTK_NUMBER` reconstructs a face-on H I
column from the full VTK ranks and compares it with a version that excludes
cells below a chosen neutral-fraction threshold.

After all three tracer summaries exist, `plot-suite-tracer-correlations`
compares gas, H I, and EM PDF widths and the \(L_{\rm in}\), \(\alpha\), and
\(A_2\) spectrum diagnostics, including temporal ranges and pairwise Spearman
coefficients.

## Six-phase zprof diagnostics

`plot-suite-phases` combines the NCR phase z-profiles into CNM+CMM, UNM, WNM,
WIM, WHIM, and HIM. It stores full temporal scatter for whole-box and
within-`H_gas` mass/volume fractions, mass/volume RMS heights, component and
3D velocity dispersions, PRFM effective vertical support speeds
`sqrt(int(Pturb,z + Pth + Pi_B) dz / int(rho) dz)`, and mean-field versus
perturbed-field Alfvén speeds.
It also plots 400--600 Myr ensemble summaries against the four direct
environmental inputs \(\Sigma_*,H_*,\Omega,q\) and the two derived predictors
\(\kappa,\rho_*\), as well as the 200--600 Myr mean \(\Sigma_{\rm SFR,10}\).
Fractions have direct phase-by-parameter panels and correlation matrices,
including separate neutral and ionized reductions. Scale heights and dynamics
have matching full-scatter products and Spearman heatmaps with neutral,
ionized, and true whole-gas rows. Evolution y-limits use 200--600 Myr values
so startup anomalies do not compress the later histories.
The omitted UIM is retained as an explicit closure residual rather than
renormalizing the requested six phases. Definitions and output schemas are in
[`docs/phase_resolved_analysis.md`](docs/phase_resolved_analysis.md).

```bash
plot-suite-phases /tigress/changgoo/anvil/TIGRESS-NCR-suite --workers 8
```

## Suite PRFM diagnostics

`plot-suite-prfm` constructs two-phase midplane pressure and whole-column
vertical weight directly from the raw z-profiles. Following the reference NCR
analysis, two-phase gas is `phase7 + phase11 + phase12 + phase13`; pressures
are averaged over `-10 <= z <= 10` pc and divided by the two-phase area
fraction. External and self-gravitating weights are integrated inward from
both vertical boundaries. The pressure components are turbulent, thermal,
turbulent-field Maxwell stress, and mean-field Maxwell stress.

The reducer also measures the stress in reference slabs near $z=\pm L_z/4$
and stores $\Delta P=P_{\rm mid}-P_{\rm top}$ for each component and their
sum.

The default `t=200--600` reduction interpolates `sfr10` and `sfr40` from each
primary history. The evolution plot retains this full interval, while scalar
summaries and vertical profiles default to averages over `t=400--600`.
Feedback yields are formed snapshot by snapshot as `pressure / sfr40` and
converted to `km/s` before temporal averaging.
The mean-`sfr10` figures use `plasma`. Alternate parameter figures use
distinct sequential maps: `viridis` for orbital frequency, `cividis` for
stellar midplane density $\Sigma_*/(2H_*)$, and `magma` for shear parameter
$q$.

The output directory `SUITE/prfm_diagnostics/` contains the complete
time-series and vertical-profile CSV caches, a per-model summary with means
and 16/50/84 percentiles, the model/color key, midplane and pressure-drop
balance figures, a `2x4` pressure-component/yield figure, and separate `2x3`
vertical-profile figures for two-phase and total gas. The three alternate
parameter color mappings are applied to the vertical profiles as well as the
scalar relation figures. `prfm_pressure_weight_time_evolution.png` shows the
full-suite midplane pressure and vertical weight histories.

The physical definitions, discrete reduction equations, unit conversions,
output schema, and interpretation are documented in
[`docs/prfm_analysis.md`](docs/prfm_analysis.md).

```bash
plot-suite-prfm /tigress/changgoo/anvil/TIGRESS-NCR-suite

# Re-read all z-profiles after simulation output changes.
plot-suite-prfm /tigress/changgoo/anvil/TIGRESS-NCR-suite --overwrite
```

## Surface-density statistics

`surface-density-stats` reads the late-run `proj2d/theta0` maps in stored-time
order, independent of snapshot number. It reads effective `qshear` and `Omega`
from runtime batch overrides when present, falling back to `athinput*`, and
applies Athena's residual shear remap before the FFT,
including the physical shearing-wave correction
`kx = kx0 + q*Omega*t_remap*ky`.

Each run receives
`proj2d/theta0/surface_density_statistics.npz`. The archive contains time
series of area- and mass-weighted PDFs for `log10(Sigma)`,
`delta = Sigma/<Sigma> - 1`, and `s = ln(Sigma/<Sigma>)`, plus angle-averaged
power spectra for `delta` and `s`. The suite directory receives
`surface_density_pdf_summary.png` and
`surface_density_power_summary.png`, showing temporal medians and shaded
5--95 percentiles for every late-run box size.

Full-domain spectra are the default. A centered local spectrum can be
apodized and zero padded, for example:

```bash
surface-density-stats /tigress/changgoo/nasa_athena/TIGRESS-NCR \
  --subregion-size 512 --window tukey --tukey-alpha 0.25 --pad-factor 2
```

`--window hann` is also available. `--pdf-min/--pdf-max`,
`--delta-min/--delta-max`, `--s-min/--s-max`, `--pdf-bins`, and `--k-bins`
control the fixed time-series grids. The subregion, window, and padding
options apply only to the spectra; PDFs remain full-domain.

## Snapshot archiving

`archive-tigress-snapshots` converts completed `vtk/NNNN` and `rst/NNNN`
directories into pyathena-compatible `PROBLEM.NNNN.tar` files. With
`--remove-originals`, it removes a snapshot directory only after checking that
the archive contains exactly the original files and uncompressed sizes.

Start with a dry run:

```bash
archive-tigress-snapshots \
  /nobackup/$USER/TIGRESS-NCR/R8_8pc_NCR_* \
  --keep-latest 1 --min-age-minutes 30 --remove-originals --dry-run
```

Then remove `--dry-run` for the cron command. The default processes both VTK
and restart snapshots. Keeping the newest directory prevents racing Athena and
preserves the restart directory needed by PBS auto-resubmission. The command
also infers the expected MPI-rank count from the copied PBS script, rejects
incomplete/non-contiguous rank sets, creates the tar through a temporary file,
verifies it, and atomically renames it before deleting originals. For runs
without a generated PBS script, pass `--expected-ranks N` explicitly when
using `--remove-originals`.

Example crontab entry, every 20 minutes:

```cron
*/20 * * * * $HOME/myenv/bin/archive-tigress-snapshots /nobackup/$USER/TIGRESS-NCR/R8_8pc_NCR_* --keep-latest 1 --min-age-minutes 30 --remove-originals >>$HOME/archive-tigress-snapshots.log 2>&1
```

## Tests

```bash
python -m pytest
```

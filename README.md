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

## Slice story movie

The slice-story renderer builds a deterministic presentation sequence from
`slicevtk` and star-particle outputs. Inspect the available times, planes, and
required fields first:

```bash
slice-story-movie /path/to/RUN --problem-id R8_8pc_NCR --preflight
```

The renderer creates the density/species/temperature sequence, a
density-opacity temperature volume with a registered XY-to-XZ camera turn,
the evolving XZ view, velocity and magnetic streamlines, and the FUV/LyC
composite. Install the volume interpolation dependency with
`pip install -e '.[movie3d]'`. The command writes numbered PNGs plus
`frame_manifest.csv`:

```bash
slice-story-movie /path/to/RUN --problem-id R8_8pc_NCR \
  --preview --output-dir preview

slice-story-movie /path/to/RUN --problem-id R8_8pc_NCR \
  --movie --output-dir movie_slice_story
```

`--start-frame`, `--stop-frame`, and the default skip-existing behavior make
long renders resumable. Use `--overwrite` to replace existing PNGs. Volume
rendering assembles the complete selected 3D fields, then uses render-only
subsampling: preview defaults to `--volume-stride 4`, production to stride 2,
and `--volume-stride 1` preserves native resolution. `--volume-max-gib` limits
the full-array allocation, `--volume-opacity-scale` tunes visibility, and
`--no-volume` retains the lighter slice-only path. Full-volume inputs may be
live `vtk/NNNN/` directories or the uncompressed `vtk/PROBLEM.NNNN.tar`
archives created by `archive-tigress-snapshots`; tar members are read directly
without extracting a second copy.

## Surface-density statistics

`surface-density-stats` reads the late-run `proj2d/theta0` maps in stored-time
order, independent of snapshot number. It reads `qshear` and `Omega` from each
run's `athinput*` and applies Athena's residual shear remap before the FFT,
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

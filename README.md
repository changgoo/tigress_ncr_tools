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

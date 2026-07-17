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
```

`check-suite` combines Slurm or PBS accounting, current-attempt error logs,
history progress, and output age. Run directories are discovered from their
generated batch scripts or simulation outputs rather than a fixed model-name
pattern. Use `--scheduler` or `--model-glob` to override auto-detection.
`plot-suite-hst` writes `hst_summary.png` and
`hst_sfr_grid.png` in the suite directory unless `--output-dir` is supplied.

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

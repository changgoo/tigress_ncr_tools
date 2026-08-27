# Repository Guidelines

## Project Structure & Module Organization

Python packages use the `src/` layout. `src/tigress_ncr_tools/` contains the
analysis and command-line modules, while `src/pathena/` provides lightweight
TIGRESS output readers. Tests live in `tests/` and generally mirror a module or
feature (for example, `plot_suite_xz.py` is covered by `test_suite_xz.py`).
Scientific method notes belong in `docs/`; exploratory workflows belong in
`notebooks/`. Keep generated plots, movies, caches, and simulation outputs out
of version control.

## Build, Test, and Development Commands

- `python3 -m pip install -e .` installs both packages and all console scripts
  for iterative development.
- `python3 -m pytest` runs the complete test suite configured in
  `pyproject.toml`.
- `python3 -m pytest tests/test_archive_snapshots.py -q` runs one focused test
  module while developing.
- `check-suite --help` (or another script listed under `[project.scripts]`)
  verifies an installed CLI and displays its supported options.

Run analysis commands against scratch or copied simulation data; many commands
write figures and archives beside the input suite.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, PEP 8 spacing,
snake_case functions and modules, PascalCase test classes, and UPPER_CASE
constants. Prefer `pathlib.Path` for filesystem work and small, testable helper
functions around CLI orchestration. Use descriptive command options and keep
CLI parsing in a `main()` function. No formatter or linter is configured, so
match nearby code and keep imports grouped as standard library, third-party,
then local modules.

## Testing Guidelines

Tests use `pytest` discovery, with many cases written using `unittest.TestCase`.
Name files `test_<feature>.py` and methods `test_<behavior>`. Use temporary
directories and synthetic arrays/files instead of depending on site-specific
TIGRESS datasets. Add regression coverage for bug fixes and exercise dry-run
or non-destructive paths for archival code.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative subjects such as `Add phase effective
vertical support diagnostics`. Keep each commit focused and include related
tests or documentation. Pull requests should explain the scientific or
operational goal, summarize validation commands, link relevant issues, and
note output-schema or CLI changes. Include representative plots when visual
results change, but do not commit bulky generated output.

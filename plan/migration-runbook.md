# Migration runbook

## Objective

Move private suite planning, scripts, designs, manuscript sources, and selected
artifacts from `$HOME/PRFM` branch `project/tigress-phangs-ili` into a new
private `tigress-learn-ism` repository. Rebuild the reusable PHANGS integration
on a clean PRFM branch, validate both destinations, and only then remove the
old project branch.

This runbook deliberately favors a reviewed snapshot plus recorded source SHAs.
The user has no requirement to preserve the complete branch history, and the
snapshot workflow makes the public/private boundary auditable.

## Non-negotiable safety rules

- Do not delete or rewrite the source branch during extraction.
- Do not overwrite the four currently observed untracked project files.
- Do not copy credentials, private PHANGS payloads, HPC logs, simulation
  binaries, caches, or bulk generated output into Git.
- Do not automatically promote existing PNG/PDF files to final figures.
- Do not push a new repository, publish a site, or delete a remote branch
  without explicit authorization at execution time.
- Record source SHAs and inventories before modifying either repository.
- A fresh clone/build/test is the migration acceptance test; the existence of
  copied files is not sufficient.

## Phase 0: freeze and inventory

1. In `$HOME/PRFM`, record:
   `git status --short --branch`, `git rev-parse HEAD`,
   `git ls-files project`, and an inventory of untracked files.
2. Save a private backup of the working tree or create a Git bundle in a
   nonpublic backup location. Do not push a tag containing private history to
   a public PRFM remote.
3. Record the source branch and SHA in
   `manifests/source-inventory/prfm-migration.yaml` in the new private repo.
4. Generate a file classification table with these dispositions:
   `move-private`, `keep-public`, `duplicate-temporarily`, `generated-review`,
   or `exclude`.
5. Inventory the core suite independently. Record 32 expected model IDs,
   design row mapping, run status, restart history where relevant, and all
   suite-level derived-product directories.

Acceptance criteria:

- every tracked file changed from PRFM `main` is classified;
- every untracked PRFM file is classified;
- the inventory has sizes and checksums for manuscript sources and candidate
  final artifacts; and
- the source branch remains unchanged and recoverable.

## Phase 1: create the private project repository

1. Create private repository `tigress-learn-ism` under the selected GitHub
   owner and clone it beside, not inside, either existing repository.
2. Add the structure in `repository-boundaries.md`, a restrictive `.gitignore`,
   and an `AGENTS.md` that requires scoped commits and validation.
3. Add dependencies on PRFM and `tigress_ncr_tools`. During migration, pin
   both by local editable paths for development and record exact SHAs in run
   manifests. Replace local paths with release/tag references when practical.
4. Add `data/README.md` explaining that data remain at HPC/Globus locations
   and that local paths come from an ignored configuration file.
5. Commit this empty scaffold before importing legacy material.

Acceptance criteria:

- a fresh private clone installs its environment;
- tests can run with no access to the simulation suite; and
- `git status` does not reveal ignored data or build products.

## Phase 2: import private material

Use the classification inventory, not a blanket copy of the PRFM branch.

### Move as project source

- `project/TIGRESS-PHANGS-ILI-project.md` and related planning/design notes;
- core-suite mapping, completion, correlation, and synthesis documents;
- project workflow and TODO material after reconciling stale tasks;
- suite YAML/design CSV files that exactly define simulations already run;
- project orchestration, validation, and paper-specific plotting scripts;
- manuscript TeX, sections, bibliography, and genuinely shared TeX support;
- tests for project-specific scripts; and
- all four untracked project files observed in the planning inventory.

Place historical prose that is no longer normative under `docs/legacy/` with
a header giving its original path and source SHA. Put active decisions into
the new `plan/` rather than silently treating every legacy TODO as current.

### Review instead of automatically committing

- all existing `project/output/*.png`;
- all `paper-figures/*.png`;
- manuscript `main.pdf` files;
- duplicated AASTeX/BST files; and
- generated CSVs that can be reproduced and are not the authoritative suite
  design or a release table.

Copy these to ignored `build/migration-review/` if needed for comparison.
Only a confirmed final figure moves into a tracked `figures/final/` directory,
and it must be committed together with a provenance manifest.

### Exclude

- Python caches, notebook checkpoints, temporary files, logs, credentials,
  machine-specific paths, SLURM output, and raw/bulk simulation data;
- generated job scripts unless they are required as immutable run provenance
  and pass a secrets/path review; and
- copied public package implementations that should be imported from PRFM or
  `tigress_ncr_tools`.

Acceptance criteria:

- manuscript sources build from the new tree;
- project scripts resolve their inputs through configuration, not `$HOME/PRFM/project`;
- core design CSV/YAML content matches the simulations actually run;
- project tests pass; and
- no candidate figure is tracked as final without confirmation.

## Phase 3: retain reusable PHANGS work in PRFM

Create a clean PRFM integration branch from the current public base (`main` at
execution time). Do not try to make the private source branch itself public by
deleting only obvious manuscript files; review every diff.

Public candidates currently include:

- reusable changes in `prfm/phangs.py` and `prfm/phangs_plot.py`;
- `prfm/phangs_sampling.py` after removing project defaults and paths;
- public configuration/schema and a minimal public example;
- book/notebook documentation suitable for general users;
- general sampling-method documentation; and
- focused unit tests for PHANGS loading, transformations, sampling, and plots.

Keep out of the clean branch:

- the entire private manuscript/program narrative;
- fixed 32-model design files and model mapping;
- TIGRESS job generation and machine-specific orchestration;
- paper-specific plotting/analysis scripts;
- candidate paper figures and compiled manuscripts; and
- assumptions that only make sense for the core suite.

Refactor thin private scripts to call public PRFM APIs. If a reusable feature
cannot be separated cleanly, move it to the private repo first and open a
small, independently reviewable PRFM change later.

Validation:

1. run the full PRFM test suite;
2. build its public documentation/book;
3. run at least one public PHANGS sampling example with documented data
   requirements;
4. install PRFM into a clean environment; and
5. run the private project's sampling/design tests against that exact PRFM
   commit.

## Phase 4: decouple this repository

No project-specific code is to be added here during migration. Confirm instead
that generic tools expose everything the private project needs through stable
CLI/API inputs and documented archive fields.

For any missing capability:

1. implement it first in the private project if its generality is uncertain;
2. promote it here only when it accepts arbitrary suites, has a documented
   schema, and has synthetic tests;
3. keep paper layout, model selection, and cross-package joins private; and
4. pin the validated `tigress_ncr_tools` SHA in project manifests.

Run `python3 -m pytest` in this repository before freezing a paper/release
analysis version.

## Phase 5: cut over manuscripts and figures

1. Select the active suite manuscript tree in the private repository.
2. Replace legacy relative paths with private-repo paths and manifest-driven
   inputs.
3. Build all candidate figures into ignored `build/figures/`.
4. Review each candidate for scientific correctness, labels/units, legibility,
   consistency, and reproducibility.
5. After explicit confirmation, copy only the selected artifact to the
   manuscript's `figures/final/` and commit it with its manifest.
6. Build the paper in a clean clone using only committed final figures and
   tables.

Acceptance criteria:

- the manuscript build does not read `$HOME/PRFM/project`;
- every included figure/table has an ID and manifest;
- the build records package/project SHAs; and
- removing ignored `build/` does not break a clean manuscript build.

## Phase 6: retire the old PRFM project branch

This is a final gate, not part of extraction.

Retire `project/tigress-phangs-ili` only after:

- the private repository is remotely backed up and passes a fresh-clone test;
- the PRFM reusable branch is merged or otherwise safely retained;
- tracked and untracked source inventories reconcile with both destinations;
- manuscript and project-script builds pass from the new repository;
- collaborators have been told the new location; and
- an explicit deletion authorization is given.

Then remove the local branch and, separately, the remote branch. Record the
retirement date, final SHA, and replacement locations in both repositories.
Do not publish a backup tag or bundle containing private project content in
the public PRFM repository.

## Rollback

Until branch retirement, rollback means switching back to the untouched PRFM
source branch. After retirement, recover from the private backup/bundle and the
new private repository. The migration inventory must make it possible to
prove that each source file was kept, intentionally excluded, or regenerated.

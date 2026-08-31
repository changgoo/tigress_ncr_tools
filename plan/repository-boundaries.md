# Repository boundaries and interfaces

## Governing principle

Put code at the lowest layer where its scientific meaning is general, and put
decisions at the project layer where their context is explicit. The private
project repository may depend on both public packages. Neither public package
may depend on the private repository.

```text
PRFM --------------------+
                         +--> tigress-learn-ism --> reviewed release manifests
tigress_ncr_tools -------+                              |
                                                        v
                                                tigress-learn-ism-data
                                                        |
                                                        v
                                                Globus data collections
```

The release site can cite package versions and consume exported manifests. It
must not import private project code or require access to private Git history.

## Placement tests

### Put functionality in PRFM when

- it ingests or standardizes PHANGS/observational data for general PRFM use;
- it implements a reusable prior, transformation, sampler, or diagnostic that
  is meaningful without the TIGRESS-NCR core-suite model list;
- its API can be documented with public, nonsensitive examples;
- it can be tested without private paths or unpublished project tables; and
- a non-TIGRESS user could plausibly use it.

Examples from the current branch are the general parts of
`prfm/phangs_sampling.py`, PHANGS field transformations, plotting helpers,
configuration schema, notebook/book examples, and unit tests.

### Put functionality in tigress_ncr_tools when

- it reads a documented TIGRESS-NCR output format;
- it accepts model or suite locations as inputs instead of encoding the core
  suite or its 32 rows;
- the estimator or reduction applies to other TIGRESS-NCR suites;
- outputs have a documented, reusable schema; and
- tests use synthetic arrays/directories rather than private project data.

Examples are history/z-profile/projection readers, phase reductions, PRFM
diagnostics, surface-density PDFs and spectra, and suite health checks.

### Put functionality in tigress-learn-ism when

- it joins PHANGS sampling outputs to the named core suite;
- it encodes the fixed row-to-model mapping, paper time intervals, plot panel
  selection, narrative grouping, or project-specific quality decisions;
- it combines products from PRFM and `tigress_ncr_tools`;
- it generates a numbered manuscript figure or table;
- it prepares a particular Globus release inventory; or
- it is exploratory or inference-specific and has not earned a reusable API.

Project-specific code should not be placed in a project namespace inside
`tigress_ncr_tools`. If a project routine becomes genuinely reusable, first
give it synthetic tests and a suite-independent interface, then promote it to
the appropriate public package in a separate change.

### Put material in tigress-learn-ism-data when

- it is necessary for a user to discover, cite, download, verify, or interpret
  a released dataset;
- it is safe and approved for public release; and
- it does not require private dependencies to render the static site.

## Private project repository layout

Create the following baseline. Names may evolve, but the separation between
source, build products, accepted artifacts, and external data should remain.

```text
tigress-learn-ism/
  README.md
  AGENTS.md
  CITATION.cff                 # add when title/authors are stable
  pyproject.toml               # project environment and internal code only
  plan/
    decisions.md
    roadmap.md
  config/
    paths.example.toml         # symbolic/local configuration; no private paths
    analyses/
    figures/
  designs/
    core/
    extended/
  manifests/
    source-inventory/
    figures/
    releases/
  src/tigress_learn_ism/       # tested project-specific analysis/plot logic
  scripts/                     # thin command entry points
  tests/
  papers/
    suite/
      main.tex
      sections/
      figures/final/           # only explicitly confirmed figures
      tables/final/
    core-results/              # activate only if the first paper is split
    inference/                 # planning only until readiness gate
    shared/                    # bibliography and author-approved TeX style
  docs/
    science/
    methods/
    legacy/                    # migrated notes retained for provenance
  build/                       # ignored: candidates, PDFs, caches, staging
  data/                        # ignored except README/schema pointers
```

Use `src/tigress_learn_ism/` for logic that is shared by multiple project
scripts. Scripts should be thin wrappers so that scientific calculations are
testable. This is still private project code; packaging it internally does not
make it part of either public package.

## Figure ownership and approval

The private repository has three figure states:

1. **Candidate:** generated into ignored `build/figures/<paper>/<figure-id>/`.
2. **Confirmed:** the PI explicitly approves scientific content and visual
   presentation. Copy the exact selected artifact to
   `papers/<paper>/figures/final/` and commit it with its manifest.
3. **Superseded:** retain provenance in Git history or the manifest, but do not
   keep multiple ambiguous "final" files in the active manuscript tree.

Every confirmed figure must have a small YAML or JSON manifest recording:

- stable figure ID and manuscript destination;
- caption-purpose summary;
- generation command;
- project, PRFM, and `tigress_ncr_tools` commit IDs;
- input analysis product paths and checksums where practical;
- suite/release version and exact model selection;
- configuration file and important time/quality cuts;
- output filename and checksum;
- confirmation date and confirmer; and
- known caveats.

Rendered manuscript PDFs are build products. Commit one only when it serves a
deliberate collaboration/submission checkpoint, not after every local build.

## Data interfaces

### Simulation to generic tools

Inputs are TIGRESS-NCR model directories. Generic tools write derived archives,
tables, figures, and provenance beside the suite or into an explicit output
directory. Absolute paths are execution details and must not become public
identifiers.

### Generic tools to private project

The private project reads documented archives and tables. It must record the
`tigress_ncr_tools` commit and should fail clearly when an expected schema
version or field is absent. Paper scripts must not scrape values from PNGs or
depend on undocumented filename coincidences.

### PRFM to private project

The project imports public PRFM APIs for PHANGS data preparation and sampling.
Private configuration supplies project cuts and design choices. The project
pins a tested PRFM commit or release in each environment lock/report.

### Private project to public release site

Only an approved export crosses this boundary: release manifests, schemas,
documentation, thumbnails, citation metadata, and public Globus identifiers.
The export must be reviewable as a diff and contain no private filesystem
paths, credentials, unpublished coauthor material, or restricted PHANGS data.

## Dependency recording

Each published figure, table, and data release must identify immutable source
versions. During active development these may be Git SHAs; for publication
they should become tagged package/repository releases. A minimal run record is:

```yaml
run_id: core-v1-prfm
suite_id: tigress-ncr-core-32
suite_manifest: manifests/source-inventory/core-32.yaml
tigress_ncr_tools_commit: <40-character SHA>
prfm_commit: <40-character SHA>
project_commit: <40-character SHA>
config: config/analyses/core-prfm.yaml
created_utc: <ISO-8601 timestamp>
```

Do not record only branch names; branches move.

## Boundary review cadence

Review placement whenever project code is reused by a second suite or paper.
Promotion to a public package is warranted only after removing fixed project
assumptions, documenting the interface, and adding synthetic tests. Do not
block paper work on premature generalization.

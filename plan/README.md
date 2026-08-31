# Learning the ISM with TIGRESS: program plan

Status: adopted planning baseline
Last updated: 2026-08-31
Provisional umbrella name: **Learning the ISM with TIGRESS**
Provisional repository slug: `tigress-learn-ism`

## Purpose

This directory is the handoff point for reorganizing the TIGRESS-NCR suite
work without relying on context stored in a shell history, an unmerged branch,
or one person's memory. It records the agreed repository boundaries, the safe
migration sequence, the publication program, and the public data-release
workflow.

The immediate scientific asset is the 32-model core suite at
`/tigress/changgoo/anvil/TIGRESS-NCR-suite`. The current analysis software is
in this repository. PHANGS-informed sampling code and private project material
are currently mixed on branch `project/tigress-phangs-ili` of
`$HOME/PRFM`. That mixture is temporary and is the first issue to resolve.

## Decisions already made

1. Create a new **private** project repository, provisionally named
   `tigress-learn-ism`.
2. Keep `$HOME/PRFM` public-facing. Reusable PHANGS integration and sampling
   code stays there; private project material moves out.
3. Keep `tigress_ncr_tools` reusable and suite-agnostic. Project-specific
   scripts move to the private project repository.
4. Commit only final figures after explicit scientific/visual confirmation.
   Candidate figures and bulk generated products remain untracked.
5. Create a separate **public** data-release site repository, provisionally
   named `tigress-learn-ism-data`. Data bytes are distributed through Globus;
   the repository contains the site, metadata, schemas, manifests, checksums,
   and citation material.
6. Organize the science program into three components:
   simulation-suite design and philosophy, core-suite results, and later
   inference. The first two may become one paper or two. Inference is deferred
   until an inference-ready extended suite exists.
7. Migration may use a reviewed snapshot rather than preserving every source
   commit. The source commit IDs and a file-level migration inventory must be
   retained for provenance.

## Repository map

| Component | Visibility | Owns | Must not own |
|---|---|---|---|
| `PRFM` | Public | General PRFM analysis, PHANGS ingestion/integration, reusable informed-prior and sampling APIs, public examples and tests | TIGRESS suite run lists, private manuscripts, paper-specific scripts |
| `tigress_ncr_tools` | Public when ready | General TIGRESS-NCR readers, generic model/suite analysis CLIs, documented output schemas, synthetic tests | Paper assembly, fixed core-suite selections, private planning, accepted paper figures |
| `tigress-learn-ism` | Private | Program context, suite designs, project-specific orchestration and plots, manuscripts, figure manifests, confirmed final figures | Raw/bulk simulation output, reusable functionality that belongs upstream |
| `tigress-learn-ism-data` | Public | Release documentation/site, schemas, machine-readable catalogs, checksums, Globus links, release notes and citations | Simulation payloads, private drafts, unreviewed science claims |
| Globus collections | Controlled/public data plane | Versioned simulation and derived-data payloads | Source code or the canonical documentation site |

See [repository-boundaries.md](repository-boundaries.md) for the detailed
ownership tests and interfaces.

## Plan documents

- [repository-boundaries.md](repository-boundaries.md): durable ownership
  rules, dependency directions, and the proposed private-repository layout.
- [migration-runbook.md](migration-runbook.md): ordered, reversible migration
  from the PRFM branch and validation gates before branch deletion.
- [publication-roadmap.md](publication-roadmap.md): the three-stage scientific
  program, the one-versus-two-paper decision gate, and reproducible figure and
  manuscript workflows.
- [data-release-plan.md](data-release-plan.md): Globus curation, public-site
  contents, manifests, versioning, validation, and release gates.
- [execution-checklist.md](execution-checklist.md): the cross-repository order
  of operations and concrete completion criteria.

## Current-state inventory

### Core simulation suite

- Location: `/tigress/changgoo/anvil/TIGRESS-NCR-suite`
- Models: `R8_8pc_NCR_row0000` through `R8_8pc_NCR_row0031`
- Core design: 32 PHANGS-informed environments near
  `Sigma_gas = 10 M_sun pc^-2`
- Principal suite-level product directories observed on 2026-08-31:
  `hst_evolution`, `phase_evolution_zprof`, `prfm_diagnostics`,
  `density_pdf_theta0`, `density_power_spectrum_theta0`,
  `hi_pdf_theta0`, `hi_power_spectrum_theta0`, `em_pdf_theta0`,
  `em_power_spectrum_theta0`, `tracer_correlations_theta0`,
  `surface_density_evolution_theta0`,
  `surface_density_evolution_x1-x3`, and
  `hydrogen_phase_evolution_theta0`.
- The simulation directory is a working/data location, not a publication
  archive and not a Git repository.

### PRFM source branch

- Repository: `$HOME/PRFM`
- Source branch: `project/tigress-phangs-ili`
- Observed source commit: `57aa9b4` (`Record completion of the core suite`)
- Reusable public candidates include `prfm/phangs.py`,
  `prfm/phangs_plot.py`, `prfm/phangs_sampling.py`, public configuration,
  book material, and their tests.
- Private material is concentrated under `project/`: suite designs, workflow
  scripts, analysis synthesis, two manuscript drafts, and generated products.
- Untracked project files observed during planning must be included in the
  migration inventory before any cleanup:
  `project/doc/paper1-integration-plan.md`,
  `project/scripts/plot_sfr_corner.py`,
  `project/scripts/suite_analysis.py`, and
  `project/doc/paper1-suite/paper-figures/design_Sgas10.0_n0032_sfr_corner.png`.

### This repository

- Branch observed during planning: `analysis/ncr-suite`
- Generic readers and analysis CLIs are under `src/pathena/` and
  `src/tigress_ncr_tools/` with tests under `tests/`.
- Scientific method and archive-schema documentation is under `docs/`.
- Existing untracked `rendered/`, `scripts/`, and
  `docs/TIGRESS-NCR_suite_paper_planning.md` are user-owned work and are not
  part of this planning commit. The last file currently contains only `#`.

## First executable milestone

The first milestone is complete when all of the following are true:

- the private `tigress-learn-ism` repository exists and can be cloned;
- every tracked and untracked private file on the PRFM source branch appears
  in a reviewed migration inventory;
- private project text and scripts run from the new repository using PRFM and
  `tigress_ncr_tools` as dependencies;
- a clean PRFM integration branch contains only reusable public changes and
  passes its tests;
- a manuscript can build in the private repository without reading source
  files from `$HOME/PRFM/project`;
- no PRFM branch has been deleted yet.

Only after those checks should the old local and remote project branches be
removed, with explicit authorization for remote changes.

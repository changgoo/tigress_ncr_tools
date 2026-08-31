# Publication and scientific-development roadmap

## Program thesis

The program uses PHANGS-informed experimental design and TIGRESS-NCR
simulations to learn how the multiphase, star-forming ISM responds across
realistic galactic environments. It begins with a controlled core suite,
establishes its physical and statistical results, and expands to an
inference-ready suite only after the observables, parameters, and validation
requirements are clear.

The provisional public-facing phrase is **Learning the ISM with TIGRESS**.
This is a working name, not yet a permanent project, repository, or dataset
identifier.

## Three scientific components

### Component A: suite design and philosophy

Question: how should a finite simulation campaign sample the correlated local
environments observed by PHANGS while remaining useful for controlled theory
and future inference?

Core deliverables:

- public PRFM APIs for PHANGS data preparation and informed-prior sampling;
- a documented separation between simulation input fields and validation
  observables;
- the exact 32-model core design and TIGRESS input mapping;
- coverage, validity, and fairness diagnostics;
- an explicit statement of what the core suite can and cannot infer; and
- a reproducible link from design row to simulation model.

### Component B: core-suite results

Question: across the sampled environments at approximately fixed gas surface
density and fixed microphysics, how do star formation, vertical equilibrium,
phase structure, turbulence/magnetic support, and projected gas statistics
respond to environmental parameters?

Existing analysis families to synthesize rather than merely enumerate:

- run completeness and star-formation histories;
- face-on and edge-on evolution;
- phase fractions, scale heights, velocity dispersions, and effective support;
- PRFM pressure balance, dynamical-equilibrium weight, and feedback yields;
- gas, H I, and emission-measure column-density PDFs;
- shear-aware gas, H I, and emission-measure power spectra;
- cross-tracer and parameter correlation summaries; and
- robustness checks for the repaired `row0000`, time windows, resolution, and
  estimator choices.

The manuscript must identify a small set of primary claims and figures.
Correlation grids and full diagnostic inventories belong in appendices or the
data release unless they directly support those claims.

### Component C: inference and extended suites

Question: which environmental and uncertain-physics parameters can be
recovered from PHANGS-comparable observables, with calibrated uncertainty?

This component is deferred. The current 32-model core suite is a concept and
sensitivity study, not automatically an adequate inference training set.
Before committing to an inference paper, design an extended suite spanning the
targets and nuisance physics needed for interpolation and held-out validation.

Potential extension axes include gas surface density, metallicity,
dust-to-gas ratio/model, population-synthesis assumptions, IMF/feedback
strength or channel, and environmental variables not sufficiently covered by
the core suite. The choices must follow an identifiability study and compute
budget, not a desire to vary everything.

## Initial publication strategy

Start with one integrated working manuscript containing Components A and B.
Maintain section and figure IDs so it can be split without rewriting the
analysis pipeline.

Provisional integrated outline:

1. Motivation and program concept
2. PHANGS-informed environmental design
3. TIGRESS-NCR setup, model mapping, and numerical status
4. Analysis definitions and reproducibility
5. Core evolution and self-regulation
6. Vertical equilibrium, phase structure, and PRFM results
7. Projected structure across gas/H I/EM
8. Environmental dependence and physical synthesis
9. Limitations, public data products, and path to inference
10. Conclusions

### Paper split decision gate

Evaluate the split after the claims-to-figures matrix and first complete
integrated draft exist. Split into a **suite design/data paper** and a
**core-results paper** if at least two of these are true:

- the integrated main text materially exceeds the target journal length;
- design/release validation requires its own coherent methods and tables;
- core physics claims are obscured by setup detail;
- the data release can be cited independently before the results paper;
- coauthor/review timelines differ enough that one component blocks the other;
- each proposed paper has at least one independent central question and a
  nonduplicative figure set.

Do not split solely because many diagnostics exist. If the design story alone
cannot support a clear scientific conclusion, keep A and B integrated.

If split:

- Paper 1: design philosophy, PHANGS-informed sampling, numerical suite,
  validation, data products, and a concise physical overview.
- Paper 2: phase/vertical-equilibrium/PRFM and projected-structure results,
  referencing Paper 1 for design and release details.
- Paper 3 (later): inference using a separately justified extended suite.

## Claims-to-evidence workflow

Before polishing figures, create `plan/claims.yaml` in the private repository.
Each proposed claim should record:

- claim ID and one-sentence wording;
- whether it is descriptive, causal, comparative, or predictive;
- primary statistic and uncertainty definition;
- required models/time range and exclusions;
- primary figure/table IDs;
- robustness checks;
- competing interpretation or limitation; and
- manuscript destination.

No figure should enter the main paper merely because a script already makes
it. Main figures are selected because they resolve a claim or essential method
question.

## Figure-production workflow

```text
TIGRESS suite
  -> versioned generic reductions (tigress_ncr_tools)
  -> project-level joined tables/manifests (tigress-learn-ism)
  -> ignored candidate figures
  -> scientific and visual review
  -> confirmed final figure + manifest
  -> manuscript
```

Requirements for publication-quality figures:

- use a stable figure ID independent of its eventual paper number;
- read numerical archives/tables, not another figure;
- declare model selection, time interval, statistic, and uncertainty;
- use consistent labels, units, colors, phase/tracer naming, and panel order;
- remain legible at intended journal column width;
- include accessible color choices and distinguish important series without
  color alone where practical;
- provide a generation command and immutable code versions; and
- land in Git only after explicit confirmation.

Candidate figures live under ignored `build/figures/`. Confirmation should be
recorded in the manifest, after which the exact PDF/PNG is copied to
`papers/<paper>/figures/final/` and committed. Prefer vector PDF for plots and
high-resolution raster output only for image/map panels.

## Core-suite work packages

### WP1: freeze suite identity and quality

- reconcile the design CSV/YAML with all 32 model directories;
- record completion/restart/repair status, especially `row0000`;
- define authoritative analysis windows and explain any difference between
  200--600 Myr and 400--600 Myr products;
- freeze units, parameter names, and derived predictors; and
- produce the first release-ready model catalog.

Done when one manifest drives analysis, papers, and release curation.

### WP2: freeze generic reductions

- validate all current `tigress_ncr_tools` tests;
- assign schema versions to paper-consumed NPZ/CSV products;
- run generic analyses into a versioned output/staging location;
- capture commands, configs, environment, SHAs, warnings, and checksums; and
- distinguish authoritative archives from diagnostic plots/movies.

Done when another user can regenerate every numerical input to a paper figure.

### WP3: scientific synthesis

- reduce the current diagnostic inventory to a claims-to-evidence matrix;
- quantify robustness to time windows and relevant analysis choices;
- separate direct environmental trends from correlations mediated by SFR;
- state the limited causal scope of a correlated 32-point design; and
- identify what belongs in main text, appendix, and release only.

Done when the integrated draft has a coherent results narrative and no figure
exists without a stated role.

### WP4: manuscript and figure freeze

- build candidate figures through private project commands;
- confirm final figures one at a time with manifests;
- complete an integrated A+B draft;
- apply the paper split gate; and
- freeze a submission tag only after clean-clone build and coauthor review.

### WP5: release coordination

- map every paper-used dataset to a release object;
- ensure captions and tables cite stable release IDs, not HPC paths;
- synchronize nomenclature, versions, and caveats with the public site; and
- prepare the data availability statement and citation metadata.

## Inference readiness gate

Begin production inference work only when all are true:

- inference targets and nuisance parameters are explicit;
- PHANGS-comparable observables and their measurement process are defined;
- the core-suite sensitivity analysis identifies informative summaries and
  degeneracies;
- a simulation-based power/coverage study estimates the required design size;
- the extended design covers interpolation volume with held-out simulations;
- stochastic simulation variance and time sampling are represented;
- out-of-distribution detection and emulator uncertainty are planned;
- compute/storage budgets and failure replacement rules are approved; and
- the extended suite has its own versioned manifest and release strategy.

The inference paper should not reuse “Paper 2” numbering from the legacy PRFM
tree until the A+B split decision is made.

## Decisions still open

Record these in the private repository as dated decisions when resolved:

- final program and repository names;
- integrated versus split A+B publication;
- primary core-results claims and main-figure budget;
- authoritative time interval for each statistic;
- public-release payload tiers and licensing;
- observational products and summary statistics for inference; and
- extended-suite parameter ranges, design size, and compute allocation.

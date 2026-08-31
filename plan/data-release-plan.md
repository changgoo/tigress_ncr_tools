# Public data-release plan

## Scope and provisional identity

Create a public static-site repository provisionally named
`tigress-learn-ism-data`, under the umbrella **Learning the ISM with TIGRESS**.
The name is deliberately provisional and should be finalized before public
URLs, DOI metadata, or citations are circulated.

The Git repository is the documentation and metadata control plane. Versioned
payloads are curated and served through Globus collections. GitHub must not be
used as a substitute for the data store.

## Release principles

- A release is a fixed, checksummed manifest, not a mutable directory listing.
- Public identifiers and relative paths replace site-specific HPC paths.
- Raw/native data and derived analysis products are clearly distinguished.
- Every documented field has units, dimensions, cadence, and provenance.
- Users can verify downloads without trusting the transfer mechanism.
- Paper figures cite release/version identifiers and reconstructible products.
- The public site never exposes private PHANGS data, credentials, logs,
  collaborator notes, or unreviewed claims.
- Corrections create a new manifest/version and an explicit changelog; they do
  not silently replace released bytes.

## Proposed repository layout

```text
tigress-learn-ism-data/
  README.md
  LICENSE
  CITATION.cff
  mkdocs.yml                    # provisional static-site implementation
  docs/
    index.md
    quickstart.md
    suite-design.md
    model-catalog.md
    data-products.md
    formats/
    examples/
    access.md
    verification.md
    citations.md
    known-issues.md
    changelog.md
  catalog/
    models.csv
    products.csv
  manifests/
    v0.1.0/
      release.json
      files.csv
      checksums.sha256
  schemas/
    release.schema.json
    files.schema.json
    model-catalog.schema.json
  scripts/
    validate_manifest.py
    verify_download.py
  tests/
  assets/                        # small approved diagrams/thumbnails only
```

MkDocs is a provisional choice because it is simple, reviewable, and deploys
to GitHub Pages. The manifest/catalog formats are independent of the site
generator, so the presentation layer can change later.

## Data product tiers

Define tiers before copying bytes into a release staging area.

### Tier 0: catalog and reproducibility metadata

- model IDs and design-row mapping;
- physical/numerical input parameters and units;
- run status, time coverage, cadence, code version, and relevant repairs;
- source and analysis software versions;
- product inventory, sizes, checksums, schemas, and known issues.

This tier is mandatory for every release.

### Tier 1: compact analysis-ready products

- selected histories and z-profile summaries;
- model/suite summary tables;
- PRFM, phase, PDF, spectrum, and cross-tracer numerical archives;
- downsampled or selected maps needed for documented examples; and
- small reference outputs for reader validation.

Tier 1 should be sufficient to reproduce publication tables and most plots
without downloading full simulation snapshots.

### Tier 2: standard simulation products

- curated native histories, z-profiles, star particles, projections, slices,
  or other products selected for broad reuse;
- exact field and format documentation; and
- a declared time/cadence policy.

### Tier 3: bulk/restart/full-state products

- VTK/full-volume snapshots, restart files, or other high-volume material;
- released only when storage, transfer, documentation, and scientific value
  justify the cost; and
- potentially placed in a distinct Globus path/collection with clearer usage
  warnings.

The first public release need not include Tier 3. Decide inclusions by user
value and reproducibility requirements, not by what happens to exist on disk.

## Required model catalog fields

At minimum, one row per model:

```text
release_id
suite_id
model_id
design_row
Sigma_gas
Sigma_star
H_star
Omega
qshear
kappa
rho_star
Zgas
Zdust
resolution_pc
domain_size_pc
t_start_Myr
t_end_Myr
status
repair_notes
simulation_code_commit
input_parameter_record
```

Use machine-safe ASCII column names and document display notation separately.
Every numeric column needs units and a definition in the schema/site.

## Required file-manifest fields

One row per released object:

```text
release_id
object_id
model_id                 # blank for suite-level products
product_family
product_tier
relative_path
media_type
format_version
size_bytes
sha256
time_start_Myr
time_end_Myr
cadence_Myr
field_names
units_reference
source_object_ids
generator_repository
generator_commit
generator_command_id
license
known_issue_ids
```

For very large collections, `files.csv` may be partitioned by model/product,
but `release.json` must list every partition and checksum.

## Release workflow

### 1. Define the release

- choose semantic release ID (start with private/internal `v0.x`; reserve
  `v1.0.0` for the first stable public release);
- freeze the suite/model manifest and product-tier policy;
- decide licensing for code, metadata/docs, and data separately;
- assign maintainers and an issue/contact route; and
- document embargo/access constraints if any.

### 2. Build a staging tree

- create a dedicated release staging directory outside the live simulation
  suite and outside Git;
- copy only inventory-approved objects using manifest-relative paths;
- never curate by deleting files from the live suite;
- generate sizes and SHA-256 checksums after staging; and
- mark the staging tree read-only once validation begins.

### 3. Validate

- validate JSON/CSV manifests against committed schemas;
- prove all manifest paths exist and no extra payload is present;
- verify checksums from an independent pass;
- open/read representative files from every product family;
- verify model counts, time coverage, field lists, units, and expected cadence;
- scan text/metadata for private absolute paths, email/log content, tokens, and
  restricted PHANGS material;
- reproduce at least one documented example from a clean environment; and
- reconcile every paper-used data product with a public object ID.

### 4. Publish through Globus

- upload/copy the immutable staging tree to the release collection/path;
- configure least-privilege public read access;
- test access with an account that does not own the collection;
- verify a sample download against `checksums.sha256`;
- record the public collection UUID, path, and transfer URL in
  `release.json`; and
- do not reuse the same release path for corrected bytes.

### 5. Publish the site

- merge the exact approved manifest, catalog, schemas, docs, and Globus links;
- run link, schema, and site-build tests in CI;
- tag the site repository with the release ID;
- archive/cite the metadata repository through a DOI service if appropriate;
- publish release notes and known issues; and
- update paper data-availability text to the stable site and release ID.

### 6. Post-release maintenance

- triage user issues against a release ID/object ID;
- record documentation-only corrections separately from payload corrections;
- issue patch releases for corrected metadata or payloads;
- preserve old manifests and clearly mark superseded versions; and
- periodically test Globus links and checksum examples.

## Release gates

### Internal preview (`v0.1.0`)

- complete core model catalog;
- Tier 0 manifest and draft Tier 1 inventory;
- schemas validate;
- private Globus test collection works; and
- project members can reproduce one core paper figure from staged products.

### Coauthor/data-paper preview (`v0.9.0`)

- intended public Tier 1/2 payload frozen;
- file checksums independently verified;
- documentation covers every product family;
- known issues and repaired-row history are explicit;
- clean-account Globus transfer succeeds; and
- paper tables/figures map to release objects.

### Public stable release (`v1.0.0`)

- licensing, authorship, citation, and naming approved;
- site and Globus locations are stable;
- payload is immutable and restorable;
- all validation and privacy checks pass;
- DOI/citation plan is complete; and
- release announcement is coordinated with the relevant paper milestone.

## Relationship to the private project

The private repo owns release preparation scripts and the pre-publication
inventory. The public repo receives an approved export and may contain small,
general validation scripts. If an export script is needed, it should produce a
reviewable directory and report rejected private fields/paths; it should never
publish directly.

## Open release decisions

- final umbrella and repository names;
- GitHub owner/organization;
- data, metadata, and documentation licenses;
- exact Tier 1/2/3 contents and cadence;
- whether the first DOI represents the site metadata, a separate archive, or
  both;
- Globus collection ownership and long-term stewardship; and
- support/contact policy after release.

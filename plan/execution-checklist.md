# Cross-repository execution checklist

This is the recommended order of work. Check items in the repository where the
work is performed; this document is the baseline, not a live multi-repository
status tracker.

## Stage 1: establish safe destinations

- [ ] Approve the provisional private and public repository names/owner.
- [ ] Create private `tigress-learn-ism`; do not create it inside PRFM.
- [ ] Add scaffold, ignore rules, environment, local-path example, and tests.
- [ ] Push the scaffold privately and pass a fresh-clone installation test.
- [ ] Create (or reserve) public `tigress-learn-ism-data` but keep Pages
      unpublished until an approved preview exists.

Exit condition: there is a safe, backed-up destination before source cleanup.

## Stage 2: inventory before migration

- [ ] Record current PRFM source branch/SHA/status.
- [ ] Inventory and classify every `main...project/tigress-phangs-ili` change.
- [ ] Include the four known untracked PRFM project files.
- [ ] Record sizes/checksums for manuscript sources and candidate artifacts.
- [ ] Build the 32-model core suite manifest and design-row mapping.
- [ ] Record current PRFM and `tigress_ncr_tools` dependency SHAs.

Exit condition: every source item has a disposition and rollback copy.

## Stage 3: move private project work

- [ ] Import active plans/designs and preserve stale notes as labeled legacy.
- [ ] Import authoritative core design CSV/YAML and model mapping.
- [ ] Import and test project-specific scripts in the private namespace.
- [ ] Import TeX/bibliography sources for the integrated A+B manuscript.
- [ ] Place existing generated figures/PDFs in ignored migration review space.
- [ ] Confirm and commit final figures only through figure manifests.
- [ ] Make all suite/project paths configurable.
- [ ] Pass private tests and clean manuscript build.

Exit condition: ongoing project work no longer requires `$HOME/PRFM/project`.

## Stage 4: clean public PRFM integration

- [ ] Branch from current PRFM `main`.
- [ ] Port only reusable PHANGS integration/sampling APIs, docs, and tests.
- [ ] Remove fixed suite choices, private paths, and project narratives.
- [ ] Run the full PRFM tests and documentation build.
- [ ] Validate private project workflows against the clean PRFM commit.
- [ ] Merge/release the clean public change through the normal PRFM workflow.

Exit condition: public PRFM has the needed general capability and no private
project content.

## Stage 5: freeze core analysis inputs

- [ ] Resolve suite completeness, repairs, restarts, and authoritative windows.
- [ ] Version the generic archive schemas consumed by paper scripts.
- [ ] Run the complete `tigress_ncr_tools` test suite.
- [ ] Regenerate authoritative numerical reductions with captured configs/logs.
- [ ] Checksum outputs and record package/environment commits.
- [ ] Build project-level joined tables from documented archive fields.

Exit condition: paper numbers do not depend on mutable or undocumented files.

## Stage 6: develop the first publication(s)

- [ ] Write the claims-to-evidence matrix.
- [ ] Select the main-figure budget and appendix/release-only diagnostics.
- [ ] Complete an integrated suite-design + core-results draft.
- [ ] Generate candidates into ignored build space.
- [ ] Confirm and commit each final figure with provenance.
- [ ] Apply the integrated-versus-split decision gate.
- [ ] Run clean-clone manuscript build and coauthor review.

Exit condition: the paper structure is evidence-driven and reproducible.

## Stage 7: prepare public data preview

- [ ] Create schemas, model catalog, product catalog, and release manifest.
- [ ] Decide product tiers, cadence, licenses, and long-term Globus owner.
- [ ] Curate into a separate staging tree; do not alter the live suite.
- [ ] Validate files, checksums, formats, privacy, and representative reads.
- [ ] Publish/test a private Globus preview.
- [ ] Build/test the static site and download verification example.
- [ ] Demonstrate reproduction of a paper figure from staged Tier 1 products.

Exit condition: `v0.9.0` is ready for coauthor/data review.

## Stage 8: retire legacy branch

- [ ] Reconcile the migration inventory with private and public destinations.
- [ ] Verify fresh clones of both destinations.
- [ ] Notify collaborators of the cutover.
- [ ] Obtain explicit approval to delete the old remote branch.
- [ ] Delete local and remote `project/tigress-phangs-ili` separately.
- [ ] Record final source SHA, retirement date, and replacement locations.

Exit condition: PRFM no longer hosts the private branch and recovery remains
possible through private backups and the new repository.

## Stage 9: inference decision and extended-suite design

- [ ] Finish the core sensitivity/identifiability synthesis.
- [ ] Define inference targets, nuisance parameters, and observables.
- [ ] Quantify stochastic/time-sampling variance.
- [ ] Estimate training, validation, and held-out simulation requirements.
- [ ] Choose extension axes/ranges and active-learning strategy if applicable.
- [ ] Approve compute/storage budget and failure replacement policy.
- [ ] Create a separate versioned extended-suite design and release plan.

Exit condition: new simulations have a defensible inference role; inference is
not being attempted merely because the core suite exists.

## Validation command baseline

Run commands from clean environments where practical and record exact versions.

```bash
# tigress_ncr_tools
python3 -m pip install -e .
python3 -m pytest

# PRFM (adapt to its maintained environment/tooling)
python3 -m pytest

# private project
python3 -m pytest
# Then run the repository's documented manuscript build command.

# public release site
python3 scripts/validate_manifest.py manifests/<version>/release.json
# Then run the chosen static-site build and link checker.
```

Do not encode `/tigress/changgoo/anvil/TIGRESS-NCR-suite` in committed code.
Supply it through ignored local configuration or an environment-specific
command argument, while manifests use stable suite/model/object IDs.

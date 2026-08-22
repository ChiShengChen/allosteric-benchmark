# Provenance

This directory is a **verbatim vendored copy** of

    https://github.com/leo07010/allosteric-dataset-pipeline
    commit 4082cb1d986c5facafba7c2a75cfa982e7f9b728 (2026-08-22)

brought in whole so this repository runs standalone, as opposed to referencing it.

## What was copied

Everything the upstream repository contains: `pipeline/` (five build scripts),
`metadata/` (manifest and summary — identifier-level only), `LICENSE`, and the
upstream README, renamed to `README.upstream.md` so it is not mistaken for ours.

**No pipeline code was modified.** Our adapter lives outside this directory, in
[`../../scripts/adapt_allobench.py`](../../scripts/adapt_allobench.py), so that this
copy stays diffable against upstream.

## Licence

Upstream is **MIT**, Copyright (c) 2026 leo07010. `LICENSE` is preserved verbatim
and applies to everything in this directory. Our own MIT licence covers the rest of
the repository.

## What is *not* here, and cannot be

The upstream repository ships **no ASD- or AlloBench-derived residue annotations and
no structures** — deliberately, because those may not be redistributed under
AlloBench/ASD terms. That constraint transfers with the code: this directory is the
machinery that rebuilds the dataset, not the dataset.

Running it needs `AlloBench.csv`, obtained separately under its own terms, plus a
PDB download (~2 GB). So "standalone" here means *no dependency on the upstream
repository*, not *no external data*. That distinction cannot be engineered away.

## Why it was brought in

Sample size is this benchmark's binding constraint. Our curated set is 97 targets;
the literature survey in [`../../docs/ai-model-landscape.md`](../../docs/ai-model-landscape.md)
measured the field's standard training sets at 90–235 proteins. This pipeline builds
1,439 samples over 327 unique UniProt accessions, with the active-site annotation our
seeded formulation requires present in every sample.

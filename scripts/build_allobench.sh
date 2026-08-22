#!/bin/bash
# One command for the whole AlloBench route: filter -> download -> build -> verify
# -> convert into this repository's schema.
#
# The pipeline stages are the vendored upstream code, run unmodified
# (external/allobench-pipeline/, MIT, see its PROVENANCE.md). Only the last stage
# is ours.
#
# The one thing this cannot do for you: AlloBench.csv is not redistributable, so
# you supply it. Everything after that is automatic.
#
#   ALLOBENCH_CSV=/path/to/AlloBench.csv bash scripts/build_allobench.sh
#
# Resumable at every stage -- the download skips files it already has, and the
# build skips samples already written.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/external/allobench-pipeline"
export REPO_ROOT="$VENDOR"
export FILTERED_JSON="${FILTERED_JSON:-$VENDOR/metadata/allobench_filtered.json}"
export PDB_DIR="${PDB_DIR:-$VENDOR/data/allobench_pdbs}"
export OUT_DIR="${OUT_DIR:-$VENDOR/data/processed}"
export MANIFEST_OUT="${MANIFEST_OUT:-$VENDOR/metadata/manifest.built.json}"

if [ -z "${ALLOBENCH_CSV:-}" ]; then
  cat <<'MSG'
ALLOBENCH_CSV is not set.

AlloBench.csv may not be redistributed under its own terms, so it is not in this
repository and cannot be fetched automatically. Obtain it, then:

    ALLOBENCH_CSV=/path/to/AlloBench.csv bash scripts/build_allobench.sh

To check the conversion code without any licensed data present:

    python3 scripts/adapt_allobench.py --selftest
MSG
  exit 1
fi
export ALLOBENCH_CSV

echo "==> 1/5  filter AlloBench rows into an entry list"
python3 "$VENDOR/pipeline/filter_allobench.py"

echo "==> 2/5  download structures from RCSB (~2 GB, resumable)"
bash "$VENDOR/pipeline/download_allobench.sh"

echo "==> 3/5  build labelled samples, manifest and UniProt-grouped folds"
python3 "$VENDOR/pipeline/build_dataset_v2.py"

echo "==> 4/5  verify (upstream's own consistency checks)"
python3 "$VENDOR/pipeline/verify_dataset.py" || {
  echo "!! upstream verification failed — stopping before conversion." >&2
  exit 1
}

echo "==> 5/5  convert into this repository's schema"
python3 "$ROOT/scripts/adapt_allobench.py" --src "$OUT_DIR"

cat <<'MSG'

Done. The converted set is in data/targets_allobench/.

Three things that are not optional when you use it:
  * coordinates are CA, not CB — methods tuned on CB (ALPS RADIUS = 12.0) must be
    re-tuned, holding out identity AND size (README 10.5)
  * labels are "4 A heavy-atom to modulator", not the expert annotation the curated
    set uses — report the two sets separately, never pooled (README 10)
  * the folds carried into each npz are UniProt-grouped; use them rather than
    re-splitting, or you lose the family declustering
MSG

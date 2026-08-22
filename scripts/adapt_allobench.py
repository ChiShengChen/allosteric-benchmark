#!/usr/bin/env python
"""Convert the vendored AlloBench pipeline's output into this repository's schema.

The pipeline lives verbatim in `external/allobench-pipeline/` (MIT, see its
PROVENANCE.md). It writes one npz per sample in its own schema; every method and
script here expects ours. This is the only new code the integration needed, and it
lives outside the vendored tree so that copy stays diffable against upstream.

    theirs                          ours
    coords  (N,3) float32   ->      cb      (N,3)      see the warning below
    active_site_mask (N,)   ->      anchor  (A,)       nonzero indices
    allo_labels (N,)        ->      y       (N,)
    resnums (N,)            ->      resnums (N,)
    meta JSON               ->      chain_id (N,), uniprot, pdb, modulator
    manifest folds          ->      fold    scalar     UniProt-grouped, carried over

**The coordinates are C-alpha, ours are C-beta.** They go into the `cb` slot because
that is the one-point-per-residue field every loader here reads, but the file also
carries `coord_type` so nothing downstream can quietly assume C-beta. This matters
concretely: `methods/alps.py` has `RADIUS = 12.0` tuned on C-beta contact geometry,
and C-alpha--C-alpha distances for the same residue pair run systematically
different. Any method applied to this set must be re-tuned, and the re-tuning has to
hold out identity *and* size -- section 10.5 of the README records what happens when
only identity is held out.

**The labels are not our labels.** Theirs is "within 4 A heavy-atom of the
allosteric modulator"; our curated set is expert annotation. Section 10 of the README
records three published conclusions that reversed when the label construction
changed. So the two sets are built side by side and evaluated separately. Pooling
them would make it impossible to tell whether a change came from more data or from a
different definition of the answer.

**Both sets are holo.** Theirs states it as a limitation; ours says the same thing in
README section 12.4. This is a shared, already-disclosed constraint, not something
the integration introduces -- but it is also not something the integration fixes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from methods.common import distal_nonanchor_mask                     # noqa: E402

VENDOR = os.path.join(HERE, "external", "allobench-pipeline")
MIN_SITE = 3


def convert(path, fold=None):
    """One sample. Returns None with a reason if it cannot be used here."""
    z = np.load(path, allow_pickle=True)
    for k in ("coords", "allo_labels", "active_site_mask", "resnums"):
        if k not in z:
            return None, f"missing {k}"
    cb = np.asarray(z["coords"], float)
    y = np.asarray(z["allo_labels"], int)
    anchor = np.nonzero(np.asarray(z["active_site_mask"], int))[0]
    if len(cb) < 50:
        return None, "chain too short"
    if len(anchor) < MIN_SITE:
        return None, "no usable active site (the seeded formulation needs a seed)"
    if y.sum() == 0:
        return None, "no positive labels"

    # our evaluation only scores residues that are distal and non-anchor; a sample
    # with no positive there cannot contribute, so drop it here rather than let it
    # become a silent nan downstream
    pool = distal_nonanchor_mask(cb, anchor, 8.0)
    if (pool & (y == 1)).sum() == 0:
        return None, "no positive survives the distal non-anchor pool"

    meta = {}
    if "meta" in z:
        try:
            meta = json.loads(str(z["meta"]))
        except Exception:
            meta = {}
    chain = str(meta.get("chain", "A"))
    rec = dict(cb=cb.astype(np.float32),
               anchor=anchor.astype(np.int64),
               y=y.astype(np.int64),
               resnums=np.asarray(z["resnums"], int),
               chain_id=np.asarray([chain] * len(cb), dtype="U4"),
               coord_type=np.asarray("CA"),
               label_rule=np.asarray("4A heavy-atom to allosteric modulator"),
               uniprot=np.asarray(str(meta.get("uniprot", ""))),
               pdb=np.asarray(str(meta.get("pdb", ""))),
               modulator=np.asarray(str(meta.get("modulator", ""))))
    if fold is not None:
        rec["fold"] = np.asarray(int(fold))
    return rec, None


def load_folds():
    """UniProt-grouped fold assignment from the vendored manifest, keyed by sample."""
    p = os.path.join(VENDOR, "metadata", "manifest.json")
    if not os.path.exists(p):
        return {}
    man = json.load(open(p))
    return {k: int(f) for f, keys in man.get("folds", {}).items() for k in keys}


def selftest():
    """Round-trip a synthetic sample in their schema, with no AlloBench data present.

    The point is to verify the conversion path itself -- field mapping, anchor
    derivation, pool filtering -- which is testable without the licensed data.
    """
    import tempfile
    rng = np.random.default_rng(0)
    n = 200
    cb = rng.normal(0, 14, (n, 3)).astype(np.float32)
    mask = np.zeros(n, dtype=np.int8); mask[:8] = 1              # active site
    y = np.zeros(n, dtype=np.int8)
    far = np.argsort(-np.linalg.norm(cb - cb[:8].mean(0), axis=1))[:10]
    y[far] = 1                                                    # distal positives
    d = tempfile.mkdtemp()
    p = os.path.join(d, "1ABC_A_LIG_deadbe.npz")
    np.savez(p, coords=cb, resnums=np.arange(1, n + 1, dtype=np.int32),
             allo_labels=y, active_site_mask=mask,
             meta=json.dumps({"pdb": "1ABC", "chain": "A", "uniprot": "P00000",
                              "modulator": "LIG"}))
    rec, why = convert(p, fold=3)
    assert rec is not None, f"conversion failed: {why}"
    assert rec["cb"].shape == (n, 3) and rec["y"].shape == (n,)
    assert len(rec["anchor"]) == 8 and set(rec["anchor"]) == set(range(8))
    assert str(rec["coord_type"]) == "CA" and int(rec["fold"]) == 3
    assert str(rec["uniprot"]) == "P00000"
    out = os.path.join(d, "out.npz"); np.savez_compressed(out, **rec)
    z = np.load(out, allow_pickle=True)
    assert z["cb"].shape == (n, 3) and int(z["y"].sum()) == 10
    print("selftest OK — field mapping, anchor derivation, pool filter, round-trip")
    # and the negative path
    bad = os.path.join(d, "bad.npz")
    np.savez(bad, coords=cb, resnums=np.arange(n, dtype=np.int32),
             allo_labels=np.zeros(n, dtype=np.int8), active_site_mask=mask,
             meta="{}")
    r2, why2 = convert(bad)
    assert r2 is None and "no positive" in why2, why2
    print(f"selftest OK — unusable samples are rejected with a reason ({why2!r})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(VENDOR, "data", "processed"),
                    help="the pipeline's output directory")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "targets_allobench"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if not os.path.isdir(a.src):
        print(f"no pipeline output at {a.src}\n\n"
              f"The vendored pipeline has to be run first, and it needs AlloBench.csv\n"
              f"obtained separately under its own terms — see\n"
              f"  external/allobench-pipeline/PROVENANCE.md\n"
              f"  external/allobench-pipeline/README.upstream.md\n\n"
              f"Run `python3 scripts/adapt_allobench.py --selftest` to check this\n"
              f"converter without any licensed data present.")
        return 1

    os.makedirs(a.out, exist_ok=True)
    folds = load_folds()
    kept, reasons = 0, {}
    for f in sorted(glob.glob(os.path.join(a.src, "*.npz"))):
        key = os.path.basename(f)[:-4]
        rec, why = convert(f, folds.get(key))
        if rec is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        np.savez_compressed(os.path.join(a.out, f"{key}.npz"), **rec)
        kept += 1

    print(f"converted {kept} samples -> {a.out}")
    if reasons:
        print("dropped:")
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {why}")
    print("\nreminders that are not optional:")
    print("  * coordinates are CA, not CB — re-tune before comparing to curated numbers")
    print("  * labels are 4 A to modulator, not expert annotation — report separately")
    print("  * folds carried over are UniProt-grouped; use them rather than re-splitting")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

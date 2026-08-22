#!/usr/bin/env python
"""Build protein-level NEGATIVES: structures with no annotated allosteric site.

Every target in this repository has a known allosteric site by construction, so
every result so far answers one question: *given a protein that has a site, can
the method rank it highly?* The question nobody here has asked is the one a user
actually faces: **given a protein with no allosteric site, does the method say so,
or does it confidently name five residues anyway?**

A method that always returns five residues scores well on a positives-only
benchmark and sends an experimental group after sites that do not exist.

Negatives are built the way the ALLO benchmarking paper built its 87
orthosteric-only proteins: a structure carrying a cofactor (which defines the
active site, exactly as for the positives) and **no drug-like ligand at all**, so
no allosteric site is annotated anywhere in it.

⚠️ This is absence of evidence, not evidence of absence. "No allosteric modulator
has been crystallised with this protein" is not "this protein has no allosteric
site". The negatives are therefore a *lower bound* on the false-positive problem:
any separation measured against them is real, but a failure to separate could
partly reflect genuine sites we have mislabelled as negative.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from build_dataset import (COFACTORS, DRUGLIKE_ELEMS, MIN_HEAVY,          # noqa: E402
                           MIN_SITE, chain_arrays, contacts, fetch, parse,
                           uniprot_of)

OUT = os.path.join(HERE, "data", "targets_negative")
MIN_LEN, MAX_LEN = 150, 1200


def build_one(pdb_id):
    """Keep only if a cofactor site exists and NO drug-like ligand is present."""
    text = fetch(pdb_id)
    if not text:
        return None
    residues, order, ligands = parse(text)
    if not residues:
        return None

    by_chain = {}
    for (chain, resname, rkey), d in ligands.items():
        xyz = np.asarray(d["xyz"], float)
        if len(xyz):
            by_chain.setdefault(chain, []).append((resname, xyz, set(d["elem"])))

    for chain in sorted(residues):
        arr = chain_arrays(residues, order, chain)
        if arr is None:
            continue
        cb, heavy, resnums = arr
        if not (MIN_LEN <= len(cb) <= MAX_LEN):
            continue
        ligs = by_chain.get(chain, [])
        cofs = [l for l in ligs if l[0] in COFACTORS]
        drugs = [l for l in ligs if l[0] not in COFACTORS and len(l[1]) >= MIN_HEAVY
                 and (l[2] & DRUGLIKE_ELEMS)]
        if not cofs or drugs:            # need an active site, and no modulator
            continue
        anchor = np.unique(np.concatenate([contacts(heavy, l[1]) for l in cofs]))
        if len(anchor) < MIN_SITE:
            continue
        return dict(pdb=pdb_id, chain=chain, n=len(cb), cb=cb, anchor=anchor,
                    resnums=resnums, n_anchor=len(anchor))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(HERE, "data", "candidate_ids_b.json"))
    ap.add_argument("--want", type=int, default=90)
    ap.add_argument("--limit", type=int, default=4000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # never reuse a UniProt that appears among the curated positives
    pos_unp = set()
    man = os.path.join(HERE, "data", "manifest_b.json")
    if os.path.exists(man):
        for r in json.load(open(man)):
            pos_unp |= set(r.get("uniprot", []))

    ids = json.load(open(a.candidates))
    kept, seen = [], set(pos_unp)
    done = {f.split("_")[0] for f in os.listdir(OUT)} if os.path.isdir(OUT) else set()
    for pdb_id in ids[:a.limit]:
        if len(kept) >= a.want:
            break
        if pdb_id in done:
            continue
        try:
            r = build_one(pdb_id)
        except Exception:
            r = None
        if r is None:
            continue
        unp = uniprot_of(pdb_id)
        if unp & seen:
            continue
        seen |= unp
        y = np.zeros(r["n"], dtype=int)          # no allosteric site: y is all zero
        np.savez_compressed(os.path.join(OUT, f"{r['pdb']}_{r['chain']}.npz"),
                            cb=r["cb"], anchor=r["anchor"], y=y, resnums=r["resnums"])
        kept.append({k: r[k] for k in ("pdb", "chain", "n", "n_anchor")})
        print(f"[{len(kept):3d}] {r['pdb']}_{r['chain']} N={r['n']:4d} "
              f"anchor={r['n_anchor']:3d}", flush=True)
    json.dump(kept, open(os.path.join(HERE, "data", "manifest_negative.json"), "w"), indent=1)
    print(f"\nkept {len(kept)} negatives -> {OUT}")


if __name__ == "__main__":
    main()

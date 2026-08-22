#!/usr/bin/env python
"""Rebuild positives and negatives through one identical procedure.

Section 10.1 found the protein-level experiment was not measuring what it claimed.
Two procedures differed between the classes, and every method is seeded at the
anchor, so every score inherited both:

1. **Anchor definition.** Positives took the active site from the curated table
   (expert annotation, tight: 4.2% of residues); negatives took it from a 4.5 Å
   cofactor shell (looser: 7.2%).
2. **Chain handling.** Positives kept every chain of the deposited structure;
   negatives kept one. That alone explains much of the size gap the earlier
   matching was trying to correct for — median N 532 against 327.

This builds both classes the same way: every protein chain, anchor = residues
within 4.5 Å of a cofactor ligand. The only thing that differs is the label —
whether a distal drug-like ligand (positives, taken from the curated allosteric
annotation) is present.

The cost is that positives now need a cofactor in the structure, which not every
curated entry has; those are dropped rather than mixed in under a different rule.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from build_dataset import COFACTORS, DRUGLIKE_ELEMS, MIN_HEAVY, contacts   # noqa: E402
from build_dataset_curated import (fetch, parse_allosteric,                # noqa: E402
                                   parse_structure)
from build_dataset import parse as parse_pdb                               # noqa: E402

CONTACT = 4.5
MIN_SITE = 3
MIN_LEN, MAX_LEN = 150, 1200


def heavy_atoms_by_residue(text, keys):
    """Heavy-atom coordinates per (chain, resnum), aligned to `keys`."""
    by = {k: [] for k in keys}
    for line in text.splitlines():
        if line[:6] == "ENDMDL":
            break
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        if line[16] not in (" ", "A"):
            continue
        elem = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        if elem in ("H", "D"):
            continue
        chain = line[21].strip() or "A"
        try:
            key = (chain, int(line[22:26]))
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if key in by:
            by[key].append(xyz)
    return [np.asarray(by[k], float) if by[k] else np.zeros((0, 3)) for k in keys]


def build(pdb_id, allo_field=None):
    """One protein, both classes, identical procedure."""
    text = fetch(pdb_id)
    if not text:
        return None
    cb, keys = parse_structure(text)
    if not (MIN_LEN <= len(cb) <= MAX_LEN):
        return None
    _res, _order, ligands = parse_pdb(text)

    cof = [np.asarray(d["xyz"], float) for (c, rn, rk), d in ligands.items()
           if rn in COFACTORS and len(d["xyz"])]
    if not cof:
        return None
    heavy = heavy_atoms_by_residue(text, keys)
    anchor = np.unique(np.concatenate(
        [contacts(heavy, x, CONTACT) for x in cof]))
    if len(anchor) < MIN_SITE:
        return None

    y = np.zeros(len(cb), dtype=int)
    if allo_field is not None:
        index = {k: i for i, k in enumerate(keys)}
        idx = sorted(index[k] for k in parse_allosteric(allo_field) if k in index)
        idx = [i for i in idx if i not in set(anchor.tolist())]
        if len(idx) < MIN_SITE:
            return None
        y[idx] = 1
    else:
        drugs = [d for (c, rn, rk), d in ligands.items()
                 if rn not in COFACTORS and len(d["xyz"]) >= MIN_HEAVY
                 and (set(d["elem"]) & DRUGLIKE_ELEMS)]
        if drugs:                       # a modulator is present: not a negative
            return None
    return dict(cb=cb, anchor=anchor, y=y,
                resnums=np.asarray([k[1] for k in keys], int),
                chain_id=np.asarray([k[0] for k in keys], dtype="U4"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.path.join(HERE, "data", "allo_tableS2.csv"))
    ap.add_argument("--neg-ids", default=os.path.join(HERE, "data", "manifest_negative.json"))
    a = ap.parse_args()
    import json

    pos_dir = os.path.join(HERE, "data", "matched_pos")
    neg_dir = os.path.join(HERE, "data", "matched_neg")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    npos = nneg = 0
    for r in csv.DictReader(open(a.table)):
        tag = r["pdb"].strip()
        out = os.path.join(pos_dir, f"{tag}.npz")
        if os.path.exists(out):
            npos += 1
            continue
        try:
            rec = build(tag.split("_")[0], r["allo_site_residues"])
        except Exception:
            rec = None
        if rec is None:
            continue
        np.savez_compressed(out, **rec)
        npos += 1
        print(f"  pos {tag} N={len(rec['cb'])} anchor={len(rec['anchor'])} "
              f"pos={int(rec['y'].sum())}", flush=True)

    for e in json.load(open(a.neg_ids)):
        pdb = e["pdb"]
        out = os.path.join(neg_dir, f"{pdb}.npz")
        if os.path.exists(out):
            nneg += 1
            continue
        try:
            rec = build(pdb, None)
        except Exception:
            rec = None
        if rec is None:
            continue
        np.savez_compressed(out, **rec)
        nneg += 1
        print(f"  neg {pdb} N={len(rec['cb'])} anchor={len(rec['anchor'])}", flush=True)

    print(f"\nmatched sets: {npos} positives, {nneg} negatives — "
          f"same parsing, same anchor rule")


if __name__ == "__main__":
    main()

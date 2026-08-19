#!/usr/bin/env python
"""Build a benchmark from *curated* allosteric annotations, not geometric proxies.

Every other target set here labels a site by "a drug-like molecule crystallised
there, far from the catalytic site" — the field's operational definition, but not
an experimentally curated allosteric site. That caveat qualifies every conclusion
in this repository.

This builder removes it. Supplementary Table S2 of

    Wu, Amor, Schaub, Barahona et al., "Prediction of allosteric sites and
    signaling: Insights from benchmarking datasets", Patterns (2021),
    doi:10.1016/j.patter.2021.100408   (open access; supplementary via Europe PMC)

lists, for 118 proteins drawn from ASBench with sites taken from ASD Release 4.10,
both the **allosteric site residues** and the **active site residues**, by chain and
residue number. That is exactly the (anchor, y) pair this benchmark needs, curated
by the people who built the databases rather than inferred by us.

The two annotation columns use different formats, both handled here:
    allosteric   "ASP14 A,ASN24 A,..."   resname + resnum, space, chain
    active site  "A41,A43,..."           chain + resnum

Output: data/targets_curated/<pdb>.npz with cb, anchor, y, resnums, chain_id —
the same format as every other target set, so all existing evaluation code runs on
it unchanged.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys
import urllib.request
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "targets_curated")
CACHE = os.path.join(HERE, "data", "pdb_cache")
TABLE = os.path.join(HERE, "data", "allo_tableS2.csv")

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE"}
MIN_LEN, MAX_LEN = 60, 1400
MIN_SITE = 3


def fetch(pdb_id):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{pdb_id.upper()}.pdb.gz")
    if not os.path.exists(path):
        try:
            req = urllib.request.Request(
                f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb.gz",
                headers={"User-Agent": "qasc-plus/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
        except Exception:
            return None
    try:
        with gzip.open(path, "rt", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def parse_structure(text):
    """Per-residue Cbeta (Calpha fallback) keyed by (chain, resnum)."""
    res = {}
    order = []
    for line in text.splitlines():
        if line[:6] == "ENDMDL":
            break
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        if line[16] not in (" ", "A"):
            continue
        rname = line[17:20].strip()
        if rname not in AA3:
            continue
        atom = line[12:16].strip()
        if atom not in ("CA", "CB"):
            continue
        chain = line[21].strip() or "A"
        try:
            rnum = int(line[22:26])
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        key = (chain, rnum)
        if key not in res:
            res[key] = {}
            order.append(key)
        res[key][atom] = xyz
    cb, keys = [], []
    for k in order:
        pos = res[k].get("CB") or res[k].get("CA")
        if pos is not None:
            cb.append(pos)
            keys.append(k)
    return np.asarray(cb, float), keys


def parse_allosteric(field):
    """'ASP14 A,ASN24 A' -> {(chain, resnum)}"""
    out = set()
    for tok in str(field).split(","):
        m = re.match(r"\s*([A-Z]{3})\s*(-?\d+)\s+([A-Za-z0-9])\s*$", tok)
        if m:
            out.add((m.group(3), int(m.group(2))))
    return out


def parse_active(field):
    """'A41,A43' -> {(chain, resnum)}"""
    out = set()
    for tok in str(field).split(","):
        m = re.match(r"\s*([A-Za-z0-9])(-?\d+)\s*$", tok)
        if m:
            out.add((m.group(1), int(m.group(2))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = list(csv.DictReader(open(a.table)))
    if a.limit:
        rows = rows[:a.limit]
    kept, skipped = [], defaultdict(int)

    for r in rows:
        pdb = r["pdb"].split("_")[0].strip()
        path = os.path.join(OUT, f"{pdb}.npz")
        if os.path.exists(path):
            skipped["already built"] += 1
            continue
        text = fetch(pdb)
        if not text:
            skipped["no structure"] += 1
            continue
        cb, keys = parse_structure(text)
        if not (MIN_LEN <= len(cb) <= MAX_LEN):
            skipped["size"] += 1
            continue
        index = {k: i for i, k in enumerate(keys)}

        allo = parse_allosteric(r["allo_site_residues"])
        act = parse_active(r["active_site_residues"])
        y_idx = sorted(index[k] for k in allo if k in index)
        a_idx = sorted(index[k] for k in act if k in index)
        if len(y_idx) < MIN_SITE or len(a_idx) < MIN_SITE:
            skipped["annotation did not map"] += 1
            continue
        overlap = set(y_idx) & set(a_idx)
        if overlap:                      # curated sites should be disjoint
            y_idx = [i for i in y_idx if i not in overlap]
            if len(y_idx) < MIN_SITE:
                skipped["sites overlap"] += 1
                continue

        y = np.zeros(len(cb), dtype=int)
        y[y_idx] = 1
        np.savez_compressed(
            path, cb=cb, anchor=np.asarray(a_idx, int), y=y,
            resnums=np.asarray([k[1] for k in keys], int),
            chain_id=np.asarray([k[0] for k in keys], dtype="U4"))
        kept.append(dict(pdb=pdb, n=len(cb), n_anchor=len(a_idx), n_pos=int(y.sum()),
                         mapped_allo=f"{len(y_idx)}/{len(allo)}",
                         mapped_act=f"{len(a_idx)}/{len(act)}"))
        print(f"[{len(kept):3d}] {pdb} N={len(cb):4d} anchor={len(a_idx):3d} "
              f"pos={int(y.sum()):3d}  allo mapped {len(y_idx)}/{len(allo)}", flush=True)

    print(f"\nkept {len(kept)} of {len(rows)} table rows -> {OUT}")
    for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  skipped, {k}: {v}")


if __name__ == "__main__":
    main()

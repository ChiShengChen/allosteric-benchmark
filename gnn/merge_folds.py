#!/usr/bin/env python
"""Merge the UniProt accession and its grouped fold assignment into a graph cache.

`data.py` predates the AlloBench route and does not carry `fold`. Rebuilding the
cache to add one scalar per target would cost another four hours of eigensolves, so
this reads the field back out of the converted targets and writes it in.

Run once after `data.py --targets data/targets_allobench`.
"""
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=os.path.join(HERE, "gnn", "graphs_allobench.npz"))
ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_allobench"))
a = ap.parse_args()

folds, unis = {}, {}
for f in glob.glob(os.path.join(a.targets, "*.npz")):
    z = np.load(f, allow_pickle=True)
    key = os.path.basename(f)[:-4]
    if "fold" in z.files:
        folds[key] = int(z["fold"])
    if "uniprot" in z.files:
        unis[key] = str(z["uniprot"])

data = list(np.load(a.cache, allow_pickle=True)["data"])
hit = 0
for r in data:
    if r["t"] in folds:
        r["fold"] = folds[r["t"]]
        hit += 1
    if r["t"] in unis:
        r["uniprot"] = unis[r["t"]]
np.savez_compressed(a.cache, data=np.array(data, dtype=object))
print(f"merged folds into {hit}/{len(data)} targets")
if hit != len(data):
    print("!! some targets have no fold — run.py will fall back to re-splitting, "
          "which loses the UniProt grouping. Fix before trusting the numbers.")
else:
    u = sorted({r["fold"] for r in data})
    print(f"fold ids {u}, sizes " +
          ", ".join(str(sum(1 for r in data if r['fold'] == k)) for k in u))
    nu = sum(1 for r in data if "uniprot" in r)
    print(f"uniprot merged into {nu}/{len(data)} targets "
          f"({len({r.get('uniprot') for r in data})} distinct) — "
          f"lets run.py group the INNER validation split too")

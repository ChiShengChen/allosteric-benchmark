#!/usr/bin/env python
"""Graph tensors for the curated targets, plus the ALPS baseline on the same set.

The literature search (`docs/ai-model-landscape.md`) found that of every AI model
family applied to this task, exactly two run on our input signature -- Cbeta
coordinates and active-site indices, no sequence, no side chains, no MSA, no
trajectory. One of them is the elastic-network route we already built. The other,
a residue-graph GNN, had not been tried. This builds its input.

**What the node features deliberately exclude.** Distance to the active site is
not a base node feature. It is the confound section 10 of the main README spent
its length removing, and a learner handed it will reproduce it -- the learned
combiner did exactly that on proxy labels. Here the model gets the anchor as an
*indicator*, and has to propagate along the graph to discover anything about
reach. Whether that matters is testable rather than assumed: `--with-dist` adds
the distance channel back, and the two are compared.

Node features, all derivable from Cbeta alone:
    anchor    1.0 if the residue is in the active site, else 0.0
    degree    contact-graph degree, scaled
    burial    neighbours within 12 A, scaled
    (dist)    optional, rank-percentile distance to anchor -- the ablation

Edges are the 10 A contact graph, matching every other method in the repository,
with an RBF expansion of the edge length as the edge feature.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from methods.alps import alps_scores                                    # noqa: E402
from methods.common import (contact_graph, distal_nonanchor_mask,       # noqa: E402
                            min_dist_to_anchor, rank_percentile)

RADIUS = 10.0
BURIAL_R = 12.0
N_RBF = 16
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs.npz")


def rbf(d, n=N_RBF, lo=0.0, hi=RADIUS):
    """Smooth distance encoding -- a raw scalar length is a poor edge feature."""
    mu = np.linspace(lo, hi, n)
    gamma = n / (hi - lo)
    return np.exp(-gamma * (d[:, None] - mu[None, :]) ** 2)


def build_one(cb, anchor, with_dist=False):
    A = contact_graph(cb, RADIUS)
    n = len(cb)
    src, dst = np.nonzero(A)
    dxyz = cb[src] - cb[dst]
    elen = np.linalg.norm(dxyz, axis=1)

    deg = A.sum(1).astype(float)
    D2 = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=2)
    burial = (D2 < BURIAL_R).sum(1).astype(float) - 1.0

    is_anchor = np.zeros(n)
    is_anchor[np.asarray(anchor, int)] = 1.0
    cols = [is_anchor, deg / 12.0, burial / 30.0]
    if with_dist:
        cols.append(rank_percentile(min_dist_to_anchor(cb, anchor)))
    return dict(x=np.column_stack(cols).astype(np.float32),
                src=src.astype(np.int64), dst=dst.astype(np.int64),
                eattr=rbf(elen).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_curated"))
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args()

    data = []
    if os.path.exists(a.cache):
        data = list(np.load(a.cache, allow_pickle=True)["data"])
    done = {r["t"] for r in data}

    for f in sorted(glob.glob(os.path.join(a.targets, "*.npz"))):
        name = os.path.basename(f).replace(".npz", "")
        if name in done:
            continue
        z = np.load(f)
        cb, anchor, y = z["cb"], z["anchor"], z["y"].astype(int)
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if y.sum() == 0 or (pool & (y == 1)).sum() == 0:
            continue                       # no positive is evaluable under the metric
        g = build_one(cb, anchor, with_dist=False)
        gd = build_one(cb, anchor, with_dist=True)
        rec = dict(t=name, y=y, pool=pool,
                   dist=min_dist_to_anchor(cb, anchor),
                   alps=alps_scores(cb, anchor, pool),
                   x=g["x"], xd=gd["x"], src=g["src"], dst=g["dst"],
                   eattr=g["eattr"])
        data.append(rec)
        np.savez_compressed(a.cache, data=np.array(data, dtype=object))
        print(f"  {name} N={len(cb)} E={len(g['src'])} anchor={int(g['x'][:,0].sum())} "
              f"pos={int(y.sum())} pos_in_pool={int((pool & (y==1)).sum())}", flush=True)

    tot = sum(len(r["y"]) for r in data)
    pos = sum(int((r["pool"] & (r["y"] == 1)).sum()) for r in data)
    inpool = sum(int(r["pool"].sum()) for r in data)
    print(f"\n{len(data)} targets, {tot} residues, {inpool} in pool, "
          f"{pos} evaluable positives ({100*pos/max(inpool,1):.1f}% of pool)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Re-estimate the noise floor on the AlloBench set, in both control variants.

Two reasons this cannot reuse the 0.4963 constant.

**Different data.** That number was estimated on the curated set, whose pool is
78,509 residues at 1.3% positive. The AlloBench route is 369,988 residues at 2.63%,
with a shorter median chain. A floor is a property of the data and the metric, not a
universal constant, and carrying one across is the same class of error this repository
already recorded: a reference value estimated in one setting, reused as a fixed number
in another.

**Different control.** `scripts/floor_and_tests.py` smooths the random field over the
contact graph before scoring it; `gnn/run.py` scores raw uniform noise. Smoothing
induces spatial correlation, so the two are not obviously the same null. Both are
estimated here rather than assumed equal.

Geometry is computed once per target and reused across seeds -- the original script
rebuilt the contact graph inside the seed loop, which at 1,042 targets would mean
26,050 rebuilds instead of 1,042.
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

from methods.common import (contact_graph, distal_nonanchor_mask,   # noqa: E402
                            min_dist_to_anchor, pocket_smooth, rank_percentile)
from partial_auc import stratified_auc                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_allobench"))
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.targets, "*.npz")))
    geom = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if y.sum() == 0 or (pool & (y == 1)).sum() == 0:
            continue
        geom.append((y, pool, min_dist_to_anchor(cb, anchor),
                     contact_graph(cb, 10.0), len(cb)))
    print(f"{len(geom)} evaluable targets, geometry cached", flush=True)

    out = {}
    for label in ("raw", "pocket-smoothed"):
        per_seed = []
        for seed in range(a.seeds):
            rng = np.random.default_rng(1000 + seed)
            vals = []
            for y, pool, dist, A, n in geom:
                s = rng.random(n)
                if label == "pocket-smoothed":
                    s = pocket_smooth(rank_percentile(s), A)
                auc, _ = stratified_auc(y, s, pool, dist, a.tol)
                vals.append(auc)
            per_seed.append(np.nanmean(vals))
        v = np.asarray(per_seed)
        out[label] = (v.mean(), v.std(), v.min(), v.max())
        print(f"{label:16s} mean {v.mean():.4f}  sd {v.std():.4f}  "
              f"range [{v.min():.4f}, {v.max():.4f}]", flush=True)

    r, p = out["raw"], out["pocket-smoothed"]
    print(f"\nsmoothing shifts the floor by {p[0]-r[0]:+.4f}")
    print(f"gnn/run.py scores a RAW random control, so its floor is {r[0]:.4f} "
          f"+/- {r[1]:.4f}")
    print(f"(the curated-set constant currently hard-coded there is 0.4963)")


if __name__ == "__main__":
    main()

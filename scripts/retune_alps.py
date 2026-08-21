#!/usr/bin/env python
"""Re-tune ALPS on curated labels — with a split, because the last tuning had none.

ALPS's hyperparameters (radius 10 A, K = 3 modes, kappa = 1.0) were selected on an
11-target proxy-labelled set using plain permutation significance. Sections 9.1-9.4
then established that proxy labels are dominated by distance and that plain AUC
inherits that confound, so those parameters were chosen against an artefact. The
distance z-score in particular was credited with lifting the method from 82% to 91%,
and on curated labels it turns out to be neutral.

This re-tunes on curated annotations with the distance-stratified metric, and does
the thing the original tuning did not: **splits the targets**. Odd-indexed targets
tune, even-indexed targets report, and then the split is reversed so neither half is
privileged. A configuration that only wins on the half it was chosen on has not won.

Cost note: K selects how many of the lowest eigenvalues enter the score, so the
lowest ten are computed once per (radius, kappa) and every K is read off for free.
That turns a 36-setting grid into 9 heavy passes.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import warnings

import numpy as np
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from methods.alps import _low_eigs, distance_zscore                       # noqa: E402
from methods.common import (contact_graph, distal_nonanchor_mask,         # noqa: E402
                            laplacian, min_dist_to_anchor, pocket_smooth,
                            rank_percentile)
from partial_auc import stratified_auc                                    # noqa: E402

RADII = (8.0, 10.0, 12.0)
KAPPAS = (0.5, 1.0, 2.0)
KS = (2, 3, 5, 10)
N_KEEP = 10


def eigen_table(cb, radius, kappa, cutoff=10.0):
    """Lowest N_KEEP non-zero eigenvalues, baseline and per-residue perturbed."""
    A = contact_graph(cb, cutoff)
    sparse = len(cb) > 400
    base = _low_eigs(laplacian(A), N_KEEP, sparse)
    D = cdist(cb, cb)
    out = np.zeros((len(cb), N_KEEP))
    for i in range(len(cb)):
        nb = np.where(D[i] <= radius)[0]
        W = A.copy()
        sub = np.ix_(nb, nb)
        W[sub] = W[sub] * (1.0 + kappa)
        p = _low_eigs(laplacian(W), N_KEEP, sparse)
        m = min(len(p), N_KEEP)
        out[i, :m] = p[:m]
    return base, out, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_curated"))
    ap.add_argument("--max-n", type=int, default=520)
    ap.add_argument("--cache", default=os.path.join(HERE, "data", "retune_cache.json"))
    a = ap.parse_args()

    cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
    files = [f for f in sorted(glob.glob(os.path.join(a.targets, "*.npz")))
             if len(np.load(f)["cb"]) <= a.max_n]

    for f in files:
        name = os.path.basename(f).replace(".npz", "")
        if name in cache:
            continue
        d = np.load(f)
        cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if y.sum() == 0 or (pool & (y == 1)).sum() == 0:
            continue
        dist = min_dist_to_anchor(cb, anchor)
        entry = {}
        for radius, kappa in itertools.product(RADII, KAPPAS):
            base, pert, A = eigen_table(cb, radius, kappa)
            for K in KS:
                m = min(K, len(base))
                raw = np.sum(np.abs(pert[:, :m] - base[:m]) / (base[:m] + 1e-12), axis=1)
                for resid in (True, False):
                    sc = distance_zscore(raw, dist, pool) if resid else raw
                    sm = pocket_smooth(rank_percentile(sc), A)
                    auc, _ = stratified_auc(y, sm, pool, dist, 2.0)
                    entry[f"r{radius}_k{kappa}_K{K}_{'z' if resid else 'raw'}"] = auc
        cache[name] = entry
        json.dump(cache, open(a.cache, "w"))
        print(f"  [{len(cache):3d}] {name}", flush=True)

    names = sorted(cache)
    if len(names) < 8:
        print("not enough targets cached yet")
        return
    keys = sorted({k for v in cache.values() for k in v})

    def mean_on(subset, key):
        v = [cache[n].get(key) for n in subset]
        v = np.array([x for x in v if x is not None], float)
        return np.nanmean(v)

    halves = {"odd tunes / even reports": (names[1::2], names[0::2]),
              "even tunes / odd reports": (names[0::2], names[1::2])}
    default = "r10.0_k1.0_K3_z"
    print(f"\n=== ALPS re-tune, curated labels, distance-stratified, "
          f"{len(names)} targets, floor 0.496 ===")
    for label, (tune, report) in halves.items():
        best = max(keys, key=lambda k: mean_on(tune, k))
        print(f"\n{label}   (tune n={len(tune)}, report n={len(report)})")
        print(f"  best on tuning half : {best:22s} tune {mean_on(tune, best):.3f} "
              f"-> report {mean_on(report, best):.3f}")
        print(f"  current default     : {default:22s} tune {mean_on(tune, default):.3f} "
              f"-> report {mean_on(report, default):.3f}")
    print("\ntop settings by mean over ALL targets (for inspection only — this is the "
          "number that overfits):")
    for k in sorted(keys, key=lambda k: -mean_on(names, k))[:8]:
        print(f"  {k:24s} {mean_on(names, k):.3f}")
    print(f"  {'-- default --':24s} {mean_on(names, default):.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Per-residue feature matrix on the curated targets.

The features are the scores of the methods already implemented upstream. Using
them rather than raw geometry keeps the comparison honest in a specific way: a
quantum model that beats a classical one *on the same features* has demonstrated
something about the model, whereas one fed different inputs has not.

Cached incrementally — the ALPS and apop columns cost N eigensolves per target.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from methods.alps import alps_scores, spectral_response                    # noqa: E402
from methods.btb import btb_scores                                         # noqa: E402
from methods.common import (contact_graph, distal_nonanchor_mask,          # noqa: E402
                            min_dist_to_anchor, rank_percentile)
from methods.enm import apop_scores, corrsite_scores, prs_scores           # noqa: E402
from methods.quantum import (ctqw_communicability, ipr_resonant_transfer,  # noqa: E402
                             noisy_or)

FEATS = ["alps", "apop", "corrsite", "prs", "btb", "ctqw", "dist", "burial"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.npz")


def build_one(cb, anchor):
    A = contact_graph(cb, 10.0)
    pool = distal_nonanchor_mask(cb, anchor, 8.0)
    d = min_dist_to_anchor(cb, anchor)
    ctqw = ctqw_communicability(A, anchor)
    f = {
        "alps": alps_scores(cb, anchor, pool),
        "corrsite": corrsite_scores(cb, anchor),
        "prs": prs_scores(cb, anchor),
        "btb": btb_scores(cb, anchor, distance_corrected=False),
        "ctqw": noisy_or(ctqw, ipr_resonant_transfer(A, anchor)),
        "dist": d,
        "burial": A.sum(axis=1),
        "apop": apop_scores(cb, anchor) if len(cb) <= 700 else spectral_response(cb),
    }
    # rank percentiles: cross-protein comparability, which a learner needs
    X = np.column_stack([rank_percentile(f[k]) for k in FEATS])
    return X, pool, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_curated"))
    ap.add_argument("--max-n", type=int, default=700)
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
        if len(cb) > a.max_n:
            continue
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if y.sum() == 0 or (pool & (y == 1)).sum() == 0:
            continue
        X, pool, d = build_one(cb, anchor)
        data.append(dict(t=name, X=X, y=y, pool=pool, dist=d, n=int(len(cb))))
        np.savez_compressed(a.cache, data=np.array(data, dtype=object),
                            feats=np.array(FEATS))
        print(f"  [{len(data):3d}] {name} N={len(cb)}", flush=True)
    print(f"\n{len(data)} targets, {len(FEATS)} features -> {a.cache}")


if __name__ == "__main__":
    main()

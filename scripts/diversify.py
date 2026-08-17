#!/usr/bin/env python
"""Does spatially diversifying the top-5 improve it?

The cooperative-selection experiment turned up a control result worth acting on:
a set of residues chosen purely for spatial spread hit true allosteric sites as
often as the QUBO optimum, and far more often than the plain top-k by score. The
plain top-k is highly redundant — the highest-scoring residues are usually
neighbours of each other, so five of them describe one site, not five.

This script tests the cheap fix directly: keep ALPS's ranking, but pick the k
residues greedily subject to a minimum separation, so the selection describes k
distinct places instead of one place five times.

  top-k        the current selection: five highest scores
  diverse-k    highest score first, then each next residue must be >= MIN_SEP A
               from every residue already chosen
  spread-k     control: ignore the scores entirely, maximise spread over the
               same candidate pool

Usage: python3 scripts/diversify.py [--targets data/targets_b] [--min-sep 10]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from methods.alps import alps_scores                                      # noqa: E402
from methods.common import (contact_graph, distal_nonanchor_mask,          # noqa: E402
                            pocket_smooth, rank_percentile)


def diverse_topk(order, cb, k, min_sep):
    """Highest-scoring residues subject to a minimum pairwise separation."""
    chosen = []
    for i in order:
        if all(np.linalg.norm(cb[i] - cb[j]) >= min_sep for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    for i in order:                      # top up if the constraint was too tight
        if len(chosen) == k:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    return chosen


def max_spread(cand, cb, k):
    D = cdist(cb[cand], cb[cand])
    sel = [int(np.argmax(D.sum(1)))]
    while len(sel) < k:
        sel.append(int(np.argmax(np.min(D[:, sel], axis=1))))
    return [int(cand[s]) for s in sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_b"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-sep", type=float, default=10.0)
    ap.add_argument("--pool-m", type=int, default=26)
    a = ap.parse_args()

    res = {key: [] for key in ("topk", "diverse", "spread")}
    spreads = {key: [] for key in ("topk", "diverse", "spread")}
    for f in sorted(glob.glob(os.path.join(a.targets, "*.npz"))):
        d = np.load(f)
        cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if pool.sum() < a.pool_m or y.sum() == 0:
            continue
        A = contact_graph(cb, 10.0)
        s = pocket_smooth(rank_percentile(alps_scores(cb, anchor, pool)), A)
        order = np.where(pool)[0][np.argsort(s[pool])[::-1]]

        sets = {
            "topk": [int(i) for i in order[:a.k]],
            "diverse": diverse_topk(order, cb, a.k, a.min_sep),
            "spread": max_spread(order[:a.pool_m], cb, a.k),
        }
        for key, idx in sets.items():
            res[key].append(int(bool(y[idx].sum())))
            spreads[key].append(float(cdist(cb[idx], cb[idx]).mean()))
        print(f"  {os.path.basename(f):14s} " +
              "  ".join(f"{key}:{int(bool(y[idx].sum()))}" for key, idx in sets.items()),
              flush=True)

    n = len(res["topk"])
    print(f"\n=== top-{a.k} selection strategies on {os.path.basename(a.targets)} "
          f"(n={n}, min separation {a.min_sep:.0f} A) ===")
    print(f"{'selection':28s} {'hit rate':>9s} {'mean spread':>12s}")
    name = {"topk": "top-k by score (current)",
            "diverse": f"score + >={a.min_sep:.0f} A separation",
            "spread": "max spread, scores ignored"}
    for key in ("topk", "diverse", "spread"):
        print(f"{name[key]:28s} {np.mean(res[key])*100:8.1f}% "
              f"{np.mean(spreads[key]):11.1f} A")


if __name__ == "__main__":
    main()

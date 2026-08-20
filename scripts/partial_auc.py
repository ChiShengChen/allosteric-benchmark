#!/usr/bin/env python
"""What does a method add *after* conditioning on distance?

Section 9.3 of the README established that a one-line closeness control reaches
AUC 0.617 on curated labels and that nothing beats it. That makes plain AUC
nearly uninterpretable: a score correlated with distance inherits most of its
discrimination from geometry rather than from any model of allostery.

This measures the part that is not geometry, non-parametrically.

**Distance-stratified AUC.** Consider only (positive, negative) pairs drawn from
the candidate pool whose distances to the active site differ by at most ``tol``
Angstrom, and ask how often the method ranks the positive above the negative.
Within such a pair the distance information is spent, so anything above 0.5 is
information the method carries beyond proximity.

The control validates the metric rather than the method: ``ctrl_closeness`` must
collapse to ~0.5 here by construction, because within a distance stratum it is
nearly constant. If it does not, the stratification is too loose.

Usage: python3 scripts/partial_auc.py [--targets data/targets_curated] [--tol 2.0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from methods.common import (contact_graph, distal_nonanchor_mask,          # noqa: E402
                            min_dist_to_anchor, pocket_smooth, rank_percentile)
from eval_curated_full import score_all                                     # noqa: E402


def stratified_auc(y, score, pool, dist, tol=2.0, min_pairs=30):
    """Pairwise AUC restricted to positive/negative pairs matched on distance."""
    pool = np.asarray(pool, bool)
    s = np.asarray(score, float)
    ok = pool & np.isfinite(s)
    pos = np.where(ok & (np.asarray(y) == 1))[0]
    neg = np.where(ok & (np.asarray(y) != 1))[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, 0
    dp, dn = dist[pos], dist[neg]
    sp, sn = s[pos], s[neg]
    # |d_pos - d_neg| <= tol  ->  distance carries no information within the pair
    close = np.abs(dp[:, None] - dn[None, :]) <= tol
    if close.sum() < min_pairs:
        return np.nan, int(close.sum())
    diff = sp[:, None] - sn[None, :]
    wins = (diff > 0).astype(float) + 0.5 * (diff == 0)
    return float(wins[close].sum() / close.sum()), int(close.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_curated"))
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results_partial_auc.json"))
    a = ap.parse_args()

    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {r["target"] for r in res}
    for f in sorted(glob.glob(os.path.join(a.targets, "*.npz"))):
        name = os.path.basename(f).replace(".npz", "")
        if name in done:
            continue
        d = np.load(f)
        cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if y.sum() == 0 or (pool & (y == 1)).sum() == 0:
            continue
        dist = min_dist_to_anchor(cb, anchor)
        raw, A = score_all(cb, anchor, pool)
        row = {}
        for k, s in raw.items():
            sm = pocket_smooth(rank_percentile(s), A)
            auc, npairs = stratified_auc(y, sm, pool, dist, a.tol)
            row[k] = [auc, npairs]
        res.append(dict(target=name, n=int(len(cb)), rows=row))
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"  [{len(res):3d}] {name}", flush=True)

    names = sorted({k for r in res for k in r["rows"]})
    print(f"\n=== distance-stratified AUC, |Δd| ≤ {a.tol:.0f} Å, n={len(res)} targets ===")
    print("a score that only reproduces proximity lands at 0.500\n")
    print(f"{'method':16s} {'stratified':>11s} {'targets':>8s}")
    out = []
    for m in names:
        v = [r["rows"][m][0] for r in res if m in r["rows"]]
        v = np.array([x for x in v if x is not None], float)
        out.append((np.nanmean(v), m, np.sum(~np.isnan(v))))
    for au, m, n in sorted(out, reverse=True):
        flag = "" if abs(au - 0.5) < 0.02 else ("  <-- above proximity" if au > 0.52 else "")
        print(f"{m:16s} {au:11.3f} {n:8d}{flag}")


if __name__ == "__main__":
    main()

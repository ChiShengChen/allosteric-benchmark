#!/usr/bin/env python
"""Fuse the GNN's per-residue scores with ALPS, with the weight chosen honestly.

The GNN and ALPS tie on average -- +0.011 +/- 0.009 over eight identical runs on the
AlloBench set -- and that tie is what this repository reported for months. What it did
not report, because nobody had measured it, is that the two agree on *which* targets
they get right barely at all: per-target Pearson r = 0.146, Spearman 0.100, with the
GNN ahead on 47% of targets. Two methods of equal average skill that fail on different
proteins are the textbook case where combining them wins, and the oracle ceiling
(cross-fitted, so not selection on noise) sits at 0.708 against 0.612 for ALPS.

So this script combines them. Two independent gains are available and they are
reported separately, because they cost different things:

**Averaging runs.** Training is not reproducible run to run -- see RESULTS.md -- and at
the per-residue level that noise is largely independent, so averaging the scores of k
identical runs removes it. This costs k times the training and needs no new idea.

**Fusing with ALPS.** Rank-average the GNN score with ALPS. This costs nothing at all:
both score vectors already exist.

Two rules keep the number honest.

* **The weight is selected per fold on the other folds**, never on the fold being
  scored. A 70/30 split scored better than 50/50 when both were read off the test
  numbers, which is exactly how this repository has previously talked itself into a
  result; the selection is therefore nested, and the unnested 50/50 constant is
  reported next to it so the cost of choosing is visible.
* **Scores are rank-percentiled before mixing.** ALPS is a distance-conditional
  z-score and the GNN is an unnormalised logit; mixing them raw would be a mixture in
  name and a rescaling in fact.

The GNN scores come from `gnn/run.py --dump`. Every dumped score is out-of-fold -- the
model that produced a target's score never trained on it -- and all runs share one fold
assignment, so averaging across runs does not leak.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from methods.common import rank_percentile                       # noqa: E402
from partial_auc import stratified_auc                            # noqa: E402

WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def load(cache, dumps):
    data = list(np.load(cache, allow_pickle=True)["data"])
    ds = [np.load(f, allow_pickle=True) for f in dumps]
    idx = {t: i for i, t in enumerate(ds[0]["t"])}
    fold = ds[0]["fold"]
    out = []
    for r in data:
        i = idx.get(r["t"])
        if i is None:
            continue
        gs = [d["gnn_score"][i] for d in ds]
        if any(g is None for g in gs):
            continue
        out.append(dict(
            t=r["t"], y=r["y"], pool=r["pool"], dist=r["dist"], fold=int(fold[i]),
            g=[rank_percentile(np.asarray(x, float)) for x in gs],
            a=rank_percentile(np.asarray(r["alps"], float))))
    return out


def auc(rec, s):
    return stratified_auc(rec["y"], s, rec["pool"], rec["dist"], 2.0)[0]


def mix(rec, w, k):
    """w * (mean of k GNN runs) + (1 - w) * ALPS, all rank-percentiles."""
    g = np.mean(rec["g"][:k], axis=0)
    return w * g + (1.0 - w) * rec["a"]


def main():
    ap = argparse.ArgumentParser()
    g = os.path.join(HERE, "gnn")
    ap.add_argument("--cache", default=os.path.join(g, "graphs_allobench.npz"))
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="one or more gnn/run.py --dump files over the same cache")
    ap.add_argument("--floor", type=float, default=0.5020)
    a = ap.parse_args()

    recs = load(a.cache, a.dumps)
    k = len(recs[0]["g"])
    folds = sorted({r["fold"] for r in recs})
    print(f"{len(recs)} targets, {k} runs, {len(folds)} folds, floor {a.floor:.4f}\n")

    # nested weight selection: choose w on the other folds, score on this one
    per_fold_w, nested = {}, {}
    for f in folds:
        tr = [r for r in recs if r["fold"] != f]
        best_w, best_v = None, -1.0
        for w in WEIGHTS:
            v = float(np.nanmean([auc(r, mix(r, w, k)) for r in tr]))
            if v > best_v:
                best_w, best_v = w, v
        per_fold_w[f] = best_w
        for r in recs:
            if r["fold"] == f:
                nested[r["t"]] = auc(r, mix(r, best_w, k))
    print("weight chosen on the held-out-from-this-fold data: " +
          ", ".join(f"fold {f} -> {per_fold_w[f]:.1f}" for f in folds) + "\n")

    cols = {
        "ALPS": {r["t"]: auc(r, r["a"]) for r in recs},
        "GNN, 1 run (mean)": {r["t"]: float(np.nanmean([auc(r, g) for g in r["g"]]))
                              for r in recs},
        f"GNN, {k}-run mean": {r["t"]: auc(r, np.mean(r["g"], axis=0)) for r in recs},
        "fuse 50/50 (fixed)": {r["t"]: auc(r, mix(r, 0.5, k)) for r in recs},
        "fuse (nested w)": nested,
        "CONTROL ctrl_dist": {r["t"]: auc(r, -r["dist"]) for r in recs},
    }
    rng = np.random.default_rng(0)
    cols["CONTROL ctrl_random"] = {r["t"]: auc(r, rng.random(len(r["y"]))) for r in recs}

    base = np.array([cols["ALPS"][r["t"]] for r in recs], float)
    print(f"{'method':22s} {'strat AUC':>10s} {'vs floor':>9s} {'vs ALPS':>9s} "
          f"{'paired p':>10s}")
    for name, col in cols.items():
        v = np.array([col[r["t"]] for r in recs], float)
        ok = ~np.isnan(v) & ~np.isnan(base)
        p = ("reference" if name == "ALPS"
             else f"{stats.wilcoxon(v[ok], base[ok]).pvalue:.2e}")
        print(f"{name:22s} {np.nanmean(v):10.3f} {np.nanmean(v)-a.floor:+9.3f} "
              f"{np.nanmean(v[ok]-base[ok]):+9.4f} {p:>10s}")

    if k > 1:
        # Average over subsets of size kk, not the first kk runs. Taking the first kk
        # makes every row inherit whichever run happens to be dump 0: on the AlloBench
        # set that run scored 0.610 against a 0.623 mean, which inflated the reported
        # ensemble gain by the better part of a hundredth before this was fixed.
        import itertools
        rng = np.random.default_rng(0)
        print(f"\n{'k runs':>7s} {'GNN mean':>9s} {'fuse 50/50':>11s}   "
              f"(averaged over subsets)")
        for kk in range(1, k + 1):
            combos = list(itertools.combinations(range(k), kk))
            if len(combos) > 10:
                combos = [tuple(rng.choice(k, kk, replace=False)) for _ in range(10)]
            e = np.mean([np.nanmean([auc(r, np.mean([r["g"][i] for i in c], axis=0))
                                     for r in recs]) for c in combos])
            f = np.mean([np.nanmean([auc(r, 0.5 * np.mean([r["g"][i] for i in c], axis=0)
                                         + 0.5 * r["a"]) for r in recs])
                         for c in combos])
            print(f"{kk:7d} {e:9.3f} {f:11.3f}")


if __name__ == "__main__":
    main()

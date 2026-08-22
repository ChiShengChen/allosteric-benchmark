#!/usr/bin/env python
"""Can any method tell that a protein has NO allosteric site?

Every earlier result answers "given a protein with a site, is it ranked highly".
This asks the question a user faces first: given a protein with no annotated
allosteric site, does the method say so, or name five residues anyway?

Protein-level discrimination, so two things change from the residue-level metric:

* **Raw scores, not rank percentiles.** Rank-percentile normalisation makes the
  top score 1.0 in every protein by construction, destroying exactly the
  cross-protein magnitude this question needs.
* **Size is a control.** If the negatives are systematically smaller or larger
  than the positives, any size-correlated statistic separates them for free.
  `ctrl_size` is scored alongside the methods so that confound is visible rather
  than assumed away.

Statistics tried per protein, all computed over the distal candidate pool:
  max      the single highest raw score
  top5     mean of the five highest
  peak     (top5 mean − median) / std — how concentrated the signal is, scale-free
  kurt     kurtosis of the pool scores — is there a spike at all
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

from methods.alps import spectral_response                                # noqa: E402
from methods.common import (contact_graph, distal_nonanchor_mask,         # noqa: E402
                            min_dist_to_anchor)
from methods.enm import corrsite_scores                                   # noqa: E402
from methods.quantum import (ctqw_communicability, ipr_resonant_transfer, # noqa: E402
                             noisy_or)
from scipy import stats                                                   # noqa: E402


def protein_stats(v, pool):
    v = np.asarray(v, float)[np.asarray(pool, bool)]
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return None
    top5 = np.sort(v)[-5:]
    sd = v.std() or 1e-12
    return {"max": float(v.max()), "top5": float(top5.mean()),
            "peak": float((top5.mean() - np.median(v)) / sd),
            "kurt": float(stats.kurtosis(v))}


def score_protein(cb, anchor):
    A = contact_graph(cb, 10.0)
    ctqw = ctqw_communicability(A, anchor)
    return {"ALPS_raw": spectral_response(cb),
            "ctqw_only": ctqw,
            "qasc_baseline": noisy_or(ctqw, ipr_resonant_transfer(A, anchor)),
            "corrsite": corrsite_scores(cb, anchor),
            "ctrl_burial": A.sum(axis=1),
            "ctrl_closeness": -min_dist_to_anchor(cb, anchor)}


def collect(pattern, label, cache, max_n):
    for f in sorted(glob.glob(pattern)):
        name = f"{label}:{os.path.basename(f)}"
        if name in cache:
            continue
        d = np.load(f)
        cb, anchor = d["cb"], d["anchor"]
        if len(cb) > max_n:
            continue
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if pool.sum() < 30:
            continue
        rec = {"label": label, "n": int(len(cb))}
        for m, v in score_protein(cb, anchor).items():
            st = protein_stats(v, pool)
            if st:
                rec[m] = st
        cache[name] = rec
        print(f"  {name}", flush=True)


def auc_matched(pos_v, pos_k, neg_v, neg_k, tols):
    """AUC over pairs matched on every key in `pos_k` / `neg_k`.

    Matching on residue count alone is not enough: after it, radius of gyration
    still separates the sets at AUC 0.601, and a distance-based statistic inherits
    that. Passing (N, Rg) spends both.
    """
    pv = np.asarray(pos_v, float); nv = np.asarray(neg_v, float)
    okp, okn = np.isfinite(pv), np.isfinite(nv)
    pv, nv = pv[okp], nv[okn]
    if len(pv) < 5 or len(nv) < 5:
        return np.nan, 0
    close = np.ones((len(pv), len(nv)), bool)
    for key, tol in zip(pos_k, tols):
        pass
    for (pk, nk), tol in zip(zip(pos_k, neg_k), tols):
        a = np.log(np.asarray(pk, float)[okp])
        b = np.log(np.asarray(nk, float)[okn])
        close &= np.abs(a[:, None] - b[None, :]) <= tol
    if close.sum() < 50:
        return np.nan, int(close.sum())
    d = pv[:, None] - nv[None, :]
    w = (d > 0).astype(float) + 0.5 * (d == 0)
    return float(w[close].sum() / close.sum()), int(close.sum())


def auc_size_matched(pos_v, pos_n, neg_v, neg_n, tol=0.25):
    """AUC over positive/negative pairs matched on log size.

    The raw comparison is confounded: the negatives are systematically smaller
    (median N 327 against 532), so ctrl_size alone separates the sets at 0.783.
    Restricting to pairs within `tol` in log N spends that information, exactly
    as the distance-stratified metric spends proximity at the residue level.
    """
    pv, pn = np.asarray(pos_v, float), np.asarray(pos_n, float)
    nv, nn = np.asarray(neg_v, float), np.asarray(neg_n, float)
    ok_p = np.isfinite(pv); ok_n = np.isfinite(nv)
    pv, pn, nv, nn = pv[ok_p], pn[ok_p], nv[ok_n], nn[ok_n]
    if len(pv) < 5 or len(nv) < 5:
        return np.nan, 0
    close = np.abs(np.log(pn)[:, None] - np.log(nn)[None, :]) <= tol
    if close.sum() < 50:
        return np.nan, int(close.sum())
    diff = pv[:, None] - nv[None, :]
    wins = (diff > 0).astype(float) + 0.5 * (diff == 0)
    return float(wins[close].sum() / close.sum()), int(close.sum())


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    ok_p, ok_n = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(ok_p) < 5 or len(ok_n) < 5:
        return np.nan
    wins = (ok_p[:, None] > ok_n[None, :]).mean() + 0.5 * (ok_p[:, None] == ok_n[None, :]).mean()
    return float(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results_false_positive.json"))
    ap.add_argument("--max-n", type=int, default=900)
    a = ap.parse_args()
    cache = json.load(open(a.out)) if os.path.exists(a.out) else {}
    collect(os.path.join(HERE, "data", "targets_curated", "*.npz"), "pos", cache, a.max_n)
    json.dump(cache, open(a.out, "w"))
    collect(os.path.join(HERE, "data", "targets_negative", "*.npz"), "neg", cache, a.max_n)
    json.dump(cache, open(a.out, "w"))

    pos = [v for v in cache.values() if v["label"] == "pos"]
    neg = [v for v in cache.values() if v["label"] == "neg"]
    print(f"\n=== protein-level: has an allosteric site vs does not ===")
    print(f"positives {len(pos)}   negatives {len(neg)}")
    npos = np.array([v["n"] for v in pos], float)
    nneg = np.array([v["n"] for v in neg], float)
    print(f"size: positives median {np.median(npos):.0f}, negatives median "
          f"{np.median(nneg):.0f}  ->  ctrl_size AUC {auc(npos, nneg):.3f}")
    print("\n(0.5 = cannot tell the two apart at all)\n")
    methods = [m for m in ("ALPS_raw", "ctqw_only", "qasc_baseline", "corrsite",
                           "ctrl_burial", "ctrl_closeness")
               if any(m in v for v in pos)]
    stat_names = ("max", "top5", "peak", "kurt")
    print("RAW (confounded by size — every column inherits ctrl_size)")
    print(f"{'method':16s} " + "  ".join(f"{s:>7s}" for s in stat_names))
    for m in methods:
        cells = []
        for st in stat_names:
            p = [v[m][st] for v in pos if m in v]
            q = [v[m][st] for v in neg if m in v]
            cells.append(f"{auc(p, q):7.3f}")
        print(f"{m:16s} " + "  ".join(cells))

    methods = methods + [m for m in ("ctrl_random", "ctrl_dist")
                         if any(m in v for v in pos) and m not in methods]
    for label, tols in (("SIZE-MATCHED (log N <= 0.25)", (0.25,)),
                        ("SIZE + SHAPE MATCHED (log N <= 0.25, log Rg <= 0.10)",
                         (0.25, 0.10))):
        keys = ("n",) if len(tols) == 1 else ("n", "rg")
        pk0 = [[v[k] for v in pos if methods[0] in v] for k in keys]
        nk0 = [[v[k] for v in neg if methods[0] in v] for k in keys]
        _, npairs = auc_matched([0.0] * len(pk0[0]), pk0,
                                [1.0] * len(nk0[0]), nk0, tols)
        print(f"\n{label}   ({npairs} pairs)")
        print(f"{'method':16s} " + "  ".join(f"{s:>7s}" for s in stat_names))
        for m in methods:
            if not any(m in v for v in pos):
                continue
            cells = []
            for st in stat_names:
                p = [v[m][st] for v in pos if m in v]
                q = [v[m][st] for v in neg if m in v]
                pk = [[v[k] for v in pos if m in v] for k in keys]
                nk = [[v[k] for v in neg if m in v] for k in keys]
                au, _ = auc_matched(p, pk, q, nk, tols)
                cells.append("      —" if au != au else f"{au:7.3f}")
            print(f"{m:16s} " + "  ".join(cells))


if __name__ == "__main__":
    main()

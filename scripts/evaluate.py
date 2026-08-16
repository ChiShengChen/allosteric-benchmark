#!/usr/bin/env python
"""Evaluate every method on every target under one identical protocol.

Metrics
-------
perm_p : QASC's own criterion -- one-sided permutation test that the known
         allosteric residues score above distal non-anchor background.
auc    : ROC-AUC of the score ranking restricted to the distal non-anchor
         candidate pool (the pool the model actually chooses from).
hit5   : does the top-5 contain any true allosteric residue.
dcc    : distance from the centroid of the top-5 predicted residues to the
         centroid of the true allosteric site. STINGAllo's success criterion is
         DCC <= 4 A; this is the "did you point at the right place" metric that
         a permutation p-value does not test.
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

from methods.common import (contact_graph, distal_nonanchor_mask, pocket_smooth,  # noqa: E402
                            rank_percentile, weighted_contact_graph)
from methods import btb as M_btb                                                  # noqa: E402
from methods import enm as M_enm                                                  # noqa: E402
from methods import quantum as M_q                                                # noqa: E402
from methods import alps as M_alps                                                # noqa: E402
from methods import qpr as M_qpr                                                  # noqa: E402

NPERM = 10000
SEED = 1234


# ---------------------------------------------------------------- metrics ----
def permutation_p(y, scores, bg_mask, nperm=NPERM, seed=SEED):
    scores = np.asarray(scores, float)
    y = np.asarray(y).astype(bool)
    valid = np.isfinite(scores)
    bg = np.asarray(bg_mask, bool) & valid
    pos = np.where(y & valid)[0]
    bgi = np.where(bg)[0]
    if len(pos) == 0 or len(bgi) < len(pos):
        return float("nan")
    obs = scores[pos].mean()
    rng = np.random.default_rng(seed)
    null = np.array([scores[rng.choice(bgi, len(pos), replace=False)].mean()
                     for _ in range(nperm)])
    return float((np.sum(null >= obs) + 1) / (nperm + 1))


def auc_in_pool(y, scores, pool_mask):
    s = np.asarray(scores, float)
    ok = np.asarray(pool_mask, bool) & np.isfinite(s)
    yy = np.asarray(y).astype(bool)
    pos = s[ok & yy]
    neg = s[ok & ~yy]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def top_k(scores, cand_mask, k=5):
    s = np.asarray(scores, float).copy()
    s[~np.asarray(cand_mask, bool)] = -np.inf
    return np.argsort(s)[::-1][:k]


def dcc(cb, top_idx, y):
    pos = np.where(np.asarray(y).astype(bool))[0]
    if len(pos) == 0 or len(top_idx) == 0:
        return float("nan")
    return float(np.linalg.norm(cb[top_idx].mean(0) - cb[pos].mean(0)))


# ---------------------------------------------------------------- methods ----
def build_methods(cb, anchor):
    """Return {name: raw per-residue score}. Every method sees only cb+anchor."""
    A = contact_graph(cb, 10.0)
    pool = distal_nonanchor_mask(cb, anchor, 8.0)
    W = weighted_contact_graph(cb, 10.0, 4.0)
    out = {}

    ctqw = M_q.ctqw_communicability(A, anchor)
    ipr_u = M_q.ipr_resonant_transfer(A, anchor, seed="uniform")
    ipr_d = M_q.ipr_resonant_transfer(A, anchor, seed="degree")
    ctqw_n = M_q.ctqw_communicability(A, anchor, normalized=True)

    out["qasc_baseline"] = M_q.noisy_or(ctqw, ipr_u)
    out["qasc_degseed"] = M_q.noisy_or(ctqw, ipr_d)
    out["qasc_normlap"] = M_q.noisy_or(ctqw_n, ipr_u)
    out["ctqw_only"] = ctqw

    out["btb"] = M_btb.btb_scores(cb, anchor, distance_corrected=True, pool=pool)
    out["btb_raw"] = M_btb.btb_scores(cb, anchor, distance_corrected=False)

    out["corrsite"] = M_enm.corrsite_scores(cb, anchor)
    out["prs"] = M_enm.prs_scores(cb, anchor)
    if len(cb) <= 600:                      # O(N) eigendecompositions: skip huge
        out["apop"] = M_enm.apop_scores(cb, anchor)

    out["enaqt"] = M_q.enaqt_transfer(A, anchor, gamma_rel=1.0)

    # distance-corrected QASC: the one-line borrow from the bond-to-bond line
    from methods.common import min_dist_to_anchor
    d = min_dist_to_anchor(cb, anchor)
    out["qasc_distcorr"] = M_btb.quantile_residual(
        np.exp(out["qasc_baseline"]), d, pool=pool)

    # fusion of the quantum baseline with the classical Green-function channel
    out["qasc+btb"] = M_q.noisy_or(out["qasc_baseline"], out["btb"])

    # the method this study converged on, plus its ablations
    out["ALPS"] = M_alps.alps_scores(cb, anchor, pool)
    out["ALPS_noresid"] = M_alps.spectral_response(cb)
    if len(cb) <= 660:
        out["qpr_coherent"] = M_qpr.qpr_scores(cb, anchor, pool)
        out["cpr_classical"] = M_qpr.cpr_scores(cb, anchor, pool)

    # promising combinations
    if "apop" in out:
        out["qasc+apop"] = M_q.noisy_or(out["apop"], out["qasc_baseline"])
        out["apop+corrsite"] = M_q.noisy_or(out["apop"], out["corrsite"])
        out["apop+dist"] = M_q.noisy_or(out["apop"], d)

    # ---- trivial controls: how much of any result is explained without
    # any allostery model at all?
    out["ctrl_burial"] = A.sum(axis=1)              # contact degree = burial
    out["ctrl_dist"] = d                            # plain distance from anchor
    rng = np.random.default_rng(0)
    out["ctrl_random"] = rng.random(len(cb))
    return out, A


def evaluate_file(path, methods_filter=None):
    d = np.load(path)
    cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
    if y.sum() == 0:
        return None
    raw, A = build_methods(cb, anchor)
    cand = distal_nonanchor_mask(cb, anchor, 8.0)
    bg = cand & (y != 1)
    if bg.sum() < max(3, int(y.sum())):
        return None

    rows = {}
    for name, s in raw.items():
        if methods_filter and name not in methods_filter:
            continue
        sm = pocket_smooth(rank_percentile(s), A)   # identical post-processing
        t5 = top_k(sm, cand, 5)
        rows[name] = dict(
            perm_p=permutation_p(y, sm, bg),
            auc=auc_in_pool(y, sm, cand),
            hit5=int(bool(y[t5].sum())),
            dcc=dcc(cb, t5, y),
        )
    return dict(target=os.path.basename(path).replace(".npz", ""),
                n=int(len(cb)), n_pos=int(y.sum()), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.targets, "*.npz")))
    if a.limit:
        files = files[:a.limit]
    print(f"evaluating {len(files)} targets\n")

    results = []
    for i, f in enumerate(files, 1):
        try:
            r = evaluate_file(f)
        except Exception as e:                       # noqa: BLE001
            print(f"  [{i}] {os.path.basename(f)} FAILED: {e}")
            continue
        if r is None:
            continue
        results.append(r)
        best = min(r["rows"].items(), key=lambda kv: kv[1]["perm_p"])
        print(f"  [{i:3d}] {r['target']:14s} N={r['n']:4d} pos={r['n_pos']:3d} "
              f"best={best[0]}({best[1]['perm_p']:.4f})", flush=True)

    json.dump(results, open(a.out, "w"), indent=1)
    summarize(results)


def summarize(results):
    if not results:
        print("no results")
        return
    names = sorted({k for r in results for k in r["rows"]})
    print(f"\n{'method':16s} {'sig<0.05':>9s} {'medianP':>9s} {'meanAUC':>8s} "
          f"{'hit5':>7s} {'DCC<=4A':>8s} {'medDCC':>7s}  (n={len(results)})")
    print("-" * 72)
    rank = []
    for m in names:
        rows = [r["rows"][m] for r in results if m in r["rows"]]
        if not rows:
            continue
        p = np.array([x["perm_p"] for x in rows], float)
        auc = np.array([x["auc"] for x in rows], float)
        h5 = np.array([x["hit5"] for x in rows], float)
        dc = np.array([x["dcc"] for x in rows], float)
        sig = np.nanmean(p < 0.05) * 100
        rank.append((sig, m))
        print(f"{m:16s} {sig:8.1f}% {np.nanmedian(p):9.4f} {np.nanmean(auc):8.3f} "
              f"{np.nanmean(h5)*100:6.1f}% {np.nanmean(dc <= 4)*100:7.1f}% "
              f"{np.nanmedian(dc):7.1f}")
    print("-" * 72)
    for s, m in sorted(rank, reverse=True)[:3]:
        print(f"  top: {m} ({s:.1f}% significant)")


if __name__ == "__main__":
    main()

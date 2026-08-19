#!/usr/bin/env python
"""Evaluate the core methods on the full curated-label benchmark.

evaluate.py runs every method including the ones that need eigenvectors per
residue (qpr/cpr), which is what makes it slow on large structures. Those
already have their answer; this runs the methods the curated check is actually
about, on all 73 targets rather than the N<=500 subset, with an incremental
cache so an interrupted run keeps its work.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts"))
from methods.alps import alps_scores, spectral_response, distance_zscore   # noqa: E402
from methods.enm import apop_scores, corrsite_scores, prs_scores           # noqa: E402
from methods.btb import btb_scores                                         # noqa: E402
from methods.quantum import (ctqw_communicability, ipr_resonant_transfer,  # noqa: E402
                             noisy_or)
from methods.common import (contact_graph, distal_nonanchor_mask,          # noqa: E402
                            min_dist_to_anchor, pocket_smooth, rank_percentile)
from evaluate import auc_in_pool, permutation_p, top_k, dcc                # noqa: E402


def score_all(cb, anchor, pool):
    A = contact_graph(cb, 10.0)
    d = min_dist_to_anchor(cb, anchor)
    ctqw = ctqw_communicability(A, anchor)
    out = {
        "ALPS": alps_scores(cb, anchor, pool),
        "ALPS_noresid": spectral_response(cb),
        "qasc_baseline": noisy_or(ctqw, ipr_resonant_transfer(A, anchor)),
        "ctqw_only": ctqw,
        "corrsite": corrsite_scores(cb, anchor),
        "prs": prs_scores(cb, anchor),
        "btb_raw": btb_scores(cb, anchor, distance_corrected=False),
        "btb": btb_scores(cb, anchor, distance_corrected=True, pool=pool),
        "ctrl_dist": d,
        "ctrl_burial": A.sum(axis=1),
        "ctrl_random": np.random.default_rng(0).random(len(cb)),
    }
    if len(cb) <= 900:
        out["apop"] = apop_scores(cb, anchor)
    return out, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_curated"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results_curated_full.json"))
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
        bg = pool & (y != 1)
        if y.sum() == 0 or bg.sum() < max(3, int(y.sum())):
            continue
        raw, A = score_all(cb, anchor, pool)
        rows = {}
        for k, s in raw.items():
            sm = pocket_smooth(rank_percentile(s), A)
            t5 = top_k(sm, pool, 5)
            rows[k] = dict(perm_p=permutation_p(y, sm, bg), auc=auc_in_pool(y, sm, pool),
                           hit5=int(bool(y[t5].sum())), dcc=dcc(cb, t5, y))
        res.append(dict(target=name, n=int(len(cb)), n_pos=int(y.sum()), rows=rows))
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"  [{len(res):3d}] {name} N={len(cb):5d} pos={int(y.sum()):3d}", flush=True)

    names = sorted({k for r in res for k in r["rows"]})
    print(f"\n=== curated labels, n={len(res)} ===")
    print(f"{'method':16s} {'hit5':>7s} {'sig':>7s} {'medP':>8s} {'AUC':>7s} {'DCC<=4':>7s}")
    rank = []
    for m in names:
        v = [r["rows"][m] for r in res if m in r["rows"]]
        p = np.array([x["perm_p"] for x in v]); au = np.array([x["auc"] for x in v])
        h = np.array([x["hit5"] for x in v]); dc = np.array([x["dcc"] for x in v])
        rank.append((np.nanmean(au), m, np.nanmean(h)*100, np.nanmean(p < .05)*100,
                     np.nanmedian(p), np.nanmean(dc <= 4)*100, len(v)))
    for au, m, h5, sig, mp, d4, n in sorted(rank, reverse=True):
        print(f"{m:16s} {h5:6.1f}% {sig:6.1f}% {mp:8.4f} {au:7.3f} {d4:6.1f}%  (n={n})")


if __name__ == "__main__":
    main()

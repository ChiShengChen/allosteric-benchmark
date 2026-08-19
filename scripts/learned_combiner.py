#!/usr/bin/env python
"""Does *learning* help at all on this task? The ceiling for any ML approach.

Before asking whether a quantum kernel is worth implementing, ask whether a
classical learned model beats the best unlearned score. Quantum ML would have to
beat the classical learner, so if learning itself does not help, the quantum
question is moot.

Features are the per-residue scores of the methods already implemented; the label
is the benchmark's y. Cross-validation is **grouped by protein**, so no residue
from a test protein is ever seen in training — the same declustering discipline
the rest of the benchmark uses.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts"))
from methods.alps import alps_scores                                   # noqa: E402
from methods.enm import apop_scores, corrsite_scores, prs_scores       # noqa: E402
from methods.btb import btb_scores                                     # noqa: E402
from methods.common import (contact_graph, distal_nonanchor_mask,      # noqa: E402
                            min_dist_to_anchor, pocket_smooth, rank_percentile)
from evaluate import auc_in_pool, permutation_p, top_k                 # noqa: E402

FEATS = ["alps", "apop", "corrsite", "prs", "btb", "dist", "burial"]


def features(cb, anchor, pool):
    A = contact_graph(cb, 10.0)
    f = {"alps": alps_scores(cb, anchor, pool),
         "corrsite": corrsite_scores(cb, anchor),
         "prs": prs_scores(cb, anchor),
         "btb": btb_scores(cb, anchor, pool=pool),
         "dist": min_dist_to_anchor(cb, anchor),
         "burial": A.sum(axis=1)}
    f["apop"] = apop_scores(cb, anchor) if len(cb) <= 660 else np.zeros(len(cb))
    X = np.column_stack([rank_percentile(f[k]) for k in FEATS])
    return X, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_b"))
    ap.add_argument("--cache", default=os.path.join(HERE, "data", "combiner_features.npz"))
    ap.add_argument("--max-n", type=int, default=660)
    ap.add_argument("--cached-only", action="store_true",
                    help="evaluate on the cache without extracting more features")
    a = ap.parse_args()

    # incremental cache: feature extraction is the expensive part, so a run that
    # is interrupted keeps everything it already computed
    data = []
    if os.path.exists(a.cache):
        data = list(np.load(a.cache, allow_pickle=True)["data"])
    done = {r["t"] for r in data}
    for f in ([] if a.cached_only else sorted(glob.glob(os.path.join(a.targets, "*.npz")))):
        name = os.path.basename(f)
        if name in done:
            continue
        d = np.load(f)
        cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
        if len(cb) > a.max_n or y.sum() == 0:
            continue
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if (pool & (y != 1)).sum() < max(3, int(y.sum())):
            continue
        X, A = features(cb, anchor, pool)
        data.append(dict(t=name, X=X, y=y, pool=pool, A=A, cb=cb))
        np.savez_compressed(a.cache, data=np.array(data, dtype=object))
        print(f"  {name} ({len(data)})", flush=True)

    n = len(data)
    print(f"\n{n} targets, {len(FEATS)} features, protein-grouped 5-fold CV")
    rng = np.random.default_rng(0)
    fold = rng.permutation(n) % 5

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    models = {"logistic": lambda: LogisticRegression(max_iter=2000, C=1.0),
              "gbdt": lambda: GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                                         random_state=0)}
    out = {m: [] for m in models}
    out["ALPS alone"] = []
    for k in range(5):
        tr = [i for i in range(n) if fold[i] != k]
        te = [i for i in range(n) if fold[i] == k]
        Xtr = np.vstack([data[i]["X"][data[i]["pool"]] for i in tr])
        ytr = np.concatenate([data[i]["y"][data[i]["pool"]] for i in tr])
        fitted = {}
        for mname, mk in models.items():
            mdl = mk(); mdl.fit(Xtr, ytr); fitted[mname] = mdl
        for i in te:
            d = data[i]; A = d["A"]; pool = d["pool"]; y = d["y"]
            bg = pool & (y != 1)
            for mname, mdl in fitted.items():
                p = np.zeros(len(y)); p[pool] = mdl.predict_proba(d["X"][pool])[:, 1]
                sm = pocket_smooth(rank_percentile(p), A); t5 = top_k(sm, pool, 5)
                out[mname].append((permutation_p(y, sm, bg), auc_in_pool(y, sm, pool),
                                   int(bool(y[t5].sum()))))
            sm = pocket_smooth(rank_percentile(d["X"][:, 0]), A); t5 = top_k(sm, pool, 5)
            out["ALPS alone"].append((permutation_p(y, sm, bg), auc_in_pool(y, sm, pool),
                                      int(bool(y[t5].sum()))))
    print(f"\n{'model':16s} {'sig':>7s} {'medP':>8s} {'AUC':>7s} {'hit5':>7s}")
    for k, v in out.items():
        p = np.array([x[0] for x in v]); au = np.array([x[1] for x in v])
        h = np.array([x[2] for x in v])
        print(f"{k:16s} {np.nanmean(p < 0.05)*100:6.1f}% {np.nanmedian(p):8.4f} "
              f"{np.nanmean(au):7.3f} {np.nanmean(h)*100:6.1f}%")


if __name__ == "__main__":
    main()

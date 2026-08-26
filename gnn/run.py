#!/usr/bin/env python
"""Train and evaluate the residue-graph GNN under this repository's protocol.

Same rules as everywhere else here, because every one of them caught a real error
upstream:

* **Curated labels**, not proxy labels.
* **Distance-stratified AUC**, not plain AUC -- plain AUC is dominated by
  proximity to the active site on both label sets.
* **Protein-grouped folds.** No residue of a test protein is trained on.
* **Model selection never touches the test fold.** Early stopping uses an inner
  validation split carved out of the training proteins.
* **Controls in the table**, `ctrl_random` and `ctrl_dist`, with the floor
  estimated from 25 seeds rather than one draw.
* **Paired tests.** A margin is not a result.

The opponent is ALPS on the same targets. That is the fair comparison: identical
input, one hand-designed spectral readout against one learned message-passing
model, evaluated by the same metric.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import MPNN                                          # noqa: E402
from partial_auc import stratified_auc                           # noqa: E402
from scipy import stats                                          # noqa: E402

# The curated-set floor, from 25 seeds. NOT a constant of nature: it is a property of
# the target set and the metric, so --floor must be passed when the data changes.
# gnn/floor_allobench.py measures 0.5020 +/- 0.0046 on the AlloBench route, where the
# floor's own sampling error is 3.4x tighter simply because there are 10x more targets.
FLOOR = 0.4963
torch.set_num_threads(8)


def to_torch(r, with_dist):
    return (torch.from_numpy(r["xd"] if with_dist else r["x"]),
            torch.from_numpy(r["src"]), torch.from_numpy(r["dst"]),
            torch.from_numpy(r["eattr"]))


def evaluate(model, data, idx, with_dist):
    model.eval()
    out = {}
    with torch.no_grad():
        for i in idx:
            r = data[i]
            s = model(*to_torch(r, with_dist)).numpy().astype(float)
            auc, _ = stratified_auc(r["y"], s, r["pool"], r["dist"], 2.0)
            out[r["t"]] = auc
    return out


def train_fold(data, tr, va, with_dist, hidden, layers, epochs, lr, seed):
    torch.manual_seed(seed)
    in_dim = data[tr[0]]["xd" if with_dist else "x"].shape[1]
    model = MPNN(in_dim=in_dim, hidden=hidden, layers=layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for i in rng.permutation(tr):
            r = data[i]
            pool = torch.from_numpy(r["pool"])
            y = torch.from_numpy(r["y"].astype(np.float32))
            if pool.sum() == 0:
                continue
            s = model(*to_torch(r, with_dist))
            yp, sp = y[pool], s[pool]
            npos = float(yp.sum())
            if npos == 0:
                continue
            w = (len(yp) - npos) / npos
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                sp, yp, pos_weight=torch.tensor(w))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (ep + 1) % 5 == 0:
            v = float(np.nanmean(list(evaluate(model, data, va, with_dist).values())))
            if v > best:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 6:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "graphs.npz"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--with-dist", action="store_true",
                    help="ablation: hand the model the distance channel it is "
                         "otherwise made to discover")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=float, default=FLOOR,
                    help="noise floor for the 'vs floor' column; re-estimate it "
                         "whenever the target set changes (gnn/floor_allobench.py)")
    a = ap.parse_args()

    data = list(np.load(a.cache, allow_pickle=True)["data"])
    n = len(data)
    rng = np.random.default_rng(a.seed)

    # Prefer folds carried in the data over folds invented here. The AlloBench route
    # ships a split grouped by UniProt accession, which is strictly stronger than the
    # protein-level grouping this script would otherwise do: it keeps homologues of the
    # same protein out of opposite sides of the split, not merely the same structure.
    # Re-splitting would throw that away silently, which is the whole reason the
    # converter carries the field.
    if all("fold" in r for r in data):
        fold = np.array([int(r["fold"]) for r in data])
        a.folds = len(np.unique(fold))
        grouping = f"UniProt-grouped ({a.folds} folds, carried from the dataset)"
    else:
        fold = rng.permutation(n) % a.folds
        grouping = f"protein-grouped ({a.folds} folds, assigned here)"
    print(f"split: {grouping}", flush=True)

    per = {k: {} for k in ("GNN", "alps", "ctrl_dist", "ctrl_random")}
    for k in range(a.folds):
        te = [i for i in range(n) if fold[i] == k]
        rest = [i for i in range(n) if fold[i] != k]
        # The inner validation split must be grouped too. The test fold is grouped by
        # UniProt, but carving validation out at random lets homologues of the same
        # accession sit on both sides -- and AlloBench carries up to six structures per
        # protein, so that is not hypothetical. It does not contaminate the test number,
        # which early stopping never sees; it corrupts the *selection criterion*, which
        # then prefers checkpoints good at recognising a family over ones that
        # generalise. On the curated set this was invisible: one structure per protein
        # means a random inner split is already effectively grouped.
        if all("uniprot" in data[i] for i in rest):
            byu = {}
            for i in rest:
                byu.setdefault(str(data[i]["uniprot"]), []).append(i)
            keys = list(byu)
            rng.shuffle(keys)
            want, va = max(3, len(rest) // 6), []
            for u in keys:
                if len(va) >= want:
                    break
                va.extend(byu[u])
            vaset = set(va)
            tr = [i for i in rest if i not in vaset]
            inner = "UniProt-grouped"
        else:
            rng.shuffle(rest)
            cut = max(3, len(rest) // 6)
            va, tr = rest[:cut], rest[cut:]
            inner = "random"
        model, best = train_fold(data, tr, va, a.with_dist, a.hidden, a.layers,
                                 a.epochs, a.lr, a.seed + k)
        print(f"fold {k}: train {len(tr)} val {len(va)} ({inner}) test {len(te)} "
              f"proteins, best val stratAUC {best:.3f}, {model.n_params} params",
              flush=True)
        per["GNN"].update(evaluate(model, data, te, a.with_dist))
        for i in te:
            r = data[i]
            for name, s in (("alps", r["alps"]),
                            ("ctrl_dist", -r["dist"]),
                            ("ctrl_random", rng.random(len(r["y"])))):
                auc, _ = stratified_auc(r["y"], np.asarray(s, float),
                                        r["pool"], r["dist"], 2.0)
                per[name][r["t"]] = auc

    tag = "GNN + dist channel" if a.with_dist else "GNN (anchor indicator only)"
    print(f"\n=== {n} targets, {grouping}, "
          f"distance-stratified AUC ===")
    print(f"    model: hidden {a.hidden}, {a.layers} layers   [{tag}]")
    print(f"    floor: {a.floor:.4f}" + ("  <-- curated-set default; re-estimate if "
          "this is not the curated set" if a.floor == FLOOR else "  (passed in)"))
    print(f"{'method':14s} {'strat AUC':>10s} {'vs floor':>9s} {'p vs random':>12s}")
    ref = per["ctrl_random"]
    order = sorted(per, key=lambda z: -np.nanmean(
        np.array(list(per[z].values()), float)))
    for name in order:
        keys = sorted(per[name])
        v = np.array([per[name][t] for t in keys], float)
        r_ = np.array([ref[t] for t in keys], float)
        ok = ~np.isnan(v) & ~np.isnan(r_)
        ps = ("reference" if name == "ctrl_random"
              else f"{stats.wilcoxon(v[ok], r_[ok]).pvalue:.4f}")
        print(f"{name:14s} {np.nanmean(v):10.3f} {np.nanmean(v)-a.floor:+9.3f} {ps:>12s}")

    keys = sorted(set(per["GNN"]) & set(per["alps"]))
    g = np.array([per["GNN"][t] for t in keys], float)
    c = np.array([per["alps"][t] for t in keys], float)
    ok = ~np.isnan(g) & ~np.isnan(c)
    print(f"\nGNN - ALPS: {np.mean(g[ok]-c[ok]):+.4f}   "
          f"paired p {stats.wilcoxon(g[ok], c[ok]).pvalue:.4f}   (n={ok.sum()})")


if __name__ == "__main__":
    main()

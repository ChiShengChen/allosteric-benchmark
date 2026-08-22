#!/usr/bin/env python
"""Gate 3 — quantum kernel against tuned classical kernels, identical features.

Design points that make this a real comparison rather than a demonstration:

* **Every kernel sees the same training subsample.** A precomputed Gram matrix is
  built once per fold from one balanced subsample, and each kernel is evaluated
  on exactly those points. Any difference is the kernel, not the sampling.
* **The classical side is tuned.** Gate 2 showed an untuned RBF sits at 0.506 and
  a tuned one at 0.563; comparing against the untuned default would hand the
  quantum kernel 0.057 of AUC for free.
* **The quantum side runs only where it can work.** Gate 1 showed the kernel
  collapses to the identity above bandwidth 0.25, so the sweep stays at 0.1 and
  below.
* **Protein-grouped folds and the stratified metric**, as everywhere else here —
  a learner given plain AUC will exploit the proximity confound instead of
  learning the task.
* **Class balance.** Positives are 2.4% of residues, so training subsamples take
  every positive and an equal number of negatives; `class_weight` alone leaves
  the kernel matrix dominated by negatives.

Cost note: the full 16k x 16k Gram matrix is not needed and not built. Training is
a balanced subsample of ~1-2k points; test residues enter through a cross-kernel.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kernels import linear_kernel, poly_kernel, quantum_kernel, rbf_kernel  # noqa: E402
from partial_auc import stratified_auc                                      # noqa: E402
from scipy import stats                                                     # noqa: E402
from sklearn.svm import SVC                                                 # noqa: E402

FLOOR = 0.4963


def kernel_bank(gammas=(10.0, 25.0, 50.0), bandwidths=(0.02, 0.05, 0.1)):
    bank = {"linear": lambda A, B: linear_kernel(A, B),
            "poly-4": lambda A, B: poly_kernel(A, B, degree=4)}
    for g in gammas:
        bank[f"RBF g={g:g}"] = (lambda g_: (lambda A, B: rbf_kernel(A, B, gamma=g_)))(g)
    for b in bandwidths:
        bank[f"quantum bw={b:g}"] = (
            lambda b_: (lambda A, B: quantum_kernel(A, B, bandwidth=b_)))(b)
    return bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "features.npz"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    z = np.load(a.cache, allow_pickle=True)
    data = list(z["data"])
    n = len(data)
    rng = np.random.default_rng(a.seed)
    fold = rng.permutation(n) % a.folds
    bank = kernel_bank()
    per_target = {k: {} for k in bank}
    per_target["ctrl_random"] = {}

    for k in range(a.folds):
        tr = [i for i in range(n) if fold[i] != k]
        te = [i for i in range(n) if fold[i] == k]
        Xtr = np.vstack([data[i]["X"][data[i]["pool"]] for i in tr])
        ytr = np.concatenate([data[i]["y"][data[i]["pool"]] for i in tr])

        pos = np.where(ytr == 1)[0]
        neg = np.where(ytr == 0)[0]
        n_neg = min(len(neg), int(len(pos) * a.neg_per_pos))
        sel = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
        if len(sel) > a.max_train:
            sel = rng.choice(sel, a.max_train, replace=False)
        Xs, ys = Xtr[sel], ytr[sel]
        print(f"fold {k}: train {len(Xs)} ({int(ys.sum())} positive), "
              f"test proteins {len(te)}", flush=True)

        for name, kf in bank.items():
            Ktr = kf(Xs, Xs)
            mdl = SVC(kernel="precomputed", C=1.0).fit(Ktr, ys)
            for i in te:
                d = data[i]
                Xte = d["X"][d["pool"]]
                s = np.zeros(len(d["y"]))
                s[d["pool"]] = mdl.decision_function(kf(Xte, Xs))
                auc, _ = stratified_auc(d["y"], s, d["pool"], d["dist"], 2.0)
                per_target[name][d["t"]] = auc
        for i in te:
            d = data[i]
            s = rng.random(len(d["y"]))
            auc, _ = stratified_auc(d["y"], s, d["pool"], d["dist"], 2.0)
            per_target["ctrl_random"][d["t"]] = auc

    names = sorted(per_target, key=lambda k_: -np.nanmean(
        np.array(list(per_target[k_].values()), float)))
    ref = per_target["ctrl_random"]
    print(f"\n=== gate 3: {n} curated targets, protein-grouped {a.folds}-fold, "
          f"distance-stratified AUC ===")
    print(f"{'kernel':16s} {'strat AUC':>10s} {'vs floor':>9s} {'p vs random':>12s}")
    best_classical = None
    for name in names:
        keys = sorted(per_target[name])
        v = np.array([per_target[name][t] for t in keys], float)
        r = np.array([ref[t] for t in keys], float)
        ok = ~np.isnan(v) & ~np.isnan(r)
        p = (np.nan if name == "ctrl_random"
             else stats.wilcoxon(v[ok], r[ok]).pvalue)
        ps = "reference" if np.isnan(p) else f"{p:.4f}"
        print(f"{name:16s} {np.nanmean(v):10.3f} {np.nanmean(v)-FLOOR:+9.3f} {ps:>12s}")
        if not name.startswith(("quantum", "ctrl")):
            if best_classical is None or np.nanmean(v) > best_classical[1]:
                best_classical = (name, float(np.nanmean(v)))

    print()
    bq = None
    for name in names:
        if name.startswith("quantum"):
            v = float(np.nanmean(np.array(list(per_target[name].values()), float)))
            if bq is None or v > bq[1]:
                bq = (name, v)
    if bq and best_classical:
        keys = sorted(set(per_target[bq[0]]) & set(per_target[best_classical[0]]))
        q = np.array([per_target[bq[0]][t] for t in keys], float)
        c = np.array([per_target[best_classical[0]][t] for t in keys], float)
        ok = ~np.isnan(q) & ~np.isnan(c)
        p = stats.wilcoxon(q[ok], c[ok]).pvalue
        print(f"best quantum  {bq[0]:16s} {bq[1]:.3f}")
        print(f"best classical {best_classical[0]:15s} {best_classical[1]:.3f}")
        print(f"quantum - classical: {np.mean(q[ok]-c[ok]):+.4f}   paired p {p:.4f}")


if __name__ == "__main__":
    main()

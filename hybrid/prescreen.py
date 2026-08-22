#!/usr/bin/env python
"""Gates 0-2. Run before writing any circuit.

Gate 0  does a classical learner beat unlearned ALPS, on curated labels and the
        distance-stratified metric?  If not, no ML component is worth adding.
Gate 1  geometric difference g between the quantum and classical Gram matrices.
        The literature result is that g well below sqrt(N) guarantees the classical
        model matches or beats the quantum one -- if that holds here, the answer is
        already proven for this feature set.
Gate 2  is there non-linear structure left to exploit?  Kernel effective rank, and
        the non-linearity gap (RBF minus linear).  A task a linear model already
        solves has no room for a richer kernel of any kind.
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

from methods.common import pocket_smooth, contact_graph, rank_percentile   # noqa: E402
from partial_auc import stratified_auc                                     # noqa: E402
from kernels import quantum_kernel, rbf_kernel                             # noqa: E402
from sklearn.linear_model import LogisticRegression                        # noqa: E402
from sklearn.svm import SVC                                                # noqa: E402
from scipy import stats                                                    # noqa: E402

FLOOR = 0.4963


def folds(n, k=5, seed=0):
    return np.random.default_rng(seed).permutation(n) % k


def gate0(data, feats):
    """Classical learner vs unlearned ALPS, protein-grouped CV, stratified metric."""
    n = len(data)
    fold = folds(n)
    alps_col = list(feats).index("alps")
    out = {"logistic": [], "ALPS alone": [], "ctrl_random": []}
    rng = np.random.default_rng(0)
    for k in range(5):
        tr = [i for i in range(n) if fold[i] != k]
        te = [i for i in range(n) if fold[i] == k]
        Xtr = np.vstack([data[i]["X"][data[i]["pool"]] for i in tr])
        ytr = np.concatenate([data[i]["y"][data[i]["pool"]] for i in tr])
        mdl = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        for i in te:
            d = data[i]
            A = contact_graph(d["Xcb"], 10.0) if "Xcb" in d else None
            p = np.zeros(len(d["y"]))
            p[d["pool"]] = mdl.predict_proba(d["X"][d["pool"]])[:, 1]
            for name, v in (("logistic", p),
                            ("ALPS alone", d["X"][:, alps_col]),
                            ("ctrl_random", rng.random(len(d["y"])))):
                auc, _ = stratified_auc(d["y"], v, d["pool"], d["dist"], 2.0)
                out[name].append(auc)
    print("GATE 0 — does learning help, on labels we trust?")
    ref = np.array(out["ctrl_random"], float)
    for name in ("logistic", "ALPS alone", "ctrl_random"):
        v = np.array(out[name], float)
        ok = ~np.isnan(v) & ~np.isnan(ref)
        p = (np.nan if name == "ctrl_random"
             else stats.wilcoxon(v[ok], ref[ok]).pvalue)
        ps = "reference" if np.isnan(p) else f"{p:.4f}"
        print(f"  {name:14s} stratified AUC {np.nanmean(v):.3f}   "
              f"vs floor {np.nanmean(v)-FLOOR:+.3f}   paired p {ps}")
    a = np.array(out["logistic"], float); b = np.array(out["ALPS alone"], float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    print(f"  learner vs ALPS: delta {np.mean(a[ok]-b[ok]):+.4f}  "
          f"p {stats.wilcoxon(a[ok], b[ok]).pvalue:.4f}")
    return float(np.nanmean(a) - np.nanmean(b))


def gate1(X, y, n_sub=400, seed=0):
    """Geometric difference between quantum and classical Gram matrices."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_sub, len(X)), replace=False)
    Xs = X[idx]
    Kc = rbf_kernel(Xs, Xs, gamma=1.0 / Xs.shape[1])
    Kq = quantum_kernel(Xs, Xs)
    # g = sqrt( || sqrt(Kq) Kc^-1 sqrt(Kq) ||_inf ) on normalised kernels
    def norm(K):
        d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
        return K / np.outer(d, d)
    Kc, Kq = norm(Kc), norm(Kq)
    m = len(Xs)
    Kc_inv = np.linalg.pinv(Kc + 1e-6 * np.eye(m))
    w, V = np.linalg.eigh(Kq)
    sq = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    M = sq @ Kc_inv @ sq
    g = float(np.sqrt(np.abs(np.linalg.eigvalsh(M)).max()))
    print("\nGATE 1 — geometric difference")
    print(f"  n = {m}   g = {g:.2f}   sqrt(n) = {np.sqrt(m):.2f}")
    print("  " + ("g >= sqrt(n): quantum kernel has headroom on this feature set"
                  if g >= np.sqrt(m) else
                  "g << sqrt(n): the classical model is GUARANTEED to match or beat it"))
    return g, float(np.sqrt(m))


def gate2(X, y, seed=0):
    """Effective rank of the classical kernel, and the non-linearity gap."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(1500, len(X)), replace=False)
    Xs, ys = X[idx], y[idx]
    K = rbf_kernel(Xs, Xs, gamma=1.0 / Xs.shape[1])
    w = np.clip(np.linalg.eigvalsh(K), 0, None)
    p = w / w.sum()
    eff = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    cut = int(0.7 * len(Xs))
    lin = SVC(kernel="linear", C=1.0).fit(Xs[:cut], ys[:cut]).score(Xs[cut:], ys[cut:])
    rbf = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xs[:cut], ys[:cut]).score(Xs[cut:], ys[cut:])
    print("\nGATE 2 — is there non-linear structure to exploit?")
    print(f"  kernel effective rank {eff:.1f} of {len(Xs)}  "
          f"({eff/len(Xs)*100:.1f}%)")
    print(f"  linear SVM {lin:.3f}   RBF SVM {rbf:.3f}   non-linearity gap {rbf-lin:+.3f}")
    return eff / len(Xs), float(rbf - lin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "features.npz"))
    a = ap.parse_args()
    z = np.load(a.cache, allow_pickle=True)
    data, feats = list(z["data"]), list(z["feats"])
    print(f"{len(data)} curated targets, features {feats}\n")
    X = np.vstack([d["X"][d["pool"]] for d in data])
    y = np.concatenate([d["y"][d["pool"]] for d in data])
    print(f"pooled residues {len(X)}   positives {int(y.sum())} "
          f"({y.mean()*100:.1f}%)\n")
    gate0(data, feats)
    gate1(X, y)
    gate2(X, y)


if __name__ == "__main__":
    main()

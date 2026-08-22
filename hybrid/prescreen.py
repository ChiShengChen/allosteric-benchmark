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
            p = np.zeros(len(d["y"]))
            p[d["pool"]] = mdl.predict_proba(d["X"][d["pool"]])[:, 1]
            for name, v in (("logistic", p),
                            ("ALPS alone", d["X"][:, alps_col]),
                            ("ctrl_random", rng.random(len(d["y"])))):
                auc, _ = stratified_auc(d["y"], v, d["pool"], d["dist"], 2.0)
                out[name].append(auc)
    print("GATE 0 — does learning help, on labels we trust?")
    print("  (no pocket smoothing here, so these are not directly comparable to")
    print("   README section 10.6; the learner-vs-ALPS contrast within the gate is)")
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
    """Geometric difference, swept over bandwidth, with the identity diagnostic.

    The criterion is one-directional: g well below sqrt(n) *proves* the classical
    model matches or beats the quantum one. A large g proves nothing in the other
    direction -- and an untuned quantum kernel is large-g precisely because it
    approaches the identity, which is the documented failure mode, not headroom.
    So the off-diagonal mass is reported next to g: a kernel whose off-diagonals
    have collapsed cannot generalise no matter what g says.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_sub, len(X)), replace=False)
    Xs = X[idx]
    m = len(Xs)

    def norm(K):
        d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
        return K / np.outer(d, d)

    Kc = norm(rbf_kernel(Xs, Xs, gamma=1.0 / Xs.shape[1]))
    Kc_inv = np.linalg.pinv(Kc + 1e-6 * np.eye(m))
    off = ~np.eye(m, dtype=bool)

    print("\nGATE 1 — geometric difference vs bandwidth")
    print(f"  n = {m}, sqrt(n) = {np.sqrt(m):.1f}")
    print(f"  {'bandwidth':>10s} {'g':>10s} {'mean |K_off|':>13s} {'verdict':>34s}")
    rows = []
    for bw in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        Kq = norm(quantum_kernel(Xs, Xs, bandwidth=bw))
        w, V = np.linalg.eigh(Kq)
        sq = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
        g = float(np.sqrt(np.abs(np.linalg.eigvalsh(sq @ Kc_inv @ sq)).max()))
        mo = float(np.abs(Kq[off]).mean())
        if mo < 0.05:
            verdict = "kernel ~ identity: cannot generalise"
        elif g < np.sqrt(m):
            verdict = "g < sqrt(n): classical guaranteed >="
        else:
            verdict = "no guarantee either way"
        print(f"  {bw:10.2f} {g:10.1f} {mo:13.3f} {verdict:>34s}")
        rows.append((bw, g, mo))
    return rows


def gate2(X, y, seed=0):
    """Is there non-linear structure to exploit -- with BOTH sides tuned?

    Two traps this gate fell into before, both worth naming because they are the
    standard ways this comparison goes wrong:

    * **Accuracy at 2.4% positives.** An all-negative classifier scores 0.976, so
      accuracy cannot separate a working model from a constant one. The first
      version reported a non-linearity gap of exactly +0.000 for that reason.
      Scored by AUC now.
    * **An untuned classical kernel.** With rank-percentile features in [0,1] and
      gamma = 1/d, every pairwise distance is small, the RBF matrix is nearly all
      ones, and its effective rank collapses to ~2. Comparing an untuned quantum
      kernel against an untuned classical one measures nothing. Gamma is swept.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(1500, len(X)), replace=False)
    Xs, ys = X[idx], y[idx]
    cut = int(0.7 * len(Xs))
    tr, te = slice(0, cut), slice(cut, None)

    print("\nGATE 2 — is there non-linear structure to exploit?")
    print(f"  positives in the sample: {ys.mean()*100:.1f}%  "
          f"(all-negative scores {1-ys.mean():.3f} on accuracy, hence AUC)")
    print(f"  {'kernel':>12s} {'param':>10s} {'eff. rank':>10s} {'AUC':>7s}")

    best = {}
    for gamma in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0):
        K = rbf_kernel(Xs, Xs, gamma=gamma)
        w = np.clip(np.linalg.eigvalsh(K), 0, None)
        p = w / w.sum()
        eff = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
        mdl = SVC(kernel="rbf", C=1.0, gamma=gamma,
                  class_weight="balanced").fit(Xs[tr], ys[tr])
        auc = roc_auc_score(ys[te], mdl.decision_function(Xs[te]))
        print(f"  {'RBF':>12s} {f'gamma={gamma:g}':>10s} {eff:10.1f} {auc:7.3f}")
        if auc > best.get("RBF", (0, None))[0]:
            best["RBF"] = (auc, gamma)
    for name, kw in (("linear", dict(kernel="linear", C=1.0)),
                     ("poly-4", dict(kernel="poly", degree=4, C=1.0, gamma="scale"))):
        mdl = SVC(class_weight="balanced", **kw).fit(Xs[tr], ys[tr])
        auc = roc_auc_score(ys[te], mdl.decision_function(Xs[te]))
        print(f"  {name:>12s} {'-':>10s} {'-':>10s} {auc:7.3f}")
        best[name] = (auc, None)

    gap = best["RBF"][0] - best["linear"][0]
    print(f"  best classical: RBF at gamma={best['RBF'][1]:g}, AUC {best['RBF'][0]:.3f}")
    print(f"  non-linearity gap (best RBF - linear) {gap:+.3f}")
    return best, float(gap)


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

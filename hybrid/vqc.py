#!/usr/bin/env python
"""Gate 3, second half — the shallow variational classifier.

The gate 3 plan named two quantum models. `run.py` ran the kernel; this runs the
circuit, on the same folds, the same features and the same metric, so the two are
directly comparable to each other and to the classical baselines.

Depth is set by the generalisation bound, not by taste. Generalisation error
scales as sqrt(T/N) in the number of trainable gates T, and 44 grouped proteins
afford tens of gates -- so the circuit is 8 qubits, 3 layers, **24 trainable
parameters**. A deeper circuit would score better in training and worse out of
fold, which is a failure mode this folder is specifically set up to avoid.

Architecture: data re-uploading. Each layer applies RY(pi*x_i) to encode, then
RY(theta) and a ring of CZ to entangle. Readout is the mean single-qubit <Z>,
through a trainable scale and bias, against a logistic loss. Simulation is exact
statevector in numpy -- at 8 qubits that is 256 amplitudes, so there is no
sampling noise to obscure the comparison.

The classical reference is a logistic regression on the identical features, which
is the fair opponent for a variational circuit: both are parametric models
trained by gradient descent on the same loss, differing only in the feature map.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from partial_auc import stratified_auc                      # noqa: E402
from scipy import stats                                     # noqa: E402
from scipy.optimize import minimize                         # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402

FLOOR = 0.4963
LAYERS = 3


def _ry(state, q, theta):
    """RY(theta) on qubit q, applied to a batch of statevectors (m, 2**n)."""
    m, dim = state.shape
    blk = 1 << q
    v = state.reshape(m, -1, 2, blk)
    a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    if np.ndim(theta):                       # per-sample angles
        c, s = c[:, None, None], s[:, None, None]
    out = np.empty_like(v)
    out[:, :, 0, :] = c * a0 - s * a1
    out[:, :, 1, :] = s * a0 + c * a1
    return out.reshape(m, dim)


def _cz_ring(state, n):
    """Ring of CZ gates: a sign flip wherever both qubits of a pair are 1."""
    dim = state.shape[1]
    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
    sign = np.ones(dim)
    for q in range(n):
        r = (q + 1) % n
        sign *= 1.0 - 2.0 * (bits[:, q] * bits[:, r])
    return state * sign[None, :]


def circuit(X, theta, layers=LAYERS):
    """Mean single-qubit <Z> of the data-re-uploading circuit, one per row."""
    X = np.asarray(X, float)
    m, n = X.shape
    dim = 1 << n
    state = np.zeros((m, dim))
    state[:, 0] = 1.0
    th = theta.reshape(layers, n)
    for L in range(layers):
        for q in range(n):
            state = _ry(state, q, np.pi * X[:, q])          # encode
        for q in range(n):
            state = _ry(state, q, th[L, q])                 # train
        state = _cz_ring(state, n)
    p = state ** 2
    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1)
    z = 1.0 - 2.0 * bits                                     # dim x n
    return (p @ z).mean(1)                                   # m


def _loss(w, X, y, layers):
    n_theta = layers * X.shape[1]
    f = circuit(X, w[:n_theta], layers) * w[n_theta] + w[n_theta + 1]
    f = np.clip(f, -30, 30)
    return float(np.mean(np.log1p(np.exp(f)) - y * f))


def fit_vqc(X, y, layers=LAYERS, restarts=2, maxiter=60, seed=0):
    """Train the circuit; keep the restart with the lowest training loss."""
    rng = np.random.default_rng(seed)
    n_theta = layers * X.shape[1]
    best = None
    for _ in range(restarts):
        w0 = np.concatenate([rng.normal(0, 0.3, n_theta), [4.0, 0.0]])
        r = minimize(_loss, w0, args=(X, y, layers), method="L-BFGS-B",
                     options=dict(maxiter=maxiter))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def predict_vqc(X, w, layers=LAYERS):
    n_theta = layers * X.shape[1]
    return circuit(X, w[:n_theta], layers) * w[n_theta] + w[n_theta + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "features.npz"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    z = np.load(a.cache, allow_pickle=True)
    data = list(z["data"])
    n = len(data)
    rng = np.random.default_rng(a.seed)
    fold = rng.permutation(n) % a.folds
    per = {k: {} for k in ("VQC (24 params)", "logistic", "ctrl_random")}

    for k in range(a.folds):
        tr = [i for i in range(n) if fold[i] != k]
        te = [i for i in range(n) if fold[i] == k]
        Xtr = np.vstack([data[i]["X"][data[i]["pool"]] for i in tr])
        ytr = np.concatenate([data[i]["y"][data[i]["pool"]] for i in tr])
        pos = np.where(ytr == 1)[0]
        neg = np.where(ytr == 0)[0]
        sel = np.concatenate([pos, rng.choice(neg, min(len(neg), len(pos)),
                                              replace=False)])
        if len(sel) > a.max_train:
            sel = rng.choice(sel, a.max_train, replace=False)
        Xs, ys = Xtr[sel], ytr[sel]

        w = fit_vqc(Xs, ys, seed=a.seed + k)
        lr = LogisticRegression(max_iter=2000).fit(Xs, ys)
        print(f"fold {k}: train {len(Xs)} ({int(ys.sum())} positive), "
              f"test proteins {len(te)}", flush=True)

        for i in te:
            d = data[i]
            Xte = d["X"][d["pool"]]
            for name, sc in (("VQC (24 params)", predict_vqc(Xte, w)),
                             ("logistic", lr.decision_function(Xte)),
                             ("ctrl_random", rng.random(len(Xte)))):
                s = np.zeros(len(d["y"]))
                s[d["pool"]] = sc
                auc, _ = stratified_auc(d["y"], s, d["pool"], d["dist"], 2.0)
                per[name][d["t"]] = auc

    print(f"\n=== gate 3b: {n} curated targets, protein-grouped {a.folds}-fold, "
          f"distance-stratified AUC ===")
    print(f"{'model':18s} {'strat AUC':>10s} {'vs floor':>9s} {'p vs random':>12s}")
    ref = per["ctrl_random"]
    order = sorted(per, key=lambda k_: -np.nanmean(
        np.array(list(per[k_].values()), float)))
    for name in order:
        keys = sorted(per[name])
        v = np.array([per[name][t] for t in keys], float)
        r = np.array([ref[t] for t in keys], float)
        ok = ~np.isnan(v) & ~np.isnan(r)
        ps = ("reference" if name == "ctrl_random"
              else f"{stats.wilcoxon(v[ok], r[ok]).pvalue:.4f}")
        print(f"{name:18s} {np.nanmean(v):10.3f} {np.nanmean(v)-FLOOR:+9.3f} "
              f"{ps:>12s}")

    keys = sorted(set(per["VQC (24 params)"]) & set(per["logistic"]))
    q = np.array([per["VQC (24 params)"][t] for t in keys], float)
    c = np.array([per["logistic"][t] for t in keys], float)
    ok = ~np.isnan(q) & ~np.isnan(c)
    print(f"\nVQC - logistic: {np.mean(q[ok]-c[ok]):+.4f}   "
          f"paired p {stats.wilcoxon(q[ok], c[ok]).pvalue:.4f}")


if __name__ == "__main__":
    main()

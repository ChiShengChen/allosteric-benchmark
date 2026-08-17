#!/usr/bin/env python
"""Which observable carries the allosteric signal?

Holds the graph, the perturbation and the distance conditioning fixed, and
varies only *what is read out* after stiffening residue i's neighbourhood.
Four readouts, all computed from the same pair of eigendecompositions:

  dlam    sum_k |lambda_k(H_i) - lambda_k(H_0)| / lambda_k(H_0)      (ALPS)
  dgap    same, on the level *spacings* lambda_{k+1} - lambda_k
          -> sensitive to a perturbation lifting a near-degeneracy, which is
             what governs long-time coherent transfer
  dpart   sum_k | V_ak(H_i)^2 - V_ak(H_0)^2 |  over the active-site rows
          -> change in how much the active site participates in the low modes
  dipr    change in the inverse participation ratio of the low modes
          -> change in how localised those modes are

dgap, dpart and dipr are the quantum-specific quantities: degeneracy structure
and eigenvector content, as opposed to plain eigenvalue magnitude. They are the
observables that would matter if interference were carrying the signal.

Usage:  python3 scripts/ablate_readouts.py [--targets data/targets]
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import sys
import warnings

import numpy as np
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from methods.alps import distance_zscore                                   # noqa: E402
from methods.common import (anchor_indices, contact_graph,                 # noqa: E402
                            distal_nonanchor_mask, laplacian,
                            min_dist_to_anchor, pocket_smooth,
                            rank_percentile)
from evaluate import auc_in_pool, permutation_p, top_k                     # noqa: E402

K = 3
RADIUS = 10.0
KAPPA = 1.0
LABEL = {"dlam": "|dlambda|  (ALPS)", "dgap": "d level spacing",
         "dpart": "d anchor participation", "dipr": "d mode IPR"}


def readouts(cb, anchor, radius=RADIUS, kappa=KAPPA, k=K, cutoff=10.0):
    A = contact_graph(cb, cutoff)
    a = anchor_indices(anchor)
    n = len(cb)

    w0, V0 = np.linalg.eigh(laplacian(A))
    nz = np.where(w0 > 1e-9)[0]
    lam0 = w0[nz][:k]                       # exactly K modes, as ALPS uses
    gap0 = np.diff(w0[nz][:k + 1])          # K spacings needs K+1 levels
    Va0 = V0[:, nz][a][:, :k]
    ipr0 = (V0[:, nz][:, :k] ** 4).sum(axis=0)

    D = cdist(cb, cb)
    out = {key: np.zeros(n) for key in LABEL}
    for i in range(n):
        nb = np.where(D[i] <= radius)[0]
        W = A.copy()
        sub = np.ix_(nb, nb)
        W[sub] = W[sub] * (1.0 + kappa)
        wp, Vp = np.linalg.eigh(laplacian(W))
        nzp = np.where(wp > 1e-9)[0]

        lam = wp[nzp][:k]
        m = min(len(lam), len(lam0))
        out["dlam"][i] = np.sum(np.abs(lam[:m] - lam0[:m]) / (lam0[:m] + 1e-12))

        gap = np.diff(wp[nzp][:k + 1])
        mg = min(len(gap), len(gap0))
        out["dgap"][i] = np.sum(np.abs(gap[:mg] - gap0[:mg]) / (gap0[:mg] + 1e-12))

        Vap = Vp[:, nzp][a][:, :k]
        mv = min(Vap.shape[1], Va0.shape[1])
        out["dpart"][i] = np.abs(Vap[:, :mv] ** 2 - Va0[:, :mv] ** 2).sum()

        iprp = (Vp[:, nzp][:, :k] ** 4).sum(axis=0)
        mi = min(len(iprp), len(ipr0))
        out["dipr"][i] = np.sum(np.abs(iprp[:mi] - ipr0[:mi]) / (ipr0[:mi] + 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets"))
    ap.add_argument("--max-n", type=int, default=660)
    a = ap.parse_args()

    agg = collections.defaultdict(list)
    files = sorted(glob.glob(os.path.join(a.targets, "*.npz")))
    for f in files:
        d = np.load(f)
        cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
        if len(cb) > a.max_n or y.sum() == 0:
            continue
        A = contact_graph(cb, 10.0)
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        bg = pool & (y != 1)
        if bg.sum() < max(3, int(y.sum())):
            continue
        dist = min_dist_to_anchor(cb, anchor)
        R = readouts(cb, anchor)
        for key, v in R.items():
            s = pocket_smooth(rank_percentile(distance_zscore(v, dist, pool)), A)
            t5 = top_k(s, pool, 5)
            agg[key].append((permutation_p(y, s, bg), auc_in_pool(y, s, pool),
                             int(bool(y[t5].sum()))))
        print(f"  {os.path.basename(f)}", flush=True)

    n = len(agg["dlam"])
    print(f"\nreadout ablation on {os.path.basename(a.targets)} (n={n}) — "
          f"same graph, same perturbation, same distance conditioning")
    print(f"{'readout':24s} {'sig':>7s} {'medP':>8s} {'AUC':>7s} {'hit5':>7s}")
    for key in ("dlam", "dgap", "dpart", "dipr"):
        v = agg[key]
        p = np.array([x[0] for x in v])
        au = np.array([x[1] for x in v])
        h = np.array([x[2] for x in v])
        print(f"{LABEL[key]:24s} {np.mean(p < 0.05) * 100:6.1f}% "
              f"{np.median(p):8.4f} {au.mean():7.3f} {h.mean() * 100:6.1f}%")


if __name__ == "__main__":
    main()

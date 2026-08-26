#!/usr/bin/env python
"""How precisely must an eigenvalue be known before ALPS stops working?

The one quantum framing this repository has not closed is the cost argument in
README section 6: ALPS reads only the lowest few Kirchhoff eigenvalues, never the
eigenvectors, so quantum phase estimation would return exactly the scalar the score
needs and would not require a full diagonalisation. That framing survives because it
never claims better predictions -- only a cheaper route to the same ones.

But "cheaper" is not free, and QPE's cost is set by the precision demanded: circuit
depth scales as O(1/epsilon). So the question that decides whether the framing is
worth implementing is not "can QPE do this" -- it can, the operator is Hermitian and
PSD -- but **how small must epsilon be**. That is a purely classical question, and
this script answers it before anyone writes a circuit.

Same shape as gate 1 in hybrid/: sweep a parameter classically until the quantum
route is either bounded or ruled out, rather than implementing first and discovering
the constraint afterwards.

Three things are measured.

**1. The size of the signal.** ALPS scores

    out[i] = sum_k |lambda_k(H_i) - lambda_k(H_0)| / lambda_k(H_0)

which is a difference of two nearly equal numbers. Its typical magnitude *is* the
precision requirement: an estimator whose error is comparable to the shift it is
trying to resolve measures nothing. This number has never been written down here.

**2. Where the AUC breaks.** Each eigenvalue is multiplied by (1 + eta) with eta drawn
uniformly from [-epsilon, epsilon] -- QPE returns a value inside a resolution window
rather than a Gaussian around the truth, so uniform is the honest noise model. The
sweep reports the distance-stratified AUC as epsilon grows, against the noiseless
score on identical targets and the same metric used everywhere else here.

**3. Whether the eigendecompositions are needed at all.** First-order perturbation
theory gives lambda_k(H_i) - lambda_k(H_0) = v_k^T dL_i v_k, which costs one
decomposition of H_0 for the whole protein instead of one per residue. kappa = 2
stiffens contacts threefold, so first order has no right to be accurate -- but ALPS
only ever uses the score as a *ranking*, and an approximation that preserves order is
enough. If it holds, the classical cost this script's own motivation rests on drops by
a factor of N, and the cost argument for QPE loses its object. Worth knowing either
way, which is why it is measured here rather than assumed.

Exact eigenvalues are computed once per target and cached, so the epsilon sweep and
the perturbation comparison are nearly free after the first pass.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

from methods.alps import (BANDWIDTH, KAPPA, K_MODES, RADIUS,     # noqa: E402
                          distance_zscore)
from methods.common import (contact_graph, distal_nonanchor_mask,  # noqa: E402
                            laplacian, min_dist_to_anchor)
from partial_auc import stratified_auc                            # noqa: E402


def exact_spectrum(cb, cutoff, radius, kappa, k_modes):
    """Base and per-residue perturbed eigenvalues, plus the first-order prediction.

    Returns (base[k], pert[i,k], first[i,k]). The eigenvectors are taken from the one
    decomposition of the unperturbed Laplacian that first-order theory needs, so the
    comparison costs a single extra eigh rather than a second sweep.
    """
    A = contact_graph(cb, cutoff)
    L0 = laplacian(A)
    w, V = np.linalg.eigh(L0)
    nz = np.where(w > 1e-9)[0][:k_modes]
    base = w[nz]
    Vk = V[:, nz]

    n = len(cb)
    D = cdist(cb, cb)
    pert = np.full((n, k_modes), np.nan)
    first = np.full((n, k_modes), np.nan)
    for i in range(n):
        nb = np.where(D[i] <= radius)[0]
        W = A.copy()
        sub = np.ix_(nb, nb)
        W[sub] = W[sub] * (1.0 + kappa)
        Li = laplacian(W)
        wi = np.linalg.eigvalsh(Li)
        wi = wi[wi > 1e-9][:k_modes]
        if len(wi) == k_modes:
            pert[i] = wi
        # first order: lambda_k + v_k^T (L_i - L_0) v_k
        dL = Li - L0
        first[i] = base + np.einsum("nk,nm,mk->k", Vk, dL, Vk)
    return base, pert, first


def score_from_spectrum(base, pert, dist, pool, eps=0.0, rng=None):
    """The ALPS raw score, optionally with each eigenvalue perturbed by up to eps.

    Both the base and the perturbed eigenvalues are corrupted, because an estimator
    that has to measure the shift has to measure both ends of it.
    """
    b, p = base.copy(), pert.copy()
    if eps > 0:
        b = b * (1.0 + rng.uniform(-eps, eps, size=b.shape))
        p = p * (1.0 + rng.uniform(-eps, eps, size=p.shape))
    raw = np.nansum(np.abs(p - b[None, :]) / (b[None, :] + 1e-12), axis=1)
    raw[np.isnan(pert).any(axis=1)] = 0.0
    return distance_zscore(raw, dist, pool, BANDWIDTH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_curated"))
    ap.add_argument("--max-n", type=int, default=400,
                    help="cap on residues; the sweep needs N+1 dense eigensolves per "
                         "target and the ranking question does not need the big ones")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "precision_budget.json"))
    a = ap.parse_args()

    eps_grid = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    cached, files = [], sorted(glob.glob(os.path.join(a.targets, "*.npz")))
    for f in files:
        if len(cached) >= a.limit:
            break
        z = np.load(f, allow_pickle=True)
        cb, anchor, y = z["cb"], z["anchor"], z["y"].astype(int)
        if len(cb) > a.max_n:
            continue
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if y.sum() == 0 or (pool & (y == 1)).sum() == 0:
            continue
        dist = min_dist_to_anchor(cb, anchor)
        base, pert, first = exact_spectrum(cb, 10.0, RADIUS, KAPPA, K_MODES)
        cached.append(dict(t=os.path.basename(f)[:-4], y=y, pool=pool, dist=dist,
                           base=base, pert=pert, first=first))
        print(f"  {cached[-1]['t']:12s} N={len(cb):4d}", flush=True)
    print(f"{len(cached)} targets cached\n", flush=True)

    # 1. how big is the signal being measured
    rel = np.concatenate([np.abs(c["pert"] - c["base"][None, :]) / c["base"][None, :]
                          for c in cached])
    rel = rel[np.isfinite(rel)]
    q = np.percentile(rel, [5, 25, 50, 75, 95])
    print("relative eigenvalue shift |dlambda|/lambda, pooled over residues and modes")
    print(f"  median {q[2]:.3e}   IQR [{q[1]:.3e}, {q[3]:.3e}]   "
          f"5-95% [{q[0]:.3e}, {q[4]:.3e}]")
    print("  -- an estimator with error at this scale resolves nothing\n", flush=True)

    # 2. where the AUC breaks
    print(f"{'epsilon':>10s} {'strat AUC':>10s} {'sd over seeds':>14s} {'vs exact':>9s}")
    rows, exact_auc = [], None
    for eps in eps_grid:
        per_seed = []
        for s in range(1 if eps == 0 else a.seeds):
            rng = np.random.default_rng(700 + s)
            v = []
            for c in cached:
                sc = score_from_spectrum(c["base"], c["pert"], c["dist"], c["pool"],
                                         eps, rng)
                auc, _ = stratified_auc(c["y"], sc, c["pool"], c["dist"], 2.0)
                v.append(auc)
            per_seed.append(np.nanmean(v))
        m, sd = float(np.mean(per_seed)), float(np.std(per_seed))
        if eps == 0:
            exact_auc = m
        rows.append(dict(eps=eps, auc=m, sd=sd))
        print(f"{eps:10.0e} {m:10.3f} {sd:14.4f} {m - exact_auc:+9.3f}", flush=True)

    # 3. does first-order perturbation theory preserve the ranking
    rho, fo = [], []
    for c in cached:
        ok = np.isfinite(c["pert"]).all(axis=1) & np.isfinite(c["first"]).all(axis=1)
        e = np.nansum(np.abs(c["pert"] - c["base"][None, :]) / c["base"][None, :], axis=1)
        f = np.nansum(np.abs(c["first"] - c["base"][None, :]) / c["base"][None, :], axis=1)
        if ok.sum() > 10:
            rho.append(spearmanr(e[ok], f[ok]).statistic)
        sc = distance_zscore(np.where(ok, f, 0.0), c["dist"], c["pool"], BANDWIDTH)
        auc, _ = stratified_auc(c["y"], sc, c["pool"], c["dist"], 2.0)
        fo.append(auc)
    print(f"\nfirst-order perturbation theory, one eigendecomposition per protein "
          f"instead of N+1")
    print(f"  Spearman against the exact raw score: median {np.nanmedian(rho):.3f}, "
          f"{np.nanmean(np.array(rho) > 0.9):.0%} of targets above 0.9")
    print(f"  stratified AUC {np.nanmean(fo):.3f} against exact {exact_auc:.3f} "
          f"({np.nanmean(fo) - exact_auc:+.3f})")

    json.dump(dict(n_targets=len(cached), shift_quantiles=q.tolist(), sweep=rows,
                   exact_auc=exact_auc, first_order_auc=float(np.nanmean(fo)),
                   first_order_spearman=float(np.nanmedian(rho))),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

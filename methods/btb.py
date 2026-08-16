"""Bond-to-bond propensity, adapted to a residue contact graph.

Reference method (Amor/Schaub/Yaliraki/Barahona line): build an energy-weighted
graph of the structure, form the edge-to-edge transfer matrix

    M = (1/2) G B^T L^dagger B

with B the incidence matrix, G the diagonal of edge weights and L^dagger the
Moore-Penrose pseudo-inverse of the weighted Laplacian; seed from the active
site by summing |M| over the source edges; then assess significance with a
conditional quantile regression of log-propensity against distance from the
active site, which removes the distance decay.

The published method uses an *atomistic* bond graph. Here the same operator is
applied to the residue contact graph, because that is the only input QASC has.
The authors of the original method explicitly warn that residue-level
coarse-graining can lose the signal for some proteins, so this is a faithful
port of the operator, not a claim of equivalent accuracy.

Never forms the full m x m transfer matrix: only its source columns are needed.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .common import (anchor_indices, incidence, laplacian, min_dist_to_anchor,
                     weighted_contact_graph)

__all__ = ["btb_propensity", "quantile_residual", "btb_scores"]


def btb_propensity(adj: np.ndarray, anchor, degree_normalize: bool = True) -> np.ndarray:
    """Per-residue raw bond-to-bond propensity seeded at the anchor."""
    a = anchor_indices(anchor)
    B, g, edges = incidence(adj)
    n, m = B.shape
    L = laplacian(adj)
    Ldag = np.linalg.pinv(L)

    # source edges: any edge with at least one endpoint in the active site
    in_anchor = np.zeros(n, dtype=bool)
    in_anchor[a] = True
    src = np.where(in_anchor[edges[:, 0]] | in_anchor[edges[:, 1]])[0]
    if len(src) == 0:
        return np.zeros(n)

    # M[:, src] = 1/2 * G B^T Ldag B[:, src]   (n x |src| intermediates only)
    X = Ldag @ B[:, src]                 # n x |src|
    Msrc = 0.5 * g[:, None] * (B.T @ X)  # m x |src|

    pi_edge = np.abs(Msrc).sum(axis=1)
    pi_edge /= (pi_edge.sum() + 1e-300)

    # residue propensity = sum of incident edge propensities
    out = np.zeros(n)
    np.add.at(out, edges[:, 0], pi_edge)
    np.add.at(out, edges[:, 1], pi_edge)
    if degree_normalize:
        # a residue with more contacts collects more incident edges; without
        # this the propensity mostly re-reads the contact degree
        ndeg = np.zeros(n)
        np.add.at(ndeg, edges[:, 0], 1.0)
        np.add.at(ndeg, edges[:, 1], 1.0)
        out = out / np.maximum(ndeg, 1.0)
    return out


def _pinball(params, x, y, tau):
    r = y - (params[0] + params[1] * x)
    return np.sum(np.where(r >= 0, tau * r, (tau - 1.0) * r))


def quantile_residual(values: np.ndarray, dist: np.ndarray,
                      tau: float = 0.5, pool: np.ndarray | None = None) -> np.ndarray:
    """Residual above a linear conditional-quantile fit of log(values) on dist.

    This is the step that removes the distance decay: a residue scores highly
    only if it beats other residues at a *comparable* distance from the active
    site.
    """
    v = np.log(np.asarray(values, float) + 1e-300)
    d = np.asarray(dist, float)
    ok = np.isfinite(v) & np.isfinite(d)
    if pool is not None:
        # fit the trend on the pool the model actually ranks within, otherwise
        # the fit is dominated by the near-anchor residues that are never
        # candidates anyway
        ok = ok & np.asarray(pool, bool)
    if ok.sum() < 5:
        return np.zeros_like(v)
    x, yv = d[ok], v[ok]
    start = np.array([np.median(yv), 0.0])
    res = minimize(_pinball, start, args=(x, yv, tau), method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-9})
    b0, b1 = res.x
    return v - (b0 + b1 * d)


def btb_scores(cb: np.ndarray, anchor, cutoff: float = 10.0,
               sigma: float = 4.0, tau: float = 0.5,
               distance_corrected: bool = True,
               pool: np.ndarray | None = None) -> np.ndarray:
    """Full residue-level bond-to-bond score from Cbeta coordinates + anchor."""
    adj = weighted_contact_graph(cb, cutoff=cutoff, sigma=sigma)
    pi = btb_propensity(adj, anchor)
    if not distance_corrected:
        return pi
    d = min_dist_to_anchor(cb, anchor)
    return quantile_residual(pi, d, tau=tau, pool=pool)

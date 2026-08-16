"""Shared graph utilities: contact graph, incidence/Laplacian, rank helpers.

Everything here works from the same minimal input QASC uses:
per-residue Cbeta coordinates + active-site (anchor) residue indices.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = [
    "contact_graph", "weighted_contact_graph", "incidence", "laplacian",
    "min_dist_to_anchor", "rank_percentile", "anchor_indices",
    "distal_nonanchor_mask", "pocket_smooth",
]


def anchor_indices(anchor) -> np.ndarray:
    return np.asarray(sorted({int(i) for i in anchor}), dtype=int)


def contact_graph(cb: np.ndarray, cutoff: float = 10.0) -> np.ndarray:
    """Unit-weighted residue contact adjacency (QASC's definition)."""
    d = squareform(pdist(np.asarray(cb, float)))
    n = d.shape[0]
    return ((d <= cutoff) & ~np.eye(n, dtype=bool)).astype(float)


def weighted_contact_graph(cb: np.ndarray, cutoff: float = 10.0,
                           sigma: float = 4.0) -> np.ndarray:
    """Distance-weighted contact adjacency, w_ij = exp(-(d/sigma)^2) inside cutoff.

    The bond-to-bond formalism weights each edge by an interaction energy; with
    only Cbeta coordinates available the smooth distance kernel is the natural
    stand-in and it removes the hard cutoff's discontinuity.
    """
    d = squareform(pdist(np.asarray(cb, float)))
    n = d.shape[0]
    w = np.exp(-(d / float(sigma)) ** 2)
    w[d > cutoff] = 0.0
    w[np.eye(n, dtype=bool)] = 0.0
    return w


def incidence(adj: np.ndarray):
    """Oriented incidence matrix B (n x m) and edge weight vector g (m,).

    Returns (B, g, edges) where edges is an (m, 2) int array of (i, j), i < j.
    """
    adj = np.asarray(adj, float)
    iu = np.triu_indices_from(adj, k=1)
    mask = adj[iu] > 0
    ei, ej = iu[0][mask], iu[1][mask]
    m = len(ei)
    n = adj.shape[0]
    B = np.zeros((n, m))
    B[ei, np.arange(m)] = 1.0
    B[ej, np.arange(m)] = -1.0
    g = adj[ei, ej].astype(float)
    return B, g, np.stack([ei, ej], axis=1)


def laplacian(adj: np.ndarray) -> np.ndarray:
    """Weighted graph Laplacian L = diag(sum_j w_ij) - W."""
    adj = np.asarray(adj, float)
    return np.diag(adj.sum(axis=1)) - adj


def min_dist_to_anchor(cb: np.ndarray, anchor) -> np.ndarray:
    a = anchor_indices(anchor)
    return cdist(np.asarray(cb, float), np.asarray(cb, float)[a]).min(axis=1)


def rank_percentile(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return np.full(n, 0.5)
    return np.argsort(np.argsort(v)) / (n - 1)


def distal_nonanchor_mask(cb: np.ndarray, anchor, distal: float = 8.0) -> np.ndarray:
    a = anchor_indices(anchor)
    d = min_dist_to_anchor(cb, a)
    is_a = np.zeros(len(cb), dtype=bool)
    is_a[a] = True
    return (d >= float(distal)) & ~is_a


def pocket_smooth(scores: np.ndarray, adj: np.ndarray, alpha: float = 0.3,
                  iters: int = 2) -> np.ndarray:
    """QASC's graph diffusion smoothing, reused so every method is compared
    under the same post-processing."""
    scores = np.asarray(scores, float)
    adj = np.asarray(adj, float)
    deg = adj.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / deg, 0.0)
    P = adj * dinv[:, None]
    finite = np.isfinite(scores)
    x = np.where(finite, scores, 0.0)
    for _ in range(int(iters)):
        x = (1.0 - alpha) * x + alpha * (P @ x)
    return np.where(finite, x, -np.inf)

"""Elastic-network readouts of the same Kirchhoff matrix QASC already builds.

QASC uses H = diag(deg) - A as a CTQW Hamiltonian and takes exactly one number
out of it (the infinite-time-averaged transfer probability). The same matrix is
the Gaussian Network Model operator, and the ENM literature reads much more
from it:

* :func:`corrsite_scores`  -- CorrSite2.0-style: correlation of each residue's
  motion with the active site, computed separately from the slowest and the
  fastest modes, scored as the maximum of the two Z-scores. The published
  finding is that allosteric/orthosteric coupling is dominated by *either* fast
  *or* slow modes depending on the pair, so taking the max is the point.
* :func:`apop_scores`      -- APOP-style: stiffen the springs across a candidate
  site to emulate ligand binding and rank by the induced shift in the global
  (slowest) mode frequencies. APOP scores whole pockets; with no pocket
  detector available the candidate site here is a residue together with its
  spatial neighbourhood, which is a residue-level surrogate.
* :func:`prs_scores`       -- perturbation-response scanning: apply a unit force
  at each residue in turn and measure the response at the active site through
  the ENM covariance.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .common import anchor_indices, contact_graph, laplacian

__all__ = ["gnm_covariance", "corrsite_scores", "apop_scores", "prs_scores"]


def gnm_covariance(adj: np.ndarray, modes: np.ndarray, w: np.ndarray,
                   V: np.ndarray) -> np.ndarray:
    """GNM covariance restricted to a subset of modes: sum_k V_k V_k^T / w_k."""
    return (V[:, modes] / w[modes]) @ V[:, modes].T


def _corr_to_anchor(C: np.ndarray, a: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    R = C / np.outer(d, d)
    return np.abs(R[:, a]).mean(axis=1)


def corrsite_scores(cb: np.ndarray, anchor, cutoff: float = 10.0,
                    n_slow: int = 10, n_fast: int = 10) -> np.ndarray:
    """Max of slow-mode and fast-mode motion-correlation Z-scores."""
    adj = contact_graph(cb, cutoff)
    K = laplacian(adj)
    w, V = np.linalg.eigh(K)
    nz = np.where(w > 1e-9)[0]
    if len(nz) < max(n_slow, n_fast):
        n_slow = n_fast = max(1, len(nz) // 2)
    a = anchor_indices(anchor)

    cs = _corr_to_anchor(gnm_covariance(adj, nz[:n_slow], w, V), a)
    cf = _corr_to_anchor(gnm_covariance(adj, nz[-n_fast:], w, V), a)
    zs = (cs - cs.mean()) / (cs.std() + 1e-12)
    zf = (cf - cf.mean()) / (cf.std() + 1e-12)
    return np.maximum(zs, zf)


def apop_scores(cb: np.ndarray, anchor=None, cutoff: float = 10.0,
                radius: float = 8.0, k_stiff: float = 1.0,
                n_global: int = 5) -> np.ndarray:
    """Shift in the slowest global mode frequencies when a local site is stiffened.

    Note this readout does not use the anchor at all -- APOP ranks pockets by
    their global mechanical leverage, independent of where the active site is.
    """
    adj = contact_graph(cb, cutoff)
    w0, _ = np.linalg.eigh(laplacian(adj))
    base = w0[w0 > 1e-9][:n_global]
    D = cdist(np.asarray(cb, float), np.asarray(cb, float))
    n = len(cb)
    out = np.zeros(n)
    for i in range(n):
        nb = np.where(D[i] <= radius)[0]
        W = adj.copy()
        sub = np.ix_(nb, nb)
        W[sub] = W[sub] * (1.0 + k_stiff)
        wp, _ = np.linalg.eigh(laplacian(W))
        pert = wp[wp > 1e-9][:n_global]
        out[i] = np.sum(np.abs(pert - base) / (base + 1e-12))
    return out


def prs_scores(cb: np.ndarray, anchor, cutoff: float = 10.0,
               n_modes: int | None = None) -> np.ndarray:
    """Perturbation-response scanning on the GNM.

    The GNM covariance C = K^dagger already is the linear response operator:
    perturbing residue i produces response C[:, i]. The score of residue i is
    its mean response magnitude at the active-site residues, i.e. how strongly
    a perturbation there is felt at the catalytic site.
    """
    adj = contact_graph(cb, cutoff)
    K = laplacian(adj)
    w, V = np.linalg.eigh(K)
    nz = np.where(w > 1e-9)[0]
    if n_modes:
        nz = nz[:n_modes]
    C = (V[:, nz] / w[nz]) @ V[:, nz].T
    a = anchor_indices(anchor)
    return np.abs(C[:, a]).mean(axis=1)

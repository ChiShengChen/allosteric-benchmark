"""Quantum-walk channels: QASC's originals plus the literature-motivated fixes.

Three things are implemented here that QASC does not do:

1. :func:`ipr_resonant_transfer` with ``seed="degree"``. QASC seeds the
   *adjacency-matrix* walk with a uniform amplitude on the anchor. The uniform
   superposition is a zero-eigenvalue eigenvector of the Laplacian, but it is
   not an eigenvector of the adjacency matrix, so under an adjacency-generated
   walk it drifts even with no driving; the natural initial state for an
   adjacency walk is degree-dependent. This switch makes that fix a one-line
   experiment.

2. :func:`ctqw_communicability` with ``normalized=True``: the symmetric
   normalized Laplacian, which removes the degree heterogeneity that makes
   coherent walks behave badly on hub-dominated graphs.

3. :func:`enaqt_transfer`: Lindblad pure-dephasing transport with the dephasing
   rate calibrated to the hopping scale (gamma in units of the largest coupling)
   and efficiency defined as a finite-window time integral of site occupation,
   which is the figure of merit used in the ENAQT literature.
"""
from __future__ import annotations

import numpy as np

from .common import anchor_indices, contact_graph, laplacian

__all__ = [
    "communicability_matrix", "ctqw_communicability", "ipr_resonant_transfer",
    "enaqt_transfer", "noisy_or",
]


def communicability_matrix(H: np.ndarray) -> np.ndarray:
    """Infinite-time-averaged CTQW transfer probability, C_ij = sum_k V_ik^2 V_jk^2."""
    _w, V = np.linalg.eigh(np.asarray(H, float))
    v2 = V ** 2
    C = v2 @ v2.T
    return 0.5 * (C + C.T)


def _normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return np.eye(len(adj)) - (adj * dinv[:, None]) * dinv[None, :]


def ctqw_communicability(adj: np.ndarray, anchor,
                         normalized: bool = False) -> np.ndarray:
    """Anchor-reduced CTQW communicability (QASC's channel 1)."""
    a = anchor_indices(anchor)
    H = _normalized_laplacian(adj) if normalized else laplacian(adj)
    return communicability_matrix(H)[a, :].mean(axis=0)


def ipr_resonant_transfer(adj: np.ndarray, anchor, gamma: float = 2.0,
                          seed: str = "uniform") -> np.ndarray:
    """IPR-weighted resonant transfer on the adjacency Hamiltonian (channel 2).

    seed="uniform" reproduces QASC exactly. seed="degree" uses the
    degree-weighted anchor state that is the natural initial condition for an
    adjacency-generated walk.
    """
    adj = np.asarray(adj, float)
    a = anchor_indices(anchor)
    n = adj.shape[0]
    deg = adj.sum(axis=1)

    _w, V = np.linalg.eigh(adj)
    p0 = np.zeros(n)
    if seed == "degree":
        # amplitude proportional to sqrt(degree): the stationary-in-A analogue
        # of a uniform state on a regular graph
        wgt = np.sqrt(np.clip(deg[a], 1e-12, None))
        p0[a] = wgt / np.linalg.norm(wgt)
    else:
        p0[a] = 1.0 / np.sqrt(len(a))

    c = V.T @ p0
    ipr = (V ** 4).sum(axis=0)
    weights = (c ** 2) * (ipr ** float(gamma))
    rt = (V ** 2) @ weights
    return rt / (deg + 1.0)


def enaqt_transfer(adj: np.ndarray, anchor, gamma_rel: float = 1.0,
                   t_max: float | None = None, nsteps: int = 120) -> np.ndarray:
    """Dephasing-assisted transport; efficiency = time-integrated occupation.

    gamma_rel is the dephasing rate *in units of the largest coupling* J_max,
    because the ENAQT optimum sits where the noise rate matches the hopping
    scale. t_max defaults to a few coherent transit times (2*pi/J_max * 5).
    """
    adj = np.asarray(adj, float)
    H = laplacian(adj)
    n = adj.shape[0]
    a = anchor_indices(anchor)

    jmax = float(np.abs(adj).max()) or 1.0
    gamma = gamma_rel * jmax
    if t_max is None:
        t_max = 5.0 * 2.0 * np.pi / jmax

    psi0 = np.zeros(n)
    psi0[a] = 1.0 / np.sqrt(len(a))
    rho = np.outer(psi0, psi0).astype(complex)
    off = 1.0 - np.eye(n)
    dt = t_max / nsteps
    acc = np.zeros(n)

    def deriv(r):
        return -1j * (H @ r - r @ H) - gamma * off * r

    for _ in range(nsteps):
        k1 = deriv(rho)
        k2 = deriv(rho + dt * k1)
        rho = rho + 0.5 * dt * (k1 + k2)
        acc += np.real(np.diag(rho)) * dt
    return acc / t_max


def noisy_or(*channels, weights=None) -> np.ndarray:
    """QASC's non-diluting fusion, generalised to any number of channels.

    s = 1 - prod_k (1 - w_k * rank(channel_k)); the first channel enters at
    full strength by convention.
    """
    from .common import rank_percentile
    if weights is None:
        weights = [1.0] + [0.5] * (len(channels) - 1)
    out = np.ones(len(channels[0]))
    for ch, w in zip(channels, weights):
        out = out * (1.0 - float(w) * rank_percentile(ch))
    return 1.0 - out

"""Chiral quantum walks: the one quantum insertion point whose precondition holds.

Every interference-based observable tested so far failed for the same reason —
they need eigenvalue degeneracy and residue contact graphs have almost none
(3.6% of low-lying gaps below 1%). A chiral walk does not need degeneracy. It
attaches complex phases to the hoppings, and those phases are physical only
through their gauge-invariant flux around **cycles**, of which contact graphs
have an abundance: 7.7-8.3 independent cycles and thousands of triangles per
protein.

Three design choices follow directly from the literature (see
`docs/quantum-observable-search.md`):

1. **Adjacency, not Laplacian.** For Laplacian-type walks the degree diagonal
   blocks transport between vertices of very different degree and chirality
   cannot overcome it. Contact graphs are degree-heterogeneous, so the chiral
   generator here is the adjacency matrix.

2. **Read the directional asymmetry, not the transport.** Raw transfer
   probability is the proximity ranker that already failed. The part of it that
   *only* chirality can produce is the forward/backward asymmetry

       d(i) = p(anchor -> i) - p(i -> anchor)

   which is **identically zero for any time-reversal-symmetric Hamiltonian**:
   for real symmetric H, U(-t) = U(t)* so the two directions match exactly.
   Any nonzero signal is therefore purely chiral in origin, and the score cannot
   silently degenerate into the observable we rejected. Allosteric signalling is
   directional; a real symmetric Laplacian structurally cannot represent that.

3. **Peierls phases, not free parameters.** The gauge-invariant degrees of
   freedom number E - N + 1, i.e. 2500-5200 per protein — not searchable. A
   uniform "magnetic field" threading flux through every cycle reduces that to
   one field vector, and it is a physical ansatz rather than a fitted one.
"""
from __future__ import annotations

import numpy as np

from .common import anchor_indices, contact_graph

__all__ = ["peierls_hamiltonian", "cycle_rank", "directional_asymmetry",
           "chiral_scores"]


def cycle_rank(adj: np.ndarray) -> int:
    """E - N + 1: the number of gauge-invariant phase parameters on this graph."""
    adj = np.asarray(adj, float)
    return int(adj.sum() // 2) - adj.shape[0] + 1


def peierls_hamiltonian(cb: np.ndarray, adj: np.ndarray,
                        field: np.ndarray) -> np.ndarray:
    """Chiral adjacency Hamiltonian via the Peierls substitution.

    In the symmetric gauge A(r) = (B x r)/2, the phase picked up on the edge
    j -> k is the line integral of A along it, which for a straight segment is

        theta_jk = 0.5 * (B x r_mid) . (r_k - r_j),   r_mid = (r_j + r_k)/2

    This is antisymmetric by construction, so H stays Hermitian, and the flux
    through any closed loop equals B . (loop area vector) — gauge-invariant and
    independent of how the gauge was written down.
    """
    r = np.asarray(cb, float)
    B = np.asarray(field, float)
    mid = 0.5 * (r[:, None, :] + r[None, :, :])
    dvec = r[None, :, :] - r[:, None, :]
    theta = 0.5 * np.einsum('ijk,ijk->ij', np.cross(np.broadcast_to(B, mid.shape), mid), dvec)
    theta = 0.5 * (theta - theta.T)                 # enforce antisymmetry exactly
    return np.asarray(adj, float) * np.exp(1j * theta)


def _spectrum(H: np.ndarray, rows: np.ndarray):
    """Eigendecomposition reused across every evaluation time."""
    w, V = np.linalg.eigh(H)
    return w, V[rows], V.conj().T


def _propagator_rows(spec, t: float) -> np.ndarray:
    """Rows of exp(-iHt) from a cached spectrum."""
    w, Vr, Vc = spec
    return (Vr * np.exp(-1j * w * t)) @ Vc


def directional_asymmetry(cb: np.ndarray, anchor, field: np.ndarray,
                          cutoff: float = 10.0, n_times: int = 12,
                          t_max: float | None = None) -> np.ndarray:
    """Time-averaged forward-minus-backward transfer probability from the anchor.

    Exactly zero for a time-reversal-symmetric walk, so this isolates the chiral
    contribution. Averaged over a window rather than read at one time, because a
    single time is an arbitrary choice and the literature warns that long-time
    chiral performance is very sensitive to the exact phase.
    """
    cb = np.asarray(cb, float)
    A = contact_graph(cb, cutoff)
    a = anchor_indices(anchor)
    H = peierls_hamiltonian(cb, A, field)

    jmax = float(np.abs(A).max()) or 1.0
    if t_max is None:
        t_max = 4.0 * 2.0 * np.pi / jmax

    spec = _spectrum(H, a)
    acc = np.zeros(len(cb))
    for t in np.linspace(t_max / n_times, t_max, n_times):
        fwd = _propagator_rows(spec, t)             # <i| U(t) |a>
        bwd = _propagator_rows(spec, -t)            # <i| U(-t)|a> = <a| U(t)|i>*
        acc += (np.abs(fwd) ** 2 - np.abs(bwd) ** 2).mean(axis=0)
    return acc / n_times


def chiral_scores(cb: np.ndarray, anchor, field_strength: float = 0.1,
                  directions: int = 3, cutoff: float = 10.0,
                  signed: bool = False) -> np.ndarray:
    """Chiral score, averaged over field directions so no axis is privileged.

    The protein has no preferred external axis, so a single field direction
    would be an arbitrary choice. Directions are taken along the structure's
    own principal axes.

    signed=False scores |asymmetry| — "this residue exchanges asymmetrically
    with the active site", direction-agnostic. signed=True keeps the sign, i.e.
    "receives more than it sends".
    """
    cb = np.asarray(cb, float)
    X = cb - cb.mean(0)
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    axes = vt[:max(1, min(3, int(directions)))]

    out = np.zeros(len(cb))
    for ax in axes:
        d = directional_asymmetry(cb, anchor, field_strength * ax, cutoff=cutoff)
        out += d if signed else np.abs(d)
    return out / len(axes)

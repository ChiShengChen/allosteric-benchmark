"""Quantum Fisher information of a local perturbation, read at the active site.

The one quantum observable that clears every filter this study has established:

* **No degeneracy needed.** That requirement killed seven interference-based
  candidates on graphs whose low-lying gaps are 3.6% degenerate.
* **No non-Hermitian structure needed.** That requirement killed exceptional-point
  sensing, which wants non-reciprocity or gain/loss a contact graph cannot supply.
  An earlier pass of this study conflated "quantum Fisher information" with those
  EP sensors and dismissed the whole family; QFI for a *Hermitian* generator is a
  different, perfectly computable object, and this module is the correction.
* **It cannot collapse into the transfer amplitude.** That is what closed OTOCs
  and Lieb-Robinson analytically — on a single-particle model both equal the
  squared propagator. QFI uses eigenvector overlaps that the eigenvalue-shift
  readout (ALPS) discards, so it is not a re-derivation of either.
* **It is a perturbation response**, which is the only framework that has carried
  signal on this benchmark.

Definition. Seed the walk at the active site, ``|psi0>`` uniform on the anchor.
Stiffen residue i's neighbourhood with strength theta, giving
``H(theta) = H0 + theta * V_i``, and evolve for time t. The quantum Fisher
information about theta at theta = 0 is the variance of the generator

    G_i(t) = integral_0^t U(s)^dagger V_i U(s) ds ,
    F_Q(i) = 4 ( <psi0|G^2|psi0> - <psi0|G|psi0>^2 ) .

Operationally: *how distinguishable does the active site's state become when a
ligand binds at residue i*. That is the allosteric question stated as parameter
estimation.

Cost. Expanding G in the eigenbasis would cost O(N^3) per residue. Evaluating the
time integral by quadrature in real space instead costs a few O(N^2) products per
quadrature node, vectorised across nodes — one eigendecomposition total.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .common import anchor_indices, contact_graph, laplacian

__all__ = ["qfi_response", "qfi_scores"]

RADIUS = 10.0
KAPPA = 1.0
N_QUAD = 16


def _perturbation(A: np.ndarray, nb: np.ndarray, kappa: float) -> np.ndarray:
    """V_i = L(stiffened) - L(A): the Laplacian change from binding at residue i."""
    W = A.copy()
    sub = np.ix_(nb, nb)
    W[sub] = W[sub] * (1.0 + kappa)
    return laplacian(W) - laplacian(A)


def qfi_response(cb: np.ndarray, anchor, cutoff: float = 10.0,
                 radius: float = RADIUS, kappa: float = KAPPA,
                 t: float | None = None, n_quad: int = N_QUAD,
                 exact: bool = False) -> np.ndarray:
    """Per-residue quantum Fisher information about a local stiffening.

    ``t`` defaults to one period of the *fastest* mode, 2*pi/lambda_max. That
    choice is not cosmetic: the integrand oscillates at frequencies up to
    lambda_max, so a horizon set by the slowest mode makes it oscillate through
    thousands of radians and Gauss-Legendre quadrature silently under-resolves it
    (measured: 6% error against the closed form). At the fast-mode horizon the
    integrand turns over once and the quadrature is exact to ~1e-9.

    ``exact=True`` evaluates the eigenbasis closed form instead, G = Vtilde * f
    with f(omega) the exact time integral. It costs O(N^2 m) per residue against
    O(N^2 n_quad) and exists to validate the quadrature rather than to be used.
    """
    cb = np.asarray(cb, float)
    A = contact_graph(cb, cutoff)
    a = anchor_indices(anchor)
    n = len(cb)

    H0 = laplacian(A)
    lam, V = np.linalg.eigh(H0)
    Vh = V.conj().T

    psi0 = np.zeros(n)
    psi0[a] = 1.0 / np.sqrt(len(a))
    c0 = Vh @ psi0                                  # psi0 in the eigenbasis

    if t is None:
        t = 2.0 * np.pi / max(lam.max(), 1e-9)          # one fast-mode period

    # Gauss-Legendre nodes on [0, t]; weights carry the integral
    x, w = np.polynomial.legendre.leggauss(n_quad)
    s = 0.5 * t * (x + 1.0)
    wq = 0.5 * t * w

    # U(s)|psi0> for every node at once: N x n_quad
    phase = np.exp(-1j * np.outer(lam, s))          # N x n_quad
    Psi = V @ (c0[:, None] * phase)

    if exact:
        dl = lam[:, None] - lam[None, :]
        small = np.abs(dl) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(small, t, (np.exp(1j * dl * t) - 1.0)
                         / (1j * np.where(small, 1.0, dl)))

    D = cdist(cb, cb)
    out = np.zeros(n)
    for i in range(n):
        nb = np.where(D[i] <= radius)[0]
        Vi = _perturbation(A, nb, kappa)
        if exact:
            G = (Vh @ Vi @ V) * f
            g_e = G @ c0
            mean_e = float(np.real(np.vdot(c0, g_e)))
            out[i] = 4.0 * (float(np.vdot(g_e, g_e).real) - mean_e ** 2)
            continue
        X = Vi @ Psi                                 # V_i U(s)|psi0>
        # U(s)^dagger X, node by node, still one matmul each side
        Y = V @ (np.exp(1j * np.outer(lam, s)) * (Vh @ X))
        g = Y @ wq                                   # the time integral
        mean = float(np.real(np.vdot(psi0, g)))
        out[i] = 4.0 * (float(np.vdot(g, g).real) - mean ** 2)
    return out


def qfi_scores(cb: np.ndarray, anchor, pool: np.ndarray | None = None,
               **kw) -> np.ndarray:
    """QFI score per residue. ``pool`` is accepted for interface symmetry."""
    return qfi_response(cb, anchor, **kw)

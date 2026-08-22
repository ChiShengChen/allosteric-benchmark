#!/usr/bin/env python
"""Classical kernels, and a simulated quantum feature-map kernel.

The quantum kernel is a ZZ-feature-map fidelity kernel,
K(x, x') = |<phi(x)|phi(x')>|^2, evaluated exactly by statevector simulation. At
7-8 features that is a 128-256 dimensional state, so numpy is faster and more
accurate than a circuit simulator and introduces no sampling noise -- which is
the right choice here, since sampling noise would only obscure the comparison we
are trying to make.

Bandwidth matters and is exposed. The literature finding this folder is testing
is that a *tuned* bandwidth makes the quantum kernel converge to an RBF, so
running it untuned would prove nothing either way.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rbf_kernel", "linear_kernel", "poly_kernel", "quantum_kernel",
           "zz_feature_state"]


def linear_kernel(A, B):
    return np.asarray(A) @ np.asarray(B).T


def poly_kernel(A, B, degree=4, coef0=1.0, gamma=None):
    A, B = np.asarray(A), np.asarray(B)
    g = 1.0 / A.shape[1] if gamma is None else gamma
    return (g * (A @ B.T) + coef0) ** degree


def rbf_kernel(A, B, gamma=None):
    A, B = np.asarray(A, float), np.asarray(B, float)
    g = 1.0 / A.shape[1] if gamma is None else gamma
    d2 = ((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T)
    return np.exp(-g * np.clip(d2, 0, None))


def zz_feature_state(X, bandwidth=1.0, reps=2):
    """Statevector of a ZZ feature map, one row per sample.

    H^{otimes n}, then diagonal phases exp(i * 2 * bandwidth * x_i) on each qubit
    and exp(i * 2 * bandwidth * (pi - x_i)(pi - x_j)) on each pair, repeated.
    Every gate is diagonal after the Hadamards, so the whole map is a product of
    phases on the uniform superposition and needs no matrix multiplication.
    """
    X = np.asarray(X, float)
    m, n = X.shape
    dim = 1 << n
    bits = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    z = 1.0 - 2.0 * bits                      # +1 / -1 per qubit per basis state

    phase = np.zeros((m, dim))
    single = 2.0 * bandwidth * X              # m x n
    pair_i, pair_j = np.triu_indices(n, 1)
    pair = 2.0 * bandwidth * (np.pi - X[:, pair_i]) * (np.pi - X[:, pair_j])
    for _ in range(reps):
        phase += single @ z.T
        phase += pair @ (z[:, pair_i] * z[:, pair_j]).T
    return np.exp(1j * phase) / np.sqrt(dim)


def quantum_kernel(A, B, bandwidth=1.0, reps=2):
    """Fidelity kernel |<phi(a)|phi(b)>|^2 of the ZZ feature map."""
    pa = zz_feature_state(A, bandwidth, reps)
    pb = zz_feature_state(B, bandwidth, reps)
    return np.abs(pa @ pb.conj().T) ** 2

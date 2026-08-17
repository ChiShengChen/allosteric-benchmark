"""Cooperative site selection: which *set* of residues jointly perturbs the active site?

Single-residue perturbation (ALPS) asks "what happens if a ligand binds here".
The biological question behind cooperative allostery is different: which **set**
of k residues, stiffened *together*, maximally retunes the active-site spectrum.

The exact objective for a set S is

    f(S) = sum_{k<=K} |lambda_k(H_S) - lambda_k(H_0)| / lambda_k(H_0)

where ``H_S`` stiffens the union of the neighbourhoods of every residue in S.
Evaluating it needs one eigendecomposition per candidate set, and there are
C(N, k) of them -- about 2e10 for N = 300, k = 5.

The standard way to make that tractable is a quadratic surrogate

    f(S) ~ sum_i h_i x_i + sum_{i<j} J_ij x_i x_j ,   x in {0,1}^N, sum x = k

with ``h_i = f({i})`` the single-residue response and
``J_ij = f({i,j}) - f({i}) - f({j})`` the non-additive part of the pair. That
is a QUBO / Ising problem, i.e. the form a quantum annealer or QAOA consumes.

**Whether any of that is warranted is an empirical question, and this module is
built to answer it first.** If the couplings J are negligible against h, the
objective is additive, greedy selection is optimal, there is no combinatorial
hardness, and a quantum solver has nothing to do. :func:`nonadditivity` reports
exactly that ratio, and the solvers below exist so greedy can be compared
against an exact search on the same surrogate.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.distance import cdist

from .common import contact_graph, laplacian

__all__ = ["set_response", "couplings", "nonadditivity",
           "solve_greedy", "solve_exact", "solve_anneal", "qubo_value"]


def _stiffened(A, D, members, radius, kappa):
    """Adjacency with the union of the members' neighbourhoods stiffened."""
    W = A.copy()
    nb = np.unique(np.concatenate([np.where(D[i] <= radius)[0] for i in members]))
    sub = np.ix_(nb, nb)
    W[sub] = W[sub] * (1.0 + kappa)
    return W


def set_response(cb, members, base_lam=None, A=None, D=None,
                 cutoff=10.0, radius=10.0, kappa=1.0, k_modes=3):
    """Exact objective f(S): relative shift of the K lowest non-zero eigenvalues."""
    cb = np.asarray(cb, float)
    if A is None:
        A = contact_graph(cb, cutoff)
    if D is None:
        D = cdist(cb, cb)
    if base_lam is None:
        w0 = np.linalg.eigvalsh(laplacian(A))
        base_lam = w0[w0 > 1e-9][:k_modes]
    if len(members) == 0:
        return 0.0
    wp = np.linalg.eigvalsh(laplacian(_stiffened(A, D, members, radius, kappa)))
    lam = wp[wp > 1e-9][:k_modes]
    m = min(len(lam), len(base_lam))
    return float(np.sum(np.abs(lam[:m] - base_lam[:m]) / (base_lam[:m] + 1e-12)))


def couplings(cb, candidates, cutoff=10.0, radius=10.0, kappa=1.0, k_modes=3):
    """Return (h, J) over ``candidates``: singles and pairwise non-additive parts.

    Cost is |candidates| + C(|candidates|, 2) eigenvalue solves, which is why the
    candidate list must be pre-screened (typically the top-M by single-residue
    response).
    """
    cb = np.asarray(cb, float)
    cand = np.asarray(candidates, int)
    A = contact_graph(cb, cutoff)
    D = cdist(cb, cb)
    w0 = np.linalg.eigvalsh(laplacian(A))
    base = w0[w0 > 1e-9][:k_modes]

    kw = dict(base_lam=base, A=A, D=D, radius=radius, kappa=kappa, k_modes=k_modes)
    h = np.array([set_response(cb, [i], **kw) for i in cand])

    m = len(cand)
    J = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            f_ab = set_response(cb, [cand[a], cand[b]], **kw)
            J[a, b] = J[b, a] = f_ab - h[a] - h[b]
    return h, J


def nonadditivity(h, J):
    """Is the objective actually non-additive? Returns a dict of diagnostics.

    ``ratio`` is median |J_ij| divided by the mean single-residue response: the
    fraction of a typical residue's effect that pairwise cooperation adds or
    removes. If this is near zero the problem is additive and greedy is optimal.
    """
    m = len(h)
    iu = np.triu_indices(m, 1)
    j = J[iu]
    scale = float(np.mean(np.abs(h))) + 1e-30
    return dict(
        n_candidates=m,
        mean_h=float(np.mean(h)),
        median_absJ=float(np.median(np.abs(j))),
        max_absJ=float(np.max(np.abs(j))) if len(j) else 0.0,
        ratio=float(np.median(np.abs(j)) / scale),
        ratio_max=float(np.max(np.abs(j)) / scale) if len(j) else 0.0,
        frac_J_over_10pct=float(np.mean(np.abs(j) > 0.1 * scale)) if len(j) else 0.0,
        frac_negative=float(np.mean(j < 0)) if len(j) else 0.0,
    )


def qubo_value(h, J, idx):
    """Surrogate objective of a chosen index set."""
    idx = list(idx)
    v = float(np.sum(h[idx]))
    for a, b in itertools.combinations(idx, 2):
        v += J[a, b]
    return v


def solve_greedy(h, J, k):
    """Greedy forward selection on the surrogate — the classical default."""
    chosen = []
    remaining = list(range(len(h)))
    for _ in range(k):
        best, best_v = None, -np.inf
        for c in remaining:
            v = qubo_value(h, J, chosen + [c])
            if v > best_v:
                best, best_v = c, v
        chosen.append(best)
        remaining.remove(best)
    return chosen, qubo_value(h, J, chosen)


def solve_exact(h, J, k):
    """Exhaustive search over C(m, k) — feasible only for a pre-screened m."""
    best, best_v = None, -np.inf
    for combo in itertools.combinations(range(len(h)), k):
        v = qubo_value(h, J, combo)
        if v > best_v:
            best, best_v = combo, v
    return list(best), best_v


def solve_anneal(h, J, k, n_restarts=40, n_steps=4000, seed=0):
    """Classical simulated annealing on the same QUBO — the fair baseline a
    quantum annealer would have to beat."""
    rng = np.random.default_rng(seed)
    m = len(h)
    best, best_v = None, -np.inf
    for _ in range(n_restarts):
        cur = list(rng.choice(m, k, replace=False))
        cur_v = qubo_value(h, J, cur)
        for step in range(n_steps):
            T = max(1e-9, 1.0 - step / n_steps)
            out_pos = rng.integers(k)
            cand = int(rng.integers(m))
            if cand in cur:
                continue
            trial = list(cur)
            trial[out_pos] = cand
            tv = qubo_value(h, J, trial)
            if tv > cur_v or rng.random() < np.exp((tv - cur_v) / (T * 0.01 + 1e-12)):
                cur, cur_v = trial, tv
            if cur_v > best_v:
                best, best_v = list(cur), cur_v
    return best, best_v

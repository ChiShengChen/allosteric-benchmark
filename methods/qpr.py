"""QPR — Quantum Perturbation Response.

A new method, designed from three measured facts about the existing ones:

1. QASC's readout is an *absolute* coherent transfer amplitude from the active
   site, and we measured its correlation with distance-to-anchor at -0.60 to
   -0.71 across three independent target sets. Any absolute amplitude on a
   contact graph is dominated by proximity, so it cannot be the signal.
2. APOP-style elastic-network stiffening generalises best of everything we
   tested, but it never looks at the active site at all: it ranks generic
   mechanical leverage, not allosteric coupling.
3. Allostery is a *conditional* statement -- does a perturbation here change
   the dynamics *at the active site*? -- so the estimator should be a
   difference, not an amplitude.

QPR therefore perturbs like APOP, reads out where PRS reads out, uses QASC's
coherent operator, and takes the difference so the baseline proximity
structure cancels:

    H0            = Kirchhoff Laplacian of the contact graph
    C(H)[i,j]     = sum_k V_ik^2 V_jk^2      (infinite-time-averaged CTQW
                                              transfer probability)
    H_i           = H0 with every edge inside residue i's neighbourhood
                    stiffened by (1 + kappa)          [APOP's ligand mimic]

    self(i)       = | mean C(H_i)[a,a] - mean C(H0)[a,a] |
    leak(i)       = || C(H_i)[a,S_i] - C(H0)[a,S_i] ||_1 / |S_i|

with ``a`` the active-site residues and ``S_i`` everything that is neither the
active site nor residue i's own perturbed neighbourhood. Excluding the
perturbed neighbourhood is what stops the estimator from simply rewarding
residues that sit near the active site: without it, ``leak`` degenerates back
into a proximity readout.

The two terms answer different questions -- ``self`` is "did the active site's
own coherent retention change", ``leak`` is "did the active site's coupling to
the rest of the protein get rerouted" -- and are combined by QASC's own
non-diluting noisy-or so neither can wash the other out.

Finally the score is residualised against distance *by construction* (a smooth
LOWESS-like local fit, not a post-hoc correction), because the perturbation
magnitude still decays with distance even after differencing.

:func:`cpr_scores` is the identical pipeline with the unitary kernel replaced
by the classical heat kernel exp(-H t). It exists so the quantum contribution
can be ablated on exactly the same graph, perturbation and readout.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .common import anchor_indices, contact_graph, laplacian
from .quantum import noisy_or

__all__ = ["qpr_scores", "qpr_inf_scores", "cpr_scores", "distance_residual"]


def _anchor_rows(H, a, kernel, t=None, ngrid=5):
    """Only the active-site rows of the propagation kernel are ever needed.

    Computing C[a, :] instead of the full C[:, :] turns the per-residue cost
    from O(N^3) into one eigendecomposition plus O(|a| N^2).

    kernel:
      "ctqw"   infinite-time-averaged coherent transfer, sum_k V_ik^2 V_jk^2
               (QASC's observable: no time scale at all)
      "ctqw_t" coherent transfer averaged over a finite window [0, T], the
               same figure of merit the ENAQT literature uses
      "heat"   classical diffusion kernel exp(-H t)
    """
    w, V = np.linalg.eigh(H)
    Va = V[a]                                    # |a| x N
    if kernel == "ctqw":
        return (Va ** 2) @ (V ** 2).T
    if kernel == "heat":
        return (Va * np.exp(-w * t)) @ V.T
    if kernel == "ctqw_t":
        acc = np.zeros((len(a), V.shape[0]))
        for tk in np.linspace(t / ngrid, t, ngrid):
            U = (Va * np.exp(-1j * w * tk)) @ V.T
            acc += np.abs(U) ** 2
        return acc / ngrid
    raise ValueError(kernel)


def distance_residual(values: np.ndarray, dist: np.ndarray,
                      pool: np.ndarray, bandwidth: float = 4.0) -> np.ndarray:
    """Local (kernel-weighted) trend removal: value minus its distance-conditional mean.

    The perturbation response still decays with distance; this makes a residue
    score high only if it beats what residues at a *comparable* distance do.
    The trend is fitted on the candidate pool only, since that is the set the
    model actually ranks within.
    """
    v = np.asarray(values, float)
    d = np.asarray(dist, float)
    pool = np.asarray(pool, bool)
    if pool.sum() < 5:
        return v - v.mean()
    dp, vp = d[pool], v[pool]
    out = np.empty_like(v)
    for i, di in enumerate(d):
        wgt = np.exp(-0.5 * ((dp - di) / bandwidth) ** 2)
        sw = wgt.sum()
        mu = (wgt @ vp) / sw if sw > 1e-12 else vp.mean()
        sd = np.sqrt(max((wgt @ (vp - mu) ** 2) / sw, 1e-24)) if sw > 1e-12 else 1.0
        out[i] = (v[i] - mu) / sd
    return out


def _response(cb, anchor, cutoff, radius, kappa, kernel):
    """Per-residue (self, leak) perturbation response at the active site."""
    cb = np.asarray(cb, float)
    A = contact_graph(cb, cutoff)
    a = anchor_indices(anchor)
    n = len(cb)
    H0 = laplacian(A)

    t = None
    if kernel != "ctqw":
        w0 = np.linalg.eigvalsh(H0)
        nz = w0[w0 > 1e-9]
        t = 1.0 / nz[0]              # inverse spectral gap: the graph's own scale
    C0 = _anchor_rows(H0, a, kernel, t)

    D = cdist(cb, cb)
    base_self = C0[:, a].mean()
    is_anchor = np.zeros(n, bool)
    is_anchor[a] = True

    self_t = np.zeros(n)
    leak_t = np.zeros(n)
    for i in range(n):
        nb = np.where(D[i] <= radius)[0]
        W = A.copy()
        sub = np.ix_(nb, nb)
        W[sub] = W[sub] * (1.0 + kappa)
        Ci = _anchor_rows(laplacian(W), a, kernel, t)

        self_t[i] = abs(Ci[:, a].mean() - base_self)

        # exclude the active site AND residue i's own perturbed neighbourhood,
        # otherwise this term degenerates back into a proximity readout
        mask = ~is_anchor
        mask[nb] = False
        if mask.any():
            self_cols = np.where(mask)[0]
            leak_t[i] = np.abs(Ci[:, self_cols] - C0[:, self_cols]).mean()
    return self_t, leak_t


def _score(cb, anchor, pool, cutoff, radius, kappa, kernel, residualise):
    from .common import distal_nonanchor_mask, min_dist_to_anchor
    if pool is None:
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
    s, l = _response(cb, anchor, cutoff, radius, kappa, kernel)
    if residualise:
        d = min_dist_to_anchor(cb, anchor)
        s = distance_residual(s, d, pool)
        l = distance_residual(l, d, pool)
    return noisy_or(s, l, weights=[1.0, 0.5])


def qpr_scores(cb, anchor, pool=None, cutoff: float = 10.0, radius: float = 8.0,
               kappa: float = 1.0, residualise: bool = True,
               kernel: str = "ctqw_t") -> np.ndarray:
    """Quantum Perturbation Response (finite-window coherent kernel by default)."""
    return _score(cb, anchor, pool, cutoff, radius, kappa, kernel, residualise)


def qpr_inf_scores(cb, anchor, pool=None, **kw) -> np.ndarray:
    """QPR using QASC's own timeless infinite-time-average observable."""
    return _score(cb, anchor, pool, kw.get("cutoff", 10.0), kw.get("radius", 8.0),
                  kw.get("kappa", 1.0), "ctqw", kw.get("residualise", True))


def cpr_scores(cb, anchor, pool=None, cutoff: float = 10.0, radius: float = 8.0,
               kappa: float = 1.0, residualise: bool = True) -> np.ndarray:
    """Classical ablation: identical pipeline, diffusive kernel exp(-H t)."""
    return _score(cb, anchor, pool, cutoff, radius, kappa, "heat", residualise)

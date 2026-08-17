#!/usr/bin/env python
"""Is cooperative site selection actually a hard combinatorial problem?

Before proposing a quantum solver for the "which set of k residues jointly
perturbs the active site" problem, this script tests the premise:

  1. **Non-additivity.** Build the exact singles h_i and pair couplings J_ij on a
     pre-screened candidate list. If |J| is negligible against |h| the objective
     is additive, greedy is optimal, and no combinatorial solver is warranted.
  2. **Optimisation gap.** Compare greedy, simulated annealing and exhaustive
     search on the same surrogate. If greedy already finds the exact optimum,
     there is nothing for a better solver to win.
  3. **Surrogate fidelity.** Check the quadratic surrogate against the exact
     objective f(S) recomputed by eigendecomposition on the chosen sets.
  4. **Biological value, against trivial controls.** Does the cooperative set find
     true allosteric residues better than the top-k singles -- and, crucially,
     better than a random k-subset of the same candidates or a set picked purely
     for maximum spatial dispersion? A spread-out set covers more of the protein
     and hits something by construction, so dispersion is the control that a
     cooperative-selection claim has to beat.

Usage: python3 scripts/cooperative.py [--targets data/targets] [--top 30] [--k 5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from methods.alps import spectral_response                                  # noqa: E402
from methods.common import distal_nonanchor_mask                            # noqa: E402
from methods.cooperative import (couplings, nonadditivity, qubo_value,      # noqa: E402
                                 set_response, solve_anneal, solve_exact,
                                 solve_greedy)
from scipy.spatial.distance import cdist                                    # noqa: E402


def run_target(path, top_m=30, k=5):
    d = np.load(path)
    cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
    pool = distal_nonanchor_mask(cb, anchor, 8.0)
    if pool.sum() < top_m or y.sum() == 0:
        return None

    # pre-screen: the top-M residues by single-residue spectral response
    single = spectral_response(cb)
    order = np.where(pool)[0][np.argsort(single[pool])[::-1]]
    cand = order[:top_m]

    h, J = couplings(cb, cand)
    diag = nonadditivity(h, J)

    g_idx, g_val = solve_greedy(h, J, k)
    a_idx, a_val = solve_anneal(h, J, k)
    e_idx, e_val = solve_exact(h, J, k)

    # surrogate fidelity: exact objective of each chosen set
    true_g = set_response(cb, cand[g_idx])
    true_e = set_response(cb, cand[e_idx])
    # additive baseline: simply the top-k singles
    top_idx = list(range(k))
    true_top = set_response(cb, cand[top_idx])

    hit = lambda idx: int(bool(y[cand[idx]].sum()))                    # noqa: E731

    # --- trivial controls on the same candidate list
    rng = np.random.default_rng(0)
    hit_rand = float(np.mean([hit(rng.choice(top_m, k, replace=False))
                              for _ in range(400)]))
    Dc = cdist(cb[cand], cb[cand])                 # max-dispersion set
    disp = [int(np.argmax(Dc.sum(1)))]
    while len(disp) < k:
        disp.append(int(np.argmax(np.min(Dc[:, disp], axis=1))))
    spread = lambda idx: float(cdist(cb[cand[idx]], cb[cand[idx]]).mean())  # noqa: E731
    return dict(
        target=os.path.basename(path).replace(".npz", ""),
        n=int(len(cb)),
        **diag,
        greedy=g_val, anneal=a_val, exact=e_val,
        gap_greedy=float((e_val - g_val) / (abs(e_val) + 1e-12)),
        gap_anneal=float((e_val - a_val) / (abs(e_val) + 1e-12)),
        greedy_is_exact=int(sorted(g_idx) == sorted(e_idx)),
        true_greedy=true_g, true_exact=true_e, true_topk=true_top,
        surrogate_err=float(abs(e_val - true_e) / (abs(true_e) + 1e-12)),
        hit_exact=hit(e_idx), hit_greedy=hit(g_idx), hit_topk=hit(top_idx),
        hit_rand=hit_rand, hit_disp=hit(disp),
        spread_topk=spread(top_idx), spread_exact=spread(e_idx),
        spread_disp=spread(disp),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets"))
    ap.add_argument("--top", type=int, default=30, help="candidates pre-screened")
    ap.add_argument("--k", type=int, default=5, help="residues to select")
    ap.add_argument("--max-n", type=int, default=660)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results_cooperative.json"))
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.targets, "*.npz"))):
        if len(np.load(f)["cb"]) > a.max_n:
            continue
        try:
            r = run_target(f, a.top, a.k)
        except Exception as e:                                          # noqa: BLE001
            print(f"  {os.path.basename(f)} FAILED: {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        print(f"  {r['target']:12s} N={r['n']:4d}  |J|/|h| med={r['ratio']:.4f} "
              f"max={r['ratio_max']:.3f}  greedy=exact:{r['greedy_is_exact']}  "
              f"gap={r['gap_greedy']*100:.2f}%", flush=True)

    if not rows:
        print("no targets")
        return
    json.dump(rows, open(a.out, "w"), indent=1)

    g = lambda key: np.array([r[key] for r in rows], float)             # noqa: E731
    print(f"\n=== cooperative selection, n={len(rows)} targets, "
          f"top-{a.top} candidates, k={a.k} ===\n")
    print("1. Non-additivity of the objective")
    print(f"   median |J_ij| / mean h        : {np.median(g('ratio')):.4f}")
    print(f"   max    |J_ij| / mean h        : {np.median(g('ratio_max')):.4f}")
    print(f"   pairs with |J| > 10% of h     : {np.mean(g('frac_J_over_10pct'))*100:.1f}%")
    print(f"   couplings that are negative   : {np.mean(g('frac_negative'))*100:.1f}%")
    print("\n2. Is the optimisation hard?")
    print(f"   greedy == exact optimum       : {np.mean(g('greedy_is_exact'))*100:.0f}% of targets")
    print(f"   greedy shortfall vs exact     : {np.mean(g('gap_greedy'))*100:.3f}%")
    print(f"   annealing shortfall vs exact  : {np.mean(g('gap_anneal'))*100:.3f}%")
    print("\n3. Surrogate fidelity (quadratic vs true eigenvalue objective)")
    print(f"   relative error                : {np.median(g('surrogate_err'))*100:.1f}%")
    print("\n4. Does cooperation help biologically -- and beat trivial controls?")
    print(f"   hit rate, top-k singles       : {np.mean(g('hit_topk'))*100:.1f}%"
          f"   (mean pairwise spread {np.mean(g('spread_topk')):.1f} A)")
    print(f"   hit rate, greedy set          : {np.mean(g('hit_greedy'))*100:.1f}%")
    print(f"   hit rate, QUBO optimum        : {np.mean(g('hit_exact'))*100:.1f}%"
          f"   (mean pairwise spread {np.mean(g('spread_exact')):.1f} A)")
    print(f"   CONTROL random k-subset       : {np.mean(g('hit_rand'))*100:.1f}%")
    print(f"   CONTROL max spatial spread    : {np.mean(g('hit_disp'))*100:.1f}%"
          f"   (mean pairwise spread {np.mean(g('spread_disp')):.1f} A)")
    print(f"   true f(S): top-k {np.mean(g('true_topk')):.4f} | "
          f"greedy {np.mean(g('true_greedy')):.4f} | exact {np.mean(g('true_exact')):.4f}")


if __name__ == "__main__":
    main()

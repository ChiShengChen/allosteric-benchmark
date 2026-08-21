#!/usr/bin/env python
"""Re-test every quantum insertion point on curated labels and the confound-free metric.

Nine quantum insertion points were measured in this repository, but only three of
them — plain CTQW communicability, the QASC baseline, and quantum Fisher
information — were ever scored on curated annotations with the proximity confound
removed. The other six rest on proxy labels and plain AUC, which sections 9.1–9.4
showed are dominated by geometry and whose rankings do not transfer.

This closes that gap. Every remaining per-residue quantum readout is re-scored on
the curated targets with distance-stratified AUC, against the same random-control
floor and the same paired test.

Not covered here, because they are not per-residue scores on this target set:
cooperative selection as a QUBO (a subset-selection problem, section 6) and the
symmetric-multimer ablation (a different target set, section 8).

Usage: python3 scripts/quantum_recheck.py [--max-n 600]
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
sys.path.insert(0, os.path.join(HERE, "scripts"))

from methods.chiral import chiral_scores                                  # noqa: E402
from methods.common import (contact_graph, distal_nonanchor_mask,         # noqa: E402
                            min_dist_to_anchor, pocket_smooth, rank_percentile)
from methods.quantum import (ctqw_communicability, enaqt_transfer,        # noqa: E402
                             ipr_resonant_transfer, noisy_or)
from methods.qpr import cpr_scores, qpr_scores                            # noqa: E402
from ablate_readouts import readouts                                      # noqa: E402
from partial_auc import stratified_auc                                    # noqa: E402


def quantum_scores(cb, anchor, pool, heavy=True):
    """Every remaining quantum readout, plus the two references already settled."""
    A = contact_graph(cb, 10.0)
    ctqw = ctqw_communicability(A, anchor)
    out = {
        "ctqw_only": ctqw,
        "qasc_baseline": noisy_or(ctqw, ipr_resonant_transfer(A, anchor)),
        "qasc_degseed": noisy_or(ctqw, ipr_resonant_transfer(A, anchor, seed="degree")),
        "qasc_normlap": noisy_or(ctqw_communicability(A, anchor, normalized=True),
                                 ipr_resonant_transfer(A, anchor)),
        "enaqt": enaqt_transfer(A, anchor, gamma_rel=1.0),
        "chiral_asym": chiral_scores(cb, anchor, field_strength=0.1),
    }
    if heavy:
        r = readouts(cb, anchor)          # one pass gives all three spectral readouts
        out["degeneracy_dgap"] = r["dgap"]
        out["eigvec_dpart"] = r["dpart"]
        out["mode_ipr_dipr"] = r["dipr"]
        out["qpr_coherent"] = qpr_scores(cb, anchor, pool)
        out["cpr_classical"] = cpr_scores(cb, anchor, pool)
    return out, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_curated"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results_quantum_recheck.json"))
    ap.add_argument("--max-n", type=int, default=600)
    ap.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args()

    res = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {r["target"] for r in res}
    for f in sorted(glob.glob(os.path.join(a.targets, "*.npz"))):
        name = os.path.basename(f).replace(".npz", "")
        if name in done:
            continue
        d = np.load(f)
        cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
        if len(cb) > a.max_n:
            continue
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if y.sum() == 0 or (pool & (y == 1)).sum() == 0:
            continue
        dist = min_dist_to_anchor(cb, anchor)
        raw, A = quantum_scores(cb, anchor, pool)
        rng = np.random.default_rng(0)
        raw["ctrl_random"] = rng.random(len(cb))
        row = {}
        for k, s in raw.items():
            sm = pocket_smooth(rank_percentile(s), A)
            auc, npairs = stratified_auc(y, sm, pool, dist, a.tol)
            row[k] = [auc, npairs]
        res.append(dict(target=name, n=int(len(cb)), rows=row))
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"  [{len(res):3d}] {name} N={len(cb)}", flush=True)

    if not res:
        return
    from scipy import stats
    names = sorted({k for r in res for k in r["rows"]})

    def vec(m):
        return np.array([r["rows"][m][0] if m in r["rows"] and r["rows"][m][0] is not None
                         else np.nan for r in res], float)
    rnd = vec("ctrl_random")
    FLOOR = 0.4963            # 25-seed estimate, scripts/floor_and_tests.py
    print(f"\n=== quantum readouts on CURATED labels, distance-stratified, n={len(res)} ===")
    print(f"{'readout':18s} {'strat AUC':>10s} {'vs floor':>9s} {'paired p':>10s}")
    rows = []
    for m in names:
        x = vec(m)
        ok = ~np.isnan(x) & ~np.isnan(rnd)
        if ok.sum() < 10:
            continue
        p = (np.nan if m == "ctrl_random" or np.allclose(x[ok], rnd[ok])
             else stats.wilcoxon(x[ok], rnd[ok]).pvalue)
        rows.append((np.nanmean(x), m, np.nanmean(x) - FLOOR, p))
    for au, m, dm, p in sorted(rows, reverse=True):
        ps = "reference" if np.isnan(p) else f"{p:.3f}"
        star = "" if np.isnan(p) or p >= 0.05 else " *"
        print(f"{m:18s} {au:10.3f} {dm:+9.3f} {ps:>10s}{star}")
    print(f"\nfloor {FLOOR:.3f} +- 0.016 (25 random controls); * = p < 0.05 uncorrected")


if __name__ == "__main__":
    main()

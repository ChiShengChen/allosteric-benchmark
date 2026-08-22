#!/usr/bin/env python
"""How much does the stratified metric depend on its one free parameter?

Every conclusion in section 9.4 onward rests on distance-stratified AUC with a
matching tolerance of 2 Å, and that tolerance was never varied. It should be,
because it trades two failure modes against each other: too tight and there are
too few matched pairs to estimate anything, too loose and distance leaks back in
and the metric quietly becomes plain AUC again.

`ctrl_closeness` is the diagnostic. It scores 0.617 on plain AUC and must fall to
~0.5 once distance is properly controlled, so its value as a function of the
tolerance shows exactly where the control stops working.

Scores are computed once per target and reused across every tolerance, so the
sweep costs one evaluation pass rather than one per setting.
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

from methods.alps import alps_scores                                       # noqa: E402
from methods.btb import btb_scores                                         # noqa: E402
from methods.common import (contact_graph, distal_nonanchor_mask,          # noqa: E402
                            min_dist_to_anchor, pocket_smooth, rank_percentile)
from methods.qfi import qfi_scores                                         # noqa: E402
from methods.quantum import (ctqw_communicability, ipr_resonant_transfer,  # noqa: E402
                             noisy_or)
from partial_auc import stratified_auc                                     # noqa: E402

TOLS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_curated"))
    ap.add_argument("--max-n", type=int, default=700)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results_tol_sweep.json"))
    a = ap.parse_args()

    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for f in sorted(glob.glob(os.path.join(a.targets, "*.npz"))):
        name = os.path.basename(f).replace(".npz", "")
        if name in res:
            continue
        d = np.load(f)
        cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
        if len(cb) > a.max_n:
            continue
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if y.sum() == 0 or (pool & (y == 1)).sum() == 0:
            continue
        A = contact_graph(cb, 10.0)
        dist = min_dist_to_anchor(cb, anchor)
        ctqw = ctqw_communicability(A, anchor)
        rng = np.random.default_rng(0)
        scores = {
            "ALPS": alps_scores(cb, anchor, pool),
            "btb_raw": btb_scores(cb, anchor, distance_corrected=False),
            "ctqw_only": ctqw,
            "qasc_baseline": noisy_or(ctqw, ipr_resonant_transfer(A, anchor)),
            "qfi": qfi_scores(cb, anchor),
            "ctrl_closeness": -dist,
            "ctrl_random": rng.random(len(cb)),
        }
        row = {}
        for k, s in scores.items():
            sm = pocket_smooth(rank_percentile(s), A)
            row[k] = {}
            for tol in TOLS:
                auc, npairs = stratified_auc(y, sm, pool, dist, tol, min_pairs=10)
                row[k][str(tol)] = [None if auc != auc else float(auc), int(npairs)]
        res[name] = row
        json.dump(res, open(a.out, "w"))
        print(f"  [{len(res):3d}] {name}", flush=True)

    if not res:
        return
    methods = list(next(iter(res.values())).keys())
    print(f"\n=== stratified AUC vs matching tolerance, n={len(res)} curated targets ===")
    hdr = "  ".join(f"{('all' if t > 100 else f'{t:g}A'):>7s}" for t in TOLS)
    print(f"{'method':16s} {hdr}")
    for m in methods:
        cells = []
        for tol in TOLS:
            v = [res[n][m][str(tol)][0] for n in res if res[n][m][str(tol)][0] is not None]
            cells.append(f"{np.mean(v):7.3f}" if v else "      —")
        print(f"{m:16s} " + "  ".join(cells))
    npairs = []
    for tol in TOLS:
        v = [res[n]["ALPS"][str(tol)][1] for n in res]
        npairs.append(f"{np.median(v):7.0f}")
    print(f"{'median pairs':16s} " + "  ".join(npairs))
    print("\n'all' = no matching, i.e. plain AUC. ctrl_closeness is the diagnostic: it "
          "must sit near 0.5\nwhile the control works and climb toward its plain-AUC "
          "value as the tolerance loosens.")


if __name__ == "__main__":
    main()

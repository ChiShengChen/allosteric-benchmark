#!/usr/bin/env python
"""How large is the noise floor, and which margins over it are real?

Section 9.4 read every method against a single random control. One draw is not a
floor: it has its own sampling error, and reading a 0.05 margin against a number
that itself moves by 0.03 is not an inference. This estimates the floor from many
independent random controls and runs paired per-target tests against it.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts"))
from methods.common import (contact_graph, distal_nonanchor_mask,        # noqa: E402
                            min_dist_to_anchor, pocket_smooth, rank_percentile)
from partial_auc import stratified_auc                                    # noqa: E402
from scipy import stats                                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_curated"))
    ap.add_argument("--results", default=os.path.join(HERE, "data", "results_partial_auc.json"))
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.targets, "*.npz")))
    per_seed = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed)
        vals = []
        for f in files:
            d = np.load(f)
            cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
            pool = distal_nonanchor_mask(cb, anchor, 8.0)
            if y.sum() == 0 or (pool & (y == 1)).sum() == 0:
                continue
            A = contact_graph(cb, 10.0)
            s = pocket_smooth(rank_percentile(rng.random(len(cb))), A)
            auc, _ = stratified_auc(y, s, pool, min_dist_to_anchor(cb, anchor), a.tol)
            vals.append(auc)
        per_seed.append(np.nanmean(vals))
    per_seed = np.asarray(per_seed)
    mu, sd = per_seed.mean(), per_seed.std()
    print(f"noise floor from {a.seeds} independent random controls, {len(files)} targets")
    print(f"  mean {mu:.4f}  sd {sd:.4f}  range [{per_seed.min():.4f}, {per_seed.max():.4f}]")
    print(f"  a single draw can land anywhere in that range — read margins against {mu:.3f}\n")

    R = json.load(open(a.results))
    def vec(m):
        return np.array([r["rows"][m][0] if m in r["rows"] and r["rows"][m][0] is not None
                         else np.nan for r in R], float)
    names = sorted({k for r in R for k in r["rows"]})
    rnd = vec("ctrl_random")
    print(f"{'method':16s} {'strat AUC':>10s} {'vs floor':>9s} {'paired p':>10s} {'n':>4s}")
    rows = []
    for m in names:
        x = vec(m)
        ok = ~np.isnan(x) & ~np.isnan(rnd)
        if ok.sum() < 10:
            continue
        if m == "ctrl_random" or np.allclose(x[ok], rnd[ok]):
            p = np.nan                       # the reference cannot be tested against itself
        else:
            p = stats.wilcoxon(x[ok], rnd[ok]).pvalue
        rows.append((np.nanmean(x), m, np.nanmean(x) - mu, p, int(ok.sum())))
    ncomp = len(rows) - 1
    for au, m, dm, p, n in sorted(rows, reverse=True):
        if np.isnan(p):
            print(f"{m:16s} {au:10.3f} {dm:+9.3f} {'reference':>10s} {n:4d}")
            continue
        mark = "**" if p < 0.05 / max(ncomp, 1) else ("*" if p < 0.05 else "")
        print(f"{m:16s} {au:10.3f} {dm:+9.3f} {p:10.2e} {n:4d} {mark}")
    print(f"\n*  p < 0.05 uncorrected   ** p < {0.05/max(ncomp,1):.4f} "
          f"(Bonferroni over {ncomp} comparisons)")


if __name__ == "__main__":
    main()

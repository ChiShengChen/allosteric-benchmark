#!/usr/bin/env python
"""Stratified AUC on the expanded 97-target curated set.

The stress test for ALPS. The set grew from 73 to 97 targets and the new ones are
larger -- median N from 579 to 806 -- which is exactly the regime where the
re-tuned radius was shown not to transfer (section 9.5). If ALPS's margin is an
artefact of small structures, this is where it breaks.

Caches are keyed on target name and the parameter set, so a stale ALPS computed
with the previous radius cannot silently survive into the new numbers.

Expensive readouts carry size caps: quantum Fisher information is O(N^2) per
residue and apop needs N dense eigensolves, so both are skipped above their cap
and their n is reported separately rather than pooled.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts"))
from methods.alps import alps_scores, spectral_response, RADIUS, KAPPA, K_MODES
from methods.btb import btb_scores
from methods.enm import apop_scores, corrsite_scores
from methods.qfi import qfi_scores
from methods.quantum import ctqw_communicability, ipr_resonant_transfer, noisy_or
from methods.common import (contact_graph, distal_nonanchor_mask, min_dist_to_anchor,
                            pocket_smooth, rank_percentile)
from partial_auc import stratified_auc

CAP_QFI, CAP_APOP = 700, 900
PARAMS = f"r{RADIUS}_k{KAPPA}_K{K_MODES}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_curated"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results_expanded.json"))
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    if res.get("_params") not in (None, PARAMS):
        print(f"cache was built with {res['_params']}, now {PARAMS} — starting fresh")
        res = {}
    res["_params"] = PARAMS

    for f in sorted(glob.glob(os.path.join(a.targets, "*.npz"))):
        name = os.path.basename(f).replace(".npz", "")
        if name in res:
            continue
        d = np.load(f); cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        if y.sum() == 0 or (pool & (y == 1)).sum() == 0:
            continue
        n = len(cb)
        A = contact_graph(cb, 10.0); dist = min_dist_to_anchor(cb, anchor)
        ctqw = ctqw_communicability(A, anchor)
        rng = np.random.default_rng(0)
        sc = {"ALPS": alps_scores(cb, anchor, pool),
              "ALPS_noresid": spectral_response(cb),
              "ctqw_only": ctqw,
              "qasc_baseline": noisy_or(ctqw, ipr_resonant_transfer(A, anchor)),
              "btb_raw": btb_scores(cb, anchor, distance_corrected=False),
              "corrsite": corrsite_scores(cb, anchor),
              "ctrl_closeness": -dist, "ctrl_dist": dist,
              "ctrl_burial": A.sum(axis=1), "ctrl_random": rng.random(n)}
        if n <= CAP_QFI:
            sc["qfi"] = qfi_scores(cb, anchor)
        if n <= CAP_APOP:
            sc["apop"] = apop_scores(cb, anchor)
        row = {"n": int(n)}
        for k, s in sc.items():
            sm = pocket_smooth(rank_percentile(s), A)
            auc, _ = stratified_auc(y, sm, pool, dist, 2.0)
            row[k] = None if auc != auc else float(auc)
        res[name] = row
        json.dump(res, open(a.out, "w"))
        print(f"  [{len(res)-1:3d}] {name} N={n}", flush=True)

    rows = {k: v for k, v in res.items() if k != "_params"}
    if not rows:
        return
    from scipy import stats
    FLOOR = 0.4963
    names = sorted({k for v in rows.values() for k in v if k != "n"})
    rnd = np.array([v.get("ctrl_random", np.nan) for v in rows.values()], float)
    print(f"\n=== expanded curated set, n={len(rows)}, params {PARAMS}, floor {FLOOR:.3f} ===")
    print(f"{'method':16s} {'strat AUC':>10s} {'vs floor':>9s} {'paired p':>9s} {'n':>4s}")
    out = []
    for m in names:
        x = np.array([v.get(m, np.nan) if v.get(m) is not None else np.nan
                      for v in rows.values()], float)
        ok = ~np.isnan(x) & ~np.isnan(rnd)
        if ok.sum() < 10:
            continue
        p = (np.nan if m == "ctrl_random" or np.allclose(x[ok], rnd[ok])
             else stats.wilcoxon(x[ok], rnd[ok]).pvalue)
        out.append((np.nanmean(x), m, np.nanmean(x) - FLOOR, p, int(ok.sum())))
    for au, m, dm, p, nn in sorted(out, reverse=True):
        ps = "reference" if np.isnan(p) else f"{p:.4f}"
        star = "" if np.isnan(p) or p >= 0.05 else " *"
        print(f"{m:16s} {au:10.3f} {dm:+9.3f} {ps:>9s} {nn:4d}{star}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Does the degeneracy-sensitive readout win where degeneracy actually exists?

Every interference-dependent observable lost on single chains. The stated reason
was that residue contact graphs of single chains have no eigenvalue degeneracies,
and interference needs degeneracies. Symmetric oligomers are the regime where
that premise is different, so this re-runs the same readout ablation there and
splits the targets by whether the chains really are symmetric copies (Kabsch
RMSD between equal-length chains).
"""
from __future__ import annotations
import argparse, collections, glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts"))
from ablate_readouts import readouts, LABEL                                # noqa: E402
from methods.alps import distance_zscore                                   # noqa: E402
from methods.common import (contact_graph, distal_nonanchor_mask,          # noqa: E402
                            laplacian, min_dist_to_anchor, pocket_smooth,
                            rank_percentile)
from evaluate import auc_in_pool, permutation_p                            # noqa: E402


def kabsch_rmsd(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    V, _S, W = np.linalg.svd(P.T @ Q)
    R = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ W))]) @ W
    return float(np.sqrt(((P @ R - Q) ** 2).sum() / len(P)))


def chain_rmsd(d):
    """RMSD between two chains, aligned on their shared author residue numbers.

    Requiring equal chain lengths is too strict: symmetric oligomers routinely
    have different numbers of resolved residues per chain, which would misfile
    a genuinely symmetric dimer as unverified.
    """
    if "chain_id" not in d.files:
        return None
    ci = d["chain_id"]; cb = d["cb"]; rn = d["resnums"]
    ks = sorted(set(ci.tolist()))
    if len(ks) < 2:
        return None
    best = None
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            ma, mb = ci == ks[i], ci == ks[j]
            ra, rb = rn[ma], rn[mb]
            common = np.intersect1d(ra, rb)
            if len(common) < 50:
                continue
            ia = np.array([np.where(ra == c)[0][0] for c in common])
            ib = np.array([np.where(rb == c)[0][0] for c in common])
            r = kabsch_rmsd(cb[ma][ia], cb[mb][ib])
            if best is None or r < best:
                best = r
    return best


def degeneracy(cb, nlow=20):
    w = np.linalg.eigvalsh(laplacian(contact_graph(cb)))
    nz = w[w > 1e-9][:nlow]
    g = np.diff(nz) / nz[:-1]
    return float(np.mean(g < 0.01))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(HERE, "data", "targets_multimer"))
    ap.add_argument("--max-n", type=int, default=560)
    ap.add_argument("--sym-cut", type=float, default=1.0)
    ap.add_argument("--cache", default=os.path.join(HERE, "data", "multimer_readouts.json"))
    a = ap.parse_args()

    # per-target cache: the readouts are the expensive part (N eigendecompositions
    # each), so a run that is interrupted does not lose the work
    cache = {}
    if os.path.exists(a.cache):
        cache = json.load(open(a.cache))
    groups = {"symmetric": collections.defaultdict(list),
              "unverified": collections.defaultdict(list)}
    deg = {"symmetric": [], "unverified": []}
    for f in sorted(glob.glob(os.path.join(a.targets, "*.npz"))):
        d = np.load(f)
        cb, anchor, y = d["cb"], d["anchor"], d["y"].astype(int)
        if len(cb) > a.max_n or y.sum() == 0:
            continue
        r = chain_rmsd(d)
        grp = "symmetric" if (r is not None and r < a.sym_cut) else "unverified"
        A = contact_graph(cb, 10.0)
        pool = distal_nonanchor_mask(cb, anchor, 8.0)
        bg = pool & (y != 1)
        if bg.sum() < max(3, int(y.sum())):
            continue
        dist = min_dist_to_anchor(cb, anchor)
        name = os.path.basename(f)
        if name in cache:
            rec = cache[name]
        else:
            R = readouts(cb, anchor)
            rec = {"deg": degeneracy(cb)}
            for key, v in R.items():
                sm = pocket_smooth(rank_percentile(distance_zscore(v, dist, pool)), A)
                rec[key] = [permutation_p(y, sm, bg), auc_in_pool(y, sm, pool)]
            cache[name] = rec
            json.dump(cache, open(a.cache, "w"))
        deg[grp].append(rec["deg"])
        for key in ("dlam", "dgap", "dpart", "dipr"):
            groups[grp][key].append(tuple(rec[key]))
        print(f"  {name:16s} N={len(cb):4d} "
              f"RMSD={'n/a' if r is None else f'{r:.2f}'} -> {grp}"
              f"{' (cached)' if name in cache else ''}", flush=True)

    for grp in ("symmetric", "unverified"):
        rows = groups[grp]
        if not rows.get("dlam"):
            continue
        n = len(rows["dlam"])
        print(f"\n=== {grp} oligomers (n={n}, mean fraction of gaps <1% = "
              f"{np.mean(deg[grp])*100:.1f}%) ===")
        print(f"{'readout':24s} {'sig':>7s} {'medP':>8s} {'AUC':>7s}")
        for key in ("dlam", "dgap", "dpart", "dipr"):
            p = np.array([x[0] for x in rows[key]])
            au = np.array([x[1] for x in rows[key]])
            print(f"{LABEL[key]:24s} {np.mean(p < 0.05)*100:6.1f}% "
                  f"{np.median(p):8.4f} {au.mean():7.3f}")


if __name__ == "__main__":
    main()

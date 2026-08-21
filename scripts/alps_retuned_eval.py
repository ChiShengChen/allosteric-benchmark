#!/usr/bin/env python
"""Stratified AUC of the re-tuned ALPS on the full curated set (incremental)."""
from __future__ import annotations
import glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(HERE,"scripts"))
from methods.alps import alps_scores, spectral_response
from methods.common import (contact_graph, distal_nonanchor_mask, min_dist_to_anchor,
                            pocket_smooth, rank_percentile)
from partial_auc import stratified_auc
OUT=os.path.join(HERE,"data","alps_retuned_strat.json")
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
for f in sorted(glob.glob(os.path.join(HERE,"data","targets_curated","*.npz"))):
    name=os.path.basename(f).replace(".npz","")
    if name in res: continue
    d=np.load(f); cb,anchor,y=d["cb"],d["anchor"],d["y"].astype(int)
    pool=distal_nonanchor_mask(cb,anchor,8.0)
    if y.sum()==0 or (pool&(y==1)).sum()==0: continue
    A=contact_graph(cb,10.0); dist=min_dist_to_anchor(cb,anchor)
    row={}
    for nm,sc in (("ALPS_retuned",alps_scores(cb,anchor,pool)),
                  ("ALPS_noresid_retuned",spectral_response(cb))):
        sm=pocket_smooth(rank_percentile(sc),A)
        a,_=stratified_auc(y,sm,pool,dist,2.0)
        row[nm]=None if a!=a else float(a)
    res[name]=row; json.dump(res,open(OUT,"w"))
    print(f"  [{len(res):3d}] {name}",flush=True)
print(f"\nfull curated set, n={len(res)}, floor 0.496")
for k in ("ALPS_retuned","ALPS_noresid_retuned"):
    v=np.array([r[k] for r in res.values() if r.get(k) is not None],float)
    print(f"  {k:22s} {v.mean():.3f}   (was 0.578 / 0.579 with the old parameters)")

# gnn — the one model family that runs on this input and had not been tried

A literature search over 2,536 deduplicated papers ([`../docs/ai-model-landscape.md`](../docs/ai-model-landscape.md),
27 verified evidence cards) asked which AI model families can do allosteric-site
prediction, and then asked the narrower question that actually matters here: which
of them run on **our** input — Cβ coordinates and active-site indices, no sequence,
no side chains, no MSA, no trajectory, ~90 labelled proteins.

Two survived. One is the elastic-network route, which is what ALPS already is. The
other is a **residue-graph GNN**, and nobody in the corpus had characterised it at
residue level on an allosteric task. This folder tries it.

## What the search says to expect

| finding from the corpus | consequence here |
|---|---|
| Protein language models hold AUROC 0.70 on allosteric sites while AUPR falls to 0.06, and the gap survives controls for sequence similarity, structural redundancy and imbalance | rank-ordering ability and usable localisation are different things; report both kinds of metric |
| PASSer2.0's own untrained FPocket baseline puts a positive pocket in the top three for 84.3% of test proteins, against 82.7% for its AutoML model | the trivial geometric baseline is the thing to beat, not an afterthought |
| Allosteric supervision is ~3,000 sites against ~14,000 for general binding sites; standard training sets are 90–235 proteins | keep the network small; this is not a regime for deep architectures |
| Distance-dependent allosteric decay "is observed in all complete allosteric maps generated to date" | distance is real signal *and* a confound — hence the stratified metric, and hence `ctrl_dist` in every table |

## The design decision worth stating

**Distance to the active site is not a base node feature.** Section 10 of the main
README spent its length showing that proximity dominates plain AUC on both label
sets, and that a learned combiner handed the distance channel will reproduce it. So
the model gets the active site as an *indicator* on the graph and has to propagate
along contacts to discover anything about reach.

Whether that restraint costs anything is measured rather than assumed: `--with-dist`
adds the channel back and the two runs are compared.

Node features, all derivable from Cβ alone: anchor indicator, contact-graph degree,
burial count. Edges are the same 10 Å contact graph every other method in this
repository uses, with an RBF expansion of edge length.

Depth is a physical choice — each layer is one hop on a 10 Å graph, so 4 layers
reach roughly 40 Å, the scale an allosteric signal travels. Width 24 over 4 layers
is ~15k parameters, which is already generous for ~90 proteins.

## Protocol

Inherited unchanged, because each rule caught a real error upstream: curated labels,
distance-stratified AUC, protein-grouped folds, early stopping on an inner
validation split so model selection never touches the test fold, `ctrl_random` and
`ctrl_dist` in the table, floor from 25 seeds, paired Wilcoxon tests.

The opponent is ALPS on the same targets under the same metric — identical input,
one hand-designed spectral readout against one learned message-passing model.

```
data.py    build graph tensors + the ALPS baseline on the curated set
model.py   the message-passing network
run.py     train, evaluate, compare
```

## Result

| | stratified AUC | GNN − ALPS | paired p |
|---|---|---|---|
| GNN, seed 0 | **0.622** | +0.030 | 0.136 |
| GNN, seed 1 | **0.630** | +0.038 | 0.151 |
| GNN + dist channel | 0.595 | +0.003 | 0.787 |
| ALPS | 0.592 | — | — |

The strongest method in this repository, decisively above the random control in both
seeds, and **not** distinguishable from ALPS — the margin reproduces across splits but
the paired test cannot separate them at n = 96.

The ablation is the finding worth keeping: giving the model the distance channel makes
it **worse**, 0.622 → 0.595, and collapses its margin over ALPS. Denying the confound
did not cost anything; it helped. Full numbers in [`RESULTS.md`](RESULTS.md).

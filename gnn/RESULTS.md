# gnn — results

96 curated targets, 86,794 residues, 78,509 in the distal non-anchor pool, 1,050
evaluable positives (1.3% of pool). Protein-grouped 5-fold CV, distance-stratified
AUC, early stopping on an inner validation split so model selection never touches the
test fold. 14,161 parameters, hidden 24, 4 layers.

## The headline

| | stratified AUC | vs floor | p vs random | GNN − ALPS | paired p |
|---|---|---|---|---|---|
| **GNN**, seed 0 | **0.622** | +0.126 | 0.0000 | +0.030 | 0.136 |
| **GNN**, seed 1 | **0.630** | +0.134 | 0.0000 | +0.038 | 0.151 |
| GNN + dist channel, seed 0 | 0.595 | +0.099 | 0.0002 | +0.003 | 0.787 |
| ALPS (deterministic) | 0.592 | +0.096 | — | — | — |
| CONTROL `ctrl_dist` | 0.509 | +0.013 | — | — | — |

The GNN is the strongest method this repository has produced, and it beats the random
control decisively in both seeds. **It does not beat ALPS.** The gap reproduces across
two independent splits — +0.030 and +0.038, so it is not a split artifact — but the
paired test cannot separate the two at n = 96, twice, at p ≈ 0.14. Under the
Bonferroni threshold this repository uses elsewhere (0.05/11 = 0.0045) it is not close.

The honest statement is: **a learned message-passing model and a hand-designed
spectral readout perform the same on this task, with the GNN nominally ahead.**

## The ablation is the interesting result

Handing the model the distance-to-anchor channel makes it **worse** — 0.622 → 0.595 —
and collapses its margin over ALPS from +0.030 to +0.003.

This is stronger than "the restraint cost nothing". Denying the confound *helped*.
Given the channel, the network spends capacity reproducing distance instead of
learning propagation on the graph; denied it, it finds something distance does not
already encode. That is the same behaviour §10 measured on the learned combiner,
reproduced now in a completely different model family — which makes it look like a
property of the task rather than of any one architecture.

So part of the base model's 0.622 exists *because* it was not given distance.

## What the two seeds say about the controls

| seed | `ctrl_random` |
|---|---|
| 0 | 0.522 |
| 1 | 0.480 |

Two draws, 0.042 apart, bracketing the 25-seed floor estimate of 0.4963 ± 0.0157. This
is why the "vs floor" column uses the multi-seed estimate and not the run's own draw —
a single draw treated as the floor is an error recorded in §10 of the main README.

The instability propagates. `ctrl_dist` scores 0.509 in both runs, being
deterministic, yet its p against the random control is 0.4120 under seed 0 and 0.0076
under seed 1 — the same number, two very different verdicts, purely because the
comparison moved. Any p-value in this repository computed against a *single* random
draw should be read with that spread in mind.

## Why more seeds will not fix this, and what would

The paired test is over 96 targets. Running more seeds averages away initialisation
and split noise, which is already small — the two seeds agree to 0.008. It does
nothing about target-level variance, which is what n = 96 limits.

That is precisely the constraint the literature survey identified
([`../docs/ai-model-landscape.md`](../docs/ai-model-landscape.md)) and the vendored
AlloBench pipeline addresses: 1,439 samples over 327 UniProt accessions against our
97 curated targets. Whether a +0.03 margin is real is answerable at that scale and
not at this one.

Two things must not be smoothed over when that comparison is run: those coordinates
are Cα where ours are Cβ, and those labels are 4 Å-to-modulator where ours are expert
annotation. §1.5 of the main README records why the two sets are evaluated separately.

## Reproducing

```bash
python3 gnn/data.py                  # ~25 min, builds graphs.npz (not committed, 72 MB)
python3 gnn/run.py                   # base model, seed 0
python3 gnn/run.py --seed 1          # the stability check
python3 gnn/run.py --with-dist       # the ablation
```

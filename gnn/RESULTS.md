# gnn — results

Two evaluations of the same model on two different target sets, reported separately
because they are not interchangeable (§1.5 of the main README): the 96 curated targets
this repository built, and the 1,042 AlloBench targets the vendored pipeline produces.
The curated set came first and is reported first; the AlloBench result is the one that
answers the question the curated set could only pose.

## The curated set

96 curated targets, 86,794 residues, 78,509 in the distal non-anchor pool, 1,050
evaluable positives (1.3% of pool). Protein-grouped 5-fold CV, distance-stratified
AUC, early stopping on an inner validation split so model selection never touches the
test fold. 14,161 parameters, hidden 24, 4 layers.

### The headline

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

### The ablation is the interesting result

Handing the model the distance-to-anchor channel makes it **worse** — 0.622 → 0.595 —
and collapses its margin over ALPS from +0.030 to +0.003.

This is stronger than "the restraint cost nothing". Denying the confound *helped*.
Given the channel, the network spends capacity reproducing distance instead of
learning propagation on the graph; denied it, it finds something distance does not
already encode. That is the same behaviour §10 measured on the learned combiner,
reproduced now in a completely different model family — which makes it look like a
property of the task rather than of any one architecture.

So part of the base model's 0.622 exists *because* it was not given distance.

### What the two seeds say about the controls

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

### Why more seeds will not fix this, and what would

The paired test is over 96 targets. Running more seeds averages away initialisation
and split noise, which is already small — the two seeds agree to 0.008. It does
nothing about target-level variance, which is what n = 96 limits.

That is precisely the constraint the literature survey identified
([`../docs/ai-model-landscape.md`](../docs/ai-model-landscape.md)) and the vendored
AlloBench pipeline addresses. Whether a +0.03 margin is real is answerable at that
scale and not at this one. The next section runs it, and the answer is no.

Two things must not be smoothed over in reading that section: those coordinates are
Cα where ours are Cβ, and those labels are 4 Å-to-modulator where ours are expert
annotation. §1.5 of the main README records why the two sets are evaluated separately
rather than pooled.

## The AlloBench set

1,042 targets over 265 distinct UniProt accessions, 369,988 residues in the pool at
2.63% positive — 9.3× the evaluable positives of the curated set. The 5-fold split is
grouped by UniProt accession and carried in the dataset rather than assigned here, so
homologues of the same protein cannot straddle the test boundary. Same model,
unchanged: 14,161 parameters, hidden 24, 4 layers.

The build came in below the 1,439 samples over 327 accessions the pipeline
advertises: 1,042 over 265 survive structure retrieval and the evaluability filter (a
target needs at least one positive inside the distal non-anchor pool). Every number
here is on what was actually built.

### The headline

| | stratified AUC | vs floor | p vs random | GNN − ALPS | paired p |
|---|---|---|---|---|---|
| **GNN**, seed 0 | **0.632** | +0.130 | 0.0000 | +0.0195 | **0.0447** |
| **GNN**, seed 1 | **0.623** | +0.121 | 0.0000 | +0.0113 | 0.1447 |
| ALPS (deterministic) | 0.612 | +0.110 | 0.0000 | — | — |
| CONTROL `ctrl_dist` | 0.537 | +0.035 | 0.0000 | — | — |
| CONTROL `ctrl_random` | 0.495 / 0.502 | −0.007 / +0.000 | reference | — | — |

Paired over the n = 1,016 targets where both methods return a defined AUC. "vs floor"
uses **0.5020**, re-estimated on this set (below) — not the 0.4963 constant
`gnn/run.py` still prints by default, which belongs to the curated set.

### The margin did not survive the larger sample

n = 96 diagnosed the problem correctly — the limit was target-level variance — but
supplying ten times the targets did not convert the margin into a win. It shrank it,
from +0.030 / +0.038 to **+0.0195 / +0.0113**, and left the verdict seed-dependent:
p = 0.0447 under seed 0, p = 0.1447 under seed 1. Against the Bonferroni threshold
this repository uses elsewhere (0.05/11 = 0.0045), neither is close.

A margin that shrinks as n grows is the signature of a small-sample effect being
partly noise, not of a real effect finally becoming measurable. The direction is
consistent across four runs on two datasets, which is worth something; the size is
not what the curated set suggested.

**The statement stands, now at n = 1,042: a learned message-passing model and a
hand-designed spectral readout perform the same on this task, with the GNN nominally
ahead.** What changed is that this is no longer an artefact of a small sample — it is
the result.

### The control instability is a small-sample artefact, and it is gone

`ctrl_random` draws 0.495 and 0.502 here, 0.007 apart, against 0.522 and 0.480 on the
curated set, 0.042 apart. Averaging over 1,042 targets instead of 96 does to the
control exactly what it is supposed to do. The warning recorded above — that any
p-value computed against a *single* random draw carries that spread — applies to the
curated numbers and no longer to these.

### The floor is 0.5020, not 0.4963

Re-estimated over 25 seeds on this set (`gnn/floor_allobench.py`):

| control | mean | sd | range |
|---|---|---|---|
| raw (what `gnn/run.py` scores) | **0.5020** | 0.0046 | [0.4936, 0.5108] |
| pocket-smoothed (what `scripts/floor_and_tests.py` scores) | 0.5022 | 0.0050 | [0.4925, 0.5126] |

Two results. The curated-set constant 0.4963 is 1.2 sd low here and cannot be carried
across — a floor is a property of the data, not a universal constant. And **smoothing
does not move the floor**: +0.0003, far inside the seed noise. The two scripts' nulls
were estimated separately rather than assumed equal, and they turn out to be the same
null on this set.

### The inner validation split is grouped too

The test fold is grouped by UniProt, but carving the inner validation split out at
random would let homologues of the same accession sit on both sides of it, and
AlloBench carries up to six structures per protein. That does not contaminate the test
number, which early stopping never sees; it corrupts the *selection criterion*. On the
curated set the bug was invisible — one structure per protein makes a random inner
split already effectively grouped — which is why it survived until this run.

Grouping it did not inflate the validation scores: best val stratAUC lands at
0.598–0.796 across the ten folds, the same range as the test numbers, with no sign of
the val-far-above-test gap that leakage produces.

## Reproducing

```bash
# curated set
python3 gnn/data.py                  # ~25 min, builds graphs.npz (not committed, 72 MB)
python3 gnn/run.py                   # base model, seed 0
python3 gnn/run.py --seed 1          # the stability check
python3 gnn/run.py --with-dist       # the ablation

# AlloBench set
scripts/build_allobench.sh                            # datasets under data/targets_allobench
python3 gnn/merge_folds.py                            # adds fold + uniprot to the cache
python3 gnn/floor_allobench.py                        # ~ the floor above, 25 seeds
python3 gnn/run.py --cache gnn/graphs_allobench.npz --floor 0.5020 --seed 0
python3 gnn/run.py --cache gnn/graphs_allobench.npz --floor 0.5020 --seed 1
```

`--floor` is not optional on this set: the default is the curated-set constant and
will overstate every margin by 0.006.

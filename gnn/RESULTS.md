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

Eight runs at the same seed, because two turned out not to be enough — see below.

| | stratified AUC | vs floor | vs ALPS | paired p |
|---|---|---|---|---|
| **GNN**, single run | **0.623 ± 0.010** | +0.121 | **+0.011 ± 0.009** | 0.05–0.70 |
| ALPS (deterministic, identical every run) | 0.612 | +0.110 | reference | — |
| CONTROL `ctrl_dist` | 0.537 | +0.035 | −0.075 | — |
| CONTROL `ctrl_random` | 0.496 | −0.006 | −0.116 | — |

Mean ± sd over eight runs at seed 0; ALPS returns the same 0.612 in all eight. Paired
over the n = 1,016 targets where both methods are defined. "vs floor" uses **0.5020**,
re-estimated on this set (below) — not the 0.4963 constant `gnn/run.py` prints by
default, which belongs to the curated set.

### The margin did not survive the larger sample, and neither did its p-value

n = 96 diagnosed the problem correctly — the limit was target-level variance — but ten
times the targets did not convert the margin into a win. It shrank it, from +0.030 /
+0.038 on the curated set to **+0.011 ± 0.009**, and the paired p across eight identical
runs ranges from 0.046 to 0.702, landing below 0.05 in **two of eight**. Against the
Bonferroni threshold used elsewhere here (0.05/11 = 0.0045), none of them is close.

A margin that shrinks as n grows is the signature of a small-sample effect that was
partly noise, not of a real one finally becoming measurable.

**The statement stands at n = 1,042: as single predictors, a learned message-passing
model and a hand-designed spectral readout perform the same on this task.** What
changed is that this is no longer an artefact of a small sample. It is the result —
and, per the fusion section below, it is also not the whole story, because two methods
of equal average skill can still be worth combining.

### Two seeds was never enough, because the same seed does not reproduce

An earlier version of this file reported +0.0195 at p = 0.0447 for seed 0 and +0.0113
at p = 0.1447 for seed 1, and read the pair as a seed effect. It is not one. Rerunning
**seed 0** eight times, changing nothing, gives:

| | |
|---|---|
| GNN stratified AUC | 0.609 – 0.634, sd 0.0095 |
| GNN − ALPS | −0.0030 to +0.0218, mean +0.0107, sd 0.0094 |
| runs reporting p < 0.05 | 2 of 8 |

**The run-to-run spread at a fixed seed is the same size as the effect being
measured.** The published +0.0195 / p = 0.0447 was a high draw from that distribution,
and seed 1's 0.623 sits in the middle of seed 0's own range — so there is no detectable
seed effect at all. What looked like two independent confirmations was one distribution
sampled twice.

The cause is not the obvious one. Twenty training steps reproduce bit-for-bit across
processes at a fixed thread count, so the first suspect was the OpenMP pool shrinking
under load. Pinning it (`OMP_DYNAMIC=FALSE`) was tested against the default in two waves
of three runs each and **did not remove the spread** (pinned 0.621/0.634/0.626, default
0.621/0.609/0.630). The remaining candidate is the parallel reduction in `index_add_` on
the larger graphs, which the twenty-step probe was too small to trigger. A reproducible
number would need single-threaded execution, at roughly eight times the cost.

The methodological point generalises past this model: **reporting two seeds proves
nothing when run-to-run variance at a fixed seed equals the effect.** The curated-set
numbers in this file were reported the same way and carry the same caveat.

### The tie hid a complementarity, and it is worth 0.07

Equal average skill does not mean the same predictions. Per target, the two barely
agree:

| GNN against ALPS, per target | |
|---|---|
| Pearson r | **0.146** |
| Spearman | 0.100 |
| targets where the GNN is ahead | 47% |
| oracle ceiling, cross-fitted | **0.708** against 0.612 |

The ceiling picks the better method per target using disjoint runs to decide and to
score, so it is not selection on noise; the naive version is 0.714, barely higher.

`gnn/fuse.py` does the combination properly — rank-percentile both scores, mix them,
choose the weight **on the other folds** and never on the fold being scored:

| method | stratified AUC | vs floor | vs ALPS | paired p |
|---|---|---|---|---|
| ALPS | 0.612 | +0.110 | reference | — |
| GNN, 1 run | 0.610 | +0.108 | −0.002 | 2.8e-01 |
| GNN, 7-run mean | 0.663 | +0.161 | +0.051 | 1.8e-07 |
| fuse 50/50, fixed | 0.673 | +0.171 | +0.061 | 2.9e-50 |
| **fuse, nested weight** | **0.685** | **+0.183** | **+0.073** | 2.4e-30 |
| CONTROL `ctrl_dist` | 0.537 | +0.035 | −0.075 | — |
| CONTROL `ctrl_random` | 0.496 | −0.006 | −0.116 | — |

All five folds independently selected w = 0.7 on their training data, so the 70/30 mix
is what honest selection picks, not what reading the test numbers suggests.

Two separable gains, at different prices:

| k runs averaged | GNN alone | fused 50/50 |
|---|---|---|
| 1 | 0.610 | **0.652** |
| 3 | 0.650 | 0.671 |
| 5 | 0.662 | 0.673 |
| 7 | 0.663 | 0.673 |

**Fusion costs nothing** — both score vectors already exist, and a single GNN run mixed
with ALPS is +0.040 over ALPS alone. **Averaging runs costs k times the training** and
adds another +0.05 to the GNN on its own, saturating around k = 5.

That second gain is the run-to-run noise from the section above, read the other way
round: at the per-residue level it is largely independent between runs, so averaging
removes it. The single-run number was being held down by its own irreproducibility.
The same nondeterminism that invalidated the two-seed claim is, once averaged rather
than sampled, the largest single improvement measured in this repository.

None of this changes the head-to-head verdict. The GNN does not beat ALPS as a
predictor. It carries information ALPS does not, which is a different claim, and only
the second one survives a paired test.

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

# gnn — results

One model, evaluated on target sets that are **not interchangeable** (§1.5 of the main
README) and are therefore reported separately and never pooled:

| section | targets | labels | coordinates |
|---|---|---|---|
| The curated set | 96 | expert-curated | Cβ |
| The AlloBench set | 1,042 | 4 Å heavy-atom to modulator | Cα |
| The coordinate convention | 1,209 / 950 matched | 4 Å heavy-atom to modulator | **Cβ** |

The curated set came first. The AlloBench set was built to answer what n = 96 could only
pose. The third section rebuilds the second on the coordinate convention the rest of this
repository uses, and is where the head-to-head question actually gets an answer — which
turns out to be "it depends on the size of the active site", not a single number.

## Read this first: what this file has said, and what replaced it

This file was rewritten several times in a few days, and **each rewrite corrected a
number the previous one published.** The corrections are kept in the git history rather
than quietly overwritten, which is the right call for a repository whose whole claim is
that it can be trusted — but it means a reader who arrives at any single commit can pick
up a figure that a later one retracted.

Every retraction, newest first. The right-hand column is what the file says now.

| this file once said | it now says |
|---|---|
| **The GNN margin reproduced across "two independent splits", +0.030 and +0.038** | Not two splits. One seed rerun four times spans 0.607–0.623, so 0.622 was a high draw and 0.630 is one more sample from the same distribution. There is no detectable seed effect |
| **+0.0195 over ALPS at paired p = 0.0447 — significant** | +0.0107 ± 0.0094 over eight runs at one seed, p < 0.05 in two of them. The published run was a high draw |
| **Averaging runs is worth +0.051, the largest single improvement here** | +0.041. The baseline was whichever run happened to be dump 0, and it scored 0.610 against a 0.623 single-run mean. It also does not replicate on the curated set (+0.003) |
| **The larger sample settles it: the GNN does not beat ALPS** | True on the full sets and not the whole picture. On 950 node-for-node identical targets rebuilt on Cβ the margin is +0.030, and across seed sizes it runs from −0.065 to +0.062 |
| **The Cβ margin is +0.023 (three runs)** | +0.0144 ± 0.0130 (eight runs). All three of the first runs sat above the eight-run mean |
| **First-order perturbation theory preserves the ranking, −0.013 AUC** | −0.032, measured on 25 targets rather than the 4-target smoke test. Same direction of error as everything above |
| **ALPS scores 0.694 on apo against 0.592/0.612 on holo** | That compares different target populations. Node-for-node matched pairs give apo 0.674 against holo 0.708 at paired p = 0.90 — the conformational cost is not measurable at n = 15 |

**Two structural notes.** Between commits `0210027` and `07f7729` this file was
physically broken: a string replacement intended for the AlloBench section matched the
curated one, because both begin `### The headline`, and the curated results, the
distance-channel ablation and the AlloBench section heading were deleted. Anything read
from those two commits is unreliable regardless of the numbers above. And the AlloBench
figures throughout are Cα unless a heading says Cβ; the two are not interchangeable and
the last section measures by how much.

**The pattern worth carrying out of this file.** Six of the seven rows above are the
same error: a noisy quantity estimated from too few draws, coming in on the flattering
side. Two seeds, one dump, three runs, four targets. None of them was a mistake in
arithmetic and none was caught by review — each was caught by *sampling the same thing
more times*. §12.6 of the main README now names all five instances.

---

## The curated set

96 curated targets, 86,794 residues, 78,509 in the distal non-anchor pool, 1,050
evaluable positives (1.3% of pool). Protein-grouped 5-fold CV, distance-stratified
AUC, early stopping on an inner validation split so model selection never touches the
test fold. 14,161 parameters, hidden 24, 4 layers.

### The headline

Four runs at the same seed. The two-seed version this section used to report is
corrected below and in the AlloBench section, which is where the spread was measured
properly.

| | stratified AUC | vs floor | vs ALPS | paired p |
|---|---|---|---|---|
| **GNN**, single run | **0.616 ± 0.007** | +0.120 | **+0.024 ± 0.007** | 0.17 – 0.50 |
| GNN + dist channel, one run | 0.595 | +0.099 | +0.003 | 0.787 |
| ALPS (deterministic, identical every run) | 0.592 | +0.096 | reference | — |
| CONTROL `ctrl_dist` | 0.509 | +0.013 | −0.083 | — |

Mean ± sd over four runs at seed 0 (0.607, 0.615, 0.618, 0.623); ALPS returns 0.592 in
all four. The GNN is the strongest single predictor this repository has produced and
beats the random control decisively in every run. **It does not beat ALPS** — the
paired test cannot separate them at n = 96, and under the Bonferroni threshold used
elsewhere here (0.05/11 = 0.0045) it is not close.

**What this table used to say, and why it was wrong.** It reported seed 0 at 0.622 and
seed 1 at 0.630 and read the pair as two independent confirmations: "the gap reproduces
across two independent splits, so it is not a split artifact." Rerunning seed 0 alone
four times spans 0.607 to 0.623, so 0.622 was a high draw and 0.630 is one more sample
from the same distribution. There was no second confirmation. The mechanism, and the
eight-run version of the same measurement, are in the AlloBench section below.

The honest statement is unchanged: **a learned message-passing model and a
hand-designed spectral readout perform the same on this task, with the GNN nominally
ahead.** What is new is that fusing them does better than either — see below.

### The ablation is the interesting result

Handing the model the distance-to-anchor channel makes it **worse** — 0.616 → 0.595 —
and collapses its margin over ALPS from +0.024 to +0.003.

This is stronger than "the restraint cost nothing". Denying the confound *helped*.
Given the channel, the network spends capacity reproducing distance instead of
learning propagation on the graph; denied it, it finds something distance does not
already encode. That is the same behaviour §10 measured on the learned combiner,
reproduced now in a completely different model family — which makes it look like a
property of the task rather than of any one architecture.

The gap is 0.021 against a run-to-run sd of 0.007, so roughly three times the noise and
it survives the correction above. The distance-channel arm was run once, though, and
its own spread has not been measured.

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

The paired test is over 96 targets, and target-level variance is what n = 96 limits.
More seeds do not touch it.

*(This section used to argue the point by saying initialisation and split noise were
"already small — the two seeds agree to 0.008". That agreement was luck: four runs at
one seed span 0.016. The conclusion survives the correction, because target-level
variance is a separate quantity from run noise and is still what n = 96 bounds.)*

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

**The two sets share most of their structures.** 75 of the 94 distinct curated PDB ids
(80%) also appear here, and at least 66 of the 265 accessions contain a curated
structure. That does not break the separation rule — the labels and coordinates differ
and the sets are never pooled — but it means the AlloBench result is not an independent
sample of proteins, and **training on one and testing on the other would be 80%
contaminated**.

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
| GNN, 1 run (mean over the 7) | 0.622 | +0.120 | +0.010 | 7.8e-01 |
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
| 1 | 0.622 | **0.657** |
| 3 | 0.652 | 0.669 |
| 5 | 0.658 | 0.671 |
| 7 | 0.663 | 0.673 |

Each row averages over subsets of that size rather than taking the first k dumps. That
matters: the first AlloBench dump happened to score 0.610 against a 0.623 single-run
mean, and reading the k = 1 row off it inflated the apparent ensemble gain from +0.041
to +0.051 in the first version of this section.

**Fusion costs nothing** — both score vectors already exist, and one GNN run mixed with
ALPS is **+0.045** over ALPS alone. **Averaging runs costs k times the training** and
adds a further +0.041 to the GNN on its own, saturating around k = 5.

That second gain is the run-to-run noise from the section above, read the other way
round: at the per-residue level it is largely independent between runs, so averaging
removes it, and the single-run number was being held down by its own irreproducibility.

### Both claims were re-run on the curated set, and only one replicated

Four runs at seed 0 on the 96 curated targets, same script, floor 0.4963:

| method | stratified AUC | vs ALPS | paired p |
|---|---|---|---|
| ALPS | 0.592 | reference | — |
| GNN, 1 run (mean over the 4) | 0.616 | +0.024 | 3.0e-01 |
| GNN, 4-run mean | 0.619 | +0.027 | 2.3e-01 |
| **fuse 50/50, fixed** | **0.625** | **+0.033** | **2.4e-03** |
| fuse, nested weight | 0.629 | +0.036 | 2.8e-02 |
| CONTROL `ctrl_dist` | 0.509 | −0.083 | 1.0e-03 |
| CONTROL `ctrl_random` | 0.497 | −0.095 | 4.0e-04 |

**Fusion replicates.** +0.033 here against +0.061 there, on a different label rule,
different coordinates and a twentieth of the targets — and at p = 2.4e-03 the fixed
50/50 mix is the first result in this repository to clear the Bonferroni threshold it
uses elsewhere (0.05/11 = 0.0045) against ALPS. The nested weight scores higher and
tests worse (p = 2.8e-02), which is the price of letting the weight vary per fold. All
nine folds across both datasets chose 0.7 or 0.8, so the mixture ratio is stable even
where the gain is not.

**Averaging runs does not replicate.** k = 1 to 4 moves the curated GNN from 0.616 to
0.619, +0.003, against +0.041 on AlloBench. Same procedure, same seed handling, two
very different answers, and nothing measured here explains which property of the two
sets decides it. Recorded as dataset-dependent rather than as a technique — it earns
its k trainings on one of the two sets tried.

None of this changes the head-to-head verdict. The GNN does not beat ALPS as a
predictor on either set. It carries information ALPS does not, which is a different
claim, and it is the fused score, not the GNN, that survives a paired test.

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

## The coordinate convention, and what the seed size decides

Everything above on the AlloBench route was computed on **Cα**, because the vendored
pipeline shipped only Cα when it was integrated. Upstream now ships a parallel `cb/`
directory — Cβ, Cα substituted at glycine, identically keyed — and every other set in
this repository is Cβ, which is also what ALPS's `RADIUS = 12.0` was tuned on. §1.6
lists the coordinate difference as one of three load-bearing incompatibilities between
the two sets. This section measures what it was worth.

The Cβ cache was built through
[`allosteric-datasets`](https://github.com/ChiShengChen/allosteric-datasets), which
converts the upstream `cb/` directory into this repository's format. **On the 950
targets the two caches share, the node sets and label vectors are identical in 100% of
cases** — only the coordinates differ, and with them the contact graph (the Cα graph
carries 63 more edges per target on average). So this is a one-variable comparison.

### Cβ is worth +0.011 to ALPS and +0.039 to the GNN

Same 950 targets, k = 7 runs on both sides:

| | Cα | Cβ | Cβ − Cα | paired p |
|---|---|---|---|---|
| ALPS | 0.6074 | 0.6171 | **+0.0113** | 2.5e-06 |
| GNN, single run | 0.6073 | 0.6468 | **+0.0388** | 4.0e-13 |
| GNN, 7-run mean | 0.6478 | 0.6784 | +0.0297 | 9.2e-07 |
| fuse 50/50 | 0.6600 | 0.6797 | +0.0198 | 2.0e-09 |

**The learned model gains three and a half times what the hand-designed one gains.**
That is mechanistically unsurprising and it is worth stating rather than assuming: the
GNN passes messages along individual contacts, and Cβ geometry carries side-chain
direction, so it is a better statement of which residues touch. ALPS uses the same
graph but reads only its three lowest eigenvalues, which individual edges barely move.

The consequence for the headline is direct. **On these 950 targets the GNN goes from an
exact tie with ALPS (−0.0001) to +0.0297.** Part of what this file recorded as "a
learned model and a hand-designed readout perform the same" was the learned model being
run on the wrong coordinate convention.

### But on the full Cβ set the margin is +0.014, and the gap between those two numbers
### is the most useful result here

Eight runs at seed 0 on the full 1,209-target Cβ set give **GNN − ALPS = +0.0144 ±
0.0130**, p < 0.05 in three of eight — against +0.0107 ± 0.0094 on Cα. Read alone, that
says nothing changed.

Both are true because the two sets are not the same targets. This repository's adapter
drops samples whose active site has fewer than three residues; the newer builder does
not, so the Cβ set carries 241 targets the Cα set never had. Splitting by seed size:

| anchor residues | n | ALPS | GNN | GNN − ALPS | present in the Cα set |
|---|---|---|---|---|---|
| 1 | 81 | 0.710 | 0.682 | **−0.028** | 0% |
| 2 | 160 | 0.622 | 0.557 | **−0.065** | 0% |
| 3–4 | 159 | 0.620 | 0.574 | −0.047 | 100% |
| 5–9 | 288 | 0.583 | 0.646 | **+0.062** | 100% |
| ≥ 10 | 512 | 0.640 | 0.678 | **+0.038** | 97% |

| subset | n | GNN − ALPS |
|---|---|---|
| all | 1,200 | +0.0144 |
| anchor ≥ 3 | 959 | +0.0313 |
| shared with the Cα set | 946 | +0.0309 |

**Whether the GNN beats ALPS is a function of how big the seed is**, and it swings by
0.13 across that range — an order of magnitude more than every effect this file has
spent its length on.

The mechanism follows from the formulation. The GNN receives the active site *only* as
an indicator on the graph and has to propagate from it; with one or two seeded nodes
there is almost nothing to propagate. ALPS stiffens a neighbourhood and reads the
spectrum, which degrades gently as the seed shrinks. So the learned model's
disadvantage is concentrated exactly where the seed is smallest, and its advantage
appears once the seed is large enough to carry signal.

One limit on that reading: `anchor ≥ 3` and `shared with the Cα set` are nearly the same
partition (+0.0313 against +0.0309), so seed size cannot be fully separated from
"whichever targets the older adapter kept". What separates them is the variation
*within* the shared population — the 3–4, 5–9 and ≥ 10 bands are all ~100% shared and
still run from −0.047 to +0.062.

### Three runs said +0.023 and eight said +0.014

The first three Cβ runs averaged +0.0229 and all three sat above the eventual eight-run
mean of +0.0144. An earlier version of this section was drafted on those three.

That is the **third** time in this work that a small-sample estimate came in high and
regressed: two seeds read as a seed effect when a fixed seed spans as much; the ensemble
gain measured against whichever run happened to be dump 0; and now this. The pattern is
consistent enough to be a rule rather than three anecdotes — **on this task, the first
few draws of any noisy quantity have come in on the flattering side.** Nothing here
should be reported from three runs again.

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

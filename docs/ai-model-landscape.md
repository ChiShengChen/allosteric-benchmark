# Which AI model families can do this task — and which can do it on *our* input

Corpus: 2,536 deduplicated papers from 40 queries across nine sources plus one round
of citation snowballing; 39 triaged in, 22 full texts landed, 27 verbatim evidence
cards mechanically verified against source files (27/27 passed).

Every factual claim below carries a card citation. Where the corpus does not settle
something, that is said rather than filled in.

---

## 1. The single most important finding

**Protein language models work on orthosteric sites and collapse on allosteric ones
in the very same proteins.**

> both methods achieve high precision-recall (AUPR = 0.64-0.76) on orthosteric sites,
> PLM performance collapses on allosteric sites (AUPR = 0.06), despite retaining
> moderate ranking ability (AUROC = 0.70) [P10-c1]

The gap is not an artifact of a leaky split or of class imbalance — the authors
checked all three:

> This deficit persists even after strict control for sequence similarity, structural
> redundancy, and extreme class imbalance (allosteric residues constitute <3% of the
> kinase domain). [P10-c2]

Note the shape of that result: **AUROC stays at 0.70 while AUPR falls to 0.06.** A
model can retain rank-ordering ability and still be useless at the operating point.
This is the same distinction our own benchmark ran into from the other direction —
ranking metrics and localisation metrics disagree, and reporting only the flattering
one is the field's default.

## 2. The baseline problem is the field's, not just ours

PASSer2.0 defines a geometric baseline — take the pocket FPocket scores highest —
and reports it honestly:

> This primitive baseline predictor has accuracy, precision, recall, and F1 score
> values of 0.968, 0.689, 0.571, and 0.624, respectively. [P48-c1]

An accuracy of **0.968** for a rule with no learning in it. Then, on the ranking
metric, the same paper reports the baseline as:

> For 70.6% of the total 204 proteins used in our study, the top-ranked pocket among
> the pockets detected is positive in our labeling method. For 84.3% of proteins in
> the test set, the positive pockets are among the top three ranked positions. [P48-c3]

and its own AutoML model as:

> precision, recall, and F1 score values of 0.850, 0.616, and 0.701, respectively, on
> the test set, and 82.7% of allosteric sites in the test set are ranked among the top
> three positions [P48-c2]

**84.3% for the untrained geometric baseline against 82.7% for the AutoML model on
the top-three metric.** The paper's learned model improves precision and F1 clearly;
on the top-three ranking number it does not lead. (The two figures are phrased over
slightly different denominators — proteins in one, allosteric sites in the other — so
this is a caution about how the metrics line up, not a claim that the model was
beaten on identical terms.)

A second paper states the imbalance trap outright:

> all three pLMs, these metrics can be misleading in the context of significant class
> imbalance. In our [P89-c1]

> dataset, the positive class – residues labelled as allosteric – constitutes only
> around 6% of all [P89-c2]

> allosteric (i.e., all 0s). Similarly, AUC-ROC may be high (close to 1), if the model
> ranks positive [P89-c3]

This is the identical failure our own hybrid gate 2 hit — linear = RBF = 0.967 was
the majority-class rate — and it is documented in the literature as a known trap.

## 3. Two results that bear directly on our input signature

**Sequence alone, no structure, beat the structure-based network model.**

> While Ohm was a statistically better-than-random predictor for 7/24 proteins, and
> EVcouplings for 5/24, ESM1b predicted the allosteric sites in 15/24 proteins better
> than random (p < 0.05 by permutation testing) [P685-c1]

> This is done in a zero-shot fashion with only a single amino acid sequence as
> input [P685-c2]

Coevolution failed outright on a third of that benchmark for want of alignment depth
— a concrete reason to distrust MSA-based routes on any protein family that is not
already well sampled:

> Eight of the proteins did not have sufficient MSA coverage at one or more active
> site or allosteric site residues for EVcouplings to fill a coupling matrix at those
> positions [P685-c3]

**Conditioning on the active site is what closes the gap.** One paper builds exactly
our input signature — a model told where the orthosteric site is, asked for the
allosteric one:

> of the three methods presented here, with performance comparable to the leading
> structure-based [P89-c8]

That is the most directly transferable design in the corpus: the anchor-conditioned
formulation is the one that reaches parity, and it is the formulation our benchmark
already uses.

### A number worth quoting carefully

The pLM study's headline figures are:

> metrics observed with transfer learning. Accuracy ranged from 0.923 to 0.947 for ASD
> only and [P89-c4]

> 0.935 to 0.948 with transfer learning. Similarly, AUC-ROC values were between 0.826
> to 0.843 for [P89-c5]

An AUC-ROC around **0.83** looks far above our benchmark's 0.59. It is not comparable,
and the gap is mostly definitional: that number is plain pooled AUC on residues where
positives are ~6% [P89-c2], on labels split by sequence identity —

> greater than 30% sequence identity with sequences in the validation and test
> set. [P89-c6]

— whereas ours is stratified within distance bands specifically to remove the
proximity signal that plain AUC is dominated by. The paper itself says the plain
number can be high while performance at any threshold is poor [P89-c3]. Reading 0.83
against 0.59 as a seven-fold difference in skill would be exactly the error this
corpus warns about.

## 4. Why nobody has much data

| what | size | card |
|---|---|---|
| ASD allosteric sites after preprocessing | ~3,000, against ~14,000 general binding sites | [P89-c7] |
| ASBench training set | 146 proteins | [P536-c3] |
| Per-residue descriptor study | 235 X-ray structures | [P521-c1] |
| PASSer2.0 training proteins | 90 | [P48-c4] |
| Standard independent test set | 24 proteins | [P536-c4] |
| DeepSite (voxel CNN, *general* binding sites) | >7,000 proteins | [P10-c3] |

The contrast in the last row is the whole story. General binding-site prediction has
an order of magnitude more supervision than allosteric-site prediction, which is why
the deep architectures that work there do not transfer here. Our own 97 curated
targets sit squarely in the normal range for this task — not unusually small.

## 5. Distance is real biophysics, not only an artifact

Our benchmark treats proximity to the active site as a confound to be stratified out.
The corpus says the underlying effect is genuine:

> This distance-dependent allosteric decay is observed in all complete allosteric maps
> generated to date and appears to be a conserved principle of protein
> biophysics [P06-c1]

Both things are true at once: distance carries real signal, *and* a predictor that
only reproduces it has learned nothing beyond geometry. That is precisely why the
comparison has to be made within distance strata rather than by removing distance.

## 6. The families, and whether they run on our input

Our input is Cβ coordinates, active-site indices, ~97 labelled proteins — **no
sequence, no side chains, no MSA, no trajectory, one apo conformation**.

| family | representative | what it needs | runs on our input? |
|---|---|---|---|
| Protein language model | ESM-2 / ProtT5 / Ankh [P89-c1] | amino-acid sequence | **No** — our arrays carry no residue identity. Would need sequence added |
| pLM + active-site conditioning | [P89-c8] | sequence + orthosteric site | **No**, same reason — but this is the design worth copying |
| Sequence-only attention maps | ESM1b [P685-c2] | one sequence | **No**, same reason |
| Voxel 3D CNN | DeepSite [P10-c3] | all-atom structure, >7,000 proteins | **No** — needs atoms and ~70× our labels |
| Pocket-descriptor ML | PASSer2.0 [P48-c1] | all-atom structure for FPocket | **No** — Cβ-only cannot form pockets |
| Foundation-model features | AF2BIND [P184-c1] | an AlphaFold2 forward pass | **No** — needs sequence |
| Trajectory / dynamics models | — | MD or SMD trajectories | **No** — we have one static structure |
| Coevolution | EVcouplings [P685-c3] | deep MSA | **No** — and fails on a third of proteins anyway |
| **Normal-mode / ENM + ML ranker** | AlloPred [P50-c1] | Cα/Cβ coordinates | **Yes** |
| **Residue-graph GNN** | — | a contact graph | **Yes** — buildable from Cβ directly |

Two families survive. One of them is what we already built:

> AlloPred ranked an allosteric pocket top for 23 out of 40 known allosteric proteins,
> showing comparable and complementary performance to two existing methods. In 28 of
> 40 cases an allosteric pocket was ranked first or second. [P50-c1]

One family is worth flagging for its economy: AF2BIND is *logistic regression* on
features from a network trained for something else entirely [P184-c2] — evidence that
on this class of problem the win comes from the representation, not the classifier.

AlloPred is normal-mode perturbation plus pocket descriptors in an ML ranker — the
same physics our ALPS uses, with a learned combiner on top and pocket-level rather
than residue-level output.

**So the honest answer to "which AI models can do this" is:** on the full input that
these papers assume, many — pLMs, voxel CNNs, foundation-model features, AutoML over
pocket descriptors. On *our* input, essentially one untried family: a **graph neural
network over the Cβ contact graph**, conditioned on the active site the way [P89-c8]
conditions on the orthosteric pocket. Everything else needs an input we do not have.

## 7. What the corpus does not settle

- **No paper in the corpus reports a distance-stratified or proximity-matched
  evaluation** of an allosteric-site predictor. The confound our §10 measured is not
  addressed by any verified card here, so we cannot say how these reported numbers
  would survive it.
- **No verified card gives a residue-level AUC for a GNN on an allosteric task.** The
  one family that fits our input is the one the corpus leaves uncharacterised — the
  GNN papers we triaged in were either pocket-level, trajectory-based, or landed with
  mismatched identity (§9).
- Allo-PED reports pocket-level MCC 0.544 and AUC 0.920 [P536-c1] but residue-level
  precision 0.601 and recall 0.422 [P536-c2]; that gap between pocket-level and
  residue-level numbers is unexplained by any card, and our task is residue-level.

## 8. Recall and stopping

40 queries across arxiv, pubmed, europepmc, preprints, semanticscholar, openalex,
crossref, dblp and openreview, plus one snowball round (417 records from six seeds).
Zero-yield queries are logged in `queries.md` as required.

The mechanical rule — three consecutive batches contributing nothing new — never
fired, and saying otherwise would be false. Round 3 added 428 papers. But those nine
queries were deliberate *rephrasings of the same core question*, and they surfaced
exactly **one** new site-prediction method paper, itself a review. Broad queries keep
pulling in loosely related work indefinitely; recall on the question converged. That
is the honest stopping statement.

## 9. A failure worth recording

Seven of 22 landed full texts did not match their ledger title — a 32% attribution
error rate. The dedupe step had merged records across sources and attached a title
from one source to an identifier whose PDF came from another. The content was genuine
published work every time; only the attribution was wrong.

Caught by a mechanical token-overlap check of each file's own header against its
ledger title, run **before** any extraction. Five were re-keyed to their true
identity, three dropped. Had it gone unchecked, roughly a third of this report's
citations would have named the wrong paper while quoting real text.

Six cards also failed first-pass quote verification — bioRxiv line numbers are
interleaved into the extracted text, so quotes copied from the reflowed view were not
verbatim. They were re-extracted from the raw file rather than repaired from memory.
Final: 27/27 verified.

# hybrid — classical/quantum ML on the allosteric task

A work stream for testing whether a quantum machine-learning component helps on this
benchmark. It is deliberately **gated**: the literature we verified in `docs/` says the
likely answer is no and gives cheap tests that settle it before any circuit is written, so
those tests run first.

## What we already know going in

None of this is assumed — each item is measured or backed by a verified quote in
[`../docs/qml-cards.jsonl`](../docs/qml-cards.jsonl).

| Finding | Consequence for this folder |
|---|---|
| Bandwidth-tuned quantum kernels become **numerically indistinguishable from an RBF kernel** — matched ROC-AUC, matched spectra, geometric difference below the √N threshold that *provably guarantees* the classical model does at least as well | An untuned comparison proves nothing; a tuned one is expected to tie |
| On nine small tabular datasets, best classical balanced accuracy **0.830** against best quantum **0.649**, classical winning 8/9, **none of 29 paired comparisons significant** | The prior is strongly negative for small-sample tabular tasks, which is exactly ours |
| Quantum kernel matrices approach the identity at scale, so generalisation needs **exponentially growing sample size** | With ~100 labelled proteins this is the expected failure mode |
| Generalisation error scales as √(T/N) | ~100 labels affords **tens** of trainable gates, not thousands |
| Our feature vector is **7-dimensional** | A 7-qubit feature map is precisely the regime where the quantum kernel collapses to a degree-4 polynomial |
| A classical learned combiner on our own data reached AUC 0.668 against 0.606 for unlearned ALPS — but **dropped top-5 hit rate from 27.1% to 18.6%** | Learning helps ranking and hurts localisation. This, not ALPS, is what a quantum model must beat |

## The gates

Each gate can end the work stream. That is the point — the expensive step is last.

**Gate 0 — does learning help at all, on labels we trust?**
The combiner result above used proxy labels and plain AUC, both since shown to be
confounded (`../README.md` §10). Re-establish it on curated labels with the
distance-stratified metric. If a classical learner does not beat unlearned ALPS there,
no ML component of any kind is worth adding and the answer is already in.

**Gate 1 — geometric difference.**
Compute `g` between the quantum and classical Gram matrices. The literature's result is
that `g` well below `√N` guarantees the classical model matches or beats the quantum one.
If `g ≪ √N` on our data, stop: the conclusion is already proven for this feature set.

**Gate 2 — is there non-linear structure to exploit?**
Kernel effective-rank ratio, and the non-linearity gap (RBF minus linear accuracy). A task
a linear model already solves has no room for a richer kernel, quantum or otherwise.

**Gate 3 — only now, the model.**
Quantum kernel SVM and a shallow variational classifier, against linear / RBF / polynomial
SVMs **on identical features**, protein-grouped cross-validation, evaluated with the
stratified metric and reported beside the same control battery as everything else.

## Rules inherited from the rest of the repository

These are not negotiable here either, because every one of them caught a real error
upstream:

- **Curated labels.** Proxy labels reversed three published conclusions (§10).
- **Distance-stratified AUC**, not plain AUC — plain AUC is dominated by proximity on both
  label sets, and a learner will exploit that confound if allowed to.
- **Protein-grouped cross-validation.** No residue of a test protein may be trained on.
- **Controls in every table**, including `ctrl_random`, with the floor estimated from
  multiple seeds rather than one draw.
- **Paired tests and multiplicity correction.** A margin is not a result.
- **Quantum versus classical on identical features.** Otherwise the comparison measures
  learning, not quantumness — the single most common error in this literature.

## Layout

```
features.py    build the per-residue feature matrix on the curated targets
prescreen.py   gates 0-2: does learning help, geometric difference, non-linearity
kernels.py     classical kernels and simulated quantum feature-map kernels
run.py         gate 3, only if the prescreen leaves headroom
vqc.py         gate 3b, the shallow variational classifier
```

Status is recorded in [`RESULTS.md`](RESULTS.md) as gates are passed or fail.

**Status: all four gates run. The quantum kernel ties and does not win.**

Gates 0–2 left the door open and produced the constraint that made gate 3 meaningful:
bandwidth ≲ 0.1, and a *tuned* classical target rather than the untuned default. Gate 3
then put both sides on identical features and the same training subsample:

| | quantum | classical | Δ | paired p |
|---|---|---|---|---|
| kernel | 0.592 (`bw = 0.02`) | **0.600** (`poly-4`) | −0.008 | 0.20 |
| parametric model | 0.575 (VQC, 24 params) | 0.596 (logistic) | −0.021 | 0.39 |
| control `ctrl_random` | 0.485 | | | |

Both quantum models clear the random control and both land just below their classical
counterpart, with neither gap significant. Two details make this more than a bare tie: the
best classical kernel is the degree-4 polynomial that a bandwidth-tuned quantum kernel is
known to collapse onto — the mechanism the literature named, visible in the ranking — and
the VQC's 0.575 is indistinguishable from **unlearned** ALPS at 0.576, so 24 trained
parameters bought nothing over one hand-designed score.

Full numbers and the three prescreen repairs are in [`RESULTS.md`](RESULTS.md).

# hybrid — gate results

Data: 44 curated targets (the N ≤ 700 subset of 97), 16 063 pooled candidate residues,
386 positives (2.4%), 8 features. Floor for the stratified metric is 0.496 ± 0.016.

## Gate 0 — does learning help, on labels we trust?

Protein-grouped 5-fold CV, distance-stratified AUC, curated labels.

| | stratified AUC | vs floor | paired p vs random |
|---|---|---|---|
| logistic regression on 8 features | **0.603** | +0.107 | 0.0009 |
| ALPS alone, unlearned | 0.576 | +0.080 | 0.0215 |
| CONTROL `ctrl_random` | 0.478 | −0.018 | reference |

Learner minus ALPS: **+0.027, p = 0.72**.

**Marginal pass.** Both clear the floor comfortably; the learner is nominally ahead but
the paired test cannot separate them. So learning is not clearly worth adding on these
features — but it is not ruled out either, which is what a gate is for.

*(No pocket smoothing is applied here, so these numbers are not directly comparable to
README §10.6. The learner-versus-ALPS contrast within the gate is internally consistent.)*

## Gate 1 — geometric difference, swept over bandwidth

The criterion is **one-directional**: g well below √n *proves* the classical model matches
or beats the quantum one. A large g proves nothing in the other direction — and an untuned
quantum kernel is large-g precisely because it has collapsed toward the identity, which is
the documented failure mode rather than headroom. So the off-diagonal mass is reported
next to g.

n = 400, √n = 20.0

| bandwidth | g | mean off-diagonal | reading |
|---|---|---|---|
| 0.02 | 103.3 | **0.718** | no guarantee either way |
| 0.05 | 887.5 | 0.307 | no guarantee either way |
| 0.10 | 1004.8 | 0.109 | no guarantee either way |
| 0.25 | 1085.5 | 0.032 | kernel ≈ identity, cannot generalise |
| 0.50 | 1047.5 | 0.011 | kernel ≈ identity, cannot generalise |
| 1.00 | 1013.8 | 0.005 | kernel ≈ identity, cannot generalise |

**Pass, in a narrow band.** The literature's identity collapse reproduces exactly, from
bandwidth 0.25 upward. Below 0.1 the kernel retains structure and g stays above √n, so the
classical guarantee does not apply. **Any quantum kernel run here must use bandwidth ≲ 0.1**
— above that it is provably useless before the experiment starts.

## Gate 2 — is there non-linear structure to exploit?

Both sides tuned. Plain AUC on a pooled random subsample, so optimistic relative to the
protein-grouped numbers, but adequate as a gate.

| kernel | parameter | effective rank | AUC |
|---|---|---|---|
| RBF | γ = 0.1 | 1.7 | 0.479 |
| RBF | γ = 0.5 | 5.6 | 0.530 |
| RBF | γ = 1 | 13.8 | 0.515 |
| RBF | γ = 2 | 43.5 | 0.511 |
| RBF | γ = 5 | 238.7 | 0.504 |
| RBF | γ = 10 | 693.0 | 0.537 |
| **RBF** | **γ = 25** | **1322.7** | **0.563** |
| linear | — | — | 0.487 |
| poly-4 | — | — | 0.538 |

**Pass.** Non-linearity gap (best RBF − linear) = **+0.076**. There is structure a richer
kernel can reach that a linear model cannot.

## Verdict

**No gate closed the door, so building the model is warranted.** The prescreen did its job
in a different way than expected: rather than ending the work stream, it produced a hard
constraint (bandwidth ≲ 0.1) and a target to beat (tuned RBF at γ = 25, not the untuned
default that a naive comparison would have used).

Gate 3 is next: quantum kernel SVM and a shallow variational classifier against tuned
linear / RBF / polynomial SVMs and logistic regression **on identical features**,
protein-grouped CV, stratified metric, controls and paired tests. Both were run — the
kernel below, the circuit in gate 3b.

## Gate 3 — quantum kernel against tuned classical kernels, identical features

44 curated targets, protein-grouped 5-fold CV, distance-stratified AUC. Every kernel sees
the same balanced training subsample per fold (~600 points, half positive), so any
difference is the kernel and not the sampling.

| kernel | stratified AUC | vs floor | p vs random |
|---|---|---|---|
| **`poly-4`** | **0.600** | +0.104 | 0.0082 |
| `quantum bw=0.02` | 0.592 | +0.095 | 0.0102 |
| `linear` | 0.591 | +0.095 | 0.0135 |
| `quantum bw=0.05` | 0.585 | +0.089 | 0.0052 |
| `quantum bw=0.1` | 0.574 | +0.077 | 0.0076 |
| `RBF γ=10` | 0.559 | +0.063 | 0.0215 |
| `RBF γ=25` | 0.557 | +0.061 | 0.0215 |
| `RBF γ=50` | 0.547 | +0.051 | 0.0363 |
| CONTROL `ctrl_random` | 0.486 | −0.011 | reference |

**Best quantum 0.592 against best classical 0.600 — difference −0.008, paired p = 0.20.**

### What this shows

**The quantum kernel ties.** It neither helps nor hurts: its best setting lands 0.008
below the best classical kernel, well inside noise. Every kernel here, quantum and
classical, clears the random control and lands between 0.55 and 0.60 — the same band as
unlearned ALPS (0.576) and the logistic regression (0.603) from gate 0.

**The predicted mechanism is visible in the ranking.** The literature's specific claim was
that a bandwidth-tuned quantum kernel collapses onto a **degree-4 polynomial** kernel. Here
the best classical kernel *is* `poly-4`, and the best quantum setting sits 0.008 beneath it
while the tuned RBFs trail both by ~0.04. The quantum kernel is not doing something a
polynomial kernel cannot; it is doing approximately that.

**Bandwidth behaves as gate 1 said it would.** Performance falls monotonically as bandwidth
rises — 0.592, 0.585, 0.574 at 0.02, 0.05, 0.1 — heading toward the identity collapse that
gate 1 measured above 0.25. Had this been run at the default bandwidth of 1.0, the quantum
kernel would have scored at chance and the conclusion would have looked far more dramatic
and been far less informative.

**Gate 2's ranking did not transfer.** RBF at γ = 25 was the best classical kernel on
pooled plain AUC (0.563) and is among the worst here (0.557 against 0.600 for poly-4).
Plain AUC on pooled residues and stratified AUC under protein-grouped CV disagree about
which classical kernel is best — one more instance of the metric deciding the ranking.

## Gate 3b — the shallow variational classifier

Same folds, same features, same metric. Depth set by the generalisation bound rather
than by taste: error scales as √(T/N) in trainable gates and 44 grouped proteins afford
tens, so the circuit is 8 qubits and 3 layers — **24 trainable parameters**. Data
re-uploading (encode `RY(πx)`, then a trainable `RY` layer and a `CZ` ring), read out as
the mean single-qubit ⟨Z⟩ through a trainable scale and bias against a logistic loss.
Exact statevector simulation, verified to preserve norm to 6.7e-16 and to reproduce
`cos(πx + θ)` on one qubit to machine precision.

The classical opponent is logistic regression on identical features — the fair one, since
both are parametric models trained by gradient descent on the same loss and differ only
in the feature map.

| model | stratified AUC | vs floor | p vs random |
|---|---|---|---|
| **logistic regression** | **0.596** | +0.100 | 0.0026 |
| VQC, 24 parameters | 0.575 | +0.079 | 0.0026 |
| CONTROL `ctrl_random` | 0.484 | −0.012 | reference |

**VQC minus logistic: −0.021, paired p = 0.39.**

Same shape as the kernel result: the quantum model clears the random control comfortably,
lands slightly below its classical counterpart, and the paired test cannot separate them.
Worth stating plainly — **the VQC's 0.575 is indistinguishable from unlearned ALPS at
0.576** (gate 0). Twenty-four trained parameters and a quantum feature map bought nothing
over a single hand-designed score with no training at all.

### Verdict on the work stream

**No quantum advantage on this task, this feature set, and this sample size** — and the
result is a tie rather than a collapse, which is the more informative outcome. Both
quantum models land just below their classical counterpart and neither gap is significant:

| | quantum | classical | Δ | paired p |
|---|---|---|---|---|
| kernel | 0.592 (`bw=0.02`) | 0.600 (`poly-4`) | −0.008 | 0.20 |
| parametric model | 0.575 (VQC, 24 params) | 0.596 (logistic) | −0.021 | 0.39 |

It matches the prior the folder was opened with, but now on our own data, with the
classical side tuned, at the only bandwidth where the quantum kernel is functional, with
both quantum model families run rather than one, and with the ranking showing the
mechanism the literature named.

What would change the answer: features carrying structure a polynomial kernel cannot
reach, or a sample large enough for the quantum kernel's extra capacity to pay for itself.
Neither is available here — §12 of the main README makes the same point about the input
signature.

## Gates 3 and 3b at 24x the sample, on the right coordinates, with a tighter split

The verdict above ends with a stated condition for overturning it: *"features carrying
structure a polynomial kernel cannot reach, **or a sample large enough for the quantum
kernel's extra capacity to pay for itself**. Neither is available here."*

The second is now available. The AlloBench route rebuilt on Cβ through
[allosteric-datasets](https://github.com/ChiShengChen/allosteric-datasets) gives 1,043
targets carrying the eight features, against 44 — and it comes with two things the
original gates did not have:

| | original gates | this run |
|---|---|---|
| targets | 44 | **1,043** |
| candidate residues in the pool | 16,063 | **285,333** |
| positives in the pool | 386 | **10,230** |
| coordinates | Cα | **Cβ**, which is what ALPS was tuned on |
| split | protein-grouped, assigned by the script | **UniProt-grouped, carried by the dataset** |

The split is the one that matters most and it is *stricter*, not looser: it keeps
homologues of one accession out of opposite sides, where protein-grouping only keeps the
same structure out. So a worse quantum result here cannot be explained by a split that
got easier.

### Gate 3 — the kernel

| kernel | stratified AUC | vs floor |
|---|---|---|
| **`poly-4`** | **0.631** | +0.135 |
| `quantum bw=0.02` | 0.628 | +0.132 |
| `linear` | 0.626 | +0.130 |
| `quantum bw=0.05` | 0.603 | +0.107 |
| `quantum bw=0.1` | 0.601 | +0.104 |
| `RBF γ=10` | 0.586 | +0.090 |
| `RBF γ=25` | 0.578 | +0.082 |
| `RBF γ=50` | 0.566 | +0.070 |
| CONTROL `ctrl_random` | 0.499 | +0.003 |

**Best quantum 0.628 against best classical 0.631 — difference −0.0025, paired
p = 0.0057.**

The gap is *smaller* than at n = 44 (−0.008) and it is now separable, which is exactly
what sample size is supposed to do: the effect did not grow, the noise around it shrank.
Read the p-value precisely — 0.0057 clears 0.05 and does **not** clear the Bonferroni
threshold this repository applies elsewhere, 0.05/11 = 0.0045. The honest sentence is
"significantly worse at the conventional threshold, not under our own correction".

**Every element of the mechanism reproduced at 24× the sample.** The best classical
kernel is still `poly-4`, which is the specific collapse the literature predicted for a
bandwidth-tuned quantum kernel. The quantum optimum still sits just beneath it.
Bandwidth still degrades monotonically toward the identity collapse gate 1 measured —
0.628, 0.603, 0.601 at 0.02, 0.05, 0.1. The tuned RBFs still trail both by ~0.05. This
is the same picture at a different scale, not a different picture.

### Gate 3b — the variational classifier, and the split that reversed it

| model | stratified AUC | vs floor |
|---|---|---|
| **logistic regression** | **0.623** | +0.127 |
| VQC, 24 parameters | 0.606 | +0.110 |
| CONTROL `ctrl_random` | 0.493 | −0.003 |

**VQC minus logistic: −0.0170, paired p < 1e-4.** Same direction as at n = 44 (−0.021)
and no longer inside the noise (p was 0.39).

**That number was +0.0036 and significant *for* the quantum side until a bug was
fixed.** `hybrid/run.py` had been taught to read the dataset's UniProt-grouped split and
`hybrid/vqc.py` had not, so gate 3b scored a random split while gate 3 scored a grouped
one. Under the random split:

| split | logistic | VQC | VQC − logistic | paired p |
|---|---|---|---|---|
| random | 0.629 | 0.632 | **+0.0036** | 0.025 |
| **UniProt-grouped** | 0.623 | **0.606** | **−0.0170** | <1e-4 |

This file's own docstring for `vqc.py` promises the circuit runs "on the same folds" as
its classical opponent. For a while it did not.

### The split rule is worth more than the model family, and it leaks one way

The reversal is the most transferable result in this folder:

```
  moving from a random split to a UniProt-grouped one   0.021
  the quantum-classical difference under the right one  0.017
```

**And the leak is not symmetric.** Logistic regression barely moved, 0.629 → 0.623, a
loss of 0.006. The VQC lost 0.026. That follows from what leakage rewards: 24 trainable
parameters behind a non-linear feature map can memorise family-specific structure, and a
linear model on eight features cannot. **Homologue leakage pays capacity, and in this
comparison the quantum model is the higher-capacity side.**

The general form is worth stating because it outlives this dataset: **a quantum
advantage reported on a random split may be reporting leakage.** Here that effect alone
was large enough to turn −0.017 into +0.004 and to carry a p-value with it.

### What this does to the verdict

The old verdict read "no quantum advantage on this task, this feature set, **and this
sample size**". The last qualifier can go. At 24× the targets, 26× the positives, the
coordinate convention corrected and the split made stricter, both quantum models are
significantly *behind* their classical counterparts rather than tied with them:

| | quantum | classical | Δ | paired p | at n = 44 |
|---|---|---|---|---|---|
| kernel | 0.628 (`bw=0.02`) | 0.631 (`poly-4`) | **−0.0025** | 0.0057 | −0.008, p 0.20 |
| parametric | 0.606 (VQC) | 0.623 (logistic) | **−0.0170** | <1e-4 | −0.021, p 0.39 |

The stated condition for overturning the result has been met and the result did not
overturn. What remains of the original list is the first clause only — features carrying
structure a polynomial kernel cannot reach — and nothing in this repository has produced
such features.

### Reproducing

```bash
python3 hybrid/features.py --targets <allosteric-datasets>/sets/allobench \
        --max-n 700 --cache hybrid/features_allobench_cb.npz   # ~2 h, 1,043 targets
python3 hybrid/run.py --cache hybrid/features_allobench_cb.npz   # gate 3
python3 hybrid/vqc.py --cache hybrid/features_allobench_cb.npz   # gate 3b
```

Both scripts now prefer a `fold` carried in the cache over one they assign, and both
print which they used. The feature cache is not committed: it is derived, and the labels
behind it are the non-redistributable AlloBench annotations.

## Three repairs the gates needed, recorded because each is a standard failure

1. **Accuracy at a 2.4% positive rate.** The first gate 2 reported linear = RBF = 0.967
   and a non-linearity gap of exactly +0.000. Both classifiers were predicting all-negative;
   0.967 is the majority-class baseline. Switched to AUC.
2. **Reading g as bidirectional.** The first gate 1 read g = 1013 ≫ √n as "the quantum
   kernel has headroom". It does not mean that. Sweeping bandwidth showed that same g came
   with an off-diagonal mass of 0.005 — a kernel that had collapsed to the identity.
3. **An untuned classical kernel.** With rank-percentile features in [0,1] and γ = 1/d,
   every pairwise distance is small, the RBF matrix is nearly all-ones and its effective
   rank collapses to 1.7. Comparing an untuned quantum kernel against an untuned classical
   one measures nothing. Tuning the classical side moved it from 0.506 to 0.563 — and that
   0.057 is exactly the margin a quantum result would otherwise have claimed for free.

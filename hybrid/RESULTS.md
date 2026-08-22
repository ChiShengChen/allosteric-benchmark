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
linear / RBF / polynomial SVMs **on identical features**, protein-grouped CV, stratified
metric, controls and paired tests.

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

### Verdict on the work stream

**No quantum advantage on this task, this feature set, and this sample size** — and the
result is a tie rather than a collapse, which is the more informative outcome. It matches
the prior the folder was opened with, but now on our own data, with the classical side
tuned, at the only bandwidth where the quantum kernel is functional, and with the ranking
showing the mechanism the literature named.

What would change the answer: features carrying structure a polynomial kernel cannot
reach, or a sample large enough for the quantum kernel's extra capacity to pay for itself.
Neither is available here — §12 of the main README makes the same point about the input
signature.

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

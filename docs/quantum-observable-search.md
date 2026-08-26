# Searching for a quantum observable that could work

Sections 5 and 8 of the README record seven ways of inserting a quantum walk into
allosteric site prediction, all of which lose to a classical spectral readout. The
mechanism identified there was that interference needs eigenvalue degeneracy and
residue contact graphs have almost none (3.6% of low-lying spectral gaps below 1%).

That diagnosis implies a follow-up: **are there quantum observables whose signal does
not come from degeneracy?** This document is a targeted literature search for exactly
that, and its result.

Every claim below is backed by a verbatim quote mechanically re-checked against the
source PDF. 91 evidence cards were extracted and **91 of 91 passed verification**.

---

## Candidate 1 — OTOCs, operator growth, Krylov complexity

An out-of-time-order correlator measures how a local perturbation spreads to a distant
site. That is the allosteric question written in quantum language, so it looked like the
strongest candidate. **It is decisively dead, and the reason is worth stating precisely.**

For a non-interacting (single-particle) hopping model — which is exactly what a residue
contact graph gives — the operator never grows: *"in non-interacting / fermionic systems,
a single-particle operator always re- / mains single-particle"* [P24-c3]. The OTOC then
collapses algebraically onto the propagator: the squared commutator is
**`C(r,t) = 4 g²(r,t)`**, four times the squared single-particle transfer amplitude
[P24-c4]. It also decays as 1/t instead of saturating, the opposite of a scrambling
system [P24-c5], and the tutorial states plainly that such a system *"is not / scrambling
and should not be expected to be generic"* [P24-c6].

**So an OTOC on our Hamiltonian is not a new observable — it is the coherent-transfer
score we already measured and rejected, squared and multiplied by four.** Building it
would re-derive a known negative.

Getting a genuinely different signal requires a real many-body interacting Hamiltonian on
the graph, and that is not affordable: exact OTOC methods reach ~15 qubits; tensor-network
methods reach a few hundred spins but are limited in accessible *time*, not size
[P24-c7]. There is no physical basis for putting an interacting many-body Hamiltonian on
a residue contact graph in the first place.

**And even if there were, the graph is the wrong shape.** Fast scrambling requires
diameter ≲ log N — regular D-dimensional lattices are explicitly excluded because their
diameter grows as N^(1/D) [P478-c4] — plus genuine expansion, boundary(A) ∝ |A|
[P478-c5]. A ~300-residue protein is geometrically closer to a 3D lattice than to an
expander. Two independent bounds put the scrambling time at Ω(log N) for bounded-degree
graphs [P04-c5, P04-c6].

Worth keeping from this line even though the candidate failed: scrambling on a graph is
controlled by the **graph Laplacian spectrum and the Cheeger constant** [P478-c6,
P478-c7], not by degeneracy — so our 3.6% gap statistic was never the relevant input
here. And rapid OTOC growth is not the same thing as fast scrambling; a circuit with
infinite Lyapunov exponent can still obey the logarithmic bound [P478-c8].

## Candidate 2 — Lieb-Robinson light cones

Same fate, same reason. The Lieb-Robinson bound constrains
`‖[W(x,t), V(0)]‖` and defines a propagation velocity that depends on interaction range
and graph structure — none of which requires degeneracy. But for a single-particle
hopping model the object it bounds is `|⟨j|e^{−iHt}|i⟩|` itself, computable in closed
form (a Bessel function for nearest-neighbour hopping). Operator dynamics in a solvable
spin chain *reduce exactly to a one-dimensional single-particle quantum walk* [P04-c8].

So Lieb-Robinson on our graph is, once again, the transfer amplitude.

## Candidate 3 — non-Hermitian sensing, exceptional points, quantum Fisher information

These promise divergent sensitivity to a perturbation, which is what an allosteric score
wants. The blocker is structural: **a real symmetric contact graph has none of the
required ingredients.** The mechanisms need non-reciprocal hopping (|H₁₂| ≠ |H₂₁|), or
gain/loss, or pairing terms — exceptional points are absent when the pairing term is zero
and the Hamiltonian is Hermitian. Manufacturing them for a protein graph would mean
inventing physics we cannot justify from Cβ coordinates.

The corpus also disagrees with itself about whether the gain is even real, and the
disagreement is informative. A full open-system noise accounting finds that reciprocal
sensors are bounded regardless of exceptional-point tuning, and that amplification
"must incorporate extra noise"; the non-Hermitian skin effect gives **no advantage at
all** once Fisher information is normalised by photon number. A dissenting paper reports
Heisenberg-limited scaling near an N-th order exceptional point — but from unitary
evolution with no Langevin noise included, which is precisely why it disagrees.

The one mechanism that looked like a free lunch is the one explicitly ruled out.

## Candidate 4 — chiral quantum walks

**The only candidate whose precondition our graphs satisfy abundantly**, and the only one
still open. A chiral walk attaches complex phases to the hoppings, breaking time-reversal
symmetry — which gives the walk *directionality*, something allosteric signalling has and
a real symmetric Laplacian structurally cannot represent.

**The precondition is cycles, not degeneracy.** Phases are physical only through their
gauge-invariant flux around loops; the number of meaningful parameters is exactly the
cycle rank E − N + 1, and on a tree every chiral Hamiltonian is gauge-equivalent to the
plain adjacency matrix — chirality does nothing there. Measured on our benchmark:

| | needed by | measured |
|---|---|---|
| eigenvalue near-degeneracy | the seven failed interference readouts | **3.6%** of low gaps — nearly absent |
| cycle rank E − N + 1 | chiral phases | **7.7–8.3 independent cycles per residue** |
| odd cycles (triangles) | chirality's payoff regime | **8 700–18 300 triangles per protein** |

Odd cycles specifically matter: the optimum phase is π/2 on odd cycles, while on even
cycles the optimum is zero, i.e. non-chiral. Protein contact graphs are dense in
triangles, so the regime the literature says is favourable is the regime we are in.

**Two red flags, both serious:**

1. **A Laplacian no-go that targets our exact Hamiltonian.** For Laplacian-type walks the
   degree diagonal hinders transport between vertices of very different degree, and this
   *cannot be overcome by chirality*. Protein contact graphs are degree-heterogeneous.
   Only the adjacency-type generator responds to chiral phases — and in our benchmark the
   adjacency generator is far weaker than the Laplacian to begin with (16.7% vs 83.3%
   significant on the subset below).
2. **The published gains are on engineered topologies.** Up to 6× transport enhancement
   on chains of triangles; but on a generic chain the advantage is marginal and
   **collapses beyond ~9 sites**. Nothing in the corpus demonstrates a gain on a large
   irregular graph.

Plus a practical problem: tuning E − N + 1 ≈ 2 500–5 200 independent loop phases is not
searchable. A principled reduction exists — minimise the leading eigenvalue modulus of
the Perron-Frobenius operator — and so does a physical one, used below.

### Measured — and it fails too

Rather than tune thousands of phases, we imposed a **Peierls substitution**: a uniform
"magnetic field" threads gauge-invariant flux through every cycle, reducing the parameter
count from thousands to one field vector, and giving a physical ansatz rather than a
fitted one. Implementation in [`methods/chiral.py`](../methods/chiral.py).

Three sanity checks pass first, which is what makes the test meaningful:

| check | result |
|---|---|
| asymmetry vanishes for a time-reversal-symmetric walk | `max|d| = 0.000e+00` at B = 0 — the observable is chiral *by construction* |
| Hamiltonian stays Hermitian and genuinely complex | yes |
| triangle fluxes invariant under a random gauge change | yes, to 1e-9 |

So the score cannot silently collapse into the transfer amplitude that failed before: at
zero field it is exactly zero, not approximately.

**Readout A — directional asymmetry** `p(anchor→i) − p(i→anchor)`, tier-A (n = 11):

| variant | sig | median p | AUC | hit5 |
|---|---|---|---|---|
| **ALPS, real symmetric (reference)** | **90.9%** | **0.0003** | **0.757** | **36.4%** |
| chiral \|asymmetry\|, B = 0.02 | 9.1% | 0.6582 | 0.477 | 9.1% |
| chiral \|asymmetry\|, B = 0.1 | 0.0% | 0.9975 | 0.318 | 9.1% |
| chiral \|asymmetry\|, B = 0.5 | 9.1% | 0.8693 | 0.426 | 9.1% |
| chiral signed asymmetry, B = 0.1 | 27.3% | 0.5088 | 0.541 | 9.1% |

**Readout B — chirality inside the perturbation framework**, i.e. how local stiffening
changes the directional asymmetry, which pairs the chiral observable with the only
framework that has worked here. Tier-A subset, N ≤ 320 (n = 5):

| variant | sig | median p | AUC | hit5 |
|---|---|---|---|---|
| **ALPS (reference)** | **80.0%** | **0.0140** | **0.725** | 40.0% |
| chiral perturbation response, B = 0.05 | 20.0% | 0.2763 | 0.565 | 40.0% |
| chiral perturbation response, B = 0.2 | 40.0% | 0.0954 | 0.547 | 20.0% |

Pairing chirality with the perturbation framework recovers some signal relative to the
raw asymmetry (AUC 0.565 vs 0.318), but stays well below the plain real-symmetric
spectral readout.

**Verdict: chirality is real, measurable and gauge-invariant on these graphs — and
allosterically uninformative.** The precondition held, unlike every previous candidate,
and the observable still carries less signal than the time-symmetric one.

Two caveats we are not entitled to wave away. The uniform-field ansatz produces triangle
fluxes spread across a range rather than concentrated at the π/2 optimum the literature
identifies for odd cycles, so a phase configuration chosen by the Perron-Frobenius
criterion could do better. And readout B is n = 5. Neither caveat changes the direction
of the result, but a determined follow-up has room to work.

---

## The pattern underneath all of it

Eleven candidate insertion points have now been measured across the two searches, and the
failures share one structure:

> A **single-particle Hermitian** walk on a graph is classically simulable and carries no
> information beyond its transfer amplitudes. Every genuinely quantum observable we found
> needs one of two things the problem does not supply: **many-body interactions** (OTOCs,
> scrambling, Krylov complexity) — exponentially expensive and physically unmotivated on
> a contact graph — or **non-Hermitian structure** (exceptional points, skin effect) —
> requiring non-reciprocity or gain/loss that Cβ coordinates cannot justify.

Chiral phases were the one exception on paper: they add genuinely new physics to a
single-particle Hermitian walk, they need only cycles, and our graphs have cycles in
abundance. The precondition held and the observable was still uninformative — so the
exception did not survive contact with the benchmark either.

The residual is small and specific: a phase configuration selected by the
Perron-Frobenius criterion rather than by a uniform field, tested at proper sample size.
We would not bet on it.

---

---

## Candidate 5 — quantum kernels, quantum ML, reservoirs, tensor networks

The previous version of this file listed these three as *not covered*: they appeared in
the corpus but no full texts were landed. A third search closed that gap — 10 queries,
one snowball round, 376 deduplicated papers, 8 full texts, **61 evidence cards, 61
verified**. Cards in [`qml-cards.jsonl`](qml-cards.jsonl).

These differ from every other candidate here in one respect that decides the whole
question: **they require training.** Every method in this repository is unlearned. So the
comparison that matters is not "quantum kernel versus ALPS" — it is **quantum kernel
versus a classical kernel on the same features**. Otherwise any gain measures *learning*,
not *quantum*.

### First: does learning help at all here?

That question bounds the entire branch, and it is answerable on our own data.
[`scripts/learned_combiner.py`](../scripts/learned_combiner.py) feeds the per-residue
scores of seven implemented methods into a classical learner, with **cross-validation
grouped by protein** so no residue of a test protein is ever trained on. 59 held-out
targets:

| model | sig | median p | AUC | hit5 |
|---|---|---|---|---|
| logistic regression on 7 method scores | **55.9%** | **0.0162** | **0.668** | 18.6% |
| gradient boosting | 47.5% | 0.0678 | 0.636 | 20.3% |
| **ALPS alone (unlearned)** | 47.5% | 0.0565 | 0.606 | **27.1%** |

> ⚠️ **These 59 targets are not the full eligible set.** 89 tier-B targets meet the
> criteria (N ≤ 660, labelled, enough background); feature extraction was interrupted
> and the cache covers the alphabetical prefix `10ZG_A` … `3R6W_A`, i.e. **59 of 89**.
> PDB IDs are not random with respect to deposition era, so this is a potential
> selection bias, not a random subsample. The comparison between rows is still like for
> like — every model is scored on the same 59 targets — but the absolute rates should not
> be quoted as the held-out numbers for this benchmark. Rebuild with
> `python3 scripts/learned_combiner.py` (the cache is incremental) and re-read before
> relying on them.

Learning helps on **ranking** — AUC 0.668 versus 0.606, significance 55.9% versus 47.5% —
and **hurts on localisation**, hit5 18.6% versus 27.1%. That is the same split the
distance control shows: good at pushing true sites up on average, bad at putting them in
the top five. So a learned model is worth having, but it is not strictly better, and
whatever a quantum model would have to beat is *this*, not ALPS.

### Then: would a quantum kernel beat the classical one?

The corpus is consistent and it says no.

- The closest analogue to our setting — quantum-kernel SVMs against linear/RBF/polynomial
  SVMs on nine small tabular datasets, some subsampled to n = 60, under nested
  cross-validation — gives best classical balanced accuracy **0.830** against best
  quantum **0.649**, an 18.1-point gap, with classical winning on 8 of 9 datasets and
  **none of 29 paired comparisons significant at α = 0.05**. Two datasets where quantum
  led at 10% training data lost at full data, and the learning-curve slopes are identical
  (0.032 both), so it is not a data-efficiency advantage either.
- There is a mechanism for why. Quantum kernels only generalise once their bandwidth is
  tuned, and optimal bandwidth tuning makes them numerically **indistinguishable from an
  RBF kernel** — matched ROC-AUC, matched spectra, and a geometric difference below the
  √N threshold, which *provably guarantees* the classical model does at least as well. At
  the small bandwidths actually selected they collapse further, to a degree-4 polynomial
  kernel.
- At scale the quantum kernel matrix approaches the identity, so good generalisation
  would demand an **exponentially growing sample size** — precisely the failure mode at
  ~100 labelled proteins. Independently, generalisation error scaling as √(T/N) means
  ~100 labels affords only tens of trainable gates.

### And is "trainable implies classically simulable" a theorem?

No — and the honest answer matters, because the sloppy version of this claim is common.
For the standard hardware-efficient ansatz it does hold in practice: it has barren
plateaus *unless* it is shallow with a local observable, and exactly in that regime it is
classically simulable. But the general statement is **refuted**: one can construct
variational models that are gradient-trainable *and* non-dequantizable. Those
constructions are cryptographic and contrived — nothing that transfers to a contact graph
— but the theorem does not exist, and this file should not pretend it does.

### Verdict, and the cheap test if you want to check anyway

Not worth implementing. Our feature vector is seven-dimensional, which would mean a
seven-qubit feature map — exactly the regime where the literature says a bandwidth-tuned
quantum kernel becomes a polynomial kernel.

If a future reader wants to check rather than take this on trust, the corpus supplies two
pre-screens that are cheap and decisive: the **geometric difference** between the quantum
and classical Gram matrices (if it is well below √N, classical is guaranteed to match or
beat quantum), and the kernel effective-rank ratio together with the non-linearity gap
(RBF minus linear accuracy). Run those before writing any circuit.

**Not covered even now:** quantum reservoir computing appeared in the corpus (papers on
particle statistics, squeezing, NARMA-10 benchmarking, non-Markovian architectures) but
no full text was landed and **no cards were extracted**, so nothing here is a verdict on
it. Tensor-network methods appear only through the dequantization results — as the
*classical* tool that removes a quantum advantage, not as a candidate scorer.

---

## Candidate 6 — phase estimation of the spectral shift

The five candidates above are all attempts to find an observable that *predicts better*,
and all five failed. This one is different in kind: it never claims a better prediction,
only a cheaper route to the same one. That is why README section 6 lists it as the single
framing left standing.

**Why the usual objection does not apply here.** Quantum linear algebra normally dies at
readout: the answer is an N-dimensional amplitude vector, and extracting it costs O(N)
measurements, which is the whole speedup. ALPS is immune to that, because it never
touches an eigenvector. Its score is

    out[i] = sum_{k<=3} |lambda_k(H_i) - lambda_k(H_0)| / lambda_k(H_0)

a sum of scalars. Phase estimation returns exactly that type. The output of the algorithm
and the input of the score match, which is rare enough to be worth stating.

**So the deciding question is precision, and precision is measurable classically.** QPE's
circuit depth scales as O(1/epsilon), so what settles whether this is worth implementing
is how small epsilon has to be. `scripts/precision_budget.py` measures it on 25 curated
targets (N <= 520) by multiplying every eigenvalue -- base and perturbed alike, since an
estimator has to measure both ends of a difference -- by (1 + eta), eta uniform on
[-epsilon, epsilon]. Uniform rather than Gaussian because QPE returns a value inside a
resolution window, not a normal deviate around the truth.

First, the size of the thing being resolved:

| relative shift `|dlambda|/lambda` | median | IQR | 5-95% |
|---|---|---|---|
| pooled over residues and modes | **2.94e-2** | [1.04e-2, 6.50e-2] | [1.60e-3, 1.56e-1] |

kappa = 2 stiffens each contact threefold, so the perturbation is not subtle and neither
is its effect on the spectrum. A ~3% relative shift is a coarse signal, not the near-
cancellation that makes small differences expensive to estimate.

Then the sweep, distance-stratified AUC, 5 noise seeds:

| epsilon | strat AUC | sd over seeds | vs exact |
|---|---|---|---|
| 0 (exact) | 0.612 | — | — |
| 1e-4 | 0.612 | 0.0002 | +0.000 |
| 1e-3 | 0.612 | 0.0008 | −0.000 |
| **1e-2** | **0.607** | 0.0033 | **−0.006** |
| 3e-2 | 0.592 | 0.0080 | −0.020 |
| 1e-1 | 0.552 | 0.0121 | −0.060 |

**One part in 10^3 is free and one part in 10^2 costs 0.006.** That is 10^2-10^3 applications
of a controlled unitary, not 10^6. Whatever eventually rules this route out, it will not
be the precision requirement — which was the objection that looked fatal before it was
measured.

### The classical shortcut that would have removed the object, and why it does not

The cost argument only has force if the classical computation is genuinely expensive.
ALPS diagonalises once per residue, so first-order perturbation theory is the obvious
attack: lambda_k(H_i) - lambda_k(H_0) = v_k^T dL_i v_k needs **one** decomposition of H_0
for the whole protein instead of N+1. kappa = 2 gives it no right to be accurate, but the
score is only ever used as a ranking, and preserving order would be enough.

It preserves order almost perfectly and still costs more than everything this repository
has been chasing:

| first-order perturbation theory | |
|---|---|
| Spearman against the exact raw score | median **0.984**, 100% of targets above 0.9 |
| stratified AUC | **0.580** against 0.612 exact, **−0.032** |

Those two rows are not in conflict, and the gap between them is the result. A rank
correlation computed over ~500 residues can sit at 0.98 while the handful of candidates
at the top of the ranking are reshuffled — and the top is the only part the metric
scores. Section 10 of the main README records the same failure in other clothing: a
flattering aggregate statistic standing in for behaviour in the region that decides the
answer.

0.032 is larger than the GNN's entire margin over ALPS, and three times the gain from
retuning ALPS itself. So the N+1 eigendecompositions are not removable this way, and the
cost the quantum framing proposes to attack is real.

### What this does and does not establish

Measured: the precision requirement, and that the classical work is not trivially
avoidable. **Not measured, and either could still consume the whole budget:**

* **State preparation.** QPE needs an input state overlapping the target eigenvector.
  H_0's eigenvector is a natural warm start and is classically in hand, but loading an
  arbitrary N-dimensional vector onto log2(N) qubits costs O(N) gates in general.
* **Block encoding.** H_i is sparse (10-20 contacts per residue), and sparse Hamiltonian
  simulation is a mature primitive, but building the oracle for an arbitrary protein's
  contact graph costs what writing the matrix down costs, O(N·k).

Both are linear in N, which is exactly the scale at which a speedup over an O(N^3)-ish
classical solve stops being obvious. This section bounds one of three costs. It does not
claim an advantage, and no circuit has been written.

**A note on the complexity figure.** README section 6 describes ALPS as "N
eigendecompositions, O(N^4)". That overstates the current implementation: `_low_eigs`
switches to a sparse shift-invert solve for the three lowest modes above N = 400, so the
real cost is N sparse solves, not N dense ones. The argument survives the correction --
it is still linear in N solves -- but the exponent quoted is not what the code does.

---

## What is still not covered

Quantum reservoir computing: present in the corpus, no full text landed, no cards. Not a
verdict.

## Method

29 queries across arXiv, OpenAlex, Europe PMC and Semantic Scholar; one round of
bidirectional citation snowballing from three seeds; 1 032 deduplicated papers; 15
triaged in, 14 full texts landed; 91 evidence cards extracted by per-paper subagents and
**91/91 verified** by mechanical re-grep against the source text.

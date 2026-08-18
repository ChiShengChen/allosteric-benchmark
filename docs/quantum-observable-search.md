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

## What this search did not cover

Quantum kernels and quantum machine learning on graph data, quantum reservoir computing,
and tensor-network / quantum-inspired classical algorithms appeared in the corpus but no
full texts were landed and **no evidence cards were extracted for them**. Nothing in this
document should be read as a verdict on those three.

## Method

29 queries across arXiv, OpenAlex, Europe PMC and Semantic Scholar; one round of
bidirectional citation snowballing from three seeds; 1 032 deduplicated papers; 15
triaged in, 14 full texts landed; 91 evidence cards extracted by per-paper subagents and
**91/91 verified** by mechanical re-grep against the source text.

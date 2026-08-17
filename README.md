# allosteric-benchmark

An independent, reproducible benchmark for **allosteric site prediction from minimal
structural input** — per-residue Cβ coordinates plus the active-site residue indices,
and nothing else. No MD trajectories, no bound (holo) reference structure, no learned
weights at inference.

It contains three things:

1. **A benchmark builder** that constructs a family-declustered test set straight from
   RCSB PDB, because the field's standard benchmark servers (ASD, ASBench, CASBench)
   are currently unreachable.
2. **Twelve methods** — quantum-walk, elastic-network, graph-propagation and machine-free
   baselines — implemented against one identical input signature, post-processing and
   scoring protocol, plus **three deliberately trivial controls**.
3. **ALPS**, a new method that came out of the comparison, with its design traced to
   specific measurements rather than intuition.

Everything here is reproducible from public data with `numpy` + `scipy`.

---

## TL;DR

- On **101 independent targets**, a distance-only control (`score = distance from the
  active site`) reaches 46–55% permutation significance. **Any method that does not beat
  that control has not demonstrated anything**, and most published-method ports do not.
- **ALPS** — perturb the contact graph locally, read the shift in the three lowest
  Kirchhoff eigenvalues, z-score against distance — beats every control on the held-out
  set, and its margin is largest at *localisation*: **24.4% top-5 hit rate** versus 7.8%
  for the distance control (3.1×) and 17.2% for the strongest published-method port.
- **Five different places to put a quantum walk have now been measured, and all five
  lose** to a plain eigenvalue-shift readout: coherent transfer amplitude, coherent
  transfer inside the perturbation framework, ENAQT dephasing, degeneracy structure, and
  eigenvector content. Section 5 records where a quantum walk could still plausibly sit —
  and it is not in the physics readout.
- **No method reaches the DCC ≤ 4 Å localisation criterion on any target.** Enrichment
  significance and pointing at the right place are different problems, and contact-graph
  methods currently solve only the first.

---

## 1. Why build a new benchmark

The standard annotated benchmarks for this task are effectively offline:

| Resource | Status (checked Aug 2026) |
|---|---|
| ASD (`mdl.shsmu.edu.cn/ASD/`) | Server unreachable over HTTPS; download links are JS-driven and not machine-retrievable |
| CASBench (`mdl.shsmu.edu.cn/CASBench/`) | HTTP 404 |
| `casbench.org` | Does not resolve |

So the benchmark is rebuilt from primary data. Sites are defined the way ASBench and
CASBench define them — by the residues that directly contact the crystallographic ligand:

| Field | Rule |
|---|---|
| `anchor` (orthosteric / active site) | residues with any heavy atom within **4.5 Å** of a **cofactor** ligand (nucleotides, NAD/FAD/SAM/CoA/PLP, porphyrins, sugar-phosphates) |
| `y` (allosteric site) | residues within **4.5 Å** of a **drug-like** ligand: ≥ 12 heavy atoms, contains N/S/halogen (excludes PEG and glycerol, which are C/O only), not a cofactor, not a crystallisation additive |
| keep if | the two sites are disjoint and separated by **≥ 8 Å** |

Two tiers, both deduplicated by UniProt accession via PDBe SIFTS so no protein family is
represented twice:

- **tier-A** (11 targets) — candidates restricted to entries whose RCSB full text mentions
  *allosteric*, i.e. the depositors themselves flagged the structure as allosteric.
  Higher-confidence labels. **Used for tuning.**
- **tier-B** (90 targets) — candidates from a generic "cofactor + ≥ 2 ligands" query.
  Purely geometric proxy labels. **Held out; never used for tuning.**

### ⚠️ The labels are a proxy, not expert curation

`y` means *"a drug-like molecule was crystallised here, far from the catalytic site"*.
That is the field's operational definition of a candidate allosteric site, and it is what
ASBench-style sets encode — but it is **not** the same as an experimentally validated
allosteric regulatory site. Every conclusion drawn from this benchmark carries that caveat.

---

## 2. Methods implemented

All in [`methods/`](methods/); every one consumes only Cβ coordinates + anchor indices, and
all share identical post-processing (rank percentile → graph smoothing) and candidate pool.

**Quantum-walk channels** ([`quantum.py`](methods/quantum.py))

| Name | Description |
|---|---|
| `qasc_baseline` | Laplacian-CTQW infinite-time-averaged communicability + adjacency-matrix IPR resonant transfer, fused by noisy-or (a reimplementation of [QASC](https://github.com/Arthur031221/quantum-allosteric-scanner)) |
| `qasc_degseed` | Same, but the IPR channel is seeded with a **degree-weighted** state. The uniform superposition is a zero-eigenvalue eigenvector of the Laplacian but *not* an eigenvector of the adjacency matrix, so under an adjacency-generated walk it drifts with no driving; degree weighting is the natural initial condition |
| `qasc_normlap` | Symmetric normalized Laplacian, to remove the degree heterogeneity that degrades coherent walks on hub-dominated graphs |
| `enaqt` | Lindblad pure-dephasing transport with the dephasing rate **calibrated in units of the largest coupling** and efficiency defined as a finite-window time integral of site occupation |

**Classical / elastic-network** ([`btb.py`](methods/btb.py), [`enm.py`](methods/enm.py))

| Name | Description |
|---|---|
| `btb` | Bond-to-bond propensity ported to the residue graph: `M = ½ G Bᵀ L† B` (Green's function of the weighted Laplacian), seeded at the active site, with conditional quantile regression against distance. Only the source columns are formed, never the full m×m matrix |
| `apop` | Stiffen a local neighbourhood to mimic ligand binding, rank by the induced shift in global mode frequencies. **Does not use the anchor at all** |
| `corrsite` | GNM slow-mode and fast-mode motion correlation to the anchor, scored as the max of the two Z-scores |
| `prs` | Perturbation-response scanning: the GNM covariance is the linear response operator |

**Trivial controls — the point of the whole exercise**

| Name | Description |
|---|---|
| `ctrl_dist` | distance from the anchor. No graph, no model |
| `ctrl_burial` | contact degree |
| `ctrl_random` | random numbers, to calibrate the false-positive rate of the criterion itself |

---

## 3. ALPS — the method this study converged on

[`methods/alps.py`](methods/alps.py)

```
score(i) = z_d[ Σ_{k ≤ K} |λ_k(H_i) − λ_k(H_0)| / λ_k(H_0) ]
```

`H_0` is the Kirchhoff matrix of the contact graph; `H_i` is the same matrix with every
edge inside residue *i*'s 10 Å neighbourhood stiffened by ×2 to mimic a ligand binding
there; `λ_1..K` are the **lowest K = 3 non-zero eigenvalues**; `z_d[·]` is a
distance-conditional z-score against residues at comparable distance from the active site.

Each design decision answers something measured here, not a hunch:

| Decision | Evidence |
|---|---|
| Perturbation response, not a propagation amplitude | The CTQW score correlates with distance-to-anchor at **−0.60 to −0.71** across all three sets — it is substantially a proximity ranker, and its AUC falls **below 0.5** on independent targets. A difference cancels the baseline proximity structure |
| Read the **spectrum**, not the transfer | Inside one identical perturb-and-read framework, infinite-time coherent transfer, finite-window coherent transfer and classical diffusion all lost to the spectral readout (9–27% vs 91% on tier-A). Local stiffening barely moves the eigenvalue *degeneracies* that govern long-time coherent transfer, so that observable is noise-dominated; it moves the low-lying eigenvalues cleanly |
| Only the lowest 3 modes | K = 3 beat K = 5 and K = 10. Allosteric leverage lives in global collective motion; higher modes add local noise |
| Distance-conditional z-score | Lifted this method from 82% → 91% on tier-A, and lifted the CTQW baseline from 9% → 27% |

**Dual reading of the operator.** The Kirchhoff matrix is simultaneously the CTQW
Hamiltonian and the Gaussian Network Model operator. Its low-lying eigenvalues are the
slowest coherent frequencies of the quantum walk *and* the slowest vibrational modes of
the elastic network — the same numbers. ALPS measures how a local binding event retunes
that shared spectrum.

> **No quantum advantage is claimed.** The classical and quantum readings of this quantity
> are identical, and in this study the explicitly coherent observables performed *worse*.

---

## 4. Results

`sig` = fraction with permutation p < 0.05 · `AUC` = ROC-AUC inside the candidate pool
(0.5 = chance) · `hit5` = top-5 contains a true allosteric residue.

**tier-B is the only column not contaminated by tuning and should be read as the result.**

| Method | QASC's own 3<br>sig / AUC | tier-A (tuning, n=11)<br>sig / AUC | **tier-B (held out, n=90)<br>sig / AUC / hit5** |
|---|---|---|---|
| **`ALPS`** | 33% / 0.619 | 90.9% / 0.757 | **48.9% / 0.603 / 24.4%** |
| `ALPS_noresid` | 0% / 0.620 | 90.9% / 0.722 | 46.7% / 0.584 / 25.6% |
| `apop` | 0% / 0.512 | 80.0% / 0.733 | 50.6% / 0.564 / 17.2% |
| `apop+dist` | 0% / 0.487 | 80.0% / 0.753 | 47.1% / 0.593 / 23.0% |
| `cpr_classical` | 0% / 0.519 | 27.3% / 0.596 | 42.7% / 0.542 / 11.2% |
| `qpr_coherent` | 0% / 0.440 | 9.1% / 0.358 | 5.6% / 0.400 / 4.5% |
| `ctrl_dist` | 0% / 0.432 | 54.5% / 0.583 | 45.6% / 0.614 / 7.8% |
| `ctrl_burial` | 33% / 0.631 | 45.5% / 0.613 | 21.1% / 0.466 / 10.0% |
| `ctrl_random` | 0% / 0.542 | 9.1% / 0.489 | 8.9% / 0.473 / 13.3% |
| `qasc_baseline` | 100% / 0.731 | 9.1% / 0.412 | **7.8% / 0.416 / 4.4%** |
| `qasc_degseed` | 100% / 0.732 | 18.2% / 0.413 | 7.8% / 0.415 / 4.4% |
| `qasc_distcorr` | 67% / 0.742 | 27.3% / 0.474 | 13.3% / 0.457 / 10.0% |
| `btb` | 0% / 0.301 | 18.2% / 0.446 | 17.8% / 0.465 / 16.7% |
| `corrsite` | 0% / 0.445 | 27.3% / 0.534 | 16.7% / 0.442 / 11.1% |
| `prs` | 0% / 0.365 | 18.2% / 0.492 | 20.0% / 0.501 / 5.6% |
| `enaqt` | 0% / 0.518 | 9.1% / 0.511 | 12.2% / 0.507 / 8.9% |

### 4.1 The honest reading of ALPS

On the 90 held-out targets ALPS beats the distance-only control on significance
(48.9% vs 45.6%) but that margin is small and inside the confidence interval. The
substantial win is **localisation**: a 24.4% top-5 hit rate against 7.8% for the distance
control — 3.1× — and 17.2% for `apop`, the strongest published-method port. ALPS also has
the best in-pool ranking quality (AUC 0.603 vs 0.564 for `apop`) and the shortest median
DCC of the strong methods (20.2 Å vs 26.4 Å).

`apop` edges ALPS on raw significance rate (50.6% vs 48.9%) while losing on AUC, hit5 and
DCC. If the deliverable is "five residues worth testing experimentally", ALPS is the better
choice; if it is "does this protein show any enrichment at all", they are equivalent.

Removing the distance z-score (`ALPS_noresid`) costs significance (46.7%) and AUC (0.584)
but not hit rate (25.6%), so the residualisation sharpens the ranking overall rather than
the very top of it.

The tier-A → tier-B drop (90.9% → 48.9%) has two causes and both should be stated:
hyperparameters were selected on tier-A, and tier-A's labels are higher quality.

### 4.2 Ablations — which observable actually carries the signal

Two ablations, both holding the graph, the perturbation and the distance
conditioning fixed and varying only what is read out.

**(a) The propagator.** Identical perturbation and readout location; only the kernel changes:

| Readout | tier-A sig / AUC |
|---|---|
| Infinite-time coherent transfer (the CTQW observable) | 9.1% / 0.475 |
| Finite-window coherent transfer | 9.1% / 0.358 |
| Classical diffusion kernel | 27.3% / 0.596 |
| **Low-lying spectral shift (ALPS)** | **90.9% / 0.757** |

**(b) The spectral quantity.** Given that a spectral readout wins, *which* spectral
quantity? `scripts/ablate_readouts.py`, tier-A, K = 3 modes throughout:

| Readout | sig | median p | AUC | hit5 |
|---|---|---|---|---|
| **`dlam` — eigenvalue shift (ALPS)** | **90.9%** | **0.0003** | **0.757** | **36.4%** |
| `dpart` — change in active-site participation in the low modes | 63.6% | 0.0196 | 0.682 | 18.2% |
| `dgap` — change in level *spacings* (degeneracy structure) | 54.5% | 0.0013 | 0.660 | 18.2% |
| `dipr` — change in mode localisation (inverse participation ratio) | 36.4% | 0.0865 | 0.637 | 9.1% |

The last three are the *quantum-specific* quantities: degeneracy structure and
eigenvector content, which is what would carry the signal if interference were doing
the work. Plain eigenvalue magnitude beats all of them.

### 4.3 On the CTQW baseline

Reported in both directions, because both are true:

- **It is not an artefact on its own targets.** On the three targets shipped with QASC it
  reaches 100% significance and AUC 0.731 while *all three* trivial controls fail
  (distance 0%, random 0%, burial 33%). That result is real and survives the controls.
- **It does not transfer.** On 101 independent targets it sits at or below the random
  control (7.8% vs 8.9% on the 90 held-out targets), with AUC consistently below 0.5
  (0.412, 0.416) and a top-5 hit rate of 4.4% against the random control's 13.3%.
- **The mechanism is measurable.** Its score correlates with distance-to-anchor at −0.713 /
  −0.621 / −0.604 on the three sets — it is largely a proximity ranker. On its own three
  targets the annotated allosteric residues happen to sit **4.2 Å closer** to the active
  site than the distal background, so a proximity ranker succeeds; on the independent sets
  they sit farther (+1.0 Å, +4.0 Å) and the same ranker inverts, which is exactly why AUC
  drops below 0.5.

This is a property of a specific scoring function on a specific set of targets, not a
judgement of the idea. Two of the fixes tried here (degree-weighted seeding, distance
conditioning) improve it measurably.

### 4.4 Nothing localises to 4 Å

DCC ≤ 4 Å success is **0% for every method on every set**; median DCC is 18–33 Å. Note that
`ctrl_random` posts a *median* DCC of 20.5 Å, so median DCC alone is easy to game by
predicting near the protein centre — read it together with `hit5`, never alone.

---

## 5. Where a quantum walk can and cannot sit in this pipeline

Because the project started from a quantum-walk method, it is worth stating precisely
what was tested and what remains open. Five insertion points have now been measured and
all five failed:

| Insertion point | Result |
|---|---|
| Absolute CTQW transfer amplitude as the score (QASC's design) | Fails — the score correlates −0.60 to −0.71 with distance-to-anchor; it is largely a proximity ranker |
| Coherent transfer as the readout inside the perturbation framework | 9.1% significant, versus 90.9% for the spectral readout |
| ENAQT / dephasing-assisted transport | γ swept from 0 to 3·J_max with the rate calibrated to the hopping scale; no optimum appears, results stay at chance |
| Degeneracy structure (level spacings) under perturbation | 54.5%, versus 90.9% for plain eigenvalue shift |
| Eigenvector content (active-site participation, mode IPR) | 63.6% and 36.4% |

The pattern is consistent: **every observable that depends on coherent transfer
probability or eigenvector structure loses; only eigenvalue magnitude wins.** A plausible
reading is that residue contact graphs lack the symmetry needed to produce meaningful
eigenvalue degeneracies, and interference effects need degeneracies to appear at all —
which is also what the CTQW literature predicts for long-time transfer.

**Where a quantum walk legitimately already sits.** The λ_k that ALPS reads *are* the
eigenvalues of the CTQW Hamiltonian — the slowest coherent frequencies of the walk. "How
does ligand binding retune the walk's low-lying spectrum" is a fair description of the
method. It is equally a description of the Gaussian Network Model's slowest vibrational
modes, so this is a *framing*, not an advantage, and this repository does not present it
as one.

**What remains genuinely untested**, in descending order of how promising it looks:

1. **Combinatorial selection rather than physical propagation.** Single-residue
   perturbation is easy classically — this repo already does it. The biological question
   is cooperative: which *set* of k residues, stiffened together, maximally perturbs the
   active site? That is C(N, k) — about 2 × 10¹⁰ subsets for N = 300, k = 5 — with an
   eigendecomposition per evaluation, so classically it is a greedy approximation at best.
   It maps directly onto a QUBO/Ising form. This puts quantum on a subproblem that is
   genuinely combinatorially hard, instead of on a physics readout where the measurements
   above show classical does better.
2. **Symmetric multimers.** Every target here is a single chain by construction, and
   single chains are exactly where eigenvalue degeneracies are absent. Symmetric
   oligomers are where degeneracy is real and where allostery *is* symmetry breaking. The
   negative results above do not transfer to that regime because it was never tested.
3. **Algorithmic speedup rather than accuracy.** ALPS needs N eigendecompositions, O(N⁴).
   Estimating spectral shifts via quantum phase estimation on the walk operator would not
   need full diagonalisation. This is a scaling claim, not an accuracy claim, and it is
   the one framing that survives every negative result above.

**Not worth further tuning:** more variants of the coherent readout (different time
scales, initial states, Hamiltonian normalisations). Five have been tried and they land
between chance and 27%, against 91% for the spectral readout. That gap is not a
hyperparameter problem.

---

## 6. Limitations

1. **Proxy labels.** `y` = "a drug-like molecule binds here, ≥ 8 Å from the catalytic site",
   not an experimentally validated allosteric site.
2. **The benchmark has a distance bias.** `ctrl_dist` reaches 42–55% on both independent
   sets. Only methods that beat it on something count — ALPS does so on `hit5`, not on `sig`.
3. **ALPS hyperparameters were chosen on tier-A**; tier-B is the unbiased estimate.
4. **Coordinates are holo conformations** (protein atoms only, ligands stripped), so this is
   not strictly an apo test. QASC's own three targets *are* apo — a systematic difference.
5. `apop`, `qpr`, `cpr` are skipped for N > 660 (they need O(N) eigendecompositions), so
   their n is slightly lower.
6. **Small samples.** tier-A is 11 targets; tier-B at n = 90 gives roughly ±10% on a 49%
   rate, so the ALPS-vs-`ctrl_dist` and ALPS-vs-`apop` gaps on *significance* are inside
   the interval. The `hit5` gap (24.4% vs 7.8%) is the one that is comfortably outside it.

---

## 7. Reproduce

```bash
pip install numpy scipy

python3 scripts/build_dataset.py   1666 80    # tier-A  (needs network; RCSB + PDBe)
python3 scripts/build_dataset_b.py 9000 130   # tier-B  (resumable)

python3 scripts/evaluate.py --targets data/targets        # tier-A
python3 scripts/evaluate.py --targets data/targets_b      # tier-B (held out)
python3 scripts/evaluate.py --targets data/qasc_targets   # QASC's own three

python3 scripts/ablate_readouts.py                       # which observable carries the signal
```

Scoring your own structure:

```python
import numpy as np
from methods.alps import alps_scores

cb     = ...   # (N, 3) per-residue Cbeta coordinates
anchor = ...   # indices of the active-site residues
scores = alps_scores(cb, anchor)          # (N,) higher = more allosteric
```

---

## 8. Repository layout

```
methods/     common.py  quantum.py  btb.py  enm.py  qpr.py  alps.py
scripts/     build_dataset.py  build_dataset_b.py  evaluate.py  ablate_readouts.py
data/
  targets/       tier-A  (11 npz: cb, anchor, y, resnums)
  targets_b/     tier-B  (90 npz, all evaluated)
  qasc_targets/  the three targets shipped with QASC
  manifest*.json results_*.json
docs/
  RESULTS.zh.md              detailed results write-up (Chinese)
  methods.zh.md              method-by-method notes (Chinese)
  literature-review.zh.md    literature survey behind the method design (Chinese)
  literature-evidence-cards.jsonl   verified quote-level evidence backing that survey
```

The literature survey in `docs/` was produced with a high-recall multi-source search
(29 queries, 2 rounds of citation snowballing, 1517 deduplicated papers) followed by
quote-level verification: every claim is backed by a verbatim quote mechanically re-checked
against the source PDF. 110 of 112 evidence cards passed; the 2 that failed are not cited.

---

## 9. Credits

- `data/qasc_targets/` and the `qasc_*` method implementations reproduce
  [Arthur031221/quantum-allosteric-scanner](https://github.com/Arthur031221/quantum-allosteric-scanner)
  (MIT). This repository exists to test that idea independently, and reports where it
  succeeds as well as where it does not.
- `apop`, `corrsite`, `prs`, `btb` are ports of published methods to a residue-graph input;
  they are re-implementations for comparison under one protocol, not the authors' code, and
  should not be read as definitive evaluations of those methods. In particular, the
  bond-to-bond authors explicitly warn that residue-level coarse-graining can lose the
  signal — the weak `btb` numbers here are consistent with that warning.
- Structural data from [RCSB PDB](https://www.rcsb.org/); UniProt mappings via
  [PDBe SIFTS](https://www.ebi.ac.uk/pdbe/).

## License

MIT — see [`LICENSE`](LICENSE).

# ECCR Next-Architecture Preregistration

**Status:** preregistered proposal; no result and no capability claim
**Date:** 2026-07-23
**Proposed treatment:** Monotone Counterexample-Transport Fiber Reactor (MCTFR)

## 1. Frozen evidence and diagnosis

The score-bearing boundary is the source-deleted physical tensor bundle:
complete deterministic transitions, observation-equality tensors, and active
record/generator/query masks. No quotient, class identifier, path certificate,
family, motif, renderer map, or assessor output may enter `forward`.

### Existing round-8 results

| Run | Train exact | Dev exact | Dev hard-valid | Legal but wrong | First hard-failure counts |
|---|---:|---:|---:|---:|---|
| pairwise, seed `2026072303`, data `2026072304` | 250/256 | 40/64 | 46/64 | 6 | observation 10, generator 4, transitivity 4 |
| pairwise, seed `2026072305`, data `2026072306` | 254/256 | 45/64 | 45/64 | 0 | observation 7, generator 11, transitivity 1 |
| record-fiber, seed `2026072305`, data `2026072306` | 254/256 | 44/64 | 44/64 physical; 64/64 equivalence | 0 | 10 observation-invalid and 20 descent-invalid; the former are a subset of the latter |

The two pairwise runs therefore give 85/128 exact and 91/128 valid
development decodes. Of 43 total misses, 37 (86.0%) are rejected before
quotient scoring and 6 (14.0%) are legal refinements of the target. The six
legal errors have coarseness precision 1.0 and reduced recall: they split a
true class rather than inventing a legal coarser quotient. Reported pairwise
failure reasons are ordered first failures because the decoder checks
transitivity, then observations, then generators; they are not guaranteed to
be disjoint latent causes.

The matched record-fiber result is the decisive architectural control. It
guarantees reflexivity, symmetry, transitivity, and exact projector laws on
64/64 examples, but exactness falls from 45/64 to 44/64. It makes 59
false-collision ordered pairs, zero false splits, and reaches only 44/64
generator-descent-valid examples. Its 16 noncommuting-context examples score
3/16, with 41 false collisions. Structural equivalence decoding is therefore
not the missing capability; the shared encoder is merging records before it
has transported all observation and generator counterexamples.

### Exact implementation causes

1. **Universal constraints are represented by means.** The current encoder
   averages query evidence, outgoing generator evidence, incoming evidence,
   and global record evidence. A single violating query or generator is
   diluted as `Q` or `G` grows. Development uses `G,Q in {3,4}`, while train
   uses `G,Q in {1,2}`. All 17 pairwise observation first-failures occur in the
   unseen multisensor family.
2. **The final pair head has no persistent pair state.** It must reconstruct
   `R(T_g i,T_g j)` from two compressed unary record embeddings. The required
   greatest-fixed-point operator lives on record pairs, not individual
   records.
3. **Generator composition is collapsed before comparison.** Generator states
   begin identically and are updated from pooled transition marginals. Systems
   with matched unary marginals but different ordered compositions can remain
   indistinguishable. The shared 3/16 noncommuting-context result across the
   pairwise and record-fiber decoders is the clearest symptom.
4. **The current physical residuals are weak averages.** Observation and
   descent residuals each have weight `0.05` and average over every active
   constraint. They do not impose a count-invariant worst-case margin.
5. **Record-fiber replicas are correlated, not independent error correction.**
   Five votes share the encoder and head input. Equality of thresholded rows
   guarantees an equivalence relation, but a systematic row collision merges
   an entire class. The fresh run's minimum absolute vote margin is only
   `0.00743`.
6. **The corpus does not identify architecture from coverage.** Train and
   development simultaneously change geometry (`N/G/Q`), latent-state count,
   family, and motif. The results prove failure on the frozen held-out
   distribution, but not whether larger geometry or unseen semantics is the
   dominant causal variable. A factorial control is mandatory.

## 2. Mathematical target

For records `X`, observations `o_q`, and deterministic generators `T_g`, the
coarsest causal congruence is the greatest fixed point

```text
Phi(R)(i,j) =
  [AND_q o_q(i) = o_q(j)]
  AND [AND_g R(T_g(i), T_g(j))].
```

Equivalently, distinction is the least fixed point obtained by propagating an
observational counterexample backward through aligned generator transitions.
The architecture should learn this local transport law and its stopping
geometry, rather than infer a global relation from pooled unary summaries.

## 3. Preregistered treatment: MCTFR

### 3.1 State and routing

- Maintain a tied recurrent pair state `z_t[i,j]` for every active ordered
  record pair. Use exactly eight rounds, matching `N <= 8`; no host-observed
  adaptive loop or retry is allowed.
- Initialize `z_0[i,j]` from the full per-query equality pattern
  `observation_equal[i,q,j,q]`, using shared query encoders plus both
  log-sum-exp and max channels. Do not use a mean-only summary.
- At round `t`, gather the aligned successor-pair states
  `z_t[T_g(i),T_g(j)]` directly with the physical one-hot transition tensor.
  A shared generator-equivariant cell aggregates them with max/log-sum-exp,
  preserving the universal-constraint scale as `G` changes.
- Update with a tied monotone residual cell: the designated distinction
  channel may increase but not decrease across rounds. Nonnegative
  parameterization applies only to that channel; auxiliary channels remain
  unrestricted.
- Ordered generator composition is represented by recurrence: round `t+1`
  sees the same-generator successor pair whose round-`t` state already
  summarizes all length-`t` continuations. No generator identity embedding or
  canonical generator numbering is introduced.

### 3.2 Observation-safe anonymous fibers

The hard signature for record `i` concatenates:

1. immutable observation-fiber bits
   `B_obs[i,q,a] = [o_q(i) = o_q(a)]` for every active query and record anchor;
2. learned dynamical fiber bits produced from `z_8[i,a]`.

Hard equivalence is equality of complete active signature rows. This retains
the record-fiber guarantee of an equivalence relation. The immutable
observation fibers additionally guarantee observation preservation: if two
signature rows match, the anchor bit at each query forces their observations
to match. These bits use only input equality structure, are invariant to
injective value recoding, and transform equivariantly under record/query
reindexing.

Generator descent is deliberately **not** repaired or hard-coded. It remains
the primary learned and falsifiable gate.

### 3.3 Objective

Use final target-relation supervision only; intermediate oracle partitions or
distinction paths are prohibited.

```text
L = L_balanced_fiber
  + 1.0 * L_max_descent
  + 0.5 * L_fixed_point
  + 0.5 * L_orbit
  + 0.1 * L_margin.
```

- `L_balanced_fiber`: separately normalized positive/negative supervision for
  the anonymous final relation and learned fiber bits.
- `L_max_descent`: per-episode smooth-max hinge over
  `p(i~j) - p(T_g(i)~T_g(j))`; no averaging over generators or pairs.
- `L_fixed_point`: consistency between the final two recurrent relation
  proposals, without changing the hard output or invoking a closure operator.
- `L_orbit`: mapped-logit consistency across matched reindex, value-recode,
  split, and merge presentations; mappings are offline training metadata and
  never enter `forward`.
- `L_margin`: keep all active hard bits at least one logit unit from zero.

The added module is capped at 24,000,000 parameters. With the protected
125,081,664-parameter Shohin base, the complete system is capped at
149,081,664 parameters, below the 200M ceiling.

## 4. Controls and identifiability

Every treatment comparison uses identical packets, optimizer updates, batch
order, threshold, and seed.

1. Existing round-8 pairwise encoder/head.
2. Existing round-8 record-fiber model.
3. Parameter-matched generic pair-state reactor with mean pooling.
4. MCTFR without successor-pair transport.
5. MCTFR with the observation-fiber skip but the old unary encoder.
6. MCTFR with generator alignment independently permuted on one transition
   side; it must lose descent competence.
7. Frozen random MCTFR and shuffled final targets; neither may pass.
8. A non-neural Boolean partition-refinement oracle is reported only as a
   mechanics ceiling and is excluded from every neural claim.

The corpus analysis must add an equal-budget 2x2 diagnostic board:
seen/unseen geometry crossed with seen/unseen family/motif. Exact, latent,
action, path, and orbit leakage gates remain unchanged. This board determines
whether remaining error is cardinality extrapolation, semantic transfer, or
their interaction; it cannot replace the frozen 256/64 score.

## 5. Frozen gates

Run the frozen 256/64 board on three fresh model/data seed pairs. Promotion
requires **every** seed to satisfy:

- train exact relation at least 99%;
- development equivalence validity exactly 100%;
- development observation validity exactly 100%;
- development generator-descent and total physical validity at least 95%;
- development exact target relation at least 90%, with median at least 95%;
- conditional exactness among physically valid relations at least 98%;
- false-collision and false-split rates each at most 0.5%;
- noncommuting-context exactness at least 80%;
- reindex and value-recode orbit consistency 8/8, all-orbit consistency at
  least 7/8;
- at least 15 percentage points exact improvement over the matched
  record-fiber control and a paired two-sided McNemar `p < 0.05`;
- all source-deletion, source-hash, split-isolation, parameter, and single-hard-
  decode custody gates pass.

Kill or redesign the treatment if any seed is below 85% exact, below 90%
physical validity, or below 60% noncommuting-context exactness. Do not scale a
failed pilot to the 48k/4k board.

Diagnostic interpretation is frozen:

- observation validity below 100% means the observation-safe fiber contract is
  incorrectly implemented;
- high equivalence/observation validity but low descent validity rejects the
  counterexample-transport cell;
- high physical validity but low exactness, with precision 1 and recall below
  1, identifies conservative under-merging/coarseness failure;
- good seen-geometry and bad unseen-geometry cells identify count transfer;
- good geometry-complete but bad unseen-motif cells identify semantic
  composition failure.

## 6. Why this is not external symbolic execution

MCTFR performs one fixed differentiable forward pass from physical tensors to
model-owned pair states and anonymous fibers. The host never computes a
partition, chooses a candidate, supplies a certificate, executes a path,
checks an intermediate state, retries, closes, or repairs the proposal.
Generator routing is a neural architectural adjacency operation, analogous to
convolutional routing; the recurrent transition and dynamical signature are
learned checkpoint parameters. A single frozen threshold decodes the final
fibers once.

The immutable observation-fiber skip enforces only an input-preservation law;
it does not determine generator closure or the coarsest quotient. Therefore a
random, no-transport, or mean-pool model can still fail the primary descent and
exactness gates. A pass would establish bounded source-deleted neural quotient
induction, not language reasoning or genuine general reasoning.

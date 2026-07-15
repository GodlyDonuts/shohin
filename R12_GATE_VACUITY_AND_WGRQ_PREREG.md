# R12 Gate Vacuity Correction and WGRQ Preregistration

**Status:** gate correction adopted. Independent audit rejects WGRQ as a new
state, algorithm, or oracle advantage. The narrower Stage-A neural optimization
falsifier is frozen separately in `R12_WGRQ_CPU_PREREG.md`; this document alone
authorizes no implementation or job.

## 1. Why the gate changed

The first R12 wording accidentally made acceptance impossible. Every bounded
finite-precision classical mechanism has a finite acyclic unrolling, while the
old exact-collapse gate treated successful unrolling as rejection evidence.
Therefore every realizable candidate failed before its resource claim could be
examined.

There was a second tautology. If a comparator class contains the candidate,
then the best member of that class cannot be asymptotically worse than the
candidate under the same resource measure. A universal recurrent control that
is explicitly allowed to execute the candidate's identical algorithm is a
correct expressivity ceiling but cannot test whether a training protocol finds
that algorithm more reliably or with different data/compute.

The corrected rule is narrow:

> A reduction rejects a novelty or resource claim only when it preserves
> behavior, information access, and the preregistered resource vector within
> constant or polylogarithmic overhead.

The vector is

```
(parameters, retained bits, precision, source bytes, training examples,
 oracle calls, training FLOPs, inference FLOPs, sequential depth,
 external memory, external execution).
```

Finite unrolling still blocks ontological claims. Known machinery still blocks
primitive-novelty claims. Neither is an automatic veto of a bounded training-
protocol experiment.

## 2. Absolute no-go results retained

The correction does not weaken:

- residual-state information lower bounds;
- arbitrary late-query information conservation;
- hidden-coordinate nonidentifiability under conjugacy;
- finite off-support and delayed-sabotage constructions;
- passive rare-witness/sample lower bounds;
- exact accounting of precision, caches, source access, and external execution;
- the closed-deliberation theorem when no new target information enters.

OOD correctness must still receive at least one honest source: a restrictive
hypothesis class, distinguishing data/interventions, or a bounded distributional
claim. Hiding that source is claim-killing.

## 3. Candidate training protocol: WGRQ

**Witness-Guided Residual Quotienting** manipulates supervision and information
access, not model-module vocabulary.

For a history `h`, an encoder commits to a query-blind state `z(h)`. After the
commit:

1. source tokens and their KV cache are deleted;
2. histories with equal future behavior receive an interchange/merge target;
3. suspected false merges receive a training-only distinguishing continuation
   and late query;
4. transition closure requires equivalent states to remain equivalent after
   every shared event;
5. counterfactual state swaps test whether consumers use causal state rather
   than lexical identity;
6. one committed state must answer many late queries and accept appended
   continuations without source recovery;
7. no simulator, witness generator, source retrieval, or verifier is available
   at inference.

The manipulated variable is future-equivalence supervision under a hard source
barrier. Recurrence, state width, decoder, token budget, and inference compute
are held identical in the principal control.

## 4. Capability and resource hypothesis

Use one protocol and one hyperparameter set on two unrelated exact families:

- noncommutative adjacent-transposition composition with late image queries;
- visible-coordinate reversible Boolean actions with late bit/readout queries.

The bounded hypothesis is:

> At matched parameters, retained bits, training examples, training/inference
> FLOPs, sequential depth, and target-oracle calls, witness-guided quotient
> supervision increases exact source-free length/scale extrapolation by at
> least five confidence-separated percentage points over identical recurrent
> controls that lack quotient supervision.

This is not a claim of sample information creation. Training-only witnesses are
oracle calls and must be counted. The test asks whether spending that fixed
oracle budget on distinguishing residual collisions is more effective than
random or answer-only supervision.

## 5. Required controls

- answer/visible-trace SFT with identical training-token and target-call budget;
- identical tied-recurrent architecture without quotient losses;
- identical architecture with random rather than adversarial witnesses;
- identical architecture without source deletion;
- a PSR/OOM, weighted-automaton, or partition-refinement control with matched
  retained state and every target call counted;
- exact symbolic realization as a ceiling, not a novelty comparator.

Every neural arm starts from identical initialization and sees an immutable,
hash-bound training generation. No arm may see confirmation examples.

## 6. Frozen CPU falsifier requirements

Before implementation, an independent audit must freeze the exact data,
architecture, optimizer, seeds, resource ledger, and decision rule. The minimum
board then requires:

- exhaustive no-collision residual checks on the smallest scales;
- randomized symbol relabelings so lexical labels cannot identify state;
- train on short compositions and evaluate at at least eight times train length;
- unseen state scale as well as unseen length;
- 32 late queries and appended continuations from one source-deleted state;
- equivalent-state interchange and non-equivalent-state separation;
- exact target-call accounting for witness, random-witness, and control arms;
- at least 95% exact candidate accuracy and a confidence-separated gain of at
  least five points over every matched neural control on both families.

One failed family, residual collision, source/KV leak, hidden solver path, or
resource mismatch rejects the protocol. A finite pass permits an isolated
Shohin-scale design review; it does not establish asymptotic reasoning.

## 7. Allowed claim if every gate passes

> WGRQ is a tiny-model training protocol that learns reusable query-blind causal
> states and improves source-free compositional extrapolation at matched state,
> compute, data, and oracle budgets on two frozen exact families.

It is not a new computational primitive, a proof of general intelligence,
compression below causal entropy, or a universal context solution.

## 8. Independent audit verdict

The original two-family draft is no-go as written:

- adjacent permutations and fully visible Boolean coordinates both have
  immediate distinguishing queries, so they do not test nonempty witness
  discovery;
- equal oracle-call counts do not equalize returned witness identities,
  response bits, adaptive rounds, or teacher search compute;
- whenever quotient labels are derived from ordinary public answers, a fair
  active answer-only learner can replay the identical transcript and derive
  the identical labels;
- residual merging is automata minimization/bisimulation, distinguishing
  suffixes are active automata learning/CEGIS, and query-blind future state is
  PSR/OOM machinery;
- the current flat Shohin SFT corpus lacks certified semantic states and cannot
  support a residual-equivalence language claim.

`R12_ACTIVE_WITNESS_ALLOCATION_NO_GO.md` freezes the active-answer-only
simulation theorem. `R12_CANONICAL_RESIDUAL_NAMING_CONTROL.md` defines the
symbolic ceiling. `R12_CERTIFIED_LANGUAGE_BRIDGE_BOUNDARY.md` defines the later
language barrier.

The only surviving empirical question is whether a behavioral relational loss
optimizes an information-identical recurrent learner better than favorable
controls under hard source deletion. `R12_WGRQ_CPU_PREREG.md` replaces the two
immediate-readout families with a delayed-witness edge-parity ring and freezes
that bounded falsifier. A pass would not restore any rejected novelty claim.

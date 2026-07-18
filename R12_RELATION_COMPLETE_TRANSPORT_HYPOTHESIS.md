# R12 Relation-Complete Transport Hypothesis

**Status:** theory and finite CPU collapse-test candidate only. No neural
implementation, data generation, fit, score, accelerator allocation, reasoning
claim, or novelty claim is authorized.

## 1. Motivation

Shohin's direct DRS probes show a characteristic asymmetry: many local digit
transitions are correct, while operation order, long-range state transport, and
terminal serialization fail. Generic replication is a poor match for this
failure because three lanes can agree on the same wrong semantic action.
Invertible transport is also insufficient because a bijection preserves an
error rather than correcting it.

The remaining cross-domain analogy is more specific. Physical gauge systems,
error-correcting constraint complexes, and biological proofreading all exploit
closed consistency relations. A missing or corrupted local transition can be
identified when it creates nonzero defect around enough independent closed
loops. The proposed object is therefore not another hidden scratchpad. It is a
globally relation-constrained transition law whose local errors create algebraic
syndromes.

## 2. Capability object

Use the R12 late-query permutation witness. For `m` objects, the event alphabet
contains adjacent transpositions `tau_i`, a history is a word in those
generators, and a query revealed after the history asks where one object moved.
The exact causal state is the resulting permutation.

The smallest noncommutative instance is `S_3` with generators `s=(01)` and
`t=(12)`. Its Coxeter presentation is

```text
s^2 = e
t^2 = e
sts = tst
```

A six-state action table assigns one successor to every one of the twelve
`(state, generator)` pairs. A relation-complete table must satisfy all three
relations from every state and its generated orbit must contain all six states.
Checking the relators only at the identity is explicitly insufficient.

## 3. Finite identification theorem

For labeled six-state deterministic actions of `s` and `t`:

1. exactly 120 transitive action tables satisfy `s^2=t^2=e` and `sts=tst`
   globally;
2. these 120 tables are precisely the labeled regular actions of `S_3`;
3. if one directed edge is erased from the canonical table while the other
   eleven remain fixed, only the canonical successor completes all global
   relations;
4. among the 120 globally valid tables, four suitably chosen labeled edges are
   sufficient and necessary to identify the canonical table.

The CPU falsifier must derive all four statements by enumeration. It may not
encode the counts as acceptance literals.

This is a finite identification result, not an asymptotic reasoning theorem.
In particular, it assumes a fixed six-state carrier and exact global relation
enforcement. A neural hidden state does not arrive with those semantic labels.

## 4. Resource hypothesis

An unconstrained six-state, two-generator atlas has `6^12 = 2,176,782,336`
possible tables and needs twelve successor labels. Using a fixed-width
three-bit state identifier, that is 36 labeled target bits.

The global presentation reduces the candidate set to 120 tables. Four exact
edge labels can then identify the canonical table, using 12 labeled target
bits. This apparent threefold reduction is not free:

- the generator and relation presentation is retained side information;
- all relations are checked from every state, costing 60 transition
  applications for one full six-state check;
- the carrier size and transitivity requirement are supplied;
- exact state labels or an equivalent observation decoder are supplied;
- a hard-coded group action can realize the same behavior with no learned
  atlas at all.

The surviving conjecture is therefore narrow:

> At matched trainable parameters and total optimization compute, global
> relation-syndrome supervision can reduce the number of labeled transition
> examples needed to learn a uniform action, compared with endpoint-only
> training, while generalizing to withheld edges and longer equivalent words.

The claim is about learnability and data allocation. It is not a new state
ontology or a separation from finite-state recurrence.

## 5. Equivalence dossier

### Tied recurrence

A deterministic relation-aware recurrent updater over the same six states has
exactly the same feasible action tables. This is the mandatory favorable
control. Any neural treatment must beat an equally parameterized recurrence
given the same relation checks, labeled edges, updates, and inference depth.

### Finite atlas

An untied atlas can represent every candidate but receives no structural
generalization unless the presentation is imposed. It is the endpoint-only
control, not the strongest comparator.

### Hard-coded execution

Swapping permutation coordinates implements the witness exactly. That is a
known symbolic algorithm and a capability ceiling. It cannot be described as a
learned reasoning primitive.

### SFT and recurrence

Relation losses are structured supervision. Unrolling their optimizer yields
ordinary training computation, and the learned updater is still a recurrent
finite-state transducer. The only reopenable question is whether the structural
constraints improve sample efficiency or scale extrapolation at matched
resources.

### Retrieval, fast weights, and external execution

The finite CPU board uses explicit tables only to test identifiability. A future
neural experiment may not retrieve a gold table, generate weights from the
source, or execute the group action on the host at inference. Oracle relations
may supervise training, but no oracle state, query answer, repair loop, or
verifier may enter autonomous evaluation.

## 6. Exact collapse test

The CPU artifact must:

1. enumerate all 76 involutions on six labels and every pair;
2. retain only globally relation-valid, transitive pairs and derive the count
   120;
3. independently enumerate every completion of one erased edge and prove the
   unique valid completion is exact;
4. enumerate all 4,096 edge-observation subsets and derive the minimum exact
   identifying size and version-space profile;
5. exhibit a wrong patch that passes identity-only relators but fails global
   relators;
6. show that the relation-complete atlas candidate set equals the matched tied
   relation-aware recurrence candidate set;
7. emit target-bit, relation-check, candidate-count, and presentation-byte
   ledgers.

A CPU pass permits only independent review and, if that review accepts the
resource accounting and prior-art boundary, drafting a neural preregistration.
It does not authorize neural source or fitting.

## 7. Future neural falsifier boundary

Any later preregistration must use increasing unseen scales and lengths, not
only `S_3`. It must freeze:

- relation treatment, endpoint-only control, shuffled-relation control, and a
  favorable relation-aware tied recurrence;
- identical trainable parameter ceilings and optimizer-update budgets;
- exact labeled-edge and oracle-relation counts;
- held-out transitions, unseen equivalent words, non-equivalent order twins,
  and every late query;
- state bits, source bytes, training examples, training/inference FLOPs,
  sequential depth, and external execution;
- an autonomous evaluation with no oracle cursor, state, schedule, repair, or
  host executor.

The direct Shohin relevance would remain conditional. A permutation witness
pass would establish relation-guided state transport, not natural-language
compilation, decimal arithmetic, or terminal serialization.

## 8. Prior-art and claim boundary

The mathematical ingredients are known: Coxeter presentations, Cayley graphs,
constraint syndromes, cycle consistency, group-equivariant learning, and
finite-state recurrence. No primitive-novelty claim is allowed. The potentially
new contribution is only a rigorously controlled training protocol for a tiny
language model whose supervision is allocated through global algebraic defect.
That delta requires a dedicated prior-art review after the CPU object is frozen.

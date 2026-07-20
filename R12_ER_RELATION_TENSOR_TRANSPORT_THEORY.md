# R12 Episodic Relation Tensor Transport

**Protocol:** `R12-ER-TT-v1-theory`

**Status:** CPU/tensor mechanics and a local neural adapter are implemented.
The adapter is not source-frozen and no board seed, training seed, GPU run, or
scored access exists.

## Motivation

ER-CST Witness Equality Bus v1.1 confirms 99.023% sealed joint accuracy when
fresh operation meanings belong to the six permutations of three positions.
The remaining limit is now explicit: rule cards are class IDs in a hard-coded
`S_3` ontology, and the recurrent motor is a learned 36-cell composition table.
Adding width to that solved interface would not test more general reasoning.

ER-TT removes both finite enumerations. Each episode chooses cardinality
`N in {3,4,5,6}`, two to four fresh operations, and up to twelve invocations.
A determining witness lists `N` distinct input symbols and `N` output symbols.
Outputs may repeat, so an operation is any of the `N^N` total copy relations,
not only a permutation. The compiler must emit each operation directly as an
`N x N` one-hot relation matrix, along with initial state, event bindings, HALT,
and query. No operation class ID exists.

## Architecture hypothesis

For output position `i` and input position `j`, learned occurrence fingerprints
produce equality logit `E[i,j]`. Masked row-wise categorical selection yields
relation matrix `R`. If state matrix `S` maps current positions to entity roles,
the next state is

`S_next = R @ S`.

This matrix multiplication is a parameter-free in-model primitive, like an
attention value mix. A persistent pre-apply HALT mask selects whether each
recurrent step commits. Source text, token memory, and residuals are destroyed
before the tensor motor runs. The final query selects one row of terminal `S`.

This construction has three useful properties:

1. it represents `N^N` operations with `N^2` emitted categorical cells rather
   than an `N^N` classifier;
2. the same tied motor executes bijections, copying, erasure, and many-to-one
   transport without learning a transition table; and
3. relation composition is associative, so depth adds recurrent work but no new
   operation-specific parameters.

## Falsifiable neural contract

- Reuse the confirmed 192,726,827-parameter parent where valid.
- Replace fixed three-position/card-class heads and the learned motor/reader with
  masked coordinate-generated occurrence queries, direct relation rows, the
  parameter-free tensor motor, and categorical row readout.
- Support variable cardinality, two to four rule records, and depth one to twelve.
- Measure the exact complete-system parameter count before any board seed; it
  must remain strictly below 200M.
- Training may supervise active witness pointers, relation rows, record/event
  bindings, initial rows, HALT, and query. It may not supervise final state,
  answer, recurrent trajectory, development, or confirmation.
- The primary controls remain family-deranged semantics and equality-ablated
  witnesses. Add relation-row derangement, cardinality-mask corruption, source
  poison after packet sealing, storage reindex, alpha rename, and post-HALT suffix.
- Development and sealed confirmation must be split-disjoint in names, latent
  families, renderer compositions, exact prompts, and word 13-grams.

## Admission sequence

1. Prove CPU witness inference, arbitrary relation execution, associativity,
   alpha invariance, storage invariance, padding invariance, source deletion,
   HALT persistence, and causal-control sensitivity.
2. Prove the torch relation motor against exhaustive/sampled CPU references,
   with zero trainable parameters and no source input.
3. Implement and parameter-audit the smallest neural compiler extension.
4. Freeze source before any board seed; build, independently reproduce, and seal
   a fresh board before a training seed.
5. Run one equal-budget development qualification and only one separately frozen
   confirmation if every absolute and causal gate passes.

## Claim boundary

A future pass would establish variable-cardinality episodic finite-relation
compilation and source-deleted recurrent composition. It would still not prove
free-form language grounding, unbounded algorithms, arithmetic, branching,
planning, or general intelligence. Those require later boards whose program
structure itself is inferred rather than supplied by the finite grammar.

## Adapter implementation receipt

The implemented adapter totals 192,740,854 parameters, of which 12,037,293 are
trainable, leaving 7,259,146 below the strict 200M ceiling. It removes the
finite permutation buffer, direct card classifier, learned categorical motor,
and learned reader from the deployed path. Actual reconstruction from the
confirmed v1.1 checkpoint copies every retained parent tensor byte-identically.
Exact architecture, supervision, controls, and gates are preregistered in
`R12_ER_RELATION_TENSOR_ADAPTER_PREREG.md`. No scientific board may be seeded
until this source and contract are committed.

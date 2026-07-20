# R12 Episodic Relation Tensor Transport Neural Adapter

**Protocol:** `R12-ER-TT-v1-neural-adapter`

**Status:** implementation and local qualification before source freeze. No
scientific board seed, training seed, GPU run, or scored access exists.

## Scientific objective

The confirmed ER-CST v1.1 system compiles fresh opaque witnesses into one of six
enumerated permutations and composes those classes with a learned 36-cell
motor. ER-TT asks a strictly broader question: can the confirmed compiler emit
the relation itself for variable cardinality and can a source-deleted in-model
tensor operation compose it without a class ontology or learned transition
table?

Episodes will use `N in {3,4,5,6}`, two to four fresh rule records, up to twelve
active updates, one persistent pre-apply HALT, and arbitrary total copy
relations. Outputs may repeat. The relation space at cardinality `N` is `N^N`,
but the compiler emits only `N^2` row logits.

## Confirmed parent and exact parameter contract

| Component | Parameters |
|---|---:|
| Frozen Shohin base | 125,081,664 |
| ER-TT compiler | 67,659,190 |
| Learned recurrent motor | 0 |
| Learned answer reader | 0 |
| **Complete deployed system** | **192,740,854** |
| **Trainable** | **12,037,293** |
| **Headroom below 200M** | **7,259,146** |

The parent is the independently confirmed ER-CST v1.1 checkpoint with SHA-256
`917c1a1fce67c02258d0f90f04398ab433d18ba63c2dca92450cc5856c022ae7`.
Its confirmation assessment SHA-256 is
`4a0fb47233d86887bb46aa853560bf81d319840610d62abe4f1dfaa899671310`.

Local reconstruction produces parent state SHA-256
`775a3fa1b42c0a578492d36e8b8001ba59f672c0a7752033259512de13f0df75`.
Every retained parent tensor copies byte-identically; the copied-subset digest is
`ccb1420f9b27846efdd7696d360421210e8afcd818cc62b46f69a909f841d1a8`.
The excluded frozen-state digest is
`d45d757f17e1ddb88c98a6005ea7a746df880333805d1966cb3c42c169ca636c`.

## Architecture

The adapter uses eighteen shuffled physical records: one declaration, four rule
records, and thirteen event records. Coordinate-generated witness queries are
the sum of one of two side embeddings and one of six position embeddings. They
select six `before` and six `after` occurrences per rule. The inherited learned
byte fingerprints produce direct equality logits

`E[r,i,j] = scale * <after[r,i], before[r,j]>`.

These logits are the relation rows. There is no permutation classifier. Six
declaration binding pointers and six initial-state pointers similarly produce a
direct `6 x 6` initial-state matrix. A four-way cardinality head selects active
rows/columns for `N=3..6`. Rule-active, event-card, HALT, and six-way query heads
emit the remaining structural fields.

After hard sealing, no source bytes, token memory, record residuals, pointer
logits, or model module are passed to execution. The recurrent motor has zero
parameters:

`S_next = R_selected @ S`.

HALT is persistent and pre-apply. The answer reader has zero parameters: a hard
query row selects one row of terminal state. The inherited six-permutation
buffer, direct permutation head, fixed three-position query head, and learned
categorical motor/reader are absent from the deployed path.

## Allowed supervision

Training may supervise only source-compiler fields:

- physical record roles and line pointers;
- cardinality and active rule slots;
- declaration binding and initial-entity pointers;
- all active witness occurrence pointers;
- active initial-state rows and active relation rows;
- event-to-rule references, HALT, and query position; and
- renderer-consistency terms over compiler fields.

Training must not expose final state, answer, recurrent trajectory, intermediate
state, development oracle, or confirmation oracle. The parameter-free motor and
reader receive no fitted weights.

## Required board design

- 48,000 training, 2,048 development, and 2,048 sealed confirmation rows;
- exactly balanced cardinality and scored depth, with balanced rule counts;
- at least 90% of episodes containing a non-bijective rule;
- split-disjoint family IDs, names, complete semantic-family signatures,
  renderer compositions, exact prompts, and word 13-grams. Atomic relation rows
  may recur because there are only 27 total relations at `N=3`; generalization
  is measured on unseen combinations, programs, symbols, and renderers rather
  than an impossible atomic-row exclusion;
- independent parser, witness-inference implementation, and executor validating
  every row before training;
- fixed-width compact names so every program fits the inherited 640-byte source
  window and every line fits the inherited 144-byte record window;
- confirmation mode `0600` and access counters `0/0`; and
- no board bytes before exact source is committed and pushed.

## Equal-budget arms and interventions

1. **Treatment:** correct witness equality and rule/event binding.
2. **Family derangement:** relation tensors belong to another family while all
   source structure and budgets remain matched.
3. **Equality ablation:** witness surface lengths/offsets remain matched but
   before/after identity is independently destroyed.

Evaluation must also include relation-row derangement, cardinality-mask
corruption, rule-storage reindex, physical-record reindex, witness alpha rename,
opcode alpha rename, source poisoning after packet sealing, post-HALT suffix,
state reset, and query swap.

## Frozen development gates

The sole development read may pass only if all conditions hold:

1. complete treatment packet, state, answer, and joint are each at least 90%;
2. relation rows, initial rows, record roles, event references, HALT, query, and
   every active pointer field are each at least 95%;
3. minimum joint by `N=3,4,5,6` is at least 80%;
4. minimum joint by scored depth and unseen renderer is at least 80%;
5. minimum joint on episodes containing a non-bijective rule is at least 85%;
6. treatment packet and joint exceed both equal-budget controls by at least 50
   percentage points;
7. family-deranged and equality-ablated packet/joint are each below 35%;
8. storage reindex, alpha rename, and post-HALT/source-poison invariances are
   exact on every otherwise valid packet;
9. relation-row, cardinality, state-reset, and query interventions cause their
   preregistered answer/state changes;
10. complete deployed parameters are exactly 192,740,854 and below 200M;
11. excluded parent state remains byte-identical; and
12. custody is exactly one development access and zero confirmation accesses.

Passing authorizes one separately frozen confirmation evaluator with unchanged
weights and thresholds. Failure closes this architecture/version; thresholds
will not be relaxed and the same board will not be repaired and rescored.

## Current local verification

Thirteen focused mechanics/adapter tests plus three parent-reconstruction tests
pass. They cover arbitrary non-bijective equality recovery, variable
cardinality, hard masking, persistent HALT, zero-parameter readout, source-free
packet fields, detached witness gradients, excluded-parent gradient isolation,
exact parameter counts, fail-fast parent hashes, and actual confirmed-parent
reconstruction. Ruff, byte compilation, and diff checks pass.

## Claim boundary

A future pass would establish bounded variable-cardinality episodic relation
compilation and recurrent composition under source deletion. It would not by
itself establish free-form language grounding, unbounded algorithms, arithmetic,
branching, planning, or general reasoning.

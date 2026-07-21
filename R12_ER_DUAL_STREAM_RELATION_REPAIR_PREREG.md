# R12 ER-TT Dual-Stream Relation Repair Preregistration

**Status:** hard-route v1 and marginal-route v1.1 rejected before fresh data;
ordinal-route v1.2 is specified separately in
`R12_ER_DUAL_STREAM_ORDINAL_ROUTE_PREREG.md`

**Parent:** confirmed ER-CST Witness Equality Bus v1.1, checkpoint SHA-256
`917c1a1fce67c02258d0f90f04398ab433d18ba63c2dca92450cc5856c022ae7`

**Closed predecessor:** ER-TT v1, development custody `1/0`; never rescore it
and never open its confirmation split.

## 1. Falsified predecessor and repair target

ER-TT v1 learned ordinary program structure but failed the variable relation
bus. On its sole development read it reached 0.098% exact packets/joints,
15.381% recurrent state, and 32.666% answers. Before-witness localization was
95.273%, after-witness localization was 51.965%, complete witness pointers were
0.586%, and witness/opcode alpha recoding preserved complete output on only
6/2,048 and 32/2,048 rows. Relation loss remained near chance.

The frozen diagnosis is that one content-sensitive pointer path mixed two
different problems:

1. **where:** infer the structural record, role, and occurrence position; and
2. **what:** preserve the complete selected symbol identity for equality.

Hard-route v1 implemented that separation but failed its train-only probe. It
was exactly alpha-invariant on all 8,000 rows yet produced zero exact relation
rows, witness pointers, packets, or joints. Its structural role assignment was
detached from routed pointer supervision, and hard selected-symbol equality
starved gradients whenever the target record began with negligible assignment
mass. V1 is closed and will not be rerun.

V1.1 repairs the gradient topology and replaces the hard selection with exact
equality marginalized over learned routes. More capacity is not the primary
change.

## 2. Architecture

### Structural stream

Every whitespace-delimited six-byte lowercase/base36 symbol is replaced by the
same six canonical bytes before record encoding. Byte positions, delimiters,
line syntax, record labels, record order, and source length are unchanged.
Learned semantic-record assignment and learned declaration/witness/opcode
queries route only over this canonical structural memory. The route therefore
cannot depend on an entity, witness, or opcode's spelling. Semantic assignment
uses the ordinary record features. The numerically identical routing assignment
is recomputed from detached record features so pointer/equality loss can train
the shared role head without modifying the shared structural record encoder.

### Identity stream

The identity stream does not embed or classify symbol spelling. For learned
route distributions `p` and `q`, it computes

`P(equal) = sum_ij p_i q_j 1[source[i:i+6] = source[j:j+6]]`.

The indicator is exact whole-symbol equality and the output log probability is
used as the categorical relation/binding logit. This gives dense gradients to
both routes while remaining exactly alpha-equivariant. Inference uses only
model-owned route distributions and source bytes; no target range or board
dictionary is available.

### Relation and event transport

- declaration initial-state rows are exact identity equality between routed
  initial occurrences and routed declaration bindings;
- each relation row is exact equality between routed after-witness and
  before-witness symbols;
- event-to-rule binding is exact equality between a routed event opcode and
  routed rule opcodes;
- the existing parameter-free source-deleted motor composes relation matrices;
- motor and reader add zero parameters.

The old content-sensitive occurrence head, side/position embeddings, and
record-similarity event binding are absent.

## 3. Model and host ownership

### Model-owned

- physical-record to semantic-role assignment;
- declaration binding and initial occurrence routes;
- all before/after witness routes;
- all rule-opcode and event-opcode routes;
- cardinality, active rules, HALT, and late query;
- the resulting source-deleted relation packet.

### Architectural primitives

- recognition of bounded six-byte symbol-shaped tokens;
- structure-preserving canonicalization of their payload bytes;
- marginalization over model-owned categorical routes;
- exact equality of every routed six-byte symbol pair;
- fixed categorical masking and the preregistered relation-matrix motor.

### Forbidden

- parsing a selected symbol's semantic role from its prefix or bytes;
- board candidate dictionaries or target ranges at inference;
- gold relations, state, answer, trajectory, development, or confirmation
  fields during fitting;
- host repair, retry, search, answer selection, or target-aware routing;
- any trained state from rejected ER-TT v1.

Initialization reconstructs the independently confirmed witness parent, creates
the old ER-TT adapter only as an untrained shape-compatible bridge, copies every
shared tensor byte-identically, removes the failed v1-only path, and initializes
the dual-stream path from a fresh seed.

## 4. Exact parameter certificate

| Component | Parameters |
|---|---:|
| Shohin base | 125,081,664 |
| Dual-stream compiler | 60,450,632 |
| Motor | 0 |
| Reader | 0 |
| **Complete deployed system** | **185,532,296** |
| Trainable | 11,129,504 |
| Headroom below 200M | 14,467,704 |

The cap is absolute: the complete deployed system must remain strictly below
200,000,000 parameters. V1.1 removes 7,197,795 dead v1 parameters rather than
counting or optimizing unused fingerprint, witness-query, occurrence, and scale
tensors.

## 5. Train-only matched-route diagnostic before fresh data

The canary may open only the already-public ER-TT `train.jsonl`. A deterministic
post-source-commit seed partitions all 12,000 families into:

- 10,000 fit families / 40,000 rows;
- 2,000 family-disjoint probe families / 8,000 rows.

Treatment receives two epochs, family batch eight, 2,500 updates, AdamW at
`2e-4`, 100-step warmup, cosine decay, and exactly the old source-only compiler
loss. Final state and answer are mechanically derived only for the held-out
train-family probe after fitting. They never enter an optimizer loss.

The same 8,000 probe rows contain two explicitly separated route controls:

1. **Oracle-route identity transport.** Source-only compiler target spans set
   one-hot declaration, witness, rule-opcode, and event-opcode routes. The same
   exact marginal-equality operator must recover initial rows, arbitrary
   relation rows, and event binding at 100%. This control receives no final
   state, answer, trajectory, development, or confirmation field and cannot
   authorize promotion by itself.
2. **Learned soft-route treatment.** The trained structural role and local
   pointer distributions feed the same equality operator. Only this arm is
   compared with the frozen capability thresholds and may authorize a fresh
   board.

Rejected hard-route v1 is retained only as immutable prior evidence. It is not
rerun, rescored, or given another seed.

A pre-freeze CPU production-row audit stratified the old training split by all
four renderers, cardinalities 3--6, rule counts 2--4, and depths 1--12. The
same oracle-route equality operator recovers initial rows, relation rows,
events, and their joint on 1,152/1,152 rows in 576 strata. This is a mechanics
qualification only, not a neural score.

For the alpha test, every entity, witness, and opcode symbol is bijectively
renamed into one neutral `z.....` namespace. Prefix-category information is
therefore removed rather than merely shuffled within `e`/`w`/`o` classes.

No development or confirmation path is accepted as an input argument by the
job. This canary creates no scored-split ledger and changes no old custody.

## 6. Immutable canary gates

A fresh board is authorized only if all gates pass:

1. oracle-route initial, relation, event, and joint transport are each exactly
   8,000/8,000 through the same equality operator;
2. exact 10,000/2,000 family-disjoint split with zero overlap;
3. learned-soft-route packet, state, answer, and joint each at least 85%;
4. learned-soft-route complete relation rows at least 90%;
5. learned-soft-route complete active witness pointers at least 90%;
6. learned-soft-route events and HALT each at least 95%;
7. minimum cardinality-specific learned-soft-route joint at least 75%;
8. every learned hard packet field and every pointer is bit-identical on all 8,000
   original/neutral-alpha probe pairs;
9. the confirmed parent remains byte-identical outside the declared trainable
   set;
10. the exact parameter certificate remains below 200M; and
11. outcome supervision, development reads, and confirmation reads remain zero.

Failure closes this repair before new board generation. No threshold may be
relaxed after seeing the canary.

## 7. Fresh-board requirements after a pass

A passing canary admits only a new source commit and fresh board. The new board
must put entities, witness symbols, and opcodes in the same neutral six-byte
namespace, add nonsemantic six-byte distractors, retain variable cardinality and
non-bijective relations, use disjoint renderer/name families, and preserve
matched deranged/equality/source-free controls. Development and sealed
confirmation require new independent seeds and one-read custody.

## 8. Claim boundary

A canary pass establishes only that the repair learns routing and relation
transport on held-out families from an existing training generator while
remaining exactly alpha-invariant. It is not fresh development, confirmation,
natural-language reasoning, arbitrary program induction, planning, arithmetic,
or general reasoning.

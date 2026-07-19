# R12 S9 Occurrence-Quotient Relational Compiler Preregistration

**Status:** CPU representation admitted; neural source and board not yet frozen

**Parent result:** S8.1 rejected end to end at 514/2,048 exact graphs, with
514/514 exact execution conditional on graph validity

## 1. Target failure

S8.1 is not failing after compilation. Its valid, exact-graph, exact-state, and
exact-answer sets are the same 514 development rows. The compiler instead loses
complete rosters, state permutations, card witnesses, or repeated-name bindings
under unseen renderers and nonce vocabularies. Its architecture independently
labels BPE tokens, although the source repeatedly refers to the same entity,
position, operation, and event tag.

S9 tests whether **identity before semantics** is a better factorization:

1. propose nonempty surface islands from source tokens;
2. quotient byte-identical proposed islands into occurrence classes;
3. classify local relation types and argument slots using both local context and
   the shared class representation;
4. emit class-level roster, state, card, event, entry, next/nil, and query
   relations; and
5. compile those relations into the unchanged S8 graph and execute through the
   confirmed S7/S8 runtime.

No arithmetic or recurrent component is widened in this phase.

## 2. Model and host ownership

### Model-owned

- each selected island's start and end;
- whether an island participates in the graph;
- its local relation kind and argument slot;
- all roster/state/card/event/entry/query records;
- every event tag, operation, entity, next link, and nil decision.

### Architectural

- enumeration of bounded contiguous token-span proposals without a candidate-
  name dictionary;
- exact byte equality over spans selected by the model;
- quotient class renaming and relation-storage invariance;
- strict categorical relation/graph validation;
- the already disclosed S8 linked traversal, node-count safety bound, S7 cyclic
  compiler, and pop-insert transition.

### Forbidden at inference

- gold spans, names, or candidate-name dictionaries;
- board `spans`, `entities`, `positions`, `cards`, `nodes`, `execution_tags`,
  `initial_state`, `final_state`, `answer`, depth, or source event order;
- host alias resolution, graph repair, retry/search, state transition, or answer
  selection.

The host may tokenize source, enumerate all spans up to the preregistered maximum
width, compare the exact bytes of model-selected spans, validate the emitted
relation graph, and invoke the frozen runtime once.

## 3. CPU theorem and result

Source seed `792451398761220486` runs the representation over all 2,048 already-
closed S8.1 development sources. Frozen labeled spans stand in only for future
model emissions; the compiler function receives source and emitted spans, not
the row's structured graph fields. Expected graph/state/answer are scorer-only.

Report SHA-256:
`f77dce825314cc38b0630cd574b450284c00fc8afa23dc0ab39cfc5be8ef2c94`

| CPU arm | Exact graph | Exact state | Valid/rejected |
|---|---:|---:|---:|
| Oracle-emitted quotient | 2,048/2,048 | 2,048/2,048 | 2,048 valid |
| Class-ID reindex | 2,048/2,048 | 2,048/2,048 | 2,048 valid |
| Relation-storage reindex | 2,048/2,048 | 2,048/2,048 | 2,048 valid |
| Swapped card witnesses | 0/2,048 | 30/2,048 | 2,048 valid |
| Reversed links | 0/2,048 | 154/2,048 | 2,048 valid |
| Split repeated operation | 0 | 0 | 2,048 rejected |
| Merge two entity classes | 0 | 0 | 2,048 rejected |
| Unique free word per occurrence | 0 | 0 | 2,048 rejected |
| Corrupt relation kind | 0 | 0 | 2,048 rejected |
| Swap event argument slots | 0 | 0 | 2,048 rejected |

All 13 frozen CPU gates pass. This proves representational sufficiency and
causal dependence only. It does not show that Shohin can emit the quotient.

## 4. Neural architecture freeze requirements

Before any board seed is drawn, source and tests must freeze:

- a frozen Shohin residual extractor;
- a trainable contextual encoder;
- a bounded span proposal scorer that cannot inspect board span labels at
  inference;
- exact source-byte class grouping only after model selection;
- a class-aware relation/slot decoder;
- a shuffled-relation control with equal architecture and updates;
- the complete parameter count below 150,000,000;
- fail-fast checkpoint/base/tokenizer/board hashes;
- an evaluator that records selected-span, class, relation, graph, state, answer,
  depth, renderer, and causal-control scores; and
- access counters fixed at zero before the board and incremented once only by
  the sole development evaluation.

Teacher forcing may label span proposals and relation slots in training. It may
not provide final state, answer, recurrent trace, or development/confirmation
relation records. Development and confirmation names, renderers, and laws must
remain disjoint from training.

## 5. Fresh-board controls

The future board must preserve S8.1's counts and scientific exclusions unless a
new source commit preregisters a smaller infrastructure canary:

- 48,000 graph-field-only training rows;
- 2,048 development rows;
- 2,048 sealed-confirmation rows;
- 23 successor cells and three zero anchors only for the S7 generator;
- depths one through eight in training and three through eight in scoring;
- split-disjoint names, laws, and renderer families;
- zero exact prompt, 13-gram, and split-name overlap;
- no train final states or answers; and
- no development or confirmation access before source, assessor, thresholds,
  and seeds are frozen.

Required controls are gold quotient, gold graph, S8.1 token-role parser, a
same-parameter span model with equality-class messages disabled, shuffled
relation labels, class-ID reindex, relation-storage reindex, operation-nonce
rotation, split-reference, merged-class, swapped-witness, reversed-link,
state-reset, and early-nil interventions.

## 6. Immutable development gates

The sole development read advances S9 only if every gate passes:

1. selected-span F1 at least 98%;
2. class membership exact at least 95%;
3. complete relation tuple exact at least 90%;
4. valid graph at least 90%;
5. exact graph at least 85%;
6. recurrent state at least 80%;
7. answer at least 85%;
8. every depth-three-through-eight state at least 70%;
9. at least +20 percentage points exact graph over frozen S8.1's 25.098%;
10. at least +5 points exact graph over the same-parameter no-class-message
    control;
11. shuffled relations below 10% exact graph;
12. class and relation-storage reindexing bit-identical on every valid graph;
13. operation nonce recoding bit-identical on every originally valid graph;
14. swapped witnesses, reversed links, reset, and early nil each cause their
    preregistered causal drop;
15. split/merge corruptions are rejected rather than repaired;
16. complete system below 150,000,000 parameters; and
17. one development access and zero confirmation accesses.

Failure closes this architecture/version. Passing authorizes one separately
frozen confirmation evaluator with unchanged weights. It does not by itself
establish alias resolution, unconstrained language grounding, arbitrary
algebra, unbounded planning, or broad native reasoning.

## 7. Honest free-word boundary

Exact occurrence quotienting helps only when references share exact surface
bytes. The free-word control makes every occurrence unique and correctly causes
the current compiler to abstain on 2,048/2,048 cases. S9 must never describe
this as general coreference. A future alias-capable phase would need a distinct,
causally tested learned equivalence relation and a negative set containing
similar-but-nonidentical distractors.

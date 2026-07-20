# R12 S9.1 Alpha-Closed Structured Compiler Preregistration

**Status:** CPU mechanics admitted; no fresh board seed drawn

**Parent:** S9 development rejected at 1,941/2,048 exact graphs (94.775%)
because five additional exact-class rows were required and 18 originally valid
operation recodes failed to remain valid. Confirmation remains sealed.

## Claim under test

S9 already learned the central occurrence quotient. S9.1 tests whether its
remaining failures are a narrow alpha-closure and typed-assignment defect:

1. explicitly train structural role logits to agree when operation names are
   rotated in source text; and
2. replace independent greedy child selection with one deterministic,
   model-logit-only assignment under the frozen card/event grammar.

This is not a new arithmetic runtime, search procedure, or answer repair.

## Frozen ownership boundary

The model still owns every roster/state island, every card and event anchor,
the number of cards and events, all child scores, entry, next/nil, and query.
The host may enforce only non-overlap and the existing arities: two children per
selected card anchor, three per selected event anchor, one entry, and one query.
The decoder receives no graph, depth, executor output, state, answer, gold
span, retry signal, or semantic consistency score.

For each forced child role, the assignment enumerates the eight highest
role-versus-none candidates in the anchor's source region and chooses the
highest-scoring non-overlapping tuple. This beam width is frozen before the
fresh board.

Uniform logits must create no anchors and therefore no graph. A deliberately
wrong high-scoring child must remain wrong or be rejected; it may not be
repaired from semantics.

## Frozen training budget

Each arm uses exactly:

- 24,000 unique training sources sampled before training from the admitted
  48,000-row pool;
- one original and one operation-recoded view per source;
- 48,000 charged views total;
- batch 64 views (32 alpha pairs), 750 optimizer updates;
- 128 sampled negative span proposals per view; and
- ordinary weighted role cross-entropy plus 0.25 mean-squared error between
  aligned original/recoded gold-occurrence log-softmax vectors.

Treatment, no-class, and shuffled-label arms have identical architecture,
initialization, charged views, update count, and orbit objective. Shuffled
labels use the same permutation in both views of a pair.

## Frozen architecture

The frozen 300k Shohin trunk, closed S8.1 initializer, five-layer width-384 S9
encoder, exact-byte class mean, relation head, width-four token proposals, S7
generator, and S8 graph/runtime are unchanged. S9.1 adds no trainable
parameters. The complete system must remain below 150,000,000 parameters.

## Required pre-board mechanics

Before a fresh board seed is drawn, CPU tests must establish:

- oracle logits reconstruct every closed S9 development graph;
- lowering one required child's role below `none` breaks old greedy decoding
  but is recovered by structured assignment;
- uniform logits produce zero valid graphs;
- a wrong high-scoring child is not semantically repaired;
- original and recoded positive occurrences align exactly;
- every recoded gold span remains within the width-four proposal cap; and
- syntax leaves multiple candidate assignments rather than uniquely revealing
  the gold graph.

Closed S9 scores are used only for mechanics and are never rescored as S9.1.

The frozen CPU falsifier ran over all 2,048 closed S9 development sources with
seed `917431867`. All nine gates passed. Its report SHA-256 is
`a43824595c513226f52f54a629bad5d52f0d7f3c2a67e672103d0a16284dc563`.

## Sole fresh-development read and gates

S9.1 retains every immutable S9 absolute, attribution, causal, resource, and
access gate. In addition:

- structured exact graph must be at least 95%;
- structured exact graph must not regress below closed S9's 94.775%;
- unconstrained decoding is reported as an ablation;
- shuffled, uniform, and lexical-source-free structured controls are reported;
- every originally valid graph must have a valid recoded counterpart; and
- canonical graph, recurrent state, and answer must be identical on every such
  operation recode.

Only a full development pass authorizes one separately frozen confirmation
read with unchanged weights. Failure closes S9.1 without opening confirmation.

## Honest boundary

Passing would establish robust exact-surface alpha closure for this bounded
reasoning language. It would not establish learned aliasing, free-form natural
language understanding, arbitrary algebra, unbounded planning, or general
reasoning. Those remain separate stages with separate falsifiers.

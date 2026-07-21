# R12 ER-TT Addressed Marginal Route Preregistration

## Status

Pre-freeze, train-only architectural repair. It may read only the existing
ER-TT `train.jsonl`. Development and confirmation remain forbidden.

## Frozen predecessor result

Marginal-route v1.1 job `694909` is rejected because complete witness-pointer
rows reached `7,194/8,000 = 89.925%` against the frozen `90%` gate. No threshold
is relaxed. The same run reached `90.9375%` packet/joint, `97.0625%` state,
`98.5375%` answer, `90.9375%` relation rows, 100% alpha invariance, and 100%
oracle-route identity transport.

Independent reconstruction shows all 806 witness-pointer failures contain
exactly one wrong occurrence. Most are adjacent-source ambiguities in late
after-witness positions. The identity bus and recurrent relation executor are
not the observed failure.

## Hypothesis

The structural router encodes opaque candidate locations through byte-level
token memory and absolute line positions. That leaves neighboring occurrences
with weakly separated route keys, especially in longer fourth-rule records.

Add an identity-free address channel with two learned components:

1. ordinal index of each opaque occurrence within its physical record; and
2. total opaque-occurrence count for that record.

The original six bytes remain available only to the exact equality `what`
stream after routing. Addresses are computed only from the alpha-invariant
opaque-start mask, so renaming any opaque symbol cannot change them. The model
still learns every route; no target span, outcome, executor state, answer, or
scored split is used at inference.

## Architecture and budget

- Parent: reconstructed confirmed witness-equality lineage, never a rejected
  canary checkpoint.
- New parameters: two `14 x 384` embeddings (`10,752` parameters).
- Zero learned motor parameters.
- Zero learned reader parameters.
- Expected complete system: `185,543,048` parameters.
- Expected trainable parameters: `11,140,256`.
- Headroom below the absolute 200M ceiling: `14,456,952`.

## Data and optimization

Use the existing 48,000-row ER-TT training split only. A new post-commit seed
deterministically chooses 10,000 fit families and 2,000 disjoint probe families,
four renderer views each. Preserve the v1.1 budget exactly:

- two epochs;
- 2,500 updates;
- 32 rows/update from eight complete families;
- AdamW, LR `2e-4`, 100-step warmup, cosine decay;
- no state, trajectory, answer, development, or confirmation supervision.

## Frozen gates

All must pass on the one train-only probe read:

- packet/state/answer/joint each at least 85%;
- complete relation rows at least 90%;
- complete witness-pointer rows at least 90%;
- events and HALT each at least 95%;
- minimum cardinality-specific joint at least 75%;
- all hard outputs exactly invariant on 8,000/8,000 neutral-namespace alpha
  recodes;
- source-span oracle-route initial/relation/event/joint exactly 8,000/8,000
  through the same equality operator;
- exact parameter certificate below 200M;
- unchanged confirmed parent; and
- train-only/development/confirmation custody exactly `1/0/0`.

Failure closes this addressed route before any fresh board. Passing authorizes
only a fresh-board development experiment. It does not establish broad,
natural-language, or unrestricted reasoning.

# R12 ER-CST Fresh Board Preregistration

**Protocol:** `R12-ER-CST-v1.2`

**Status:** builder and audits locally admitted before source freeze. No scientific
board seed, board bytes, training seed, H100 job, output, development access, or
confirmation access exists.

## 1. Purpose

This board tests whether the 192,421,936-parameter ER-CST v1.2 system can infer
episode-local operation semantics from determining witnesses and recurrently reuse
those inferred cards. It does not reuse any closed scored row, operation name,
entity name, witness symbol, or renderer composition.

## 2. Fixed split

| Split | Latent families | Renderer views | Rows | Fit visibility |
|---|---:|---:|---:|---|
| train | 12,000 | 4 | 48,000 | compiler fields only |
| development | 512 | 4 | 2,048 | sealed scorer oracle |
| confirmation | 512 | 4 | 2,048 | mode `0600`, unopened |

Every family contains three distinct opaque operations, three entities, nine
determining witness symbols, an initial entity order, a depth from one through
eight, nine event slots, one query, and exactly one explicit HALT immediately after
the active prefix. Records after HALT contain valid but inactive opcode invocations.

Each operation card is one of the six `S_3` permutations. One before/after witness
with three distinct symbols uniquely determines each card. The three cards are
distinct inside a family and balanced across each split. Depths are exactly balanced;
queries and card identities differ by at most one latent family.

## 3. Renderer orbit

The source has four binary factors: declaration, witness, event, and query grammar.
Training uses the four even compositions `0000`, `0011`, `1100`, and `1111`.
Scored splits use the disjoint odd coset `1000`, `1011`, `0100`, and `0111`.
Every factor value appears equally in both sets, but no full composition crosses
from training to scoring.

The thirteen semantic records are independently shuffled into physical storage for
every renderer view. The compiler must emit one declaration role, three ordered rule
roles, and nine ordered event roles. Complete physical programs must remain within
512 bytes and each line within the inherited 144-byte local window. Queries are
separate single-line sources.

## 4. Training information boundary

Training JSONL may contain only:

- physical role and line ranges;
- three entity bindings and initial-order occurrence ranges;
- initial six-state category;
- three rule-card categories;
- nine opcode-card references and HALT labels;
- one categorical query and query pointer; and
- renderer/family identifiers needed for batching and matched controls.

Training rows contain no `oracle`, final state, answer, trajectory, executor output,
correctness signal, retry target, development statistic, or confirmation byte.
Development and confirmation oracles reside only in their respective files.

## 5. Admission audit

Before writing any scientific board, the pure builder and independent grammar parser
must establish:

1. exact 48,000/2,048/2,048 rows and 12,000/512/512 complete four-view families;
2. all 52,096 rows round-trip through the production grammar, determining-witness
   inference, categorical executor, query, and answer oracle;
3. exactly one HALT, depth one through eight, and nine-state trajectory at depth eight;
4. zero train oracle fields and complete scored oracle fields;
5. all programs at most 512 bytes and all physical lines at most 144 bytes;
6. balanced depth, card, and query distributions;
7. globally unique opaque names by latent family and zero cross-split name overlap;
8. zero exact prompt, raw word-13-gram, latent-family, and renderer-composition overlap;
9. family-deranged card state exactness below 40%;
10. byte-identical full rebuild from the same source and seed;
11. confirmation mode exactly `0600`; and
12. development/confirmation access exactly `0/0`.

Any failed gate exits before output creation. A malformed line, repeated/missing HALT,
wrong cardinality, invalid witness, unknown operation, or overlength source is rejected,
not repaired.

## 6. Neural controls and gates

The treatment, family-deranged-card arm, and equality-ablated-witness arm must use
identical architecture, initialization, optimizer, update count, batch order, and
compute. Their only differences are the preregistered card labels or witness-identity
relation. All inherited excluded tensors must remain byte-identical.

The immutable development thresholds and causal controls remain those in
`R12_ER_CST_EPISODIC_RULE_CARD_THEORY.md`. Passing all gates authorizes one separately
committed confirmation evaluator. Failure closes the board without rescore or
threshold repair.

## 7. Ordered custody

1. Commit and push the exact v1.2 architecture, builder, parser, tests, and this
   preregistration.
2. Draw one signed-safe board seed after that commit.
3. Build, audit, write, seal, and independently rebuild the board; hash every file.
4. Commit the board receipt before drawing a training seed.
5. Freeze training/evaluation/assessment source and submit one development job.

No board seed may be chosen because it yields favorable model behavior. Rebuilds may
verify bytes only; they may not alter the admitted board.

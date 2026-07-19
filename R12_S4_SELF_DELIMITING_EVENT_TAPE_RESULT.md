# R12 S4 Self-Delimiting Event Tape Result

## Decision

`REJECT_S4_V1_AND_V1_1`; retain one causal discovery. The treatment learns exact autonomous event
count on every public-development source and, whenever it emits a structurally valid tape, the
entire program/state/answer is exact. It misses the frozen 95% overall program gate by 14 rows.
The zero-fit pointer-anchored v1.1 repair collapses on variable-width referential boundaries and is
also rejected. No confirmation board was generated or read.

## Frozen objects

- Immutable base: 125,081,664 parameters at the protected 300k checkpoint.
- Parser adapter: 8,608,271 parameters; total 133,689,935, below 150M.
- Training: 48,000 whole-source rows, depths 1--4, 120,000 events, one epoch / 750 updates.
- Development: 2,048 matched rows, depths 3--8, zero train/development exact, 13-gram, name, or
  factor overlap.
- Treatment training job `693152`; shuffled-label control `693154`.
- Original evaluations `693153` and `693155`; diagnostics `693157` and `693158`.
- Pointer-anchored evaluations `693160` and `693161`.

## Original autonomous result

| Arm/control | Count | Exact program/state | Answer | Valid tapes |
|---|---:|---:|---:|---:|
| Treatment strict | **2048/2048 = 100%** | **1932/2048 = 94.336%** | **1932/2048 = 94.336%** | 1932/2048 |
| Treatment gold count | 2048/2048 | 1938/2048 = 94.629% | 1939/2048 = 94.678% | 1940/2048 |
| Treatment gold intro/query | 2048/2048 | **1991/2048 = 97.217%** | same exact-consumption boundary | 1991/2048 |
| Shuffled strict | 26/2048 = 1.270% | **0/2048** | **0/2048** | 0/2048 |

All 1,932 valid strict tapes have the exact program, state, and answer. Strict treatment program by
depth is 97.093%, 94.767%, 94.706%, 94.118%, 90.294%, and 95.000% for depths 3--8. The failures
are 66 intro-cardinality, 47 event-component-cardinality, and three entity-identity errors. Gold
intro/query boundaries raise every depth to at least 96.471%. This supports learned variable event
count and event semantics, while localizing the remaining miss to source-span emission.

## Frozen zero-fit repair and rejection

Source/prereg commit `34657ea` and corrected training-width receipt commit `36f06ed` precede any
v1.1 score. The first training-only lexicon build failed closed because it incorrectly required one
entity width; contextual BPE spans are widths 4/5/6. Its rejected SHA-256 is
`f487d1cb98bebd84137c1b0b7839e2241603cc4f920f4f1a09205f502e9015e6`. The lawful set-valued
receipt passes all gates at SHA-256
`eb49f75d969c999d4bcb8f2e350658f76a5491d4a91a7e4abff83b086ba4fd38`.

The pointer-anchored treatment preserves exact count at 2048/2048 but falls to 25/2048 = 1.221%
exact programs and 300/2048 = 14.648% answers; shuffled remains 0/2048 programs. Treatment has
1,176 `event_entity` failures and only 306/2048 exact initial rosters. A global token-role maximum
cannot determine the correct 4/5/6-token boundary, and shared event-role scores do not pair each
direction anchor with its own entity as depth grows. Assessment SHA-256
`fd0479b0737af49313b0cebf1863c4826c21de336f51e240ece3e4d60d11d587` records
`reject_s4_v1_1_public_development`.

## What survives

S4 v1 is the first whole-source result here to recover the exact number of complete known-atom
events on all 2,048 held-out rows without padding, host count, or hidden `active_operations`.
Shuffling supervision destroys the result. The locked S3 executor is not the bottleneck once a tape
is valid. The open interface is now narrower: variable-width start/end binding and event-relative
argument pairing.

## Next admissible experiment

Do not tune another deterministic decoder on this public board. S4 v2 must replace shared token-role
segmentation with learned start/end pointers for the three roster entries and event-relative pointer
queries conditioned on each model-found direction anchor. It must retain source-order count, train
only on depths 1--4, and evaluate once on a newly frozen, disjoint development board through depth
eight. A favorable parameter-matched shared-role parser and shuffled-label arm remain mandatory.

## Claim boundary

This is evidence for bounded known-atom schedule counting and conditional tape execution, not a
confirmed autonomous parser, semantic halt, unseen action semantics, planning, open-language
reasoning, benchmark improvement, or architectural novelty.

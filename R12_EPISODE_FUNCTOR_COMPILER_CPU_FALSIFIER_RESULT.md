# R12 EPISODE Functor Compiler CPU Falsifier Result

**Status:** mechanics-only corrective result. No neural fit, source freeze, new
board seed, development read, GPU job, capability claim, or continuation
pretraining is authorized.

**Date:** 2026-07-23

**Theory draft audited:** SHA-256
`e3c7420fd7aef36834cee79af58afe681359e1cbf5ca35a1ad855d14bfcabd36`

**Frontier commentary audited:** SHA-256
`a83536547b121d000cd8c28d9ce4beb059a661f49a294eb6492d7db0e61e3531`

**Reproducible audit command:**

```text
python3 -m pipeline.episode_functor_compiler_falsifiers
```

**Deterministic report SHA-256:**
`95b1157ca7f017826bf689430c6b00cfe9e56fb36551437d829fd9541b5881fb`

## Decision

The Episodic Functor Compiler is a promising architecture family, but
`R12_EPISODE_FUNCTOR_COMPILER_THEORY_DRAFT.md` is **NO-GO AS WRITTEN** for two
independent reasons:

1. its finite-query theorem is correct, but its application to current EPISODE
   substitutes the two sampled score rows for the post-seal query support; and
2. its committed schema has an `initial_state` but no retained opaque
   `state_key` records, even though the current late query supplies an opaque
   start-state token that must be bound after source deletion.

The old frozen transformer workspace remains a control architecture. The EFC
candidate remains open only after these two specification defects are repaired.

## 1. What Was Implemented

`pipeline/episode_functor_compiler_falsifiers.py` now provides:

- a hard anonymous categorical EPISODE machine;
- explicit retained state keys, action keys, transition tables, and observer
  maps;
- a parser for source-only committed worlds;
- a separate parser for late queries;
- hard ordered execution;
- exact Moore-machine causal-quotient refinement;
- shortest separating-word search;
- binding/key-only, operator-only, compensated key/operator, and local
  transition-row interventions;
- exact machine-versus-answer-table resource accounting;
- SHA-256 verification of every consumed corpus and custody JSONL against its
  sealed manifest;
- a lawful world-only two-entry cache;
- a deliberately leaky cache receiving hidden query identities and assessor
  answers; and
- an audit over the complete already-consumed 1,920-packet EPISODE corpus and
  its physically separated development custody artifacts.

The implementation is CPU-only and does not modify the frozen neural source.

## 2. Test Result

The two separately implemented Python CPU-audit suites and inherited EPISODE
mechanics tests pass after reconciliation:

```text
49 passed in 2.05s
```

They are **not** admitted as two independent deployed runtimes. Both use sorted
opaque keys, integer state slots, row-major transition tables, and the same
left-to-right update convention. Neither consumes only a sealed fixed-width
machine artifact. A shared representation or composition-order defect could
therefore survive agreement between them. The required independent C and Rust
runtimes plus a third assessor are specified in
`R12_EPISODE_FUNCTOR_COMPILER_SEAL_FIRST_PROTOCOL.md`.

The independent identifiability auditor source hashes to
`e3a55ae0b5a6d17fa0cd9fb9152c5ac2bea456fa3b87df2a2cbc9fd1827d6af5`
and independently reports 1,677,312 exhaustive nonempty depth-at-most-six
executions over the 192 custody development worlds.

The focused tests cover:

- both hidden query orders from one committed world;
- the two-entry cache custody distinction;
- exact query-support counting;
- state/action key and semantic-bit accounting;
- causal-quotient refinement;
- nontrivial future separators;
- independent key and operator interventions;
- compensated intervention invariance;
- all six action-record permutations;
- one-row transition transplant locality; and
- failure to bind the current query start without retained state keys.

## 3. Full Frozen-Corpus Mechanics

The categorical machine reconstructed from model-visible demonstrations
executes:

```text
1,920 / 1,920 packets exact
960 unique committed worlds
8 / 8 causal-quotient classes in every world
28 / 28 state pairs separated
```

The shortest separator has depth zero because the current identity observer
already assigns a different opaque output to every state. If empty observation
is excluded, any bijective action followed by the identity observer still
separates every distinct state pair at depth one.

This is an oracle mechanics result. It proves that the visible source is
sufficient to construct the machine; it does not prove that Shohin can learn
the compilation from bytes.

## 4. The Two-Answer Cache Does Not Cross Current Custody

Current physical custody declares:

```text
compiler:  development_worlds.jsonl
executor:  development_queries.jsonl
assessor:  development_assessor.jsonl
```

There are 192 development worlds and 384 sampled query rows, hence two scored
queries per committed world. But a compiler does not receive the identities of
those two queries.

For the current interface, a late query contains:

```text
one of 8 opaque start states
an action word over 3 opaque actions
depth 1 through 6
```

Therefore the post-seal query support per world is:

```text
8 * sum_(depth=1)^6 3^depth = 8,736 queries.
```

The audit obtains:

```text
lawful canonical world-only two-entry cache coverage: 0 / 384
leaky hidden-query-plus-target cache exactness:       384 / 384
```

The leaky construction is exactly the finite-query-cache theorem instantiated
with the two realized rows. It requires both
`development_queries.jsonl` and `development_assessor.jsonl` before sealing, so
it violates the existing process boundary.

For a fixed pair of cache keys and a uniformly sampled hidden pair from the
declared support, the probability of matching both is:

```text
1 / choose(8,736, 2) = 2.620924200775374e-08.
```

This probability is descriptive, not a claim that the current generator
samples uniformly from all unordered query pairs. The decisive point is
logical: the draft's construction needs the identities of `q_1` and `q_2`,
while the compiler input does not contain them.

### Correct theorem boundary

The finite-query theorem applies to the complete query support known at compile
time, or to realized future queries if their identities leak before the seal.
It does not reduce the committed state to two answers merely because the
assessor later samples two rows.

The old board can still be underidentified for other reasons, including a
finite challenge support, generous continuous state capacity, or exploitable
generator correlations. Those are legitimate falsifier targets. They are not
established by the two-answer construction in the supplied draft.

## 5. Exact Resource Receipt

For the current board with eight states, three actions, one identity observer,
depth at most six, and opaque token IDs below 32,768:

| Object | Semantic bits |
|---|---:|
| complete query-indexed answer table | 26,208 |
| transition destinations | 72 |
| identity observer table | 24 |
| eight retained opaque state keys | 120 |
| three retained opaque action keys | 45 |
| complete explicit machine fields above | 261 |

The answer-table count includes all 8,736 supported queries at three answer
bits each. The machine count includes the opaque start-state keys omitted from
the draft. It excludes schema, masks, fixed framing, precision receipts, and
cryptographic custody metadata; those must be counted in a later byte-level
preregistration.

The independent auditor reports a conservative 276 semantic bits when the
draft's initial-state field and twelve active-mask bits are also retained.
Thus 261 is the current-interface minimum counted above, while 276 is the
conservative provisional EFC schema count. Neither substitutes for an exact
serialized-byte receipt.

Every current action is a permutation of eight states. An information-
theoretically compressed arbitrary triple of such permutations therefore needs
only `ceil(log2((8!)^3)) = 46` transition bits. The 72-bit transition table is
an explicit intervention-friendly representation, not a minimum code.

For a sparse `k`-entry cache over 8,736 canonical queries, the favorable lower
bound `ceil(log2(choose(8736,k))) + 3k` permits at most 20 entries under 261
bits and 21 under 276 bits. Under uniformly sampled all-start challenges, a
default-answer cache with those exceptions is bounded near 12.73% because
permutation actions make outputs exactly balanced. These bounds exclude only
that cache family. An unrestricted decoder that stores and composes the action
generators is already functionally a transition machine.

This is a strong resource argument for explicit transitions, but not a proof
that an unconstrained real-valued workspace cannot encode the table. A future
comparison must freeze precision and committed bytes, not count tensor
coordinates alone.

## 6. Intervention Result

On one deterministic current-EPISODE world, all 960 start/word combinations
through depth four were audited:

```text
key-only intervention changed:      840 / 960
operator-only intervention changed: 840 / 960
compensated intervention changed:     0 / 960
```

All six action-record permutations preserve behavior when keys and transitions
are permuted together. Independent key and transition permutations are
nontrivial. Local transition-row transplantation changes the selected
state/action edge while leaving all other one-step edges unchanged.

These mechanics provide the clean causal intervention interface that the
frozen four-slot workspace lacks.

## 7. Missing Start-State Binding

Current late queries have the form:

```text
QUERY <opaque-start-state> <opaque-action-word> ANSWER EOS
```

The draft's hard machine retains:

- one `initial_state`;
- action keys and transitions; and
- observer keys and outputs.

It does not retain opaque state keys. Consequently its post-seal parser cannot
map `<opaque-start-state>` to an anonymous causal-state index.

One of two repairs is mandatory:

1. retain a fixed-shape `state_key[K,d_key]` field and let the late parser
   select the start state; or
2. redesign the board so the source fixes the initial state and the late query
   never supplies a state referent.

The two designs test different capabilities and cannot be interchanged after a
score.

## 8. Corrected Next Sequence

No neural fitting is authorized. The next lawful sequence is:

1. revise the EFC theory to distinguish sampled rows from query support;
2. choose and freeze either retained state-key binding or a source-fixed
   initial state;
3. define the exact challenge distribution and prove that query bytes and seed
   are unavailable before sealing;
4. freeze field precision and compare complete committed bytes against matched
   answer-cache and generic recurrent controls;
5. retain the hard categorical runtime and intervention suite implemented here;
6. add exhaustive dual-oracle STOP, observer, equivalent-word, noncommuting,
   and transition-row audits for the revised schema;
7. only then design a fresh mechanics split and expanded board; and
8. only after every CPU gate passes instantiate a neural compiler below the
   strict 200M complete-system ceiling.

The frozen `80dc07a` workspace is retained as a favorable control and custody
reference. The current old board remains a useful action-binding,
order-sensitivity, and source-deletion diagnostic. Neither is an advancement
claim.

## 9. Hostile Seal Audit

The existing corpus is rejected for an advancement claim even though its
physical files are separated:

- world mechanics, opaque keys, demonstration order, and hidden query
  coordinates share one deterministic PRNG trajectory;
- candidate-world acceptance inspects sampled hidden query outcomes; and
- custody files are split only after complete packets already contain world,
  query, and answer information.

This proves file separation, not temporal nonexistence of future challenges.
The corrected protocol requires a world beacon before world generation and an
independent challenge beacon only after external publication of `machine_root`.

The old depth-six cache receipt is also too weak as a byte-capacity exclusion.
A complete nonempty depth-one-through-six answer table is 26,208 bits, or 3,276
packed bytes, so it fits inside a hypothetical 16 KiB machine budget. A
depth-zero-through-twelve support has 6,377,288 queries and needs 2,391,483
packed answer bytes, approximately 146 times that budget. Exact depth support,
byte length, precision, and state-key fields must be frozen before any future
seal.

## 10. Corrected CPU Gate Implementation

The corrected mechanics and protocol suite now passes:

```text
121 passed in 14.37s
```

This includes:

- 49 current-board, independent-audit, and inherited corpus tests;
- 47 strict C/Rust fixed-wire mutation and cross-runtime tests;
- 16 two-beacon seal-order, immutability, taint, source-poison, and independent-
  assessor tests; and
- nine nontrivial-observer quotient-board tests.

The standalone C runtime uses typed linear key lookup and flat
`next[action*K + state]` transitions. The standalone Rust runtime independently
parses the same 1,536-byte wire record and applies Boolean-relation images to a
one-hot state bitset. Their transcripts are byte-identical and bind both exact
machine and query payload hashes. They import no Shohin runtime code.

The consumed two-beacon rehearsal compiles exactly once and reuses identical
machine bytes under two distinct later challenge seeds. For the deterministic
rehearsal fixture:

| Receipt | Value |
|---|---|
| protocol root | `2b039882911524486eb6c5793ef0ef46824e821ade5c90402ce08849516e13e4` |
| world root | `7637102cf47900718ba5d7b65d92ef01efc29a94a8418b9e4497f3572a740e6c` |
| machine root | `77af3e77f57036ed17b78f7ef01e81bb11b06faf75e18613c40557bcb1244532` |
| challenge A coordinate root | `73d13b1fb38404038ba1cb9ffc62a8b3b2f67c478100693414f8fb1598de302d` |
| challenge B coordinate root | `c1f3d2655d7ea7ce1c2d07fe92f79b589ba5b47d21555020561403317343a129` |
| compile count | 1 |
| independently assessed answers | 200/200 |

The rehearsal's 113-byte compact big-endian machine is deliberately not the
deployed C/Rust wire format. It proves protocol phase mechanics only.

The consumed nontrivial-observer fixture contains twelve machines covering
quotient sizes three through eight. Partition refinement, pair-product
reachability, and depth-seven exhaustive future behavior agree on every
machine. All thirteen mechanics gates pass, train/development structural
signatures are disjoint, and the fixture SHA-256 is
`349b8f4c4a163afd5eab288727c7bd59a1e52fb6782d25071820fac1151c678d`.

These results establish a credible CPU contract for a future compiler. They do
not show that Shohin can compile the machine from raw evidence. No neural
weights were fitted, no GPU was used, no official board was opened, and no
reasoning or continuation-pretraining claim is made.

# R12 SD-CST Complete Physical-Record Front-End Preregistration

**Status:** locally admitted before source freeze, seed, or H100 execution;
consumed-training mechanics only

**Parent compiler:** retained independent-assignment Physical-Record Write Bus,
checkpoint SHA-256
`89ab7d7417918e72da60028e6d5936908a3ee29c0981f5fdac9dc385c3099419`

**Claim class:** local-front-end completion gate; no fresh generalization,
native reasoning, architecture novelty, or Shohin promotion claim

## 1. Why this gate is necessary

Job `694136` establishes perfect physical-record event compilation on 48,000
fit and 8,000 held-out consumed renderer views. However, that version still
delegates declaration/initial-state and late-query parsing to the old frozen
global parent. A fresh renderer board could therefore fail because inherited
global interfaces do not recognize new declaration or query language, rather
than because physical-record factorization fails.

The fresh-board candidate must be source-facing through one coherent local
contract before any scored bytes are generated or read.

## 2. Fixed architecture

The retained independent-assignment record bus remains byte-identical and
frozen. Ten new `local_*` tensors add:

1. six declaration-local pointer queries: three canonical binding roles and
   three initial-order occurrences;
2. one declaration-query projection over the already-trained local token
   memory;
3. one local late-query selector with query/key projections;
4. one position-free raw-byte value projection; and
5. one normalization and three-class query head.

Declaration pointers are computed inside every physical record and mixed only
by the model's retained declaration-role probability. Their content
fingerprints feed the frozen matcher and six-way initial permutation scorer.
The query is encoded as one bounded local record with the shared retained line
encoder; contextual state may choose a byte, but the value classifier receives
only the selected raw-byte value projection.

The complete compiler does not call the inherited global source encoder or
orbit encoder for either program or query compilation. A test replaces both
methods with raising sentinels and the complete program/query forward still
passes.

## 3. Frozen ownership and parameter certificate

Only the ten `local_*` names train in this gate. The 88 retained `record_*`
tensors, joint parent, fingerprint matcher, categorical tape/executor, motor,
reader, and Shohin trunk remain frozen under one excluded-state digest.

| Quantity | Exact count |
|---|---:|
| immutable Shohin trunk | 125,081,664 |
| complete compiler | 66,426,124 |
| new trainable completion parameters | 594,435 |
| all local-front-end parameters, frozen plus new | 11,701,265 |
| categorical motor | 19,206 |
| categorical reader | 835 |
| **complete deployed system** | **191,527,829** |
| **strict-200M headroom** | **8,472,171** |

The historical 150M contracts remain immutable. No parameter may be added to
this gate after source freeze.

## 4. Fixed data and optimization

The gate reads only the already-consumed projected-v2 training JSONL SHA-256
`b7756dbf8d4401dbc5fb897dee53f68758e27200b1ce0d2387631f2f0205ec25`.
It reuses the fixed 12,000-semantic even-parity fit and disjoint 2,000-semantic
odd-parity heldout partitions. Development, confirmation, answer, state,
trajectory, and executor outputs remain unreachable.

Optimization is fixed to two epochs / 3,000 updates, family batch eight,
AdamW lr `2e-4`, betas `(0.9, 0.95)`, weight decay `0.01`, 100-update warmup,
cosine decay, gradient clipping `1.0`, and renderer consistency weight `1.0`.
Every inherited event loss remains in the objective but cannot update frozen
record tensors. Initial heldout metrics are recorded before optimization.

## 5. Frozen gates

The gate passes only if:

1. minimum fit-renderer complete packet is at least 99%;
2. minimum heldout-renderer complete packet is at least 95%;
3. heldout initial state is at least 95%;
4. heldout query and query pointer are each at least 99%;
5. heldout declaration and initial-occurrence pointers are each at least 99%;
6. heldout event pointer is at least 99%;
7. heldout kind, identity, and amount are each at least 99%;
8. every excluded tensor is byte-identical;
9. complete deployed size is strictly below 200M; and
10. scored access is `0/0`.

All gates passing yields
`retain_complete_local_front_end_for_fresh_board`. Any failure yields
`reject_or_revise_complete_local_front_end`. No threshold, parameter, epoch,
or same-output retry may change after seed.

## 6. Honest boundary and next phase

A pass means only that every source-facing field can be compiled through
bounded local evidence on already-consumed renderer factors. It does not test
new language, names, distributions, recurrent state, answers, or reasoning.

Only a pass authorizes a separately committed board builder with fresh names,
renderer families, train/development/sealed-confirmation bytes, access ledger,
model-logit-only compiler outputs, source deletion, and the retained fixed
executor. Existing development and confirmation remain permanently unavailable
to this architecture.

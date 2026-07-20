# R12 SD-CST Complete Physical-Record Front-End v1.1 Preregistration

**Status:** closed and rejected under the frozen gates. Sole job `694203`
completed cleanly on H100 `evc22`; full result:
`R12_SD_CST_COMPLETE_PHYSICAL_RECORD_BUS_V1_1_RESULT.md`.

Exact scientific source `b93b17b3ee5c096509cd1ab0d903ef7a9287d3a3`
preceded raw beacon `14330060956843215829` and signed-safe seed
`5106688919988440021`. The dedicated key raises minimum binding pointer to
61.0% and packet to 47.45%, but minimum initial-occurrence pointer remains 0%.
Decision: `reject_declaration_key_repair`.

**Claim class:** consumed-training declaration-address repair only. This cannot
establish fresh-language generalization, native reasoning, or Shohin promotion.

## 1. Frozen diagnosis

Complete local-front-end v1 reaches 100% minimum held-out query, query pointer,
line pointer, event pointer, kind, and amount, but only 5.15% binding pointer,
0% initial-occurrence pointer, 15.0% initial state, and 13.45% complete packet.
Fit is nearly identical. Its declaration queries and projection move and retain
large gradients, while their address losses remain near uniform.

V1 addresses six declaration occurrences through `record_entity_key`, a frozen
projection learned for event-line entity extraction. V1.1 tests the smallest
representation repair: a declaration-specific trainable key projection.

## 2. Fixed parent and ownership

V1.1 reconstructs the exact joint parent and retained independent physical bus,
then loads only the successful eight-tensor `local_query_*` endpoint from v1
checkpoint SHA-256
`30b75305031b1e2f67a24f98b4907d2d65bc847310ea406d25f34c7b9611e1b4`.
The failed declaration query table and query projection are reset from the new
post-source-commit seed. One new bias-free 384 by 384 declaration-key projection
is initialized from the same seed.

Only these three tensors train:

1. `local_declaration_queries`;
2. `local_declaration_query_projection.weight`; and
3. `local_declaration_key_projection.weight`.

The successful query path, 88-tensor physical record bus, joint parent,
fingerprint matcher, tape, executor, motor, reader, and Shohin trunk remain
frozen under one excluded-state digest.

## 3. Exact parameter contract

| Quantity | Exact count |
|---|---:|
| immutable Shohin trunk | 125,081,664 |
| complete compiler | 66,573,580 |
| trainable declaration repair | 297,216 |
| categorical motor | 19,206 |
| categorical reader | 835 |
| **complete deployed system** | **191,675,285** |
| **strict-200M headroom** | **8,324,715** |

No parameter may be added after the scientific source commit. Historical 150M
contracts remain unchanged.

## 4. Fixed data, optimization, and gates

The pilot reads only consumed train SHA-256
`b7756dbf8d4401dbc5fb897dee53f68758e27200b1ce0d2387631f2f0205ec25`
with the same fixed 12,000-semantic fit and disjoint 2,000-semantic heldout
partition, four renderer views each, two epochs / 3,000 updates, family batch
eight, AdamW lr `2e-4`, betas `(0.9, 0.95)`, weight decay `0.01`, 100-update
warmup, cosine decay, clipping `1.0`, and renderer-consistency weight `1.0`.
Development, confirmation, answers, states, trajectories, and executor feedback
remain unreachable.

The frozen v1 absolute gates are unchanged: minimum fit packet 99%; heldout
packet and initial state 95%; query, query pointer, binding pointer,
initial-occurrence pointer, event pointer, kind, identity, and amount 99%;
excluded state byte-identical; system strictly below 200M; scored access `0/0`.

All gates passing yields `retain_declaration_key_repair_for_fresh_board`. Any
failure yields `reject_declaration_key_repair`. A pass credits the three-tensor
repair package only; it remains conventional compiler mechanics and merely
authorizes a separately committed fresh-board contract.

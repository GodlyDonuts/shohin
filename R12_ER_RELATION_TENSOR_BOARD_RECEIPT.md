# R12 ER-TT Fresh Board Receipt

**Protocol:** `R12-ER-TT-v1-board`

**Decision:** admit the board for score-bearing source implementation. No
training seed, H100 run, development read, or confirmation read exists.

## Provenance

| Field | Value |
|---|---|
| Exact board source | `bd77c0fafdbc527688ba57aedd74ccdfbe2ed1cf` |
| Public beacon round | `6305283` |
| Beacon payload SHA-256 | `0fda3a9977fbed0805e38b16332036b0e8af6d646646078d367e4b62709c3d99` |
| Board seed | `1209366536012979338` |
| Report SHA-256 | `64ea4c0e19ea029102af240d44242c830d7b014e49a59af09a836b2d3efb6010` |

The board lives at
`artifacts/r12/er_relation_tensor_board_1209366536012979338/` locally. A second
complete build under `/tmp` was byte-identical for all four files.

## Immutable files

| File | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `train.jsonl` | 48,000 | 155,836,848 | `1982aeb272ab630472b9326149fc0e9c4f653c2e54a8e70b2b09162d0c95e734` |
| `development.jsonl` | 2,048 | 6,674,519 | `59be0c40656d5f7cedbbf5cea8e384896ed0eeb5cbed5083df4ac1281ff81090` |
| `confirmation.jsonl` | 2,048 | 6,678,600 | `cac2515bcdec2f2be2bac30162954592de0a12af4823bc1ec25cbe821030c5e6` |
| `report.json` | n/a | n/a | `64ea4c0e19ea029102af240d44242c830d7b014e49a59af09a836b2d3efb6010` |

`confirmation.jsonl` is mode `0600`. Development and confirmation access
counters are `0/0`.

## Gate result

All fifteen preregistered gates pass:

- 13,024 rows at each cardinality `N=3,4,5,6`;
- rule-count rows 17,368 / 17,368 / 17,360;
- depth rows are 4,336 or 4,344 for every depth 1–12;
- every one of 13,024 semantic families contains a non-bijective rule;
- every generated row passes independent grammar parsing, equality inference,
  execution, HALT, query, pointer-span, and oracle checks;
- pairwise names, semantic families, exact prompts, and word 13-grams are all zero;
- training and scored renderer cosets are disjoint;
- maximum program and line lengths are 610 and 96 bytes;
- family-deranged state exactness is 1,036/13,024 = 7.955%; and
- equality-ablated state exactness is 1,063/13,024 = 8.162%.

## Boundary

This receipt establishes clean, reproducible score-bearing data only. It is not
a neural result. The confirmation file remains sealed unless a separately
frozen development evaluator passes every preregistered gate and authorizes one
read.

# R12 ER-CST Fresh Board Result

**Decision:** `admit_er_cst_v1_2_board_before_training_seed`

**Scientific source:** `fba34cdc9bfab75882dee8093b07ab96042d4a07`

**Board seed:** `1686667709479653771`

## Closed predecessor

Source `c06eab3e2476d9805bf1079c698143410e25eef5` and seed
`2459068742837489615` closed before byte write because independent 32-bit name
hashes collided at full scale. The builder created no output directory. V1.1 changed
only name allocation to a seed-keyed bijection; the failed seed was not reused.

## Admitted board

| Split | Families | Views | Rows | SHA-256 |
|---|---:|---:|---:|---|
| train | 12,000 | 4 | 48,000 | `57abe77eadde0d5bb6573d8f70db73567c4ee85f6621554a20f190fb54556361` |
| development | 512 | 4 | 2,048 | `8ef7f8249a5d1b1e0383a9211b9e9a36573baf0a56eaf6dd5eae5a1fc624ac22` |
| sealed confirmation | 512 | 4 | 2,048 | `a6dc26a50784122d993b3151686b5289af97eec877521f32dab295295203a207` |

Board report SHA-256:
`5c9b5b812e55dd5d19b32262d0c4af5b4b5f1af477959ed3901dc76df98d2f13`.

The confirmation file is mode `0600`. Development and confirmation access are
exactly `0/0`.

## Audit

All thirteen admission gates pass:

- exact row and complete four-view family counts;
- 52,096/52,096 independent grammar, witness, executor, query, and oracle agreement;
- no oracle/final-state/answer/trajectory fields in training;
- complete scored oracles isolated to development and confirmation files;
- exactly one explicit HALT and exact depth-one-through-eight balance;
- exact training card/query balance and scored near-balance;
- 195,360/195,360 globally unique compact opaque names;
- zero train/development/confirmation overlap in names, exact prompts, raw word
  13-grams, and latent families;
- disjoint training/scored renderer compositions with every factor value balanced;
- maximum program 402/512 bytes and maximum line 74/144 bytes; and
- family-deranged cards retain only 2,068/13,024 = **15.878%** exact final state.

A second complete build under `/tmp` from the same exact source and seed is
byte-identical for train, development, confirmation, and report files.

## Boundary

This result admits only one training-seed draw and the separately frozen neural
development experiment. It is data integrity evidence, not a model capability or
reasoning result. No H100 job, neural output, development read, or confirmation read
exists at this point.

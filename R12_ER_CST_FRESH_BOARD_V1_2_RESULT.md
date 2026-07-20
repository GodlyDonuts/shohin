# R12 ER-CST Fresh Board v1.2 Result

**Decision:** `admit_addressed_er_cst_board_before_training_seed`

**Scientific source:** `9cf9d043d0e86a30d18c6d5e3b838c80ec054d7c`

**Board seed:** `8277659525319823840`

## Board

| Split | Families | Views | Rows | SHA-256 |
|---|---:|---:|---:|---|
| train | 12,000 | 4 | 48,000 | `b5cb2f14949b68e9f005b03d11356a24a3d75df76876aebd2da20a3d9efc2a28` |
| development | 512 | 4 | 2,048 | `5cd0395f6b171c5b61e741694ffe3f584d3d60d0795bcdfbfbc27966cfc56ef9` |
| sealed confirmation | 512 | 4 | 2,048 | `7404b2473da5d76835f4d5bf69d852b62572683cd878c6c968740421cae54189` |

Board report SHA-256:
`589b203fb4fec3c55b1b4d77efaf78d127ce127fc98633279c69b305dad2e704`.

Confirmation is mode `0600`; access is exactly `0/0`.

## Admission evidence

All thirteen immutable gates pass. The production parser requires each addressed
rule slot exactly once, every witness identifies its card, every event binds to a
known opaque opcode, and the independent executor/query oracle agrees on all 52,096
rows. Training contains no final state, answer, trajectory, or oracle field.

Depths one through eight are exactly balanced. Names are 195,360/195,360 unique.
Train/development/confirmation overlap is zero for names, exact prompts, raw word
13-grams, and latent families. Renderer compositions are disjoint while every factor
value remains balanced. Maximum program and line sizes are 405/512 and 75/144 bytes.
Family-deranged cards retain 2,033/13,024 = **15.610%** exact final states.

A second complete build from the same exact source and seed is byte-identical for all
four files.

## Boundary

The earlier unaddressed source `fba34cd` board is closed and must never be trained.
This v1.2 board admits only frozen training/evaluation source and one later training-
seed draw. It is data-integrity evidence, not neural reasoning evidence. No training
seed, H100 job, neural output, development read, or confirmation read exists.

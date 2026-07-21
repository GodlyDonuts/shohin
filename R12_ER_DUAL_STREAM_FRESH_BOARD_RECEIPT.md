# R12 ER-TT Dual-Stream Fresh Board Receipt

**Decision:** admit the board for score-bearing source implementation

**Development/confirmation custody:** `0/0`

## Provenance

| Field | Value |
|---|---|
| Exact board source | `627b6c3d97a885017041aacb5971874680e1b289` |
| Derivation label | `r12-er-dual-stream-fresh-board-v1` |
| Derivation SHA-256 | `d6b8fae41eb5cffec723437078f00abfb2913076fa3e9cbe2de08ec43c4cb6d4` |
| Raw 64-bit value | `15472392377506058238` |
| Signed-safe board seed | `6249020340651282430` |
| Report SHA-256 | `6b0a011c26c40628cb1db5547715c9f11292cba9af3a9eb10af01714df456b8f` |

The Mac production build and clean-capsule Newton CPU rebuild job `694938` are
byte-identical for every immutable file. The admitted Newton copy is
`/lustre/fs1/home/sa305415/shohin_runs/er_dual_stream_fresh_board_6249020340651282430_rebuild`.

## Immutable files

| File | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `train.jsonl` | 48,000 | 160,020,611 | `ce47a5b823c383375646f807bcd66bb9734bdffdbacca4a199c5d0cf7f703ecd` |
| `development.jsonl` | 2,048 | 6,853,013 | `086d5784fd871d6f3642d1e028466591e2241df06220493c2698ebaf3b920de7` |
| `confirmation.jsonl` | 2,048 | 6,857,347 | `508e583b2abdbf705f599678f83e3a87472c014375788b722996392c00ccc55b` |
| `report.json` | n/a | 3,438 | `6b0a011c26c40628cb1db5547715c9f11292cba9af3a9eb10af01714df456b8f` |

Newton train/development/report files are read-only. Confirmation remains mode
`0600`; the access directory is empty.

## Gate result

All preregistered board gates pass:

- exact 48,000/2,048/2,048 row counts and four views per family;
- exact or arithmetic-minimum balance over cardinality 3--6, rule count 2--4,
  and depth 1--12;
- all 13,024 semantic families contain a non-bijective relation;
- all 52,096 rows public-parse and execute exactly;
- all semantic and distractor tokens occupy one neutral `z.....` namespace;
- distractor rotation preserves state/answer on 52,096/52,096 rows;
- cross-split name/family/prompt/word-13-gram overlap is zero;
- train and scored renderer compositions are disjoint;
- maximum program/line length is 627/97 under 640/144; and
- family-deranged/equality-ablated state is 1,060/13,024 = 8.139% and
  1,034/13,024 = 7.939%, both below 40%.

## Boundary

This receipt establishes reproducible score-bearing data only. It is not a
neural result. A separate committed trainer/assessor and post-commit training
seed are required before the sole development read. Confirmation remains
forbidden unless every frozen development gate passes.

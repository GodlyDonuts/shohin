# R12 S6 Contextual Affine Law Development Board Receipt

**Status:** frozen after source commit `c09024b` and before fit, checkpoint,
development access, or score.

## Seeds

- Development board seed: `4930377975126057597`
- Training seed: `412095620685111169`

Both seeds were drawn after the architecture, optimizer, evaluator, assessor,
tests, and H100 wrapper were committed and pushed. Neither seed may be replaced.

## Frozen Board

Directory:
`artifacts/r12/s6_contextual_affine_law_development_4930377975126057597`

| File | Rows | SHA-256 |
|---|---:|---|
| `atomic_train.jsonl` | 961 | `c4312b99dddbad5c3c44e0af1b80b5fd281040291b30d05a999330deec64b9b7` |
| `development.jsonl` | 2,048 | `8fd78f761207e8446562c75e1816d1a0821d90ecd64fcb6b837f7a92fe808047` |
| `scale_diagnostic.jsonl` | 512 | `4016a6df9f9681df8599a4ab19b3804d8627c833aef5169f24f705bc28344984` |
| `report.json` | 1 report | `9e222b942613a5837775031210d3fc32bcf938eb161307c50eac01964f186394` |

The board has 103 training laws and 34 primary development laws with zero law
overlap. Every modulus/depth cell has 113 or 114 rows, every program uses at
least two laws, and treatment fields are exactly `card_y0`, `card_y1`,
`current_location`, and `modulus`. Confirmation programs and accesses are zero.

## Sole Authorized Run

Run the committed serial H100 wrapper once with this board and training seed.
It must fit treatment and law-ID control on atomic cells, write one checkpoint,
perform one development read, and apply the frozen assessor. Failure of a fit or
capability gate rejects S6. No optimizer, architecture, seed, board, threshold,
or rescore repair is authorized. Confirmation remains ungenerated unless the
assessment qualifies every primary gate.

# R12 SSC First-Integer Offline Rescore

**Status:** diagnostic only — does **not** change the frozen SSC last-integer
contract or confirmation claim.

**Source:** `artifacts/evals/source_scheduled_reasoning_confirmation_raw260k.json`
(SHA `be2e64c8…45de6a0d`).

**Script:** `train/rescore_ssc_first_integer.py`

## Numbers (256 whole decode)

| Metric | Count | Rate |
|---|---:|---:|
| Frozen last-integer (echo) | 9 | 3.5% |
| First integer == answer | 12 | 4.7% |
| Answer appears in segment | 60 | 23.4% |
| Appears but last-integer wrong (parser-ish loss) | 51 | 19.9% |

Family note: first-integer hits are almost all `base_conversion` (12/12), where
the model often states the decimal early. Multiply/modular/sequential show
`answer_appears` without first-integer match — intermediates precede the final.

## Relation to taxonomy 45/256

Taxonomy “reaches answer” (45) was a **trajectory-aware** post-hoc class
(scored 9 + correct-then-parser-loss 36). This offline pass uses a weaker
token heuristic (`answer_appears=60`), so it is an upper envelope, not a
replication of the exclusive taxonomy table.

## Use

Cashable without training: prefer decode/stop policies that keep the first
correct final and emit EOS, rather than last-integer under loop tails.

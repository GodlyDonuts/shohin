# R12 Factorized-Language Complete-Compiler Corpus Admission

**Status:** CPU-admitted; matched development arms running; confirmation sealed

## Frozen identity

The generator and tests were committed as `e6d957e` before any production seed
existed. The schema-aware compiler evaluator was committed as `6c416cb`, and
the matched ordinary-parser control plus common Slurm contract were committed
as `9b85463`. Production seeds were chosen only after the generator commit.

| Split | Rows | Quartets | Source SHA-256 |
|---|---:|---:|---|
| train | 96,000 | 24,000 | `e6feb311c37f34a88ce7bda59ebb4f968c9ce3b4052cb5c0f6c2ef2e3fca44a8` |
| development compositional | 2,048 | 512 | `e69fb70bddfb827a428c297352a72e45612ff3528a9fa107dec38c04189e1922` |
| development lexical OOD | 2,048 | 512 | `40a059024770d3785ac27f7b02365d7741f631f7413aa5e69631efbc3af73dc0` |
| confirmation | 8,192 | 2,048 | `e2bc25d8d95bb48c8d2915e6b966f3b96a01d3726177e6469f7824d0cf4b1a0f` |

The full local report SHA-256 is
`fd2c26580a1b164ad1095e0ad7940ffc2420c16f4882cbb0372c923c32bdc8f7`.
The Newton/GitHub development report removes both the confirmation seed and
the confirmation artifact entry; its SHA-256 is
`d481114232e438294bd1ea7f5b739f6068c2bf10fe02c1ee3c216c2e56aa3be3`.
Confirmation JSONL remains local only and is absent from Newton.

## Accounting

| Quantity | Value |
|---|---:|
| Total rows | 108,288 |
| Source tokens | 10,858,878 |
| Source UTF-8 bytes | 38,214,563 |
| Model-owned pointer labels | 1,082,880 |
| Teacher/model calls during generation | 0 |
| Checkpoint reads during generation | 0 |
| Production evaluation answer reads | 0 |

All rows contain ten nonempty source-owned spans: three initial entities, two
operation kind/entity/literal triples, and one query position. Independent
pop/insert and adjacent-swap CPU executors agree on every row.

## Structural gates

Every frozen CPU gate passes:

- all IDs are unique;
- every semantic quartet preserves canonical/paraphrase behavior and separates
  both order and binding twins;
- canonical/order/binding token bags are identical;
- train covers every known atomic language factor;
- compositional development uses only known atoms in unseen combinations;
- lexical OOD direction words are absent from all known-lexicon strata;
- exact prompt, word-13-gram, nonce-name, and full factor-combination overlap
  are zero across every split pair;
- token-bag shortcut accuracy is exactly chance at 1/3; absolute-position and
  source-length shortcut ceilings remain below the admitted chance tolerance.

The factor cross-product independently varies intro frame, list style,
operation ordinal vocabulary, operation frame, argument order, direction
lexicon, distractor frame/location, query frame, and punctuation/case style.
A split-disjoint neutral nonce anchor prevents shared generic prose from
creating cross-split 13-grams; it is sampled independently of every semantic
label and is included in the name-overlap audit.

## Matched development arms

Both initial arms use immutable raw Shohin 300k, 96,000 examples, one epoch,
1,517 optimizer updates, seed `2026071810`, source token IDs plus length mask as
their only inference inputs, and the same evaluator.

| Arm | Adapter parameters | Total parameters | Newton job |
|---|---:|---:|---|
| v1.3 parameter islands | 8,658,701 | 133,740,365 | `693048` on `evc25` |
| favorable ordinary bidirectional tagger | 8,607,886 | 133,689,550 | `693049` on `evc28` |

The adapter-budget difference is 50,815 parameters, or 0.587% of the islands
adapter. The ordinary control directly labels source tokens with five
bidirectional Transformer encoder layers and pools direction classes at the
predicted kind spans. It receives no learned program slots or parameter
islands. Both jobs are currently running; no score or promotion claim exists
until both compositional and lexical-OOD outputs complete.

## Decision boundary

Do not open confirmation unless v1.3 passes every frozen primary development
gate and beats the favorable ordinary parser by at least five percentage
points in exact semantic programs. Lexical OOD is diagnostic and must never be
pooled into the compositional score. No result on this board alone establishes
state transition, halting, autonomous rollout, or native reasoning.

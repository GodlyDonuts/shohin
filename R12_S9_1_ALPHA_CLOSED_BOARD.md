# R12 S9.1 Alpha-Closed Structured Compiler Board

**Status:** frozen and unevaluated

**Neural source commit:** `863a210`

**Board seed:** `1370124171784245712`

**Training seed:** `8076551815802451212`

## Frozen experiment

S9.1 retains the frozen 300k Shohin trunk, closed S8.1 initializer, S9
occurrence-quotient encoder, S7 cyclic generator, and S8 graph/runtime. The two
changes frozen at the source commit are:

1. equal-budget original/operation-recoded training pairs with a 0.25 aligned
   role-logit orbit loss; and
2. syntax-only top-eight non-overlapping child assignment around model-selected
   card/event anchors.

Each treatment/control arm receives 24,000 unique sources, 48,000 charged views,
batch 64, and 750 optimizer updates. No train state, answer, recurrent trace, or
score-board relation is provided.

## Board custody

| Payload | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| generator train | 23 | 3,763 | `754906d994121253889c3ebbbbd59e7f41e69fff5ca5f0a882fc1e5568dfff7f` |
| graph-only train | 48,000 | 406,390,454 | `db7649172ecfee9e62397918fc916678aa7929b5079761ebc27783b66879c20a` |
| development | 2,048 | 18,766,053 | `4b5d0e397e4df769f15b0ad34497ca2430fda641e8aaf184c92fcaa16a52bafe` |
| sealed confirmation | 2,048 | 18,738,735 | `ee7e19fc3d589d5e3831385f5f061705e44ccbc61ab444cefd863159c17f076b` |

Board report SHA-256:
`92cde7e7b1215dad66cd48ea8fc26937959d5b5751ea3c519256853fd8e27122`

Audit results:

- 52,096/52,096 independent executor agreements;
- 52,096/52,096 noncanonical node-storage rows;
- no train final state or answer;
- zero exact-prompt, 13-gram, or split-name overlap;
- development/confirmation access `0/0`;
- original and recoded maximum 455/512 tokens; and
- 8,568 nonce recodes change token width.

## Authorized next action

Commit this receipt plus the report and generator cells. Sync exact source,
large board payloads, tokenizer, protected base, and closed initializer to
Newton and hash-verify them. Then run one serial treatment/no-class/shuffled
training and sole development assessment. The sealed confirmation file must
remain unread by model/evaluator code unless every development gate passes.

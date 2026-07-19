# R12 S8 Nil-Linked Law Graph Board

**Status:** closed after one evaluator non-result; do not rescore

**Source commit:** `598e405ffdc08b0e03e999b715ec1d04f17f1b20`

**Board seed:** `4026952256631032219`

**Training seed:** `5532971934318350109`

**Board report SHA-256:** `067d97d790c0a2cadb0158ee013a74e0e0264e7dd099afa2ca0c5389294ddd31`

## Frozen board

| Payload | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `generator_train.jsonl` | 23 | 3,763 | `5c3b2e1ef13261b0e872305a495d70931232e4352baaa65b41f078995ce9c918` |
| `train.jsonl` | 48,000 | 406,632,176 | `d2925e0062051d11133f438ba7dfb8fb26e48886f802fc3cc1a97662f5818446` |
| `development.jsonl` | 2,048 | 18,752,468 | `58953f1d1dfa51e6913ce9674548f2b0d3f094019b106649f6173e7ea1e86754` |
| `confirmation.sealed.jsonl` | 2,048 | 18,697,579 | `ea0d242f9315e7d8a185162dde02b9d50b896b33149778705f59c97a5cbf2bd4` |

The builder records zero development and zero confirmation accesses. Training
contains graph-field supervision only and no final state or answer. Development
and confirmation use disjoint law pools, nonce names, and language renderers.
Across all 52,096 sources:

- independent reference and graph executors agree;
- node storage is noncanonical;
- split-name overlap is zero;
- exact prompt overlap is zero;
- 13-gram overlap is zero; and
- maximum tokenized length is 453 under the frozen 512-token tokenizer.

## Frozen neural system

The 125,081,664-parameter 300k Shohin trunk is frozen. The whole-source graph
compiler adds 8,610,966 parameters and initializes its five-layer memory encoder
from the confirmed S4 parser family. The learned cyclic generator adds 218
parameters. Complete-system accounting is **133,692,848 parameters**, below the
150M project ceiling.

The compiler is trained for one epoch on the 48,000 graph-field rows. It never
receives final states, answers, recurrent transitions, development laws, or
confirmation laws. The generator receives only 23 successor cells and three
zero anchors. A matched shuffled-label compiler starts from the identical
adapter state. The favorable ordinary parser emits execution ranks and receives
host list traversal; the S8 treatment must instead emit entry/next pointers and
nil termination.

## Closure

Job `693462` completed both frozen 750-update fits and wrote checkpoint SHA-256
`3c7154f2e31dd4f3e86534f8b007b7457585b85f7f7ffad4d13d8354721143af`.
The evaluator then opened development and failed before scoring or writing an
evaluation because token-ID nonce rotation assumed equal contextual BPE widths.
This board may not be patched or rescored. Its confirmation file remains sealed
and must not be opened. The sole admissible continuation is the separately
preregistered S8.1 source-level nonce repair on wholly fresh board bytes.

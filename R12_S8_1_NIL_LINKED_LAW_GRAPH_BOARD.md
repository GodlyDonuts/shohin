# R12 S8.1 Nil-Linked Law Graph Board

**Status:** frozen and admitted before neural access

**Source commit:** `ce2a5e47496f326c8dbd949c5ea3955b62ad4a49`

**Board seed:** `5943437777437228096`

**Training seed:** `8354164228219389085`

**Report SHA-256:** `1dcd576d9706c011ff8164994f0424f4bdc96a16525cdda400559b255b3aa831`

| Payload | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `generator_train.jsonl` | 23 | 3,763 | `263e215960d84097b4fd298f5a48f7e3823993435a01b91bd783ff88e2fd1215` |
| `train.jsonl` | 48,000 | 406,135,235 | `9e917ae5f09f9df623e354da6e66f4c5b92d39ac59f6910dd23a737e0ad80a28` |
| `development.jsonl` | 2,048 | 18,731,454 | `d16a1a8f773ff627b5f47ebd06b2344cad67a6473c431d78d79a5ef41f360d54` |
| `confirmation.sealed.jsonl` | 2,048 | 18,707,232 | `e951ac173135ee7791528ae78206cb48900865dbeade65085fcf948a2da2977d` |

The original-source audit matches S8 v1's exclusions: zero state/answer in
training, zero cross-split names, exact prompts, or 13-grams, independent
executor agreement for all 52,096 rows, noncanonical node storage, and maximum
length 455/512.

S8.1 additionally rotates operation nonce strings, adjusts all source spans,
retokenizes, and recompiles every generated source before sealing. All 52,096
pass; 9,018 change token count, and the maximum recoded length is 457/512.
Development and confirmation access counters are zero/zero.

The sole authorized next action is the unchanged S8 serial train/development
job using training seed `8354164228219389085` on a CUDA-preflighted H100.
Confirmation remains sealed unless every unchanged development gate passes.

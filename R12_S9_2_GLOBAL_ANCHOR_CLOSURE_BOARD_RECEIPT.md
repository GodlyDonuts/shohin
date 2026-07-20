# R12 S9.2 Global Anchor Closure Board Receipt

**Decision:** admit the sole fresh S9.2 development/confirmation board

**Frozen scientific source commit:**
`38c934cf9f360e1fd13258c23be310e948cafba1`

**Board seed:** `3823077847356570601`

**Training seed:** `1277007704479652588`

**Report SHA-256:**
`f22401e82690f8240abe89d6083e3a387619243bc68bb9d9e380540d90b1899e`

## Frozen files

| File | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `generator_train.jsonl` | 23 | 3,763 | `4bea22f931eb1288c42c4f53e0a7ec6aedb2b2650f1b9193e6f475a4bbefb575` |
| `train.jsonl` | 48,000 | 406,537,768 | `1b7e9029f530d8df9a64613346285ccd94bb35e7b806cbbde4a81c6f428c66a3` |
| `development.jsonl` | 2,048 | 18,720,807 | `a186df62c9fe030d71f8c3734e6e6d570667c2c2ddc340d81718813258b96550` |
| `confirmation.sealed.jsonl` | 2,048 | 18,723,696 | `84be3808f1f385740edf6acb8129985075e90506e31a2d087d8f34a79703c054` |

Tokenizer SHA-256 is
`87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`.

## Admission audit

- all 52,096 source graphs agree with the independent executor;
- all 52,096 rows use noncanonical graph storage;
- training contains no final state or answer;
- exact-prompt overlap is zero across every split pair;
- 13-gram overlap is zero across every split pair;
- split-name overlap is zero;
- original maximum length is 453/512 tokens;
- operation-recoded maximum length is 452/512;
- 9,345 rows change token width under operation recoding;
- development and confirmation depth/modulus/renderer cells differ by at most
  one row; and
- development/confirmation access is `0/0`.

The failed preliminary invocation supplied an incorrect expanded commit hash
and exited inside the clean-HEAD guard before creating the output directory.
The board seed was therefore unused until the successful invocation above.

## Custody

Neither `development.jsonl` nor `confirmation.sealed.jsonl` was opened after
generation. Training may read only `train.jsonl` and `generator_train.jsonl`.
The evaluator must atomically claim the deterministic board-hash development
ledger before its sole read. Confirmation remains sealed unless all 43 frozen
development gates pass.

# R12 S9 Occurrence-Quotient Relational Compiler Board

**Status:** frozen and unevaluated

**Neural source commit:** `9fd8aea`

**Bounded-memory source commit:** `ba9e4c6`

**Board seed:** `7563652620455132721`

**Training seed:** `1782702123750965299`

The first post-`9fd8aea` candidate board was discarded before training or score
access because full-corpus span materialization wasted host memory. Commit
`ba9e4c6` changes only proposal materialization from corpus-wide to active-
batch generation; it preserves every candidate, label, model parameter, loss,
control, and gate. Fresh seeds were drawn after that repair.

## Architecture

The frozen Shohin trunk feeds a five-layer width-384 contextual encoder. The
model scores all contiguous token spans up to width four, pools start/end/mean
residuals, groups candidates by exact trimmed source bytes within one example,
and predicts relation slots from local span plus shared class context. The
treatment receives that class message. The equal-parameter no-class control
zeros it. A third equal-architecture arm receives shuffled relation labels.

Parameter count from the frozen implementation:

- Shohin base: 125,081,664
- occurrence-quotient compiler: 9,498,382
- learned cyclic generator: 218
- complete system: **134,580,264**

The checkpoint will record the runtime count and fail if it differs or reaches
150 million.

## Board custody

| Payload | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| generator train | 23 | 3,763 | `bc691adff44fd12b4fa5419379a9ee1c5f90c97bdaafa7576f36685ff2d7dad5` |
| graph-only train | 48,000 | 406,246,861 | `fea64b033f4ff418e4b4af194b17bbedd0b4e601f9ed3224aaf9feb70f034327` |
| development | 2,048 | 18,774,014 | `193df4513e9b7186aefbe3890931be85f4a7f0b154ab39c0409614603a686ff0` |
| sealed confirmation | 2,048 | 18,752,717 | `2f0967bc35ee4b01f1adb59e6f0278c18394f3a0f84a5727b21e8df21e256419` |

Board report SHA-256:
`fb81b75f5963ad4bcd513d9e4a14e2fa36ad02dabd1085b9f4387c270755cd93`

Audit results:

- 52,096/52,096 independent executor agreements;
- 52,096/52,096 noncanonical node-storage rows;
- no train final state or answer;
- zero exact prompt, 13-gram, or split-name overlap;
- development/confirmation access `0/0`;
- original maximum 459/512 tokens;
- source-recoded maximum 458/512;
- 8,890 nonce recodings change token width;
- maximum gold island width 3 under the frozen width-4 proposal cap; and
- oracle logits pass the complete neural proposal/assembly path on 2,048/2,048
  development graphs without fitting or score access.

## Authorized next action

Commit the report, generator cells, this receipt, and updated ledgers. Sync exact
large payload bytes, source, tokenizer, protected 300k base, and closed S8.1
initializer to Newton and hash-verify them. Then run one serial development job:
treatment, no-class-message, shuffled-relations, evaluation, and assessment.
Confirmation remains unopened unless every immutable development gate passes.

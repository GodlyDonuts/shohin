# R12 Referential Literal-Pointer Compiler CPU Result

**Decision:** **CPU BOARD PASS; ONE ISOLATED COMPILER PILOT AUTHORIZED.**

**Claim boundary:** this is a deterministic bounded-language and leakage result.
It is not a Shohin score, neural result, native-reasoning result, arithmetic
result, source-deleted executor result, halt result, or novelty claim.

## 1. Question tested

R4's binding-first compiler improved held-out exact programs from `469/896` to
`624/896`, but its exact-program metric omitted initial quantities and event
values. A deterministic host lexer supplied those values to execution. The
new preregistration asks whether a future neural compiler can own the complete
typed interface:

```text
[operation kind, entity token-span pointer, literal token-span pointer]
```

plus query and STOP, with no structured value supplied at inference.

Before fitting, the CPU falsifier had to establish that its board is
semantically well-defined and not directly solved by the named lexical,
position, or template features.

## 2. Frozen command

```bash
python3 pipeline/semantic_compiler_falsifier.py \
  --tokenizer artifacts/shohin-tok-32k.json \
  --out artifacts/r12/semantic_compiler_falsifier_v1.dev.json \
  --receipt artifacts/r12/semantic_compiler_falsifier_v1.dev.receipt.json
```

The development seed is `20260718`. No confirmation seed exists.

## 3. Exact result

All 14 frozen gates pass:

| Gate | Result |
|---|---:|
| Quartets / surfaces | **32 / 128** |
| Typed-AST round trips | **128/128** |
| Independent executor agreement | **128/128** |
| Equivalent canonical/paraphrase groups | **32/32** |
| Noncommuting order twins separated | **32/32** |
| Argument-binding twins separated | **32/32** |
| Canonical/order/binding token bags equal | **32/32** |
| Nonempty exact kind/entity/literal/query token spans | **128/128** |
| Disjoint nonce-name quartets | **32/32** |
| Named shortcut features at or below `1/3` | **7/7** |
| Teacher/model/checkpoint/production-answer reads | **0** |
| Confirmation seed present | **no** |

The matched-surface Bayes-optimal exact-program shortcut ceilings are:

| Feature available to shortcut | Best possible exact programs |
|---|---:|
| exact Shohin-tokenizer bag | **32/96 = 33.33%** |
| entity/literal bag | **32/96 = 33.33%** |
| absolute pointer positions | **7/96 = 7.29%** |
| span widths | **5/96 = 5.21%** |
| source token length | **4/96 = 4.17%** |
| operation bag | **3/96 = 3.13%** |
| renderer identity | **1/96 = 1.04%** |

The two `33.33%` ceilings are the intended chance ceiling for each matched
canonical/order/binding triple. They do not establish that an unlisted neural
shortcut is impossible. They establish only that the named direct leaks do not
solve the board.

## 4. Acquisition and execution ledger

| Resource | Count |
|---|---:|
| UTF-8 source bytes | 34,144 |
| source tokens | 10,116 |
| target pointer labels | 896 |
| typed-program oracle calls | 128 |
| separator-oracle calls | 96 |
| executor A calls | 128 |
| executor B calls | 128 |
| teacher-model calls | 0 |
| checkpoint reads | 0 |
| production-evaluation answer reads | 0 |
| training examples / FLOPs | 0 / 0 |
| sequential instruction depth | 2 |

Executor A uses remove/reinsert semantics. Executor B uses repeated adjacent
swaps. They agree on every surface.

## 5. Evidence identity

| Artifact | SHA-256 |
|---|---|
| preregistration | `96deb2da5a2e63fab124e5d34219dd2e0ebba934120aae0c87bf89f71dbdcb6a` |
| generator/falsifier | `400cdfb23b8bc49a3ad23c4a4b5374a54656fadef982d2c8bc732d073e20d10f` |
| tests | `59855a26af7682ee7860a7abfa203386986bd4814240f2877448f02436898a24` |
| Shohin tokenizer | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| development artifact | `a13bee354d847844ba6db27a65a68a8f7ce540f1558692fa06f31be9919193c1` |
| receipt | `52e66e2d96f19e30bb49f85f1a3e0c6336c4e9c184199bc58432b8eeb9df3ea4` |

Verification passes `py_compile`, four unit tests, and `git diff --check`.

## 6. Consequence

The result authorizes exactly one next stage: freeze a fresh
train/development/confirmation corpus and fit the preregistered complete
compiler against R4, absolute-role, ordinary pointer-network, text-AST, joint,
shuffled, and oracle controls. The immutable 300k base must remain frozen.

Executor integration remains blocked. A compiler that parses a two-step list
machine is still a semantic parser, not a reasoner. Only after the compiler
passes its untouched confirmation gate may it be connected to a separately
preregistered learned source-deleted packet updater. HALT remains a third
independent gate.

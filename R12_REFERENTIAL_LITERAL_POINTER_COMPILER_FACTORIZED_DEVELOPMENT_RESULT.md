# R12 Factorized-Language Complete-Compiler Development Result

**Status:** absolute compiler gates pass; parameter-islands attribution fails;
confirmation remains sealed

## Bottom line

Broad factorized language supervision changed the source-pointer compiler from a
renderer-indexed partial parser into a nearly or fully exact known-atom
compiler. The preregistered parameter-islands mechanism did not earn credit for
that improvement:

- parameter islands: **2,048/2,048 exact compositional programs**;
- favorable ordinary tagger: **2,048/2,048**;
- structured parser: **2,048/2,048**;
- free learned slots: **2,012/2,048 = 98.242%**;
- shuffled-label islands: **3/2,048 = 0.146%**.

Every absolute v1.3 primary gate passes, but its exact-program advantage over
the ordinary parser is **0.0 percentage points**, below the frozen +5-point
attribution gate. The committed assessor therefore returns:

```text
retain_as_conventional_compiler_baseline_confirmation_sealed
```

This is a strong **data-identifiability result** and a useful conventional
compiler baseline. It is not evidence that parameter islands are necessary,
not a sealed-confirmation pass, and not an executor, halt, autonomous rollout,
or native-reasoning result.

## Frozen contract

All five neural arms use:

- immutable raw-300k Shohin, SHA-256
  `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`;
- the same 96,000 training rows and 1,517 optimizer updates;
- seed `2026071810`;
- source token IDs and a source-length mask as their only inference inputs;
- the same 2,048-row compositional and 2,048-row lexical-OOD evaluators;
- zero confirmation access.

The corpus generator was committed before production seeds. Train,
compositional-development, and lexical-OOD SHA-256 values are respectively
`e6feb311c37f34a88ce7bda59ebb4f968c9ce3b4052cb5c0f6c2ef2e3fca44a8`,
`e69fb70bddfb827a428c297352a72e45612ff3528a9fa107dec38c04189e1922`,
and `40a059024770d3785ac27f7b02365d7741f631f7413aa5e69631efbc3af73dc0`.
Confirmation bytes were never copied to Newton or read by training,
evaluation, or assessment.

## Jobs and resources

| Arm | Job | Node | Adapter parameters | Total parameters | Elapsed |
|---|---:|---|---:|---:|---:|
| parameter islands | `693048` | `evc25` | 8,658,701 | 133,740,365 | 12:04 |
| ordinary token tagger | `693049` | `evc28` | 8,607,886 | 133,689,550 | 11:17 |
| shuffled-label islands | `693098` | `evc25` | 8,658,701 | 133,740,365 | 10:02 |
| free learned slots | `693101` | `evc28` | 3,241,091 | 128,322,755 | 5:46 |
| structured parser | `693102` | `evc28` | 6,402,701 | 131,484,365 | 9:50 |

All jobs exited `0:0`. Oracle jobs `693099` and `693100` each completed in 45
seconds on `evc28`.

## Primary compositional development

| Arm | Answer | Program | Full ten-pointer | Kind | Initial | Canonical+paraphrase | All-four |
|---|---:|---:|---:|---:|---:|---:|---:|
| free slots | 98.340% | 98.242% | 98.242% | 100.000% | 98.340% | 496/512 | 496/512 |
| structured | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% | 512/512 | 512/512 |
| parameter islands | **100.000%** | **100.000%** | **100.000%** | **100.000%** | **100.000%** | **512/512** | **512/512** |
| ordinary tagger | **100.000%** | **100.000%** | **100.000%** | **100.000%** | **100.000%** | **512/512** | **512/512** |
| shuffled islands | 0.488% | 0.146% | 0.000% | 49.756% | 59.229% | 0/512 | 0/512 |

The shuffled arm's chance-level operation loss and near-zero exact program and
answer scores rule out evaluator leakage, a label-free answer shortcut, or a
base-only solution. The three exact supervised architectures and the 98.2%
free-slot arm show that broad factor coverage, not specialized parameter
separation, produced the main gain.

## Lexical-OOD diagnostic

The lexical-OOD split replaces all trained direction words with unseen
direction pairs. It is diagnostic and was never pooled into the primary score.

| Arm | Answer | Program | Full ten-pointer | Kind | Initial | Canonical+paraphrase | All-four |
|---|---:|---:|---:|---:|---:|---:|---:|
| free slots | **89.307%** | **85.010%** | 71.777% | **86.621%** | 98.633% | 262/512 | 249/512 |
| structured | 82.373% | 72.852% | 72.852% | 74.072% | 99.854% | 279/512 | 190/512 |
| parameter islands | 85.352% | 77.881% | **77.881%** | 78.979% | **99.805%** | **318/512** | **231/512** |
| ordinary tagger | 76.367% | 63.721% | 63.721% | 70.776% | **99.805%** | 212/512 | 163/512 |
| shuffled islands | 0.391% | 0.049% | 0.000% | 47.998% | 54.932% | 0/512 | 0/512 |

No one architecture dominates every lexical metric. Free slots have the best
row-level answer/program/kind scores; islands have the best full-pointer and
quartet consistency. This is secondary evidence only because the lexemes were
absent from training and no architecture received definitions for them.

## Oracle localization

| Arm | Oracle | Answer | Exact program/full pointer | All-four exact |
|---|---|---:|---:|---:|
| islands | none | 85.352% | 77.881% | 231/512 |
| islands | gold operation kinds | **99.463%** | **99.316%** | **505/512** |
| islands | gold structural pointers | 85.840% | 78.516% | 234/512 |
| islands | full | 100.000% | 100.000% | 512/512 |
| ordinary | none | 76.367% | 63.721% | 163/512 |
| ordinary | gold operation kinds | **99.219%** | **98.975%** | **501/512** |
| ordinary | gold structural pointers | 76.904% | 64.307% | 165/512 |
| ordinary | full | 100.000% | 100.000% | 512/512 |

Supplying only operation polarity closes almost the entire lexical-OOD gap;
supplying every structural pointer barely changes it. Initial, entity, literal,
and query pointers are already about 99--100% exact. The residual failure is
therefore unseen-word semantic polarity, not role binding or execution.

## Frozen gate outcome

| Gate | Floor | Result | Pass |
|---|---:|---:|---|
| answer accuracy | 85% | 100% | yes |
| semantic-program exact | 75% | 100% | yes |
| full ten-pointer exact | 65% | 100% | yes |
| operation-kind accuracy | 95% | 100% | yes |
| initial-state joint exact | 80% | 100% | yes |
| canonical+paraphrase both exact | 192/512 | 512/512 | yes |
| all-four exact | 96/512 | 512/512 | yes |
| islands program advantage over ordinary | +5 points | +0.0 points | **no** |

The assessment SHA-256 is
`ca8cab2ef9dbaa9d894857438e72193476259fd659e8423b85af47e13e37fc0d`.

## Decision and next use

1. Keep confirmation sealed. The preregistered attribution condition failed.
2. Retain the ordinary tagger as the favorable conventional compiler baseline.
3. Treat factorized language generation as the durable discovery: it supplies
   the coverage that all supervised parsers previously lacked.
4. Build the next source-deleted transition/consumer experiment against this
   conventional compiler, with a fresh untouched qualification board and no
   current-confirmation reuse.
5. Keep unseen-lexeme semantics separate. Definitions, contrastive lexical
   grounding, or pretrained-language support may be tested as explicit
   resources; they may not be disguised as an executor improvement.

No result here authorizes reporting Shohin as a native reasoner. The compiler is
one independently gated component of the larger compiler/executor/state/
consumer/halt program.

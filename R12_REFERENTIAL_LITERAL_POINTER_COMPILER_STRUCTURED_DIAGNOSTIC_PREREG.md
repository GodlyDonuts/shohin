# R12 Structured Complete-Compiler Diagnostic Preregistration

**Protocol:** `r12_referential_literal_pointer_compiler_v1_2_structured_development`
**Status:** frozen before v1.2 fitting or scoring
**Selection boundary:** the v1.1 development set is exposed; this is a mechanism-repair diagnostic,
not an untouched generalization result

## 1. Failed parent

The v1.1 six-slot compiler fit its two training renderers to near-zero loss but reached only
45/2,048 full-pointer exact, 313/2,048 semantic-program exact, and 602/2,048 answers on frozen
development. Its unseen paraphrase renderer scored 1/512 answers and systematically selected
renderer-coordinate words such as `unaffected` and `travel`.

The parent result is immutable in
`R12_REFERENTIAL_LITERAL_POINTER_COMPILER_DEVELOPMENT_RESULT.md`. Confirmation remains sealed.

## 2. Single treatment change

Keep the exact parent base, train/development bytes, examples, update count, batch size, optimizer,
schedule, six program slots, ten pointer targets, two kind classifiers, and host dereference/executor.
Change only the source parser:

1. Project frozen layer-19 token states to width 256 as before.
2. Apply four bidirectional Transformer encoder layers, width 256, eight heads, FF 1,024.
3. Predict ten token-role logits at every source position.
4. Add each role logit to its corresponding pointer score.
5. Train a balanced binary role-label objective at weight `0.5` using the same gold source spans as
   the pointer objective.

No line boundary, renderer identifier, structured program, initial value, event value, answer,
gold span, or role label is supplied at inference. The model input remains source token IDs and a
source-length mask only.

## 3. Resource ledger

| Component | Parameters |
|---|---:|
| immutable Shohin base | 125,081,664 |
| structured compiler adapter | 6,402,701 |
| total | 131,484,365 |

The total remains below 150M. The base is frozen. The treatment uses one H100 and four CPUs.

## 4. Frozen fit

- train bytes: v1.1 `train.jsonl`, SHA-256
  `f47c6d6ce316be6765641f61a294481605fa53c7b12388741fb753c238b2f36e`;
- 96,000 examples, one epoch, batch 64, 1,514 expected updates;
- AdamW, betas `(0.9, 0.95)`, weight decay `0.01`;
- peak LR `0.001`, 50-update warmup, cosine decay to 10%;
- gradient clip 1.0;
- role loss weight 0.5;
- seed `2026071804`;
- distinct output `train/referential_literal_pointer_compiler_v1_2_structured/`.

## 5. Frozen diagnostic gates

The v1.1 development set is intentionally reused only to answer whether the structural field repairs
the observed renderer-coordinate failure. It cannot authorize confirmation.

| Metric | Gate |
|---|---:|
| overall answer accuracy | >=50% and >=15 percentage points above v1.1 |
| semantic-program exact | >=40% and >=20 percentage points above v1.1 |
| full ten-binding pointer exact | >=20% and >=15 percentage points above v1.1 |
| paraphrase answer accuracy | >=25% |
| paraphrase semantic-program exact | >=20% |
| canonical answer accuracy | no more than 5 percentage points below v1.1 |
| canonical + paraphrase both pointer-exact | >=64/512 |
| all four surfaces pointer-exact | >=32/512 |

## 6. Decisions

- **Fails:** reject bidirectional role supervision as an adequate repair under two-renderer training.
- **Passes:** authorize construction of a fresh factorized-language board and favorable matched
  controls. Do not read v1.1 confirmation.
- **Regardless of result:** no compiler/executor integration, native-reasoning claim, novelty claim,
  or production promotion follows from this exposed-development diagnostic.

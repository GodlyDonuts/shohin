# R12 Complete-Compiler Parameter-Islands Preregistration

**Protocol:** `r12_referential_literal_pointer_compiler_v1_3_islands_development`
**Status:** frozen before v1.3 fitting or scoring
**Selection boundary:** exposed-development mechanism diagnostic only

## Hypothesis

The v1.2 structural parser learned substantially better ordered bindings, but its operation-kind
head stayed at chance because token-role supervision bypassed the free semantic slot. The zero-fit
hybrid of v1.2 structure and v1.1 operation classes recovered 48.193% answers and 33.740% semantic
programs. Therefore:

> A physically separate semantic reader over the original frozen causal states can preserve
> operation meaning while a bidirectional structural reader learns source roles.

## Single treatment change from v1.2

Keep v1.2 unchanged. Add a separate two-layer semantic slot decoder with its own layer norm and
576-to-256 memory projection over frozen layer-19 states. Only the operation-kind classifier reads
this island. The structural encoder, role head, ten pointers, role loss, data, optimizer, and host
executor remain unchanged.

No island receives line spans, structured values, operation labels, renderer IDs, or answers at
inference. The only inference inputs remain source token IDs and the source-length mask.

## Resource and fit ledger

| Component | Parameters |
|---|---:|
| immutable Shohin | 125,081,664 |
| complete compiler islands | 8,658,701 |
| total | 133,740,365 |

- v1.1 train SHA-256 `f47c6d6ce316be6765641f61a294481605fa53c7b12388741fb753c238b2f36e`;
- 96,000 examples, one epoch, batch 64, 1,514 expected updates;
- same AdamW, LR, warmup, cosine, clip, and loss weights as v1.2;
- seed `2026071805`;
- one H100, four CPUs;
- distinct output `train/referential_literal_pointer_compiler_v1_3_islands/`.

## Frozen gates

| Metric | Gate |
|---|---:|
| operation-kind accuracy | >=90% |
| initial-state joint exact | >=45% |
| answer accuracy | >=45% |
| semantic-program exact | >=30% |
| full pointer exact | >=10% |
| paraphrase answer accuracy | >=45% |
| canonical answer accuracy | >=35% |
| canonical + paraphrase both pointer-exact | >=1/512 |

Passing all gates authorizes a new untouched factorized-language board and favorable controls only.
It does not authorize v1.1 confirmation, an executor connection, a native-reasoning claim, or a
novelty claim.

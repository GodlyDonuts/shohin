# R12 Complete Pointer Compiler Development Pilot Manifest

**Status:** **FROZEN BEFORE GPU FIT OR DEVELOPMENT SCORE.**

**Scientific commit:** `51a5a6410b93ec277bc0d0adc0821f0a3674283f`

**Authorization:** one treatment-only development feasibility fit. Confirmation
access is forbidden. A development pass authorizes implementation and matched
development selection of the full control suite; it does not authorize direct
confirmation scoring.

## 1. Hypothesis

A small learned slot decoder over frozen Shohin token states can compile the
complete bounded program interface without structured inference inputs:

```text
three initial-order entity pointers
two [operation-kind grounding, operation class, entity pointer, literal pointer] codons
one late-query pointer
fixed STOP after two operations
```

The model receives only source token IDs and a source-length mask. It receives
no line boundaries, target spans, entity list, initial order, values, typed
program, query, answer, renderer ID, or semantic state.

This is a learnability hypothesis for a supervised semantic parser. It is not a
new reasoning primitive.

## 2. Frozen implementation

The immutable 300k Shohin base is frozen through layer 19. Six learned slot
queries cross-attend over all source-token states through a two-layer,
256-wide, eight-head decoder with a 1,024-wide feed-forward block. Ten
independent pointer projections score every valid source token. Two operation
slots also classify `LEFT` versus `RIGHT`.

| Component | Parameters |
|---|---:|
| immutable Shohin base | 125,081,664 |
| complete pointer compiler | 3,241,091 |
| strict total | **128,322,755** |
| remaining below 150M | 21,677,245 |

Pointer loss is the negative log probability mass assigned to every token in
the gold source span, averaged across ten targets. Operation-kind
cross-entropy has weight 1.0. The base receives no gradients.

At evaluation, the host expands each model-selected token to its containing
source word, then applies the frozen list-machine semantics. It may not select,
repair, reorder, or infer an operand. Full pointer exactness requires all ten
pointers and both operation classes. Fixed STOP is not a halt result.

## 3. Frozen inputs

| Input | SHA-256 |
|---|---|
| base 300k checkpoint | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |
| train, 96,000 rows | `f47c6d6ce316be6765641f61a294481605fa53c7b12388741fb753c238b2f36e` |
| development, 2,048 rows | `20611bf4ddbdb42d7e2f9dd76759b86f3f4dd16d5942f207bf7b325984da5ad6` |
| v1.1 corpus report | `176435d8c544948468f81cb23dc65ff51bf8010af212fb737984bbed1d1265cc` |
| tokenizer | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| v1.1 amendment | `f7d8f6f23ceb2f91d33c8a46340e10e298a2b1aa39ca6b1b5e6264d80bbcd72a` |

Confirmation SHA-256
`84005921b5fca93f9c2567655c4345bced78fc74ed7f49c8f72189b9f87fbf03`
is recorded for custody but its bytes must not be copied into the pilot runtime
or read by any pilot process.

## 4. Frozen source identities

| Source | SHA-256 |
|---|---|
| architecture/data helpers | `080f1bf22eb5fe62d7e9aecec0fc7d351110b549d264f9381fc0158e7666437f` |
| trainer | `3fcc1f4574b61c29ff14df4462c8e939ee30e90ee6aa6ee484996854435ded91` |
| evaluator | `b52962333126f2a10232811755e35843fe6afaad3252aafd14dca6c02020b004` |
| Slurm job | `cb733cb8650cd450a491a21fea1b10a5d5cb7e428c346e0b7c24cbaf325c7cc7` |
| unit tests | `0120ae49a9a19c6fec76443deea2d827566bfabde12d280bb164d6d8b2ee73c0` |

Local verification before freeze:

- three architecture tests pass;
- five corpus tests pass;
- `py_compile`, Ruff, Bash syntax, and `git diff --check` pass;
- a real CPU forward from the immutable 300k checkpoint produces ten pointer
  distributions of shape `[1,81]` and operation logits `[1,2,2]`;
- gold-pointer dereference reconstructs the answer without structured values;
- strict total parameters are below 150M.

## 5. Frozen optimization

| Setting | Value |
|---|---:|
| training examples | 96,000 |
| epochs | 1 |
| batch size | 64 |
| nominal updates | approximately 1,500, subject only to exact length buckets |
| optimizer | AdamW, betas `(0.9,0.95)`, weight decay `0.01` |
| peak LR | `0.001` |
| warmup | 50 updates |
| schedule | cosine to 10% of peak |
| gradient clip | 1.0 |
| seed | `2026071803` |
| precision | frozen base/adapter forward under BF16 autocast; losses in FP32 |
| accelerator | one H100; four CPUs; 96 GiB host memory |

No checkpoint, epoch, layer, width, loss weight, or seed selection is permitted
after viewing development. This pilot has one artifact and one development
score.

## 6. Frozen development gate

The slot-decoder realization advances to a matched-control build only if all
floors pass on the untouched development split:

| Metric | Floor |
|---|---:|
| all ten pointers + both operation classes exact | **40%** |
| host-dereferenced semantic program exact | **50%** |
| answer accuracy from predicted program | **50%** |
| joint three-pointer initial order | **70%** |
| operation-0 joint kind/pointers | **60%** |
| operation-1 joint kind/pointers | **60%** |
| canonical/paraphrase both fully exact | **128/512 groups** |
| all four surfaces fully exact | **64/512 groups** |

If any floor fails, this exact slot-decoder realization is rejected. The
failure does not reject complete pointer compilation in general, but no
confirmation access or executor integration follows.

If every floor passes, the next required work is to implement and freeze the
R4 privileged-lexer baseline, absolute-role compiler, ordinary pointer network,
text-AST decoder, joint adapter, shuffled-pointer sanity control, and oracle
ceiling under matched development budgets. Arm identities and the selection
rule must be committed before confirmation is copied into the runtime.

## 7. Claim boundary

A development pass establishes only that the complete source-pointer interface
can be learned on the bounded synthetic machine. It does not establish natural-
language reasoning, arithmetic, source-deleted execution, autonomous state
update, serialization, halt, scale transfer, or novelty.

# R12 Referential Literal-Pointer Compiler Development Result

**Protocol:** `r12_referential_literal_pointer_compiler_v1_1_development`
**Decision:** **REJECT THIS REALIZATION**
**Claim boundary:** development-only complete compiler feasibility; no confirmation, executor,
halt, native-reasoning, or novelty claim

## 1. Immutable identities

| Item | Value |
|---|---|
| Scientific source commit | `51a5a6410b93ec277bc0d0adc0821f0a3674283f` |
| Frozen pilot manifest commit | `a1a6af0` |
| Base checkpoint step | 300,000 |
| Base SHA-256 | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |
| Train JSONL SHA-256 | `f47c6d6ce316be6765641f61a294481605fa53c7b12388741fb753c238b2f36e` |
| Development JSONL SHA-256 | `20611bf4ddbdb42d7e2f9dd76759b86f3f4dd16d5942f207bf7b325984da5ad6` |
| Corpus report SHA-256 | `176435d8c544948468f81cb23dc65ff51bf8010af212fb737984bbed1d1265cc` |
| Tokenizer SHA-256 | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Initial adapter-state SHA-256 | `0ec56b21df404be2a7cbd73765d8ec64a58474de919d01b76ee29d56b7b2c38d` |
| Final adapter-state SHA-256 | `ad52a12239d1d00b1877a505cb12b73c4803f2b0b763b3bf94d8bab6abf8f8c2` |
| Final adapter file SHA-256 | `6815f2fb68e94701630eaece6fff740e54a6f69d2fb226468bbcc1989b7e3cfa` |
| Development result SHA-256 | `070d148c9d0031fea83218f4a941ecbc4a50c00f0961fcf7f0f8435bdc2a4a25` |
| Slurm log SHA-256 | `277e22e0e24bc8270f29edcea42272990fdcbee2f6bfee7a29867732627ceeb4` |

The adapter, full development record, and Slurm log are hash-matched between Newton and the Mac.
The confirmation JSONL was not copied to Newton. Evaluation metadata records
`confirmation_access = 0`.

## 2. Execution record

The first allocation, job `692965` on `evc26`, exposed no CUDA device. It was canceled before an
update or artifact. This is a hardware-allocation invalidation, not a model result.

Job `692966` ran unchanged on a verified NVIDIA H100 PCIe on `evc28` and completed with exit code
zero in 8 minutes 22 seconds. The fit used:

- 96,000 examples, one epoch, 1,514 updates;
- batch 64, AdamW, peak LR `0.001`, 50-update warmup, cosine decay;
- frozen Shohin through layer 19;
- 125,081,664 frozen base parameters plus 3,241,091 trainable compiler parameters;
- 128,322,755 total parameters, below the strict 150M cap;
- source token IDs and a source-length mask as the only neural inference inputs.

Fit elapsed time was 304.965 seconds. Training loss fell from `4.6947` at update zero to
`0.0000018793` at update 1,500. This is fit evidence only.

## 3. Frozen development gate

The development set contains 2,048 rows in 512 semantic quartets. All names and renderer templates
are disjoint from training.

| Metric | Frozen gate | Result | Count | Pass |
|---|---:|---:|---:|---:|
| Full ten-binding pointer exact | >=40% | **2.197%** | 45/2,048 | no |
| Semantic program exact | >=50% | **15.283%** | 313/2,048 | no |
| Executed answer accuracy | >=50% | **29.395%** | 602/2,048 | no |
| Initial-state joint exact | >=70% | **18.848%** | 386/2,048 | no |
| Operation-0 joint exact | >=60% | **38.086%** | 780/2,048 | no |
| Operation-1 joint exact | >=60% | **61.426%** | 1,258/2,048 | yes |
| Canonical + paraphrase both pointer-exact | >=128/512 | **0/512** | 0 | no |
| All four surfaces pointer-exact | >=64/512 | **0/512** | 0 | no |

Only one of eight frozen gates passes. The treatment is rejected before confirmation.

## 4. Failure localization

### 4.1 Surface decomposition

| Surface | Initial joint | Op-0 joint | Op-1 joint | Program exact | Answer |
|---|---:|---:|---:|---:|---:|
| canonical | 24.414% | 18.359% | 79.883% | 20.508% | 41.992% |
| order twin | 23.438% | 19.336% | 82.422% | 20.508% | 38.281% |
| binding twin | 24.219% | 16.602% | 82.617% | 20.117% | 37.109% |
| paraphrase | **3.320%** | **98.047%** | **0.781%** | **0%** | **0.195%** |

The paraphrase renderer is the decisive collapse. The model nearly always locates operation zero
but nearly never locates operation one. Inspection of predictions shows systematic selection of
renderer-position words such as `unaffected` for `intro.entity1` and `travel` for `op1.entity`.
This is not random uncertainty; it is a learned coordinate shortcut.

### 4.2 Target decomposition

| Target | Accuracy |
|---|---:|
| intro entity 0 | 52.637% |
| intro entity 1 | 45.947% |
| intro entity 2 | 96.094% |
| operation-0 kind pointer | 41.113% |
| operation-0 entity pointer | 94.873% |
| operation-0 literal pointer | 100% |
| operation-1 kind pointer | 89.160% |
| operation-1 entity pointer | 74.756% |
| operation-1 literal pointer | 100% |
| query-position pointer | 100% |
| operation-kind class | 96.265% |

Literal values, query position, operation-kind class, and most entity copying are learnable. The
failure is primarily renderer-invariant role assignment, especially ordered initial bindings and
the second operation under an unseen paraphrase grammar.

## 5. Scientific consequence

This experiment rejects the specific architecture of six free learned slot queries directly
cross-attending to linearly projected frozen causal states after training on two renderer families.
It does **not** reject source-pointer compilation, the compiler/executor decomposition, or the
possibility of a complete learned compiler under 150M parameters.

The result identifies three factors that the first pilot conflated:

1. **Lexical grounding:** map words such as `front`, `rear`, `earlier`, and `later` to operation
   classes.
2. **Structural role parsing:** distinguish intro entity order, operation index, entity argument,
   literal argument, and query argument without absolute renderer coordinates.
3. **Program execution:** consume the compiled bindings and apply transitions.

The current host executor tests factor 3 only after factors 1 and 2. A repair must measure those
factors separately and must not compensate for parser failure with host-provided spans or values.

## 6. Authorized next work

Confirmation remains sealed. No control suite or confirmation evaluation is authorized for this
failed arm.

A successor may proceed only after it freezes a fresh language board and separates the following
causes:

- same architecture with broad factorized renderer coverage;
- a bidirectional structured token parser with the original training budget;
- a favorable ordinary sequence tagger / pointer-network control;
- a lexical-oracle control that supplies operation-class grounding but no entity, literal, or
  query bindings;
- a structure-oracle control that supplies role boundaries but no selected source values;
- a full oracle ceiling.

Any successor development language observed during this experiment is contaminated for model
selection and may not serve as its untouched gate.

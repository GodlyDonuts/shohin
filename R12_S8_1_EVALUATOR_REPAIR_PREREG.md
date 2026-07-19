# R12 S8.1 Source-Level Nonce Repair Preregistration

**Status:** source freeze before a fresh board or score access

**Parent:** S8 nil-linked law graph v1

## Why S8 v1 is closed

The valid H100 fit in job `693462` completed both frozen 750-update arms and
wrote checkpoint SHA-256
`3c7154f2e31dd4f3e86534f8b007b7457585b85f7f7ffad4d13d8354721143af`.
The evaluator then opened the development file and completed original-source
forward passes, but failed before scoring or writing an evaluation because its
operation-nonce intervention assumed that contextual BPE spans have equal token
width. They do not. No result statistic exists. The development board is still
closed because access occurred; it may not be repaired or rescored. Its sealed
confirmation file remains unopened.

Jobs `693457` and `693459` are separate scoreless infrastructure failures.
They stopped in CUDA preflight on `evc25` and `evc26`, respectively, before
model or board access and wrote no checkpoint.

## Sole repair

S8.1 changes only operation-nonce recoding. It rotates nonce **strings** in all
card and event-operation source spans, adjusts every subsequent character span,
retokenizes the complete source with the frozen tokenizer, and recompiles token
targets. This is the intervention originally intended by the S8 preregistration.
It makes no assumption about token width or contextual segmentation.

The following remain bit-identical in design and frozen before a new seed:

- the 125,081,664-parameter base and S4 parser initializer;
- the 8,610,966-parameter graph compiler and 218-parameter generator;
- one epoch, 48,000 graph-only rows, batch 64, optimizer, and schedule;
- all role/rank labels, controls, causal perturbations, and parameter caps;
- all development thresholds in the S8 preregistration; and
- zero state, answer, recurrent, development-law, or confirmation-law training.

## Fresh-custody requirement

After this repair and its tests are committed, draw a fresh board seed and a
fresh training seed. Regenerate all bindings, law examples, names, train,
development, and sealed confirmation bytes. Before submission, execute the
source-level nonce recoder over every generated source and require:

1. successful retokenization and span recompilation;
2. unchanged graph semantics under consistent nonce rotation;
3. maximum recoded length at most 512 tokens; and
4. zero development and confirmation accesses.

One serial H100 development job is authorized on a CUDA-preflighted node. A
failure closes S8.1; passing every unchanged S8 gate authorizes only a separately
committed unchanged-weight sealed confirmation evaluator.

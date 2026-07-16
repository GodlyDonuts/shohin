# R12 Cursor-Conditioned Token-Tape Diagnostic

**Status:** frozen development-only candidate. It is not a confirmation,
architecture promotion, novelty result, internal-cursor result, or reasoning
claim.

## Motivation

The final-token linear diagnostic memorized train cells but reached only 43.13%
pre-final and 41.67% post-final on new operand/renderers, versus a 40% cursor
shortcut and 0/192 exact groups. This rejects a simple renderer-invariant code
at the final selector position. It does not test whether operation order remains
distributed over the prompt token states.

## Frozen data boundary

Use only the immutable confirmation-free development view and audit with
SHA-256 values:

- view: `24abd93737be57c6792a1d44c8f2e3a28d7c5fbc1666b083383350f410ce6ec9`;
- audit: `33fb4792ed0a8027d49de157c295cb9ba651cdd9c59ab5cfa04a71e99af8ea25`.

The H100 process receives no source canary, source audit, tokenizer path, or
confirmation row. Development contains all 24 operation permutations, so this
diagnostic cannot establish unseen-permutation extrapolation.

## Readout arms

All arms use frozen raw-260k token states and the externally supplied cursor.

1. **Shared token tape:** one query per cursor attends over valid prompt-token
   states; one shared linear decoder maps the attended state to five actions.
2. **Cursor-specific token tape:** the same attention, with an independent
   action decoder per cursor. This is a favorable upper bound.
3. **Mean joint:** masked mean pooling followed by the same cursor-indexed
   linear family used in the failed final-token diagnostic.
4. **Source-only tape:** one cursor-independent query and one shared action
   decoder.
5. **Cursor-only:** a five-by-five table with no token state.
6. **Embedding-only shared tape:** the identical 5,765-parameter shared probe
   over frozen input embeddings, separating lexical parsing from deep-state use.
7. **Position-only shared tape:** the identical probe over deterministic
   sinusoidal positions plus the real length mask.
8. **Source-deranged shared tape:** the identical probe receives a seeded
   derangement of source tapes while retaining each original label/cursor.
9. **Raw and token-RMS pre-final shared tapes:** fixed preprocessing controls
   for the primary train-standardized surface.

The shared arm has `5d + 5d + 5 = 5,765` trainable scalars at `d=576`. The
cursor-specific upper bound has `5d + 25d + 25 = 17,305`. These are ordinary
attention/gating mechanisms. A pass means token-tape readability with an oracle
cursor, not internal cursor state or autonomous reasoning.

Source-only and cursor-only are structural leakage bounds, not
parameter-matched competitors. Embedding, position, source-deranged, raw, and
RMS arms are the matched controls for the shared probe. Attention weights are
descriptive and do not establish causal clause retrieval; no clause-swap claim
is authorized by v1.

## Optimization

- seeds `2026071602`, `2026071603`, and `2026071604` for the shared and
  cursor-specific deep-state families; all controls use `2026071602`;
- 100 epochs, deterministic epoch shuffle;
- batch 256;
- AdamW, learning rate 0.03, betas 0.9/0.95, epsilon 1e-8;
- zero weight decay, gradient clip 5;
- primary train-valid-token feature standardization, plus frozen raw and
  per-token RMS controls;
- exact masks exclude right padding.

No output valve or full-vocabulary calibration is fit in this experiment.
Model states, query norms, preprocessing statistics, attention entropy/peaks,
and per-example development predictions must be preserved. The 1,152
train-derived mean/std scalars are reported as preprocessing state.

## Decision

A replicate passes only with at least 99% train cell accuracy, 95% development
cell accuracy, 90% exact five-action development groups, at least 95% on every
development renderer, and at least 95% separately at each of the four
non-DONE cursors. Source-only must be at most 20% and cursor-only at most 40%.
The shared or cursor-specific family passes only if at least two of three fixed
replicates pass. If fewer than two replicates reach 99% train accuracy, the
family is optimization-inconclusive rather than a representation no-go.

A deep shared-state claim additionally requires its median development cell
accuracy to exceed the embedding-only, position-only, and source-deranged
matched controls by at least 10 percentage points. Raw/RMS results determine
whether any apparent effect depends on train-derived feature scaling.

- A deep shared-tape pass supports only a shared attention-bottleneck v2 design.
- An embedding-only pass supports only a lexical controller design.
- Only a cursor-specific pass supports at most a hard-gated upper-bound v2.
- Only a mean-joint pass shows distributed aggregation without retrieval.
- If fitted deep families fail, stop the external-cursor/token-tape branch; if
  they do not fit train, report optimization-inconclusive.

Any v2 must be a new preregistration with held-out permutations and fresh
operand/renderers. No v1 confirmation reuse is authorized.

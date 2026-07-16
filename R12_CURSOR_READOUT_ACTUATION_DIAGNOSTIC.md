# R12 Cursor Readout / Actuation Diagnostic

**Status:** frozen development-only diagnostic. This is not a selector
confirmation, architecture promotion, reasoning result, or novelty claim.

## Question

Cursor-action v1 failed because its final-block, head-zero Q intervention moved
the five action logits by about 0.10 while the median gap to the
full-vocabulary winner was about 2.54. The next experiment must distinguish two
possibilities without tuning on the exposed v1 confirmation:

1. the frozen representation does not make operation order jointly readable
   with the supplied cursor; or
2. the representation is readable, but the Q-only path cannot write the result
   into vocabulary logits.

## Allowed data and custody

Only the immutable `train` and `development` splits from the v1 canary may be
used. A deterministic builder projects them into a separate hash-bound
artifact. An independent auditor rederives every projected field from the
source canary and rejects any `confirmation` key. The H100 process receives
only this confirmation-free artifact and its audit; it is not given the source
canary, source audit, or tokenizer paths.

The development-view and audit SHA-256 values are respectively
`24abd93737be57c6792a1d44c8f2e3a28d7c5fbc1666b083383350f410ce6ec9`
and `33fb4792ed0a8027d49de157c295cb9ba651cdd9c59ab5cfa04a71e99af8ea25`.

This diagnostic may choose a pre-final versus post-final feature surface on
development. A later score-bearing experiment must withhold operation-order
permutations as well as use fresh operands and renderers after one
implementation is frozen. V1 development contains all 24 order permutations,
so a pass here establishes only oracle-cursor-indexed linear separability
across new operands/renderers, not compositional extrapolation.

## Readout family

For frozen source feature `h in R^d` and cursor `c in {0,...,4}`, the joint
readout is

```
s_a(h,c) = <W[c,a], standardize(h)> + b[c,a].
```

It is a tensor-product linear probe with `5 * 5 * (d + 1)` trainable scalars.
It is deliberately favorable and diagnostic. It is a hard-gated linear
mixture of experts with an externally supplied cursor, not evidence that
Shohin represents a cursor internally and not an R12 primitive.

Controls are:

- source-only: one linear five-action readout shared across all cursors;
- cursor-only: a five-by-five bias table independent of source features.

The shortcut theorem bounds their deterministic development accuracy at 20%
and 40%, respectively.

## Readout and calibrated write requirement

1. **Restricted readout:** cross-entropy over the five probe scores. This tests
   oracle-cursor-indexed linear separability without vocabulary competition.
2. **Frozen-readout calibration:** freeze each restricted readout, then fit
   only two scalars in
   `delta_action_logits = softplus(raw_alpha) * restricted_scores + beta`.
   The loss is exact full-vocabulary cross-entropy. The non-action log-sum-exp
   and maximum are cached from the frozen base. This does not independently
   prove an internal actuation path: any positively separated finite readout
   can eventually beat a finite vocabulary margin by scaling. It instead
   measures the exact gain, common action bias, and intervention magnitude a
   direct write path would require.

The calibrated write is active only at the selector position in this diagnostic. It does
not update a cursor, execute arithmetic, emit COMMIT, emit DONE/EOS in one
call, or modify the flagship.

## Frozen optimization

- base: immutable raw 260k;
- seed: `2026071601`;
- 100 epochs;
- deterministic epoch shuffle;
- batch size 256;
- AdamW, learning rate 0.03, betas 0.9/0.95, epsilon 1e-8;
- zero weight decay;
- gradient clip 5.0;
- per-feature standardization from training sources only, standard deviation
  clamped to at least 1e-5.

Pre-final and post-final frozen features are both measured. No rank, layer,
threshold, seed, or optimizer search is allowed under v1.

## Development decision

Representation availability passes only if at least one joint restricted arm
has all of:

- at least 99% train cell accuracy;
- at least 95% development cell accuracy;
- at least 90% exact five-action development source groups;
- source-only development accuracy at most 21%;
- cursor-only development accuracy at most 41%.

The calibrated full-vocabulary fit is reported on the same feature surface and
must have the following before it can inform a v2 design:

- at least 95% full-vocabulary development cell accuracy;
- at least 90% exact five-action development source groups;
- at least 95% full-vocabulary cell accuracy on each development renderer;
- at least a 10 percentage-point cell-accuracy advantage over both valve
  calibrated controls.

The report must also preserve the frozen readout parameters, standardization
statistics, scalar calibration, per-example development predictions, base and
adjusted margins, delta magnitudes, runtime identity, and immutable input/code
bindings. The gain and bias are descriptive measurements, not an actuation
gate.

If representation fails, stop this branch. If it passes, the only authorized
next action is to use the measured write requirement to design a matched
direct-write v2 comparison of orbit/interchange training versus ordinary CE,
with held-out permutations and fresh confirmation. No result here authorizes
a reasoning, compositionality, novelty, or internal-cursor claim.

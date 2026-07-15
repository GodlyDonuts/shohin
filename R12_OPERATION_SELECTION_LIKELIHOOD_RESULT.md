# R12 Raw-260k Operation-Selection Likelihood Result

**Status:** complete, hash-bound, negative cursor-awareness diagnostic.

## Bottom line

The free-decoding cursor probe was format-confounded, so this separately frozen
diagnostic scored only four exact one-token operation candidates with one model
forward per prompt. Newton job `689796` completed all `528/528` forwards and the
independent full-result reconstruction passed.

The full-source arm is `80/176 = 45.45%`, versus `64/176 = 36.36%` for both
residual controls. That aggregate improvement is **not** evidence of an internal
operation scheduler. The full-source prediction changes across cursors in only
`1/112` adjacent cursor transitions and only `1/64` multi-step sources. It
recovers the complete operation schedule in `0/64` sources. It predicts `add`
145 times, `subtract` 31 times, and never predicts `multiply` or `remainder`.
Fifteen of sixteen multiply-subtract
sources receive `subtract` at both cursors. The apparent gain is therefore a
family-level lexical preference for `subtract`, not cursor-indexed schedule
recovery.

Raw 260k remains an externally steerable executor. It does not contain the
source compiler, cursor-dependent action policy, state transport, or halt policy
needed for autonomous reasoning.

## Custody

| Object | Value |
|---|---|
| Newton job | `689796` on `evc42`, `COMPLETED`, exit `0:0`, elapsed `00:04:19` |
| Frozen implementation commit | `7ad37cbd05a1683c3fd6a22377ae16094f7cc535` |
| Pre-score evidence commit | `59836bc3b76ff738dc6bc94d2492162f0ca4612c` |
| Result-receipt commit | `34573c1` |
| Cases / transitions / arms | `64 / 176 / 3` |
| Model forwards / candidate logits | `528 / 2,112` |
| Prompt tokens / generated tokens | `33,160 / 0` |
| Checkpoint | immutable raw 260k, SHA-256 `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d` |
| Result SHA-256 | `772050a9c30c229ff200f81895a01377c63a7e07a8ccc7e944afc54779bca5b6` |
| Receipt SHA-256 | `73e4241a00e40d4ed7491039f4b9410931a5e46164dc59c86ae07893857b3dd1` |
| Local result | `artifacts/eval_history/raw260k_operation_selection_likelihood_20260715_h100.json` |

The wrapper wrote and fsynced a score-free receipt before publishing either
score-bearing copy. That receipt was mirrored locally, validated, committed,
and pushed before the result was downloaded or opened. The local result hash
matches the receipt.

## Exact aggregate results

| Arm | Correct | Unique top-1 | Ties | Interpretation |
|---|---:|---:|---:|---|
| Full source + cursor | `80/176` | `176/176` | `0/176` | Only model-owned source-conditioned diagnostic |
| Residual suffix head | `64/176` | `176/176` | `0/176` | Literal-label copy control; predicts `add` everywhere |
| Residual suffix + oracle state | `64/176` | `176/176` | `0/176` | Literal-label/state control; predicts `add` everywhere |

The paired full-source comparison has 16 correctness gains and zero losses
against either control. The exact two-sided sign/McNemar probability is
`2 / 65536 = 0.000030517578125`. This establishes that source text changes the
restricted operation preference. It does **not** establish cursor-sensitive
selection because all 16 gains are the second `subtract` step in the same
multiply-subtract family.

## Full-source confusion matrix

| Gold operation | Predicted `add` | Predicted `subtract` | Predicted `multiply` | Predicted `remainder` |
|---|---:|---:|---:|---:|
| `add` | 64 | 0 | 0 | 0 |
| `subtract` | 16 | 16 | 0 | 0 |
| `multiply` | 49 | 15 | 0 | 0 |
| `remainder` | 16 | 0 | 0 | 0 |

The median gold margin to the best incorrect candidate is negative
(`-0.5659969` logit), and the mean is `-0.3044649`. Every prompt has a unique
restricted top-1, so ties do not explain the failure. Per-operation recall is
`100% add`, `50% subtract`, `0% multiply`, and `0% remainder`, for `37.5%`
macro recall.

## Cursor and family analysis

| Family | Full-source correct | Prediction behavior |
|---|---:|---|
| Multiply-subtract | `16/32` | 15/16 sources predict `subtract` at both cursors; one changes `add -> subtract` |
| Base conversion | `32/64` | predicts `add` at every cursor |
| Sequential state | `16/48` | predicts `add` at every cursor |
| Modular update | `16/32` | predicts `add` at every cursor |

All `112/112` adjacent gold operations change. The model changes its prediction
on only `1/112` of those transitions, and no source has a fully correct predicted
schedule. The residual controls change in `0/112`. Full-source micro-accuracy
also exactly equals the best family-constant and best index-constant shortcut
baselines, both `80/176`. A controller must distinguish operation order within a
fixed source; a family or position classifier cannot do that.

## Decision

This result closes the branch that assumed a pre-existing model-owned operation
signal could simply be stitched to the known atomic executor and a stop token.
No full controller fit is authorized from this diagnostic.

The next smallest admissible intervention is a separately preregistered
**counterfactual cursor-action induction** canary. It must use operation-order
twins with the same lexical inventory, cursor swaps within each source,
paraphrase invariance, source-blind and ordinary completion-loss controls, and
an exact held-out cursor-sensitivity gate. Training may target only control
boundaries; it may not use gold state at inference or modify the protected
flagship. Before any uninterrupted-chain experiment, it must show all of:

1. held-out source+cursor operation selection beyond the majority and matched
   completion-loss controls;
2. operation-order-twin separation rather than family-word classification;
3. cursor-dependent predictions on sources with multiple distinct operations;
4. preservation of the raw atomic executor; and
5. a separately trained and tested DONE/EOS boundary.

Passing that canary would establish only a learned action policy. Autonomous
reasoning still requires one uninterrupted model call that carries state,
advances its own cursor, executes each action, and halts without an external
scheduler, parser, repair loop, or verifier.

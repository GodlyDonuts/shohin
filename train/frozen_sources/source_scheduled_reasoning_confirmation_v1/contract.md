# Source-Scheduled Reasoning: Fresh Confirmation Contract

**Status:** frozen 2026-07-15 before board generation or model evaluation.
This is a capability-system confirmation, not an R12 novelty claim. The
controller is external, deterministic, and fully counted.

## 1. Development evidence and hypothesis

On the immutable 20-case raw-260k development board, a misleading `Next state`
renderer produced `input_state + 1` on 43/55 calls and scored 0/20 chains. A
fixed no-demonstration format matrix then obtained:

```
renderer          atomic transitions   full model-carried chains
Question/Answer       40 / 55                    7 / 20
bare equation          8 / 55                    1 / 20
Problem/Work          44 / 55                   10 / 20
```

`Problem/Work` is therefore frozen as the only scheduler renderer. A separate
crossed-prefix audit found six of six crossed add/multiply/subtract cells favor
the intervened visible state over the source-implied state. The hypothesis is
that raw Shohin contains a renderer-indexed visible-state arithmetic executor
that can be composed by a deterministic public operation schedule.

## 2. Fresh board

Generation seed is `2026071502`. Generate exactly 64 unique cases in each of
four families, 256 total, in frozen family order:

- `multiply_subtract`: `a in [20,99]`; first 32 use multiplier `[2,9]`, last
  32 use `[10,19]`; subtractor is positive and leaves a positive result.
- `base_conversion`: three-digit numerals; first 32 use bases `[2,9]`, last 32
  use `[10,12]`; every rendered digit is decimal and less than the base.
- `sequential_state`: start `[5,50]`, addend `[1,25]`; first 32 use multiplier
  `[2,5]`, last 32 use `[6,7]`; subtraction leaves a positive result.
- `modular_update`: two addends in `[10,99]`; first 32 use modulus `[3,14]`,
  last 32 use `[15,25]`.

The generator writes question, final answer, initial state, and exact public
operation schedule before loading a model. An independent structural audit in
the evaluator must reparse every question, replay every operation, verify the
answer, reject duplicates, and bind the board hash.

## 3. Frozen arms

All decoding is greedy. No demonstration, retry, repair, search, candidate
sampling, or verifier feedback is allowed.

1. **Direct QA:** one call on
   `Question: <question> Return only the final integer.\nAnswer:`.
2. **Whole Problem/Work:** one call on `Problem: <question>\nWork:`.
3. **Atomic oracle-state ceiling:** one independent `Problem/Work` call per
   public operation using the gold input state. This measures the executor and
   never contributes a state to the scheduled arm.
4. **Source-scheduled:** begin with the public initial state, issue one
   `Problem: Compute ...\nWork:` call per operation, parse the last integer on
   the first nonempty line, and carry only that model-produced integer into the
   next call. A parse failure terminates the chain. Gold intermediates are never
   shown to this arm.

The controller may parse the structured source and retain the operation
schedule, initial state, and its own model-produced integer. It may not retain
the final answer, gold intermediates, model activations/KV, or source text after
schedule extraction. Base conversion uses the explicit Horner schedule; this
algorithmic structure is external execution and must be reported as such.

## 4. Scoring and locked gate

For direct and whole-work calls, score the last integer before the first newly
generated `Question:` or `Problem:` header. For atomic calls, score the last
integer on the first nonempty line. Every transcript is preserved.

Let `S`, `D`, and `A` be source-scheduled final accuracy, direct final accuracy,
and oracle-state atomic transition accuracy. Advance to an internalization
experiment only if all conditions hold:

```
S >= 0.35
S - D >= 0.10
two-sided exact paired McNemar p(S versus D) < 0.01
S_family >= D_family in all four families
S_sequential_state >= 0.70
A >= 0.70
```

Any malformed board, hash mismatch, missing transcript, extra call, mutable
output, renderer deviation, retry, or near miss closes this version. No family,
threshold, parser, prompt, or schedule may be changed after scores are read.

## 5. Allowed claim and next step

Passing permits only:

> A deterministic counted scheduler exposes and composes source-free visible-
> state arithmetic already present in raw Shohin better than one-shot decoding
> on a fresh procedural board.

It does not establish standalone model reasoning, latent reasoning, context
compression, or a new primitive. The next experiment would have to train the
model to emit and execute the schedule itself, with the external scheduler as
the favorable control and matched total calls/tokens/FLOPs.

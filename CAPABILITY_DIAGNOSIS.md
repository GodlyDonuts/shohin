# Capability Diagnosis: 2026-07-12

## Bottom line

Shohin's raw checkpoint is a stable language model, not yet a general interactive
reasoner. The first broad SFT attempt did teach an answer-and-derivation style and
can solve some fresh multi-step arithmetic, but it does not reliably transfer that
skill to algebra, base conversion, logic, algorithms, or code.

The original SFT benchmark numbers were also materially understated: the shared
decoder stopped on any blank line, while many SFT targets deliberately put a blank
line before their final answer. That flaw is fixed in `train/eval_suite.py`; every
post-fix result must be kept separate from earlier scores.

The live pretrain is healthy and must remain untouched while these gates run.

## Evidence

### Raw 166.25k qualitative behavior

`pretrain_166250_qualitative_686255.json` shows that the raw checkpoint often:

- substitutes a memorized template for the supplied arithmetic values;
- completes a partially learned algebra trace but does not follow the requested
  task reliably;
- copies an input list rather than sorting/deduplicating it;
- repeats a generated pattern instead of applying a string transformation; and
- produces an incorrectly indented, semantically inverted `is_even` function.

It does retain fragments of symbolic knowledge: the linear-equation transcript
reaches the correct `x = 16/5` before its completion drifts. This is evidence of
partial token-level pattern learning, not robust multi-step control.

### SFT-v2 qualitative behavior

With the decoder fixed, `sft_v2_120k_ep1_fixed_qualitative_686263.json` shows a
real change in behavior:

- the discount-and-tax problem is solved correctly with a complete derivation and
  final `68.04`;
- the fraction problem is solved correctly as `4.5` cups;
- linear algebra starts in the right direction but loses the equation invariant;
- base conversion uses an invalid repeated-division procedure and returns the
  wrong representation;
- syllogism invents multiple-choice options;
- string and list tasks still degenerate; and
- the generated `is_even` reverses True/False.

So v2 taught *solution formatting plus a narrow arithmetic routine*. It has not
taught a reusable problem-solving process.

### Quantitative gates before decoder correction

These values are retained as diagnostics only, not as final capability claims:

| Checkpoint | GSM8K pass@1 | MATH-500 | HumanEval | MBPP | RG held-out |
|---|---:|---:|---:|---:|---:|
| raw 120k | 1/100 | 3/100 | 7/164 | 0/100 | 29/800 (3.625%) |
| v2 SFT, old decoder | 12/100 | 0/100 | 6/164 | 0/100 | 19/800 (2.375%) |

The GSM change is directionally consistent with the qualitative arithmetic gain,
but the old decoder cut many valid SFT completions at their first blank line.
Fixed-decoder public, held-out, and in-training gates are queued before using
these scores for a recipe decision.

## Ranked causes

1. **Benchmark decoder truncation**: `generate()` stopped on `"\n\n"`. This
   undercounted SFT completions that place their answer after a paragraph break.
   Fixed; all affected results are being rerun.
2. **Pretraining distribution is far too narrow for the stated target**: the live
   loader round-robins four directories, three math-oriented and one code-oriented.
   It contains no substantial curated logic, algorithmic, or instruction-following
   source. Directory round-robin makes this roughly 75% math-oriented / 25% code,
   independent of corpus size.
3. **The SFT curriculum is narrow and format-heavy**: v2 has 349,449 rows, but
   83,611 procedural traces come from only the six families with hand-built
   tracers. It contains only 444 code examples. This cannot plausibly move MBPP
   or HumanEval, and it provides little coverage for the 32-family held-out RG
   battery.
4. **The raw base has not been taught the interaction contract**: raw transcripts
   continue with unrelated `Question:` templates and mutate the prompt. A small
   completion model needs an explicit, diverse instruction/answer phase before
   benchmark-style interaction is reliable.
5. **Post-training is too shallow to repair missing substrate**: v2's one epoch is
   85.34M packed tokens. It can impose a style, but cannot create broad reasoning
   circuits absent from the base and absent from its data coverage.

## Rejected explanations

- **GPU utilization or microbatch size** is not the capability bottleneck. Live
  utilization is high; BS32 raises throughput only modestly.
- **Training instability** is not the main problem. Loss and gradient norms are
  stable, and isolated guard skips recover.
- **The SFT checkpoint was written to the wrong directory** is not the issue. Its
  header, checkpoint path, and filesystem prove it initialized from 120k and wrote
  only `train/sft_v2_120k/sft_ep1.pt`.

## Required next moves

1. Let the healthy raw run continue. At its next natural transition, include the
   verified `openmath_pt` corpus (5.0B tokens, 50 shards) with an explicit mixture
   policy; never change the live `SHARDS` in place.
2. Extend procedural supervision beyond six hand-written trace families. Add
   execution-verified derivations for logic, algorithms, strings, and code-like
   tasks before scaling the trace corpus.
3. Build a separate execution-verified code SFT source. The current 444 code rows
   are inadequate for HumanEval/MBPP.
4. Keep three fixed gates for every SFT candidate: direct qualitative transcript,
   fixed-decoder public board, and balanced RG in-distribution plus held-out tests.
5. Do not promote v2 as the final general-reasoning recipe unless the fixed gates
   overturn the current broad-generalization diagnosis.

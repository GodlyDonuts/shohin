# Capability Diagnosis: 2026-07-12

## Executive conclusion

Shohin is healthy as a training run but is not yet an intelligent general reasoner.
At raw step 166,250 it behaves primarily as a text-completion model: it can emit
fragments of familiar templates but does not reliably execute arithmetic, preserve
equation invariants, apply transformations, or follow a concise interaction
contract. The first v2 SFT pilot made its reasoning *look* more coherent and did
teach a few narrow routines, but it did not produce transferable problem-solving.

This is not a GPU-utilization or loss-stability failure. It is a curriculum,
coverage, and interface-contract failure. Continuing the current live pretrain is
safe, but simply continuing the same mix cannot be treated as the complete route to
the stated reasoning goal.

The live pretrain remains protected. All diagnosis, curation, and SFT work is
isolated from its checkpoint/output directory.

## What was measured

### Corrected benchmark status

The original shared decoder stopped at any blank line. Many SFT targets place a
blank line before `The answer is ...`, so those old scores are diagnostic only.
`train/eval_suite.py` now stops only after a complete explicit final-answer line.

The CUDA-only corrected public board (`686277`) has already completed its first
two GSM8K measurements for `sft_v2_120k/sft_ep1.pt`:

| Metric | Result | Interpretation |
|---|---:|---|
| GSM8K maj@4 | 6 / 100 | no useful self-consistency gain |
| GSM8K pass@1 | 14 / 100 | slightly above the old 12 / 100 diagnostic result, but still far below the target |

The remainder of that board plus fresh held-out/in-training RG gates must finish
before any numerical SFT promotion decision. Earlier results must not be compared
directly to corrected-decoder results.

### Direct interaction, not only benchmarks

I ran the raw 166,250 checkpoint and the v2 SFT checkpoint on twelve fresh,
hand-authored prompts using the same `Question: ... Answer:` contract used for SFT.
The complete verbatim transcript is
`artifacts/eval_history/interactive_v1_686293.json`.

| Capability | Expected | Raw 166.25k | v2 SFT |
|---|---|---|---|
| exact instruction | `saffron` only | restates the instruction | misspells and invents a letter-count template |
| 19 x 17 | 323 | 319 | 343 |
| linear equation | x = 3 | x = 3.5 | loses the equation mid-derivation |
| base-6 `254` | 106 | copies `254` | applies an invalid division procedure, returns 0 |
| syllogism | no | says no but gives a false explanation | says no, then contradicts itself with a multiple-choice answer |
| string insertion | `orcXYhard` | unrelated XOR template | returns `XYXY` |
| sort and deduplicate | `[2, 4, 9]` | copies most of the input | emits generic procedure without an answer |
| count `a` in `bananas` | 3 | claims ten letters | claims two occurrences |
| state tracking | 20 | computes 4 | computes 20, then adds unrelated narration |
| correct base-5 claim | no, 13 | returns 1000 | derives 13 but runs out before a final answer |
| minimal Python predicate | executable code | invalid/incomplete code | correct `n % 3 == 0` function |
| use supplied r = 14 | 42 | returns 38 | ignores r and switches to an arithmetic-series template |

The SFT model therefore has isolated wins (state tracking and a minimal code
predicate) and partial correct intermediates, but no robust rule execution. This
is direct evidence, not an inference from a loss curve.

### Training state and corpus replay

The raw checkpoint at step 166,250 has seen 87.16B tokens: 645.7 tokens per
nominal 135M parameters, or 696.7 per the 125.1M parameter count printed by the
actual SFT loader. The codebase/runbook headline should use **125.1M trained
parameters** unless a new checkpoint proves otherwise; calling it 135M does not
make the model stronger, and hides a roughly 10% target mismatch.

The live loader uses equal directory round-robin, not corpus-proportional sampling.
At step 166,250 it has drawn about 21.79B tokens from each enabled directory:

| Directory | Manifest tokens | Approximate passes so far | Consequence |
|---|---:|---:|---|
| `finemath4` | 2.00B | 10.90 | repeatedly replayed high-quality but narrow math |
| `openwebmath` | 14.06B | 1.55 | math web, not general educational language |
| `code_python` | 16.76B | 1.30 | raw code, not code instruction/completion pairs |
| `finemath3` | 25.00B | 0.87 | the largest math source has not completed one pass |

The active source mix is therefore exactly 75% math-oriented text and 25% raw
code. It has no substantial general educational English, logic/deduction, or
instruction-following pretraining source. That differs materially from the stated
strategy of a language floor plus a reasoning-tilted mix. It also makes the
smallest source replay almost eleven times while the best large math source has
not completed one pass.

## Why the first SFT did not repair it

The v2 pilot was cleanly isolated and trained as intended: 349,317 examples,
85.34M packed tokens, 64.29M answer-supervised tokens, and one 2,605-step epoch.
The loss fell from roughly 0.99 to 0.46-0.55. That demonstrates the model learned
to imitate the answers in the mix; it does not demonstrate broad reasoning.

The actual content makes the result unsurprising:

- 240,297 retained rows are OpenMath-derived, so arithmetic-style derivation is
  overwhelmingly represented.
- 83,611 procedural traces came from only six earlier hand-built families.
- The mix has only 444 code rows, and just 50 examples contain a code fence.
- Logic, strings, stateful algorithms, and error correction are too sparse to
  support the 32-family held-out RG battery.
- The answer-only loss is correct for SFT, but one epoch cannot install missing
  algorithms that neither the base nor the data has represented broadly.

There is also a code-specific contract error. `train/sft.py` teaches every example
as `Question: {problem}\nAnswer: {code}`, while `train/eval_code.py` asks HumanEval
for a raw Python continuation and MBPP with a separate `[BEGIN]` prompt. A 125M
model is highly sensitive to this mismatch. The code result remains genuinely weak
(the direct audit found a correct trivial predicate but the public board is low),
but it is additionally penalized by training and evaluation on different formats.
Future code SFT must include a verified raw-completion form matching the evaluation
contract, alongside instruction-form code examples.

## Latent reasoning / context compaction status

There is no trained latent-reasoning or self-compaction capability in the current
flagship. `GPTConfig.n_loop` exists as an experimental weight-shared repeat of the
block stack, but the live checkpoint has `n_loop=1`. It has never been trained or
validated with recurrence. The model has a conventional KV cache for inference;
that speeds token decoding but does not compact context or let the model summarize
its own reasoning state.

Switching `n_loop` on at inference would be an untrained architecture change, not
extended thinking. The correct path is a separate Mame-scale proxy ablation with
recurrence enabled during training, a fixed test-time loop budget, and gates against
an equally trained `n_loop=1` control. It must not be injected into the live run.

## Root causes, ranked

1. **Missing reasoning substrate and uneven replay in pretraining.** The active
   equal-domain mix is math/raw-code only, with severe replay imbalance. Stable
   loss here is not evidence of broad skill acquisition.
2. **SFT coverage is narrow and format-heavy.** It teaches concise derivation
   style more strongly than reusable algorithms. The direct transcript shows
   plausible prose without reliable state transitions or invariants.
3. **Code is underrepresented and prompt-misaligned.** Four hundred forty-four
   code rows cannot move HumanEval/MBPP, and the SFT/eval prompt mismatch wastes
   what little code supervision exists.
4. **Early evaluation was partially invalid.** The blank-line stop depressed
   older SFT scores. This was fixed before drawing the current conclusion, but it
   delayed a clear diagnosis.
5. **The advertised latent-reasoning feature is only dormant scaffolding.** It
   has no trained behavior and no context-compression objective.
6. **Capacity is constrained.** The model is 125.1M parameters, not a general
   frontier model. The viable route is a focused math/code/logic specialist with
   exceptional data and calibrated decoding, not an unsupported claim of broad
   general intelligence at this stage.

## Rejected explanations

- **GPU underutilization:** rejected. The live H100 holds 99-100% utilization;
  BS32 is validated and provides a modest throughput increase.
- **Training divergence:** rejected. Loss and gnorm are stable; isolated guard
  skips recover immediately.
- **Incorrect SFT output routing:** rejected. The pilot initialized from the
  intended 120k checkpoint and wrote only to `train/sft_v2_120k/`.
- **A single bad prompt or parser:** rejected. The raw and SFT failures recur
  across fresh direct prompts, corrected GSM8K, and the broad procedural gate.

## Remediation plan and promotion gates

1. **Keep the protected pretrain running.** Do not rewrite its live shard list.
   At a natural checkpoint handoff, use explicit domain weights to prevent
   FineMath4 replay from dominating and add the already-tokenized 5.0B-token
   `openmath_pt` source.
2. **Restore a language/instruction floor.** `fineweb_edu_probe.sbatch` is
   validating a decontaminated educational-English source before any tokenization.
   A full source is admitted only after schema, contamination, manifest, and
   quality checks; it remains future-relaunch-only.
3. **Replace the six-family procedural SFT component.** `rg_v4` already contains
   374,659 answer-checked, deduplicated traces across 25 families. Build a new
   frozen mix from it, not from writer files or the older six-family set.
4. **Scale and audit code before a new SFT.** The first APPS scan retained 234
   verified rows from 5,000 candidates, so a 75,000-candidate isolated scan is
   running rather than pretending the pilot met its 3,000-row goal. CodeContests
   is a separate train-only, execution-verified source. Neither enters a mix
   until its final quality report is clean.
5. **Use explicit source balancing for v4 SFT.** Do not let code remain a
   sub-1% residue merely because the math corpus is larger. Support source-level
   sampling or controlled duplication, and include a raw-code-completion template
   that matches code evaluation. Freeze the resulting mix with hashes and a
   contamination report before training.
6. **Gate every candidate on four comparisons.** Require: corrected public board,
   balanced held-out RG, a fixed direct-interaction transcript, and code execution
   under a prompt format represented in training. Promote only if it improves the
   relevant axes without a material regression on the others.
7. **Treat latent reasoning as an ablation, not a promise.** Train and compare
   `n_loop=1` versus `n_loop=2` at Mame scale first. No live architecture change
   is justified until that measured proxy wins.

The immediate success criterion is not a prettier loss or longer derivation. It is
repeatable improvement on fresh answer-checked tasks with direct transcripts that
show correct state updates, transformations, and final answers.

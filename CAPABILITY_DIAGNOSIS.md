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

The CUDA-only corrected public board (`686277`) completed for
`sft_v2_120k/sft_ep1.pt`:

| Metric | Result | Interpretation |
|---|---:|---|
| GSM8K maj@4 | 6 / 100 | sampling does not create a useful self-consistency gain |
| GSM8K pass@1 | 14 / 100 | format-sensitive improvement, still far below the target |
| MATH-500 pass@1 | 6 / 100 | narrow arithmetic style does not transfer to contest math |
| HumanEval pass@1 | 6 / 164 | code remains weak |
| MBPP pass@1 | 0 / 100 | no usable simple-program synthesis yet |

Earlier results must not be compared directly to corrected-decoder results. The
corrected board is enough to reject v2 as a promotion candidate. Its corrected
held-out RG result is **90/800 = 11.25%**, which is above the raw 120k baseline
(29/800 = 3.625%) but remains highly concentrated: chain sums 20/25, string
insertion 19/25, basic arithmetic and decimal-chain sums 13/25 each, and products
9/25. It is zero on most transformation, logic, cipher, geometry, and search-like
families. The in-training control is only **98/800 = 12.25%**, so the roughly
one-point gap does not support an exact-trace-memorization explanation. The pilot
did learn a few routines that transfer within its limited family coverage; it did
not learn a general algorithmic substrate, which is why the public board still
rejects the recipe.

The first raw-base board attempt (`686314`) is invalid as a board: it loaded the
rotating `ckpt_0168000.pt`, completed only GSM8K maj@4 at 1/100, then the source
checkpoint was deleted by normal rotation before the other four metrics loaded.
This is an evaluation lifecycle defect, not a model result. The evaluator now
pins its source before decoding. The corrected raw board (`686315`) ran from a
reflink-pinned `best_step168750.pt` and completed cleanly: GSM8K maj@4 **5/100**,
GSM8K pass@1 **2/100**, MATH-500 **2/100**, HumanEval **7/164**, and MBPP
**0/100**. This is the valid raw baseline for the v4 SFT experiment, not the
rotated-checkpoint partial result.

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

### V4 has procedural signal, but is not a broad promotion

The corrected V4 r3 held-out procedural evaluator (`686337`) scored **209/800 =
26.125%**, compared with the corrected V2 held-out evaluator's **90/800 =
11.25%** on the same `rg_v2/rg_eval.jsonl` sample seed and complete-answer
decoder. Its strongest gains are simple equations (1 -> 17 / 25), number sorting
(0 -> 16), isomorphic strings (0 -> 16), AIW word problems (0 -> 14), and
decimal-chain sums (13 -> 20). Basic arithmetic and LCM each changed by only one
item downward.

This is a real reason to retain V4 as a diagnostic/generator candidate, but not
proof that the V4 data alone caused the gain: its base checkpoint is raw 168.75k,
whereas V2 used raw 120k. More importantly, V4's public board is still weak
(GSM8K majority@4 5/100, MATH-500 1/100, HumanEval 2/164, MBPP 0/100), so it is
not a broad promotion candidate. Its adaptive direct interaction (`686338`) is
also only 1/6 initial, 1/6 review, and 0/6 scaffolded. The remaining matched
capability matrix will separate prompt-format effects from this local procedural
transfer signal.

That matched matrix and the raw-versus-V4 transcript are now complete. On the
same 48 cases and seed, raw 168.75k scored Q/A 4/48, direct 5/48, CoT 0/48, and
one-shot 7/48; V4 scored 4/48, 4/48, 4/48, and 10/48. The V4 CoT gain is four
arithmetic cases, while the one-shot gain is mainly syllogisms; native Q/A and
direct instruction remain at or below the raw checkpoint. The complete
transcript (`interactive_raw_vs_v4_168750_686343.json`) shows why the aggregate
is not enough: it turns `saffron` into `nroffas`, emits `19 * 17 = 303`, copies
base-6 `254` as decimal 254, counts one `a` in `bananas`, and fails to carry
`r = 14` into `3r`. V4's extra structure is sometimes useful for sampled
procedural tasks, but it is not a dependable execution trace.

### V5 proves missing primitives are trainable, not that reasoning is solved

The source-balanced V5 ablation retained broad V4 sources and added a 30% share
of 210,000 solver-verified primitive examples. Its 3,500-row held-out bank uses
separate prompts, random seeds, and numeric ranges. On the fixed 700-case
sample, raw scored **0/700** and V5 scored **272/700 = 38.86%**. The family
breakdown is decisive: syllogism **100/100**, string insertion **88/100**,
correction **47/100**, state update **19/100**, arithmetic **12/100**, base
conversion **4/100**, and sort/deduplicate **2/100**.

Its direct interview reached **2/8 initial, 2/8 review, 1/8 scaffold, and 3/8
compact-state reuse**, against raw 170k's 1/8, 0/8, 1/8, and 0/8. The three
reuse successes are state update, sort/deduplication, and precedence correction:
all close to explicit V5 curriculum families. It still emits `43 * 17 = 701`,
fails base-8 conversion and string splice, and cannot write a valid minimal
Python predicate. V5 therefore establishes that explicit compact-state
supervision can install a few execution moves; it does not establish autonomous
compaction, broad self-correction, or general reasoning. Its prompt matrix and
public board remain the promotion gates.

On the fixed 48-case matrix, V5 moves native Q/A **4/48 -> 17/48** and explicit
CoT **0/48 -> 11/48**, driven by arithmetic, sorting, and state updates. Plain
direct instruction is only **5/48 -> 6/48** and one-shot falls **7/48 -> 5/48**.
The result is a useful warning against reporting only the best prompt: V5 has
learned some execution under the supervised Q/A or CoT contract, but has not
become a reliable instruction-following solver.

The completed public board makes the promotion decision unambiguous. V5 (`686401`)
scored GSM8K majority@4 **10/100**, greedy GSM8K **9/100**, MATH-500 **3/100**,
HumanEval **2/164**, and MBPP **0/100**. Relative to the pinned raw 168.75k board
(5/100, 2/100, 2/100, 7/164, and 0/100), that is a narrow arithmetic-format gain
alongside a severe code regression. V5 is rejected as a broad SFT recipe.

A separate fresh seven-case transcript probe (`686425`) was run after the board
rather than inferred from aggregate metrics. Raw 168.75k scored **1/7 initial,
0/7 review, 1/7 after a verified fact, and 0/7 state reuse**. V5 scored **3/7,
3/7, 2/7, and 3/7**. Its exact wins are the trained arithmetic, sorting, and
logic patterns; it still fails base conversion, sequential state updates, string
insertion, and syntax-valid Python. V5 normally does not emit the requested
`state=` representation. Its reuse wins are final-answer matches after a new
prompt, not evidence that it produced or faithfully continued from a compact
state. The hash-matched transcript is
`artifacts/eval_history/manual_capability_raw168750_vs_sft_v5_20260712_JOBID.json`
(md5 `28dd0b15de2af16a10a2012f630072a1`).

### Controlled prompt matrix at 168k

The first twelve hand-authored prompts established the failure qualitatively. A
second reproducible audit (`train/capability_matrix.py`, job `686306`) then used
48 fresh generated tasks across arithmetic, base conversion, state updates,
sorting/deduplication, string insertion, and syllogisms. It tested the raw 168k
checkpoint and v2 SFT under four prompt contracts. The complete transcripts and
per-family scores are in `artifacts/eval_history/capability_matrix_v1_686306.json`.

| Checkpoint | Q/A contract | Plain instruction | Ask for chain of thought | One worked example |
|---|---:|---:|---:|---:|
| raw 168k | 4 / 48 (8.3%) | 4 / 48 (8.3%) | 0 / 48 (0.0%) | 5 / 48 (10.4%) |
| v2 SFT | 7 / 48 (14.6%) | 4 / 48 (8.3%) | 5 / 48 (10.4%) | 4 / 48 (8.3%) |

The raw model only solved four negative syllogisms under its native Q/A format.
The v2 model gained three arithmetic cases only in that exact format; it scored
zero on all eight base-conversion, state-update, sorting, and string tasks in the
same condition. A request to think step by step did not unlock latent computation.
For example, raw 168k correctly wrote `18 + 9 = 27`, `27 * 5 = 135`, and
`135 - 14 = 121`, then continued into a different question and emitted a final
`1`. V2 instead applied the wrong precedence (`18 + 9*5 = 63`). This separates a
weak output/stopping contract from the deeper missing algorithmic competence.

### Multi-turn correction and scaffold test at 168.75k

I also interacted directly with the same preserved raw checkpoint through a
six-case, three-turn audit (`interactive_adaptive_168750_686316.json`). Each
case received an initial question, an explicit independent-review request using
its prior answer, and a fresh version with one verified intermediate fact. The
scores were **1/6 initial**, **1/6 review**, and **1/6 scaffold**. The only exact
success in all three conditions was the simple negative syllogism.

The failures identify the missing operation rather than merely a bad stopping
token. For `27 * 14 + 9`, it asserted `27 * 14 = 398`; on review it repeated the
same result; with the verified product `378` it repeated the fact but did not add
9. It maps base-7 `356` to `1000`, turns a state update into repeated additions,
returns a generic `[1,2,3,4,5,6]` for an unrelated sort/deduplicate task, and
collapses a string insertion into empty code fences or `pq`. Thus review and
provided state do not activate an unexpressed solver. The model needs training
on state transitions, transformations, answer contracts, and correction moves;
prompt engineering alone is not a credible remedy.

### Compact-state interview at 170k

To test the specific latent-reasoning/compaction claim rather than infer it from
the earlier six cases, job `686370` ran a second pinned raw checkpoint interview
against `best_step170000.pt`. It used eight fresh cases across arithmetic,
base-8 conversion, state transitions, sort/deduplication, string splice, logic,
counterexample correction, and a minimal Python contract. Each case was tested
as an initial answer, after independent review, with a verified intermediate
fact, and after the model had been asked to create then reuse a compact `state=`
representation.

The canonical, syntax-checked result is **1/8 initial, 0/8 review, 1/8
scaffolded, and 0/8 compact-state reuse**. The sole success is the simple logic
constraint. The model repeats `43 x 17 = 651`, treats base-8 `725` as 8 or 1000,
uses wrong operator order for the state transition, and emits generic code/search
templates for list and string tasks. Its apparent initial code success in the
first instrumentation pass was rejected: the generated `is_even` body was not
syntactically valid Python. The scorer was tightened to parse the function AST
without executing model-produced code, then the interview was rerun from the
same pinned checkpoint. The transcript is
`artifacts/eval_history/deep_interaction_raw170k_r2_686370.json` (md5
`1979bcc79cb18830cb3080a7cab85e82`).

This is direct negative evidence for the desired feature: the current model does
not create a usable internal summary, cannot continue faithfully from one it
generated, and does not repair simple errors when prompted. It does not rule out
training an explicit compact-state curriculum later; it rules out claiming that
the capability already exists.

### Training state and corpus replay

At step 168,300 the run has processed 88.24B nominal tokens, or 705.4 tokens per
the 125.1M parameter count printed by the actual SFT loader. The codebase/runbook
headline should use **125.1M trained parameters** unless a new checkpoint proves
otherwise; calling it 135M does not make the model stronger, and hides a roughly
10% target mismatch. Mean training loss has been essentially flat across the
extension: 1.659 (60k-80k), 1.655 (80k-100k), 1.663 (100k-120k), 1.645
(120k-140k), 1.635 (140k-160k), and 1.640 so far after 160k. This is a healthy
optimization trace, but it is not evidence of capability growth.

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
strategy of a language floor plus a reasoning-tilted mix. At 168,300, the equal
four-way loader has supplied about 22.06B tokens to each source: FineMath4 has
been replayed 11.03 times while FineMath3 has received only 0.88 pass.

There was an additional handoff defect. Checkpoints stored model and optimizer
state but not the asynchronous loader cursor. Every resumed Slurm job constructed
`ShardLoader` with the same `DSEED=777`, so it could restart the shuffled stream
from its beginning. The exact replay fraction cannot be recovered after the fact,
but the logs confirm multiple completed handoffs with that fixed seed; treating
their nominal token count as fully new data would be unjustified. This is now
fixed forward-only: checkpoints record a data-stream generation and every resume
uses a deterministic new stream seed. It prevents repeated stream prefixes but
does not pretend to serialize an exact prefetched byte cursor. The active job is
not modified; the first next handoff from an older checkpoint becomes generation
1 and therefore cannot reuse its seed-0 ordering.

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
- This older frozen mix predates source-balanced sampling: all 349,449 rows have
  no `training_group`, so its 2,605 packed sequences were shuffled in their
  natural source proportions rather than deliberately sampling code, procedural,
  math, and teacher supervision.
- Logic, strings, stateful algorithms, and error correction are too sparse to
  support the 32-family held-out RG battery.
- The answer-only loss is correct for SFT, but one epoch cannot install missing
  algorithms that neither the base nor the data has represented broadly.

The completion-mask implementation itself was checked on 1,360 deterministic
samples from the v2 mix: every tokenized prompt was an exact prefix of its full
prompt-plus-answer tokenization. The poor result is not caused by a shifted label
boundary or accidental prompt-token supervision.

There is also a code-specific contract error. `train/sft.py` teaches every example
as `Question: {problem}\nAnswer: {code}`, while `train/eval_code.py` asks HumanEval
for a raw Python continuation and MBPP with a separate `[BEGIN]` prompt. A 125M
model is highly sensitive to this mismatch. The code result remains genuinely weak
(the direct audit found a correct trivial predicate but the public board is low),
but it is additionally penalized by training and evaluation on different formats.
Future code SFT must include a verified raw-completion form matching the evaluation
contract, alongside instruction-form code examples.

The first v4 code-completion pilot uncovered a second, more subtle contract bug
before any candidate artifact was accepted. For **461 of 3,542** completion-form
code rows, the tokenizer's IDs for the separately encoded prompt were not a
prefix of IDs for `prompt + completion`, most often at CRLF plus indentation.
The old packer computed a prompt-length mask from the former but trained the
latter, shifting labels at exactly the code boundary that matters. The pilot was
canceled and preserved as invalid. `train/sft.py` now independently encodes the
prompt and continuation and concatenates those IDs, which matches autoregressive
inference exactly; `test_sft_prompt_boundaries.py` covers a normal Q/A boundary
and the CRLF Python case. The clean v4 rerun starts from the same raw checkpoint
only after that regression test passes locally and on Newton.

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

That first mechanical ablation is now complete. Identical 31.5M Mame runs over
800 steps with the same data seed were both stable and each had one recovered
grad-norm skip. `n_loop=1` finished in 886 seconds at 472.7k tok/s with final
logged loss 2.4899; `n_loop=2` finished in 1,466 seconds at 286.0k tok/s with
final logged loss 2.4890. This proves the implementation can train recurrently,
but not a capability benefit: the 1.65x wall-time cost has no measurable short-run
loss advantage. Keep recurrence off the flagship until a longer paired capability
evaluation earns that cost.

## Root causes, ranked

1. **Missing reasoning substrate and uneven replay in pretraining.** The active
   equal-domain mix is math/raw-code only, with severe replay imbalance and,
   before the forward-only handoff fix, potentially repeated stream prefixes.
   Stable loss here is not evidence of broad skill acquisition.
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
7. **The model has not learned a usable compact-state protocol.** A later turn
   can be correct without using the earlier model text; the transcript must
   validate both the emitted state contract and faithful continuation before any
   score is described as latent reasoning.

## Rejected explanations

- **GPU underutilization:** rejected. The live H100 holds 99-100% utilization;
  BS32 is validated and provides a modest throughput increase.
- **Training divergence:** rejected. Loss and gnorm are stable; isolated guard
  skips recover immediately.
- **Incorrect SFT output routing:** rejected. The pilot initialized from the
  intended 120k checkpoint and wrote only to `train/sft_v2_120k/`.
- **A single bad prompt or parser:** rejected. The raw and SFT failures recur
  across fresh direct prompts, corrected GSM8K, and the broad procedural gate.
- **A broken SFT label mask:** rejected. The prompt-prefix/token-mask audit found
  no sampled boundary mismatch.
- **A magical prompt or hidden latent mode:** rejected. One-shot prompts gave the
  raw model only 5/48 and asking for chain of thought gave it 0/48; v2 remained
  tied to its Q/A template.

## Throughput reality check

The H100 is not idling: the live BS32/ACC8 run is holding about 154.2k tokens/s
with 99-100% reported GPU utilization. At that rate it processes about 13.32B
tokens/day. A claim of 30T tokens in ten days would require 34.72M tokens/s,
225x this run's already-saturated rate, and would take about 6.17 years here.
That claim is therefore not a comparable single-GPU pretraining result; it is not
evidence that a missing graph-fusion flag explains the capability gap. The
measured BS32 change gained about 4%, and the whole-update CUDA graph canary only
about 1.8%, which is why they are not the central remediation path.

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
5. **Use measured source balancing for v4 SFT.** Do not let code remain a
   sub-1% residue merely because the math corpus is larger, but do not turn a
   small verified set into a memorization loop. The frozen v4 mix has 62,926
   packed sequences: math 34,848, procedural 24,847, code 1,225, teacher 2,006.
   Its pilot uses 40/47/8/5 math/procedural/code/teacher, which gives code about
   4.1 replays and teacher 1.6 per epoch rather than the unearned 7.7/3.1 of the
   prior 40/35/15/10 proposal. It includes raw-code-completion templates that
   better match code evaluation and is frozen with quality/contamination reports.
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

# Raw-260k Interaction Capability Map

**Status:** static, read-only behavioral audit of the existing raw-260k
manual, continuation, renderer, scheduler, and packet-interface artifacts.
This is development evidence. It is not a public benchmark, a training result,
an architecture localization, or evidence of broad or latent reasoning.

## 1. Bottom line

Raw 260k has one narrow, reproducible positive behavior: under the familiar
`Problem: ...\nWork:` renderer, it often applies a supplied arithmetic operation
to a visible integer state. A deterministic external scheduler, first-line
integer parser, and state carrier compose that behavior into **10/20** correct
development chains. The strongest family is sequential add/multiply/subtract:
**15/15** gold-state atomic transitions and **5/5** model-carried chains.

The result does not reduce to one global failure label:

- **Executor:** strong for sequential add/multiply/subtract, mixed for general
  multiplication/subtraction, and weak for remainder. It is not a general
  arithmetic executor.
- **Compiler/selector:** ordinary task prompts often omit an operation, treat a
  non-decimal numeral as decimal-looking text, or enter an unrelated template.
  External schedule compilation materially changes the result.
- **Updater:** the external parser/carrier can recur over visible integers, but
  the model fails the explicit compact `State`/`Plan` packet update contract
  **0/2**. These are different capabilities.
- **Terminator/serializer:** correct content often appears before unrelated
  continuation. All 110 `Problem/Work` atomic-plus-chained calls contain text
  after the parsed first line, and the packet halt response starts with the
  correct integer but does not stop.
- **Parser:** there are two separate parser facts. The model-facing integer
  parser succeeds on every scheduled development call. The original
  continuation evaluator had its own numbered-header bug; the hash-bound
  assessor corrects direct QA from **1/20** to **4/20** without regenerating a
  response. That discrepancy is an evaluator error, not an executor, compiler,
  or updater error.

The strongest justified description is therefore **renderer-indexed,
externally scheduled visible-state execution on a small development board**.
It is not autonomous model recurrence or broad reasoning.

## 2. Behavioral definitions

These labels describe transcript behavior. They do not assert separable neural
modules.

| Label | Observable contract |
|---|---|
| Compiler/selector | Convert a source task into the correct initial state, ordered operations, operands, and output contract. |
| Executor | Apply exactly one requested operation to the displayed state. |
| Updater | Consume an observed result, replace the current state, remove the completed operation, and preserve the remaining plan. |
| Recurrence | Feed a model-produced visible state into the next operation. In the scheduled artifacts this is done by external code, not hidden model state. |
| Terminator | Stop at the requested answer or state boundary. |
| Parser | Extract the state used by the external scheduler or score the requested answer segment. A parser success is not model format compliance. |

`Atomic gold-state` calls use the correct input to isolate the executor.
`Chained local-operation` correctness asks whether the model correctly applied
an operation to the state it actually received, even if an earlier error made
that state globally wrong. `Full chain` requires the final visible state to
equal the gold answer.

## 3. Evidence custody

The local checkpoint and tokenizer were rehashed directly:

| Object | SHA-256 |
|---|---|
| `train/flagship_out/ckpt_0260000.pt` | `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d` |
| `artifacts/shohin-tok-32k.json` | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Frozen 20-case payload | `3bae0add841e403d01251ae6e6ff110f3c6a07324b28de1b671a59f012071f7c` |

Every model-bearing artifact below records checkpoint step `260000` and the
same checkpoint/tokenizer SHA-256 pair, except the older manual artifact, which
records the checkpoint path and step but predates those hash fields.

| Evidence | Role | File SHA-256 |
|---|---|---|
| `manual_capability_raw260k_20260715_mps.json` | Seven direct/review/fact/state/reuse tasks | `42590202834294cea182821f09613503c5ca91f6a1676d020d9f2cc2100c0aac` |
| `raw260k_continuation_modes_20260715_mps.json` | Exploratory four-case, five-format discovery probe | `f462391f3351a8491955587c036e7579559deb6d52e3c44827a236f245d41290` |
| `raw260k_continuation_confirmation_20260715_mps.json` | Immutable 20-case, 60-generation confirmation transcript | `f333c8f54383c411813551bc2001077b88e49514923b76c3cfe0331e9fd6bb47` |
| `raw260k_continuation_confirmation_20260715_mps.assessment.json` | Corrected, transcript-bound first-segment assessment | `058aa9dafdc741efc181e6377db5d46b233875504b4b4b6d92837a0db71ea62b` |
| `raw260k_ssc_diagnostic_20260715_mps.json` | Failed `Next state` scheduled renderer | `a152e85294d02173a697e29d8537bf4b53428d747d16c7e3baf692095d9b6a2f` |
| `raw260k_atomic_operation_formats_20260715_mps.json` | Three-renderer atomic and model-carried matrix | `b33c26b3963296c0d97b2a6d3332c0be18af40f460137c25652b881824a1ca4b` |
| `raw260k_renderer_interchange_20260715_mps.json` | Crossed visible-state likelihood diagnostic | `963177139b6abb333710f0db19a521c341a039fce3f65743ebdd698be6f12170` |
| `raw260k_residual_packet_interface_20260715_mps.json` | Five-call compiler/updater/halt interface probe | `1ca48442013a69f8fa53e25a0e063ea38063d7cd9e245c731b2b5fa295e1376c` |
| `raw260k_updater_subskill_probe_20260715_mps.json` | Twelve-call copy/delete/joint updater decomposition | `4505602994a0e337b99359e580a6f2f04fad4d365b2dac59f4c339fac13a7593` |
| `raw260k_updater_subskill_probe_20260715_mps.assessment.json` | Transcript-bound exact updater assessment | `26da3205d50301f6a9accf27ed22d4a6d92d7efcc451460bb1a73bb02dcff536` |
| `source_scheduled_reasoning_confirmation_v1.json` | Immutable 256-case board only | `19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474` |

The corrected assessor binds the unchanged transcript, case payload, and
generation script (`d4bdb45be00d161f1948c7f9669f7ced3201d33a199c8301c7b64634b34ac801`).
The 256-case file has 256 board rows and **zero response fields**. It is not a
scheduled transcript and contributes no model score to this map.

## 4. Manual interaction map

The seven-case manual artifact scores **1/7 initial**, **0/7 independent
review**, **1/7 supplied verified fact**, and **0/7 state reuse**. No compact
response begins with the exact requested `state=` prefix. The sole initial
answer-level success is logic; its text starts `No` but immediately gives the
false rationale that a zol is not a mar, so even that success is not a stable
faithful derivation.

| Task | Prompt form | First correct state or output | Recurrence, termination, and format | Primary behavioral error |
|---|---|---|---|---|
| `29 * 16 - 37` | Direct QA, review, supplied product, compact state, reuse | None. Direct emits `29 times 16 = 496`; the supplied-fact arm emits `464` but never subtracts 37. | Repeats numbered questions; compact product is `436`; reuse continues a multiplication template. | Executor error on product, then selector/updater failure to perform the second operation; termination failure. |
| Base-6 `425` to decimal | Same five manual forms | None. Direct returns the numeral `425`; the supplied decomposition is echoed but not evaluated. | Compact state uses decimal-looking `400 + 20 + 5`; reuse returns `400`. | Compiler/representation failure, with no successful final executor chain. |
| Start 14, add 9, multiply 3, subtract 20 | Same five manual forms | First addition is correct: `14 + 9 = 23`. | Direct skips multiplication and computes `23 - 20 = 3`; supplied fact stops at `n = 23`; no usable state is emitted or reused. | Compiler/selector omission and failed recurrence on this case. Later controlled cases show that the executor itself is not absent. |
| Sort and deduplicate | Same five manual forms | None. | Copies the unsorted list, then enters malformed Java/C++ templates; supplied distinct values are not sorted. | Executor plus response-mode/format failure. |
| String insertion | Same five manual forms | None. | Enters repeated `Input:` number templates; the supplied prefix/suffix arm returns `lantern`. | Compiler/selector and format failure. |
| Zol/mar/tiv syllogism | Same five manual forms | Initial answer token `No`; supplied-fact answer is also `No`. | Initial rationale contradicts a premise; review loses the answer; compact/reuse alternates contradictory `State:` lines. | Answer token is available, but review, faithful constraint use, state serialization, and recurrence are unstable. |
| Python divisibility predicate | Same five manual forms | Predicate concept `n % 3 == 0` is present. | Generated function is not syntax-valid because indentation is missing and extra driver/explanation text follows; reuse emits `10`. | Code compiler/serializer and format failure, not absence of the predicate concept. |

This manual board alone would support a negative verdict. The controlled
arithmetic artifacts explain why that verdict is too coarse for one narrow
family.

## 5. Whole-task prompt form

The corrected 20-case confirmation uses five fixed-seed cases in each family.
`Intermediates` means every required intermediate value is present in the
assessed first answer segment. Worked completion includes two solved examples,
so it is not a pure format-only intervention.

| Family | Direct QA final / intermediates | Bare final / intermediates | Worked final / intermediates | What first becomes correct |
|---|---:|---:|---:|---|
| Multiply then subtract | `0/5` / `2/5` | `0/5` / `0/5` | `1/5` / `2/5` | Product is correct in two cases; only one continues to the correct subtraction. |
| Base conversion | `0/5` / `0/5` | `0/5` / `0/5` | `0/5` / `0/5` | No required place-value state appears in any mode. |
| Add, multiply, subtract | `4/5` / `5/5` | `0/5` / `1/5` | `5/5` / `5/5` | The first addition and every later state are correct in all direct and worked cases. One direct case continues from correct `55` to `55 / 3 = 18`, making it a strict termination failure. |
| Add then remainder | `0/5` / `5/5` | `1/5` / `0/5` | `2/5` / `5/5` | The sum is correct in all direct and worked cases; the remainder is correct only `2/5` worked. |
| **All** | **`4/20` / `12/20`** | **`1/20` / `1/20`** | **`8/20` / `12/20`** | Narrow state traces are more available than strict finals. |

The byte-frozen v1 transcript embedded direct `1/20`. Its parser failed to
terminate on numbered headers such as `Question 2:` and included later text in
the answer segment. The separate v2 assessor finds direct `4/20`; no response
was regenerated. The model still has one genuine strict termination miss:
direct sequential case 3 contains the correct complete chain and final `55`,
then performs an unrequested division before the first new header.

Observed continuation beyond the corrected first answer segment is **18/20
direct**, **3/20 bare**, and **20/20 worked**. Segment parsing prevents most of
that run-on text from changing the score, but the behavior is still a model
termination defect.

## 6. Scheduled renderer map

### 6.1 Failed `Next state` renderer

The first scheduler prompt used:

```text
Current state: <integer>
Requested operation: <operation>
Apply exactly this one operation. Return only the next integer.
Next state:
```

The external integer parser succeeds **55/55**, but the model emits
`input_state + 1` on **43/55** calls. Only **1/55** outputs is accidentally
correct for the locally requested operation, every first transition is wrong
(`0/20`), and every chain is wrong (`0/20`). All 55 responses contain extra
text after the first line. For example, state 32 with `multiply by 12` starts
with `33`.

This is a **renderer/format failure**, not a parser failure. Because the prompt
strongly elicits lexical integer succession, it is also not clean evidence
that the arithmetic executor is absent.

### 6.2 Three-renderer atomic and recurrence matrix

Each renderer makes 110 calls: 55 independent calls from gold states plus 55
calls in chains carrying the parser-extracted model state. There are no
demonstrations, retries, repairs, search, or verifier feedback.

| Renderer | Gold-state atomic | First transition | Chained local operation | Chained gold state | Full chains | Parse success | Extra text after first line |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Question/Answer` | `40/55` | `16/20` | `40/55` | `35/55` | `7/20` | `110/110` | `110/110` |
| Bare equation | `8/55` | `2/20` | `13/55` | `4/55` | `1/20` | `110/110` | `95/110` |
| `Problem/Work` | **`44/55`** | **`18/20`** | **`44/55`** | **`40/55`** | **`10/20`** | `110/110` | `110/110` |

`Problem/Work` has four chained transitions that correctly apply the local
operation to an already wrong carried state: local-operation correctness is
44/55 while gold-state correctness is 40/55. This is evidence that visible
state carry is behaviorally active, but also shows how executor errors compound
under recurrence.

The `Problem/Work` operation breakdown is:

| Operation | Gold-state atomic | Chained local operation | Chained gold state |
|---|---:|---:|---:|
| Add | `19/20` | `18/20` | `16/20` |
| Multiply | `15/20` | `15/20` | `15/20` |
| Subtract | `8/10` | `9/10` | `7/10` |
| Remainder | `2/5` | `2/5` | `2/5` |

The family-level behavior map is more diagnostic than the aggregate:

| Task family | First correct state under `Problem/Work` | Atomic arithmetic | Model-carried recurrence | Error attribution |
|---|---:|---:|---:|---|
| Multiply then subtract | First product `3/5` | `6/10` | `7/10` local operations; `2/5` full chains | Executor-dominant. Worked examples also fail, so this is not only a compiler or halt defect. |
| Base conversion via external Horner schedule | First transition `5/5` | `16/20` | `15/20` local operations; `1/5` full chains | Whole-task compilation fails all five cases, while externally compiled arithmetic is mostly available. Later executor errors compound; the external updater itself parses and carries every state. |
| Add, multiply, subtract | First addition `5/5` | `15/15` | `15/15` local operations; `5/5` full chains | Executor and visible-state recurrence pass on this narrow family. Remaining direct defect is principally termination/format brittleness. |
| Add then remainder | First sum `5/5` | `7/10` | `7/10` local operations; `2/5` full chains | Addition passes; remainder executor is `2/5`. Direct prompts also select the sum and omit the remainder. |

The parser is an essential external resource. It chooses the last integer on
the first nonempty line and supplies that integer to the next prompt. The model
does not autonomously select the next operation, retain the plan, or stop at
the state boundary.

## 7. Does the model read the carried state?

The renderer-interchange artifact holds the operation fixed and crosses the
visible state against the state implied by retained source text. In all six
crossed add/multiply/subtract cells, the candidate computed from the displayed
state has higher summed log probability than the source-implied candidate:

- displayed/local wins: **6/6**
- source-implied wins: **0/6**
- minimum absolute summed-log-probability margin: **0.793863810133189**
- candidate-sequence evaluations: **18**
- generated tokens: **0**

This rejects pure source-answer replay for those six `Problem/Work` cells. It
supports causal use of the displayed integer, not a latent program, a general
parser, or model-internal recurrence.

## 8. Explicit packet compiler, updater, and halt

The five-call residual-packet probe asks for a compact visible object with a
current `State` and remaining `Plan`.

| Probe role | Exact result | Interpretation |
|---|---:|---|
| Compile two fresh three-step sources | Arithmetic traces contain every expected result in order: `2/2` | The narrow sequential executor can solve these instances. The model does not emit the requested packet: `0/2` responses contain the required `State:` plus `Plan:` form. This is packet compiler/serializer failure. |
| Update after observed add or multiply | Exact next packet: `0/2` | Both responses repeat the observed-result/prompt material (`2/2`) instead of replacing state and deleting the completed operation. This is direct updater failure. |
| Halt after observed subtraction | First integer correct: `1/1`; exact integer-only response: `0/1` | The answer is available, but termination fails. |
| Decode stop | `5/5` end at `max_new` | None of the five calls self-terminates within the recorded cap. |

This resolves an important ambiguity. Successful externally carried recurrence
does **not** imply that raw Shohin can author or update the recurrent packet.
The external controller currently owns parsing, state replacement, plan
advancement, operation selection, and call scheduling.

A second twelve-call diagnostic decomposes that updater into independent
subskills using three fresh examples in each cell:

| Updater subskill | Strict result | Observed failure |
|---|---:|---|
| Copy an observed result into a new state | `0/3` | Echoes the prompt instead of emitting a state. |
| Delete exactly the completed plan head | `0/3` | Executes operand-looking arithmetic or repeats the full plan. |
| Joint copy and delete in natural language | `0/3` | Ignores/reorders the state or replays a consumed operation. |
| Joint copy and delete in canonical packet form | `0/3` | Repeats the source packet or executes arithmetic instead of serializing the residual packet. |

All twelve calls reach `max_new`. The most informative near miss starts from
observed state 37, correctly executes the two remaining operations to reach
92, and then applies the already-consumed subtraction. The missing behavior is
therefore not simply arithmetic. It is a **consume-and-transport queue-state
transducer**: bind the observation as state, delete one instruction, preserve
the residual order, and stop at the packet boundary. Combined strict joint
accuracy is `0/6`.

## 9. Error attribution by capability

| Capability | Evidence | Verdict |
|---|---|---|
| Natural-language task compiler | Direct/manual omission of multiply or remainder; decimal-like base conversion; unrelated sort/string/code templates | **Weak and family-dependent.** Sequential direct compilation succeeds on the fresh five-case family, but there is no general compiler. |
| One-step visible-state executor | `Problem/Work` atomic `44/55`; add `19/20`, multiply `15/20`, subtract `8/10`, remainder `2/5` | **Real but narrow and renderer-indexed.** Remainder and some larger multiplication/subtraction remain executor failures. |
| Visible-state recurrence with external scheduler | `Problem/Work` local operations `44/55`, full chains `10/20`; sequential `5/5` | **Works narrowly.** It is a capability of the model-plus-parser-plus-scheduler system. |
| Model-authored updater | Packet exact next state/plan `0/2` | **Not demonstrated.** Explicit updater behavior fails. |
| State causality | Crossed displayed state favored `6/6` | **Supported for six add/multiply/subtract cells.** This does not establish a reusable latent state. |
| Termination | Correct sequential result followed by extra operation; `Problem/Work` extra text `110/110`; packet exact halt `0/1` | **Systematic defect.** External parsing hides much of it. |
| Output format | Manual exact `state=` prefix `0/7`; packet form `0/2`; bare renderer `8/55` atomic | **Systematic and behaviorally consequential.** |
| Model-facing integer parser | Scheduled development calls parse `55/55` in failed SSC and `330/330` across all three atomic/chained renderers | **No observed parse failures on this board.** The parser is permissive and external. |
| Continuation evaluator parser | Frozen v1 direct `1/20`; corrected hash-bound v2 direct `4/20` | **Evaluator bug corrected.** It must not be attributed to the model. |

## 10. Claim boundary and next evidence

The `Problem/Work` renderer was selected after the 20-case development
transcripts were observed. The format matrix and renderer interchange are
post-hoc diagnostics on the same cases. They establish a useful hypothesis and
a local mechanism boundary, not an independent confirmation.

The immutable 256-case fresh board is suitable for confirmation, but the local
artifact is only a board manifest with no responses. Until a completed,
hash-bound transcript and independent assessment exist, this map authorizes no
fresh-board score, internalization result, broad reasoning claim, or R12
novelty claim.

The static replay command is:

```bash
python3 train/analyze_raw260k_interactions.py
```

It verifies all eleven file hashes, recomputes the statistics above, and makes no
model calls or file writes.

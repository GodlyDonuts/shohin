# Raw-260k Operation-Cursor Diagnostic

**Status:** frozen diagnostic contract. The implementation is read-only with
respect to the checkpoint, tokenizer, source board, training data, and training
outputs. This change does not submit or run the GPU job.

## 1. Question and claim boundary

This diagnostic measures three narrower interfaces on immutable raw 260k:

1. select the operation and operand at a supplied cursor in the full source;
2. select the head operation and operand from a supplied residual suffix,
   without returning, copying, deleting, or updating that suffix; and
3. select that residual head and apply it once to a supplied numeric state.

The artifact reports exact structured-parse and correctness counts. It does not
report a reasoning score, choose a bottleneck label, authorize training,
promote a model, or support a production submission. Pairwise tables are
descriptive cells, not hypothesis tests or advancement gates.

## 2. Frozen inputs

| Input | Frozen identity |
|---|---|
| Source board | `artifacts/evals/source_scheduled_reasoning_confirmation_v1.json` |
| Source artifact SHA-256 | `19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474` |
| Source rows SHA-256 | `4afc6c4b0c271ea2f723078ab183e8d1ac1851fd1728898384ef52275887b0e4` |
| Raw-260k checkpoint SHA-256 | `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d` |
| Tokenizer SHA-256 | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |

The evaluator rejects any other source, checkpoint, or tokenizer. It also
requires checkpoint metadata `step == 260000`.

The score-blind subset rule is positional and has no model-dependent branch:
take rows `000` through `015` in each source family, preserving the source
family order. Its canonical row-list SHA-256 is
`c48ad18103b7971e7cd3c29be172ed40baccaa10d5d255011a22d3c023dc17e6`.

| Family | Cases | Transitions |
|---|---:|---:|
| `multiply_subtract` | 16 | 32 |
| `base_conversion` | 16 | 64 |
| `sequential_state` | 16 | 48 |
| `modular_update` | 16 | 32 |
| **Total** | **64** | **176** |

The evaluator independently reconstructs each schedule from the natural-
language question, then requires equality with the source schedule and replayed
answer. It never trusts a schedule or score copied into a result transcript.

## 3. Arms

Every source transition receives exactly one call in every arm. Calls are
stateless and independent; no response from one arm enters another prompt.

### A. `source_step_selector`

Exposed:

- the complete natural-language source question;
- the zero-based step index; and
- the fixed output schema.

Required output keys:

```json
{"operation":"multiply","operand":5}
```

The concrete values above illustrate types only and are not used as a prompt
demonstration. The actual prompt names the allowed operation vocabulary and
requires exactly `operation` and `operand`.

Not exposed: source-board schedule, residual suffix, current intermediate
state, expected next state, final answer, prior responses, scores, or verifier
feedback.

### B. `residual_suffix_selector`

Exposed:

- the oracle schedule suffix beginning at the tested transition, serialized as
  read-only compact JSON; and
- the same two-key output schema.

The model returns only the suffix head's operation and operand. It is not asked
to preserve, copy, delete, or emit a replacement suffix. The external harness
owns the suffix and advances it independently for the next test call.

Not exposed: full source question, step index, numeric state, expected next
state, final answer, prior responses, scores, or verifier feedback.

### C. `residual_suffix_state_update`

Exposed:

- the same oracle residual suffix as Arm B;
- the oracle current numeric state for this one transition; and
- a three-key output schema.

Required output keys:

```json
{"operation":"multiply","operand":5,"next_state":405}
```

Again, the concrete values are descriptive here, not a prompt demonstration.
The evaluator asks for the suffix head and the result of applying it once. It
does not ask for a new suffix. Later oracle current states are gold
intermediates intentionally exposed only in this arm so local state update can
be separated from model-carried error compounding.

Not exposed: full source question, step index, expected next state, final
answer, prior responses, scores, or verifier feedback. At a final transition,
the expected next state equals the source answer, but that value is never put in
the prompt.

## 4. Parsing and scoring

The parser consumes the entire decoded response as one JSON value. Leading or
trailing JSON whitespace is allowed. There is no regex extraction, substring
salvage, second parse, repair prompt, retry, or verifier feedback.

A parse succeeds only when all of the following hold:

- the value is a JSON object;
- there are no duplicate keys or non-standard constants such as `NaN`;
- the key set is exact for the arm;
- `operation` is exactly one of `add`, `subtract`, `multiply`, or `remainder`;
- `operand` is a JSON integer, excluding booleans; and
- Arm C's `next_state` is also a JSON integer, excluding booleans.

Extra prose, Markdown fences, extra keys, floats, numeric strings, aliases, and
case variants fail parsing. A parse failure scores every semantic field false
for that call, is retained verbatim, and does not suppress any later call.

Each selector arm reports exact counts for parse success, operation correctness,
operand correctness, and joint operation-plus-operand selection. Arm C also
reports next-state correctness and joint selection-plus-state correctness.

The immutable result includes the same exact summaries:

- globally;
- by family;
- by zero-based schedule index;
- by gold operation; and
- in paired cells for Arm A versus B selection, Arm B versus C selection, and
  Arm B selection versus Arm C joint correctness.

All rates are stored as integer numerator/denominator objects. No floating
percentage, pass threshold, p-value, gate, or inferred bottleneck is emitted.

## 5. Model-call and token accounting

There are `176 * 3 = 528` required greedy calls. Each call has a frozen cap of
32 sampled tokens, so the absolute cap is 16,896 sampled tokens. EOS, the
32-token cap, or the model context limit are the only decode stops. Prompts may
not be truncated.

The resource ledger preserves, globally and by arm:

- model-call count;
- prompt-token count;
- sampled-token count;
- decoded-token count; and
- exact parse-success and parse-failure counts.

It also records zero retries, repairs, search calls, verifier calls, and calls
omitted after parse failure. The evaluator compares an in-memory call counter
against all preserved call records before writing output.

## 6. Hashes and immutability

Before model loading, the evaluator hashes the checkpoint, tokenizer, source,
and implementation manifest. The implementation manifest covers:

- this contract;
- `train/eval_operation_cursor.py`;
- `train/test_eval_operation_cursor.py`;
- `train/jobs/eval_operation_cursor.sbatch`; and
- `train/model.py`.

It hashes all inputs and implementation files again after the 528 calls and
aborts on any change. The result stores every input and code SHA-256.

Output uses exclusive creation, refuses an existing path or symlink, fsyncs the
payload, and changes the mode to `0444` before close. There is one result file
and no mutable sidecar. The batch wrapper restricts that file to a fresh
`raw260k_operation_cursor_*.json` child of `artifacts/eval_history`.

## 7. Verification

The focused CPU test command is:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_eval_operation_cursor.py
```

Run it from `train/`. The tests reconstruct the immutable subset, source
schedules, all 528 prompts, strict parses, scores, summaries, and resource
ledger. Mutation cases reject changed prompts, responses, parse records,
scores, schedules, row order, call counts, token accounting, hashes, and summary
cells. Tests also verify exclusive read-only output and the non-training batch
wrapper.

`train/jobs/eval_operation_cursor.sbatch` is an isolated diagnostic wrapper.
It contains no training or downstream submission command, and it is not
submitted as part of this change.

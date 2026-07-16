# Raw-300k Researcher-Selected Freeform Interaction

**Status:** qualitative diagnostic only. This is not a benchmark, promotion
gate, or statistically powered capability estimate.

**Checkpoint:** immutable local `train/flagship_out/ckpt_0300000.pt`, raw 300k,
greedy decoding on MPS with at most 80 new tokens.

## 1. Why this interview exists

The fixed seven-case interaction can hide whether one response is a brittle
template or a general update behavior. Six fresh prompts were chosen only after
the 300k checkpoint and fixed benchmark results were already known. They probe
ordinary arithmetic, one atomic state update, source-deleted continuation, a
few-shot state format, a late query after two swaps, and explicit internal
control. Nothing from this interview enters training data.

## 2. Observations

| Prompt | Required behavior | Observed behavior | Semantic result |
|---|---|---|---:|
| `17 + 26` | return only `43` | begins `17 + 26 = 43`, then emits `Question:` | correct value, wrong contract |
| `n=23; multiply by 3` | `state=n:69` | copies the literal schema `state=n:<integer>` and enters textbook prose | fail |
| committed `n=69; subtract 20` | return `49` | returns `100`, then loops over source availability | fail |
| example followed by `x=7; add 6; multiply 3` | states `7,13,39` and answer `39` | emits `7,14,21`, answer `21`, then invents another example | fail |
| two adjacent swaps over `[A,B,C]` | track order and answer `B` | emits repeated empty code fences | fail |
| `14+9`, multiply by 3, subtract 20 | maintain one current value and answer `49` | enters a repeated C++ include sequence | fail |

The semantic count is **1/6** and the strict requested-output count is **0/6**.
All four prompts that directly require maintaining or updating a numeric state
fail. The few-shot response is especially diagnostic: it reproduces the visual
shape of a state trajectory while replacing computation with a superficial
increment pattern.

## 3. Interpretation boundary

Raw 300k has useful local arithmetic associations and can occasionally emit a
correct visible multi-step trace. It does not reliably:

- bind an operation to the current state;
- replace rather than repeat that state;
- preserve source-deleted state for a later update;
- follow an output contract or terminate after the answer;
- use an explicit request for internal control to prevent mode collapse.

This strengthens the existing diagnosis that the missing capability is an
internally controlled update-and-termination process, not merely a hidden
answer coordinate or a need for longer visible rationale text. Because the
prompts were selected adaptively, this report cannot estimate population
accuracy and must not be compared numerically with a frozen public board.

# R12 Researcher Adaptive Interaction Result

**Status:** descriptive negative; no checkpoint or architecture promotion

## Frozen evidence

- Evaluator source commit: `6c2046e4b1e31d4fb6b3512e5a687b42d846ac64`
- Evaluator source SHA-256: `4606782af85e1adbcbf7f242ac90a536e8de52e2710c379e63c8767ab7d0a7e9`
- Transcript artifact:
  `artifacts/eval_history/researcher_interview/adaptive_raw200_drs_raw300_fe46ba9.json`
- Transcript artifact SHA-256:
  `b0dff205fa870a3ce07bc8f3c5ea882d877a2e9b665d7df9239cff5323a90abc`
- Tokenizer SHA-256: `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`
- Generation: greedy, temperature `0.0`, at most 64 new tokens, seed `20260717`

The probe contains ten adaptive turns per checkpoint. Later turns may quote at most 400 characters
from an earlier response by the same checkpoint. Host code does not extract, repair, execute, or
replace model state.

## Checkpoints

| Arm | Checkpoint SHA-256 | Recorded step |
|---|---|---:|
| raw 200k | `675af7cffdc87ccd43c56a15f0616d368442aad56deb0df3fe11b5a5064aac2a` | 200,000 |
| DRS r3 from 200k | `d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459` | `sft_ep1` |
| raw 300k | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` | 300,000 |

## Locked scores

Every arm scores `0/10` semantic correctness, `0/10` exact first line, and `0/10` strict exact.
These are output-contract scores, not a claim that every generated token is unrelated to the target.

| Mechanism | raw 200k | DRS r3 | raw 300k |
|---|---:|---:|---:|
| direct scalar compute | fail | fail | fail |
| independent review | fail | fail | fail |
| serialize gold scalar | fail | fail | fail |
| serialize model scalar | fail | fail | fail |
| local digit-column compute | fail | fail | fail |
| packetize digit/carry | fail | fail | fail |
| copy trusted memo | fail | fail | fail |
| consume trusted memo, two steps | fail | fail | fail |
| consume trusted memo, one step | fail | fail | fail |
| consume model-produced memo | fail | fail | fail |

## Direct transcript reading

### Raw 200k

The model contains a narrow local arithmetic skill that the official contract correctly refuses to
count. On `58 + 27`, it writes `27 + 58 = 85` and `58 + 27 = 85`; during the independent-review turn
it repeats the correct equality three times. It nevertheless ignores the requested integer-only
interface, so neither response is a usable answer.

The value cannot be transported. Given the gold instruction `quill=85`, the model invents an
unrelated definition and formula. When asked to serialize its own earlier computation, it retains
the token `85` but writes the false equality `58 + 27 + 58 = 85` and never emits `quill=85`.
It also fails the smaller `6 + 7 + 0` column sum, digit/carry packetization, literal memo copying,
one-step memo update, two-step memo update, and reuse of its own prior response. The memo turns fall
into unrelated textbook continuations.

Interpretation: this checkpoint sometimes retrieves or computes a familiar local scalar, but it has
no demonstrated reliable actuator, typed write, state update, or state reuse mechanism.

### DRS r3 from 200k

All ten natural-language prompts produce an empty decoded response, consistent with immediate EOS.
This is not evidence that the late residual digit signal disappeared: the separate causal swap probe
established that signal under its registered interface. It is evidence that this narrow SFT candidate
does not preserve an ordinary natural-language generation interface. The residual channel therefore
cannot be treated as a usable autonomous reasoner or even as a usable controller without an explicit
actuator and preservation control.

### Raw 300k

The additional pretraining does not preserve the raw-200k scalar behavior on this probe. The 300k
model answers the scalar turn with a long run of `1` followed by zeros, produces generic decimal and
fraction text for the digit sum, and emits corpus-like headings or repeated definitions for all state
turns. It copies neither trusted values nor seals and performs no correct registered update.

Interpretation: more next-token pretraining improved neither this interface nor the missing state
transition. On these fresh prompts the observable behavior regressed from a narrow local scalar hit to
template loops.

## Causal diagnosis

The adaptive probe agrees with the stronger registered evidence while sharpening the intervention:

1. Raw pretraining can create isolated local computation without producing a controllable answer.
2. DRS can create a causally active digit-bearing late residual while collapsing ordinary decoding.
3. Neither property establishes a reusable state machine.
4. A host that executes predicted operations can expose controller information, but host execution is
   an external executor and cannot by itself establish model reasoning.
5. The next learned mechanism must separately test **read**, **update**, **write**, **consume**, and
   **halt**, with source deletion and counterfactual state interventions. It must preserve ordinary
   language behavior and must still work when no answer/result tape is supplied.

The admissible architecture target is therefore a controller/executor split with a learned discrete
carry/cursor packet and a trained residual-to-token or residual-to-register actuator. The arithmetic
executor may be deterministic in a diagnostic upper bound, but the promotion arm must perform its
registered update internally and autonomously. Matched ordinary-SFT and recurrent controls remain
required.

## Claim boundary

This is a ten-turn descriptive interaction, not a benchmark, architecture comparison, or promotion
gate. The incidental `85` in raw-200k transcripts is useful localization evidence but is not exact,
semantic, or deployable success. The DRS empty responses do not negate its registered residual swap
effect; they close the stronger claim that the existing DRS checkpoint already exposes that effect
through a preserved natural-language interface.

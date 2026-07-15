# Raw-260k Source-Scheduled Reasoning Confirmation Result

**Decision:** **FAIL the immutable internalization gate.** Preserve the causal
decomposition result, but do not launch RSP-C2 or any successor whose
prerequisite is this gate.

## Custody

- Frozen board SHA-256:
  `19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474`
- Canonical board-row SHA-256:
  `4afc6c4b0c271ea2f723078ab183e8d1ac1851fd1728898384ef52275887b0e4`
- Raw checkpoint SHA-256:
  `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d`
- Tokenizer SHA-256:
  `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`
- Exact patched Newton job: `689542`, 256/256 cases, immutable result mode
  `0444`.
- Primary result SHA-256:
  `be2e64c8df2797c3b35c7431b3b6af4d6d7fb3600cd25e5a0371415b45de6a0d`
- Independent assessment SHA-256:
  `0e1e49ea864d3958a765e11ac395aac7e2d87a4b9433950b00a3bb213a7933bd`

The independent assessor does not import the generator or evaluator. It
reconstructs all 256 board rows, all four renderers, every parse, all 704
operations, exact call accounting, the paired test, and every gate from the
raw call records. It rehashes the exact five runtime sources preserved under
`train/frozen_sources/source_scheduled_reasoning_confirmation_v1/`, including
the historical Newton model loader used by the running job. Primary and
independent counts agree exactly.

## Primary result

| Arm | Correct | Accuracy |
|---|---:|---:|
| Direct final answer | 16/256 | 6.25% |
| Whole `Problem/Work` final answer | 9/256 | 3.52% |
| Source-scheduled final answer | 115/256 | 44.92% |
| Oracle-state atomic transition | 534/704 | 75.85% |

The paired scheduled-versus-direct table has 101 scheduler-only successes and
2 direct-only successes. The exact two-sided McNemar probability is

```text
5357 / 5070602400912917605986812821504
= 1.0564819673172401e-27
```

The external scheduler therefore exposes a real capability. This is not a
sampling fluctuation and it is not a direct-answer parser artifact.

## Family result

| Family | Direct | Whole work | Scheduled | Atomic transitions |
|---|---:|---:|---:|---:|
| base conversion | 1/64 | 2/64 | 44/64 | 233/256 |
| modular update | 0/64 | 3/64 | 4/64 | 60/128 |
| multiply then subtract | 1/64 | 4/64 | 29/64 | 84/128 |
| sequential state | 14/64 | 0/64 | 38/64 | 157/192 |

Operation-local accuracy, recomputed from each recorded model input rather
than from gold chain state, is:

| Operation | Atomic oracle-state | Scheduled-input local execution |
|---|---:|---:|
| add | 230/256 | 233/256 |
| multiply | 204/256 | 204/256 |
| subtract | 96/128 | 100/128 |
| remainder | 4/64 | 7/64 |

The scheduled chain is strongest on low-base conversion (`28/32`) and falls
on bases 10--12 (`16/32`), two-digit multiplication (`9/32`), and sequential
multipliers 6--7 (`15/32`). The original five-case sequential development
success was therefore an optimistic small sample.

## Locked gates

| Gate | Requirement | Result | Pass |
|---|---:|---:|---|
| scheduled absolute | at least 35% | 44.92% | yes |
| scheduled advantage | at least +10 points | +38.67 points | yes |
| paired significance | exact p below 0.01 | `1.056e-27` | yes |
| family nonregression | scheduled at least direct in all four | 4/4 | yes |
| sequential absolute | at least 70% | 59.38% | **no** |
| atomic ceiling | at least 70% | 75.85% | yes |

All twelve integrity gates pass. The miss is a capability miss, not an
evidence-integrity failure. Thresholds, prompts, parsers, families, schedules,
or board rows must not be changed after reading this result.

## Resource and termination boundary

The experiment made exactly 1,920 model calls: 256 direct, 256 whole-work, 704
oracle-state atomic, and 704 scheduled. It decoded 133,120 tokens from 33,122
prompt tokens, with no retries, repairs, search, verifier feedback, or gold
intermediates in the scheduled arm.

Every call hit its frozen generation cap: all 512 full calls emitted 128
tokens and all 1,408 atomic/scheduled calls emitted 48. The model therefore has
no demonstrated halt policy even when its first-line integer is correct.

## Interpretation

Raw 260k contains a renderer-indexed scalar executor

```text
E(current_state, supplied_operation) -> next_state
```

that a counted external scheduler can compose substantially better than
one-shot decoding. It does **not** yet contain a reliable compiler, operation
selector, remainder operator, consume-and-transport updater, or halt policy.
The scheduler owns source parsing, operation order, queue advancement, integer
parsing, and recurrence. Calling the 115/256 result autonomous reasoning would
misattribute those resources to the model.

The next admissible work is diagnostic, not a threshold-repaired C2 fit:

1. test model-owned operation selection separately from arithmetic execution;
2. rank exact updater candidates to distinguish decoding/termination failure
   from an absent queue-update preference;
3. test for a causal, future-verbalizable operation/state workspace before
   trying counterfactual interruption training;
4. keep the external scheduler as the favorable systems baseline.

RSP-C2 is closed under its own prerequisite contract. A future experiment must
receive a new name and a new preregistration; it may not reinterpret this near
miss as a pass.

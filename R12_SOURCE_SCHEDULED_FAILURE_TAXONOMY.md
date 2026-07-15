# R12 Raw-260k Source-Scheduled Failure Taxonomy

**Status:** post-result analysis of the immutable 256-case confirmation. This
document does not change the frozen evaluator, result, parser, gates, or claim.

## Bottom line

Whole `Problem/Work` decoding scored `9/256`, but the original-task prefix
reached the correct final value in `45/256` cases. Continued generation and the
last-integer parser destroyed 36 otherwise correct trajectories. Source
scheduling scored `115/256`; the all-atomic-steps-correct ceiling was `113/256`
(two sequential chains diverged and accidentally recovered).

The scheduler's main contribution is therefore control: it owns the operation
cursor, scalar state transfer, parse boundary, and truncation boundary. It does
not repair the remainder primitive, which remains a genuine executor defect.

## Exclusive whole-decode taxonomy

Classification precedence is scored correct, correct trajectory lost to
tail/parser, wrong first operation, wrong first arithmetic, loop/replay before
completion, then later arithmetic/controller failure.

| Family | Depth | Wrong first op | Wrong first arithmetic | Loop/replay | Later failure | Correct then parser loss | Scored correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multiply-subtract | 2 | 0 | 29 | 14 | 1 | 16 | 4 |
| Modular update | 2 | 0 | 6 | 52 | 0 | 3 | 3 |
| Sequential state | 3 | 44 | 2 | 5 | 6 | 7 | 0 |
| Base conversion | 4 | 52 | 0 | 0 | 0 | 10 | 2 |
| **Total** | | **96** | **37** | **71** | **7** | **36** | **9** |

All 64 base-conversion responses failed to follow the frozen Horner schedule.
Twelve base-10 cases nevertheless stated the correct value through an
alternative decimal expansion; ten then lost it to continuation.

## Correct leading scheduled transitions

| Family | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Multiply-subtract | 29 | 15 | 20 | | |
| Modular update | 6 | 52 | 6 | | |
| Sequential state | 46 | 9 | 2 | 7 | |
| Base conversion | 64 | 0 | 0 | 0 | 0 |

| Family | Whole reaches answer | Whole scored | All atomic steps correct | Scheduled final correct |
|---|---:|---:|---:|---:|
| Multiply-subtract | 20/64 | 4/64 | 29/64 | 29/64 |
| Modular update | 6/64 | 3/64 | 4/64 | 4/64 |
| Sequential state | 7/64 | 0/64 | 36/64 | 38/64 |
| Base conversion | 12/64 | 2/64 | 44/64 | 44/64 |
| **Total** | **45/256** | **9/256** | **113/256** | **115/256** |

## Local execution and compounding

Successive chain-position counts, denominator 64 per position:

| Family | Atomic, oracle input | Scheduled, local arithmetic | Scheduled, oracle state |
|---|---|---|---|
| Multiply-subtract | `[39,45]` | `[39,48]` | `[39,29]` |
| Modular update | `[56,4]` | `[56,7]` | `[56,4]` |
| Sequential state | `[57,49,51]` | `[57,49,52]` | `[57,43,38]` |
| Base conversion | `[64,62,52,55]` | `[64,62,52,58]` | `[64,62,50,44]` |

Operation totals are atomic add `230/256`, multiply `204/256`, subtract
`96/128`, remainder `4/64`; scheduled-local add `233/256`, multiply `204/256`,
subtract `100/128`, remainder `7/64`.

## Looping and termination

A conservative loop signature (new-question continuation, duplicate canonical
equation, duplicate substantive line, repeated operation/operand chain, or
three repeated additive operands) occurs in `214/256` whole responses:

- multiply-subtract `63/64`
- modular update `64/64`
- sequential state `26/64`
- base conversion `61/64`

Every call in every arm hit its cap: whole `256/256`, atomic `704/704`, and
scheduled `704/704`, for `1920/1920` cap stops and zero EOS stops. Scheduled
execution survives only because the evaluator parses the first nonempty line
of each isolated transition and externally starts the next call.

## Controller implication

The minimum autonomous target is not a larger free-form chain. It is a typed
controller carrying `(state, next_operation, operand, cursor, done)` with one
operation per transition, one cursor advance, one fixed scalar-state write,
consumed-operation suppression, and an explicit DONE/EOS policy. It must be
evaluated with one uninterrupted model call. Remainder needs a separate
executor intervention and cannot be used to rescue a failed controller gate.

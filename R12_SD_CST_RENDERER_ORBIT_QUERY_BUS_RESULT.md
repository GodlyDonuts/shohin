# R12 SD-CST Renderer-Orbit Query Bus Result

**Decision:** reject the residual renderer-orbit program front end; retain the
late-query bus and the localization that renderer-native program decoding is
required

**Claim boundary:** consumed projected-v2 training rows only; no development,
confirmation, reasoning, or benchmark score

## Frozen run

- exact source: `05fb94a8193640b01a9548b6772996f907bdfbe5`
- raw post-commit beacon: `9602980233009144166`
- signed-safe seed, raw modulo `2^63`: `379608196154368358`
- sole valid job: `694063` on H100 `evc36`
- elapsed: 6m15s, 3,000 updates, two frozen epochs
- complete/trainable parameters: 172,723,071 / 26,665,476
- strict-200M headroom: 27,276,929
- checkpoint SHA-256:
  `2e019b81406bb90e539665271c9893a0e568e0177396243ac427f17d8ca51eca`
- report SHA-256:
  `5cce5d9c73001e9bb1345936d97b07d3cad9da642a55074891e8cd6c9a5eafe8`
- access: development `0`, confirmation `0`

The exact report is preserved at
`artifacts/r12/sd_cst_renderer_orbit_pilot_379608196154368358/report.json`.
The checkpoint is preserved locally and on Newton but is not committed to Git.

## Result

The run is numerically clean but fails five of ten frozen gates. Minimum rates
over the four held-out renderer combinations are:

| Field | Minimum exact rate |
|---|---:|
| initial state | 100.000% |
| complete kind tape | 0.300% |
| complete active identities | 4.450% |
| complete active amounts | 1.600% |
| late query | 100.000% |
| query ordinal pointer | 100.000% |
| source-line pointers | 0.350% |
| declaration binding pointers | 99.850% |
| initial-occurrence pointers | 100.000% |
| event-occurrence pointers | 0.450% |
| whole program tape | 0.000% |
| complete packet | 0.000% |

Fit renderers also fail, so this is not merely odd-parity renderer OOD. Minimum
fit exact kind/identity/amount/line/event-pointer/packet rates are
0.292%/4.667%/1.583%/0.200%/0.258%/0.000%. The final fit event-pointer gate is
therefore far below 99%.

Training is not static. Query and query-pointer losses reach zero. Average
event support rises from 21.778% in epoch one to 51.460% in epoch two; total
loss falls from 27.882 to 15.716 and line-address loss from 12.615 to 5.822.
This is evidence that the staged support path is active, but not that the
frozen program interface is close to transfer under the registered budget.

## Causal interpretation

The treatment solves exactly the interfaces given their own trainable motors:
ordinal query address/class, declaration address, initial-occurrence address,
and initial state. It fails the interfaces that must translate the new renderer
through one scalar-gated residual into frozen exact-surface line, kind, amount,
and hard line-conditioned event heads.

The failure therefore rejects the hypothesis that a large generic residual
encoder plus orbit consistency is sufficient to make a frozen exact-surface
program compiler renderer invariant. It does not reject the categorical
executor, which remains exact conditional on a correct packet, and it does not
reject the raw-byte query bus, which transfers at 100%.

## Pre-artifact closures

- `694057`: `evc26` exposed no CUDA device and failed bf16 preflight before
  model initialization or data access.
- `694059`: v1 reached epoch one with infinite event/total loss because zero
  target mass was multiplied by masked negative infinity under bf16. It was
  canceled before output creation.
- `694061`: float32 arithmetic repair source `ca67217` failed safely on update
  one because some true event spans were outside the frozen hard line support.
  The finite guard raised before an optimizer update or output creation.

These are implementation diagnostics, not additional score-bearing arms.

## Next admissible hypothesis

Do not add epochs to this rejected contract and do not widen the executor. Use
the remaining 27,276,929 parameters for a renderer-native program decoder over
the already-trained orbit memory:

1. trainable program line queries and line keys;
2. trainable slot composition plus kind and amount motors;
3. trainable event-name address queries under the model-owned decoded line;
4. frozen retained declaration/initial binding, query bus, categorical tape,
   recurrent executor, motor, and reader; and
5. exact parameter accounting below 200M with the same fit/held-out renderer
   orbit and no scored-split access.

This tests whether specialization of the program interface, rather than more
generic encoder capacity, is the missing mechanism. A pass may authorize only
a new preregistered fresh board with equal-budget controls.

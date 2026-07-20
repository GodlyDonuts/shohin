# R12 SD-CST Renderer-Native Program Decoder Result

**Decision:** reject the frozen-memory conventional decoder control

**Claim boundary:** consumed training rows only; no fresh score, novelty, or
reasoning claim

## Frozen run

- source: `dd25c388c381ca94141797ce27098aa388e80548`
- raw post-commit beacon: `18113454053950047972`
- signed-safe seed: `8890082017095272164`
- sole job: `694073` on H100 `evc36`
- elapsed: 4m46s; two epochs / 3,000 updates
- complete/trainable parameters: 179,826,564 / 7,103,493
- strict-200M headroom: 20,173,436
- frozen-parent digest:
  `2d7176300c1e00ed40d55d61b06b4a6774f8a687495c13784d06684489443748`
- checkpoint SHA-256:
  `fe95385d47dc5b737d1921db84e06a206ef4eb97802a6e1a79f258bb2432b518`
- report SHA-256:
  `6fffe15d551c66cca026bfbba3eda3576da983b0f393219eb97b7c4eb9305155`
- development/confirmation access: `0/0`

The exact report is preserved at
`artifacts/r12/sd_cst_renderer_native_pilot_8890082017095272164/report.json`.

## Result

The decoder fails on its own fit renderers, not only on held-out renderer
recombinations. Every fit renderer has 0% exact line pointers, event pointers,
whole tapes, and packets. Minimum fit complete kind/identity/amount are
0.167%/0.058%/1.467%. Minimum held-out values are:

| Field | Minimum exact rate |
|---|---:|
| initial state | 100.000% |
| complete kind tape | 0.050% |
| complete active identities | 0.050% |
| complete active amounts | 1.450% |
| source-line pointers | 0.000% |
| declaration binding pointers | 99.850% |
| initial-occurrence pointers | 100.000% |
| event-occurrence pointers | 0.000% |
| late query / query pointer | 100.000% / 100.000% |
| whole tape / packet | 0.000% / 0.000% |

Six of eleven gates pass: parameter cap, frozen-parent preservation, initial,
query, query pointer, and zero scored access. Kind, identity, amount, packet,
and fit event-pointer gates fail. Average event support falls from 8.893% to
5.660%; line loss remains near uniform at 5.691 -> 5.661 and kind loss remains
near chance at 1.022 -> 1.004.

## Interpretation

Adding a renderer-native line/slot/event decoder is not sufficient when the
v1.2 orbit memory is frozen. The retained successful interfaces are exactly
preserved, which rules out destructive interference as the explanation. The
new heads receive no readily decodable program structure from the frozen
memory. This closes “add decoder heads” and localizes the next control to joint
representation/decoder co-adaptation.

The next admissible training-only control may unfreeze only the shared orbit
byte/position embeddings, orbit encoder, orbit normalization, and the native
program decoder. Query motors, residual projection/scale, declaration and
initial-binding machinery, packet executor, motor, and reader remain frozen.
All preservation losses and final gates remain active. The complete parameter
count stays 179,826,564; only the trainable count changes. This remains a
conventional compiler control, not a reasoning invention.

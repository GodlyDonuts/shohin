# R12 Separated Controller–Executor Bus (SCEB)

**Status:** theory + implementation authorized under creative directive.
Protocol `R12-SCEB-v1`.

## Conjecture

On Shohin-scale (~125M), multi-step exact arithmetic fails when a single
autoregressive head must jointly (a) select the next operator/cursor/DONE and
(b) emit the integer result. Evidence:

- SSC external schedule: **115/256** finals (executor works under control).
- Typed v1: DONE **86%**, rollout **16%**, atomic **27%** (controller format
  learned; value emission weak).
- Typed v2 mixture: DONE **0%**, rollout **0.8%** (mixing native+typed
  destroys the controller).
- Post-DRS residual: digit directions **+31 Δlogodds** (value lives in residual
  but is not reliably consumed across steps).

**Conjecture C-SCEB (amended).** Separating controller from executor yields a
useful **systems upper bound** and localization clue. Host arithmetic does
**not** count as internal model reasoning. Claims of Shohin execution require
in-model serialization (residual motors) without external `apply_op`.


## Realizations (ordered)

### A. Host-executed typed bus (runtime; no new weights)

Model emits one typed step line. Host parses `(op, args)`, **ignores** the
model’s `-> next` integer, applies `apply_op` in Python, rebuilds the register
prompt, repeats until `done=1` or depth cap. Primary metrics: final exact,
op-match rate vs gold schedule, host-corrected vs joint-LM baseline (v1).

Gates (256 heldout rollouts, v1 ckpt):

| Gate | Threshold |
|---|---|
| Host-exec final exact | ≥50% |
| Advantage vs joint typed v1 (16.4%) | ≥+20pp |
| Op-match rate (emitted op matches schedule[cursor]) | ≥80% |

### B. Stateful Residual Register (SRR) — architecture

Frozen GPT backbone. New module `StatefulResidualRegister`:

- `n_reg` digit registers (default 8 decimal digits + sign), each embedding in
  `d_model`.
- At write layer `L_w` (default 17): last-token residual → per-register digit
  logits; Gumbel-softmax / straight-through into register ids.
- At read layers `L_r ⊆ {17…29}`: add `Σ_i E[digit_i]` to residual (broadcast
  last token or all tokens).
- Aux loss: CE on register digits vs teacher state integer.
- Optional: keep LM loss on typed step lines with **host-corrected** next values
  in the target (model not graded on arithmetic tokens).

Gates: beat host-exec-only if SRR adds ≥5pp on heldout without host at
inference; else SRR is a representation aid, not an autonomy claim.

### C. Dual-head (later)

Separate linear heads for `{op,cursor,done}` vs digit value; out of scope for
v1 score unless A/B underperform.

## Comparator / integrity

- Init: immutable `sft_typed_controller_v1_200k_r1/sft_ep1.pt` or
  `best_step200000.pt` as stated per arm.
- Never write `train/flagship_out/`.
- Do not collide with ACW Track S custody.
- No threshold shopping after scores land.

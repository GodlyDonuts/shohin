# R12 Causal Result-Digit Motor — Prereg (sibling to carry motor)

**Status:** exploratory implementation authorized; confirmation custody
deferred until carry-motor reports.

**Parameter budget (2026-07-17):** the frozen flagship has exactly
**125,081,664 unique parameters** (verified by instantiating the 300k checkpoint
configuration with tied embeddings counted once). The system must remain
strictly below **150,000,000 total parameters**. Default `DigitMotor` is
`576→4096→4096→10` (**19,185,674** trainable; **144,267,338** total),
leaving at most **5,732,661** further parameters under the strict ceiling. Tiny
rank-8 motors are obsolete for this lane.
pass their CPU unit tests, and may be run with `--allow-non-canonical` for
exploratory fits. No result from this lane may be treated as confirmatory,
and the canonical git-source-commit seal (and any advertised claim beyond a
development-board fit/eval) stays gated until Codex's carry-motor (`691928`)
reports, and until this draft passes the same review bar as
`R12_CAUSAL_CARRY_MOTOR_PREREG.md`.

**Custody:** do not share outputs, confirmation secrets, or fit boards with the
carry-motor lane. Same frozen DRS backbone class, different grammar site.

## 1. Why this exists

Codex Sol’s critique of SCEB is accepted: host `apply_op` proves **control is
learnable**, not that Shohin executes. Post-DRS residual probes already show
actionable **digit** directions (~+31 Δlogodds at L17–29). Carry motor asks
whether carry is held and only fails to serialize. This sibling asks the same
for the **result digit** at the grammar site after `;r=` (position `p`).

Together they are the honest “two-motor bundle” for the one-bit / one-digit
local transition — not a novel primitive.

## 2. Question

> Does the frozen late residual contain enough information for a tiny learned
> output motor, activated only at the grammar-defined result-digit site, to
> serialize the correct next digit and improve autonomous multi-step DRS
> execution **without** host arithmetic?

## 3. Architecture (mirror carry motor)

- Frozen DRS checkpoint (same SHA family as carry prereg when reused).
- Residual `h` after block 29 at the prefix ending at the digit site.
- Motor: `m(h) = W_up SiLU(W_mid SiLU(W_down h + b_down) + b_mid) + b_up`
  with digit logits over `{0…9}` only (10-way, exactly 19,185,674 trainable
  parameters at the frozen `d_model=576`).
- Active **only** when the generated prefix is at the exact `;r=` digit
  position for cursor `p` (grammar router; no solver in the router).
- All other logits untouched.

## 4. Arms

Base / treatment / shuffled-label / dead motor / linear diagnostic — same
logic as carry prereg. No host ALU. No SCEB host bus.

## 5. Success

Advance only if treatment beats shuffled and dead on autonomous episode exact
and first-failure shifts later, on a secret-bound confirmation board.

## 6. Relation to SCEB

| Artifact | Role |
|---|---|
| SCEB 25.4% host loop | **Control** — upper envelope when ALU is external |
| NL SCEB 15.7% | **Controller** signal without schedule text |
| Carry motor | Codex lane — serialize `c=` |
| This digit motor | This lane — serialize `r[p]` |

Neither motor may be advertised as a new computational class.

# R12 SCEB Results (reframed)

**Claim class:** controller / systems **controls**, not internal Shohin reasoning.
Codex Sol’s critique is accepted and locked here.

## What SCEB is allowed to mean

- Host arithmetic + discrete op heads shows **control is learnable** and
  localizes the joint-LM failure (cursor/op vs value emission).
- It does **not** establish that Shohin internally executes multi-step arithmetic.

## Results

### SCEB typed closed-loop — CONTROL (65/256 = 25.4%)

Heads propose op; **host** `apply_op` updates state. Beats typed v1 joint LM
(16.4%) as a systems envelope. Oracle schedule ceiling = 100% when the
schedule is visible in the prompt.

### NL SCEB — CONTROLLER SIGNAL (8/51 = 15.7%)

Schedule **not** in the prompt. Op+done step accuracy ~62%; full-chain 15.7%.
Useful localization clue for op selection; execution remains external.

### Halt-first — DECODE POLICY (61/256 = 23.8%)

Cashes latent answers without new weights. Not an executor claim.

### Failures / traps

| Arm | Outcome |
|---|---|
| Typed v2 native mixture | DONE wipe (0.8%) |
| Host-exec of LM step text | 1.2% (LM ignores cursor) |
| Heads r1 “90%” | Metric trap (final-step collapse) |
| SRR integer readout | 0% |

## Hand-off

In-model execution → grammar-gated residual motors (Codex carry motor
`691928` running; sibling result-digit prereg
`R12_CAUSAL_RESULT_DIGIT_MOTOR_PREREG.md`). No further SCEB host-ALU
threshold shopping.

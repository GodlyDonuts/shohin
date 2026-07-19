# R12 S5.2 Learned Generator-Factored Executor Preregistration

**Status:** frozen after the scoreless S5.1 corpus-gate failure and before one
replacement seed, fit, model load, or score.

S5.2 imports every architecture, training, control, evaluation, gate, and claim
boundary byte from `R12_S5_LEARNED_GENERATOR_PREREG.md` and the corrected sealed
data policy from `R12_S5_LEARNED_GENERATOR_PREREG_V1_1.md`.

Seed `7741142465189679834` is retired because 4/2,048 generated rows reused a
public roster token multiset. It had zero model, fit, development-score, and
confirmation access. No threshold or mechanism changes are authorized.

After this receipt is committed, draw exactly one replacement seed and run the
same deterministic 512-group builder against the same source/public/development
exclusions. Never open prior confirmation bytes. The board must pass every
existing gate before the sole matched fit and serial H100 evaluation. Failure
closes S5.2; passing uses the original frozen assessor without repair or rescore.

## Frozen Development Board

Replacement seed `1639560669058669827` yields 2,048 rows / 512 groups over
depths three through eight, with maximum length 345 tokens. Every corpus gate
passes, including zero exact-prompt, 13-gram, factor, nonce-name, and roster
token-multiset overlap against the admitted source/public/development inputs.
Confirmation access is zero. Development data SHA-256 is
`5d58f97f6763ac4b6550b4b2aeb959993537c185994b0ed49ac4b102c568582f`;
report SHA-256 is
`7e667a519f7cb7f3462edd51f75279bfc66260a34baa7414f06424be6cd71dd9`.
Both Newton and local copies are read-only. Commit this receipt and aggregate
report before the sole fit/evaluation job; no further seed or board is allowed.

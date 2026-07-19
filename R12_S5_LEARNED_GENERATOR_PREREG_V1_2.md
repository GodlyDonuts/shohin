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

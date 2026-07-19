# R12 S7 Learned Cayley Law Compiler: CPU Result

**Date:** 2026-07-19  
**Decision:** `admit_s7_learned_cayley_preregistration`  
**Neural score:** none; no score-bearing board seed exists yet

## Result

All frozen CPU gates pass. The treatment compiler uses only a successor
permutation, a zero symbol, equality, and bounded repeated generator
application. Source audit finds no modulo operator, affine solver call,
destination oracle, or slope multiplication in `compile_destination`.

The falsifier evaluates every hidden symbol permutation at moduli 5 and 7 and
deterministic sampled permutations at moduli 11 and 13:

| Modulus | Hidden bindings | Mode | Exact destination cells | Exact recurrent programs | `S^2` accuracy | One-witness unit-default |
|---:|---:|---|---:|---:|---:|---:|
| 5 | 120 | exhaustive | 12,000/12,000 | 120/120 | 20.000% | 40.000% |
| 7 | 5,040 | exhaustive | 1,481,760/1,481,760 | 5,040/5,040 | 14.286% | 28.571% |
| 11 | 256 | sampled | 309,760/309,760 | 256/256 | 9.091% | 18.182% |
| 13 | 128 | sampled diagnostic | 259,584/259,584 | 128/128 | 7.692% | 15.385% |

Totals are **2,063,104/2,063,104 exact destination cells** and **5,544/5,544
exact recurrent programs** across 5,544 hidden coordinate systems.

## Resource boundary

- Primary learned successor cells: 23
- Primary learned zero anchors: 3
- Trainable treatment parameters: 218
- Complete promoted-stack plus treatment parameters: 133,695,087
- Law-specific parameters: zero
- External arithmetic at inference: zero
- Maximum fixed nested successor depth: 121
- Exact equality: structural
- State mutation: structural pop-insert

The CPU result proves mechanics, not learning. It does not show that gradient
training recovers the true generator, that the ordinary transformer control
fits, or that development laws transfer. Those remain the sole fresh-board
neural gate.

## Evidence

- CPU report:
  `artifacts/r12/s7_learned_cayley_cpu_falsifier.json`
- CPU report SHA-256:
  `a933c174cb8c81dd076a5e37277a08b0eb1b075d52695b65b58ded4959937929`
- Unit tests: 14 passing S7 tests before board generation
- Sandbox optimization: both true and `S^2` 218-parameter generators fit all
  23 synthetic successor cells and all three zero anchors under the frozen
  1,000-update schedule. This sandbox contains no S7 development law or score.

## Consequence

Commit the theorem, mechanics, model, controls, builder, trainer, evaluator,
assessor, tests, and this report together. Only after that commit may the board
and training seeds be drawn. Development must use never-read reserved laws and
hidden symbol permutations; confirmation remains sealed unless every immutable
development and causal gate passes.

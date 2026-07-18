# Reasoning Attack Plan — 2026-07-17 (post-Codex alignment)

## Claim hygiene (locked)

| Result | May claim | Must not claim |
|---|---|---|
| SCEB typed 25.4% | Controller localization; host-exec **control** | Internal model reasoning / Shohin arithmetic |
| NL SCEB 15.7% | Op-selection without schedule in prompt | Full executor internalization |
| Halt-first 23.8% | Decode/stop policy cashes latent answers | New weights or deeper compute |
| SSC 115/256 | External schedule owns cursor | Autonomous reasoner |

Codex Sol is correct: **host arithmetic is a systems result**, not a reasoning
breakthrough. Use SCEB as a strong control and localization clue only.

## Live scoreboard (honest)

| System | Score | Class |
|---|---:|---|
| SSC scheduled | 115/256 | External control |
| SCEB typed closed-loop | 65/256 | **Control** (host math) |
| Halt-first decode | 61/256 | Decode policy |
| NL op-selection closed-loop | 8/51 | **Controller signal** (no schedule in prompt) |
| Typed v1 joint LM | 42/256 | Internal joint emission (weak) |
| Direct / whole | 16 / 9 per 256 | Baselines |

## Active lanes (do not collide)

| Owner | Experiment | Status |
|---|---|---|
| **Codex** | Causal carry motor (`691928`) | RUNNING ~5h |
| **This lane** | Result-digit motor (`692100`) | **~19.2M** motor (total ≈144M <150M); r2 wide MLP |
| **This lane** | NL op heads | Frozen as **controller control** only |

## Theorem reminder (Codex)

Every deterministic single-pass one-bit consumer collapses to a two-motor
bundle. Do not invent cosmetic one-bit “primitives.” Prefer grammar-gated
output motors on frozen residuals (carry / digit sites).

## Goal for this lane

Move **execution serialization** into Shohin:
post-DRS residual already moves digit log-odds (~+31). Build a tiny
grammar-gated **result-digit motor** (parallel to Codex’s carry motor), frozen
backbone, no host `apply_op`. Success = autonomous multi-step exactness without
external arithmetic.

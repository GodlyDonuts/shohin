# R12 Typed Controller v2 Result

**Status:** `advance=false` — **CLOSED NEGATIVE** relative to v1.
Decision SHA-256 `1fb898e9842415bc2656dab8fafebd0ee7b95f258a3008fd9cf2f3ef5e50fa36`.
Job `691792` (evc25), 6m wall.

## Metrics vs v1

| Metric | v1 | v2 | Gate |
|---|---:|---:|---|
| Typed rollout exact | **16.4%** | **0.8%** | ≥35% FAIL |
| Done rate | **86.3%** | **0.0%** | ≥80% FAIL |
| Typed − direct | +16pp | 0pp | ≥10pp FAIL |
| Atomic step | 27.3% | 26.6% | ≥50% FAIL |
| Δ vs v1 rollout | — | **−15.6pp** | ≥+5pp FAIL |

## What happened

Native `Problem: Compute … / Work:` bank at weight 0.45 + 2 epochs from the
v1 checkpoint **catastrophic-forgot** the typed DONE/rollout contract. Sample
rollouts stop after the first step with `done=0` and wrong products (e.g.
`99*13→1167` vs `1287`). Atomic accuracy did not improve.

## Locked lesson

Controller format and native executor format are **not freely mixable** under
a single LM objective on this scale. The renderer-indexed arithmetic foothold
must be coupled through a **separate channel** (host bus, register module, or
dual head), not through more SFT mixture weight.

## Next

1. Host-executed typed controller (model proposes op; host applies arithmetic).
2. Stateful Residual Register (SRR) — architectural register bank in the residual.
3. Do **not** reopen naive typed∪native SFT mixtures.

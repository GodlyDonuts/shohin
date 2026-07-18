# R12 Typed Controller v1 Result (rescored)

**Status:** `advance=false` on locked floors, but **real partial win** on
controller contract. Decision SHA-256
`bd25abeac4cf775eeac96664f29200ccf05eee7e2e79eea0ad83f289e3271e69`.

**Jobs:** SFT `691764` (from `best_step200000.pt`); rescored eval `691782`.

## Metrics (256 rollouts / 128 atomics / 128 direct)

| Metric | Raw | SFT | Gate |
|---|---:|---:|---|
| Typed rollout exact | 0.0 | **0.164** | ≥0.35 FAIL |
| Done rate | 0.0 | **0.863** | ≥0.80 PASS |
| Typed − direct | — | **+0.164** | ≥0.10 PASS |
| Atomic step exact | — | **0.273** | ≥0.70 FAIL |

## What worked

- Explicit `done=1` / structured step lines internalized (86% done).
- When multiply is correct, the full chain can complete (e.g. `76*17→1292;
  −28→1264; answer=1264`).
- First eval understated accuracy (~1%) due to mid-integer early-stop; rescored
  with fixed decode.

## What failed

- Multiply executor in the typed register format is weak (~1/5 first multiplies
  correct in samples). Errors compound on step 2.
- Atomic 27% << SSC's native Problem/Work atomic ceiling (~76% on confirmation).

## Diagnosis

Controller/DONE ≠ executor. The typed format teaches stopping and cursor syntax
but under-uses the renderer-indexed arithmetic already present in raw Shohin
under `Problem: Compute … / Work:`.

## Next (v2)

Hybrid curriculum: keep typed rollout/DONE, add large **native SSC-style
atomic** bank (`Problem: Compute <state> <op> <arg>\nWork:` → integer only),
continue from v1 checkpoint, 2 epochs, raise atomic/native weight.

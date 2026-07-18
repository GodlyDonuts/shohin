# R12 CDRL Neural Optimization Result

**Status:** `advance=false`. Conjecture C is rejected on the frozen
`R12-CDRL-NEURAL-v1` board. Not a Shohin, ACW, or reasoning result.

**Job:** Newton `691750` on `evc22`, exit `0:0`, elapsed `00:03:25`.
**Decision SHA-256:** `ad94ac15ca17eaa2c5381aa0a3f94fc60a49dbbf2a528552a1212b3ecf1cabdb`

## Locked outcome

Median depth-OOD margins (`core - control`), required `>= +0.05`:

| Margin | Median | Gate |
|---|---:|---|
| core − full | **-0.7759** | FAIL |
| core − rand | **-0.0205** | FAIL |
| core − hard | **-0.7783** | FAIL |

Depth-OOD exact state accuracy by seed/arm:

| Seed | core | full | hard | rand |
|---:|---:|---:|---:|---:|
| 2026071601 | 0.0449 | 0.8774 | 0.8232 | 0.0586 |
| 2026071602 | 0.0420 | 0.8179 | 0.9248 | 0.0625 |
| 2026071603 | 0.0381 | 0.7041 | 0.7549 | 0.0835 |

## Interpretation

Core-only supervision learns a residual predictor that never sees identity
padding `P` in training, then fails when depth-OOD evaluation restores full
distractor-laden histories. Random length-matched subsequences fail similarly.
Full-history and hard-mined full-history arms both solve the board.

This rejects pure Nerode-core allocation as stated in Conjecture C under the
frozen eval contract (train allocation may differ; **eval is always on full
histories**). It does not reject mixture curricula, ACW/CGBR collision
injection, or other residual-transport mechanisms.

## Non-claims

- No Shohin adapter was trained
- No ACW Track S/C custody bytes were touched
- No language or autonomous-reasoning claim

## Next

Close Conjecture C. Preserve artifacts under
`artifacts/r12/cdrl_neural_v1/`. Do not retune thresholds or re-run with
altered eval. Any successor must be a new preregistration (for example a
mixture `core∪full` arm with matched label budget).

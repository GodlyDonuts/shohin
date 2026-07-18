# Post-DRS Workspace Residual Probe Result

**Status:** POSITIVE DIAGNOSTIC on digit residual broadcast. Not a promotion of DRS.

**Job:** Newton `691756` on `evc33`, completed in 10m06s.
**Checkpoint:** `train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt`
**Artifact:** `artifacts/evals/workspace_probe_post_drs_r2.json`

## Contrast with raw-200k baseline

Raw baseline (artifact SHA-256 `78b5efa4...`) had near-zero causal residual
action: carry deltas +0.001 to +0.028 (18-22/40 positive), digit deltas -0.042
to +0.0002 (14-20/40 positive).

## Post-DRS result (10 directions / layer)

| Field | Layer | Positive | mean toward-source Δlogodds |
|---|---:|---:|---:|
| digit | 17 | 10/10 | **+31.00** |
| digit | 21 | 10/10 | **+30.94** |
| digit | 25 | 10/10 | **+30.98** |
| digit | 29 | 10/10 | **+31.02** |
| carry | 29 | 10/10 | **+2.96** |
| carry | 25 | 8/10 | +0.66 |

Early layers remain weak; late-layer digit residual is strongly causally
actionable under matched carry/digit swaps.

## Interpretation

DRS training induced a last-position residual that *can* broadcast the result
digit. The closed-loop failure is therefore not "no workspace exists" but
"workspace is not reliably updated / consumed across compounding steps."

## Next attacks unlocked

1. Late-layer residual intervention during multi-step DRS rollouts (force-correct digit residual each step).
2. Typed controller / ACW packet that reads this residual as a hard register.
3. Do not revive DRS promotion from this alone.

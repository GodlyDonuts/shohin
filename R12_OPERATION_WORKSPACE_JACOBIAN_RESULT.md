# R12 Raw-260k Operation-Workspace Jacobian Result

**Status:** failed closed before a result artifact; inconclusive causal probe.

## Bottom line

Newton job `689718` loaded immutable raw 260k and evaluated all 12 frozen
primary cases. It then aborted on the first replication case because the
norm-matched intervention was below the preregistered minimum relative norm.
The evaluator wrote no score artifact.

This is neither positive nor negative evidence for a causal future-operation
workspace. The intervention contract could not be satisfied on the frozen
replication cell. The threshold, layer set, directions, cases, and decision
rule will not be changed after this observation, and this protocol will not be
rerun as a favorable rescue.

## Custody

| Object | Value |
|---|---|
| Newton job | `689718` on `evc42` |
| Slurm state | `FAILED`, exit `1:0`, elapsed `00:04:07` |
| Primary cases reached | `12/12` |
| Replication cases scored | `0` |
| Result artifacts written | `0` |
| Preserved local log | `logs/operation_workspace_jacobian_689718.out` |
| Log SHA-256 | `60e26d88432675f233b3b1a2c58e0d06814d12eee03bacb2954ad15a0d2c3804` |

The terminal exception was:

```text
RuntimeError: norm-matched swap is below the frozen minimum relative norm
```

The log contains progress identifiers only and no preserved primary scores.
Consequently, partial console output cannot be converted into a result.

## Decision

Classify the frozen Jacobian diagnostic as **inconclusive / failed closed**.
It does not authorize a controller fit, a workspace claim, threshold tuning,
or a replacement intervention selected from these outcomes. Any future causal
probe must be a newly preregistered protocol with a mathematically defined
zero-signal branch and an immutable result for every valid execution path.

# R12 S3 Equivariant Local Action Result

**Decision:** `reject_s3_equivariant_v1_1_for_confirmation`

The equivariant input contract repairs the coordinate-frame failure of S3 v1
and makes two-step execution almost exact. It does not make the learned local
action stable under long-context operation transport, so the frozen depth
gates fail and confirmation remains unauthorized.

## Custody

- Source/prereg commit `b6dc983` preceded fit and every v1.1 score.
- Job `693131` completed once on H100 `evc25` in 5m33s, exit `0:0`.
- Training used the unchanged 96,000 public rows / 192,000 independent atomic
  targets / 1,517 updates / one epoch / seed `2026071903`.
- The 710,411-parameter executor kept the full system at 134,399,961
  parameters and recorded zero confirmation access.
- Training took 71.18 seconds. Final executor-state SHA-256 is
  `15f640d7482de592ed8394335c8e755c61ac9917cc916e680a83946ead93ace2`.
- Assessment SHA-256 is
  `3c9ed4f0891f7afbe2f8f2fc64c2685d8ed6b2073942fca798067be15c4d2fb2`.
- The local checkpoint mirror is SHA-256
  `39e77e4355b31314de1be5c8349d029c8d25027f29f4607da2b98672683e9830`;
  it remains ignored and is not committed.

## Results

| Evaluation | Answers | Exact state | All transitions | Entity match |
|---|---:|---:|---:|---:|
| Two-step mean | 99.463% | 99.854% | 99.854% | 99.927% |
| Two-step ordered | 99.512% | 100.000% | 100.000% | 100.000% |
| Two-step gold | 99.512% | 100.000% | 100.000% | 100.000% |
| Lexical-OOD mean | 74.512% | 72.705% | 63.086% | 85.522% |
| Depth 3--8 mean | 84.180% | 80.713% | 60.059% | 86.051% |
| Depth 3--8 ordered | 86.719% | 84.473% | 66.211% | 89.331% |
| Depth 3--8 gold | 87.109% | 84.912% | 66.895% | 89.829% |

Depth-eight mean answers are 82.941%, but exact state is 78.235% and complete
transition chains are 47.353%. Gold identity changes those depth-eight values
only to 82.647%, 78.824%, and 50.882%. The two-step gates and the depth-eight
answer floor pass. The frozen mean-state, mean-chain, ordered/gold ceiling, and
ten-point attribution gates fail.

Relative to S3 v1, this is a large architectural recovery: long mean answers
rise 53.906% -> 84.180%, exact state 41.260% -> 80.713%, and complete chains
17.627% -> 60.059%. Relative to the continuous RGDE public-board comparator,
mean answers fall 1.465 points below its 85.645% rebound rather than exceeding
it by the required ten points. The result is therefore evidence for local
equivariance, not a promoted executor.

## Mechanistic diagnosis

The exact S3 state no longer drifts, and the cell can execute the public
two-step distribution. The remaining transition inputs are still continuous:
`kind_context`, soft kind probabilities, and a literal embedding. The separate
amount head is 100% accurate at depth, while the same learned local action
misclassifies transitions. Gold identity does not repair the gap. This
localizes the remaining failure to action transport: continuous encodings of
the same finite `(direction, amount)` action move under long-context surfaces,
and the MLP uses nuisance variation that the discrete amount classifier has
already discarded.

The next bounded arm may therefore replace only the learned transition MLP
with a closed S3 action table driven by model-predicted categorical direction
and amount. It must retain the frozen language compiler, query consumer,
source deletion, register, boards, and favorable identity ceilings. That arm
tests closure by construction; it is an internal neural-symbolic execution
component with an externally supplied schedule and halt, not autonomous
reasoning.

No confirmation, free-form language reasoning, planning, learned halt, or
novelty claim is authorized by v1.1.

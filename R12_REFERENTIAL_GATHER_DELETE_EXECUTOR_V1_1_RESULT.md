# R12 Referential Gather-Delete Executor v1.1 Result

**Decision:** `qualify_rgde_v1_1_for_fresh_depth_confirmation`

**Boundary:** this is a passed development component: a model-owned,
source-deleted, tied atomic update rule composes two list operations. It is not
yet broad natural-language reasoning, autonomous rollout, halting, or a
state-of-the-art claim.

## Run custody

The source/preregistration commit `ac03e46` preceded every fit. Jobs
`693118--693121` completed `0:0` on H100 nodes `evc29`, `evc33`, and `evc34`.
The no-refit repaired control job `693122` completed `0:0` on `evc29` after
control-amendment commit `3b718d4`.

| Arm | Job | Elapsed | Executor SHA-256 |
|---|---:|---:|---|
| tied predicted atomic | 693118 | 2m44s | `adb6323202f6d25280f3a1cfd34a5b88fbc876331643726e38db389ead746b74` |
| untied predicted atomic | 693119 | 2m06s | `29b00408c369135fd94154b6521d0b30f13e10e2dd70db7f018fd80c9f18ed48` |
| tied gold atomic | 693120 | 2m44s | `bb238a69fc2cb1aafeba5ba55473e2b1abe11f07172c4085d0aade93ec445d51` |
| tied composed supervision | 693121 | 2m50s | `43a6fd11d8cec76d09096a8112c452695053e700aa7cf6f91143838692cd2843` |

All local executor and log hashes match Newton. Base and compiler were frozen;
only the 1,491,279-parameter tied or 2,459,161-parameter untied executor was
trainable. Each system remains below 150M total parameters. Confirmation access
is zero.

## Primary results

| Arm | Answers | Exact final state | Both transitions | Query | All-four answers |
|---|---:|---:|---:|---:|---:|
| **tied predicted atomic** | **99.707%** | **99.902%** | **99.756%** | **99.805%** | **507/512** |
| tied predicted, gold rescore | 99.805% | 100.000% | 100.000% | 99.805% | 509/512 |
| untied predicted atomic | 99.512% | 99.609% | 99.414% | 99.805% | 506/512 |
| tied gold atomic | 99.902% | 100.000% | 100.000% | 99.902% | 510/512 |
| tied predicted, composed supervision | 99.609% | 99.756% | 99.707% | 99.805% | 507/512 |

The treatment saw 192,000 independent one-operation examples. Op0 and op1
each started from identity; no two-step state, transition, or full answer was a
treatment training target. At evaluation, the same neural cell was applied
twice. Its minimum answer accuracy across canonical, paraphrase, order-twin,
and binding-twin surfaces is 99.609%.

The tied arm is 0.195 points better than untied on answers and 0.293 points
better on exact final state while using 967,882 fewer parameters. The fully
composed training ceiling does not improve it. This is evidence for transfer of
the shared atomic rule rather than slot-specific memorization or missing
two-step supervision.

## Causal interventions

The first within-batch row rotation was invalid because surface quartets often
share the same semantic field. It is retained but excluded. The committed
no-refit amendment globally deranges semantic keys and replaces only one
bounded packet field. All 2,048 requested fields actually change.

| Evaluation | Answers | Exact final state | Both transitions | Query |
|---|---:|---:|---:|---:|
| treatment | 99.707% | 99.902% | 99.756% | 99.805% |
| operation-program derangement | 36.963% | 24.365% | 10.156% | 99.805% |
| query-position derangement | 0.146% | 99.902% | 99.756% | 0.049% |

Operations reduce answers by 62.744 points and final state by 75.537 points.
Query replacement destroys answer selection while leaving the state unchanged.
This factorization is the expected causal signature: operation packets update
state; the query packet consumes it.

## What changed from v1

RGDE v1 gathered one contextual token with pointer softmax and achieved only
48.340% answers / 18.701% final state. Its compiler pointed inside the correct
multi-token span 99.982% of the time, but selected the same subtoken across
occurrences only 59.326% of the time. The no-fit v1.1 carrier treats each role
as a set and averages frozen lexical embeddings with normalized sigmoid role
weights, recovering entity identity at 99.854%.

That zero-parameter interface repair raises two-step answers by 51.367 points
and exact final state by 81.201 points. Contextual features still carry
operation and query semantics; lexical spans carry stable referential identity;
both source memories are deleted before execution.

## Frozen assessment

All ten preregistered gates pass. Canonical assessment SHA-256 is
`60ec1bb3794c123595801f24f14c431d63fc9eabbf5616e1c9825ee556f30f20`;
assessment file SHA-256 is
`b1c9e348ee558fa78785d6211b27b0516ce284b7cf7b8b101d1fbb1a3a258654`.
The safe evidence archive SHA-256 is
`aca02c1661b0c62ca90affddbda4aa65fd979c61c6805ca3e1fd3cdcb1e48930`.

This authorizes one fresh commit-before-seed confirmation board with
three-to-eight operations, unseen names, new factor combinations, source
deletion, and the frozen tied executor. It does not authorize opening any old
confirmation or claiming general reasoning.

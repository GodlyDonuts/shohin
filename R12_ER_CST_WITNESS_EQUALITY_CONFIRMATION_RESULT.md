# R12 ER-CST Witness Equality Bus v1.1 Confirmation Result

**Decision:** `confirm_er_cst_witness_equality_v1_1`

## Custody

Evaluator source `4a930c032adc04ec580ce7272df473365fe57a4a` was
committed and pushed before confirmation access. Its isolated Newton capsule
reproduced scientific manifest `4f32348020163dcec4eeee970443120ca652be63ac3056d7b4c85cf9c2c1a6ac`
and evaluator manifest `1c710acd0e5a4cec3db240f6f21836867602d6e39a49fcbf3b024732d9c6d26d`.
Preflight found exactly one development ledger, confirmation mode `0600`, and a
new output path. Sole no-training job `694641` ran on H100 `evc22` for 2m42s.
It wrote immutable authorization before the `O_EXCL` confirmation ledger, read
the sealed split once, and ended at final custody `1/1`.

## Sealed result

| Metric | Treatment | Family deranged | Equality ablated |
|---|---:|---:|---:|
| Initial state | 99.219% | 28.906% | 15.967% |
| Complete cards | 99.805% | 0.195% | 0.146% |
| Events | 100% | 100% | 50.000% |
| HALT | 100% | 100% | 100% |
| Late query | 100% | 100% | 100% |
| All 18 witness pointers | 99.805% | 42.480% | 0% |
| Complete packet | 99.023% | 0.098% | 0% |
| Recurrent state | 99.023% | 17.334% | 16.357% |
| Answer | 99.023% | 35.010% | 34.424% |
| Packet/state/answer joint | 99.023% | 0.098% | 0% |

Treatment joint accuracy is 100% at depths one through five, 92.969% at depth
six, 99.219% at depth seven, and 100% at depth eight. All four unseen renderer
compositions are 99.023% joint. Every one of the 14 unchanged scientific gates
passes. The separately invoked assessor recomputes the raw predictions and
passes all six artifact, parameter, authorization, metric, gate-vector, and
custody gates.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| Authorization | `84e99ce3747196c3457292890102c807bc154ebe21444d3d92eed1c569b2384f` |
| Confirmation evidence | `2138a4b631d0ed0c28388edaf313726b856a8ce2afecdb1122e1adb516375090` |
| Confirmation report | `92de586aa10b8ab68651ff6b9eadcc9977a68042546095e86aee0a6c6a290dd6` |
| Independent assessment | `4a0fb47233d86887bb46aa853560bf81d319840610d62abe4f1dfaa899671310` |
| Confirmation ledger | `137a88106c2fffb003c6837aab153496538cd8534f9f5fb5bede9d49bc30270e` |
| Promoted checkpoint | `917c1a1fce67c02258d0f90f04398ab433d18ba63c2dca92450cc5856c022ae7` |

The artifacts are mirrored read-only locally. Re-running the independent
assessor on the local mirrors produces byte-identical assessment SHA
`4a0fb472...`. Newton promotion path is
`/lustre/fs1/home/sa305415/shohin_promoted/er_cst_witness_equality_v1_1`.

## What is confirmed

The model can infer a fresh operation's complete categorical permutation from
opaque before/after witnesses, bind fresh operation names, delete the source,
compose up to eight selected operation cards recurrently with internal halt, and
answer a late categorical query. The causal controls show that the result
depends on true cross-occurrence equality and true episodic card semantics.

This closes the specific failure that defeated ER-CST v1. The result is not a
claim of general reasoning: state cardinality is fixed at three, every operation
is a permutation, the architecture enumerates six `S_3` cards, and the task is a
formal synthetic language. The next valid frontier is to remove those finite
ontology constraints through variable-cardinality relation matrices and direct
model-owned transition composition, not merely widen this solved compiler.

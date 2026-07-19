# R12 S3 Closure-Complete Local Action Result

**Decision:** `reject_closed_s3_v1_2_for_confirmation`

Exact finite action closure is mechanically correct but does not rescue the
end-to-end long program. The remaining dominant error is the frozen compiler's
direction prediction under long source surfaces, not state recurrence, amount,
or the S3 action algebra.

## Custody

- Source/prereg commit `82e92d5` preceded every v1.2 score.
- Job `693134` completed once on H100 `evc25` in 2m01s, exit `0:0`.
- The arm reused the exact v1.1 checkpoint and performed zero optimizer updates,
  zero data fit, and zero confirmation access.
- The action table adds no parameter; the full system remains 134,399,961
  parameters including the frozen, unused v1.1 transition MLP.
- Assessment SHA-256 is
  `603ec1ffad20325061a9ac7f1cb7cf9e1995f64314517fd04898448497bc27b2`.

## Results

| Evaluation | Answers | Exact state | All transitions | Direction | Amount |
|---|---:|---:|---:|---:|---:|
| Two-step mean | 99.463% | 99.854% | 99.854% | 100.000% | 100.000% |
| Two-step ordered | 99.512% | 100.000% | 100.000% | 100.000% | 100.000% |
| Two-step gold | 99.512% | 100.000% | 100.000% | 100.000% | 100.000% |
| Lexical-OOD mean | 75.195% | 73.242% | 63.086% | 70.776% | 100.000% |
| Depth 3--8 mean | 85.303% | 82.031% | 63.281% | 93.403% | 100.000% |
| Depth 3--8 ordered | 87.939% | 85.840% | 69.775% | 93.403% | 100.000% |
| Depth 3--8 gold | 88.379% | 86.328% | 70.508% | 93.403% | 100.000% |

Depth-eight mean is 83.824% answers / 79.706% state / 51.471% complete chains.
Gold identity is 83.529% / 80.294% / 55.000%. Short and lexical gates pass;
the frozen mean attribution/state/chain, ordered, gold, and direction gates
fail.

Relative to learned equivariant v1.1, closed action raises long mean answers
84.180% -> 85.303%, exact state 80.713% -> 82.031%, and complete chains
60.059% -> 63.281%. Gold improves by 1.270 / 1.416 / 3.613 points. These modest
gains show that continuous transition selection contributed error but was not
the principal long-context failure.

## Mechanistic diagnosis

The action table itself is exhaustive and exact. The frozen amount classifier
is 100% accurate on every evaluated arm. Direction is also 100% on the short
compositional board, but falls to 93.403% on the known-atom depth board and
70.776% on lexical OOD. Because one wrong direction changes the exact state,
later entity-location measurements and complete-chain accuracy compound that
upstream compiler error even under gold identity.

A corpus audit finds that all 12 direction token sequences on the depth board
are exact training atoms: six left and six right forms, with zero cross-class
collisions. The contextual kind head is therefore discarding a relation that
the source-pointer channel can in principle retain. The next bounded arm may
decode the frozen operation-kind pointer through a lexicon built only from
training spans, fall back to the neural kind head for unmatched sequences, and
feed that categorical direction into the same exact action table. No
development labels, executor fit, or confirmation may enter that lexicon.

V1.2 does not authorize confirmation, autonomous planning, learned halt,
free-form language reasoning, or novelty.

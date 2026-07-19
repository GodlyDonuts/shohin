# R12 S3 Training-Lexicon Action Development Result

**Decision:** `qualify_lexical_closed_s3_v1_3_for_fresh_confirmation`

The training-only lexical relation decoder closes the known-atom direction
interface. Combined with ordered referential identity and exact S3 action, the
source-deleted system executes three-to-eight-step programs at near-exact
accuracy without training on composed or long programs.

## Custody

- Source/prereg commit `a2fc8da` preceded lexicon construction and every score.
- Job `693136` completed once on H100 `evc25` in 2m15s, exit `0:0`.
- The lexicon contains exactly 12 collision-free training patterns from
  192,000 operation references: six left and six right.
- The arm performed zero optimizer updates, added zero parameters, and recorded
  zero development-label and confirmation access during construction.
- Lexicon SHA-256 is
  `dda061ccc4e3ba5ba4d0df0186fae01e3ab09b1feaa319701e320533b7ac3189`.
- Assessment SHA-256 is
  `41102547acd8755661192b43473b8dbddf291cb8f3fc223d8b93ccc883266f39`.

## Results

| Evaluation | Answers | Exact state | All transitions | Direction | Amount | Lexicon coverage |
|---|---:|---:|---:|---:|---:|---:|
| Two-step mean | 99.463% | 99.854% | 99.854% | 100.000% | 100.000% | 100.000% |
| Two-step ordered | 99.512% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| Two-step gold | 99.512% | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| Lexical-OOD mean | 75.195% | 73.242% | 63.086% | 70.776% | 100.000% | 0.000% |
| Depth 3--8 mean | 94.434% | 94.336% | 89.453% | 100.000% | 100.000% | 99.982% |
| Depth 3--8 ordered | 98.340% | 99.463% | 98.730% | 100.000% | 100.000% | 99.982% |
| Depth 3--8 gold | 98.779% | 100.000% | 100.000% | 100.000% | 100.000% | 99.982% |

Ordered depth-eight is 98.529% answers / 100% state / 98.824% complete chains.
Mean depth-eight is 96.176% / 96.176% / 90.882%. Every frozen public gate
passes.

## Causal interpretation

The v1.2-to-v1.3 intervention changes no weight, state register, identity arm,
amount head, query head, board, or executor. Restoring known direction atoms
raises mean depth answers 85.303% -> 94.434%, state 82.031% -> 94.336%, and
complete chains 63.281% -> 89.453%. Gold state/chains rise 86.328% / 70.508%
to exactly 100% / 100%. This is the predicted signature of an upstream
direction-transport failure.

The lexical-OOD control is equally important. None of its unseen direction
phrases crosses the 0.5 training-pattern mass threshold, so coverage is 0% and
scores remain byte-for-byte at the closed-action fallback baseline. The gain is
not caused by matching known distractors or reading development labels.

The strongest complete arm is ordered identity plus lexical direction plus
model-predicted amount and query plus exact S3 state/action. It receives source
text only through frozen compiler pointers, deletes source states, and carries
one categorical register through repeated calls. It was trained only on
independent atomic updates; no composed or depth supervision fits any weight.

## Boundary

This public development pass authorizes one independently seeded confirmation.
It does not yet confirm the score. It handles known direction atoms, three
referential identities, two bounded amounts, and externally supplied operation
count/halt. It does not establish unseen direction semantics, autonomous plan
induction, learned stopping, free-form language reasoning, or novelty.

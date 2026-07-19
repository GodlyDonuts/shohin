# R12 S3 Lexical Closed-Action Confirmation Result

**Decision:** `reject_lexical_closed_s3_confirmation`

The fresh board strongly reproduces the public score and passes both causal
controls, but it misses the preregistered perfect gold execution gate by two
direction decisions out of 11,248. The result is a near-confirmation, not a
confirmed claim.

## Custody

- Confirmation source/prereg commit `29c7607` preceded the first production
  seed. Mechanical balance correction commit `9a37b22` preceded the scored seed.
- Seed `3906227011763392781` was rejected before model access for infeasible
  query derangement. Seed `3664953321459551042` is the sole scored board.
- The scored board has 512 quartets / 2,048 rows / 6,136 cards / 798,115 source
  tokens. Board SHA-256 is
  `9b3895639cd74c2abd24309c795d403934f675c535aac0acbf4a3d4b19f8c180`.
- Job `693138` completed once on H100 `evc25`, exit `0:0`.
- Assessment SHA-256 is
  `8d69dd5d4461e07f7ad2530cc31d92776d162e4fa9ddccef73b3457b661239bc`.
- No weight fit, old-confirmation access, threshold change, or rerun occurred.

## Scores

| Arm | Answers | Exact state | All transitions | Direction | Amount |
|---|---:|---:|---:|---:|---:|
| Ordered primary | 99.121% | 99.268% | 98.682% | 99.982% | 100.000% |
| Mean identity | 95.215% | 93.652% | 88.818% | 99.982% | 100.000% |
| Gold identity | 99.609% | 99.951% | 99.902% | 99.982% | 100.000% |
| Operation derangement | 35.010% | 17.871% | 1.221% | 65.203% | 59.122% |
| Query derangement | 0.439% | 99.268% | 98.682% | 99.982% | 100.000% |

Ordered scores clear the overall and every-depth gates. Mean clears 90% /
90% / 82%. Operation replacement changes all 2,048 rows and drops answers by
64.111 points and state by 81.396 points. Query replacement changes all rows,
drops answers by 98.682 points, and leaves state exactly unchanged. Receipts
and hashes pass.

The sole failed gate is gold exact execution. Lexicon coverage is 99.813%; the
neural fallback repairs most unmatched references, but two direction decisions
remain wrong. Gold therefore has one incorrect final state and two rows with a
non-exact full chain. The frozen gate required exactly 100% state and chains.

## Consequence

Do not relax the 0.5 threshold or rerun this board. Seal it. Return to public
development with a structural decoder that asks whether the compiler pointer's
global maximum lies inside a known exact token pattern. That rule has no tuned
mass threshold: known pointed atoms use their class, while an unseen pointed
phrase falls back to the neural kind head. It must pass public lexical-OOD
fallback and exact-depth gates before a wholly new independent confirmation.

This result supports a strong known-atom source-deleted execution component but
does not confirm it under the locked contract. It does not establish unseen-
phrase semantics, autonomous planning, learned halt, free-form language
reasoning, or novelty.

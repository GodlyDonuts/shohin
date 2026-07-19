# R12 RGDE Consumer Transport Diagnostic Result

**Decision:** `transport_failure_not_localized`

The public board reproduces the end-to-end degradation, and identity rebinding
helps materially, but even gold identity cannot recover the frozen executor's
gold-packet ceiling. The failure is distributed across identity transport and
continuous recurrent consumption rather than one bad comparator.

## Custody

- Preregister/source commit `fba4be0` preceded executor-level score access.
- Board SHA-256:
  `ba2b0d4817ffe68f004978b6a403aba893db17aa49878afb6548a71b9219b596`.
- Frozen executor file SHA-256:
  `adb6323202f6d25280f3a1cfd34a5b88fbc876331643726e38db389ead746b74`.
- Frozen executor state SHA-256:
  `d31fd3e6150dd352cd0eea5063f960393e9017ac3ab729b5c536fd1f1c432184`.
- Job `693126` completed once on H100 `evc29` in 26 seconds, exit `0:0`,
  with zero fit updates and zero confirmation access.
- Result SHA-256:
  `d7822b395069569e43aa9d591bfc30769ed32c7be61140f71e37ff5c63b8201a`.

## Matched results

| Arm | Answers | Exact state | All transitions | Entity match |
|---|---:|---:|---:|---:|
| Untouched packet | 77.393% | 70.312% | 46.045% | 79.979% |
| Mean-selected rebound | 85.645% | 81.543% | 62.012% | 86.691% |
| Ordered-selected rebound | 88.281% | 85.400% | 68.311% | 89.989% |
| Gold-identity rebound | 88.672% | 85.840% | 68.994% | 90.478% |

Amount is 100% and query is 99.609% in every arm. Identity selection itself is
97.413% for mean, 99.653% for ordered, and 100% for gold.

The untouched arm clearly reproduces the prior transport failure. Mean
rebinding gains 8.252 answer points and 11.230 state points: it misses the
predeclared ten-point answer threshold, so consumer-matcher localization is
not authorized. Ordered rebinding adds 2.637 answer points over mean and gold
adds only another 0.391 points. Most importantly, gold identity remains far
below the 99% answer/state ceilings.

## Mechanistic consequence

The executor does not merely fail to decide which introduced name an operation
mentions. It re-encodes identity into continuous entity vectors, compares those
vectors after soft permutation mixing, and repeatedly feeds the resulting
distributed state back into the next match. Exact rebinding removes the first
comparison error but not state-representation drift.

The next admissible architecture should therefore remove continuous entity
embeddings from the recurrent loop. A categorical identity packet can address
one of three immutable identities directly; recurrent state can be only a
three-by-three permutation register; and a tied neural update cell can consume
current location, kind, and amount without relearning semantic equality at
every step. Mean-selected identity is the primary conventional compiler;
ordered and gold identity remain favorable ceilings. This requires a new
preregistration and fit on public atomic data before any fresh confirmation.

No autonomous planning, learned halt, language reasoning, or novelty claim is
authorized by this diagnostic.

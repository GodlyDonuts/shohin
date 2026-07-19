# R12 RGDE Consumer Transport Diagnostic Preregistration

**Status:** closed negative after job `693126`. See
`R12_RGDE_CONSUMER_TRANSPORT_DIAGNOSTIC_RESULT.md`.

**Claim class:** no-fit public matched-consumer diagnosis. This cannot promote
the rejected relational carrier and does not read confirmation.

## Question

The public relational probe found 97.413% raw lexical-mean identity, whereas
the frozen executor's learned matcher reached only 77.952% on the closed depth
board. Does the learned consumer interface destroy identity that is present in
the packet, or does this public board fail to reproduce the end-to-end problem?

## Frozen arms

All arms use the exact tied RGDE v1.1 executor state
`d31fd3e6150dd352cd0eea5063f960393e9017ac3ab729b5c536fd1f1c432184`
and the admitted public board
`ba2b0d4817ffe68f004978b6a403aba893db17aa49878afb6548a71b9219b596`.
There is no fit.

1. `current`: exact existing packet and executor.
2. `mean_rebound`: choose one of the three introduced identities by existing
   lexical-mean cosine, then replace only the operation entity vector with the
   corresponding introduced vector.
3. `ordered_rebound`: same replacement using the rejected ordered kernel as a
   diagnostic oracle, not a promotion path.
4. `gold_rebound`: same replacement using the true introduced identity.

Every operation context, kind, literal, query, state, cell, and weight is
otherwise byte-identical. Full source memories are deleted before execution.

## Frozen interpretation

- Localize transport loss to the learned consumer matcher only if current
  entity match is below 90% and mean rebinding gains at least 10 points on both
  answers and exact final state.
- If current answers and exact state are both at least 95%, record that the
  public board does not reproduce the failure.
- Otherwise record `transport_failure_not_localized`.
- Ordered rebinding adds independent evidence only if it gains at least one
  answer point over mean rebinding. Gold answer/state ceilings are recorded
  against 99% but do not alter the primary diagnosis.

One H100 job must exit `0:0`, hash every input and this evaluator, record zero
fit updates and zero confirmation access, and write all four depth/surface
tables. No threshold changes, retry, confirmation read, reasoning, halt, or
novelty claim is permitted.

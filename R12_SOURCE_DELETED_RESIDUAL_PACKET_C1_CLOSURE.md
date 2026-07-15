# RSP-C1 Research-Integrity Closure

**Decision:** RSP-C1 is permanently closed as a claim-bearing experiment on
2026-07-15. This decision is irreversible and does not depend on the eventual
source-scheduled prerequisite result.

## 1. Closure finding

The frozen C1 contract stated that RSP-C1 could generate a board, acquire data,
fit, or evaluate only after the prerequisite confirmation reported
`advance_to_internalization=true` and an independent recomputation agreed with
every locked gate. It also stated that no RSP board or training data existed at
freeze time and that the production board would be generated exactly once.

Before the prerequisite completed, C1 implementation work did all of the
following:

- computed and hardcoded the exact production board and canonical-row hashes;
- instantiated the exact 256-case production board in generator and scorer
  tests;
- materialized a temporary production board, treatment corpus, sham corpus,
  and generation manifest in audit tests;
- reconstructed the exact 4,096-program training set and both 16,384-row arms;
- continued changing uncommitted generator, auditor, evaluator, scorer, and
  test code after those exact artifacts were known.

The affected C1 board identities are:

```text
artifact SHA-256  ad6be48f5952a142c0684f304ba6393b66c25b68b2d6c97d8a0b5d80cfedd9e7
rows SHA-256      fcc2970f9bbd8890a6e3d8cb495ddb45cb7c0825d9adb7318d1b2e0807b9a20e
```

No C1 model score was read, and this is not a finding of outcome fabrication.
It is a custody failure: the exact evaluation board and training arms became
development fixtures before conditional authorization. The C1 board therefore
cannot function as a fresh confirmatory test.

## 2. Disposition

RSP-C1 is closed under the literal contract. There will be no exception for
temporary files, in-memory generation, score blindness, or the fact that a
model had not yet consumed the board.

The following are quarantined as C1 development material and are forbidden in
any C2 production path:

- C1 board rows, sources, packets, trajectories, answers, and hashes;
- C1 board, training, observation, sham, fit, sampling, or intervention seeds;
- C1 treatment and sham rows, manifests, audits, token-accounting receipts,
  checkpoints, transcripts, and scores;
- any fixture, cache, serialized object, or generated file containing C1
  production rows or training examples;
- C1 `v1` generator, auditor, evaluator, scorer, and job files as executable
  production dependencies;
- any C1 result as confirmatory evidence, even if generated later without code
  changes.

The files must be retained long enough to audit the closure rather than erased
or rewritten. They may be described only as non-claim-bearing engineering
evidence.

## 3. What may survive conceptually

C2 may restate the bounded scientific hypothesis that a learned compiler and
source-free updater can control an immutable arithmetic executor. It may also
use ordinary, independently reimplemented utilities whose behavior is not
conditioned on C1 cases. This does not authorize copying any C1 production
artifact, seed, exact case, generated corpus, hash, or executable `v1` path.

Any C2 implementation must live in separately versioned paths, be audited
against toy fixtures, and be pushed before its production seed can be known.
The companion C2 preregistration controls that salvage.

## 4. Claim boundary

Allowed statement:

> RSP-C1 was closed before fitting or scoring because exact production board
> and data generation occurred before its prerequisite gate completed.

Forbidden statements include that C1 passed, failed scientifically, validated
source-deleted reasoning, or supplied an independent held-out estimate. A later
prerequisite pass cannot reopen C1.

# ACSO Deep-Fault Oracle Recovery Protocol

**Status:** v2 preregistration; no v2 result has been consumed.

## Retired v1

The uncommitted v1 draft is void. Its fixture accidentally read one
confirmation world, its cyclic objective was not oracle-fixed, its receipt was
too aggregated, and its tie/recoding gates were incomplete. No v1 official
artifact was produced. V2 makes no unseen-board claim and uses every eligible
deep fault on the already frozen 200-world mechanics board.

## Question

Before ACSO is connected to HSC or fitted, does depth-three causal closure
repair transition faults that an oracle-fixed one-step objective cannot
distinguish? This is a targeted oracle upper-bound mechanics test. Failure
kills the present multi-step revision rule. Success does not establish
learnability, raw-source compilation, transfer, or reasoning.

## Frozen board and eligibility

- Generate the existing 200-world identifiable board with exact seed
  `efc-identifiable-pilot-v1` and frozen 96/48/32/24 counts.
- Consume every structurally unique world across all four split labels. Split
  labels have no inferential role because this is an exhaustive mechanics
  audit, not a fit or held-out capability claim.
- Build target base signatures through action-word depth three and
  action-prefix derivative signatures through suffix depth three, hence total
  action depth four, independently from each exact finite machine.
- A transition fault is eligible only when the wrong destination and correct
  destination have the same answer under **both** observers. Such a fault is
  invisible to all depth-zero/one observer behavior but can be separated by
  deeper future behavior.
- Exclude observer-row faults and immediately distinguishable transition
  faults. They do not test the proposed multi-step causal signal.

The frozen board contains 672 eligible faults across 88 worlds: 112 worlds
have zero, 64 have six, and 24 have twelve. The implementation must recompute
and verify this exact distribution and independently confirm that every
eligible destination pair differs within suffix depth three before any
official decision.

## Fault construction

For every eligible fault and each margin in `{0.05, 0.10, 0.20}`:

- initialize each correct transition/observer logit to `+margin` and every
  incorrect logit to `-margin`; and
- swap the correct and selected wrong transition logits in exactly one row.

Every initial hard machine must differ from its oracle in exactly one untied
row. There are 2,016 primary cases and 2,016 recoded pairs.

## Revision arms

Both arms start from byte-identical corrupted logits, compute the full
depth-three closure, and run four cycles.

- **Causal treatment:** optimize all base and action-prefix derivative
  signatures through depth three.
- **One-step control:** optimize only base words of length zero/one and
  action-prefix derivatives of total length one. Deeper signatures are still
  computed but masked before the reverse dynamic program. The exact oracle is
  a fixed point of both objectives.

For each categorical row, divide its logit adjoint by its maximum absolute
entry; zero rows remain zero. Subtract `0.1` times that row-normalized
direction. This positive diagonal preconditioner preserves descent for each
arm's objective and is the strongest parameter-free member of the
preregistered `[0.001, 0.1]` ACSO step range. There is no line search,
outcome-dependent stopping, learned controller, or margin-specific tuning.

## Evidence

For every fault, arm, recoding, and cycle zero through four, serialize:

- base, derivative, and total innovation for that arm's objective;
- exact full-machine recovery;
- intended fault-row recovery; and
- whether **any** transition or observer row is tied.

The report also records per-world, per-margin, and global aggregates plus a
SHA-256 over the ordered per-fault evidence.

## Recoding and execution controls

Derive deterministic nonidentity state, action, observer, and answer
permutations from each world identifier. Recode the exact machine and map each
fault descriptor through those permutations. Require identical exact,
fault-row, and all-row-tie decisions at every cycle, with innovation curves
equal within `1e-6`.

Run every fault independently with batch size one. Vectorized fault batches
are prohibited in the official audit, eliminating batch-reduction dependence.

## Source and custody binding

An official result is eligible only when:

- seed, counts, margins, cycles, step, and thresholds equal this document;
- all 200 worlds, 88 eligible worlds, and 672 faults are present;
- the protocol, runner, ACSO implementation, board generator, and Hankel
  codebook match their Git `HEAD` blobs;
- their file SHA-256 values, the Git commit, and a canonical board-manifest
  SHA-256 appear in the report; and
- an exclusive reservation is durably created at a previously absent output
  path before evaluation, remains unchanged throughout evaluation, and the
  report is published with no-clobber hard-link semantics plus directory
  `fsync`; the inode-bound reservation remains beside the final report as a
  permanent custody receipt.

Any nondefault seed, subset, source drift, dirty bound file, or altered
threshold is fixture-only and cannot emit GO.

## Gates

The current multi-step ACSO rule is **deep-fault oracle GO** only if:

1. every primary and recoded treatment total-innovation curve is
   nonincreasing within `1e-7`;
2. treatment exact and intended-row recovery are 100% at every margin and in
   every represented world;
3. treatment exact recovery exceeds one-step-control exact recovery by at
   least 80 percentage points at every margin and in every represented world;
4. no treatment tie appears in any row after the final cycle;
5. all primary/recoded decisions and innovation curves satisfy their gates;
6. every execution uses batch size one;
7. no nonfinite value appears; and
8. all source, board, count, and custody bindings pass.

Any violation is **NO-GO** for integrating or fitting the current multi-step
revision rule. Thresholds, margins, cycles, normalization, eligibility, and
controls cannot change after source freeze without a new named protocol.

## Claim boundary

A pass proves only that an explicit target-informed multi-step correction field
repairs these bounded synthetic deep faults better than an oracle-fixed
one-step ablation. It does not prove that Shohin infers target signatures, that
HSC generalizes from source bytes, that the 3,995,137-parameter preconditioner
learns useful scaling, that a sealed machine transfers to unseen task families,
or that Shohin reasons natively. Pretraining remains prohibited.

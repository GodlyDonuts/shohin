# R12 Cross-Domain Fault-Channel Review Result

**Decision:** the combined `0/3` no-go is rejected. Replication and pure
invertible transport retain narrow no-go results; globally enforced algebraic
relations uniquely repair the frozen missing transition and reopen only a
resource-counted learnability hypothesis.

## Frozen reviewed tuple

| Object | SHA-256 |
|---|---|
| `R12_CROSS_DOMAIN_FAULT_CHANNEL_NO_GO.md` | `d4d8e86adf4ba221bdf6d8505de58a1454d41f034034a62d69c62f388d4f9c28` |
| `pipeline/cross_domain_fault_channel_falsifier.py` | `6e54efa74e96e0a9ef2c42a078e04697a9e4395fe5166710ef972ae1630089e5` |
| `pipeline/test_cross_domain_fault_channel_falsifier.py` | `ab76e090215298d9261f07acaf61a3aa6cdfe7ee9bf179a27f04b199211a2dfc` |
| `scratchpad/cross_domain_fault_channel_no_go_v1.json` | `e410b256e5040382c5afd0875db6e72d39cd2fc359892063df2350bc473fff75` |

The report is byte-reproducible and its embedded payload SHA-256 is
`821bbf3850553beb3836884e18951fdaaed70303c851e39d4add871625a1d3a1`.
Five tests, Ruff, and `py_compile` pass. Ruff format check does not pass for the
frozen source, which remains unmodified.

## Candidate A: triadic efference commit

The frozen board correctly shows 12/12 recovery cases with one independently
flipped lane and 4/4 failures when all lanes share the same wrong semantic
action. This is a repetition code after semantic selection. It supports only
the narrow statement that majority replication cannot repair a common-mode
wrong program. It does not test whether heterogeneous learners can decorrelate
their semantic errors.

## Candidate B: reversible transport

The finite `F_5` board correctly performs 6,000 state/error/step comparisons
with zero contractions. A pure bijection cannot merge a perturbed state back
into the clean state without a noninvertible decoder or extra provenance. The
board does not test a learned observer, invariant, ancilla, syndrome extractor,
or Bayesian temporal estimator, so broader robustness and learnability claims
are rejected.

## Candidate C: relation-syndrome atlas

The frozen no-go is invalid. Its wrong patch is tested against the eleven
admitted transitions but not against the globally applied relations it claims
to preserve. Independent enumeration found five global relation violations.
Of all six possible missing successors, only the canonical successor satisfies
`s^2`, `t^2`, and `sts=tst` at every state. The already-known reverse `s` edge
also forces the missing edge through involution.

The patch passes the three relators only when they are checked from the identity.
This is evidence that identity-only cycle checks are insufficient, not evidence
that a globally constrained relation atlas cannot complete a missing
transition.

## Resource boundary

The frozen report's `surviving_advantage` fields and combined `0/3` conclusion
are literals. It does not count parameters, labeled examples, retained bits,
FLOPs, or learning curves. Functional equivalence to a recurrence does not by
itself reject a learnability or data-efficiency advantage.

The only reopened question is whether global relation-syndrome supervision has
a matched sample-efficiency or scale-extrapolation advantage over endpoint-only
training and a favorable relation-aware tied recurrence. That requires a new
frozen theory, exact resource ledger, exhaustive CPU collapse test, and fresh
independent review.

## Gate table

| Gate | Decision |
|---|---|
| Fault-neighborhood exact-recovery lemma | `GO`, narrow |
| Replication-code common-mode no-go | `GO`, narrow |
| Pure invertible state-only correction no-go | `GO`, narrow |
| Relation-atlas no-go | `NO-GO` |
| Combined `0/3` report | `NO-GO` |
| New resource-counted relation hypothesis | theory/CPU repair only |
| Neural source / data / fitting / H100 | `NO-GO` |
| Shohin reasoning or novelty claim | `NO-GO` |

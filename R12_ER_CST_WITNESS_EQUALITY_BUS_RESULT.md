# R12 ER-CST Witness Equality Bus v1.1 Development Result

**Development decision:** `authorize_one_sealed_confirmation`

**Final status:** independently confirmed; see
`R12_ER_CST_WITNESS_EQUALITY_CONFIRMATION_RESULT.md`.

## Frozen identity

| Item | Value |
|---|---|
| Scientific source | `87d53b53462d8d15660663238fd33886c010efb7` |
| Source manifest | `4f32348020163dcec4eeee970443120ca652be63ac3056d7b4c85cf9c2c1a6ac` |
| Board source | `5670ad83ae4e5806ec351337997c16990f2b5452` |
| Board seed | `2244518911844010727` |
| Training seed | `2262748995832026278` |
| Training-seed beacon | `dc2b653172beb8138b0d09acf45897365643fefe3d7b1483390fa835742a3488` |
| Sole job | `694567`, H100 `evc48`, 16m34s |
| Board | 48,000 train / 2,048 development / 2,048 sealed confirmation |
| Training | 3 arms, identical initialization, 3,000 updates each |
| Parameters | 192,726,827 complete / 12,021,276 trainable / 7,273,173 headroom |
| Custody after result | development/confirmation `1/0` |

## Result

| Metric | Treatment | Family deranged | Equality ablated |
|---|---:|---:|---:|
| Initial state | 99.609% | 32.812% | 16.113% |
| Complete cards | 99.902% | 0.293% | 0.195% |
| Events | 100% | 100% | 99.902% |
| HALT | 100% | 100% | 100% |
| Late query | 100% | 100% | 100% |
| All 18 witness pointers | 99.902% | 21.484% | 0% |
| Complete packet | 99.512% | 0.098% | 0% |
| Recurrent state | 99.609% | 17.822% | 15.479% |
| Answer | 100% | 33.740% | 31.104% |
| Packet/state/answer joint | 99.512% | 0.098% | 0% |

Treatment joint accuracy by depth is 100% at depths one through six, 99.219%
at depth seven, and 96.875% at depth eight. Its four renderer joint rates are
99.414%, 99.414%, 99.609%, and 99.609%. All 14 frozen scientific gates pass.
The independent assessor recomputes all metrics from raw hard predictions and
passes all eight artifact, parameter, source, metric, gate, and custody checks.

## Interpretation

ER-CST v1 failed despite perfect structural parsing because a generic record
residual could not infer a fresh operation's `S_3` permutation from six opaque
before/after name occurrences. V1.1 repairs exactly that interface:

1. six dedicated model-owned occurrence queries locate the three before and
   three after names for each rule;
2. learned byte-bigram fingerprints represent those occurrences;
3. a learned 3x3 equality matrix compares after names with before names;
4. finite assignment scores select one of the six legal permutations; and
5. the resulting cards are composed only by the pre-existing source-deleted
   recurrent categorical motor.

The two matched controls make the causal conclusion unusually sharp. Destroying
cross-occurrence equality or deranging card semantics collapses complete packet
and joint accuracy to approximately zero while leaving event order, HALT, and
query interfaces intact. The improvement is therefore not attributable to more
executor capacity, renderer memorization, direct answer supervision, or a host
parser. It is a learned episodic equality-and-assignment compiler feeding a
model-owned recurrent machine.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| Checkpoint | `917c1a1fce67c02258d0f90f04398ab433d18ba63c2dca92450cc5856c022ae7` |
| Raw development evidence | `1a7504eb9b08d7d123e89705360f2eb37a861f5cd75b3ebc73570c8e904327fb` |
| Development report | `d295f8f67f32916386e04674fc782a0982b9b1b55f7b82aa1eaab6f59bb1ae35` |
| Independent assessment | `29e4349225ed9523ec3b8096cd2cd16ef1b55c727797421a1ac0b39c042f11b2` |
| Development access ledger | `5b6e233b3cc9d3cf49a32525ca11f6c6f846005486df67a252e9ca4ec36b4db3` |

All artifacts are mirrored read-only on the Mac with matching hashes. A local
independent-assessor replay is byte-identical to the Newton assessment.

## Claim boundary

This development result establishes bounded fresh episodic `S_3` rule inference,
source-deleted categorical composition, internal halt, and late-query readout on
one split. It is not yet confirmed and does not establish unrestricted language
grounding, arbitrary operation induction, arithmetic, planning, or broad general
reasoning. The only authorized next score is the preregistered one-read sealed
confirmation in `R12_ER_CST_WITNESS_EQUALITY_CONFIRMATION_PREREG.md`.

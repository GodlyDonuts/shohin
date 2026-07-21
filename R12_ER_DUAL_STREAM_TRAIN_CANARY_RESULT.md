# R12 ER-TT Dual-Stream Train-Only Canary Result

**Protocol:** `r12_er_dual_stream_train_only_canary_v1`

**Decision:** reject v1 before fresh-board generation. No development or
confirmation split was read.

## Custody and provenance

- Source: `54476bcb02cc9d3f7388407fbe3700e92bfdbc28`
- Deterministic post-commit seed: `5113128174248698871`
- Seed derivation SHA-256:
  `46f57afbe522a3f7c36821ed3714dd178e1cc347fc306b13ea65d08f73ed3241`
- Sole H100 job: `694800` on `evc36`
- Runtime: 11m21s, exit code zero
- Fit/probe: 10,000/2,000 disjoint old-training families,
  40,000/8,000 rows
- Development/confirmation reads by this experiment: `0/0`
- Complete/trainable/headroom parameters:
  `192,730,091 / 18,327,299 / 7,269,909`

The probe-family overlap is zero. The fit/probe family-set hashes are
`5042c110...` and `16d86284...`.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `compiler.pt` | `829139114eeed714eea3b03074ed91b406bc37d999bcd855920e0f15c14dae03` |
| `train_probe_evidence.pt` | `e3e8646738d721092acf3f88cdf9f33426ddb7237c23cdec08184b8bec794c91` |
| `train_probe_report.json` | `697ce2830104aff034f3cb8c718f377edc1588560953033a6bc3aa3d828aeef4` |

Newton and local copies hash-match and are read-only.

## Frozen result

| Metric | Exact |
|---|---:|
| Packet | 0/8,000 |
| State | 164/8,000 = 2.050% |
| Answer | 1,666/8,000 = 20.825% |
| Joint | 0/8,000 |
| Complete relation rows | 0/8,000 |
| Complete witness pointers | 0/8,000 |
| Complete events | 0/8,000 |
| HALT | 3,252/8,000 = 40.650% |

Every cardinality-specific joint score is zero. The final relation-row loss is
`1.526640`, at chance for the mixed-cardinality task; witness-pointer loss is
`31.415091`.

The architecture did solve the construction target it was designed to solve:
all hard packet fields and all hard pointers are exactly identical on
8,000/8,000 original versus neutral-namespace alpha recodes. This includes
relations, declaration state, event binding, witness pointers, and every other
field. Alpha equivariance is therefore established by construction, but useful
routing is not.

## Independent granular recomputation

The local audit reconstructed the exact deterministic probe split and compared
the immutable prediction evidence with source-only targets.

| Quantity | Exact |
|---|---:|
| Active relation cells | 24,040/107,880 = 22.284% |
| Complete active rules | 224/23,900 = 0.937% |
| Initial-state cells | 8,000/36,140 = 22.136% |
| Binding occurrences | 2,116/36,140 = 5.855% |
| Initial occurrences | 0/36,140 |
| Before-witness occurrences | 8,000/107,880 = 7.416% |
| After-witness occurrences | 0/107,880 |
| Event cells before HALT | 40,297/96,000 = 41.976% |
| HALT cells | 98,468/104,000 = 94.681% |
| Line occurrences | 70,513/144,000 = 48.967% |
| Query occurrences | 8,000/8,000 = 100% |

Relation-cell accuracy by cardinality is 33.747%, 25.051%, 20.568%, and
16.315% at `N=3,4,5,6`, exactly the chance pattern. Every row emits three active
rules, and 7,532/8,000 rows emit cardinality five. The occurrence queries have
collapsed to a small number of structural positions rather than learning their
distinct slots.

## Failure localization and admitted diagnostic

The identity stream and exact equality are not the observed failure. The
structural router receives a physical-to-semantic assignment, but v1 detached
that assignment before the routed pointer loss. If the target semantic record
initially receives negligible assignment mass, the source-position probability
is clamped near zero; the pointer loss then cannot teach the role assignment
that would make the target record reachable. Local occurrence queries collapse
inside the wrong records.

V1.1 changes gradient topology and equality routing without changing capacity,
data budget, or the under-200M system:

1. compute semantic assignment normally for packet heads;
2. recompute the numerically identical assignment from detached record features
   for the routing path;
3. allow pointer/equality gradients to train the shared role-assignment head;
4. continue blocking those gradients from the shared structural record encoder;
5. replace hard selected-symbol equality with exact equality marginalized over
   the complete learned route distributions; and
6. require a matched source-span oracle-route control to reach 100% identity
   transport through the same marginal equality operator.

The oracle receives only source compiler spans, never outcomes or scored data.
It separates representational sufficiency of the identity bus from learned
route acquisition and cannot authorize promotion. The learned soft-route arm
retains every original capability and alpha-invariance threshold. V1 is closed
and will not be rerun.

## Marginal-route v1.1 result

**Protocol:** `r12_er_dual_stream_train_only_canary_v1_1`

**Decision:** reject before fresh-board generation. No development or
confirmation split was read, and the frozen threshold is not relaxed.

- Source: `8419c74e161f41c704d324d2b6ad72ed5587035f`
- Deterministic post-commit seed: `4412270997190025241`
- Sole H100 job: `694909` on `evc43`
- Runtime: 9m06s, exit code zero
- Fit/probe: 10,000/2,000 disjoint training families, 40,000/8,000 rows
- Complete/trainable/headroom: `185,532,296 / 11,129,504 / 14,467,704`
- Development/confirmation reads: `0/0`

| Metric | Exact |
|---|---:|
| Packet / joint / relation rows | 7,275/8,000 = 90.9375% |
| State | 7,765/8,000 = 97.0625% |
| Answer | 7,883/8,000 = 98.5375% |
| Witness pointers | 7,194/8,000 = 89.925% |
| Line pointers | 7,998/8,000 = 99.975% |
| Binding / initial pointers | 8,000/8,000 |
| Events / HALT / query | 8,000/8,000 |
| Alpha-invariant complete hard output | 8,000/8,000 |
| Oracle-route initial/relation/event/joint | 8,000/8,000 |

Every cardinality-specific joint gate passes; the minimum is `N=6` at
1,749/2,056 = 85.068%. The witness-pointer gate is the sole failed gate.

Immutable artifact SHA-256 values are:

| Artifact | SHA-256 |
|---|---|
| `compiler.pt` | `9e6115d1db01499f6cf5c7dd4763b2a43e82019c28a267131a1f71a2e95edb6f` |
| `train_probe_evidence.pt` | `52e70d017b49738a775df3c2638a3ee757ae68d0e06b67bfff3f251d18380fad` |
| `train_probe_report.json` | `a89439c861870b48983ca62e995e9da38ca8dc75d188165af2cde2612820f479` |

### Residual localization

Independent recomputation from immutable evidence finds all 806 failed
witness-pointer rows contain exactly one wrong occurrence. Individual pointer
occurrences are 214,722/215,528 = 99.626% exact. The dominant failures are the
second and third after-witness positions in the fourth rule. On representative
failures, the correct occurrence is route rank two while an adjacent duplicate
is rank one. Witness and relation failures overlap strongly but not perfectly:
7,054 rows have both exact, 585 have both wrong, 221 have only witness wrong,
and 140 have only relation wrong.

This is no longer evidence that identity equality or the recurrent relation
executor is missing. It is evidence that the alpha-invariant structural route
does not reliably distinguish repeated occurrences inside longer physical
records.

## Admitted train-only repair

The occurrence-addressed marginal repair adds two learned, identity-free
address components to the route: opaque-occurrence ordinal within the record
and total opaque candidates in that record. Raw symbol bytes remain confined to
the exact equality marginal. The repair adds 10,752 parameters, yielding
185,543,048 complete / 11,140,256 trainable / 14,456,952 headroom. It starts
from the confirmed parent, never failed canary weights, and keeps the v1.1 data,
2,500-update budget, optimizer, thresholds, and `1/0/0` custody unchanged.

Before source freeze, 22 focused tests plus Ruff, byte compilation, shell
syntax, a real-board alpha-recode equality check, a real-parent backward pass,
full trainable-gradient coverage, and zero excluded-parent leakage pass. A
single post-commit seed and single isolated H100 canary are authorized; no
fresh board is authorized unless every unchanged gate passes.

## Occurrence-addressed result

**Protocol:** `r12_er_addressed_marginal_train_only_canary_v1`

**Decision:** reject before fresh-board generation. Exact source
`7601625f2cdc0a476f9383ce9773722f64760d17` precedes drand round `6305746`,
canonical payload SHA `8884bfe6...`, derivation SHA `c24770ac...`, and seed
`4775909816533321494`. Sole job `694928` completed on H100 `evc36` in 7m38s.
Development/confirmation custody remains `0/0`.

| Metric | Exact |
|---|---:|
| Packet / joint / relation rows | 4,873/8,000 = 60.9125% |
| State | 6,411/8,000 = 80.1375% |
| Answer | 7,218/8,000 = 90.225% |
| Witness pointers | 4,760/8,000 = 59.500% |
| Binding / initial / events / HALT / query / line | 8,000/8,000 |
| Alpha-invariant complete hard output | 8,000/8,000 |
| Oracle-route initial/relation/event/joint | 8,000/8,000 |

Cardinality-specific joint is 90.041%, 64.008%, 47.194%, and 43.606% for
`N=3,4,5,6`. Four scientific gates fail. The parameter certificate remains
185,543,048 complete / 11,140,256 trainable / 14,456,952 headroom.

Artifact SHA-256 values are:

- checkpoint: `803a850af3f18a8138d6f5029f6c1d459b0f15fb6d0c42f3e0c407810108e4e3`
- evidence: `a2d8349bd6efcfc9eaca522661b24130529ad04fd08ab313ce20dbeeb3c35227`
- report: `11d230a2d2c83bbf70997bbd2fd11189f6094d43fdd1717f2a7595e04d52b0f6`

Independent reconstruction finds 3,240 witness-failed rows. Wrong-pointer
counts per failed row range from one through six rather than the predecessor's
single error. The dominant errors are adjacent ordinal swaps in after-witness
positions, especially for cardinalities four through six. Endpoint ordinal
embedding norm is 8.64 and ordinal rows six/seven have cosine 0.878; count
embedding norm is 2.02. The address representation therefore became an
entangled positional shortcut rather than a reliable tie-breaker.

The architecture/version is closed. A post-hoc scale ablation may read this
same consumed training probe solely to separate ordinal dominance from count
interaction. It has no optimizer and cannot authorize a fresh board, promotion,
or any reasoning claim.

## Closed-checkpoint scale audit

Read-only H100 job `694932` evaluates the immutable addressed endpoint on the
same consumed probe with no optimizer. Report SHA-256 is
`d958cc0507fe85a489a3b85368f52ed67cfda6caf9fc5efc8d686216f28f6934`.

| Ordinal scale | Count scale | Witness | Joint | Relation |
|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.000% | 0.200% | 0.400% |
| 0.25 | 1.00 | 1.1875% | 1.7125% | 9.075% |
| 0.50 | 1.00 | 12.1125% | 21.150% | 21.1625% |
| 0.75 | 1.00 | 34.225% | 40.7125% | 40.7875% |
| 1.00 | 1.00 | 59.500% | 60.9125% | 60.9125% |
| 1.50 | 1.00 | 70.425% | 69.3875% | 73.100% |
| 1.00 | 0.00 | 0.4875% | 3.275% | 3.275% |
| 0.00 | 0.00 | 0.000% | 0.275% | 0.3125% |

Ordinal and count information are both necessary, and stronger ordinal signal
helps rather than hurts. The failure comes from combining those embeddings with
structural record/token memory before the shared key projection, not simply
from excessive positional magnitude.

## Factorized witness-route successor

The distinct successor preserves the v1.1 structural dot-product logits and
adds a centered/bounded `14 x 12 x 14` residual with twelve zero-initialized
role gates only to witness routes. Its
indices are record candidate count, semantic witness role, and candidate
ordinal. The route has 2,364 learned scalars; all non-witness routes are
unchanged even when the table is nonzero. Complete/trainable/headroom is
185,534,660 / 11,131,868 / 14,465,340.

It starts from the reconstructed confirmed parent, never either rejected
canary. The old train-only data, 2,500-update budget, optimizer, thresholds,
alpha/oracle controls, and `1/0/0` custody remain exact. Twenty-four focused
tests plus Ruff, byte compilation, shell syntax, real-row alpha invariance,
real-parent gradient coverage, and zero excluded leakage pass before source
freeze. All four same-seed arms fit and are atomically checkpointed before the
probe is scored: treatment, residual-disabled baseline, content-disabled
structural-only, and physical-record-rotated shuffled-address. Treatment must
beat baseline and shuffled address by at least +0.5pp witness rows. Failure
closes the route; passing authorizes only a fresh-board test.

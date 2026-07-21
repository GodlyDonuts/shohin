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

# R12 ER Factorized Witness-Route Result

**Protocol:** `r12_er_factorized_witness_route_train_only_canary_v1`

**Decision:** `reject_factorized_witness_before_fresh_board`

The route is closed. It must not receive a fresh board, threshold relaxation,
or post-hoc optimizer tuning.

## Operational provenance and custody receipts

The job/node/runtime, beacon derivation, and Newton mirror statements below are
operational journal receipts. They are not independently reconstructed by the
three-file artifact audit.

- Exact source commit: `4643d1a51defe53397f9bed481051621d85c0b11`
- Public drand round: `6305851`
- Canonical drand payload SHA-256:
  `0bffb1f1ba8b9649554be76a712cf50e389aeb38084662e35203f033f065a9ea`
- Seed derivation SHA-256:
  `ddf2905f16a414fd90c3db7e2ff1f404fa5e88ecef04d11c966863154678cb93`
- Deterministic seed: `6769631927967421693`
- Sole H100 job: `694945` on `evc36`
- Runtime: 24m25s, exit code zero
- Fit/probe: 10,000/2,000 disjoint old-training families,
  40,000/8,000 rows
- Train-only/development/confirmation reads: `1/0/0`
- Complete/compiler/trainable/new parameters:
  `185,534,660 / 60,452,996 / 11,131,868 / 2,364`

All four arms began from trainable-state digest
`0cd21dcf3b4bf0f7a741ad9369535ff8157074a4c0653e93b12d6cb119e5b8be`,
used fit-order seed `746651095578902126`, ran exactly 2,500 updates, and
preserved the same frozen-parent digest.

## Immutable artifacts

| Artifact | SHA-256 |
|---|---|
| `compiler.pt` | `e93bb4cff5f316616c7a02bce272112acf454f42f56f8b4ea07ffac6074318a2` |
| `train_probe_evidence.pt` | `11d931b37ad854de9976015fc1ff38522da0812776ad67ea7d408d43820889c6` |
| `train_probe_report.json` | `87ea12a28cfaf82c4556f1730da6778cf63df9e5c2ad3df8b09a86613786ccca` |

Newton and local copies hash-match. The local copies are read-only.

## Frozen result

| Arm | Witness | Relation / joint | State | Answer | Events |
|---|---:|---:|---:|---:|---:|
| Treatment | 2,069/8,000 = 25.8625% | 2,239/8,000 = 27.9875% / 2,209/8,000 = 27.6125% | 57.8875% | 77.075% | 99.9875% |
| Same-seed baseline | 11.7125% | 12.9625% / 12.8875% | 37.975% | 59.050% | 93.0125% |
| Structural-only | 92.050% | 1.250% / 1.250% | 36.000% | 58.400% | 92.400% |
| Shuffled address | 0.2875% | 5.650% / 5.650% | 40.6625% | 63.1875% | 100.000% |

Treatment joint accuracy collapses with cardinality: 67.276% at `N=3`,
35.501% at `N=4`, 5.550% at `N=5`, and 1.892% at `N=6`. Treatment has
99.1375% initial rows/pointers and 99.9875% line pointers, so the failure is
localized to witness identity transport and the relation assembled from it.

Nine of thirteen frozen gates pass. The failures are the absolute witness,
relation, packet/state/answer/joint, and minimum-cardinality joint gates. Exact
alpha invariance, source-only oracle transport, matched initialization,
parameter accounting, train-only custody, and treatment gains over baseline
and shuffled controls all pass. Partial relative gains cannot override the
absolute failure.

## Independent artifact-only audit

`train/audit_er_factorized_witness_route.py` verifies the three exact artifact
hashes, all 33 committed source files, source/seed receipts, all arm
initialization/fit/frozen-parent receipts, the exact 400-leaf metric schema and
its cardinality/depth/renderer aggregates, evidence shapes, and every frozen
gate without reading any ER-TT board row or scored split. It independently
recomputes every retained canonical-versus-recoded alpha field and the complete
mask as 8,000/8,000. The serialized canonical audit JSON including its trailing
newline has SHA-256
`272b6b3be28fab3741c6e5c383a1c02b37d7b7e8b394dd6424d41f12de9bbe3d`;
its explicitly labeled preimage self-hash is
`3747beac310a790791d02677a050aa2e370d0a4b6153b999cd89d4a013e8a6ca`.
The read-only report is stored at
`artifacts/r12/er_factorized_witness_route_artifact_audit.json`.

The effective cardinality-dependent `1 + 2N` address slice with side-major
roles selects the grammar-correct one-based ordinal on 36/36 active cases in
treatment and structural-only, 10/36 in shuffled-address, and 0/36 in baseline
because its gate is exactly zero. Treatment gate/table L2 norms are
0.7896/3.5995; structural-only norms are 0.8211/4.2015. Thus the table learned
the deterministic grammar address, but the soft content residual did not carry
a coherent symbol identity through that address. Structural-only can point to
the right slot while producing only 1.25% exact relations, demonstrating that
address correctness is not content transport.

Control row-level predictions were not retained, so paired McNemar statistics
cannot be reconstructed. No second probe access is permitted to repair that
omission. Oracle-route targets were also not retained, so its 8,000/8,000
exactness remains an internally consistent producer receipt rather than an
independently regenerated result. Family disjointness, custody counters,
frozen-parent integrity, and total compiler/base parameter counts are likewise
hash-bound producer receipts. Treatment trainable parameters and all alpha
comparisons are independently recomputed from retained tensors.

## Consequence

The factorized lookup is a structural answer key for this grammar, not a native
reasoning mechanism. It is also seed/optimization-sensitive: its same-seed
baseline witness score is 11.7125%, far below the historical marginal v1.1
endpoint of 89.925%. Both facts make threshold tuning scientifically invalid.

The next architecture must make the committed content state itself causal and
reusable. Closure-Tied Action Algebra (CTAA) is the current pre-neural
falsifier: a trunk-conditioned hard categorical state transducer whose action
application and action composition share parameters. Its CPU mechanics are
audited independently; no neural source, seed, board, or H100 job is yet
authorized.

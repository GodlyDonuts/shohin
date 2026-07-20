# ER-CST v1 Development Result

**Decision:** reject; do not open confirmation

**Final custody:** development 1, confirmation 0

**Scientific source:** `90fd496de23ee9f12d21fa2c553df0de3fad9b23`

**Board seed:** `8277659525319823840`

**Training seed:** `7148525615058810782`

**Sole job:** `694511` on H100 `evc25`

## Artifact identity

| Artifact | SHA-256 |
|---|---|
| Immutable checkpoint | `150febfa00d1129a61876ab4a852708225a3b9ab29bece4737e1dbeb84874136` |
| Raw development evidence | `05756471521072f5024c0cbfb3ddf3d92053e7d8845b00dec7355792ed318f37` |
| Development report | `be4b5c5050bf148f5d86614433d3b53a7ed21160b6864a8c0a799f5c1dde2822` |
| Independent assessment | `39ffd483a0167a09158196b2d22a310ba85279b795ab4e7b7a291b1193858bc9` |
| Development access ledger | `5efae65cf5f15762643f8a1bac18ce2fb68cdf9fbccfb6df43cb3c84769bd1c2` |

Newton and local read-only mirrors hash-match.

## Exact development scores

| Arm | Initial | Cards | Events | HALT | Query | State | Answer | Packet/joint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Treatment | 642/2,048 (31.348%) | 0/2,048 | 2,048/2,048 | 2,048/2,048 | 2,048/2,048 | 311/2,048 (15.186%) | 682/2,048 (33.301%) | 0/2,048 |
| Family-deranged | 2,018/2,048 (98.535%) | 0/2,048 | 2,048/2,048 | 2,048/2,048 | 2,048/2,048 | 431/2,048 (21.045%) | 775/2,048 (37.842%) | 0/2,048 |
| Equality-ablated | 1,398/2,048 (68.262%) | 0/2,048 | 2,048/2,048 | 2,048/2,048 | 2,048/2,048 | 362/2,048 (17.676%) | 706/2,048 (34.473%) | 0/2,048 |

Every arm is also 2,048/2,048 on line, declaration-binding, initial-occurrence,
and query-occurrence pointers. Treatment has zero exact complete card tuples in
every renderer and depth group.

## Post-result localization

Treatment's direct card head predicts only classes 2, 3, and 4 and collapses target
classes into coarse groups. Interpreting predictions under the inverse-permutation
convention recovers only 22.27% card cells. Fitting the best global class remapping
on the first 1,024 development rows and applying it to the held-out 1,024 rows gives:

- 33.46% card-cell exactness;
- 16.80% complete three-card tuple exactness;
- 42.77% corrected initial-state exactness;
- 20.21% corrected recurrent-state exactness;
- 40.72% corrected answer exactness.

Therefore the negative is not a hidden global code permutation. The matched controls
and fit losses show two facts:

1. the structural parser, record order, opcode references, HALT, query, categorical
   motor, and reader are not the bottleneck;
2. an undifferentiated record vector fails to extract before/after equality, and its
   card loss interferes with declaration/initial features.

The admitted repair is the fresh-board Witness Equality Bus preregistered in
`R12_ER_CST_WITNESS_EQUALITY_BUS_PREREG.md`. The old development board must never be
rescored and its confirmation must remain sealed.

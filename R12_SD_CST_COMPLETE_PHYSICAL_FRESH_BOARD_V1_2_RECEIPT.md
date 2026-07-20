# R12 SD-CST Complete Physical Fresh-Board v1.2 Receipt

**Status:** admitted and sealed before training-seed draw or model fit. Receipt
commit `b5beee2d04f802771093f16b891e60a830a1144b` precedes the sole
training-seed draw.

## Custody

- Scientific source: `fab094f6e32f1e928551f1830509b8c83fbd759e`
- Raw board beacon: `13419454120885953526`
- Board seed modulo `2^63`: `4196082084031177718`
- Development/confirmation access: `0/0`
- Confirmation mode: `0600`
- Previous v1.1 board/training seeds are closed and not reused.
- Raw training beacon: `15146785326247343388`
- Training seed modulo `2^63`: `5923413289392567580`

## Board and hashes

| Artifact | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| report | - | 3,807,172 | `162b6054b74509eba35c1f5b339ba22edc555b7edc7cba36640270869cbec1d0` |
| train | 48,000 | 180,583,488 | `9bc8d0b6227fdf70b09043294ba075be6d7651c9897fbddffb58139dc2fc0547` |
| development | 2,048 | 8,122,544 | `6bc327ce34b225a5f46199cafdbfe7692212eccf0753356a6f3aafa11fa2c854` |
| sealed confirmation | 2,048 | 8,126,928 | `b8ec5d84f087fe2a22893631485387da5ef2c11e1366b7c632b2395b66827d0b` |

The development row-ID/content registrations are
`ddf848c008b24bc6f9263d9e639da7c9179c40a20ff650feafc1ac3f4f0fd160`
and
`11e0b6a9ea9e0c9c107fb56a83845c2f83e7fc3f0d76f6e1ba282e43dc48dcfc`.
The confirmation registrations are
`0dc6b430be6036de25f6da21f9bdc6e04c03a1c600de29332687819976baeef3`
and
`64ae25bfc95d8f916894311be108ef82302bc7c454c1a6e76563fa715df981fe`.

## Admission result

All seventeen frozen gates pass. In addition to counts, independent oracle
agreement, renderer parity, split/prior leakage exclusions, unique names,
outcome custody, and access sealing, every one of the 52,096 rows is accepted by
the exact production `parse_projected_row` path. This directly closes the v1.1
redundant-event-name defect before GPU execution.

A second full generation from the same source, seed, and prior inputs is
byte-identical for all four artifacts. The post-receipt training draw is now
recorded above; no optimizer, output, or scored read exists. This authorizes
only the sole preregistered development pilot; it does not authorize
confirmation or establish reasoning.

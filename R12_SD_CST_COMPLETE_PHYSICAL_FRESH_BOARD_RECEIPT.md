# R12 SD-CST Complete Physical Fresh-Board Receipt

**Status:** rejected prefit without scored access. Board
receipt commit `296af7e96e06082771b8963c524de6c23a26a445` precedes the sole
training-seed draw.

## Custody

- Scientific source: `aa1c598594aec985519e300f9207a0fa8da72ea4`
- Raw 64-bit beacon: `15994587003838256523`
- Board seed, reduced modulo `2^63`: `6771214966983480715`
- Development accesses: `0`
- Confirmation accesses: `0`
- Confirmation file mode: `0600`
- Raw training beacon: `16198579975688416761`
- Training seed, reduced modulo `2^63`: `6975207938833640953`
- Prior consumed train SHA-256:
  `b7756dbf8d4401dbc5fb897dee53f68758e27200b1ce0d2387631f2f0205ec25`
- Prior consumed development SHA-256:
  `0e0720030f4b5739b7de7320fb45f5817e1e8fadb3f7f12e62b98e2f41593191`

The previous source/seed pair `cd5a02b...` / `8056159684949768997`
failed the global-name-uniqueness admission gate before writing any board byte.
It is closed and was not reused.

## Board

| Split | Families | Views/family | Rows | Renderer parity |
|---|---:|---:|---:|---|
| training | 12,000 | 4 | 48,000 | even |
| development | 512 | 4 | 2,048 | odd |
| sealed confirmation | 512 | 4 | 2,048 | odd |

All depths one through six are balanced to within one latent family in each
scored split. Training contains compiler fields only and no final state, answer,
or trajectory. Development and confirmation contain scorer-only outcomes.

## Hashes

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| board report | 3,803,182 | `7ecb3dcfea53d82180fc99c7911b9cf9169e0b29c6154f2490019c1f89418cc4` |
| train | 180,573,152 | `bad7f8db5b8580b4f960ceb1adfa41704c8afd595f8c1b97623c0faaa927b61c` |
| development | 8,121,104 | `58aef89229821e665bef26b0d97e03d475192d86a85de2ddea9ef3759ebbdaca` |
| sealed confirmation | 8,129,968 | `afedef7565d3c6e963783f91a26262f389a4925c09ba8de4dd14bdafb46c9b4e` |

Development row registration SHA-256 values are
`ddf848c008b24bc6f9263d9e639da7c9179c40a20ff650feafc1ac3f4f0fd160`
for row IDs and
`265f4212690a75cd0f596d2b2d4789c3b3d1b85cc07e5fe21286c0c59228e9bb`
for row content. Confirmation registrations are
`0dc6b430be6036de25f6da21f9bdc6e04c03a1c600de29332687819976baeef3`
and
`2c7021db3465a85cc59a27a06a8cf3f3b19f2f6c46ef2e2fba6f641cf139c03c`.

## Admission

All sixteen board gates pass:

- exact row counts, unique IDs, four complete views per family;
- independent simulator/oracle agreement on all 52,096 rows;
- training and scored renderer orbits disjoint, with development and
  confirmation renderer orbits equal;
- zero cross-split exact prompt, 13-gram, name, and operation-sequence overlap;
- zero prior prompt/name/sequence overlap and zero prior scored 13-gram overlap;
- three fixed globally unique opaque names per family;
- no training outcomes and complete scored outcomes; and
- zero development and confirmation access.

A second full generation from the same committed source, prior inputs, and seed
is byte-identical for all four artifacts. H100 job `694333` then passed CUDA
preflight but failed while parsing training bytes: family re-keying had updated
declaration bindings and scorer answers without updating redundant active-event
entity strings. The failure occurred before model initialization, optimizer
creation, output-directory creation, access-ledger creation, or development
read. Access remains `0/0`; confirmation remains sealed. Close this board and
both seeds. The successor must update event strings, require all 52,096 rows to
pass the exact runtime parser during board admission, freeze a new source, and
draw new board/training seeds.
